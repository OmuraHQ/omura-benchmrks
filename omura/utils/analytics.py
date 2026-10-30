"""Traction analytics: a privacy-preserving local mirror of search traffic.

The production source of truth is Cloudflare (Zone GraphQL rollups plus a
Logpush ``http_requests`` job).  This module owns two things:

* :class:`AnalyticsStore` -- a small SQLite store that can ingest individual
  search events (from the ASGI middleware or from ``scripts/ingest_cloudflare_logpush.py``)
  and Cloudflare daily rollups (from ``scripts/cloudflare_analytics.py``).
* :class:`AnalyticsMiddleware` -- a non-blocking ASGI middleware that mirrors
  tracked search requests into the store.

Privacy rules
-------------
Raw client IPs are **never** written to SQLite.  We store a salted SHA-256 of
the canonical IP (``ip_hash``).  The salt comes from ``OMURA_ANALYTICS_SALT``;
if that is unset we derive one from ``OMURA_ADMIN_TOKEN``.  If neither exists,
IP hashing is disabled (``ip_hash`` is NULL) and every report is marked
``verified=false`` -- the endpoint still works, but MAU cannot be proven.

Environment variables (all optional; defaults in parentheses)
-------------------------------------------------------------
``OMURA_ANALYTICS_DB_PATH`` (``data/analytics.sqlite``)
    SQLite path.  Created lazily on first use.
``OMURA_ANALYTICS_SALT``
    Salt for IP/query hashes.  Required for a verified report in production.
``OMURA_ANALYTICS_EXCLUDE_COUNTRIES`` (empty)
    Comma-separated ISO-3166 alpha-2 codes, e.g. ``CN,RU``.
``OMURA_ANALYTICS_EXCLUDE_IPS`` (empty)
    Comma-separated IPs/CIDRs (operator host, grantee egress, CI).
``OMURA_ANALYTICS_EXCLUDE_ASNS`` (empty)
    Comma-separated ASNs, with or without the ``AS`` prefix.
``OMURA_ANALYTICS_BOT_REGEX``
    Override the default bot/automation user-agent regex.
``OMURA_ANALYTICS_ENABLED`` (``true``)
    Set to ``false`` to disable middleware recording.
``OMURA_ANALYTICS_MAU_TARGET`` (``50``)
``OMURA_ANALYTICS_QUERY_TARGET`` (``5000``)
    Thresholds used by ``meets_mau`` / ``meets_queries``.

This module is import-safe: no DB is created and no network call is made at
import time.  Lookup helpers (bot detection, country/ASN/IP filtering) are
entirely offline and run in the request path.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union
from urllib.parse import parse_qs

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

DEFAULT_DB_PATH = "data/analytics.sqlite"

# Only these request shapes count as "search traffic".
_SKIP_PREFIXES = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/static",
    "/favicon",
    "/robots.txt",
)

# Default automation fingerprint.  Kept deliberately broad: the grant requires
# *unaffiliated humans*, so erring toward exclusion is the honest direction.
DEFAULT_BOT_REGEX = (
    r"(?:curl|wget|python-requests|python-urllib|aiohttp|httpx|"
    r"go-http-client|java/|okhttp|libwww-perl|"
    r"bot\b|spider|crawler|scraper|monitor|uptime|pingdom|statuscake|"
    r"headless|phantomjs|selenium|playwright|puppeteer|"
    r"prometheus|kube-probe|blackbox|grafana|datadog|newrelic|"
    r"github-actions|gitlab-runner|jenkins|buildkite|circleci|travis|"
    r"omura-ci|grantee-ci|synthetic)"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS search_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    day           TEXT    NOT NULL,
    endpoint      TEXT    NOT NULL,
    method        TEXT    NOT NULL,
    status        INTEGER,
    latency_ms    REAL,
    ip_hash       TEXT,
    country       TEXT,
    region        TEXT,
    asn           TEXT,
    user_agent    TEXT,
    referer       TEXT,
    tenant        TEXT,
    query_hash    TEXT,
    query_len     INTEGER DEFAULT 0,
    is_bot        INTEGER DEFAULT 0,
    is_excluded   INTEGER DEFAULT 0,
    exclude_reason TEXT,
    meta          TEXT,
    UNIQUE (ts, ip_hash, endpoint)
);

CREATE INDEX IF NOT EXISTS idx_search_events_day ON search_events(day);
CREATE INDEX IF NOT EXISTS idx_search_events_ip_hash ON search_events(ip_hash);
CREATE INDEX IF NOT EXISTS idx_search_events_endpoint ON search_events(endpoint);
CREATE INDEX IF NOT EXISTS idx_search_events_tenant ON search_events(tenant);

CREATE TABLE IF NOT EXISTS daily_rollups (
    day             TEXT NOT NULL,
    source          TEXT NOT NULL,
    unique_visitors INTEGER NOT NULL DEFAULT 0,
    search_queries  INTEGER NOT NULL DEFAULT 0,
    total_requests  INTEGER NOT NULL DEFAULT 0,
    countries_json  TEXT NOT NULL DEFAULT '[]',
    generated_at    TEXT NOT NULL,
    UNIQUE (day, source)
);

-- Window-level totals as reported by Cloudflare's dashboard. A 30-day
-- "unique visitors" figure is NOT the sum of daily uniques, so it cannot live
-- in daily_rollups without inflating MAU.
CREATE TABLE IF NOT EXISTS cloudflare_windows (
    start_day       TEXT NOT NULL,
    end_day         TEXT NOT NULL,
    unique_visitors INTEGER NOT NULL DEFAULT 0,
    search_queries  INTEGER,
    total_requests  INTEGER NOT NULL DEFAULT 0,
    countries_json  TEXT NOT NULL DEFAULT '[]',
    source          TEXT NOT NULL DEFAULT 'dashboard',
    note            TEXT,
    recorded_at     TEXT NOT NULL,
    UNIQUE (start_day, end_day, source)
);
"""

_EVENT_COLUMNS = (
    "ts, day, endpoint, method, status, latency_ms, ip_hash, country, region, "
    "asn, user_agent, referer, tenant, query_hash, query_len, is_bot, "
    "is_excluded, exclude_reason, meta"
)

_INSERT_EVENT = (
    f"INSERT OR IGNORE INTO search_events ({_EVENT_COLUMNS}) "
    "VALUES (" + ", ".join("?" for _ in range(len(_EVENT_COLUMNS.split(",")))) + ")"
)


# ---------------------------------------------------------------------------
# Configuration helpers (read on every call so tests / operators can change env)
# ---------------------------------------------------------------------------


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "true" if default else "false").lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def _split_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _int_env(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)) or default)
    except ValueError:
        return default


def salt_info() -> Tuple[Optional[str], Optional[str]]:
    """Return ``(salt, source)`` without exposing the secret itself.

    ``source`` is ``"OMURA_ANALYTICS_SALT"``, ``"OMURA_ADMIN_TOKEN"`` or
    ``None`` when hashing is disabled.
    """
    explicit = _env("OMURA_ANALYTICS_SALT")
    if explicit:
        return explicit, "OMURA_ANALYTICS_SALT"
    token = _env("OMURA_ADMIN_TOKEN")
    if token:
        return f"admin-token:{token}", "OMURA_ADMIN_TOKEN"
    return None, None


def _hash(value: str, salt: str, domain: str) -> str:
    return hashlib.sha256(f"{salt}|{domain}|{value}".encode("utf-8")).hexdigest()


def hash_ip(ip: Optional[str], salt: Optional[str] = None) -> Optional[str]:
    """Return a salted SHA-256 of *ip*, or ``None`` when hashing is disabled."""
    normalised = normalize_ip(ip)
    if not normalised:
        return None
    if salt is None:
        salt, _ = salt_info()
    if not salt:
        return None
    return _hash(normalised, salt, "ip")


def hash_query(query: Optional[str], salt: Optional[str] = None) -> Optional[str]:
    if query is None:
        return None
    if salt is None:
        salt, _ = salt_info()
    # Queries are hashed even without an IP salt; this is not an identification
    # mechanism, just a way to count duplicate searches.
    return _hash(query.strip(), salt or "no-salt", "query")


def normalize_ip(value: Optional[Union[str, bytes]]) -> Optional[str]:
    """Canonicalise an IP, stripping ports/brackets and IPv4-mapped IPv6."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("["):
        # [2001:db8::1]:443
        text = text[1 : text.find("]")] if "]" in text else text.lstrip("[")
    elif text.count(":") == 1 and "." in text:
        # 203.0.113.7:443
        host, port = text.rsplit(":", 1)
        if port.isdigit():
            text = host
    try:
        parsed = ipaddress.ip_address(text)
    except ValueError:
        return None
    if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped:
        parsed = parsed.ipv4_mapped
    return str(parsed)


def _parse_networks(raw: str) -> List[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]]:
    networks = []
    for token in _split_csv(raw):
        try:
            networks.append(ipaddress.ip_network(token, strict=False))
        except ValueError:
            # Ignore malformed entries rather than taking the API down.
            continue
    return networks


def _network_is_exact(net: Union[ipaddress.IPv4Network, ipaddress.IPv6Network]) -> bool:
    return net.prefixlen == net.max_prefixlen


def _parse_asns(raw: str) -> Tuple[str, ...]:
    out = []
    for token in _split_csv(raw):
        token = token.upper()
        if token.startswith("AS"):
            token = token[2:].strip()
        if token:
            out.append(token)
    return tuple(sorted(set(out)))


def normalize_asn(value: Optional[Union[str, int]]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    if text.startswith("AS"):
        text = text[2:].strip()
    return text or None


def _country(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    text = str(value).strip().upper()
    if not text or text in ("XX", "T1", "UNKNOWN"):
        return None
    return text[:8]


def _in_networks(
    ip: Optional[str],
    networks: Sequence[Union[ipaddress.IPv4Network, ipaddress.IPv6Network]],
) -> bool:
    normalised = normalize_ip(ip)
    if not normalised:
        return False
    try:
        addr = ipaddress.ip_address(normalised)
    except ValueError:
        return False
    for net in networks:
        if addr.version == net.version and addr in net:
            return True
    return False


def _bot_pattern() -> str:
    return _env("OMURA_ANALYTICS_BOT_REGEX") or DEFAULT_BOT_REGEX


@lru_cache(maxsize=8)
def _compiled_bot_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


def is_bot_ua(user_agent: Optional[str]) -> bool:
    """Classify a user agent as automation.  Missing UA is treated as a bot."""
    if not user_agent or not user_agent.strip():
        return True
    try:
        return bool(_compiled_bot_pattern(_bot_pattern()).search(user_agent))
    except re.error:
        # A broken custom regex must not break request handling.
        return bool(_compiled_bot_pattern(DEFAULT_BOT_REGEX).search(user_agent))


def ua_class(user_agent: Optional[str]) -> str:
    """Coarse UA family used for the secondary "unique visitor" definition."""
    if not user_agent:
        return "none"
    ua = user_agent.lower()
    if "headlesschrome" in ua or "headless" in ua:
        return "headless"
    if "edg/" in ua or "edge/" in ua:
        return "edge"
    if "firefox/" in ua or "fxios" in ua:
        return "firefox"
    if "chrome/" in ua or "crios" in ua:
        return "chrome"
    if "safari/" in ua:
        return "safari"
    if "curl/" in ua:
        return "curl"
    if "python" in ua:
        return "python"
    if "okhttp" in ua or "java/" in ua:
        return "java"
    if "bot" in ua or "spider" in ua or "crawler" in ua:
        return "bot"
    return "other"


def _exclusions() -> Dict[str, Any]:
    countries = tuple(sorted({c.upper() for c in _split_csv(_env("OMURA_ANALYTICS_EXCLUDE_COUNTRIES"))}))
    networks = _parse_networks(_env("OMURA_ANALYTICS_EXCLUDE_IPS"))
    asns = _parse_asns(_env("OMURA_ANALYTICS_EXCLUDE_ASNS"))
    salt, _ = salt_info()
    exact_hashes = []
    for net in networks:
        if _network_is_exact(net):
            # Exact IPs can be excluded retroactively because we can re-hash
            # them; CIDR ranges are only enforceable at ingestion time.
            digest = hash_ip(str(net.network_address), salt=salt)
            if digest:
                exact_hashes.append(digest)
    return {
        "countries": countries,
        "networks": tuple(networks),
        "asns": asns,
        "exact_hashes": tuple(sorted(set(exact_hashes))),
    }


def _exclusion_reason(
    ip: Optional[str], country: Optional[str], asn: Optional[str], info: Dict[str, Any]
) -> Optional[str]:
    if _in_networks(ip, info["networks"]):
        return "ip"
    if country and country.upper() in info["countries"]:
        return "country"
    if asn and asn in info["asns"]:
        return "asn"
    return None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_timestamp(value: Optional[Union[str, int, float, datetime]]) -> str:
    """Return a stable UTC ISO-8601 string.

    Accepts RFC3339 strings, naive datetimes (assumed UTC), and epoch values in
    seconds / milliseconds / microseconds / nanoseconds (Cloudflare Logpush uses
    nanoseconds by default).
    """
    if value is None:
        return _utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip()
        if not text:
            return _utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        if re.fullmatch(r"\d+(?:\.\d+)?", text):
            number = float(text)
        else:
            iso = text[:-1] + "+00:00" if text.endswith("Z") else text
            try:
                dt = datetime.fromisoformat(iso)
            except ValueError:
                return _utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    # Epoch magnitude heuristic: ns > us > ms > s.
    if number > 1e17:
        number /= 1_000_000_000
    elif number > 1e14:
        number /= 1_000_000
    elif number > 1e11:
        number /= 1_000
    try:
        dt = datetime.fromtimestamp(number, timezone.utc)
    except (OverflowError, OSError, ValueError):
        return _utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# Request classification / extraction
# ---------------------------------------------------------------------------


def is_tracked_request(method: str, path: str) -> bool:
    """True only for the search endpoints we count toward traction."""
    method = (method or "GET").upper()
    path = (path or "/").split("?", 1)[0]
    if not path.startswith("/"):
        path = "/" + path
    if path == "/" or path.startswith(_SKIP_PREFIXES):
        return False

    # POST /search, POST /search/, POST /search/audio, ... /video, /reverse-image
    if method == "POST":
        return path == "/search" or path.startswith("/search/")
    # Older/alternate GET variants kept for compatibility with the grant wording.
    if method == "GET":
        return (
            path == "/search/reverse-image"
            or path.startswith("/search/reverse-image/")
            or path.startswith("/search/audio")
            or path.startswith("/search/video")
        )
    return False


def client_ip_from_request(request: Request) -> Optional[str]:
    """Resolve the client IP with Cloudflare's headers taking precedence."""
    for header in ("CF-Connecting-IP", "True-Client-IP"):
        value = request.headers.get(header)
        if value:
            return normalize_ip(value.split(",", 1)[0])
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return normalize_ip(forwarded.split(",", 1)[0])
    client = request.client
    return normalize_ip(client.host) if client else None


def _first_header(request: Request, names: Sequence[str]) -> Optional[str]:
    for name in names:
        value = request.headers.get(name)
        if value:
            return value.strip()
    return None


async def extract_search_query(request: Request) -> Optional[str]:
    """Best-effort query extraction without breaking the downstream body.

    Starlette's ``BaseHTTPMiddleware`` caches a body read in dispatch and replays
    it to the application, so JSON/form reads here are safe.  Multipart uploads
    (reverse image search) and large bodies are skipped.
    """
    for key in ("q", "query", "text", "prompt"):
        value = request.query_params.get(key)
        if value:
            return value

    content_type = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" in content_type:
        return None
    length = request.headers.get("content-length")
    if length:
        try:
            if int(length) > 65_536:
                return None
        except ValueError:
            pass

    try:
        body = await request.body()
    except Exception:
        return None
    if not body:
        return None

    if "application/json" in content_type:
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return None
        if isinstance(payload, dict):
            for key in ("q", "query", "text", "prompt"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value
    elif "application/x-www-form-urlencoded" in content_type:
        try:
            values = parse_qs(body.decode("utf-8", "replace"))
        except Exception:
            return None
        for key in ("q", "query", "text", "prompt"):
            if values.get(key):
                return values[key][0]
    return None


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class AnalyticsStore:
    """SQLite-backed event mirror + daily rollups.

    The class is cheap to construct and lazy: the schema is created on the first
    read/write, never at import.  Every public method is safe to call from
    multiple threads/processes (SQLite WAL + busy timeout).
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        raw = db_path if db_path is not None else _env("OMURA_ANALYTICS_DB_PATH", DEFAULT_DB_PATH)
        self.path = Path(raw).expanduser()
        self._schema_lock = threading.Lock()
        self._schema_ready = False

    # -- schema / connection -------------------------------------------------

    def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._schema_lock:
            if self._schema_ready:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.path), timeout=30)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA busy_timeout=30000")
                conn.executescript(_SCHEMA)
                conn.commit()
            finally:
                conn.close()
            self._schema_ready = True

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._ensure_schema()
        conn = sqlite3.connect(str(self.path), timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # -- ingestion -----------------------------------------------------------

    def _prepare_event(
        self,
        *,
        endpoint: str,
        method: str = "GET",
        status: int = 0,
        latency_ms: float = 0.0,
        ip: Optional[str] = None,
        ip_hash: Optional[str] = None,
        country: Optional[str] = None,
        region: Optional[str] = None,
        asn: Optional[Union[str, int]] = None,
        user_agent: Optional[str] = None,
        referer: Optional[str] = None,
        tenant: Optional[str] = None,
        query: Optional[str] = None,
        ts: Optional[Union[str, int, float, datetime]] = None,
        meta: Optional[Dict[str, Any]] = None,
        salt: Optional[str] = None,
        exclusions: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, ...]:
        timestamp = normalize_timestamp(ts)
        day = timestamp[:10]
        info = exclusions if exclusions is not None else _exclusions()
        if salt is None:
            salt, _ = salt_info()

        digest = ip_hash if ip_hash is not None else hash_ip(ip, salt=salt)
        normalised_country = _country(country)
        normalised_asn = normalize_asn(asn)
        reason = _exclusion_reason(ip, normalised_country, normalised_asn, info)
        bot = 1 if is_bot_ua(user_agent) else 0

        meta_dict: Dict[str, Any] = dict(meta or {})
        meta_dict.setdefault("ua_class", ua_class(user_agent))
        meta_dict.setdefault("source", "local")

        endpoint = (endpoint or "/")[:512]
        return (
            timestamp,
            day,
            endpoint,
            (method or "GET").upper()[:16],
            int(status or 0),
            float(latency_ms or 0.0),
            digest,
            normalised_country,
            (region or None) and str(region)[:64],
            normalised_asn,
            (user_agent or None) and str(user_agent)[:512],
            (referer or None) and str(referer)[:2048],
            (tenant or None) and str(tenant)[:128],
            hash_query(query, salt=salt),
            len(query.strip()) if query else 0,
            bot,
            1 if reason else 0,
            reason,
            json.dumps(meta_dict, ensure_ascii=False, default=str)[:8192],
        )

    def record(self, **event: Any) -> bool:
        """Insert one event.  Returns ``True`` when a new row was written."""
        return self.record_many([event]) > 0

    def record_many(self, events: Iterable[Dict[str, Any]]) -> int:
        """Insert many events in one transaction; duplicates are ignored.

        Deduplication key is ``(ts, ip_hash, endpoint)`` via the table's UNIQUE
        constraint.  Returns the number of rows actually inserted.
        """
        salt, _ = salt_info()
        info = _exclusions()
        prepared = [
            self._prepare_event(salt=salt, exclusions=info, **event)
            for event in events
        ]
        if not prepared:
            return 0
        with self._connect() as conn:
            before = conn.total_changes
            conn.executemany(_INSERT_EVENT, prepared)
            return conn.total_changes - before

    # -- aggregation ---------------------------------------------------------

    @staticmethod
    def _window(days: int) -> Tuple[str, str, int]:
        days = max(1, min(int(days), 3660))
        end = _utcnow().date()
        start = end - timedelta(days=days - 1)
        return start.isoformat(), end.isoformat(), days

    @staticmethod
    def _placeholders(values: Sequence[Any]) -> str:
        return ", ".join("?" for _ in values)

    def _clean_sql(self, info: Dict[str, Any], alias: str = "") -> Tuple[str, List[Any]]:
        prefix = f"{alias}." if alias else ""
        conditions = [
            f"COALESCE({prefix}is_bot, 0) = 0",
            f"COALESCE({prefix}is_excluded, 0) = 0",
        ]
        params: List[Any] = []
        if info["countries"]:
            conditions.append(
                f"UPPER(COALESCE({prefix}country, '')) NOT IN ({self._placeholders(info['countries'])})"
            )
            params.extend(info["countries"])
        if info["asns"]:
            conditions.append(
                f"COALESCE({prefix}asn, '') NOT IN ({self._placeholders(info['asns'])})"
            )
            params.extend(info["asns"])
        if info["exact_hashes"]:
            conditions.append(
                f"COALESCE({prefix}ip_hash, '') NOT IN ({self._placeholders(info['exact_hashes'])})"
            )
            params.extend(info["exact_hashes"])
        return " AND ".join(conditions), params

    def _scalar(
        self, conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()
    ) -> int:
        row = conn.execute(sql, params).fetchone()
        return int(row[0] or 0) if row else 0

    def _excluded_counts(
        self, conn: sqlite3.Connection, start: str, end: str, info: Dict[str, Any]
    ) -> Dict[str, Any]:
        base = "SELECT COUNT(*) FROM search_events WHERE day >= ? AND day <= ?"
        base_params: List[Any] = [start, end]

        bots = self._scalar(
            conn,
            base + " AND COALESCE(is_bot, 0) = 1",
            base_params,
        )

        countries = 0
        if info["countries"]:
            countries = self._scalar(
                conn,
                base
                + f" AND UPPER(COALESCE(country, '')) IN ({self._placeholders(info['countries'])})",
                base_params + list(info["countries"]),
            )

        ips = 0
        ip_parts = ["COALESCE(exclude_reason, '') = 'ip'"]
        ip_params: List[Any] = []
        if info["exact_hashes"]:
            ip_parts.append(
                f"COALESCE(ip_hash, '') IN ({self._placeholders(info['exact_hashes'])})"
            )
            ip_params.extend(info["exact_hashes"])
        ips = self._scalar(
            conn,
            base + " AND (" + " OR ".join(ip_parts) + ")",
            base_params + ip_params,
        )

        asns = 0
        if info["asns"]:
            asns = self._scalar(
                conn,
                base
                + f" AND COALESCE(asn, '') IN ({self._placeholders(info['asns'])})",
                base_params + list(info["asns"]),
            )

        # Total events removed for *any* reason (bots and filters may overlap;
        # this is an unduplicated count).
        or_parts = ["COALESCE(is_bot, 0) = 1", "COALESCE(is_excluded, 0) = 1"]
        or_params: List[Any] = []
        if info["countries"]:
            or_parts.append(
                f"UPPER(COALESCE(country, '')) IN ({self._placeholders(info['countries'])})"
            )
            or_params.extend(info["countries"])
        if info["asns"]:
            or_parts.append(
                f"COALESCE(asn, '') IN ({self._placeholders(info['asns'])})"
            )
            or_params.extend(info["asns"])
        if info["exact_hashes"]:
            or_parts.append(
                f"COALESCE(ip_hash, '') IN ({self._placeholders(info['exact_hashes'])})"
            )
            or_params.extend(info["exact_hashes"])
        total = self._scalar(
            conn,
            base + " AND (" + " OR ".join(or_parts) + ")",
            base_params + or_params,
        )

        return {
            "bots": bots,
            "countries": countries,
            "ips": ips,
            "asns": asns,
            "total_events": total,
            "country_codes": list(info["countries"]),
            "asn_codes": list(info["asns"]),
            "exclude_ip_count": len(info["networks"]),
            "note": (
                "The same event may appear in more than one category; "
                "total_events is unduplicated. CIDR/IP rules are evaluated at "
                "ingestion; country/ASN/exact-IP rules are also re-applied at read time."
            ),
        }

    def _aggregate(self, start: str, end: str) -> Dict[str, Any]:
        info = _exclusions()
        clean, clean_params = self._clean_sql(info)
        with self._connect() as conn:
            row = conn.execute(
                f"""
                WITH window AS (
                    SELECT *,
                           CASE WHEN {clean} THEN 1 ELSE 0 END AS is_clean
                    FROM search_events
                    WHERE day >= ? AND day <= ?
                )
                SELECT
                    COUNT(*) AS total_requests,
                    COALESCE(SUM(is_clean), 0) AS search_queries,
                    COUNT(DISTINCT CASE
                        WHEN is_clean = 1 AND ip_hash IS NOT NULL AND ip_hash <> ''
                        THEN ip_hash END) AS mau,
                    COUNT(DISTINCT CASE
                        WHEN is_clean = 1 AND ip_hash IS NOT NULL AND ip_hash <> ''
                        THEN ip_hash || '|' || COALESCE(
                            CASE WHEN json_valid(meta) THEN json_extract(meta, '$.ua_class') END,
                            ''
                        ) END) AS unique_visitors,
                    COUNT(DISTINCT CASE
                        WHEN ip_hash IS NOT NULL AND ip_hash <> ''
                        THEN ip_hash END) AS unique_ips
                FROM window
                """,
                [*clean_params, start, end],
            ).fetchone()

            country_rows = conn.execute(
                f"""
                SELECT
                    UPPER(COALESCE(NULLIF(country, ''), 'unknown')) AS country,
                    COUNT(DISTINCT CASE
                        WHEN ip_hash IS NOT NULL AND ip_hash <> '' THEN ip_hash END
                    ) AS uniques,
                    COUNT(*) AS queries
                FROM search_events
                WHERE day >= ? AND day <= ? AND {clean}
                GROUP BY 1
                ORDER BY uniques DESC, queries DESC
                """,
                [start, end, *clean_params],
            ).fetchall()

            excluded = self._excluded_counts(conn, start, end, info)

        countries = [
            {
                "country": r["country"],
                "uniques": int(r["uniques"] or 0),
                "queries": int(r["queries"] or 0),
            }
            for r in country_rows
        ]
        return {
            "mau": int(row["mau"] or 0) if row else 0,
            "unique_visitors": int(row["unique_visitors"] or 0) if row else 0,
            "search_queries": int(row["search_queries"] or 0) if row else 0,
            "total_requests": int(row["total_requests"] or 0) if row else 0,
            "unique_ips": int(row["unique_ips"] or 0) if row else 0,
            "countries": countries,
            "excluded": excluded,
        }

    # -- public reporting ----------------------------------------------------

    def traction(self, days: int = 30) -> Dict[str, Any]:
        """Return the grant-facing traction report.

        Primary MAU definition (``mau``): distinct ``ip_hash`` in the rolling
        window after bot/country/ASN/IP exclusions.  Secondary definition
        (``unique_visitors``): distinct ``(ip_hash, UA-class)`` pairs, which is
        stricter against one client cycling user agents.
        """
        start, end, days = self._window(days)
        aggregate = self._aggregate(start, end)
        salt, salt_source = salt_info()
        verified = bool(salt)

        mau_target = _int_env("OMURA_ANALYTICS_MAU_TARGET", 50)
        query_target = _int_env("OMURA_ANALYTICS_QUERY_TARGET", 5000)

        # Cloudflare window totals (dashboard / GraphQL rollup) are the
        # authoritative visitor count when present: the local mirror only sees
        # traffic since the middleware was deployed.
        cf_window = self.latest_cloudflare_window(start, end)
        local_mau = int(aggregate["mau"])
        cf_mau = int(cf_window["unique_visitors"]) if cf_window else 0
        mau = max(local_mau, cf_mau)
        mau_source = "cloudflare" if cf_mau > local_mau else "local"

        local_queries = int(aggregate["search_queries"])
        cf_queries = (
            int(cf_window.get("search_queries") or 0) if cf_window else 0
        )
        search_queries = max(local_queries, cf_queries)
        queries_source = "cloudflare" if cf_queries > local_queries else "local"

        notes = [
            "Source of truth: Cloudflare zone analytics + Logpush; this report reads "
            "the local mirror populated by AnalyticsMiddleware and "
            "scripts/ingest_cloudflare_logpush.py.",
            "mau = distinct salted ip_hash over the rolling UTC window after "
            "bot/country/ASN/IP exclusions (primary definition).",
            "unique_visitors = distinct (ip_hash, user-agent class) pairs after the "
            "same exclusions (secondary, stricter definition).",
            "unique_ips includes excluded/bot hashes and is shown for diagnostics only.",
            "search_queries counts only POST /search* and the tracked GET "
            "/search/audio*, /search/video*, /search/reverse-image* endpoints.",
            "total_requests counts every mirrored tracked event before exclusions; "
            "it is not the zone-wide Cloudflare request total.",
            "Bots are identified by user-agent regex (curl, python-requests, "
            "spiders/monitors/uptime probes, headless browsers, common CI runners); "
            "a missing user agent is treated as a bot.",
            "IP hashing is salted SHA-256; raw client IPs are never stored.",
            "MAU is IP-based: shared NAT can undercount and IPv6 rotation can "
            "overcount; the (ip_hash, UA-class) definition is the stricter check.",
            "Country/ASN values come from Cloudflare headers or Logpush fields, not "
            "from a local GeoIP database.",
            "CIDR exclusions are applied when an event is ingested. Country, ASN and "
            "exact-IP exclusions are also applied at read time.",
            f"verification={'enabled' if verified else 'DISABLED'}"
            f" (salt source: {salt_source or 'none'}).",
            "Cloudflare's daily GraphQL rollups have no request-path dimension, so "
            "path-filtered query counts must come from Logpush events.",
            "The local mirror only contains events recorded after "
            "AnalyticsMiddleware was deployed; historical coverage must come from "
            "Cloudflare windows (dashboard/GraphQL) or Logpush exports. Access-log "
            "files are not a complete source.",
            "When a Cloudflare window is present it is used for MAU; the report "
            "shows mau_source and cloudflare_window so the provenance is explicit.",
        ]

        return {
            "window": {
                "days": days,
                "start": start,
                "end": end,
                "timezone": "UTC",
                "generated_at": _utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            },
            "mau": mau,
            "mau_source": mau_source,
            "local_mau": local_mau,
            "cloudflare_mau": cf_mau,
            "unique_visitors": aggregate["unique_visitors"],
            "search_queries": search_queries,
            "search_queries_source": queries_source,
            "local_search_queries": local_queries,
            "cloudflare_search_queries": cf_queries,
            "total_requests": aggregate["total_requests"],
            "unique_ips": aggregate["unique_ips"],
            "countries": aggregate["countries"],
            "excluded": aggregate["excluded"],
            "cloudflare_window": cf_window,
            "method_notes": notes,
            "verified": verified,
            "targets": {"mau": mau_target, "search_queries": query_target},
            "meets_mau": mau >= mau_target,
            "meets_queries": search_queries >= query_target,
        }

    def rollup_local(self, day: str) -> Dict[str, Any]:
        """Recompute and upsert the local daily rollup for *day*."""
        day = str(day)[:10]
        aggregate = self._aggregate(day, day)
        generated_at = _utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        countries = aggregate["countries"]
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO daily_rollups
                    (day, source, unique_visitors, search_queries, total_requests,
                     countries_json, generated_at)
                VALUES (?, 'local', ?, ?, ?, ?, ?)
                ON CONFLICT(day, source) DO UPDATE SET
                    unique_visitors = excluded.unique_visitors,
                    search_queries = excluded.search_queries,
                    total_requests = excluded.total_requests,
                    countries_json = excluded.countries_json,
                    generated_at = excluded.generated_at
                """,
                (
                    day,
                    aggregate["mau"],
                    aggregate["search_queries"],
                    aggregate["total_requests"],
                    json.dumps(countries, ensure_ascii=False),
                    generated_at,
                ),
            )
        return {
            "day": day,
            "source": "local",
            "unique_visitors": aggregate["mau"],
            "search_queries": aggregate["search_queries"],
            "total_requests": aggregate["total_requests"],
            "countries": countries,
            "generated_at": generated_at,
        }

    def upsert_cloudflare_rollup(
        self,
        day: str,
        unique_visitors: int,
        total_requests: int,
        countries: Optional[Sequence[Dict[str, Any]]] = None,
        search_queries: int = 0,
        generated_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upsert a Cloudflare daily rollup (``source='cloudflare'``).

        ``search_queries`` is normally 0 because the zone GraphQL dataset has no
        request-path dimension; pass a Logpush-derived value if one is available.
        """
        day = str(day)[:10]
        generated_at = generated_at or _utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        payload = list(countries or [])
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO daily_rollups
                    (day, source, unique_visitors, search_queries, total_requests,
                     countries_json, generated_at)
                VALUES (?, 'cloudflare', ?, ?, ?, ?, ?)
                ON CONFLICT(day, source) DO UPDATE SET
                    unique_visitors = excluded.unique_visitors,
                    search_queries = excluded.search_queries,
                    total_requests = excluded.total_requests,
                    countries_json = excluded.countries_json,
                    generated_at = excluded.generated_at
                """,
                (
                    day,
                    int(unique_visitors or 0),
                    int(search_queries or 0),
                    int(total_requests or 0),
                    json.dumps(payload, ensure_ascii=False),
                    generated_at,
                ),
            )
        return {
            "day": day,
            "source": "cloudflare",
            "unique_visitors": int(unique_visitors or 0),
            "search_queries": int(search_queries or 0),
            "total_requests": int(total_requests or 0),
            "countries": payload,
            "generated_at": generated_at,
        }

    def _series(self, days: int, source: str) -> List[Dict[str, Any]]:
        start, end, _ = self._window(days)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT day, source, unique_visitors, search_queries, total_requests,
                       countries_json, generated_at
                FROM daily_rollups
                WHERE source = ? AND day >= ? AND day <= ?
                ORDER BY day ASC
                """,
                (source, start, end),
            ).fetchall()
        out = []
        for row in rows:
            try:
                countries = json.loads(row["countries_json"] or "[]")
            except (ValueError, TypeError):
                countries = []
            out.append(
                {
                    "day": row["day"],
                    "source": row["source"],
                    "unique_visitors": int(row["unique_visitors"] or 0),
                    "search_queries": int(row["search_queries"] or 0),
                    "total_requests": int(row["total_requests"] or 0),
                    "countries": countries,
                    "generated_at": row["generated_at"],
                }
            )
        return out

    def cloudflare_series(self, days: int = 30) -> List[Dict[str, Any]]:
        return self._series(days, "cloudflare")

    def record_cloudflare_window(
        self,
        start_day: str,
        end_day: str,
        unique_visitors: int,
        total_requests: int,
        *,
        search_queries: Optional[int] = None,
        countries: Optional[Sequence[Dict[str, Any]]] = None,
        source: str = "dashboard",
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record a Cloudflare **window** total (e.g. the 30-day dashboard).

        Window uniques are not additive across days, which is why they are kept
        out of ``daily_rollups``. ``search_queries`` is optional: the zone
        overview does not break requests down by path, so leave it ``None``
        unless the figure was obtained from Logpush or a path-filtered view.
        """
        start_day = str(start_day)[:10]
        end_day = str(end_day)[:10]
        recorded_at = _utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        payload = list(countries or [])
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cloudflare_windows
                    (start_day, end_day, unique_visitors, search_queries,
                     total_requests, countries_json, source, note, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(start_day, end_day, source) DO UPDATE SET
                    unique_visitors = excluded.unique_visitors,
                    search_queries = excluded.search_queries,
                    total_requests = excluded.total_requests,
                    countries_json = excluded.countries_json,
                    note = excluded.note,
                    recorded_at = excluded.recorded_at
                """,
                (
                    start_day,
                    end_day,
                    int(unique_visitors or 0),
                    None if search_queries is None else int(search_queries),
                    int(total_requests or 0),
                    json.dumps(payload, ensure_ascii=False),
                    str(source),
                    note,
                    recorded_at,
                ),
            )
        return self.latest_cloudflare_window(start_day, end_day) or {}

    def latest_cloudflare_window(
        self, start_day: str, end_day: str
    ) -> Optional[Dict[str, Any]]:
        """Newest recorded Cloudflare window that overlaps the report window."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT start_day, end_day, unique_visitors, search_queries,
                       total_requests, countries_json, source, note, recorded_at
                FROM cloudflare_windows
                WHERE end_day >= ? AND start_day <= ?
                ORDER BY end_day DESC, recorded_at DESC
                LIMIT 1
                """,
                (str(start_day)[:10], str(end_day)[:10]),
            ).fetchone()
        if row is None:
            return None
        try:
            countries = json.loads(row["countries_json"] or "[]")
        except (ValueError, TypeError):
            countries = []
        return {
            "start_day": row["start_day"],
            "end_day": row["end_day"],
            "unique_visitors": int(row["unique_visitors"] or 0),
            "search_queries": row["search_queries"],
            "total_requests": int(row["total_requests"] or 0),
            "countries": countries,
            "source": row["source"],
            "note": row["note"],
            "recorded_at": row["recorded_at"],
        }

    def local_series(self, days: int = 30) -> List[Dict[str, Any]]:
        return self._series(days, "local")

    def events(self, limit: int = 100, day: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return recent events (newest first) for the admin endpoint."""
        limit = max(1, min(int(limit), 1000))
        params: List[Any] = []
        where = ""
        if day:
            where = "WHERE day = ?"
            params.append(str(day)[:10])
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT id, ts, day, endpoint, method, status, latency_ms, ip_hash,
                       country, region, asn, user_agent, referer, tenant,
                       query_hash, query_len, is_bot, is_excluded, exclude_reason, meta
                FROM search_events
                {where}
                ORDER BY id DESC
                LIMIT ?
                """,
                params,
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            try:
                item["meta"] = json.loads(item.get("meta") or "{}")
            except (ValueError, TypeError):
                item["meta"] = {}
            out.append(item)
        return out


# ---------------------------------------------------------------------------
# Default store registry (lazy; import-safe)
# ---------------------------------------------------------------------------

_STORES: Dict[str, AnalyticsStore] = {}
_STORES_LOCK = threading.Lock()


def reset_default_store() -> None:
    """Forget cached stores (used by tests when env paths change)."""
    with _STORES_LOCK:
        _STORES.clear()


def get_store(db_path: Optional[Union[str, Path]] = None) -> AnalyticsStore:
    """Return a cached store for *db_path* (or the configured default)."""
    raw = db_path if db_path is not None else _env("OMURA_ANALYTICS_DB_PATH", DEFAULT_DB_PATH)
    path = Path(raw).expanduser()
    try:
        key = str(path.resolve())
    except OSError:
        key = str(path)
    with _STORES_LOCK:
        store = _STORES.get(key)
        if store is None:
            store = AnalyticsStore(path)
            _STORES[key] = store
        return store


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class AnalyticsMiddleware(BaseHTTPMiddleware):
    """Mirror tracked search requests into :class:`AnalyticsStore`.

    Recording is best-effort and never raises into the request path.  The
    middleware only buffers a query body for small JSON/form POSTs; multipart
    uploads are ignored.
    """

    def __init__(
        self,
        app: Any,
        store: Optional[AnalyticsStore] = None,
        enabled: Optional[bool] = None,
        capture_body: bool = True,
    ) -> None:
        super().__init__(app)
        self.store = store
        self.enabled = enabled
        self.capture_body = capture_body

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        enabled = self.enabled if self.enabled is not None else _env_bool(
            "OMURA_ANALYTICS_ENABLED", True
        )
        tracked = enabled and is_tracked_request(request.method, request.url.path)

        query: Optional[str] = None
        if tracked and self.capture_body and request.method.upper() == "POST":
            try:
                query = await extract_search_query(request)
            except Exception:
                query = None

        started = time.perf_counter()
        response = await call_next(request)

        if not tracked:
            return response

        try:
            from starlette.concurrency import run_in_threadpool

            store = self.store or get_store()
            # SQLite writes are blocking; keep them off the event loop so a
            # busy search path never queues behind analytics.
            await run_in_threadpool(
                store.record,
                endpoint=request.url.path,
                method=request.method,
                status=response.status_code,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                ip=client_ip_from_request(request),
                country=_first_header(request, ("CF-IPCountry", "X-Country-Code")),
                region=_first_header(request, ("CF-Region-Code", "CF-Region")),
                asn=_first_header(request, ("CF-ASN",)),
                user_agent=request.headers.get("user-agent"),
                referer=request.headers.get("referer"),
                tenant=_first_header(
                    request, ("X-Omura-Tenant", "X-Tenant-ID", "X-Tenant")
                ),
                query=query,
                meta={"source": "middleware"},
            )
        except Exception as exc:  # never break search because analytics failed
            print(f"[analytics] middleware record failed: {exc}")
        return response


__all__ = [
    "AnalyticsMiddleware",
    "AnalyticsStore",
    "DEFAULT_DB_PATH",
    "client_ip_from_request",
    "extract_search_query",
    "get_store",
    "hash_ip",
    "hash_query",
    "is_bot_ua",
    "is_tracked_request",
    "normalize_asn",
    "normalize_ip",
    "normalize_timestamp",
    "reset_default_store",
    "salt_info",
    "ua_class",
]
