"""Tenant registry + usage accounting for Phase 3 pilot projects.

Storage lives in a small SQLite database (``OMURA_TENANCY_DB_PATH``, default
``data/tenants.sqlite``) and is created lazily on first use. Importing this
module never touches the filesystem, so it is safe to import from the API,
scripts, and tests.

API keys
--------
``register()`` mints a key of the form ``omu_`` + 32 hex characters and returns
it **exactly once**. Only the SHA-256 digest and the first 8 characters (for
display) are persisted. ``verify_key()`` compares digests with
``hmac.compare_digest`` and returns the public tenant record for pending or
active tenants; suspended tenants are rejected.

Usage
-----
``tenant_usage`` stores one aggregate row per ``(tenant_id, day, endpoint)``;
``count`` is incremented on every call. ``day`` is an ISO UTC date and ``ts`` is
the last time that row was touched. ``stats()`` derives active days, distinct
ISO weeks, and the ``meets_recurring`` flag (>= 20 active days in the window
**or** usage in >= 4 distinct ISO weeks).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

DEFAULT_DB_PATH = "data/tenants.sqlite"
VALID_STATUSES = ("pending", "active", "suspended")
KEY_PREFIX = "omu_"
KEY_HEX_BYTES = 16  # 32 hex characters
KEY_PREFIX_DISPLAY = 8

_CREATE_TENANTS = """
CREATE TABLE IF NOT EXISTS tenants (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    slug            TEXT NOT NULL UNIQUE,
    contact         TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',
    api_key_hash    TEXT NOT NULL,
    api_key_prefix  TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    activated_at    TEXT,
    notes           TEXT,
    meta            TEXT NOT NULL DEFAULT '{}'
)
"""

_CREATE_USAGE = """
CREATE TABLE IF NOT EXISTS tenant_usage (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id   TEXT NOT NULL,
    ts          TEXT NOT NULL,
    day         TEXT NOT NULL,
    endpoint    TEXT NOT NULL,
    count       INTEGER NOT NULL DEFAULT 0,
    meta        TEXT,
    UNIQUE(tenant_id, day, endpoint),
    FOREIGN KEY(tenant_id) REFERENCES tenants(id) ON DELETE CASCADE
)
"""

_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_tenant_usage_tenant_day "
    "ON tenant_usage(tenant_id, day)",
    "CREATE INDEX IF NOT EXISTS idx_tenant_usage_day ON tenant_usage(day)",
)

_init_lock = threading.Lock()
_initialized_paths: set[str] = set()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _now_iso() -> str:
    return _utcnow().isoformat()


def _today() -> str:
    return _utcnow().date().isoformat()


def db_path() -> Path:
    """Resolve the tenancy DB path from the environment on every call.

    Resolving lazily (instead of a module constant) lets tests point
    ``OMURA_TENANCY_DB_PATH`` at a ``tmp_path`` after import.
    """
    return Path(os.getenv("OMURA_TENANCY_DB_PATH", DEFAULT_DB_PATH))


def _ensure_schema(path: Path) -> None:
    key = str(path.resolve())
    if key in _initialized_paths:
        return
    with _init_lock:
        if key in _initialized_paths:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(path), timeout=30) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(_CREATE_TENANTS)
            conn.execute(_CREATE_USAGE)
            for statement in _INDEXES:
                conn.execute(statement)
            conn.commit()
        _initialized_paths.add(key)


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    path = db_path()
    _ensure_schema(path)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower()).strip("-")
    return slug or "tenant"


def _unique_slug(conn: sqlite3.Connection, base: str) -> str:
    candidate = base
    suffix = 2
    while conn.execute("SELECT 1 FROM tenants WHERE slug = ?", (candidate,)).fetchone():
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def _public(row: sqlite3.Row | Dict[str, Any]) -> Dict[str, Any]:
    """Return a tenant dict without the key digest."""
    data = dict(row)
    data.pop("api_key_hash", None)
    try:
        data["meta"] = json.loads(data.get("meta") or "{}")
    except (TypeError, ValueError):
        data["meta"] = {}
    return data


def _row_to_tenant(row: sqlite3.Row) -> Dict[str, Any]:
    return _public(row)


def register(
    name: str,
    contact: Optional[str] = None,
    *,
    slug: Optional[str] = None,
    notes: Optional[str] = None,
    status: str = "pending",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a tenant and return ``{"tenant": ..., "api_key": ...}`` once."""
    clean_name = (name or "").strip()
    if not clean_name:
        raise ValueError("name is required")
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {status}")

    api_key = KEY_PREFIX + secrets.token_hex(KEY_HEX_BYTES)
    key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    now = _now_iso()
    tenant_id = uuid.uuid4().hex

    with _connect() as conn:
        tenant_slug = _unique_slug(conn, _slugify(slug or clean_name))
        conn.execute(
            """
            INSERT INTO tenants (
                id, name, slug, contact, status, api_key_hash, api_key_prefix,
                created_at, activated_at, notes, meta
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tenant_id,
                clean_name,
                tenant_slug,
                contact,
                status,
                key_hash,
                api_key[:KEY_PREFIX_DISPLAY],
                now,
                now if status == "active" else None,
                notes,
                json.dumps(meta or {}, ensure_ascii=False, sort_keys=True),
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM tenants WHERE id = ?", (tenant_id,)).fetchone()

    return {"tenant": _row_to_tenant(row), "api_key": api_key}


def get_tenant(tenant_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM tenants WHERE id = ?", (tenant_id,)).fetchone()
    return _row_to_tenant(row) if row else None


def verify_key(key: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return the tenant for a valid key, or ``None``.

    Both ``pending`` and ``active`` tenants verify; ``suspended`` tenants do
    not. The digest comparison is constant-time.
    """
    if not isinstance(key, str) or not key.startswith(KEY_PREFIX) or len(key) < 12:
        return None
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    prefix = key[:KEY_PREFIX_DISPLAY]
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tenants WHERE api_key_prefix = ?", (prefix,)
        ).fetchall()
    for row in rows:
        if hmac.compare_digest(row["api_key_hash"], digest) and row["status"] != "suspended":
            return _public(row)
    return None


def list_tenants() -> List[Dict[str, Any]]:
    """List tenant records (never includes key digests)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tenants ORDER BY created_at ASC, id ASC"
        ).fetchall()
    return [_row_to_tenant(row) for row in rows]


def set_status(tenant_id: str, status: str) -> Optional[Dict[str, Any]]:
    """Update a tenant's status. Returns the updated record, or ``None``."""
    if status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {status}")
    now = _now_iso()
    with _connect() as conn:
        if status == "active":
            conn.execute(
                "UPDATE tenants SET status = ?, activated_at = COALESCE(activated_at, ?) "
                "WHERE id = ?",
                (status, now, tenant_id),
            )
        else:
            conn.execute(
                "UPDATE tenants SET status = ? WHERE id = ?", (status, tenant_id)
            )
        conn.commit()
        row = conn.execute("SELECT * FROM tenants WHERE id = ?", (tenant_id,)).fetchone()
    return _row_to_tenant(row) if row else None


def delete(tenant_id: str) -> bool:
    """Hard-delete a tenant and its usage rows. Returns ``True`` if found.

    The ``DELETE /pilot/tenants/{id}`` route does a *soft* delete
    (``status='suspended'``); this helper exists for operator cleanup.
    """
    with _connect() as conn:
        conn.execute("DELETE FROM tenant_usage WHERE tenant_id = ?", (tenant_id,))
        cur = conn.execute("DELETE FROM tenants WHERE id = ?", (tenant_id,))
        conn.commit()
        return cur.rowcount > 0


def record_usage(
    tenant_id: str,
    endpoint: str,
    n: int = 1,
    *,
    day: Optional[str] = None,
    ts: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> int:
    """Increment the ``(tenant, day, endpoint)`` counter.

    ``day``/``ts`` are normally derived from the current UTC time; they can be
    supplied to backfill historical usage in tests or imports. Returns the
    updated counter for that row.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    now = ts or _now_iso()
    usage_day = day or _today()
    payload = json.dumps(meta, ensure_ascii=False, sort_keys=True) if meta is not None else None

    with _connect() as conn:
        exists = conn.execute(
            "SELECT 1 FROM tenants WHERE id = ?", (tenant_id,)
        ).fetchone()
        if not exists:
            raise KeyError(f"unknown tenant: {tenant_id}")
        conn.execute(
            """
            INSERT INTO tenant_usage (tenant_id, ts, day, endpoint, count, meta)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(tenant_id, day, endpoint) DO UPDATE SET
                count = tenant_usage.count + excluded.count,
                ts = excluded.ts,
                meta = COALESCE(excluded.meta, tenant_usage.meta)
            """,
            (tenant_id, now, usage_day, endpoint, int(n), payload),
        )
        conn.commit()
        row = conn.execute(
            "SELECT count FROM tenant_usage "
            "WHERE tenant_id = ? AND day = ? AND endpoint = ?",
            (tenant_id, usage_day, endpoint),
        ).fetchone()
    return int(row["count"]) if row else int(n)


def stats(tenant_id: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
    """Usage statistics for one tenant or all tenants.

    ``active_days`` counts distinct UTC days with any usage inside the window
    (today inclusive). ``distinct_weeks`` counts distinct ISO ``(year, week)``
    pairs in the window. ``meets_recurring`` is true when a tenant has at least
    20 active days **or** usage in at least 4 distinct weeks.
    """
    if days < 1:
        raise ValueError("days must be >= 1")

    end = _utcnow().date()
    start = end - timedelta(days=days - 1)
    start_s = start.isoformat()
    end_s = end.isoformat()

    with _connect() as conn:
        if tenant_id:
            tenant_rows = conn.execute(
                "SELECT * FROM tenants WHERE id = ?", (tenant_id,)
            ).fetchall()
        else:
            tenant_rows = conn.execute(
                "SELECT * FROM tenants ORDER BY created_at ASC, id ASC"
            ).fetchall()

        if tenant_id:
            usage_rows = conn.execute(
                """
                SELECT tenant_id, day, endpoint, count
                FROM tenant_usage
                WHERE tenant_id = ? AND day >= ? AND day <= ?
                """,
                (tenant_id, start_s, end_s),
            ).fetchall()
            last_rows = conn.execute(
                "SELECT tenant_id, MAX(ts) AS last_seen FROM tenant_usage "
                "WHERE tenant_id = ? GROUP BY tenant_id",
                (tenant_id,),
            ).fetchall()
        else:
            usage_rows = conn.execute(
                """
                SELECT tenant_id, day, endpoint, count
                FROM tenant_usage
                WHERE day >= ? AND day <= ?
                """,
                (start_s, end_s),
            ).fetchall()
            last_rows = conn.execute(
                "SELECT tenant_id, MAX(ts) AS last_seen FROM tenant_usage "
                "GROUP BY tenant_id"
            ).fetchall()

    last_seen = {row["tenant_id"]: row["last_seen"] for row in last_rows}
    per_tenant: Dict[str, Dict[str, Any]] = {}
    for row in usage_rows:
        bucket = per_tenant.setdefault(
            row["tenant_id"],
            {"active_days": set(), "weeks": set(), "total_usage": 0, "endpoints": {}},
        )
        bucket["active_days"].add(row["day"])
        try:
            bucket["weeks"].add(date.fromisoformat(row["day"]).isocalendar()[:2])
        except ValueError:
            pass
        bucket["total_usage"] += int(row["count"] or 0)
        endpoint = row["endpoint"] or "unknown"
        bucket["endpoints"][endpoint] = bucket["endpoints"].get(endpoint, 0) + int(
            row["count"] or 0
        )

    tenants: List[Dict[str, Any]] = []
    for row in tenant_rows:
        tenant = _row_to_tenant(row)
        bucket = per_tenant.get(tenant["id"], {})
        active_days = len(bucket.get("active_days", set()))
        distinct_weeks = len(bucket.get("weeks", set()))
        tenants.append(
            {
                **tenant,
                "last_seen": last_seen.get(tenant["id"]),
                "active_days": active_days,
                "distinct_weeks": distinct_weeks,
                "total_usage": int(bucket.get("total_usage", 0)),
                "endpoints": dict(sorted(bucket.get("endpoints", {}).items())),
                "meets_recurring": active_days >= 20 or distinct_weeks >= 4,
            }
        )

    tenants.sort(key=lambda t: (-t["total_usage"], t["created_at"]))
    recurring = [t["id"] for t in tenants if t["meets_recurring"]]
    result: Dict[str, Any] = {
        "days": days,
        "window_start": start_s,
        "window_end": end_s,
        "generated_at": _now_iso(),
        "total_tenants": len(tenants),
        "active_tenants": sum(1 for t in tenants if t["status"] == "active"),
        "recurring_tenants": recurring,
        "meets_recurring": bool(recurring),
        "tenants": tenants,
    }
    if tenant_id and tenants:
        result["tenant"] = tenants[0]
    return result


__all__ = [
    "DEFAULT_DB_PATH",
    "VALID_STATUSES",
    "db_path",
    "register",
    "get_tenant",
    "verify_key",
    "list_tenants",
    "set_status",
    "delete",
    "record_usage",
    "stats",
]
