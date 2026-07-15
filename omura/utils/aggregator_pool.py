"""Multi-upstream Walrus aggregator pool with health-aware smart routing + retries.

Walrus blob fetches are I/O-heavy and slow per-node. Spreading load across several
public aggregators in parallel lifts throughput linearly with the number of upstreams.

Config (env, comma-separated; first non-empty wins):
  WALRUS_AGGREGATOR_URLS   pool of upstreams, e.g. ``https://a/, https://b/``
  WALRUS_AGGREGATOR_URL    single fallback (back-compat with existing code)

If neither is set, the full DEFAULT_POOL is used (12 public nodes + omura tunnel).

Routing strategy:
  • Lowest score wins → score = ema_latency * (1 + in_flight)
  • After _FAILURES_TO_TRIP consecutive errors, upstream is cooled down for 60 s
  • startup_ping() pre-warms latency scores so the fastest nodes lead from request 1
"""

from __future__ import annotations

import itertools
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests


# ── Fallback aggregator list ──────────────────────────────────────────────────
# Sorted roughly fastest→slowest based on community benchmarks; the startup
# ping re-ranks them anyway so the order only matters for the very first request.
DEFAULT_POOL = [
    # Omura Cloudflare tunnel — proxied, always try but not relied on exclusively
    "https://agrregator.omura.fun",
    # Walrus Foundation official
    "https://aggregator.walrus-mainnet.walrus.space",
    # Community aggregators — independently operated, geographically diverse
    "https://walrus-mainnet-aggregator.redundex.com",
    "https://wal-aggregator-mainnet.staketab.org",
    "https://walrus.blockscope.net",
    "https://aggregator.suicore.com",
    "https://walrus-agg.mainnet.obelisk.sh",
    "https://walrus-aggregator.thcloud.dev",
    "https://walrus.globalstake.io",
    "https://walrus-mainnet.nodeinfra.com",
    "https://walrus.natsai.xyz",
    "https://walrus-aggregator.nodes.guru",
]

_HEALTH_COOLDOWN_SECS = 60.0   # seconds to cool-down a tripped upstream
_FAILURES_TO_TRIP = 3          # consecutive failures before cooldown
_EMA_ALPHA = 0.25              # smoothing factor for rolling latency EMA
_DEFAULT_LATENCY = 1.0         # assumed latency before first measurement (seconds)
_PING_PATH = "/v1/blobs/1V7ZMSvAaTLKKDXqtQRCxWiXWLnJpfBXirPxYiT3k1Y"  # small known blob
_PING_TIMEOUT = 6.0            # seconds per startup ping attempt


# ── Per-upstream state ────────────────────────────────────────────────────────

@dataclass
class _Upstream:
    url: str
    consecutive_failures: int = 0
    tripped_until: float = 0.0
    total_requests: int = 0
    total_failures: int = 0
    ema_latency_secs: float = _DEFAULT_LATENCY
    in_flight: int = 0
    last_used_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def score(self, now: float) -> float:
        """Lower = better. Combines latency EMA and current in-flight count.

        A fast upstream with 4 in-flight requests is treated as 5x its baseline
        latency — this naturally spreads load without central coordination.
        """
        return self.ema_latency_secs * (1.0 + self.in_flight)


# ── Pool ──────────────────────────────────────────────────────────────────────

class AggregatorPool:
    """Thread-safe smart-routing pool over Walrus aggregator base URLs."""

    def __init__(self, urls: Optional[List[str]] = None):
        if urls is None:
            urls = self._urls_from_env()
        cleaned = [u.rstrip("/") for u in urls if u and u.strip()]
        if not cleaned:
            cleaned = [DEFAULT_POOL[0]]
        self.upstreams = [_Upstream(url=u) for u in cleaned]
        self._cursor = itertools.cycle(self.upstreams)
        self._cursor_lock = threading.Lock()

    @staticmethod
    def _urls_from_env() -> List[str]:
        raw = (os.getenv("WALRUS_AGGREGATOR_URLS") or "").strip()
        configured = [s.strip().rstrip("/") for s in raw.split(",") if s.strip()]
        if not configured:
            single = (os.getenv("WALRUS_AGGREGATOR_URL") or "").strip()
            if single:
                configured = [single.rstrip("/")]
        if not configured:
            return list(DEFAULT_POOL)
        # Keep the configured URL(s) first (priority) but always add the public
        # fallbacks too, so a single-entry env config (accidental or not) never
        # collapses the pool to one node with no redundancy/failover.
        extra = [u for u in DEFAULT_POOL if u.rstrip("/") not in configured]
        return configured + extra

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _next_healthy(self) -> _Upstream:
        """Smart pick: lowest score among non-tripped upstreams."""
        now = time.monotonic()
        candidates = [u for u in self.upstreams if u.tripped_until <= now]
        if not candidates:
            # All tripped — pick the one whose cooldown expires soonest
            return min(self.upstreams, key=lambda u: u.tripped_until)
        return min(candidates, key=lambda u: u.score(now))

    def _mark_success(self, u: _Upstream, latency_secs: float) -> None:
        with u.lock:
            u.consecutive_failures = 0
            u.tripped_until = 0.0
            u.total_requests += 1
            u.last_used_at = time.monotonic()
            u.ema_latency_secs = (
                _EMA_ALPHA * latency_secs + (1.0 - _EMA_ALPHA) * u.ema_latency_secs
            )

    def _mark_failure(self, u: _Upstream, latency_secs: float = 0.0) -> None:
        with u.lock:
            u.total_requests += 1
            u.total_failures += 1
            u.consecutive_failures += 1
            u.last_used_at = time.monotonic()
            penalty = max(latency_secs, _DEFAULT_LATENCY * 4)
            u.ema_latency_secs = (
                _EMA_ALPHA * penalty + (1.0 - _EMA_ALPHA) * u.ema_latency_secs
            )
            if u.consecutive_failures >= _FAILURES_TO_TRIP:
                u.tripped_until = time.monotonic() + _HEALTH_COOLDOWN_SECS

    # ── Startup latency ping ──────────────────────────────────────────────────

    def startup_ping(self, workers: int = 8, timeout: float = _PING_TIMEOUT) -> None:
        """Probe all upstreams in parallel and seed their EMA latency scores.

        Called once at server startup so the smart picker routes to the fastest
        aggregators from the very first real request instead of relying on the
        default _DEFAULT_LATENCY placeholder.
        """

        def _ping(up: _Upstream) -> None:
            url = f"{up.url}{_PING_PATH}"
            t0 = time.monotonic()
            try:
                r = requests.get(url, timeout=timeout, stream=True)
                latency = time.monotonic() - t0
                r.close()
                if r.status_code in (200, 404):
                    self._mark_success(up, latency)
                    print(f"[AggPool] ping OK   {up.url}  ({latency*1000:.0f} ms)")
                else:
                    self._mark_failure(up, latency)
                    print(f"[AggPool] ping {r.status_code}  {up.url}  — marked down")
            except Exception as e:
                latency = time.monotonic() - t0
                self._mark_failure(up, latency)
                print(f"[AggPool] ping ERR  {up.url}  {e}")

        with ThreadPoolExecutor(max_workers=min(workers, len(self.upstreams))) as ex:
            futs = [ex.submit(_ping, up) for up in self.upstreams]
            for f in as_completed(futs):
                pass  # results already applied inside _ping

        # Log final ranking
        now = time.monotonic()
        ranked = sorted(self.upstreams, key=lambda u: u.score(now))
        print("[AggPool] Startup ranking (best→worst):")
        for i, u in enumerate(ranked, 1):
            tripped = u.tripped_until > now
            tag = "  ⛔ tripped" if tripped else ""
            print(f"  {i:2}. {u.url}  score={u.score(now):.3f}{tag}")

    # ── Public request API ────────────────────────────────────────────────────

    def request(
        self,
        method: str,
        path: str,
        *,
        session: Optional[requests.Session] = None,
        max_tries: Optional[int] = None,
        treat_404_as_success: bool = True,
        **kwargs: Any,
    ) -> Tuple[Optional[requests.Response], Optional[str]]:
        """Send a request, walking the pool on retriable failures.

        Returns ``(response, used_url)``. ``response`` is None if all attempts failed.
        404 is treated as a definitive answer (blob doesn't exist), not a failure.
        """
        if not path.startswith("/"):
            path = "/" + path
        s = session or requests.Session()
        tries = max_tries or max(len(self.upstreams), 1)

        last_err: Optional[Exception] = None
        for _ in range(tries):
            up = self._next_healthy()
            url = f"{up.url}{path}"
            with up.lock:
                up.in_flight += 1
            t0 = time.monotonic()
            try:
                resp = s.request(method, url, **kwargs)
                latency = time.monotonic() - t0
                if resp.status_code >= 500 or resp.status_code in (429,):
                    self._mark_failure(up, latency)
                    last_err = RuntimeError(f"{resp.status_code} from {up.url}")
                    continue
                if resp.status_code == 404 and treat_404_as_success:
                    self._mark_success(up, latency)
                    return resp, up.url
                if resp.status_code >= 400:
                    self._mark_success(up, latency)
                    return resp, up.url
                self._mark_success(up, latency)
                return resp, up.url
            except requests.RequestException as e:
                self._mark_failure(up, time.monotonic() - t0)
                last_err = e
                continue
            finally:
                with up.lock:
                    up.in_flight = max(0, up.in_flight - 1)
        return None, None

    def get(self, path: str, **kwargs: Any) -> Tuple[Optional[requests.Response], Optional[str]]:
        return self.request("GET", path, **kwargs)

    def head(self, path: str, **kwargs: Any) -> Tuple[Optional[requests.Response], Optional[str]]:
        return self.request("HEAD", path, **kwargs)

    def get_blob_cached(
        self, path: str, cache_dir: "str | os.PathLike", **kwargs: Any
    ) -> Tuple[Optional[bytes], Optional[str]]:
        """Fetch a full blob's raw bytes with a permanent on-disk cache keyed by `path`.

        Walrus blob content is immutable once written (content-addressed), so caching
        indefinitely is always correct. Use only for plain full-content GETs — not for
        Range/liveness requests, which must hit the network every time.

        Returns (content_bytes, source) where source is "cache" or the aggregator URL used.
        """
        import hashlib
        from pathlib import Path

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / hashlib.sha256(path.encode("utf-8")).hexdigest()
        if cache_file.exists():
            return cache_file.read_bytes(), "cache"

        kwargs.setdefault("stream", True)
        resp, used_url = self.get(path, **kwargs)
        if resp is None or resp.status_code != 200:
            return None, None
        buf = bytearray()
        for chunk in resp.iter_content(1 << 20):
            buf += chunk
        data = bytes(buf)
        tmp = cache_file.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(cache_file)
        return data, used_url

    def health_snapshot(self) -> List[Dict[str, Any]]:
        now = time.monotonic()
        out = []
        for u in sorted(self.upstreams, key=lambda x: x.score(now)):
            with u.lock:
                out.append(
                    {
                        "url": u.url,
                        "consecutive_failures": u.consecutive_failures,
                        "total_requests": u.total_requests,
                        "total_failures": u.total_failures,
                        "ema_latency_ms": round(u.ema_latency_secs * 1000, 1),
                        "in_flight": u.in_flight,
                        "score": round(u.score(now), 3),
                        "tripped": u.tripped_until > now,
                        "cooldown_remaining_secs": round(max(0.0, u.tripped_until - now), 1),
                    }
                )
        return out


# ── Module-level singleton ────────────────────────────────────────────────────

_default_pool: Optional[AggregatorPool] = None
_pool_lock = threading.Lock()


def get_pool() -> AggregatorPool:
    global _default_pool
    if _default_pool is None:
        with _pool_lock:
            if _default_pool is None:
                _default_pool = AggregatorPool()
    return _default_pool


def reset_pool() -> None:
    """Force re-init (re-reads env vars). Useful for tests."""
    global _default_pool
    with _pool_lock:
        _default_pool = None


__all__ = ["AggregatorPool", "get_pool", "reset_pool", "DEFAULT_POOL"]
