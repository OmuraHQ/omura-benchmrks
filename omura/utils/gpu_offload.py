"""Release locally-loaded GPU models after a period of inactivity.

Omura keeps several models resident for latency: the Omura embedder and CLAP in
the API process, plus the Moondream caption and InternVideo2 video sidecars. When
traffic stops, those weights sit on the GPU indefinitely. Each holder registers an
:class:`IdleOffloader` with an ``unload`` callback; a daemon thread invokes it once
nothing has used the model for ``OMURA_GPU_IDLE_UNLOAD_SECS`` seconds.

The offloader also acts as an inference gate: ``busy()`` scopes are blocked while an
unload is in progress, so a request that arrives mid-unload waits and then reloads
the model instead of racing the teardown.

Rules:
  * Wrap every inference path in ``with offloader.busy():`` (or use :func:`guard`).
  * The callback must only drop references; CUDA/GC cleanup is done by the helper.
  * ``OMURA_GPU_IDLE_UNLOAD_SECS=0`` disables offloading entirely.

Env:
  OMURA_GPU_IDLE_UNLOAD_SECS   idle seconds before unloading (default 300)
  OMURA_GPU_IDLE_CHECK_SECS    reaper poll interval (default 20)
"""
from __future__ import annotations

import functools
import gc
import glob as _glob
import os
import threading
import time
from typing import Callable, Optional


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


DEFAULT_IDLE_SECS = _env_float("OMURA_GPU_IDLE_UNLOAD_SECS", 300.0)
DEFAULT_CHECK_SECS = max(1.0, _env_float("OMURA_GPU_IDLE_CHECK_SECS", 20.0))

_cudart = None
_cudart_checked = False


def _find_cudart():
    """Locate libcudart for ``cudaDeviceReset`` (torch does not expose it)."""
    global _cudart, _cudart_checked
    if _cudart_checked:
        return _cudart
    _cudart_checked = True
    candidates = []
    try:
        import torch

        candidates += _glob.glob(
            os.path.join(os.path.dirname(torch.__file__), "lib", "libcudart.so*")
        )
    except Exception:  # noqa: BLE001
        pass
    try:
        import site

        for sp in site.getsitepackages():
            candidates += _glob.glob(
                os.path.join(sp, "nvidia", "*", "lib", "libcudart.so*")
            )
    except Exception:  # noqa: BLE001
        pass
    candidates += ["libcudart.so", "libcudart.so.13", "libcudart.so.12"]
    for path in candidates:
        try:
            lib = __import__("ctypes").CDLL(path)
            lib.cudaDeviceReset.restype = __import__("ctypes").c_int
            _cudart = lib
            return lib
        except Exception:  # noqa: BLE001
            continue
    return None


def reset_cuda_context() -> bool:
    """Destroy the process's CUDA context(s) so the driver reclaims all memory.

    Weights must be unloaded and no inference may be running (the IdleOffloader gate
    guarantees this). Torch re-creates the context lazily on the next CUDA call, so
    the next request works without restarting the process.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        lib = _find_cudart()
        if lib is None:
            print("[gpu-offload] libcudart not found; deep reset unavailable", flush=True)
            return False
        count = torch.cuda.device_count()
        for idx in range(count):
            torch.cuda.set_device(idx)
            rc = int(lib.cudaDeviceReset())
            if rc != 0:
                print(f"[gpu-offload] cudaDeviceReset({idx}) rc={rc}", flush=True)
                return False
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[gpu-offload] cudaDeviceReset failed: {exc}", flush=True)
        return False


def empty_cuda_cache(deep: bool = False) -> None:
    """Best-effort CUDA allocator cleanup. Never raises.

    ``deep=True`` also destroys the CUDA context (cudaDeviceReset), dropping the
    per-process context overhead to ~0; the next inference re-initializes it.
    """
    gc.collect()
    try:
        import torch  # imported lazily: sidecars/CPU-only hosts may not need it

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if deep:
                reset_cuda_context()
            else:
                try:
                    torch.cuda.ipc_collect()
                except Exception:  # noqa: BLE001 - not available on all builds
                    pass
    except Exception:  # noqa: BLE001
        pass


class _BusyScope:
    def __init__(self, owner: "IdleOffloader") -> None:
        self._owner = owner

    def __enter__(self) -> "_BusyScope":
        self._owner._enter()
        return self

    def __exit__(self, *exc) -> bool:
        self._owner._exit()
        return False


class IdleOffloader:
    """Tracks model activity and unloads it after an idle period."""

    def __init__(
        self,
        name: str,
        unload: Callable[[], None],
        *,
        idle_secs: Optional[float] = None,
        check_secs: Optional[float] = None,
    ) -> None:
        self.name = name
        self._unload = unload
        # Read env at construction time: sidecars load .env after this module is
        # imported, and operators may change the timeout between model reloads.
        self._idle_secs = (
            _env_float("OMURA_GPU_IDLE_UNLOAD_SECS", 300.0)
            if idle_secs is None
            else float(idle_secs)
        )
        self._check_secs = (
            max(1.0, _env_float("OMURA_GPU_IDLE_CHECK_SECS", 20.0))
            if check_secs is None
            else max(1.0, float(check_secs))
        )
        if self._idle_secs > 0:
            # Don't let a coarse poll interval overshoot a short idle timeout.
            self._check_secs = min(self._check_secs, max(1.0, self._idle_secs / 2))
        # Condition guards both the activity counters and the unloading flag, so a
        # new inference cannot start while the teardown callback is running.
        self._cond = threading.Condition()
        self._unload_lock = threading.Lock()
        self._active = 0
        self._unloading = False
        self._last = time.monotonic()
        self._unloads = 0
        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if self._idle_secs > 0:
            self._thread = threading.Thread(
                target=self._loop, name=f"gpu-offload:{name}", daemon=True
            )
            self._thread.start()
            print(
                f"[gpu-offload] {name}: unload after {self._idle_secs:.0f}s idle "
                f"(poll {self._check_secs:.0f}s)",
                flush=True,
            )
        else:
            print(f"[gpu-offload] {name}: disabled (idle timeout = 0)", flush=True)

    # ── public API ─────────────────────────────────────────────────────────
    def touch(self) -> None:
        """Mark the model as used *now* (cheap; call on every inference)."""
        with self._cond:
            self._last = time.monotonic()

    def busy(self) -> _BusyScope:
        """Context manager that blocks while an unload is in progress."""
        return _BusyScope(self)

    def unload_now(self, deep: bool = False) -> bool:
        """Force an unload when idle. Returns True when the callback ran.

        ``deep=True`` additionally destroys the CUDA context (GPU ~0); the next
        inference re-initializes it lazily.
        """
        with self._cond:
            if self._active > 0 or self._unloading:
                return False
            self._unloading = True
        try:
            with self._unload_lock:
                self._run_unload(deep=deep)
            return True
        finally:
            with self._cond:
                self._unloading = False
                self._last = time.monotonic()
                self._unloads += 1
                self._cond.notify_all()

    def stop(self) -> None:
        self._stopped.set()

    def stats(self) -> dict:
        with self._cond:
            return {
                "name": self.name,
                "idle_secs": self._idle_secs,
                "check_secs": self._check_secs,
                "active": self._active,
                "unloading": self._unloading,
                "idle_for": round(time.monotonic() - self._last, 1),
                "unloads": self._unloads,
                "enabled": self._idle_secs > 0,
            }

    # ── internals ──────────────────────────────────────────────────────────
    def _enter(self) -> None:
        with self._cond:
            while self._unloading:
                self._cond.wait(timeout=1.0)
            self._active += 1
            self._last = time.monotonic()

    def _exit(self) -> None:
        with self._cond:
            self._active = max(0, self._active - 1)
            self._last = time.monotonic()
            self._cond.notify_all()

    def _loop(self) -> None:
        while not self._stopped.wait(self._check_secs):
            with self._cond:
                if self._active > 0 or self._unloading:
                    continue
                if (time.monotonic() - self._last) < self._idle_secs:
                    continue
                self._unloading = True
            try:
                if not self._unload_lock.acquire(blocking=False):
                    continue
                try:
                    self._run_unload()
                finally:
                    self._unload_lock.release()
            finally:
                with self._cond:
                    self._unloading = False
                    self._last = time.monotonic()
                    self._unloads += 1
                    self._cond.notify_all()

    def _run_unload(self, deep: bool = False) -> None:
        started = time.monotonic()
        try:
            self._unload()
            print(
                f"[gpu-offload] {self.name}: unloaded after {self._idle_secs:.0f}s idle "
                f"({time.monotonic() - started:.1f}s){' [deep]' if deep else ''}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - never kill the reaper thread
            print(f"[gpu-offload] {self.name}: unload failed: {exc}", flush=True)
        finally:
            empty_cuda_cache(deep=deep)


def guard(offloader: IdleOffloader):
    """Decorator: run the wrapped function inside ``offloader.busy()``."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with offloader.busy():
                return fn(*args, **kwargs)

        return wrapper

    return decorator


__all__ = [
    "IdleOffloader",
    "DEFAULT_IDLE_SECS",
    "DEFAULT_CHECK_SECS",
    "empty_cuda_cache",
    "guard",
]
