"""Minimal ``.env`` loader for sidecar processes.

The API process loads ``.env`` in ``main.py``, but the caption/video sidecars are
launched by ``start.sh`` and only inherit the shell environment. This keeps the
loader tiny and stdlib-only so it can be imported in any venv.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union


def load_env_file(path: Union[str, Path], *, override: bool = False) -> int:
    """Load ``KEY=VALUE`` pairs from *path* into ``os.environ``.

    Comments/blank lines are ignored, surrounding quotes are stripped, and values
    are never overridden unless *override* is true (matching ``main.py``).
    Returns the number of variables set.
    """
    p = Path(path)
    if not p.exists():
        return 0
    applied = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            applied += 1
    return applied


__all__ = ["load_env_file"]
