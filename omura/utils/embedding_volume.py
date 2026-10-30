"""Read-only embedding volume measurement for the pilot condition.

The Phase 3 pilot condition asks for "3-4 GB of embeddings stored on the
Protocol, updated daily". This module measures what the local stores actually
contain — vectors, dimensions, on-disk index files, logical vector bytes, and
the catalogued source-blob bytes — without loading models, touching the GPU, or
writing to any store.

Two byte figures are reported side by side:

* ``logical_fp32_bytes`` / ``logical_fp16_bytes`` — ``vectors * dim * 4`` (or
  ``* 2``). This is the embedding payload that would be published to Walrus.
* ``catalog_bytes`` — ``SUM(size)`` of indexed source blobs in
  ``blob_catalog.sqlite``. Source media is *not* the embedding payload; it is
  reported for context only and never counted toward ``meets_3gb``.

Two flag families are reported and they mean different things:

* ``meets_3gb`` / ``meets_4gb`` — **local payload**. True when the local
  stores' logical fp32 bytes reach 3 / 4 GiB. These are the historical,
  backward-compatible flags and do *not* prove publication.
* ``meets_protocol_3gb`` / ``meets_protocol_4gb`` — **grant-facing**. True only
  when a *complete* publication manifest records at least 3 / 4 GiB of shard
  bytes that were actually uploaded to Walrus (see
  ``scripts/publish_index.py``). A local index on this host is not "on the
  Protocol" until it is published; these flags are the ones to cite.

The publication manifest is read from ``OMURA_PUBLISHED_INDEX_MANIFEST``
(default ``data/refresh_audit/published_index.json``). Missing or invalid
manifests never raise: they report zeros and add an entry to ``errors``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import quote

import numpy as np

FP32_BYTES = 4
FP16_BYTES = 2
GIB = 1024 ** 3

DEFAULT_PUBLISHED_MANIFEST = "data/refresh_audit/published_index.json"

#: Store definitions, in reporting order. The env var is resolved on every call
#: so tests and operators can retarget stores without reloading the module.
STORE_SPECS: tuple[Dict[str, str], ...] = (
    {
        "name": "unified",
        "env": "OMURA_VECTOR_STORE_DIR",
        "default": "data/vector_index",
        "backend": "gemma",
    },
    {
        "name": "audio",
        "env": "OMURA_AUDIO_VECTOR_STORE_DIR",
        "default": "data/vector_index_clap",
        "backend": "audio",
    },
    {
        "name": "video",
        "env": "OMURA_VIDEO_VECTOR_STORE_DIR",
        "default": "data/vector_index_iv2",
        "backend": "video",
    },
    {
        "name": "gemma_audio",
        "env": "OMURA_OMURAGEMMA_AUDIO_VECTOR_STORE_DIR",
        "default": "data/vector_index_gemma_audio",
        "backend": "gemma",
    },
    {
        "name": "gemma_video",
        "env": "OMURA_OMURAGEMMA_VIDEO_VECTOR_STORE_DIR",
        "default": "data/vector_index_gemma_video",
        "backend": "gemma",
    },
)

_INDEX_FILES = ("embeddings.npy", "vector_index.faiss", "vector_index.cagra")
_MARKER_FILES = ("embedding_backend.json", "store_identity.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return None


def _file_bytes(path: Path) -> int:
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def _npy_shape(path: Path) -> tuple[int, int]:
    """Return ``(count, dim)`` from an ``.npy`` header without reading data."""
    try:
        array = np.load(str(path), mmap_mode="r", allow_pickle=False)
    except Exception:  # noqa: BLE001 - corrupt/truncated files must not raise
        return 0, 0
    if array.ndim == 1:
        return int(array.shape[0]), 0
    if array.ndim >= 2:
        return int(array.shape[0]), int(array.shape[-1])
    return 0, 0


def _metadata_count(path: Path) -> int:
    data = _read_json(path)
    if isinstance(data, dict):
        return len(data)
    if isinstance(data, list):
        return len(data)
    return 0


def _marker_dim(directory: Path) -> int:
    for name in _MARKER_FILES:
        marker = _read_json(directory / name)
        if isinstance(marker, dict):
            for key in ("dim", "embedding_dim", "dimensions"):
                value = marker.get(key)
                try:
                    if value is not None and int(value) > 0:
                        return int(value)
                except (TypeError, ValueError):
                    continue
    return 0


def store_report(name: str, directory: str | Path, *, env: str = "", backend: str = "") -> Dict[str, Any]:
    """Measure one store directory. Never raises; errors land in ``errors``."""
    path = Path(directory)
    errors: List[str] = []
    report: Dict[str, Any] = {
        "name": name,
        "path": str(path),
        "env": env,
        "backend": backend,
        "exists": path.is_dir(),
        "vector_count": 0,
        "dim": 0,
        "on_disk_bytes": 0,
        "logical_fp32_bytes": 0,
        "logical_fp16_bytes": 0,
        "logical_gb": 0.0,
        "files": {},
        "errors": errors,
    }
    if not path.is_dir():
        errors.append("store directory does not exist")
        return report

    for filename in _INDEX_FILES:
        file_path = path / filename
        size = _file_bytes(file_path)
        if size:
            report["files"][filename] = size
            report["on_disk_bytes"] += size

    count, dim = _npy_shape(path / "embeddings.npy")
    if not count:
        metadata_count = _metadata_count(path / "metadata.json")
        if metadata_count:
            count = metadata_count
            errors.append("vector count taken from metadata.json (embeddings.npy missing/unreadable)")
    if not dim:
        dim = _marker_dim(path)
        if dim and count:
            errors.append("dimension taken from store marker (embeddings.npy missing/unreadable)")
    if count and not dim:
        errors.append("vector count known but dimension unknown")

    report["vector_count"] = int(count)
    report["dim"] = int(dim)
    fp32 = int(count) * int(dim) * FP32_BYTES
    report["logical_fp32_bytes"] = fp32
    report["logical_fp16_bytes"] = int(count) * int(dim) * FP16_BYTES
    report["logical_gb"] = fp32 / GIB
    return report


def _catalog_path(explicit: Optional[str | Path] = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    configured = os.getenv("OMURA_CATALOG_DB_PATH")
    if configured:
        return Path(configured)
    data_dir = Path(os.getenv("OMURA_DATA_DIR", "data"))
    return data_dir / "blob_catalog.sqlite"


def catalog_report(db_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Read indexed blob bytes from the catalog using a read-only SQLite URI."""
    path = _catalog_path(db_path)
    report: Dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": 0,
        "count": 0,
        "by_modality": {},
        "errors": [],
    }
    if not report["exists"]:
        report["errors"].append("catalog database does not exist")
        return report

    uri = f"file:{quote(str(path.resolve()), safe='/')}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True, timeout=2)
    except (sqlite3.Error, OSError) as exc:  # pragma: no cover - host/filesystem dependent
        report["errors"].append(f"could not open catalog read-only: {exc}")
        return report

    try:
        conn.row_factory = sqlite3.Row
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(blobs)").fetchall()
        }
        if not columns:
            report["errors"].append("blobs table not found")
            return report

        size_col = "size" if "size" in columns else (
            "actual_size" if "actual_size" in columns else None
        )
        if size_col is None:
            report["errors"].append("blobs table has no size column")
            return report

        row = conn.execute(
            f"SELECT COALESCE(SUM({size_col}), 0) AS total, COUNT(*) AS n "
            "FROM blobs WHERE indexed = 1"
        ).fetchone()
        report["bytes"] = int(row["total"] or 0)
        report["count"] = int(row["n"] or 0)

        if "kind" in columns:
            by_kind: Dict[str, Dict[str, int]] = {}
            for row in conn.execute(
                f"SELECT COALESCE(kind, 'other') AS kind, "
                f"COALESCE(SUM({size_col}), 0) AS total, COUNT(*) AS n "
                "FROM blobs WHERE indexed = 1 GROUP BY COALESCE(kind, 'other')"
            ).fetchall():
                by_kind[str(row["kind"])] = {
                    "count": int(row["n"] or 0),
                    "bytes": int(row["total"] or 0),
                }
            report["by_modality"] = dict(sorted(by_kind.items()))
    except sqlite3.Error as exc:
        report["errors"].append(f"catalog query failed: {exc}")
    finally:
        conn.close()
    return report


def _resolve_stores(
    store_paths: Optional[Mapping[str, str | Path]] = None,
) -> List[Dict[str, str]]:
    if store_paths is not None:
        return [
            {"name": str(name), "path": str(path), "env": "", "backend": ""}
            for name, path in store_paths.items()
        ]

    specs: List[Dict[str, str]] = []
    by_path: Dict[str, Dict[str, str]] = {}
    for spec in STORE_SPECS:
        env_value = os.getenv(spec["env"], spec["default"])
        path = Path(env_value)
        try:
            key = str(path.resolve()) if path.exists() else str(path)
        except OSError:
            key = str(path)
        if key in by_path:
            alias = by_path[key]
            existing = alias.get("aliases", "")
            alias["aliases"] = f"{existing},{spec['name']}" if existing else spec["name"]
            continue
        entry = {
            "name": spec["name"],
            "path": str(path),
            "env": spec["env"],
            "backend": spec["backend"],
        }
        by_path[key] = entry
        specs.append(entry)
    return specs


def _published_manifest_path(explicit: Optional[str | Path] = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    configured = os.getenv("OMURA_PUBLISHED_INDEX_MANIFEST")
    if configured:
        return Path(configured)
    data_dir = os.getenv("OMURA_DATA_DIR")
    if data_dir:
        return Path(data_dir) / "refresh_audit" / "published_index.json"
    return Path(DEFAULT_PUBLISHED_MANIFEST)


def publication_report(manifest_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Read the Walrus publication manifest written by ``scripts/publish_index.py``.

    Never raises. A missing or invalid manifest yields zeros plus an entry in
    ``errors``. ``published_bytes`` is the manifest's ``total_bytes`` (the size
    of the uploaded ``.npy`` shards); ``complete`` is only true when the
    manifest says so *and* every listed shard has a ``blob_id``.
    """
    path = _published_manifest_path(manifest_path)
    report: Dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "manifest": None,
        "published_bytes": 0,
        "published_gb": 0.0,
        "published_vectors": 0,
        "published_at": None,
        "complete": False,
        "shard_count": 0,
        "published_shards": 0,
        "epochs": 0,
        "errors": [],
    }
    if not report["exists"]:
        report["errors"].append("publication manifest does not exist")
        return report

    data = _read_json(path)
    if not isinstance(data, dict):
        report["errors"].append("publication manifest is not a JSON object")
        return report

    raw_shards = data.get("shards")
    shards = raw_shards if isinstance(raw_shards, list) else []
    shard_sizes: List[int] = []
    published_shards = 0
    for shard in shards:
        if not isinstance(shard, dict):
            continue
        if shard.get("blob_id"):
            published_shards += 1
        try:
            shard_sizes.append(int(shard.get("size") or 0))
        except (TypeError, ValueError):
            shard_sizes.append(0)

    try:
        total_bytes = int(data.get("total_bytes") or 0)
    except (TypeError, ValueError):
        total_bytes = 0
    if total_bytes <= 0:
        total_bytes = sum(shard_sizes)
    sizes_match = not shards or total_bytes == sum(shard_sizes)
    if not sizes_match:
        report["errors"].append(
            "manifest total_bytes does not match the sum of shard sizes"
        )

    try:
        published_vectors = int(data.get("vector_count") or 0)
    except (TypeError, ValueError):
        published_vectors = 0
    try:
        epochs = int(data.get("epochs") or 0)
    except (TypeError, ValueError):
        epochs = 0

    complete = bool(data.get("complete"))
    if complete and not sizes_match:
        complete = False
    if complete and shards and published_shards != len(shards):
        report["errors"].append(
            "manifest is marked complete but some shards have no blob_id"
        )
        complete = False
    if complete and not shards:
        report["errors"].append("manifest is marked complete but lists no shards")
        complete = False

    report.update(
        {
            "manifest": data,
            "published_bytes": total_bytes,
            "published_gb": total_bytes / GIB,
            "published_vectors": published_vectors,
            "published_at": data.get("created_at"),
            "complete": complete,
            "shard_count": len(shards),
            "published_shards": published_shards,
            "epochs": epochs,
        }
    )
    return report


def volume_report(
    *,
    store_paths: Optional[Mapping[str, str | Path]] = None,
    catalog_db_path: Optional[str | Path] = None,
    publication_manifest_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Compute the full read-only volume report.

    ``store_paths`` overrides the default env-driven store list (used by tests).
    Missing stores are reported with ``exists: false`` instead of raising.

    ``publication_manifest_path`` overrides ``OMURA_PUBLISHED_INDEX_MANIFEST``
    (default ``data/refresh_audit/published_index.json``). ``meets_3gb`` /
    ``meets_4gb`` describe the **local payload**; ``meets_protocol_3gb`` /
    ``meets_protocol_4gb`` describe a **complete Walrus publication**.

    ``total_dim`` is the sum of the per-store dimensions (e.g. 768 + 512 + 768
    = 2048); logical bytes are always summed per store, never
    ``total_vectors * total_dim``.
    """
    started = time.perf_counter()
    errors: Dict[str, List[str]] = {}
    stores: List[Dict[str, Any]] = []

    for spec in _resolve_stores(store_paths):
        try:
            report = store_report(
                spec["name"], spec["path"], env=spec["env"], backend=spec["backend"]
            )
        except Exception as exc:  # noqa: BLE001 - partial report beats a 500
            report = {
                "name": spec["name"],
                "path": spec["path"],
                "env": spec.get("env", ""),
                "backend": spec.get("backend", ""),
                "exists": False,
                "vector_count": 0,
                "dim": 0,
                "on_disk_bytes": 0,
                "logical_fp32_bytes": 0,
                "logical_fp16_bytes": 0,
                "logical_gb": 0.0,
                "files": {},
                "errors": [f"unexpected measurement error: {exc}"],
            }
        if spec.get("aliases"):
            report["aliases"] = spec["aliases"].split(",")
        if report["errors"]:
            errors[report["name"]] = list(report["errors"])
        stores.append(report)

    total_vectors = sum(store["vector_count"] for store in stores)
    total_dim = sum(store["dim"] for store in stores)
    on_disk_bytes = sum(store["on_disk_bytes"] for store in stores)
    logical_fp32_bytes = sum(store["logical_fp32_bytes"] for store in stores)
    logical_fp16_bytes = sum(store["logical_fp16_bytes"] for store in stores)

    try:
        catalog = catalog_report(catalog_db_path)
    except Exception as exc:  # noqa: BLE001 - never fail the whole report
        catalog = {
            "path": str(_catalog_path(catalog_db_path)),
            "exists": False,
            "bytes": 0,
            "count": 0,
            "by_modality": {},
            "errors": [f"unexpected catalog error: {exc}"],
        }
    if catalog["errors"]:
        errors["catalog"] = list(catalog["errors"])

    try:
        publication = publication_report(publication_manifest_path)
    except Exception as exc:  # noqa: BLE001 - never fail the whole report
        publication = {
            "path": str(_published_manifest_path(publication_manifest_path)),
            "exists": False,
            "manifest": None,
            "published_bytes": 0,
            "published_gb": 0.0,
            "published_vectors": 0,
            "published_at": None,
            "complete": False,
            "shard_count": 0,
            "published_shards": 0,
            "epochs": 0,
            "errors": [f"unexpected publication error: {exc}"],
        }
    if publication["errors"]:
        errors["publication"] = list(publication["errors"])

    logical_gb = logical_fp32_bytes / GIB
    catalog_gb = catalog["bytes"] / GIB
    published_gb = publication["published_gb"]
    protocol_complete = bool(publication["complete"])
    return {
        "stores": stores,
        "total_vectors": int(total_vectors),
        "total_dim": int(total_dim),
        "on_disk_bytes": int(on_disk_bytes),
        "on_disk_gb": on_disk_bytes / GIB,
        "logical_fp32_bytes": int(logical_fp32_bytes),
        "logical_fp16_bytes": int(logical_fp16_bytes),
        "logical_gb": logical_gb,
        "logical_gb_fp16": logical_fp16_bytes / GIB,
        "catalog_bytes": int(catalog["bytes"]),
        "catalog_gb": catalog_gb,
        "catalog": catalog,
        "published_bytes": int(publication["published_bytes"]),
        "published_gb": published_gb,
        "published_vectors": int(publication["published_vectors"]),
        "published_at": publication["published_at"],
        "publication_manifest": publication["manifest"],
        "meets_3gb": logical_gb >= 3.0,
        "meets_4gb": logical_gb >= 4.0,
        "meets_protocol_3gb": protocol_complete and published_gb >= 3.0,
        "meets_protocol_4gb": protocol_complete and published_gb >= 4.0,
        "computed_at": _now_iso(),
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        "errors": errors,
    }


__all__ = [
    "STORE_SPECS",
    "store_report",
    "catalog_report",
    "publication_report",
    "volume_report",
]
