"""Main entry point for Omura services.

Roles:
- api: serve FastAPI only (no cataloger/indexer threads)
- worker: run cataloger/indexer only (no HTTP server)
- all: legacy mode, run API + background threads together
"""

import os
from pathlib import Path


def _load_dotenv(dotenv_path: Path) -> None:
    """Load .env file into process env without overriding existing variables."""
    if not dotenv_path.exists():
        return

    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(Path(__file__).resolve().parent / ".env")

# Set defaults before any CUDA/torch usage (env overrides these)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("OMURA_VECTOR_INDEX_VERSION", "omni")

import uvicorn


if __name__ == "__main__":
    role = os.getenv("OMURA_INSTANCE_ROLE", "all").strip().lower()
    host = os.getenv("OMURA_HOST", "0.0.0.0")
    port = int(os.getenv("OMURA_PORT", "19353"))
    workers = int(os.getenv("OMURA_WORKERS", "2"))

    if role == "worker":
        from omura.worker import run_worker_service

        print("Starting Omura worker instance (cataloger/indexer only)")
        run_worker_service()
    else:
        if role == "api":
            # QoS mode: API nodes should not run background ingestion.
            os.environ["OMURA_ENABLE_CATALOGER"] = "false"
            os.environ["OMURA_ENABLE_INDEXER"] = "false"
            print(
                f"Starting Omura API-only instance on {host}:{port} with {workers} workers"
            )
        else:
            enable_indexer = os.getenv("OMURA_ENABLE_INDEXER", "true").lower() == "true"
            print(
                f"Starting Omura combined instance on {host}:{port} with {workers} workers"
            )
            if enable_indexer:
                print("Background indexer: ENABLED (legacy combined mode)")
            else:
                print("Background indexer: DISABLED")

        # When using multiple workers, we can't pass the app instance directly.
        uvicorn.run(
            "omura.api:app",
            host=host,
            port=port,
            workers=workers,
            log_level="info",
        )
