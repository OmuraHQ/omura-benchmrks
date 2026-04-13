# Omura Usage Guide

## Overview

Omura is a multimodal search engine for Walrus protocol blobs that uses ImageBind for embeddings and cuVS for GPU-accelerated vector search.

## Architecture

```
omura/
├── parsers/          # File type detection and content parsing
│   └── file_detection.py
├── indexers/         # Blob indexing pipelines
│   └── image_indexer.py
├── routes/          # FastAPI endpoints
│   ├── search.py    # Search API
│   └── blobs.py     # Blob metadata API
├── utils/           # Shared utilities
│   ├── blockberry.py        # Blockberry API client
│   ├── imagebind_embeddings.py  # ImageBind wrapper
│   └── vector_store.py     # cuVS vector store
└── api.py           # FastAPI app setup
```

## Running the Indexer

To index images from Walrus and generate embeddings:

```bash
# Index images and generate embeddings
uv run python -m omura.indexers.image_indexer

# With custom batch size and max batches
uv run python -m omura.indexers.image_indexer --batch-size 50 --max-batches 10
```

The indexer will:
1. Stream active blobs from Blockberry API
2. Filter for images (PNG, JPEG, GIF)
3. Generate ImageBind embeddings for each image
4. Store embeddings in cuVS vector store
5. Save index to `data/vector_index/`

## Running the API Server

The API server automatically runs the indexer in the background:

```bash
# Start the FastAPI server (indexer runs automatically)
uv run python main.py

# Or with uvicorn directly
uv run uvicorn omura.api:app --host 0.0.0.0 --port 19353

# Disable background indexer (API only)
OMURA_ENABLE_INDEXER=false uv run python main.py
```

The indexer runs asynchronously in a background thread, so the API remains responsive while indexing continues.

## API Endpoints

### Search for Images

```bash
curl -X POST "http://localhost:19353/search/" \
  -H "Content-Type: application/json" \
  -d '{"query": "a cat playing with a ball", "top_k": 5}'
```

Response:
```json
{
  "results": [
    {
      "blob_id": "DFBxFK...",
      "mime_type": "image/png",
      "size": 123456,
      "similarity": 0.85,
      "extension": "png",
      "kind": "image"
    }
  ],
  "total": 1
}
```

### Get Blob Metadata

```bash
curl "http://localhost:19353/blobs/DFBxFK..."
```

### Get Search Stats

```bash
curl "http://localhost:19353/search/stats"
```

### Reverse Image Search (Upload Any Image)

Use multipart form upload with field `file` and optional `top_k`.

```bash
curl -X POST "http://localhost:19353/search/reverse-image" \
  -F "file=@tests/fixtures/test_blob.png" \
  -F "top_k=10"
```

Notes:
- Accepts `image/*` uploads (PNG/JPEG/WebP/GIF and other supported image formats).
- Default max upload size is 25 MiB (`OMURA_REVERSE_IMAGE_MAX_BYTES`).

## Environment Variables

- `WALRUS_AGGREGATOR_URL`: Walrus aggregator URL (default: mainnet)
- `OMURA_ENABLE_INDEXER`: Enable/disable background indexer (default: `true`)
- `OMURA_INDEXER_WORKERS`: Number of parallel workers (default: 4)
- `OMURA_INDEXER_BATCH_SIZE`: Batch size for indexing (default: 100)
- `OMURA_INDEXER_MAX_BATCHES`: Maximum batches to process (default: unlimited)
- `OMURA_VECTOR_STORE_PATH`: Custom vector store path
- `OMURA_VECTOR_STORE_DIR`: Vector store directory (default: `data/vector_index`)
- `OMURA_HOST`: API server host (default: `0.0.0.0`)
- `OMURA_PORT`: API server port (default: `19353`)
- `IMAGEBIND_CACHE_DIR`: ImageBind model cache (default: `data/imagebind_cache`)
- `OMURA_IMAGEBIND_GPU_ID`: GPU ID for ImageBind embedding generation (default: `0`)
  - Example: `OMURA_IMAGEBIND_GPU_ID=0` uses GPU 0 for ImageBind
- `OMURA_CUVS_GPU_ID`: GPU ID for cuVS vector operations (index building, search) (default: `1`)
  - Example: `OMURA_CUVS_GPU_ID=1` uses GPU 1 for cuVS
  - **Important**: Should be different from ImageBind GPU to avoid conflicts

## Installation

### Basic Installation (API only, no indexing)

```bash
uv sync
```

### Full Installation (with cuVS for indexing)

```bash
uv sync --extra cuvs
# or
uv pip install 'omura[cuvs]'
```

## Notes

- ImageBind model will be downloaded on first use (~2GB)
- **Separate GPU Configuration**: By default, uses GPU 0 for ImageBind and GPU 1 for cuVS
  - ImageBind (embedding generation) uses a single GPU (configurable via `OMURA_IMAGEBIND_GPU_ID`)
  - cuVS (vector index building/search) uses a separate GPU (configurable via `OMURA_CUVS_GPU_ID`)
  - This separation prevents GPU memory conflicts and improves stability
  - Example: `OMURA_IMAGEBIND_GPU_ID=0 OMURA_CUVS_GPU_ID=1` separates operations across GPUs
- cuVS requires CUDA-capable GPU and is optional
- Without cuVS, the API will start but indexing/search will be disabled
- Vector index is saved periodically during indexing
- Only images are currently indexed (text/audio/video support can be added)

## Regular Expired Blob Cleanup

Expired blob pruning is integrated into the API main process and runs automatically
in the periodic background maintenance task (leader worker only).

Configuration:

```bash
# Save interval (default: 600)
OMURA_SAVE_INTERVAL_SECONDS=600

# Prune interval (default: 3600)
OMURA_PRUNE_INTERVAL_SECONDS=3600
```

You can still run ad-hoc/manual pruning from CLI:

```bash
uv run python scripts/prune_expired_blobs.py
uv run python scripts/prune_expired_blobs.py --epoch 22
uv run python scripts/prune_expired_blobs.py --dry-run
```
