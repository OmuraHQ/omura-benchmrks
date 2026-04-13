---
license: mit
library_name: transformers
pipeline_tag: feature-extraction
tags:
  - omura-embed
  - embeddings
  - multimodal
---

# Omura - Walrus Protocol Search Engine

A search engine for the Walrus protocol that indexes and retrieves blob IDs from Walrus mainnet, with multimodal vector search using Omura Embed.

## Features

- **Blob Discovery**: Scans Sui blockchain to discover all Walrus Blob objects (historical + real-time)
- **Multimodal Search**: Vector similarity search across text, images, audio, and video using ImageBind
- **GPU Acceleration**: Leverages A100 GPUs for embedding generation and vector search with NVIDIA cuvs
- **Persistent Storage**: PostgreSQL database and persistent vector index
- **Epoch Tracking**: Handles blob expiration and deletion based on Sui epochs
- **File Type Detection**: Magic bytes detection for accurate file type identification
- **Text Processing**: Advanced parsing and chunking for webpages, ebooks, and PDFs
- **Async API**: Fully async FastAPI server for high performance

## Requirements

- Python 3.11
- PostgreSQL database
- NVIDIA GPU (recommended for faster embedding generation)
- CUDA toolkit

## Installation

Using `uv` (recommended):

```bash
uv pip install -e .
```

Or using pip:

```bash
pip install -e .
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key configuration variables:
- `SUI_RPC_URL`: Sui mainnet RPC endpoint
- `WALRUS_PUBLISHER_URL`: Walrus publisher endpoint
- `WALRUS_AGGREGATOR_URL`: Walrus aggregator endpoint
- `DATABASE_URL`: PostgreSQL connection string
- `GPU_DEVICE`: GPU device ID (default: 0)
- `OMURA_EMBEDDING_MODEL`: Hugging Face model id used for embeddings.

### Use your Hugging Face Omura Embed model

If you duplicated/renamed your model repo on Hugging Face (for example `immortaltatsu/omura_emebd`), set:

```bash
export OMURA_EMBEDDING_MODEL="immortaltatsu/omura_emebd"
```

Omura will load this model via `transformers` at runtime.

## Usage

### Start the API server and indexer:

```bash
python -m omura.main
```

Or run with your custom Hugging Face model directly:

```bash
OMURA_EMBEDDING_MODEL="immortaltatsu/omura_emebd" python -m omura.main
```

### CLI commands:

```bash
# Show statistics
omura stats

# List blobs
omura list

# Export blobs
omura export
```

## API Endpoints

- `GET /blobs` - List all blob IDs (with pagination)
- `GET /blobs/{blob_id}` - Get blob metadata
- `GET /blobs/{blob_id}/urls` - Get frontend-usable URLs
- `POST /search` - Vector similarity search
- `GET /stats` - Indexing statistics

## Benchmark your model

Use the benchmark script to validate latency/throughput and a retrieval sanity check:

```bash
PYTHONPATH=. python benchmarks/benchmark_omura_emmbed.py --model "immortaltatsu/omura_emebd"
```

Optional image benchmark:

```bash
PYTHONPATH=. python benchmarks/benchmark_omura_emmbed.py --model "immortaltatsu/omura_emebd" --image-dir data/samples --max-images 100
```

Results are saved to `benchmarks/results/omura_emmbed_benchmark.json`.

## Architecture

1. **Blob Discovery Service**: Scans Sui blockchain for Blob objects
2. **Database Storage**: PostgreSQL for blob metadata
3. **Multimodal Embeddings**: ImageBind for unified embeddings
4. **Vector Search**: NVIDIA cuvs for GPU-accelerated similarity search
5. **REST API**: FastAPI async server

## License

MIT
