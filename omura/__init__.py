"""Omura package.

For now this just houses utilities around Walrus blob discovery and type
detection. We start with a simple, monolithic-but-modular indexer that:

- Reads blob IDs (e.g. from `blob_ids_active.txt`)
- Fetches blob contents from Walrus
- Identifies file types using magic bytes
"""

"""
Omura: Walrus blob indexing prototype.

This package currently contains:
- `file_detection`: simple magic-bytes based file type detection
- `walrus_client`: thin wrapper around walrus-python client
- `indexer`: small script that fetches blobs and prints detected types
"""

