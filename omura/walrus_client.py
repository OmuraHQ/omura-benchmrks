"""Thin wrapper around walrus-python client for fetching Walrus blobs."""

from __future__ import annotations

import os
from typing import Optional

from walrus import WalrusClient


def _get_env(name: str, default: str) -> str:
    """Helper to read env vars with a default."""
    return os.getenv(name, default)


class SimpleWalrusClient:
    """Synchronous Walrus client wrapper."""

    def __init__(self) -> None:
        publisher_url = _get_env(
            "WALRUS_PUBLISHER_URL",
            "https://publisher.walrus-mainnet.walrus.space",
        )
        aggregator_url = _get_env(
            "WALRUS_AGGREGATOR_URL",
            "https://aggregator.walrus-mainnet.walrus.space",
        )
        self.client = WalrusClient(
            publisher_base_url=publisher_url,
            aggregator_base_url=aggregator_url,
        )

    def fetch_blob(self, blob_id: str) -> Optional[bytes]:
        """Fetch blob content bytes; returns None on errors."""
        try:
            return self.client.get_blob(blob_id)
        except Exception:
            return None


__all__ = ["SimpleWalrusClient"]

