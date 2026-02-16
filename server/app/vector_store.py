from __future__ import annotations

from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import Record

from .config import Settings


class VectorStore:
    """Read-only wrapper around Qdrant for retrieving embeddings."""

    def __init__(self, settings: Settings) -> None:
        self._client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self._prefix = settings.collection_name_prefix

    def _collection_name(self, pack_id: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in pack_id.strip())
        return f"{self._prefix}{safe.lower()}"

    def download(self, pack_id: str, limit: int, offset: Optional[str] = None) -> tuple[List[Record], Optional[str]]:
        collection_name = self._collection_name(pack_id)
        return self._client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
