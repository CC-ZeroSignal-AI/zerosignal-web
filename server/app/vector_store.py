from __future__ import annotations

import uuid
from typing import List, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import Record
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

from .config import Settings
from .schemas import EmbeddingRecord


class VectorStore:
    """Wrapper around Qdrant for storing and retrieving embeddings."""

    def __init__(self, settings: Settings) -> None:
        self._client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self._prefix = settings.collection_name_prefix

    def _collection_name(self, pack_id: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in pack_id.strip())
        return f"{self._prefix}{safe.lower()}"

    def ensure_collection(self, pack_id: str, vector_size: int) -> str:
        collection_name = self._collection_name(pack_id)
        try:
            self._client.get_collection(collection_name)
        except UnexpectedResponse as err:
            if err.status_code != 404:
                raise
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
            )
        return collection_name

    def upsert(self, pack_id: str, records: Sequence[EmbeddingRecord], vector_size: int) -> int:
        if not records:
            return 0
        collection_name = self.ensure_collection(pack_id, vector_size)
        points = [
            qm.PointStruct(
                id=record.point_id or str(uuid.uuid4()),
                vector=record.embedding,
                payload={
                    "pack_id": pack_id,
                    "document_id": record.document_id,
                    "text": record.text,
                    "metadata": record.metadata,
                },
            )
            for record in records
        ]
        self._client.upsert(collection_name=collection_name, points=points)
        return len(points)

    def search(self, pack_id: str, query_vector: List[float], top_k: int) -> List[Record]:
        collection_name = self._collection_name(pack_id)
        return self._client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

    def download(self, pack_id: str, limit: int, offset: Optional[str] = None) -> tuple[List[Record], Optional[str]]:
        collection_name = self._collection_name(pack_id)
        return self._client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
