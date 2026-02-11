from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

from .config import Settings
from .schemas import PackMetadata, TopicStat


class PackRegistry:
    """Stores pack metadata inside a dedicated Qdrant collection."""

    def __init__(self, settings: Settings) -> None:
        self._client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        self._collection = settings.pack_registry_collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            self._client.get_collection(self._collection)
        except UnexpectedResponse as err:
            if err.status_code != 404:
                raise
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qm.VectorParams(size=1, distance=qm.Distance.COSINE),
            )

    def list_packs(self) -> List[PackMetadata]:
        results: List[PackMetadata] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                with_payload=True,
                limit=100,
                offset=offset,
            )
            for point in points:
                pack = self._payload_to_metadata(point.payload)
                if pack:
                    results.append(pack)
            if offset is None:
                break
        return results

    def get_pack(self, pack_id: str) -> Optional[PackMetadata]:
        result = self._client.retrieve(
            collection_name=self._collection,
            ids=[pack_id],
            with_payload=True,
        )
        if not result:
            return None
        return self._payload_to_metadata(result[0].payload)

    def upsert(self, metadata: PackMetadata) -> PackMetadata:
        payload = metadata.model_dump()
        payload["last_ingested_at"] = metadata.last_ingested_at.isoformat()
        topics = [topic.dict() for topic in metadata.topics]
        payload["topics"] = topics
        self._client.upsert(
            collection_name=self._collection,
            points=[
                qm.PointStruct(
                    id=metadata.pack_id,
                    vector=[0.0],
                    payload=payload,
                )
            ],
        )
        return metadata

    @staticmethod
    def _payload_to_metadata(payload: Optional[dict]) -> Optional[PackMetadata]:
        if not payload:
            return None
        topics_data = payload.get("topics", [])
        topics = [TopicStat(**topic) for topic in topics_data]
        last_ingested = payload.get("last_ingested_at")
        last_ingested_at = (
            datetime.fromisoformat(last_ingested)
            if isinstance(last_ingested, str)
            else datetime.utcnow()
        )
        return PackMetadata(
            pack_id=payload["pack_id"],
            total_documents=payload.get("total_documents", 0),
            topics=topics,
            source_urls=payload.get("source_urls", []),
            metadata=payload.get("metadata", {}),
            last_ingested_at=last_ingested_at,
        )
