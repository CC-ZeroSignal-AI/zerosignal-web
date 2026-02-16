from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse
from sentence_transformers import SentenceTransformer

from .schemas import DocumentChunk

logger = logging.getLogger(__name__)

PACK_REGISTRY_COLLECTION = "pack_registry"


class QdrantUploader:
    """Embeds text locally and writes vectors directly to Qdrant."""

    def __init__(
        self,
        client: QdrantClient,
        model: SentenceTransformer,
        *,
        collection_name_prefix: str = "context_pack_",
    ) -> None:
        self._client = client
        self._model = model
        self._prefix = collection_name_prefix

    # -- Collection naming (must match server/app/vector_store.py) -----------

    def _collection_name(self, pack_id: str) -> str:
        safe = "".join(
            ch if ch.isalnum() or ch in {"_", "-"} else "_"
            for ch in pack_id.strip()
        )
        return f"{self._prefix}{safe.lower()}"

    # -- Collection management -----------------------------------------------

    def _ensure_collection(self, pack_id: str, vector_size: int) -> str:
        collection_name = self._collection_name(pack_id)
        try:
            self._client.get_collection(collection_name)
        except UnexpectedResponse as err:
            if err.status_code != 404:
                raise
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=qm.VectorParams(
                    size=vector_size, distance=qm.Distance.COSINE
                ),
            )
        return collection_name

    def _ensure_registry_collection(self) -> None:
        try:
            self._client.get_collection(PACK_REGISTRY_COLLECTION)
        except UnexpectedResponse as err:
            if err.status_code != 404:
                raise
            self._client.create_collection(
                collection_name=PACK_REGISTRY_COLLECTION,
                vectors_config=qm.VectorParams(
                    size=1, distance=qm.Distance.COSINE
                ),
            )

    # -- Ingest documents ----------------------------------------------------

    def ingest(self, pack_id: str, documents: Sequence[DocumentChunk]) -> int:
        if not documents:
            return 0

        texts = [doc.text for doc in documents]
        embeddings = self._model.encode(texts, show_progress_bar=False)
        vector_size = embeddings.shape[1]

        collection_name = self._ensure_collection(pack_id, vector_size)

        points = [
            qm.PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[idx].tolist(),
                payload={
                    "pack_id": pack_id,
                    "document_id": doc.document_id,
                    "text": doc.text,
                    "metadata": doc.metadata,
                },
            )
            for idx, doc in enumerate(documents)
        ]

        self._client.upsert(collection_name=collection_name, points=points)
        logger.info(
            "Upserted %d points to collection %s", len(points), collection_name
        )
        return len(points)

    # -- Registry metadata ---------------------------------------------------

    def upsert_registry(self, pack_id: str, metadata_payload: Dict[str, Any]) -> None:
        self._ensure_registry_collection()

        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, pack_id))
        payload = {
            "pack_id": pack_id,
            "total_documents": metadata_payload.get("total_documents", 0),
            "topics": metadata_payload.get("topics", []),
            "source_urls": metadata_payload.get("source_urls", []),
            "metadata": metadata_payload.get("metadata", {}),
            "last_ingested_at": datetime.utcnow().isoformat(),
        }

        self._client.upsert(
            collection_name=PACK_REGISTRY_COLLECTION,
            points=[
                qm.PointStruct(
                    id=point_id,
                    vector=[0.0],
                    payload=payload,
                )
            ],
        )
        logger.info("Upserted registry metadata for pack %s", pack_id)
