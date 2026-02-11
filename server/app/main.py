from __future__ import annotations

from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from qdrant_client.http.exceptions import UnexpectedResponse

from .config import Settings, get_settings
from .embeddings import EmbeddingService
from .schemas import (
    DownloadResponse,
    DownloadItem,
    EmbeddingRecord,
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResult,
)
from .vector_store import VectorStore


app = FastAPI(title="Cognit-Edge Embedding Service", version="0.1.0")


def _init_services() -> tuple[Settings, EmbeddingService, VectorStore]:
    settings = get_settings()
    embedder = EmbeddingService(settings.embedding_model_name)
    store = VectorStore(settings)
    return settings, embedder, store


SETTINGS, EMBEDDER, STORE = _init_services()


def get_embedding_service() -> EmbeddingService:
    return EMBEDDER


def get_vector_store() -> VectorStore:
    return STORE


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/packs/{pack_id}/documents", response_model=IngestResponse)
async def ingest_documents(
    pack_id: str,
    payload: IngestRequest,
    embedder: EmbeddingService = Depends(get_embedding_service),
    store: VectorStore = Depends(get_vector_store),
) -> IngestResponse:
    texts = [doc.text for doc in payload.documents]
    embeddings = embedder.embed(texts)
    records = [
        EmbeddingRecord(
            document_id=doc.document_id,
            text=doc.text,
            metadata=doc.metadata,
            embedding=embeddings[idx],
        )
        for idx, doc in enumerate(payload.documents)
    ]
    stored = store.upsert(pack_id, records, embedder.vector_size())
    return IngestResponse(stored=stored)


@app.post("/packs/{pack_id}/search", response_model=List[SearchResult])
async def search_pack(
    pack_id: str,
    payload: SearchRequest,
    embedder: EmbeddingService = Depends(get_embedding_service),
    store: VectorStore = Depends(get_vector_store),
) -> List[SearchResult]:
    top_k = payload.top_k or SETTINGS.default_top_k
    try:
        query_vector = embedder.embed([payload.query])[0]
        records = store.search(pack_id, query_vector, top_k)
    except UnexpectedResponse as err:
        if err.status_code == 404:
            raise HTTPException(status_code=404, detail="Context pack not found")
        raise

    results: List[SearchResult] = []
    for record in records:
        payload_data = record.payload or {}
        results.append(
            SearchResult(
                document_id=payload_data.get("document_id", ""),
                text=payload_data.get("text", ""),
                metadata=payload_data.get("metadata") or {},
                score=record.score or 0.0,
            )
        )
    return results


@app.get("/packs/{pack_id}/download", response_model=DownloadResponse)
async def download_pack(
    pack_id: str,
    limit: int = Query(50, ge=1, le=500),
    offset: Optional[str] = Query(None, description="Opaque cursor from the previous response"),
    store: VectorStore = Depends(get_vector_store),
) -> DownloadResponse:
    try:
        records, next_offset = store.download(pack_id, limit, offset)
    except UnexpectedResponse as err:
        if err.status_code == 404:
            raise HTTPException(status_code=404, detail="Context pack not found")
        raise

    items: List[DownloadItem] = []
    for record in records:
        payload_data = record.payload or {}
        vector = record.vector
        if isinstance(vector, dict):
            # If multiple vectors are stored, grab the default one
            vector = next(iter(vector.values()))
        items.append(
            DownloadItem(
                document_id=payload_data.get("document_id", ""),
                text=payload_data.get("text", ""),
                metadata=payload_data.get("metadata") or {},
                embedding=list(vector) if vector is not None else [],
            )
        )

    return DownloadResponse(
        pack_id=pack_id,
        limit=limit,
        offset=offset,
        next_offset=next_offset,
        items=items,
    )
