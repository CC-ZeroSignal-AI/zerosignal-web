from __future__ import annotations

from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from qdrant_client.http.exceptions import UnexpectedResponse

from .config import Settings, get_settings
from .schemas import (
    DownloadResponse,
    DownloadItem,
    PackMetadata,
)
from .registry import PackRegistry
from .vector_store import VectorStore


app = FastAPI(title="ZeroSignal Read-Only API", version="0.2.0")


def _init_services() -> tuple[Settings, VectorStore, PackRegistry]:
    settings = get_settings()
    store = VectorStore(settings)
    registry = PackRegistry(settings)
    return settings, store, registry


SETTINGS, STORE, REGISTRY = _init_services()


def get_vector_store() -> VectorStore:
    return STORE


def get_registry() -> PackRegistry:
    return REGISTRY


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/packs", response_model=List[PackMetadata])
def list_packs(registry: PackRegistry = Depends(get_registry)) -> List[PackMetadata]:
    return registry.list_packs()


@app.get("/packs/{pack_id}", response_model=PackMetadata)
def fetch_pack_metadata(
    pack_id: str, registry: PackRegistry = Depends(get_registry)
) -> PackMetadata:
    pack = registry.get_pack(pack_id)
    if not pack:
        raise HTTPException(status_code=404, detail="Pack metadata not found")
    return pack


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
