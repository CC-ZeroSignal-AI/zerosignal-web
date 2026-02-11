from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class DocumentChunk(BaseModel):
    document_id: str = Field(..., description="Unique identifier inside the context pack")
    text: str = Field(..., description="Raw text that will be embedded")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("text")
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text cannot be empty")
        return value


class IngestRequest(BaseModel):
    documents: List[DocumentChunk]

    @validator("documents")
    def validate_documents(cls, value: List[DocumentChunk]) -> List[DocumentChunk]:
        if not value:
            raise ValueError("provide at least one document")
        return value


class IngestResponse(BaseModel):
    stored: int


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = Field(None, ge=1, le=50)


class SearchResult(BaseModel):
    document_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class DownloadItem(BaseModel):
    document_id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]


class DownloadResponse(BaseModel):
    pack_id: str
    limit: int
    offset: Optional[str]
    next_offset: Optional[str]
    items: List[DownloadItem]


@dataclass
class EmbeddingRecord:
    document_id: str
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]
    point_id: Optional[str] = None
