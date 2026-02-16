from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


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


class TopicStat(BaseModel):
    name: str
    document_count: int


class PackMetadata(BaseModel):
    pack_id: str
    total_documents: int
    topics: List[TopicStat]
    source_urls: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    last_ingested_at: datetime
