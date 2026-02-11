from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, HttpUrl, validator


class SourceConfig(BaseModel):
    url: HttpUrl
    metadata: Dict[str, Any] = Field(default_factory=dict)
    title: Optional[str] = None


class PackConfig(BaseModel):
    pack_id: str
    sources: List[SourceConfig]
    chunk_size: int = Field(900, description="Approximate characters per chunk before summarization")
    chunk_overlap: int = Field(150, description="Number of overlapping characters between chunks")
    summary_model: str = Field(
        "gpt-4o-mini",
        description="LLM identifier for summarization (OpenAI style)",
    )
    summary_temperature: float = 0.2
    summary_max_words: int = Field(180, description="Target max words for each summarized chunk")
    summarization_enabled: bool = True
    ingest_base_url: str = Field(
        "http://localhost:8000",
        description="Base URL of the embedding server",
    )
    request_timeout: int = Field(30, description="HTTP timeout when calling embeddings server")
    batch_size: int = Field(16, gt=0, le=64)
    default_metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("chunk_overlap")
    def validate_overlap(cls, value: int, values: Dict[str, Any]) -> int:
        chunk_size = values.get("chunk_size", 1)
        if value >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value

    @classmethod
    def from_file(cls, path: str | Path) -> "PackConfig":
        config_path = Path(path)
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return cls(**data)
