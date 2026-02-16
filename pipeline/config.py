from __future__ import annotations

import os
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
    qdrant_url: str = Field(
        "http://localhost:6333",
        description="Qdrant server URL (or Qdrant Cloud endpoint)",
    )
    qdrant_api_key: Optional[str] = Field(
        None,
        description="Qdrant API key (required for Qdrant Cloud)",
    )
    embedding_model_name: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="SentenceTransformer model for local embedding",
    )
    collection_name_prefix: str = Field(
        "context_pack_",
        description="Prefix for Qdrant collection names",
    )
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
        # Fall back to env vars when not specified in YAML
        env_fallbacks = {
            "qdrant_url": "QDRANT_URL",
            "qdrant_api_key": "QDRANT_API_KEY",
            "embedding_model_name": "EMBEDDING_MODEL_NAME",
            "collection_name_prefix": "COLLECTION_NAME_PREFIX",
        }
        for field, env_var in env_fallbacks.items():
            if field not in data and os.getenv(env_var):
                data[field] = os.environ[env_var]
        return cls(**data)
