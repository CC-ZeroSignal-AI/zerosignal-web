from __future__ import annotations

from typing import Any, Dict

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
