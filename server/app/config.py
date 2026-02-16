from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    collection_name_prefix: str = "context_pack_"
    default_top_k: int = 5
    pack_registry_collection: str = "pack_registry"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
