from __future__ import annotations

import logging
from typing import Iterable, List

import requests

logger = logging.getLogger(__name__)


class EmbeddingIngestor:
    def __init__(self, base_url: str, timeout: int = 30) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def ingest(self, pack_id: str, documents: Iterable[dict]) -> int:
        docs_list = list(documents)
        if not docs_list:
            return 0
        url = f"{self._base_url}/packs/{pack_id}/documents"
        logger.info("Uploading %s documents to %s", len(docs_list), url)
        response = requests.post(url, json={"documents": docs_list}, timeout=self._timeout)
        response.raise_for_status()
        data = response.json()
        return int(data.get("stored", 0))

    def update_metadata(self, pack_id: str, payload: dict) -> None:
        url = f"{self._base_url}/packs/{pack_id}/metadata"
        logger.info("Updating metadata for %s", pack_id)
        response = requests.put(url, json=payload, timeout=self._timeout)
        response.raise_for_status()
