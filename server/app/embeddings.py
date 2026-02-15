from __future__ import annotations

from threading import Lock
from typing import Iterable, List

from fastembed import TextEmbedding


class EmbeddingService:
    """Creates sentence embeddings using a lazy-loaded fastembed model."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: TextEmbedding | None = None
        self._lock = Lock()

    def _get_model(self) -> TextEmbedding:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        model = self._get_model()
        vectors = list(model.embed(list(texts)))
        return [v.tolist() for v in vectors]

    def vector_size(self) -> int:
        return len(next(iter(self._get_model().embed(["test"]))))
