from __future__ import annotations

from threading import Lock
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Creates sentence embeddings using a lazy-loaded transformer model."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._lock = Lock()

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        model = self._get_model()
        vectors = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        if isinstance(vectors, np.ndarray):
            return vectors.astype(np.float32).tolist()
        return [list(map(float, vec)) for vec in vectors]

    def vector_size(self) -> int:
        return self._get_model().get_sentence_embedding_dimension()
