from __future__ import annotations

import re
from typing import List


class TextChunker:
    """Splits cleaned text into overlapping chunks suitable for embeddings."""

    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 150) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []
        if len(normalized) <= self.chunk_size:
            return [normalized]

        chunks: List[str] = []
        start = 0
        text_length = len(normalized)

        while start < text_length:
            end = min(text_length, start + self.chunk_size)
            split = self._find_split(normalized, start, end)
            chunk = normalized[start:split].strip()
            if chunk:
                chunks.append(chunk)
            if split >= text_length:
                break
            next_start = max(split - self.chunk_overlap, split)
            if next_start <= start:
                next_start = split
            start = next_start

        return chunks

    @staticmethod
    def _find_split(text: str, start: int, end: int) -> int:
        if end >= len(text):
            return len(text)
        split = text.rfind(" ", start, end)
        if split == -1 or split <= start:
            split = end
        return split
