from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from server.app.schemas import DocumentChunk

from .chunker import TextChunker
from .config import PackConfig
from .scraper import SourceDocument, WebScraper
from .summarizer import BaseSummarizer
from .uploader import EmbeddingIngestor

logger = logging.getLogger(__name__)


@dataclass
class ChunkPayload:
    document: DocumentChunk
    original_length: int


class PackCreator:
    def __init__(
        self,
        scraper: WebScraper,
        chunker: TextChunker,
        summarizer: BaseSummarizer,
        ingestor: EmbeddingIngestor,
        *,
        pack_id: str,
        batch_size: int,
        output_path: Path | None = None,
        dry_run: bool = False,
    ) -> None:
        self.scraper = scraper
        self.chunker = chunker
        self.summarizer = summarizer
        self.ingestor = ingestor
        self.pack_id = pack_id
        self.batch_size = batch_size
        self.output_path = output_path
        self.dry_run = dry_run

    def run(self, config: PackConfig) -> List[ChunkPayload]:
        aggregated: List[ChunkPayload] = []
        for source_index, source in enumerate(config.sources):
            doc = self.scraper.fetch(str(source.url))
            base_metadata = {
                **config.default_metadata,
                **source.metadata,
                "source_url": doc.url,
                "source_title": source.title or doc.title,
            }
            aggregated.extend(
                self._process_source(
                    doc,
                    base_metadata=base_metadata,
                    max_words=config.summary_max_words,
                    source_index=source_index,
                )
            )

        if self.output_path:
            self._write_to_disk(aggregated)

        if not self.dry_run:
            self._upload(aggregated)
        else:
            logger.info("Dry run enabled; skipping upload")

        return aggregated

    def _process_source(
        self,
        doc: SourceDocument,
        *,
        base_metadata: Dict[str, object],
        max_words: int,
        source_index: int,
    ) -> List[ChunkPayload]:
        chunks = self.chunker.split(doc.text)
        logger.info(
            "Split %s into %s chunks (avg len %.0f chars)",
            doc.url,
            len(chunks),
            sum(len(chunk) for chunk in chunks) / max(len(chunks), 1),
        )
        payloads: List[ChunkPayload] = []
        for chunk_index, chunk_text in enumerate(chunks):
            summary = self.summarizer.summarize(chunk_text, max_words=max_words)
            document = DocumentChunk(
                document_id=f"{self.pack_id}-{source_index:02d}-{chunk_index:04d}",
                text=summary,
                metadata={
                    **base_metadata,
                    "chunk_index": chunk_index,
                    "original_char_count": len(chunk_text),
                },
            )
            payloads.append(ChunkPayload(document=document, original_length=len(chunk_text)))
        return payloads

    def _upload(self, payloads: List[ChunkPayload]) -> None:
        logger.info("Uploading %s chunks to pack %s", len(payloads), self.pack_id)
        for batch in self._batched(payloads, self.batch_size):
            stored = self.ingestor.ingest(
                self.pack_id,
                [chunk.document.model_dump() for chunk in batch],
            )
            logger.info("Server stored %s documents", stored)

    def _write_to_disk(self, payloads: List[ChunkPayload]) -> None:
        output = [
            {
                "document_id": chunk.document.document_id,
                "text": chunk.document.text,
                "metadata": chunk.document.metadata,
                "original_char_count": chunk.original_length,
            }
            for chunk in payloads
        ]
        assert self.output_path is not None
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Wrote %s chunks to %s", len(payloads), self.output_path)

    @staticmethod
    def _batched(iterable: List[ChunkPayload], batch_size: int) -> Iterable[List[ChunkPayload]]:
        for idx in range(0, len(iterable), batch_size):
            yield iterable[idx : idx + batch_size]
