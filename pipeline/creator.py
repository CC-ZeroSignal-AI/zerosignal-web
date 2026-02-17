from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from .chunker import TextChunker
from .config import PackConfig
from .schemas import DocumentChunk
from .scraper import SourceDocument, WebScraper
from .summarizer import BaseSummarizer
from .uploader import QdrantUploader

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
        ingestor: QdrantUploader,
        *,
        pack_id: str,
        batch_size: int,
        output_path: Path | None = None,
        dry_run: bool = False,
        clean: bool = False,
    ) -> None:
        self.scraper = scraper
        self.chunker = chunker
        self.summarizer = summarizer
        self.ingestor = ingestor
        self.pack_id = pack_id
        self.batch_size = batch_size
        self.output_path = output_path
        self.dry_run = dry_run
        self.clean = clean

    def run(self, config: PackConfig) -> List[ChunkPayload]:
        if self.clean and not self.dry_run:
            logger.info("--clean: deleting existing collection for pack %s", self.pack_id)
            self.ingestor.delete_collection(self.pack_id)

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
            self._report_metadata(aggregated, config)
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
                [chunk.document for chunk in batch],
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

    def _report_metadata(self, payloads: List[ChunkPayload], config: PackConfig) -> None:
        if not payloads:
            logger.info("No payloads to report for metadata")
            return
        topic_counts: Counter[str] = Counter()
        source_urls: set[str] = set()
        for chunk in payloads:
            topic = chunk.document.metadata.get("topic") or "unspecified"
            topic_counts[str(topic)] += 1
            source_url = chunk.document.metadata.get("source_url")
            if source_url:
                source_urls.add(str(source_url))
        metadata_payload = {
            "total_documents": len(payloads),
            "topics": [
                {"name": name, "document_count": count}
                for name, count in sorted(topic_counts.items())
            ],
            "source_urls": sorted(source_urls),
            "metadata": {
                "default_metadata": config.default_metadata,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "summary_model": config.summary_model if config.summarization_enabled else None,
            },
        }
        self.ingestor.upsert_registry(self.pack_id, metadata_payload)
