from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from .chunker import TextChunker
from .config import PackConfig
from .creator import PackCreator
from .scraper import WebScraper
from .summarizer import LLMSummarizer, NoOpSummarizer
from .uploader import QdrantUploader

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and upload Cognit-Edge context packs")
    parser.add_argument("--config", required=True, help="Path to the pack YAML config")
    parser.add_argument(
        "--output", help="Write the processed chunks to a JSON file before upload"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process and export chunks without hitting the embedding server",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip the LLM summarization layer (uses raw chunks)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing collection before ingesting (prevents duplicates on re-run)",
    )
    parser.add_argument(
        "--override-pack-id",
        help="Optional: override the pack_id defined in the YAML file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PackConfig.from_file(args.config)
    pack_id = args.override_pack_id or config.pack_id

    scraper = WebScraper()
    chunker = TextChunker(config.chunk_size, config.chunk_overlap)

    summarizer = NoOpSummarizer()
    if config.summarization_enabled and not args.no_summary:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set; falling back to raw chunks")
        else:
            base_url = os.getenv("OPENAI_API_BASE")
            summarizer = LLMSummarizer(api_key=api_key, model=config.summary_model, base_url=base_url)

    qdrant_client = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
    embedding_model = SentenceTransformer(config.embedding_model_name)
    ingestor = QdrantUploader(
        client=qdrant_client,
        model=embedding_model,
        collection_name_prefix=config.collection_name_prefix,
    )

    output_path = Path(args.output).expanduser() if args.output else None

    creator = PackCreator(
        scraper=scraper,
        chunker=chunker,
        summarizer=summarizer,
        ingestor=ingestor,
        pack_id=pack_id,
        batch_size=config.batch_size,
        output_path=output_path,
        dry_run=args.dry_run,
        clean=args.clean,
    )

    creator.run(config)


if __name__ == "__main__":
    main()
