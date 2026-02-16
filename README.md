# ZeroSignal Embedding Platform

This repository contains:
1. **Read-only FastAPI server** – deployed on Vercel, serves pack metadata and embeddings from Qdrant Cloud for Android/ObjectBox clients to sync offline knowledge bases.
2. **Local pipeline** – scrapes vetted URLs, optionally summarizes them with an LLM, embeds text locally with `sentence-transformers`, and writes vectors directly to Qdrant Cloud.

---

## Architecture at a glance
- **Vector database**: Qdrant Cloud (free tier: 1M vectors / 1GB).
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, ~1.5 KB per chunk). Generated locally by the pipeline — the server has no ML dependencies.
- **Pipeline** (runs locally): Scraper → chunker → optional LLM summarizer → sentence-transformers embedding → direct Qdrant write. No server involved.
- **Server** (deployed on Vercel): Read-only API over Qdrant. Serves `GET /packs`, `GET /packs/{pack_id}`, and `GET /packs/{pack_id}/download`.
- **Pack registry**: Stored in a dedicated Qdrant collection (`pack_registry`). The pipeline updates this automatically after every ingestion.
- **Android flow**: Device calls `GET /packs` to discover available packs, then `GET /packs/{pack_id}/download` with cursor pagination, persisting each `{document_id, text, metadata, embedding}` record in ObjectBox for offline RAG.

---

## Repository layout
```
server/
  app/
    config.py        # Pydantic settings loaded from env vars
    main.py          # FastAPI app (read-only endpoints)
    schemas.py       # Pydantic response models
    vector_store.py  # Qdrant read-only client wrapper
    registry.py      # Pack registry reader
  requirements.txt   # Server deps (no ML libraries)
  .env.example       # Template for Qdrant settings
pipeline/
  cli.py             # Entry point
  config.py          # Pack YAML config + env var fallbacks
  schemas.py         # DocumentChunk model
  creator.py         # Orchestrator: scrape → chunk → summarize → upload
  uploader.py        # QdrantUploader: embeds locally + writes to Qdrant
  chunker.py         # Text chunking with overlap
  scraper.py         # Web scraper (BeautifulSoup)
  summarizer.py      # LLM summarizer (OpenAI-compatible API)
  requirements.txt   # Pipeline deps (sentence-transformers, qdrant-client, etc.)
  examples/
    sample-pack.yaml # Starter pack definition
```

---

## Environment setup

### Server (Vercel)
The server only needs Qdrant credentials. Set these as Vercel environment variables:

| Variable | Purpose |
| --- | --- |
| `QDRANT_URL` | Qdrant Cloud endpoint |
| `QDRANT_API_KEY` | API key from Qdrant dashboard |
| `COLLECTION_NAME_PREFIX` | Collection prefix (default `context_pack_`) |

### Pipeline (local)
1. **Python & virtualenv**
   ```bash
   cd server
   python -m venv .venv
   source .venv/bin/activate
   pip install -r ../pipeline/requirements.txt
   ```
2. **`.env` file** – create `server/.env` from the template:
   ```bash
   cp .env.example .env
   ```
   | Variable | Purpose |
   | --- | --- |
   | `QDRANT_URL` | Qdrant Cloud endpoint |
   | `QDRANT_API_KEY` | API key from Qdrant dashboard |
   | `EMBEDDING_MODEL_NAME` | SentenceTransformers model (default `sentence-transformers/all-MiniLM-L6-v2`) |
   | `COLLECTION_NAME_PREFIX` | Collection prefix (default `context_pack_`) |
   | `OPENAI_API_KEY` | API key for LLM summarization |
   | `OPENAI_API_BASE` | Base URL for OpenAI-compatible API (e.g. `https://llm-api.arc.vt.edu/api/v1`) |

   When `OPENAI_API_KEY` is unset, the pipeline skips summarization and uploads raw chunks.

---

## Building a context pack with the pipeline
1. **Understand the YAML config** – each pack has:
   ```yaml
   pack_id: emergency-field-pack
   default_metadata:
     domain: emergency-response
     language: en
   sources:
     - url: https://en.wikipedia.org/wiki/Hyperbaric_medicine
       metadata:
         topic: hyperbaric
   chunk_size: 5000
   chunk_overlap: 540
   summary_model: gpt-oss-120b
   summary_max_words: 500
   summarization_enabled: true
   # qdrant_url and qdrant_api_key are read from env vars
   embedding_model_name: sentence-transformers/all-MiniLM-L6-v2
   batch_size: 16
   ```
   - `qdrant_url` and `qdrant_api_key` fall back to `QDRANT_URL` / `QDRANT_API_KEY` env vars when not specified in the YAML.
   - Each source inherits `default_metadata` plus its own metadata. The scraper stores `source_url` and `source_title` automatically.

2. **Dry-run first** (optional sanity check):
   ```bash
   python -m pipeline.cli --config pipeline/examples/sample-pack.yaml --dry-run --output /tmp/pack.json
   ```
   Open `/tmp/pack.json` to inspect the summarized text + metadata before uploading.

3. **Ingest for real**:
   ```bash
   set -a && source server/.env && set +a
   python -m pipeline.cli --config pipeline/examples/sample-pack.yaml
   ```
   - The pipeline fetches all sources, runs the summarizer (or not), embeds text locally with sentence-transformers, and writes vectors directly to Qdrant.
   - After uploads succeed, the pipeline also writes pack registry metadata to Qdrant so `GET /packs` reflects topic stats, total embeddings, and source URLs.
   - Re-running the same config adds new vectors (use deterministic `document_id` format: `<pack_id>-<source_idx>-<chunk_idx>`).

4. **Cleaning up / re-ingesting**: to nuke a pack entirely, delete the Qdrant collection named `context_pack_<pack_id>` via the Qdrant UI or client, then rerun the pipeline. The registry entry will be overwritten on the next run.

---

## API reference (server)

| Method & Path | Description |
| --- | --- |
| `GET /health` | Readiness probe |
| `GET /packs` | List all packs with topic counts and metadata |
| `GET /packs/{pack_id}` | Metadata for one pack |
| `GET /packs/{pack_id}/download?limit=…&offset=…` | Download stored chunks + embeddings (paginated) |

### Pack catalog (`GET /packs`)
```bash
curl https://zerosignal-web.vercel.app/packs
```
Response:
```json
[
  {
    "pack_id": "emergency-field-pack",
    "total_documents": 36,
    "topics": [
      {"name": "first-aid", "document_count": 16},
      {"name": "hyperbaric", "document_count": 20}
    ],
    "source_urls": [
      "https://en.wikipedia.org/wiki/First_aid",
      "https://en.wikipedia.org/wiki/Hyperbaric_medicine"
    ],
    "metadata": {
      "default_metadata": {"domain": "emergency-response", "language": "en"},
      "chunk_size": 5000
    },
    "last_ingested_at": "2026-02-10T19:23:27Z"
  }
]
```

### Pack metadata (`GET /packs/{pack_id}`)
Returns a single entry (same shape as above) for the requested pack.

### Download (`GET /packs/{pack_id}/download`)
```bash
curl "https://zerosignal-web.vercel.app/packs/emergency-field-pack/download?limit=5"
```
Response:
```json
{
  "pack_id": "emergency-field-pack",
  "limit": 5,
  "offset": null,
  "next_offset": "261299ac-9961-48b8-90a9-81eee781577b",
  "items": [
    {
      "document_id": "emergency-field-pack-00-0000",
      "text": "Summarized content...",
      "metadata": {
        "domain": "emergency-response",
        "topic": "hyperbaric",
        "source_url": "https://en.wikipedia.org/wiki/Hyperbaric_medicine",
        "chunk_index": 0
      },
      "embedding": [0.0123, -0.084, ...]
    }
  ]
}
```
Keep calling the endpoint with the last `next_offset` until it returns `null` to retrieve the entire pack. Android/ObjectBox should persist the cursor so it can resume interrupted syncs.

### Pack identifiers (`pack_id`)
- Choose stable, URL-safe slugs: `emergency-field-pack`, `naval-maintenance-v1`, etc.
- The same ID must be used by the pipeline YAML and the Android client.
- Internally, collections are named `context_pack_<pack_id>` (sanitized). Deleting a pack = deleting that collection in Qdrant.

---

## Android/offline sync flow
1. Call `GET /packs` to discover what packs/topics exist and decide which ones to download.
2. Run the pipeline locally whenever content changes so the registry stays fresh.
3. Device hits `GET /packs/{pack_id}/download?limit=…` and stores each `DownloadItem` into ObjectBox: `{document_id, text, metadata, embedding}`.
4. Continue paging with the supplied cursor until `next_offset` becomes `null`. Persist the last cursor so the device can resume if the sync is interrupted.

---

## Validation checklist
- **Server health**: `curl https://zerosignal-web.vercel.app/health` returns `{"status": "ok"}`.
- **Pipeline dry-run**: inspect exported JSON to ensure summaries + metadata look right.
- **End-to-end ingestion**: run the pipeline, then verify `GET /packs/<id>/download?limit=5` returns the newly ingested text and embeddings.
- **Android smoke test**: import a batch into ObjectBox, run a local cosine search, and ensure the embeddings produce relevant nearest neighbors.

---

## Next steps / production hardening
- Add authentication/authorization (API keys, JWTs) before exposing the endpoints publicly.
- Configure HTTPS termination and CDN according to your deployment environment.
- Introduce duplicate detection or human QA in the pipeline if you ingest regulated material.
