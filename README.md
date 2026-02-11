# Cognit-Edge Embedding Platform

This repository contains:
1. **FastAPI embedding server** – turns uploaded context chunks into embeddings, stores them in Qdrant (cloud or self-hosted), and exposes APIs for ingestion, search, and offline download.
2. **Automation pipeline** – scrapes vetted URLs, optionally summarizes them with an LLM, and uploads the resulting pack to the server so Android/ObjectBox clients can sync a compact, offline-ready knowledge base.

Everything here is geared toward an MVP that can be run locally but deploys cleanly once you point it at Qdrant Cloud.

---

## Architecture at a glance
- **Vector database**: Qdrant Cloud (free tier: 1M vectors / 1GB). Swap `server/app/vector_store.py` if you move to Pinecone/Weaviate/etc.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim, ~1.5 KB per chunk). Change `EMBEDDING_MODEL_NAME` in `.env` if you need another model.
- **Pipeline**: Scraper → chunker → optional OpenAI summarizer → uploader. Chunk size determines how many vectors land on-device. We currently target ~5 k characters per chunk (≈36 vectors for the sample pack), but you can tune per pack.
- **Android flow**: Device calls `GET /packs/{pack_id}/download`, following the cursor until `next_offset` is `null`, and persists each `{document_id, text, metadata, embedding}` record inside ObjectBox for offline RAG.

---

## Repository layout
```
server/
  app/
    config.py        # Pydantic settings loaded from server/.env
    embeddings.py    # Lazy SentenceTransformer wrapper
    main.py          # FastAPI app + routes
    schemas.py       # Pydantic request/response models
    vector_store.py  # Qdrant client wrapper
  requirements.txt  # Shared deps for server + pipeline
  .env.example      # Template for Qdrant + model settings
pipeline/
  chunker.py, scraper.py, summarizer.py, uploader.py, creator.py, cli.py
  examples/sample-pack.yaml  # Starter pack definition
initial-idea         # Original product brief (reference only)
```

---

## Environment setup
1. **Python & virtualenv**
   ```bash
   cd server
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Server `.env`** – copy the template and fill in secrets (never commit the real file):
   ```bash
   cp .env.example .env
   ```
   | Variable | Purpose |
   | --- | --- |
   | `QDRANT_URL` | Qdrant Cloud HTTPS endpoint (or `http://localhost:6333` for self-hosting) |
   | `QDRANT_API_KEY` | API key from Qdrant dashboard |
   | `EMBEDDING_MODEL_NAME` | SentenceTransformers identifier |
   | `COLLECTION_NAME_PREFIX` | Helps isolate dev/test packs (default `context_pack_`) |
   | `DEFAULT_TOP_K` | Fallback for `/search` if `top_k` not provided |
   | `OPENAI_API_KEY` (optional) | Passed through for the pipeline CLI so you don’t have to export it elsewhere |
3. **Pipeline env** – the CLI reads `OPENAI_API_KEY` from the shell. You can keep it in `server/.env` and `source .env` before running `python -m pipeline.cli`, or load it via your shell profile/direnv. When unset, the pipeline skips the LLM summarizer and uploads raw chunks.

---

## Running the server locally
From the repo root:
```bash
cd server
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir .
```
- Leave this terminal running while the pipeline ingests packs.
- In production, drop `--reload`, add a process manager (systemd, supervisord, Docker, etc.), and make sure `.env` is present or env vars are injected another way.

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
   chunk_size: 5000        # characters per chunk after cleanup
   chunk_overlap: 400      # overlap keeps continuity between chunks
   summary_model: gpt-4o-mini
   summary_max_words: 250  # target length of the LLM-compressed chunk
   summarization_enabled: true
   ingest_base_url: http://localhost:8000
   batch_size: 16
   ```
   - Each source inherits `default_metadata` plus its own metadata. The scraper stores `source_url` and `source_title` automatically so provenance survives all the way to Android.
   - Chunk count ≈ `ceil(total_chars / (chunk_size - chunk_overlap))`. Each chunk becomes one embedding, so tune the numbers to hit your storage budget.
2. **Dry-run first** (optional sanity check):
   ```bash
   python -m pipeline.cli --config pipeline/examples/sample-pack.yaml --dry-run --output /tmp/pack.json
   ```
   Open `/tmp/pack.json` to inspect the summarized text + metadata before uploading.
3. **Ingest for real**:
   ```bash
   python -m pipeline.cli --config pipeline/examples/sample-pack.yaml
   ```
   - The CLI fetches all sources, runs the summarizer (or not), then pushes batches to `POST /packs/{pack_id}/documents`.
   - Re-running the same config simply overwrites matching `document_id`s (format: `<pack_id>-<source_idx>-<chunk_idx>`).
4. **Cleaning up / re-ingesting**: if you want to nuke a pack entirely (e.g., before changing chunk sizes), delete the Qdrant collection named `context_pack_<pack_id>` via the Qdrant UI or client, then rerun the pipeline.

---

## API reference
| Method & Path | Description |
| --- | --- |
| `GET /health` | Simple readiness probe |
| `POST /packs/{pack_id}/documents` | Accepts `{document_id, text, metadata}` chunks, embeds them, and upserts into Qdrant |
| `POST /packs/{pack_id}/search` | Runs semantic similarity search; useful for server-side RAG or QA |
| `GET /packs/{pack_id}/download?limit=…&offset=…` | Streams stored chunks + embeddings for offline/ObjectBox sync |

### Ingestion (`POST /packs/{pack_id}/documents`)
```bash
curl -X POST http://localhost:8000/packs/emergency-field-pack/documents \
     -H "Content-Type: application/json" \
     -d '{
           "documents": [
             {
               "document_id": "emergency-field-pack-00-0001",
               "text": "Condensed instructions...",
               "metadata": {"domain": "emergency-response", "topic": "hyperbaric"}
             }
           ]
         }'
```
Response:
```json
{"stored": 1}
```

### Search (`POST /packs/{pack_id}/search`)
```bash
curl -X POST http://localhost:8000/packs/emergency-field-pack/search \
     -H "Content-Type: application/json" \
     -d '{"query": "hyperbaric chamber safety", "top_k": 5}'
```
Returns a list of `{document_id, text, metadata, score}` sorted by cosine similarity.

### Download (`GET /packs/{pack_id}/download`)
```bash
curl "http://localhost:8000/packs/emergency-field-pack/download?limit=5"
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
Keep calling the endpoint with the last `next_offset` until it returns `null` to retrieve the entire pack. Android/ObjectBox should store the cursor so it can resume interrupted syncs.

### Pack identifiers (`pack_id`)
- Choose stable, URL-safe slugs: `emergency-field-pack`, `naval-maintenance-v1`, etc.
- The same ID must be used by the pipeline YAML, the ingestion API, and the Android client.
- Internally, collections are named `context_pack_<pack_id>` (sanitized). Deleting a pack = deleting that collection in Qdrant.

---

## Android/offline sync flow
1. Trigger the pipeline (or other ingestion tool) to push updated packs whenever content changes.
2. Device hits `GET /packs/{pack_id}/download?limit=…` on a schedule (or after receiving a push) and stores each `DownloadItem` into ObjectBox: `{document_id, text, metadata, embedding}`.
3. Continue paging with the supplied cursor until `next_offset` becomes `null`. Persist the last cursor so the device can resume if the sync is interrupted.
4. Optionally invoke `POST /packs/{pack_id}/search` when the device needs cloud-side re-ranking (e.g., to fetch only the top-N authoritative chunks before syncing).

---

## Validation checklist
- **Server health**: `curl http://localhost:8000/health` → `{ "status": "ok" }`.
- **Pipeline dry-run**: inspect exported JSON to ensure summaries + metadata look right.
- **End-to-end ingestion**: run the pipeline (without `--dry-run`), then verify `GET /packs/<id>/download?limit=5` returns the newly ingested text and embeddings.
- **Android smoke test**: import a batch into ObjectBox, run a local cosine search, and ensure the embeddings produce relevant nearest neighbors.

---

## Next steps / production hardening
- Add authentication/authorization (API keys, JWTs, mTLS) before exposing the endpoints publicly.
- Configure HTTPS termination and WAF/CDN according to your deployment environment.
- Introduce duplicate detection or human QA in the pipeline if you ingest regulated material.
- Swap out Qdrant Cloud for another vector DB by replacing `VectorStore` with a compatible client.
