# Retrieval Infrastructure Platform

[![CI](https://github.com/bablamon/Retrieval-Infrastructure-Platform/actions/workflows/ci.yml/badge.svg)](https://github.com/bablamon/Retrieval-Infrastructure-Platform/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Docker Compose](https://img.shields.io/badge/docker-compose-2496ED.svg?logo=docker&logoColor=white)](docker-compose.yml)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)

Open-source alternative to Tavily + Firecrawl, purpose-built for autonomous AI agents, RAG pipelines, coding copilots, and deep research systems.

## Pipeline

```
Query → Query Expansion → SearXNG Search → URL Dedup → Async Crawl (HTTPX + Playwright)
      → Trafilatura Extraction → Markdown Cleaning → Semantic Chunking
      → Local Embeddings (BGE) → Qdrant Vector Store → BGE Reranking → Context Output
```

## Stack

| Component | Library |
|-----------|---------|
| API | FastAPI + uvicorn |
| Search | SearXNG (self-hosted) |
| Crawling | HTTPX + Playwright |
| Extraction | Trafilatura + BeautifulSoup4 |
| Embeddings | sentence-transformers (`BAAI/bge-small-en-v1.5`) |
| Reranking | FlagEmbedding (`BAAI/bge-reranker-base`) |
| Vector DB | Qdrant |
| Cache | Redis |
| Database | PostgreSQL (asyncpg) |
| Task Queue | ARQ |

## Quick Start

```bash
# 1. Copy environment config
cp .env.example .env

# 2. Build and start all services
docker compose up --build -d

# 3. Initialize the database (first time only)
docker compose exec api python scripts/init_db.py

# 4. Wait for health
curl http://localhost:8000/health

# 5. Ingest a URL
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)"]}'

# 6. Retrieve relevant context
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "how does self-attention work", "top_k": 20, "rerank_top_k": 5}'
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health check |
| POST | `/query/generate` | Expand query into multiple search queries |
| POST | `/search` | Search via SearXNG |
| POST | `/crawl` | Async crawl URLs |
| POST | `/extract` | Extract + clean content from URL or HTML |
| POST | `/ingest` | Full pipeline: crawl → extract → embed → store |
| POST | `/retrieve` | Semantic search + BGE reranking |

Full API docs: http://localhost:8000/docs

## Architecture

```
app/
├── api/          FastAPI routers (one per endpoint)
├── core/         Config, logging, exceptions, DI
├── crawler/      HTTP + Playwright crawlers, robots.txt, sitemap
├── extractor/    Trafilatura → BS4 → cleaner → chunker
├── embeddings/   SentenceTransformer with Redis cache
├── reranker/     BGE cross-encoder reranker
├── retrieval/    Full pipeline orchestration
├── search/       SearXNG client, query expansion, caching
├── storage/      Redis, Postgres, Qdrant clients
├── workers/      ARQ background task definitions
├── models/       Pydantic v2 request/response/DB schemas
└── utils/        URL utils, text utils, retry, concurrency
```

### Key Design Decisions

- **Single uvicorn worker** — Playwright is not fork-safe; scale via container replicas
- **ARQ over Celery** — native asyncio, Redis-backed, zero extra broker overhead
- **Two-layer URL dedup** — Redis set (fast O(1)) + Postgres `crawl_state` (persistent)
- **Models loaded at startup** — avoids 2-5s cold start per request
- **CPU-bound ops in executor** — trafilatura, sentence-transformers, FlagReranker wrapped in `run_in_executor`
- **BGE query prefix** — `embed_query()` prepends `"query: "` per BAAI instruction-tuning spec

## Configuration

All config via environment variables. See `.env.example` for full reference.

Key settings:
```bash
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5    # or bge-large-en-v1.5 for better quality
RERANKER_MODEL=BAAI/bge-reranker-base     # ~1.1GB; use bge-reranker-v2-m3 for higher quality
EMBEDDING_DEVICE=cpu                       # or cuda for GPU
RERANKER_USE_FP16=false                    # keep false on CPU; true only on CUDA
APP_MAX_CRAWL_CONCURRENCY=10              # concurrent crawls
APP_MAX_PLAYWRIGHT_PAGES=5               # concurrent Playwright pages
```

### Memory footprint

The image and runtime are tuned for low-cost CPU deployment:

- **CPU-only torch** — the Dockerfile installs the CPU wheel, avoiding ~5GB of
  unused CUDA libraries that the default `torch` package pulls in.
- **Worker skips the reranker** — the ARQ worker only ingests/crawls, so it never
  loads the reranker model (~1GB saved per worker).
- **Lazy Playwright** — Chromium (~300MB) is only launched on the first crawl that
  needs JS rendering, not at startup.
- **Bounded threads + caches** — BLAS/OpenMP thread pools are capped and
  Trafilatura's internal caches are reset periodically to keep memory flat.

## Running Tests

```bash
# Inside container
docker compose exec api pytest tests/ -v

# Or locally (with services running)
pip install -r requirements.txt
pytest tests/ -v
```

## Scaling

For higher throughput:
- Scale API replicas: `docker compose up --scale api=3 -d`
- Scale workers: `docker compose up --scale worker=2 -d`
- Use GPU: set `EMBEDDING_DEVICE=cuda` and `RERANKER_USE_FP16=true`
- Increase Qdrant memory limit in docker-compose.yml for larger collections

## Development

```bash
# Install deps
pip install -r requirements.txt
playwright install chromium

# Run locally (requires Redis, Postgres, Qdrant, SearXNG running)
uvicorn app.main:app --reload --port 8000
```
