# Demo: SAP knowledge retrieval, no hallucinations

A 90-second, self-contained demo of the retrieval engine running over a
curated SAP knowledge corpus. Designed for teams who want **high-precision
retrieval over consultant-authored knowledge** — not open-web crawling.

## What it proves

The demo runs four scenes back to back. Each one targets a property that
matters when retrieval is sitting underneath an enterprise AI product:

| Scene | What it shows | Why it matters for SAP / consulting workflows |
|------|---------------|-----------------------------------------------|
| **Ingest** | Curated URLs in → chunks indexed → idempotent on re-run | Re-publishing a consultant's doc replaces its chunks cleanly. No stale knowledge accumulates. |
| **Semantic retrieval** | Natural-language question → cited answer | The everyday case. Citations on every passage so every claim is auditable. |
| **Exact-token retrieval** | `internal tables` query → BM25 + dense fusion → exact compound surfaces | Pure dense retrieval blurs ABAP identifiers, transaction codes, error numbers. The hybrid lexical channel pulls them back. |
| **Refusal** | Off-corpus question → engine returns `refused: true` | The corpus contains no answer, so the engine says so explicitly. An LLM consuming this response has *nothing to fabricate from*. |

## Run it

```bash
docker compose up --build -d
docker compose exec api python scripts/init_db.py    # first run only
docker compose exec api bash demo.sh
```

Or from the host:

```bash
bash demo.sh
```

Override the host or collection if needed:

```bash
BASE_URL=http://my-host:8000 COLLECTION=my_corpus bash demo.sh
```

## What you'll see

```
════════════════════════════════════════════════════════════════════════
  3/5  SEMANTIC RETRIEVAL  —  natural-language question, cited answer
════════════════════════════════════════════════════════════════════════
  Q: What is ABAP and how is it different from a general-purpose language like Java?

  -> returned 3 cited passages in 1.4s
  top rerank score: 0.812

  [1] (source: https://en.wikipedia.org/wiki/ABAP)
  ABAP is a high-level programming language created by SAP SE. It is one
  of the few languages provided for application servers in the SAP NetWeaver
  platform and SAP S/4HANA. ...

  SOURCES (every passage is auditable):
    [1] relevance 0.812  https://en.wikipedia.org/wiki/ABAP
    [2] relevance 0.604  https://en.wikipedia.org/wiki/ABAP
    [3] relevance 0.531  https://en.wikipedia.org/wiki/SAP_S/4HANA
```

And — critically — when asked an off-corpus question:

```
════════════════════════════════════════════════════════════════════════
  5/5  REFUSAL  —  off-corpus question, no hallucination
════════════════════════════════════════════════════════════════════════
  Q: What is the warranty period for the Tesla Model 3?

  REFUSED  —  the engine returned an explicit non-answer
  below_threshold: top rerank score 0.087 is under the min_relevance gate of 0.300

  response:
    refused          = true
    answer_context   = null         <- the LLM has nothing to fabricate from
    chunks           = []
    citations        = []
    top_relevance    = 0.087
    total_retrieved  = 8            <- candidates considered, all rejected
```

## How to adapt it for your knowledge base

The demo uses public Wikipedia pages as stand-ins. Swap them for your own:

```bash
SOURCES='["https://docs.your-co.com/sap/playbook","file:///shared/abap-style.md"]'
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d "{\"urls\": $SOURCES, \"collection\": \"qorelo_kb\", \"replace\": true}"
```

`replace: true` is the idempotent default — re-ingesting the same URL
overwrites its prior chunks rather than accumulating duplicates, so
publishing a new version of a consultant's doc cleanly supersedes the old
one.

## The relevant pipeline knobs

Every `/retrieve` call accepts these per-request:

```json
{
  "query": "...",
  "collection": "qorelo_kb",
  "num_queries": 3,        // multi-query expansion (HyDE + declarative)
  "use_hybrid": true,      // BM25 + dense fusion
  "use_hyde": true,        // hypothetical-answer expansion
  "min_relevance": 0.30    // refusal threshold — raise for stricter mode
}
```

- `min_relevance: 0` disables the refusal gate (best-effort / benchmarking).
- `min_relevance: 0.5` is "only confident answers".
- `min_relevance: 0.3` (the default) is the sweet spot for "looks like an
  answer, not just topically adjacent".

## Footprint

- Fully self-hosted. Postgres + Redis + Qdrant + the API in Docker Compose.
- No paid APIs. No data leaves your infrastructure.
- CPU-only inference by default (BGE small embedding + BGE reranker).
  Set `EMBEDDING_DEVICE=cuda` and `RERANKER_USE_FP16=true` for GPU.
- ~4GB RAM per replica at idle.
