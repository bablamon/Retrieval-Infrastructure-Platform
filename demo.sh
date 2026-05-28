#!/usr/bin/env bash
#
# Demo: retrieval engine over a curated SAP knowledge corpus.
#
#   docker compose exec api bash demo.sh        # from inside the container
#   bash demo.sh                                # from the host (needs curl + python3)
#   BASE_URL=http://host:8000 bash demo.sh      # different host
#
# What this demo shows (in 90 seconds, four scenes):
#
#   1. INGEST     — curated, trusted SAP reference pages only
#                   (NO open-web crawling at retrieve time)
#   2. SEMANTIC   — natural-language question, cited answer
#   3. EXACT-TOKEN — query containing a specific ABAP identifier;
#                   hybrid BM25 + dense recovers the exact match that
#                   pure semantic retrieval would dilute
#   4. REFUSAL    — off-corpus question; engine REFUSES instead of
#                   inventing an answer (the no-hallucination guardrail)
#
# This is the same pipeline that would sit underneath a consultant-
# authored knowledge base — swap the URLs for your own docs and the
# behavior is identical.

set -uo pipefail
BASE_URL="${BASE_URL:-http://localhost:8000}"
COLLECTION="${COLLECTION:-sap_knowledge}"

# ---------------------------------------------------------------------------
# Cosmetics (terminal colors + section banners)
# ---------------------------------------------------------------------------
if [ -t 1 ]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'
  CYAN=$'\033[36m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'
  RESET=$'\033[0m'
else
  BOLD=""; DIM=""; CYAN=""; GREEN=""; YELLOW=""; RED=""; RESET=""
fi
LINE="════════════════════════════════════════════════════════════════════════"

section () {
  echo
  echo "${CYAN}${LINE}${RESET}"
  echo "${CYAN}${BOLD}  $1${RESET}"
  echo "${CYAN}${LINE}${RESET}"
}

# Pause for Enter between scenes when recording.  Opt-in via env var so
# unattended runs (CI, tests, the unattended demo path) are unaffected:
#
#     bash demo.sh             # auto-run (default — no pauses)
#     PAUSE=1 bash demo.sh     # step mode — wait for Enter between scenes
#
pause_for_step () {
  if [ "${PAUSE:-0}" != "0" ]; then
    echo
    echo -en "  ${DIM}▸ press Enter to continue${RESET}"
    read -r _
  fi
}

# ---------------------------------------------------------------------------
# Curated corpus. In production these would be your consultants' authored
# docs, internal SAP knowledge bases, training material, ticket post-mortems.
# Public Wikipedia pages stand in for that here so the demo is reproducible.
# ---------------------------------------------------------------------------
SOURCES='["https://en.wikipedia.org/wiki/ABAP","https://en.wikipedia.org/wiki/SAP_S/4HANA","https://en.wikipedia.org/wiki/SAP_ERP","https://en.wikipedia.org/wiki/SAP_HANA"]'

# ---------------------------------------------------------------------------
# 1/5  Health
# ---------------------------------------------------------------------------
section "1/5  HEALTH  —  self-hosted stack, no external APIs"
curl -s -m 10 "$BASE_URL/health" | python3 -m json.tool
pause_for_step

# ---------------------------------------------------------------------------
# 2/5  Ingest
# ---------------------------------------------------------------------------
section "2/5  INGEST  —  load ONLY the curated corpus"
echo "  ${DIM}Corpus: ABAP, SAP S/4HANA, SAP ERP, SAP HANA (Wikipedia stand-ins)${RESET}"
echo "  ${DIM}Ingest is idempotent: re-running this demo overwrites prior chunks${RESET}"
echo "  ${DIM}for the same URLs instead of accumulating duplicates.${RESET}"
echo
curl -s -m 240 -X POST "$BASE_URL/ingest" \
  -H "Content-Type: application/json" \
  -d "{\"urls\": $SOURCES, \"collection\": \"$COLLECTION\", \"replace\": true}" \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'  ${GREEN}->${RESET} ingested {len(d[\"ingested_urls\"])} docs into {d[\"total_chunks\"]} chunks ({d[\"elapsed_ms\"]/1000:.1f}s)')
print(f'  ${DIM}collection: {d[\"collection\"]}${RESET}')
if d['failed_urls']:
    print(f'  ${YELLOW}warn:${RESET} {len(d[\"failed_urls\"])} URLs failed to ingest')
"
pause_for_step

# ---------------------------------------------------------------------------
# Helper: send a /retrieve request and pretty-print the result
# ---------------------------------------------------------------------------
retrieve () {
  local query="$1"
  curl -s -m 90 -X POST "$BASE_URL/retrieve" \
    -H "Content-Type: application/json" \
    -d "$(python3 -c "
import json, sys
print(json.dumps({
  'query': sys.argv[1],
  'collection': sys.argv[2],
  'top_k': 12,
  'rerank_top_k': 3,
  'num_queries': 3,
  'use_hybrid': True,
  'use_hyde': True,
  'min_relevance': 0.30,
}))
" "$query" "$COLLECTION")"
}

# ---------------------------------------------------------------------------
# 3/5  Semantic retrieval
# ---------------------------------------------------------------------------
section "3/5  SEMANTIC RETRIEVAL  —  natural-language question, cited answer"
SEMANTIC_Q="What is ABAP and how is it different from a general-purpose language like Java?"
echo "  ${BOLD}Q:${RESET} $SEMANTIC_Q"
echo
retrieve "$SEMANTIC_Q" | python3 -c "
import sys, json
d = json.load(sys.stdin)
if d.get('refused'):
    print(f'  ${RED}REFUSED${RESET} — unexpected: {d.get(\"refusal_reason\")}')
    sys.exit(0)
print(f'  ${GREEN}->${RESET} returned {d[\"total_reranked\"]} cited passages in {d[\"elapsed_ms\"]/1000:.1f}s')
print(f'  ${DIM}top rerank score: {d[\"top_relevance\"]:.3f}${RESET}')
print()
ctx = (d.get('answer_context') or '')[:1500]
for line in ctx.splitlines():
    print('   ', line)
print()
print(f'  ${BOLD}SOURCES${RESET} ${DIM}(every passage is auditable):${RESET}')
for c in d['citations']:
    print(f'    [{c[\"index\"]}] relevance {c[\"relevance\"]:.3f}  {c[\"source\"]}')
"
pause_for_step

# ---------------------------------------------------------------------------
# 4/5  Exact-token retrieval (hybrid BM25 + dense)
# ---------------------------------------------------------------------------
section "4/5  EXACT-TOKEN RETRIEVAL  —  hybrid BM25 + dense"
EXACT_Q="What are internal tables in ABAP and what are they used for?"
echo "  ${BOLD}Q:${RESET} $EXACT_Q"
echo "  ${DIM}'internal tables' is an ABAP-specific compound noun. The words${RESET}"
echo "  ${DIM}'internal' and 'tables' are generic on their own, so pure dense${RESET}"
echo "  ${DIM}retrieval can blur them; the BM25 sparse channel pulls the exact${RESET}"
echo "  ${DIM}compound back to the top via reciprocal rank fusion.${RESET}"
echo
retrieve "$EXACT_Q" | python3 -c "
import sys, json
d = json.load(sys.stdin)
if d.get('refused'):
    print(f'  ${YELLOW}REFUSED${RESET} (top relevance {d.get(\"top_relevance\", 0):.3f})')
    print(f'  reason: {d.get(\"refusal_reason\")}')
    sys.exit(0)
print(f'  ${GREEN}->${RESET} {d[\"total_reranked\"]} cited passages, top relevance {d[\"top_relevance\"]:.3f}')
print()
top = d['chunks'][0] if d['chunks'] else None
if top:
    snippet = top['text'][:500]
    for line in snippet.splitlines():
        print('   ', line)
print()
print(f'  ${BOLD}SOURCES${RESET}')
for c in d['citations']:
    print(f'    [{c[\"index\"]}] relevance {c[\"relevance\"]:.3f}  {c[\"source\"]}')
"
pause_for_step

# ---------------------------------------------------------------------------
# 5/5  Refusal — the no-hallucination guardrail
# ---------------------------------------------------------------------------
section "5/5  REFUSAL  —  off-corpus question, no hallucination"
OFF_Q="What is the chemical formula for water?"
echo "  ${BOLD}Q:${RESET} $OFF_Q"
echo "  ${DIM}A trivially-known fact — but it's NOT in the SAP corpus, so the${RESET}"
echo "  ${DIM}engine declines. A generic LLM would confidently answer H2O even${RESET}"
echo "  ${DIM}though that knowledge didn't come from your curated sources.${RESET}"
echo
retrieve "$OFF_Q" | python3 -c "
import sys, json
d = json.load(sys.stdin)
if d.get('refused'):
    print(f'  ${RED}${BOLD}REFUSED${RESET}  —  the engine returned an explicit non-answer')
    print(f'  ${DIM}{d[\"refusal_reason\"]}${RESET}')
    print()
    print(f'  ${BOLD}response:${RESET}')
    print(f'    refused          = true')
    print(f'    answer_context   = null         ${DIM}<- the LLM has nothing to fabricate from${RESET}')
    print(f'    chunks           = []')
    print(f'    citations        = []')
    print(f'    top_relevance    = {d[\"top_relevance\"] if d[\"top_relevance\"] is not None else \"null\"}')
    print(f'    total_retrieved  = {d[\"total_retrieved\"]}    ${DIM}<- candidates considered, all rejected${RESET}')
else:
    print(f'  ${YELLOW}WARNING${RESET}: engine did NOT refuse')
    print(f'  top relevance: {d.get(\"top_relevance\", 0):.3f}  (threshold was 0.30)')
    print(f'  -> consider raising min_relevance for stricter mode')
"
pause_for_step

# ---------------------------------------------------------------------------
# Outro
# ---------------------------------------------------------------------------
section "DEMO COMPLETE"
cat <<EOF
  ${BOLD}What this proves:${RESET}
    • Curated-corpus retrieval — no live web crawling, no surprise sources
    • Hybrid BM25 + dense — exact identifiers (ABAP keywords, transaction
      codes, error numbers) come back accurately
    • Structured refusal — when the corpus doesn't contain an answer, the
      engine says so explicitly instead of letting an LLM make one up

  ${DIM}Swap the SOURCES array for your own knowledge base and the behavior
  is identical. All retrieval data stays on your infrastructure.${RESET}

EOF
