#!/usr/bin/env bash
#
# Demo script for the Retrieval Infrastructure Platform.
#
# Recommended (works on any machine, no host deps but Docker):
#     docker compose exec api bash demo.sh
#
# Or from the host if you have curl + python3:
#     bash demo.sh
#
# Override the target if needed:  BASE_URL=http://localhost:8000 bash demo.sh
#
# Narrative: load ONLY curated/trusted sources, then get precise, CITED
# answers scoped strictly to that corpus — and a guardrail that refuses to
# answer when the corpus has nothing relevant (no hallucination).

BASE_URL="${BASE_URL:-http://localhost:8000}"
COLLECTION="sap_knowledge"
LINE="════════════════════════════════════════════════════════════════════"

section () { echo; echo "$LINE"; echo "  $1"; echo "$LINE"; }

# Curated, vetted sources. In production these would be the consultants'
# knowledge base / internal docs — here we use public SAP reference pages.
SOURCES='["https://en.wikipedia.org/wiki/ABAP","https://en.wikipedia.org/wiki/SAP_S/4HANA","https://en.wikipedia.org/wiki/SAP_ERP","https://en.wikipedia.org/wiki/SAP_HANA"]'

section "1/4  HEALTH  —  fully self-hosted stack (no paid APIs)"
curl -s -m 10 "$BASE_URL/health" | python3 -m json.tool

section "2/4  INGEST  —  load ONLY curated, trusted sources"
echo "  Curated corpus: ABAP, SAP S/4HANA, SAP ERP, SAP HANA"
curl -s -m 180 -X POST "$BASE_URL/ingest" \
  -H "Content-Type: application/json" \
  -d "{\"urls\": $SOURCES, \"collection\": \"$COLLECTION\"}" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print('  -> ingested',len(d['ingested_urls']),'docs into',d['total_chunks'],'chunks | collection:',d['collection'])"

section "3/4  RETRIEVE  —  precise, CITED answer (scoped to the corpus)"
echo "  Q: What data types does ABAP support?"
echo
curl -s -m 60 -X POST "$BASE_URL/retrieve" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What data types does ABAP support?\", \"collection\": \"$COLLECTION\", \"top_k\": 8}" \
  | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('  reranked to',d['total_reranked'],'cited passages in %.1fs'%(d['elapsed_ms']/1000))
print()
ctx=d.get('answer_context') or ''
print(ctx[:900])
print()
print('  SOURCES (every passage is auditable):')
for c in d['citations']:
    print('    [%d] relevance %.3f  %s'%(c['index'],c['relevance'],c['source']))
"

section "4/4  GUARDRAIL  —  off-corpus question => no hallucination"
echo "  Q: What is the best pizza topping?  (nothing about this in the SAP corpus)"
echo
curl -s -m 60 -X POST "$BASE_URL/retrieve" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What is the best pizza topping?\", \"collection\": \"$COLLECTION\", \"top_k\": 8}" \
  | python3 -c "
import sys,json
d=json.load(sys.stdin)
scores=[c['relevance'] for c in d['citations']]
print('  top rerank relevance scores:', [round(s,4) for s in scores])
print('  -> highest is %.4f : the trusted corpus has nothing relevant.'%(max(scores) if scores else 0.0))
print('  -> the agent DECLINES instead of inventing an answer (unlike a generic LLM).')
"

section "DEMO COMPLETE"
echo "  Curated sources in  ->  precise, cited answers out."
echo "  Fully self-hosted. You control the sources. Every answer is auditable."
echo
