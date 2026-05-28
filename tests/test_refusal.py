"""Tests for the no-hallucination guardrail.

The guardrail is a hard relevance gate applied AFTER reranking: any chunk
with a rerank score below ``request.min_relevance`` is dropped, and if the
filter removes everything the pipeline returns a structured refusal rather
than picking the least-bad weak chunk.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.models.requests import RetrieveRequest
from app.retrieval.pipeline import RetrievalPipeline


def _chunk_point(point_id: str, url: str, chunk_index: int, text: str):
    p = MagicMock()
    p.id = point_id
    p.score = 0.5
    p.payload = {
        "text": text,
        "url": url,
        "title": "Doc",
        "chunk_index": chunk_index,
        "total_chunks": 1,
        "word_count": len(text.split()),
    }
    p.vector = None
    return p


def _make_pipeline(
    *,
    qdrant_search_result: list,
    rerank_side_effect,
    expander_variants: list[str] | None = None,
) -> RetrievalPipeline:
    """Build a pipeline with the heavy collaborators stubbed.

    ``rerank_side_effect`` controls what scores the cross-encoder hands back;
    each test uses it to simulate either confident answers or noise.
    """
    mock_embedding_service = MagicMock()
    mock_embedding_service.embed_texts = AsyncMock(return_value=[[0.1] * 384])

    mock_qdrant = MagicMock()
    mock_qdrant.search = AsyncMock(return_value=qdrant_search_result)
    mock_qdrant.search_sparse = AsyncMock(return_value=[])
    mock_qdrant.ensure_collection = AsyncMock(return_value=None)
    mock_qdrant.is_hybrid = MagicMock(return_value=False)
    mock_qdrant._default_collection = "retrieval_chunks"

    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(side_effect=rerank_side_effect)

    mock_expander = MagicMock()
    mock_expander.expand_for_retrieval = AsyncMock(
        return_value=expander_variants or ["the original query"]
    )

    return RetrievalPipeline(
        search_service=MagicMock(),
        query_expander=mock_expander,
        crawler=MagicMock(),
        extraction_pipeline=MagicMock(),
        embedding_service=mock_embedding_service,
        reranker_service=mock_reranker,
        qdrant_store=mock_qdrant,
        postgres=MagicMock(),
        sparse_encoder=None,
    )


# ---------------------------------------------------------------------------
# Refusal triggers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refuses_when_no_candidates_exist():
    """Empty collection / over-filtered search → ``no_candidates`` refusal."""

    async def rerank(query, chunks, top_k):
        return chunks[:top_k]  # never called with anything anyway

    pipeline = _make_pipeline(qdrant_search_result=[], rerank_side_effect=rerank)
    request = RetrieveRequest(
        query="anything", num_queries=1, use_hybrid=False, use_hyde=False
    )
    response = await pipeline.retrieve(request)

    assert response.refused is True
    assert response.chunks == []
    assert response.citations == []
    assert response.answer_context is None
    assert response.total_retrieved == 0
    assert response.refusal_reason is not None
    assert response.refusal_reason.startswith("no_candidates:")
    assert response.top_relevance is None


@pytest.mark.asyncio
async def test_refuses_when_all_chunks_below_threshold():
    """Candidates exist but every cross-encoder score is below
    min_relevance → ``below_threshold`` refusal. The pipeline must NOT
    return the least-bad chunk."""
    points = [
        _chunk_point("p1", "https://a.example.com", 0, "vaguely related text"),
        _chunk_point("p2", "https://b.example.com", 0, "another weak match"),
    ]

    async def rerank(query, chunks, top_k):
        # Both candidates score way below the default 0.3 threshold.
        chunks[0].rerank_score = 0.12
        chunks[1].rerank_score = 0.08
        return chunks[:top_k]

    pipeline = _make_pipeline(qdrant_search_result=points, rerank_side_effect=rerank)
    request = RetrieveRequest(
        query="how does quantum chromodynamics work",
        num_queries=1,
        use_hybrid=False,
        use_hyde=False,
        min_relevance=0.3,
    )
    response = await pipeline.retrieve(request)

    assert response.refused is True
    assert response.chunks == []
    assert response.citations == []
    assert response.answer_context is None
    # We DID consider candidates, just rejected them
    assert response.total_retrieved == 2
    assert response.refusal_reason is not None
    assert response.refusal_reason.startswith("below_threshold:")
    # The best score we saw is surfaced so the caller can tune the gate
    assert response.top_relevance is not None
    assert abs(response.top_relevance - 0.12) < 1e-6


# ---------------------------------------------------------------------------
# Non-refusal paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_passes_when_at_least_one_chunk_above_threshold():
    """One strong chunk + one weak chunk: the weak one is dropped, the
    strong one is returned, refused stays False."""
    points = [
        _chunk_point("p1", "https://a.example.com", 0, "strong relevant answer"),
        _chunk_point("p2", "https://b.example.com", 0, "weak tangential text"),
    ]

    async def rerank(query, chunks, top_k):
        chunks[0].rerank_score = 0.82
        chunks[1].rerank_score = 0.05
        return chunks[:top_k]

    pipeline = _make_pipeline(qdrant_search_result=points, rerank_side_effect=rerank)
    request = RetrieveRequest(
        query="what is attention",
        num_queries=1,
        use_hybrid=False,
        use_hyde=False,
        rerank_top_k=2,
        min_relevance=0.3,
    )
    response = await pipeline.retrieve(request)

    assert response.refused is False
    assert response.refusal_reason is None
    assert len(response.chunks) == 1
    assert response.chunks[0].rerank_score == 0.82
    assert response.top_relevance == 0.82
    # Context pack contains only the surviving chunk
    assert response.answer_context is not None
    assert "strong relevant answer" in response.answer_context


@pytest.mark.asyncio
async def test_threshold_zero_disables_gate():
    """min_relevance=0 is the legacy behavior: return whatever the reranker
    ranks highest, even if scores are near zero. Useful for benchmarking
    and for users who explicitly want best-effort retrieval."""
    points = [_chunk_point("p1", "https://a.example.com", 0, "weak match")]

    async def rerank(query, chunks, top_k):
        chunks[0].rerank_score = 0.05
        return chunks[:top_k]

    pipeline = _make_pipeline(qdrant_search_result=points, rerank_side_effect=rerank)
    request = RetrieveRequest(
        query="anything",
        num_queries=1,
        use_hybrid=False,
        use_hyde=False,
        min_relevance=0.0,
    )
    response = await pipeline.retrieve(request)

    assert response.refused is False
    assert len(response.chunks) == 1
    assert response.chunks[0].rerank_score == 0.05


@pytest.mark.asyncio
async def test_refusal_message_reports_top_score():
    """The structured refusal reason should include the actual top score
    seen, so a caller can decide whether to lower the threshold and retry."""
    points = [_chunk_point("p1", "https://a.example.com", 0, "barely related")]

    async def rerank(query, chunks, top_k):
        chunks[0].rerank_score = 0.27
        return chunks[:top_k]

    pipeline = _make_pipeline(qdrant_search_result=points, rerank_side_effect=rerank)
    request = RetrieveRequest(
        query="off-corpus question",
        num_queries=1,
        use_hybrid=False,
        use_hyde=False,
        min_relevance=0.30,
    )
    response = await pipeline.retrieve(request)

    assert response.refused is True
    assert response.refusal_reason is not None
    assert "0.270" in response.refusal_reason
    assert "0.300" in response.refusal_reason
