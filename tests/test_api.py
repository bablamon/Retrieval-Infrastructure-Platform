from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_health_returns_structure():

    mock_redis = MagicMock()
    mock_redis.health_check = AsyncMock(return_value=True)
    mock_postgres = MagicMock()
    mock_postgres.health_check = AsyncMock(return_value=True)
    mock_qdrant = MagicMock()
    mock_qdrant.health_check = AsyncMock(return_value=True)

    with (
        patch("app.core.dependencies.get_redis", return_value=mock_redis),
        patch("app.core.dependencies.get_postgres", return_value=mock_postgres),
        patch("app.core.dependencies.get_qdrant", return_value=mock_qdrant),
    ):
        from app.api.health import health
        result = await health(
            redis=mock_redis,
            postgres=mock_postgres,
            qdrant=mock_qdrant,
        )
        assert result.status == "healthy"
        assert result.services["redis"] is True
        assert result.services["postgres"] is True
        assert result.services["qdrant"] is True


@pytest.mark.asyncio
async def test_health_degraded_when_one_service_down():
    from app.api.health import health

    mock_redis = MagicMock()
    mock_redis.health_check = AsyncMock(return_value=True)
    mock_postgres = MagicMock()
    mock_postgres.health_check = AsyncMock(return_value=False)  # down
    mock_qdrant = MagicMock()
    mock_qdrant.health_check = AsyncMock(return_value=True)

    result = await health(
        redis=mock_redis,
        postgres=mock_postgres,
        qdrant=mock_qdrant,
    )
    assert result.status == "degraded"
    assert result.services["postgres"] is False


@pytest.mark.asyncio
async def test_query_generate():
    from app.api.query import generate_queries
    from app.models.requests import QueryGenerateRequest
    from app.search.query_expander import QueryExpander

    request = QueryGenerateRequest(query="how does attention work", num_queries=5)
    result = await generate_queries(request, expander=QueryExpander())

    assert result.original == "how does attention work"
    assert len(result.queries) >= 1
    assert result.elapsed_ms >= 0


@pytest.mark.asyncio
async def test_search_cached_result(redis_cache):
    from app.models.requests import SearchRequest
    from app.models.responses import SearchResult
    from app.search.cache import CachedSearchService

    mock_searxng = MagicMock()
    mock_searxng.search = AsyncMock(return_value=[
        SearchResult(url="https://example.com", title="Test", snippet="snippet", engine="google")
    ])

    service = CachedSearchService(mock_searxng, redis_cache, cache_ttl=900)

    request = SearchRequest(query="transformer architecture", use_cache=True)

    # First call
    r1 = await service.search(request)
    assert r1.cached is False
    assert mock_searxng.search.call_count == 1

    # Second call — from cache
    r2 = await service.search(request)
    assert r2.cached is True
    assert mock_searxng.search.call_count == 1  # not called again


@pytest.mark.asyncio
async def test_retrieve_pipeline_end_to_end(redis_cache):
    from app.models.requests import RetrieveRequest

    mock_embedding_service = MagicMock()
    mock_embedding_service.embed_query = AsyncMock(return_value=[0.1] * 384)

    fake_point = MagicMock()
    fake_point.score = 0.85
    fake_point.payload = {
        "text": "Transformers use self-attention mechanisms.",
        "url": "https://example.com",
        "title": "Transformers",
        "chunk_index": 0,
        "total_chunks": 1,
        "word_count": 6,
    }

    mock_qdrant = MagicMock()
    mock_qdrant.search = AsyncMock(return_value=[fake_point])

    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(side_effect=lambda q, chunks, top_k: chunks[:top_k])

    from app.retrieval.pipeline import RetrievalPipeline

    pipeline = RetrievalPipeline(
        search_service=MagicMock(),
        query_expander=MagicMock(),
        crawler=MagicMock(),
        extraction_pipeline=MagicMock(),
        embedding_service=mock_embedding_service,
        reranker_service=mock_reranker,
        qdrant_store=mock_qdrant,
        postgres=MagicMock(),
    )

    request = RetrieveRequest(query="how does attention work", top_k=10, rerank_top_k=5)
    response = await pipeline.retrieve(request)

    assert response.query == "how does attention work"
    assert len(response.chunks) >= 1
    assert response.total_retrieved == 1
