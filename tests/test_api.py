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
    """Dense-only retrieve path on a legacy collection: no hybrid, no
    multi-query, no MMR. Confirms the simplest configuration still works
    end-to-end through the new pipeline."""
    from app.models.requests import RetrieveRequest

    mock_embedding_service = MagicMock()
    # New pipeline uses embed_texts (batched) for both single and multi-query.
    mock_embedding_service.embed_texts = AsyncMock(return_value=[[0.1] * 384])

    fake_point = MagicMock()
    fake_point.id = "point-1"
    fake_point.score = 0.85
    fake_point.payload = {
        "text": "Transformers use self-attention mechanisms.",
        "url": "https://example.com",
        "title": "Transformers",
        "chunk_index": 0,
        "total_chunks": 1,
        "word_count": 6,
    }
    fake_point.vector = None

    mock_qdrant = MagicMock()
    mock_qdrant.search = AsyncMock(return_value=[fake_point])
    mock_qdrant.search_sparse = AsyncMock(return_value=[])
    mock_qdrant.ensure_collection = AsyncMock(return_value=None)
    mock_qdrant.is_hybrid = MagicMock(return_value=False)
    mock_qdrant._default_collection = "retrieval_chunks"

    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(side_effect=lambda q, chunks, top_k: chunks[:top_k])

    mock_expander = MagicMock()
    mock_expander.expand_for_retrieval = AsyncMock(return_value=["how does attention work"])

    from app.retrieval.pipeline import RetrievalPipeline

    pipeline = RetrievalPipeline(
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

    # Use num_queries=1 so we exercise the single-query path; hybrid off so
    # we don't need a sparse encoder. min_relevance=0 disables the
    # refusal gate (the mock reranker doesn't set realistic scores, so the
    # gate would otherwise correctly refuse here — see test_refusal.py).
    request = RetrieveRequest(
        query="how does attention work",
        top_k=10,
        rerank_top_k=5,
        num_queries=1,
        use_hybrid=False,
        use_mmr=False,
        use_hyde=False,
        min_relevance=0.0,
    )
    response = await pipeline.retrieve(request)

    assert response.query == "how does attention work"
    assert len(response.chunks) >= 1
    assert response.total_retrieved == 1


@pytest.mark.asyncio
async def test_retrieve_pipeline_multi_query_and_hybrid(redis_cache):
    """Exercises the multi-query + hybrid path: 3 expanded queries → 3
    dense searches + 1 BM25 sparse search → RRF fusion → rerank."""
    from app.models.requests import RetrieveRequest

    mock_embedding_service = MagicMock()
    # 3 variants × 1 vector each (all the same dummy for simplicity).
    mock_embedding_service.embed_texts = AsyncMock(return_value=[[0.1] * 384] * 3)

    def _point(id_, url, chunk_index, text):
        p = MagicMock()
        p.id = id_
        p.score = 0.8
        p.payload = {
            "text": text,
            "url": url,
            "title": "Doc",
            "chunk_index": chunk_index,
            "total_chunks": 2,
            "word_count": len(text.split()),
        }
        p.vector = None
        return p

    p1 = _point("id-1", "https://a.example.com", 0, "self-attention drives transformer power")
    p2 = _point("id-2", "https://b.example.com", 0, "multi-head attention attends to multiple positions")
    p3 = _point("id-3", "https://a.example.com", 1, "transformers replaced rnn architectures")

    mock_qdrant = MagicMock()
    # Each dense call returns the same two top hits; sparse adds a third.
    mock_qdrant.search = AsyncMock(return_value=[p1, p2])
    mock_qdrant.search_sparse = AsyncMock(return_value=[p3, p1])
    mock_qdrant.ensure_collection = AsyncMock(return_value=None)
    mock_qdrant.is_hybrid = MagicMock(return_value=True)
    mock_qdrant._default_collection = "retrieval_chunks"

    mock_reranker = MagicMock()
    mock_reranker.rerank = AsyncMock(side_effect=lambda q, chunks, top_k: chunks[:top_k])

    mock_expander = MagicMock()
    mock_expander.expand_for_retrieval = AsyncMock(
        return_value=[
            "how does attention work",
            "attention work declarative",
            "attention refers to attention work hypothetical",
        ]
    )

    mock_sparse = MagicMock()
    mock_sparse.encode_query = AsyncMock(
        return_value=MagicMock(indices=[1, 2, 3], values=[0.5, 0.4, 0.3])
    )

    from app.retrieval.pipeline import RetrievalPipeline

    pipeline = RetrievalPipeline(
        search_service=MagicMock(),
        query_expander=mock_expander,
        crawler=MagicMock(),
        extraction_pipeline=MagicMock(),
        embedding_service=mock_embedding_service,
        reranker_service=mock_reranker,
        qdrant_store=mock_qdrant,
        postgres=MagicMock(),
        sparse_encoder=mock_sparse,
    )

    request = RetrieveRequest(
        query="how does attention work",
        top_k=10,
        rerank_top_k=3,
        num_queries=3,
        use_hybrid=True,
        use_hyde=True,
        use_mmr=False,
        # See note in test_retrieve_pipeline_end_to_end; refusal-gate tests
        # live in tests/test_refusal.py.
        min_relevance=0.0,
    )
    response = await pipeline.retrieve(request)

    # 3 dense calls (one per variant) + 1 sparse call
    assert mock_qdrant.search.await_count == 3
    assert mock_qdrant.search_sparse.await_count == 1
    # All 3 unique chunks should make it through fusion → rerank
    assert response.total_retrieved == 3
    assert len(response.chunks) <= 3


@pytest.mark.asyncio
async def test_ingest_replace_deletes_before_upsert(redis_cache):
    """Idempotent ingest: when ``replace=True`` (the default), the pipeline
    must call delete_by_urls for every ingested URL before upserting."""
    from app.models.requests import IngestRequest
    from app.models.responses import CrawlResult
    from app.retrieval.pipeline import RetrievalPipeline

    crawl_result = CrawlResult(
        url="https://example.com",
        status_code=200,
        html="<html><body>hi</body></html>",
        final_url="https://example.com",
        content_type="text/html",
        elapsed_ms=10.0,
        used_playwright=False,
        cached=False,
    )
    mock_crawler = MagicMock()
    mock_crawler.crawl_many = AsyncMock(return_value=([crawl_result], []))

    mock_extraction = MagicMock()
    mock_extraction.extract_and_chunk = AsyncMock(
        return_value=[
            {
                "text": "first chunk",
                "url": "https://example.com",
                "title": "T",
                "chunk_index": 0,
                "total_chunks": 1,
                "word_count": 2,
            }
        ]
    )

    mock_embedding_service = MagicMock()
    mock_embedding_service.embed_texts = AsyncMock(return_value=[[0.1] * 384])

    mock_qdrant = MagicMock()
    mock_qdrant.ensure_collection = AsyncMock(return_value=None)
    mock_qdrant.is_hybrid = MagicMock(return_value=False)
    mock_qdrant.delete_by_urls = AsyncMock(return_value=None)
    mock_qdrant.upsert_chunks = AsyncMock(return_value=1)
    mock_qdrant._default_collection = "retrieval_chunks"

    pipeline = RetrievalPipeline(
        search_service=MagicMock(),
        query_expander=MagicMock(),
        crawler=mock_crawler,
        extraction_pipeline=mock_extraction,
        embedding_service=mock_embedding_service,
        reranker_service=MagicMock(),
        qdrant_store=mock_qdrant,
        postgres=MagicMock(),
        sparse_encoder=None,
    )

    request = IngestRequest(urls=["https://example.com"])  # type: ignore[arg-type]
    response = await pipeline.ingest(request)

    assert response.total_chunks == 1
    # delete_by_urls must be called with the URL we ingested, BEFORE upsert
    mock_qdrant.delete_by_urls.assert_awaited_once()
    args = mock_qdrant.delete_by_urls.await_args
    urls_arg = args.args[0] if args.args else args.kwargs.get("urls")
    assert "https://example.com" in urls_arg
    mock_qdrant.upsert_chunks.assert_awaited_once()
