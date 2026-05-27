import asyncio
import time

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Singletons
_redis_cache = None
_postgres_client = None
_qdrant_store = None
_embedding_service = None
_reranker_service = None
_searxng_client = None
_cached_search_service = None
_crawler = None
_retrieval_pipeline = None
_start_time: float = 0.0


async def initialize_dependencies(load_reranker: bool = True) -> None:
    """Initialize all shared services.

    Args:
        load_reranker: When False (e.g. for the ARQ worker, which only ingests
            and crawls), the reranker cross-encoder model is NOT loaded into
            memory, saving ~1GB+ of RAM. Reranking only happens in /retrieve,
            which is served by the API process where this stays True.
    """
    global _redis_cache, _postgres_client, _qdrant_store
    global _embedding_service, _reranker_service, _searxng_client
    global _cached_search_service, _crawler, _retrieval_pipeline, _start_time

    from app.crawler.async_crawler import AsyncCrawler
    from app.crawler.http_crawler import HTTPCrawler
    from app.crawler.playwright_crawler import PlaywrightCrawler
    from app.crawler.robots import RobotsCache
    from app.embeddings.embedding_service import EmbeddingService
    from app.extractor.bs4_extractor import BS4Extractor
    from app.extractor.markdown_cleaner import MarkdownCleaner
    from app.extractor.pipeline import ExtractionPipeline
    from app.extractor.trafilatura_extractor import TrafilaturaExtractor
    from app.reranker.reranker_service import RerankerService
    from app.retrieval.pipeline import RetrievalPipeline
    from app.search.cache import CachedSearchService
    from app.search.query_expander import QueryExpander
    from app.search.searxng_client import SearXNGClient
    from app.storage.postgres_client import PostgresClient
    from app.storage.qdrant_client import QdrantStore
    from app.storage.redis_client import RedisCache

    _start_time = time.monotonic()

    # Storage
    _redis_cache = RedisCache(settings.redis_url, default_ttl=settings.redis_cache_ttl)
    await _redis_cache.connect()

    _postgres_client = PostgresClient(
        settings.postgres_dsn,
        min_size=settings.postgres_pool_min,
        max_size=settings.postgres_pool_max,
    )
    await _postgres_client.connect()

    _qdrant_store = QdrantStore(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection=settings.qdrant_collection,
        vector_size=settings.qdrant_vector_size,
    )
    await _qdrant_store.connect()

    # Search
    _searxng_client = SearXNGClient(settings.searxng_url, timeout=settings.app_request_timeout)
    await _searxng_client.start()

    _cached_search_service = CachedSearchService(
        _searxng_client,
        _redis_cache,
        cache_ttl=settings.redis_search_cache_ttl,
    )
    query_expander = QueryExpander()

    # Crawler
    playwright_semaphore = asyncio.Semaphore(settings.app_max_playwright_pages)
    crawl_semaphore = asyncio.Semaphore(settings.app_max_crawl_concurrency)

    http_crawler = HTTPCrawler(timeout=settings.app_request_timeout)
    await http_crawler.start()

    playwright_crawler = PlaywrightCrawler(playwright_semaphore, timeout=settings.app_request_timeout * 1000)
    await playwright_crawler.start()

    robots_cache = RobotsCache(
        _redis_cache,
        cache_ttl=settings.redis_robots_cache_ttl,
    )

    _crawler = AsyncCrawler(
        http_crawler=http_crawler,
        playwright_crawler=playwright_crawler,
        robots_cache=robots_cache,
        redis_cache=_redis_cache,
        postgres=_postgres_client,
        crawl_semaphore=crawl_semaphore,
        cache_ttl=settings.redis_cache_ttl,
    )

    # Extractor
    extraction_pipeline = ExtractionPipeline(
        trafilatura=TrafilaturaExtractor(),
        bs4=BS4Extractor(),
        cleaner=MarkdownCleaner(),
    )

    # Embeddings + Reranker (model loading happens here — blocking)
    _embedding_service = EmbeddingService(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
        cache=_redis_cache,
    )
    _embedding_service.load_model()

    _reranker_service = RerankerService(
        model_name=settings.reranker_model,
        use_fp16=settings.reranker_use_fp16,
        top_k=settings.reranker_top_k,
    )
    if load_reranker:
        _reranker_service.load_model()
    else:
        logger.info("reranker_load_skipped", reason="not_needed_in_this_process")

    # Retrieval pipeline
    _retrieval_pipeline = RetrievalPipeline(
        search_service=_cached_search_service,
        query_expander=query_expander,
        crawler=_crawler,
        extraction_pipeline=extraction_pipeline,
        embedding_service=_embedding_service,
        reranker_service=_reranker_service,
        qdrant_store=_qdrant_store,
        postgres=_postgres_client,
    )

    logger.info("dependencies_initialized")


async def disconnect_all_dependencies() -> None:
    global _redis_cache, _postgres_client, _qdrant_store, _searxng_client

    if _redis_cache:
        await _redis_cache.disconnect()
    if _postgres_client:
        await _postgres_client.disconnect()
    if _qdrant_store:
        await _qdrant_store.disconnect()
    if _searxng_client:
        await _searxng_client.stop()

    logger.info("dependencies_disconnected")


# FastAPI Depends providers

def get_redis():
    if _redis_cache is None:
        raise RuntimeError("RedisCache not initialized")
    return _redis_cache


def get_postgres():
    if _postgres_client is None:
        raise RuntimeError("PostgresClient not initialized")
    return _postgres_client


def get_qdrant():
    if _qdrant_store is None:
        raise RuntimeError("QdrantStore not initialized")
    return _qdrant_store


def get_embedding_service():
    if _embedding_service is None:
        raise RuntimeError("EmbeddingService not initialized")
    return _embedding_service


def get_reranker_service():
    if _reranker_service is None:
        raise RuntimeError("RerankerService not initialized")
    return _reranker_service


def get_crawler():
    if _crawler is None:
        raise RuntimeError("AsyncCrawler not initialized")
    return _crawler


def get_retrieval_pipeline():
    if _retrieval_pipeline is None:
        raise RuntimeError("RetrievalPipeline not initialized")
    return _retrieval_pipeline


def get_search_service():
    if _cached_search_service is None:
        raise RuntimeError("CachedSearchService not initialized")
    return _cached_search_service


def get_query_expander():
    from app.search.query_expander import QueryExpander
    return QueryExpander()


def get_extraction_pipeline():
    from app.extractor.bs4_extractor import BS4Extractor
    from app.extractor.markdown_cleaner import MarkdownCleaner
    from app.extractor.pipeline import ExtractionPipeline
    from app.extractor.trafilatura_extractor import TrafilaturaExtractor
    return ExtractionPipeline(
        trafilatura=TrafilaturaExtractor(),
        bs4=BS4Extractor(),
        cleaner=MarkdownCleaner(),
    )


def get_uptime() -> float:
    return time.monotonic() - _start_time if _start_time else 0.0
