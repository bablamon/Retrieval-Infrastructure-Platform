import time

from app.core.logging import get_logger
from app.crawler.async_crawler import AsyncCrawler
from app.embeddings.embedding_service import EmbeddingService
from app.extractor.pipeline import ExtractionPipeline
from app.models.requests import IngestRequest, RetrieveRequest
from app.models.responses import (
    ChunkMetadata,
    IngestResponse,
    RetrievedChunk,
    RetrieveResponse,
)
from app.reranker.reranker_service import RerankerService
from app.search.cache import CachedSearchService
from app.search.query_expander import QueryExpander
from app.storage.postgres_client import PostgresClient
from app.storage.qdrant_client import ChunkPayload, QdrantStore

logger = get_logger(__name__)


class RetrievalPipeline:
    def __init__(
        self,
        search_service: CachedSearchService,
        query_expander: QueryExpander,
        crawler: AsyncCrawler,
        extraction_pipeline: ExtractionPipeline,
        embedding_service: EmbeddingService,
        reranker_service: RerankerService,
        qdrant_store: QdrantStore,
        postgres: PostgresClient,
    ) -> None:
        self._search = search_service
        self._expander = query_expander
        self._crawler = crawler
        self._extraction = extraction_pipeline
        self._embeddings = embedding_service
        self._reranker = reranker_service
        self._qdrant = qdrant_store
        self._postgres = postgres

    async def ingest(self, request: IngestRequest) -> IngestResponse:
        start = time.monotonic()
        urls = [str(u) for u in request.urls]
        collection = request.collection

        await self._qdrant.ensure_collection(collection)

        results, failures = await self._crawler.crawl_many(
            urls=urls,
            use_playwright=request.use_playwright,
            respect_robots=True,
            use_cache=True,
        )

        all_chunks: list[dict] = []
        ingested_urls: list[str] = []

        for crawl_result in results:
            if not crawl_result.html:
                failures.append({"url": crawl_result.url, "error": "empty HTML"})
                continue

            chunks = await self._extraction.extract_and_chunk(
                html=crawl_result.html,
                url=crawl_result.final_url,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                extra_metadata=request.metadata,
            )

            if chunks:
                all_chunks.extend(chunks)
                ingested_urls.append(crawl_result.url)

        total_chunks = 0
        if all_chunks:
            texts = [c["text"] for c in all_chunks]
            embeddings = await self._embeddings.embed_texts(texts)

            chunk_payloads = [
                ChunkPayload(
                    text=chunk["text"],
                    embedding=emb,
                    metadata={k: v for k, v in chunk.items() if k != "text"},
                )
                for chunk, emb in zip(all_chunks, embeddings, strict=False)
            ]

            total_chunks = await self._qdrant.upsert_chunks(chunk_payloads, collection)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "ingest_complete",
            urls=len(ingested_urls),
            chunks=total_chunks,
            elapsed_ms=round(elapsed_ms, 1),
        )

        return IngestResponse(
            ingested_urls=ingested_urls,
            failed_urls=[{"url": f["url"], "error": f["error"]} for f in failures],
            total_chunks=total_chunks,
            collection=collection or self._qdrant._default_collection,
            elapsed_ms=elapsed_ms,
        )

    async def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        start = time.monotonic()

        query_vector = await self._embeddings.embed_query(request.query)

        scored_points = await self._qdrant.search(
            query_vector=query_vector,
            top_k=request.top_k,
            collection=request.collection,
            filters=request.filters,
            score_threshold=request.score_threshold,
        )

        chunks: list[RetrievedChunk] = []
        for point in scored_points:
            payload = point.payload or {}
            text = payload.pop("text", "")
            metadata = ChunkMetadata(
                url=payload.get("url", ""),
                title=payload.get("title"),
                chunk_index=payload.get("chunk_index", 0),
                total_chunks=payload.get("total_chunks", 1),
                word_count=payload.get("word_count", 0),
                extra={k: v for k, v in payload.items()
                       if k not in ("url", "title", "chunk_index", "total_chunks", "word_count")},
            )
            chunks.append(
                RetrievedChunk(text=text, score=point.score, metadata=metadata)
            )

        total_retrieved = len(chunks)
        reranked = await self._reranker.rerank(request.query, chunks, request.rerank_top_k)
        elapsed_ms = (time.monotonic() - start) * 1000

        logger.info(
            "retrieve_complete",
            query=request.query,
            retrieved=total_retrieved,
            reranked=len(reranked),
            elapsed_ms=round(elapsed_ms, 1),
        )

        return RetrieveResponse(
            query=request.query,
            chunks=reranked,
            total_retrieved=total_retrieved,
            total_reranked=len(reranked),
            elapsed_ms=elapsed_ms,
        )

    async def search_and_ingest(
        self,
        query: str,
        num_results: int = 10,
        collection: str | None = None,
    ) -> IngestResponse:
        from app.models.requests import SearchRequest

        search_request = SearchRequest(query=query, num_results=num_results)
        search_response = await self._search.search(search_request)
        urls = [r.url for r in search_response.results if r.url]

        ingest_request = IngestRequest(urls=urls, collection=collection)  # type: ignore[arg-type]
        return await self.ingest(ingest_request)
