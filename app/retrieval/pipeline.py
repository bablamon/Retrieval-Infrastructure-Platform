"""End-to-end retrieval pipeline.

Ingest path:
    crawl → extract → chunk → embed (dense) → encode (BM25 sparse)
                                  → delete-by-url (if replace) → upsert

Retrieve path:
    query expansion → embed each variant → dense search per variant
                    ↘ BM25 sparse search ↗
                    → reciprocal-rank fusion → cross-encoder rerank
                    → (optional) MMR diversity → citation pack

Key design points:

* **Multi-query**: the QueryExpander produces N variants (declarative
  rewrite, HyDE hypothetical answers, synonyms). Each variant is embedded
  and run as its own dense search; results are fused via RRF. This costs N
  embed calls (small after cache warm-up) and N Qdrant searches but
  measurably improves recall on long-form / question-form queries.

* **Hybrid**: in parallel with the dense searches, a single BM25 sparse
  search is run and its ranking joins the same RRF fusion. RRF is
  rank-based so dense and sparse scores don't need calibration.

* **Idempotent ingest**: on ``request.replace=True`` (the default), all
  prior chunks for each ingested URL are deleted before the new chunks
  are upserted. Combined with deterministic point IDs from
  ``(url, chunk_index)``, this gives true "set-to-this-state" semantics
  rather than the previous "accumulate forever" behavior.

* **MMR**: optional post-rerank diversity step that suppresses near-
  duplicate chunks (common when a single source page produces several
  overlapping passages).
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Hashable
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.crawler.async_crawler import AsyncCrawler
from app.embeddings.embedding_service import EmbeddingService
from app.extractor.pipeline import ExtractionPipeline
from app.models.requests import IngestRequest, RetrieveRequest, SearchRequest
from app.models.responses import (
    ChunkMetadata,
    Citation,
    IngestResponse,
    RetrievedChunk,
    RetrieveResponse,
)
from app.reranker.reranker_service import RerankerService
from app.search.cache import CachedSearchService
from app.search.fusion import maximal_marginal_relevance, reciprocal_rank_fusion
from app.search.query_expander import QueryExpander
from app.search.sparse_encoder import BM25SparseEncoder
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
        sparse_encoder: BM25SparseEncoder | None = None,
    ) -> None:
        self._search = search_service
        self._expander = query_expander
        self._crawler = crawler
        self._extraction = extraction_pipeline
        self._embeddings = embedding_service
        self._reranker = reranker_service
        self._qdrant = qdrant_store
        self._postgres = postgres
        self._sparse = sparse_encoder

    # ==================================================================
    # Ingest
    # ==================================================================
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
            # Idempotent replace: wipe prior chunks for every URL we are
            # about to re-ingest. Deterministic point IDs alone (see
            # qdrant_client._point_id) handle the common case, but a
            # re-crawl that produces *fewer* chunks would otherwise leave
            # orphans at higher chunk indices.
            if request.replace and ingested_urls:
                urls_to_replace = list({c.get("url", "") for c in all_chunks if c.get("url")})
                await self._qdrant.delete_by_urls(urls_to_replace, collection=collection)

            texts = [c["text"] for c in all_chunks]
            embeddings = await self._embeddings.embed_texts(texts)

            sparse_pairs: list[tuple[list[int], list[float]]] = []
            if self._sparse and self._qdrant.is_hybrid(collection):
                resolved_collection = collection or self._qdrant._default_collection
                sparse_vecs = await self._sparse.encode_documents(
                    texts, collection=resolved_collection
                )
                sparse_pairs = [(s.indices, s.values) for s in sparse_vecs]
            else:
                sparse_pairs = [([], []) for _ in texts]

            chunk_payloads = [
                ChunkPayload(
                    text=chunk["text"],
                    embedding=emb,
                    metadata={k: v for k, v in chunk.items() if k != "text"},
                    sparse_indices=sparse_idx,
                    sparse_values=sparse_val,
                )
                for chunk, emb, (sparse_idx, sparse_val) in zip(
                    all_chunks, embeddings, sparse_pairs, strict=False
                )
            ]

            total_chunks = await self._qdrant.upsert_chunks(chunk_payloads, collection)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "ingest_complete",
            urls=len(ingested_urls),
            chunks=total_chunks,
            replace=request.replace,
            elapsed_ms=round(elapsed_ms, 1),
        )

        return IngestResponse(
            ingested_urls=ingested_urls,
            failed_urls=[{"url": f["url"], "error": f["error"]} for f in failures],
            total_chunks=total_chunks,
            collection=collection or self._qdrant._default_collection,
            elapsed_ms=elapsed_ms,
        )

    # ==================================================================
    # Retrieve
    # ==================================================================
    async def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        start = time.monotonic()

        resolved_collection = request.collection or self._qdrant._default_collection
        collection_is_hybrid = self._qdrant.is_hybrid(resolved_collection)

        # ---- 1. Query expansion --------------------------------------
        variants = await self._build_query_variants(request)

        # ---- 2. Dense searches (one per variant), in parallel --------
        # Embed all variants in a single batched call (the embedding service
        # already handles per-text caching) so we pay one model invocation
        # for the whole expansion set.
        prefixed = [f"query: {v}" for v in variants]
        variant_vectors = await self._embeddings.embed_texts(prefixed)

        per_variant_top_k = self._overfetch_top_k(request.top_k, len(variants))
        dense_searches = [
            self._qdrant.search(
                query_vector=vec,
                top_k=per_variant_top_k,
                collection=request.collection,
                filters=request.filters,
                score_threshold=request.score_threshold,
                with_vectors=request.use_mmr,  # only fetch vectors if MMR will use them
            )
            for vec in variant_vectors
        ]

        # ---- 3. Sparse (BM25) search --------------------------------
        sparse_task = None
        if (
            request.use_hybrid
            and self._sparse is not None
            and collection_is_hybrid
        ):
            sparse_task = asyncio.create_task(
                self._sparse_search(
                    query=request.query,
                    collection=resolved_collection,
                    top_k=per_variant_top_k,
                    filters=request.filters,
                    with_vectors=request.use_mmr,
                )
            )

        dense_results_per_variant = await asyncio.gather(*dense_searches)
        sparse_results = await sparse_task if sparse_task else []

        # ---- 4. RRF fusion across all ranked lists -------------------
        ranked_lists: list[list[Hashable]] = [
            [p.id for p in points] for points in dense_results_per_variant
        ]
        if sparse_results:
            ranked_lists.append([p.id for p in sparse_results])

        # Weight the original query slightly higher than expansions, and
        # match dense:sparse weighting per the BM25+dense literature where
        # dense gets the larger share on semantic queries.
        weights = self._fusion_weights(
            num_variants=len(variants),
            has_sparse=bool(sparse_results),
        )
        fused = reciprocal_rank_fusion(
            ranked_lists, k=settings.retrieval_rrf_k, weights=weights
        )

        # ---- 5. Materialize chunks from points -----------------------
        # We need the full payload (+ optionally the dense vector for MMR).
        # Collect points by id from all the searches, deduping.
        point_by_id: dict[Hashable, Any] = {}
        for points in dense_results_per_variant:
            for p in points:
                point_by_id.setdefault(p.id, p)
        for p in sparse_results:
            point_by_id.setdefault(p.id, p)

        chunks: list[RetrievedChunk] = []
        # Preserve RRF order, cap at request.top_k for the reranker input.
        for pid, fused_score in fused[: request.top_k]:
            point = point_by_id.get(pid)
            if point is None:
                continue
            chunk = self._point_to_chunk(point, fused_score)
            chunks.append(chunk)

        total_retrieved = len(chunks)

        # ---- 6. Cross-encoder rerank --------------------------------
        # Pull more candidates than the requested rerank_top_k when MMR is on,
        # so MMR has room to swap out duplicates without losing relevance.
        rerank_input_k = request.rerank_top_k
        if request.use_mmr:
            rerank_input_k = max(request.rerank_top_k * 3, request.rerank_top_k)
        reranked = await self._reranker.rerank(request.query, chunks, rerank_input_k)

        # ---- 7. No-hallucination relevance gate ---------------------
        # Record the best score we saw BEFORE filtering, so the caller can
        # adapt the threshold if all candidates fell just under.
        top_relevance = self._top_rerank_score(reranked)

        # Drop chunks below the threshold. We do this before MMR so the
        # diversity step never has to "pick the least bad" weak chunk.
        if request.min_relevance > 0.0:
            kept = [
                c for c in reranked
                if (c.rerank_score if c.rerank_score is not None else c.score)
                >= request.min_relevance
            ]
            dropped = len(reranked) - len(kept)
            reranked = kept
        else:
            dropped = 0

        # If nothing survives, refuse outright. We return an empty chunk
        # list + answer_context=None + refused=True so a downstream LLM has
        # literally nothing to hallucinate from.
        if not reranked:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.info(
                "retrieve_refused",
                query=request.query,
                top_relevance=top_relevance,
                threshold=request.min_relevance,
                candidates_considered=total_retrieved,
                elapsed_ms=round(elapsed_ms, 1),
            )
            reason = self._refusal_reason(
                top_relevance=top_relevance,
                threshold=request.min_relevance,
                had_candidates=total_retrieved > 0,
            )
            return RetrieveResponse(
                query=request.query,
                answer_context=None,
                citations=[],
                chunks=[],
                total_retrieved=total_retrieved,
                total_reranked=0,
                elapsed_ms=elapsed_ms,
                refused=True,
                refusal_reason=reason,
                top_relevance=top_relevance,
            )

        # ---- 8. MMR diversity (optional) ----------------------------
        if request.use_mmr and reranked:
            reranked = self._apply_mmr(
                query_vector=variant_vectors[0] if variant_vectors else None,
                point_by_id=point_by_id,
                reranked=reranked,
                top_k=request.rerank_top_k,
                lambda_=request.mmr_lambda,
            )

        # ---- 9. Citation context pack -------------------------------
        citations, answer_context = self._build_citations(reranked, request.include_context)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "retrieve_complete",
            query=request.query,
            variants=len(variants),
            hybrid=bool(sparse_results),
            mmr=request.use_mmr,
            retrieved=total_retrieved,
            reranked=len(reranked),
            dropped_below_threshold=dropped,
            top_relevance=top_relevance,
            elapsed_ms=round(elapsed_ms, 1),
        )

        return RetrieveResponse(
            query=request.query,
            answer_context=answer_context,
            citations=citations,
            chunks=reranked,
            total_retrieved=total_retrieved,
            total_reranked=len(reranked),
            elapsed_ms=elapsed_ms,
            refused=False,
            refusal_reason=None,
            top_relevance=top_relevance,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    async def _build_query_variants(self, request: RetrieveRequest) -> list[str]:
        """Generate the set of queries to actually retrieve with. Always
        includes the user's original query first so RRF weighting keeps it
        as the strongest signal."""
        if request.num_queries <= 1:
            return [request.query]
        variants = await self._expander.expand_for_retrieval(
            query=request.query,
            num_queries=request.num_queries,
            use_hyde=request.use_hyde,
        )
        # Defensive: never come back empty.
        return variants or [request.query]

    def _overfetch_top_k(self, top_k: int, num_variants: int) -> int:
        if num_variants <= 1:
            return top_k
        # When we have multiple variants, each searches for a slightly larger
        # candidate set so RRF can find consensus picks even when individual
        # variants disagree on the tail.
        return max(top_k, int(top_k * settings.retrieval_multi_query_overfetch))

    def _fusion_weights(self, num_variants: int, has_sparse: bool) -> list[float]:
        """Per-list weight vector for RRF.

        The user's original query gets the highest dense weight; expansion
        variants get a lower weight (otherwise a noisy HyDE rewrite can
        outvote the real query). Sparse gets ~70% of the original-query
        weight — strong enough to recover exact-match hits, light enough
        that semantically-correct dense matches still win when the lexical
        overlap is weak.
        """
        weights: list[float] = []
        for i in range(num_variants):
            weights.append(1.0 if i == 0 else 0.6)
        if has_sparse:
            weights.append(0.7)
        return weights

    async def _sparse_search(
        self,
        query: str,
        collection: str,
        top_k: int,
        filters: dict[str, Any] | None,
        with_vectors: bool,
    ) -> list[Any]:
        assert self._sparse is not None
        sparse_vec = await self._sparse.encode_query(query, collection=collection)
        if not sparse_vec.indices:
            return []
        return await self._qdrant.search_sparse(
            indices=sparse_vec.indices,
            values=sparse_vec.values,
            top_k=top_k,
            collection=collection,
            filters=filters,
            with_vectors=with_vectors,
        )

    @staticmethod
    def _top_rerank_score(reranked: list[RetrievedChunk]) -> float | None:
        """Highest normalized rerank score in the list, or None if empty.
        Used both for telemetry and for the refusal-reason message so the
        caller can see *how close* it got to the threshold."""
        if not reranked:
            return None
        top = reranked[0]
        score = top.rerank_score if top.rerank_score is not None else top.score
        return float(score) if score is not None else None

    @staticmethod
    def _refusal_reason(
        top_relevance: float | None,
        threshold: float,
        had_candidates: bool,
    ) -> str:
        """Compose a structured, machine-parseable refusal reason.

        Two distinct failure modes:

        * ``no_candidates`` — Qdrant returned nothing for any variant. The
          collection is empty, or filters / score thresholds excluded
          everything. Nothing to rerank, nothing to gate.
        * ``below_threshold`` — There were candidates, but every one of
          them came in below ``min_relevance`` after reranking. The corpus
          contains topically-related material but nothing that the
          cross-encoder considers an actual answer.
        """
        if not had_candidates:
            return (
                "no_candidates: the collection returned no results for this "
                "query (empty collection, or filters excluded everything)"
            )
        score_str = (
            f"{top_relevance:.3f}" if top_relevance is not None else "n/a"
        )
        return (
            f"below_threshold: top rerank score {score_str} is under the "
            f"min_relevance gate of {threshold:.3f} — the corpus does not "
            f"contain sufficiently relevant information to answer this query"
        )

    def _point_to_chunk(self, point: Any, fused_score: float) -> RetrievedChunk:
        # ``point.payload`` is shared across searches, so don't mutate it.
        payload = dict(point.payload or {})
        text = payload.pop("text", "")
        payload.pop("_fp", None)  # internal content fingerprint, not user-facing
        metadata = ChunkMetadata(
            url=payload.pop("url", ""),
            title=payload.pop("title", None),
            chunk_index=payload.pop("chunk_index", 0),
            total_chunks=payload.pop("total_chunks", 1),
            word_count=payload.pop("word_count", 0),
            extra=payload,
        )
        return RetrievedChunk(text=text, score=float(fused_score), metadata=metadata)

    def _apply_mmr(
        self,
        query_vector: list[float] | None,
        point_by_id: dict[Hashable, Any],
        reranked: list[RetrievedChunk],
        top_k: int,
        lambda_: float,
    ) -> list[RetrievedChunk]:
        """Reorder ``reranked`` to maximize MMR, then truncate to ``top_k``.

        We look up each chunk's dense vector from the originating point
        (fetched with ``with_vectors=True`` upstream). Chunks without a
        vector get a zero similarity contribution — they are still ranked
        but contribute nothing to the diversity penalty.
        """
        if not query_vector:
            return reranked[:top_k]

        chunk_index_by_url_idx: dict[tuple[str, int], int] = {}
        for i, chunk in enumerate(reranked):
            chunk_index_by_url_idx[(chunk.metadata.url, chunk.metadata.chunk_index)] = i

        # Build (id, vector) candidates aligned with the reranked list.
        candidates: list[tuple[int, list[float]]] = []
        for i, chunk in enumerate(reranked):
            vec: list[float] = []
            # Find the matching point by URL + chunk_index. The point id is
            # deterministic, but we don't recompute it here — we just scan,
            # because reranked is small (top tens of items).
            for point in point_by_id.values():
                payload = point.payload or {}
                if (
                    payload.get("url") == chunk.metadata.url
                    and payload.get("chunk_index") == chunk.metadata.chunk_index
                ):
                    raw_vec = self._extract_dense_vector(point)
                    if raw_vec:
                        vec = raw_vec
                    break
            candidates.append((i, vec))

        # If no vectors made it through, MMR is a no-op.
        if not any(vec for _, vec in candidates):
            return reranked[:top_k]

        selected_indices = maximal_marginal_relevance(
            query_vector=query_vector,
            candidates=candidates,
            top_k=top_k,
            lambda_=lambda_,
        )
        return [reranked[i] for i in selected_indices]

    @staticmethod
    def _extract_dense_vector(point: Any) -> list[float]:
        """Pull the dense vector out of a ScoredPoint regardless of layout
        (named vs unnamed). Hybrid collections store it under the ``dense``
        key; legacy collections use a bare list."""
        from app.storage.qdrant_client import DENSE_VECTOR_NAME

        vec = getattr(point, "vector", None)
        if vec is None:
            return []
        if isinstance(vec, dict):
            return list(vec.get(DENSE_VECTOR_NAME, []) or [])
        if isinstance(vec, list):
            return list(vec)
        return []

    def _build_citations(
        self, reranked: list[RetrievedChunk], include_context: bool
    ) -> tuple[list[Citation], str | None]:
        citations: list[Citation] = []
        context_parts: list[str] = []
        for i, chunk in enumerate(reranked, 1):
            source = chunk.metadata.url or "unknown"
            citations.append(
                Citation(
                    index=i,
                    source=source,
                    title=chunk.metadata.title,
                    relevance=round(
                        chunk.rerank_score
                        if chunk.rerank_score is not None
                        else chunk.score,
                        4,
                    ),
                )
            )
            context_parts.append(f"[{i}] (source: {source})\n{chunk.text}")
        answer_context = (
            "\n\n".join(context_parts) if (include_context and reranked) else None
        )
        return citations, answer_context

    # ==================================================================
    # Convenience: search → ingest
    # ==================================================================
    async def search_and_ingest(
        self,
        query: str,
        num_results: int = 10,
        collection: str | None = None,
    ) -> IngestResponse:
        search_request = SearchRequest(query=query, num_results=num_results)
        search_response = await self._search.search(search_request)
        urls = [r.url for r in search_response.results if r.url]

        ingest_request = IngestRequest(urls=urls, collection=collection)  # type: ignore[arg-type]
        return await self.ingest(ingest_request)
