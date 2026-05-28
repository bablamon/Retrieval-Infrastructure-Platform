"""Qdrant client wrapper.

Supports two collection layouts side-by-side:

* **Dense-only** (legacy): a single unnamed dense vector per point. Kept for
  backwards compatibility with collections created before hybrid retrieval.
* **Hybrid** (default for new collections): a *named* dense vector plus a
  named BM25 *sparse* vector per point. The pipeline searches both and
  fuses results via reciprocal rank fusion in ``app.retrieval.pipeline``.

The wrapper auto-detects which layout a collection uses on first access
and uses the appropriate query / upsert API. New collections are created
hybrid; old collections keep working unchanged.

Idempotent ingest: ``upsert_chunks`` derives point IDs from
``(url, chunk_index)`` so re-ingesting a URL overwrites in-place. For full
"replace" semantics (handle the case where a re-crawl produces *fewer*
chunks), the retrieval pipeline calls ``delete_by_urls`` before upsert.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PointStruct,
    ScoredPoint,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # UUID namespace URL

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"


@dataclass
class ChunkPayload:
    text: str
    embedding: list[float]
    metadata: dict[str, Any]
    # Sparse vector (parallel arrays of u32 indices + float weights). Empty
    # by default so callers that don't compute BM25 still work.
    sparse_indices: list[int] = field(default_factory=list)
    sparse_values: list[float] = field(default_factory=list)


def _point_id(url: str, chunk_index: int) -> str:
    """Deterministic UUIDv5 derived from (url, chunk_index). Re-ingesting
    the same URL therefore upserts in place — see module docstring."""
    return str(uuid.uuid5(_NAMESPACE, f"{url}:{chunk_index}"))


def _text_fingerprint(text: str) -> str:
    """Short stable hash of chunk text. Stored on the payload so callers can
    detect content changes without re-fetching the original document."""
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


class QdrantStore:
    def __init__(
        self,
        url: str,
        api_key: str,
        collection: str,
        vector_size: int,
    ) -> None:
        self._url = url
        self._api_key = api_key or None
        self._default_collection = collection
        self._vector_size = vector_size
        self._client: AsyncQdrantClient | None = None
        # Cache layout (hybrid / legacy) per collection — avoids a
        # ``get_collection`` call on every search/upsert.
        self._is_hybrid: dict[str, bool] = {}
        self._known_collections: set[str] = set()

    async def connect(self) -> None:
        self._client = AsyncQdrantClient(url=self._url, api_key=self._api_key)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    def _client_or_raise(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RuntimeError("QdrantStore not connected — call connect() first")
        return self._client

    def _resolve_collection(self, collection: str | None) -> str:
        return collection or self._default_collection

    async def ensure_collection(self, collection: str | None = None) -> None:
        """Create the collection if missing. New collections are always
        created in hybrid layout; existing legacy collections are detected
        and used as-is."""
        name = self._resolve_collection(collection)
        if name in self._known_collections:
            return
        client = self._client_or_raise()

        existing = await client.get_collections()
        existing_names = [c.name for c in existing.collections]

        if name not in existing_names:
            await client.create_collection(
                collection_name=name,
                vectors_config={
                    DENSE_VECTOR_NAME: VectorParams(
                        size=self._vector_size, distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: SparseVectorParams(index=SparseIndexParams())
                },
            )
            self._is_hybrid[name] = True
        else:
            # Detect layout by inspecting the existing config. Legacy
            # collections have an unnamed dense vector (the config exposes
            # a single VectorParams instead of a dict-of-named-vectors).
            info = await client.get_collection(collection_name=name)
            vec_cfg = info.config.params.vectors
            sparse_cfg = getattr(info.config.params, "sparse_vectors", None)
            is_named_dense = isinstance(vec_cfg, dict) and DENSE_VECTOR_NAME in vec_cfg
            has_sparse = bool(sparse_cfg and SPARSE_VECTOR_NAME in (sparse_cfg or {}))
            self._is_hybrid[name] = is_named_dense and has_sparse

        self._known_collections.add(name)

    def is_hybrid(self, collection: str | None = None) -> bool:
        """True iff the collection was created with the named dense + sparse
        layout. Surface this on the pipeline so callers don't issue sparse
        searches against legacy dense-only collections."""
        name = self._resolve_collection(collection)
        return self._is_hybrid.get(name, False)

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------
    async def upsert_chunks(
        self,
        chunks: list[ChunkPayload],
        collection: str | None = None,
    ) -> int:
        if not chunks:
            return 0

        name = self._resolve_collection(collection)
        await self.ensure_collection(name)
        client = self._client_or_raise()
        hybrid = self.is_hybrid(name)

        batch_size = 100
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            points: list[PointStruct] = []
            for j, chunk in enumerate(batch):
                url = chunk.metadata.get("url", "")
                chunk_index = chunk.metadata.get("chunk_index", i + j)

                payload = {
                    **chunk.metadata,
                    "text": chunk.text,
                    "_fp": _text_fingerprint(chunk.text),
                }

                if hybrid:
                    vector: Any = {
                        DENSE_VECTOR_NAME: chunk.embedding,
                    }
                    if chunk.sparse_indices and chunk.sparse_values:
                        vector[SPARSE_VECTOR_NAME] = SparseVector(
                            indices=chunk.sparse_indices,
                            values=chunk.sparse_values,
                        )
                else:
                    vector = chunk.embedding

                points.append(
                    PointStruct(
                        id=_point_id(url, chunk_index),
                        vector=vector,
                        payload=payload,
                    )
                )
            await client.upsert(collection_name=name, points=points)
            total += len(batch)

        return total

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    async def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        collection: str | None = None,
        filters: dict[str, Any] | None = None,
        score_threshold: float = 0.0,
        with_vectors: bool = False,
    ) -> list[ScoredPoint]:
        name = self._resolve_collection(collection)
        await self.ensure_collection(name)
        client = self._client_or_raise()
        hybrid = self.is_hybrid(name)

        qdrant_filter = self._build_filter(filters)

        # qdrant-client's ``query_points`` dispatches the ``query`` arg by
        # type: a bare list is a dense vector, a ``SparseVector`` is a
        # sparse query, etc. For named vectors we pass the raw vector and
        # specify the name via ``using=`` — wrapping in NamedVector is the
        # old multi-query API and isn't accepted as a ``query`` value here.
        response = await client.query_points(
            collection_name=name,
            query=query_vector,
            using=DENSE_VECTOR_NAME if hybrid else None,
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold if score_threshold > 0.0 else None,
            with_payload=True,
            with_vectors=with_vectors,
        )
        return response.points

    async def search_sparse(
        self,
        indices: list[int],
        values: list[float],
        top_k: int = 20,
        collection: str | None = None,
        filters: dict[str, Any] | None = None,
        with_vectors: bool = False,
    ) -> list[ScoredPoint]:
        """BM25 lexical search against the named sparse vector. Returns []
        on legacy (non-hybrid) collections so the hybrid pipeline degrades
        cleanly when pointed at old data."""
        if not indices:
            return []
        name = self._resolve_collection(collection)
        await self.ensure_collection(name)
        if not self.is_hybrid(name):
            return []

        client = self._client_or_raise()
        qdrant_filter = self._build_filter(filters)

        # Same convention as the dense path: pass the raw ``SparseVector``
        # as ``query`` and name the target sparse index via ``using=``.
        response = await client.query_points(
            collection_name=name,
            query=SparseVector(indices=indices, values=values),
            using=SPARSE_VECTOR_NAME,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
            with_vectors=with_vectors,
        )
        return response.points

    def _build_filter(self, filters: dict[str, Any] | None) -> Filter | None:
        if not filters:
            return None
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()
        ]
        return Filter(must=conditions)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    async def delete_by_url(self, url: str, collection: str | None = None) -> None:
        await self.delete_by_urls([url], collection=collection)

    async def delete_by_urls(
        self, urls: list[str], collection: str | None = None
    ) -> None:
        """Bulk delete all points whose payload.url matches any of ``urls``.

        Used by idempotent ingest: we wipe stale chunks for a URL *before*
        upserting the new ones, so re-ingesting a page that now has fewer
        chunks doesn't leave orphans behind. The previous upsert-by-id
        strategy alone was insufficient because chunk_index N+1 from an
        old crawl would survive a new crawl that produced only N chunks.
        """
        if not urls:
            return
        name = self._resolve_collection(collection)
        await self.ensure_collection(name)
        client = self._client_or_raise()

        delete_filter = Filter(
            must=[FieldCondition(key="url", match=MatchAny(any=list(urls)))]
        )
        try:
            await client.delete(
                collection_name=name,
                points_selector=FilterSelector(filter=delete_filter),
            )
        except Exception:
            # Fallback for older qdrant-client versions
            await client.delete(
                collection_name=name,
                points_selector=delete_filter,  # type: ignore[arg-type]
            )

    async def health_check(self) -> bool:
        try:
            await self._client_or_raise().get_collections()
            return True
        except Exception:
            return False
