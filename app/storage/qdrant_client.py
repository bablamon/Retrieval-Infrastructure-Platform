import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    ScoredPoint,
    VectorParams,
)

_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # UUID namespace URL


@dataclass
class ChunkPayload:
    text: str
    embedding: list[float]
    metadata: dict[str, Any]


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
        name = self._resolve_collection(collection)
        client = self._client_or_raise()
        existing = await client.get_collections()
        names = [c.name for c in existing.collections]
        if name not in names:
            await client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
            )

    async def upsert_chunks(
        self,
        chunks: list[ChunkPayload],
        collection: str | None = None,
    ) -> int:
        if not chunks:
            return 0

        name = self._resolve_collection(collection)
        client = self._client_or_raise()

        batch_size = 100
        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            points = [
                PointStruct(
                    id=str(
                        uuid.uuid5(
                            _NAMESPACE,
                            f"{chunk.metadata.get('url', '')}:{chunk.metadata.get('chunk_index', i + j)}",
                        )
                    ),
                    vector=chunk.embedding,
                    payload={**chunk.metadata, "text": chunk.text},
                )
                for j, chunk in enumerate(batch)
            ]
            await client.upsert(collection_name=name, points=points)
            total += len(batch)

        return total

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        collection: str | None = None,
        filters: dict[str, Any] | None = None,
        score_threshold: float = 0.0,
    ) -> list[ScoredPoint]:
        name = self._resolve_collection(collection)
        client = self._client_or_raise()

        qdrant_filter: Filter | None = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        # query_points is the stable API; search() is deprecated in qdrant-client.
        response = await client.query_points(
            collection_name=name,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            score_threshold=score_threshold if score_threshold > 0.0 else None,
            with_payload=True,
        )
        return response.points

    async def delete_by_url(self, url: str, collection: str | None = None) -> None:
        name = self._resolve_collection(collection)
        client = self._client_or_raise()
        delete_filter = Filter(
            must=[FieldCondition(key="url", match=MatchValue(value=url))]
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
