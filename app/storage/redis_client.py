import hashlib

import orjson
import redis.asyncio as aioredis

from app.models.responses import CrawlResult, SearchResponse
from app.utils.url_utils import normalize_url


class RedisCache:
    def __init__(self, redis_url: str, default_ttl: int = 3600) -> None:
        self._url = redis_url
        self._default_ttl = default_ttl
        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        self._client = aioredis.from_url(self._url, decode_responses=False)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _client_or_raise(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("RedisCache not connected — call connect() first")
        return self._client

    async def get(self, key: str) -> bytes | None:
        return await self._client_or_raise().get(key)

    async def set(self, key: str, value: bytes, ttl: int | None = None) -> None:
        await self._client_or_raise().set(key, value, ex=ttl or self._default_ttl)

    async def delete(self, key: str) -> None:
        await self._client_or_raise().delete(key)

    async def exists(self, key: str) -> bool:
        return bool(await self._client_or_raise().exists(key))

    async def sadd(self, key: str, *values: str) -> int:
        return await self._client_or_raise().sadd(key, *values)

    async def sismember(self, key: str, value: str) -> bool:
        return bool(await self._client_or_raise().sismember(key, value))

    async def get_json(self, key: str) -> dict | None:
        raw = await self.get(key)
        if raw is None:
            return None
        return orjson.loads(raw)

    async def set_json(self, key: str, value: dict, ttl: int | None = None) -> None:
        await self.set(key, orjson.dumps(value), ttl)

    def _page_key(self, url: str) -> str:
        return f"page:{hashlib.sha256(normalize_url(url).encode()).hexdigest()}"

    def _search_key(self, query_hash: str) -> str:
        return f"search:{query_hash}"

    def _embed_key(self, text_hash: str) -> str:
        return f"embed:{text_hash}"

    def _robots_key(self, netloc: str) -> str:
        return f"robots:{netloc}"

    async def get_cached_page(self, url: str) -> CrawlResult | None:
        data = await self.get_json(self._page_key(url))
        if data is None:
            return None
        return CrawlResult(**data)

    async def set_cached_page(self, url: str, result: CrawlResult, ttl: int) -> None:
        await self.set_json(self._page_key(url), result.model_dump(), ttl)

    async def get_cached_search(self, query_hash: str) -> SearchResponse | None:
        data = await self.get_json(self._search_key(query_hash))
        if data is None:
            return None
        return SearchResponse(**data)

    async def set_cached_search(self, query_hash: str, result: SearchResponse, ttl: int) -> None:
        await self.set_json(self._search_key(query_hash), result.model_dump(), ttl)

    async def get_cached_embedding(self, text_hash: str) -> list[float] | None:
        raw = await self.get(self._embed_key(text_hash))
        if raw is None:
            return None
        return orjson.loads(raw)

    async def get_cached_embeddings(self, text_hashes: list[str]) -> list[list[float] | None]:
        """Batch lookup via a single MGET round-trip (avoids N sequential calls)."""
        if not text_hashes:
            return []
        keys = [self._embed_key(h) for h in text_hashes]
        raws = await self._client_or_raise().mget(keys)
        return [orjson.loads(r) if r is not None else None for r in raws]

    async def set_cached_embedding(self, text_hash: str, vector: list[float], ttl: int) -> None:
        await self.set(self._embed_key(text_hash), orjson.dumps(vector), ttl)

    async def set_cached_embeddings(
        self, items: list[tuple[str, list[float]]], ttl: int
    ) -> None:
        """Batch store via a pipeline (single round-trip)."""
        if not items:
            return
        pipe = self._client_or_raise().pipeline(transaction=False)
        for text_hash, vector in items:
            pipe.set(self._embed_key(text_hash), orjson.dumps(vector), ex=ttl)
        await pipe.execute()

    async def get_robots_txt(self, netloc: str) -> str | None:
        raw = await self.get(self._robots_key(netloc))
        if raw is None:
            return None
        return raw.decode("utf-8", errors="replace")

    async def set_robots_txt(self, netloc: str, content: str, ttl: int) -> None:
        await self.set(self._robots_key(netloc), content.encode(), ttl)

    async def is_url_crawled(self, url: str) -> bool:
        return await self.sismember("crawled_urls", normalize_url(url))

    async def mark_url_crawled(self, url: str) -> None:
        await self.sadd("crawled_urls", normalize_url(url))

    async def health_check(self) -> bool:
        try:
            return await self._client_or_raise().ping()
        except Exception:
            return False
