import hashlib
import time

from app.models.requests import SearchRequest
from app.models.responses import SearchResponse
from app.search.searxng_client import SearXNGClient
from app.storage.redis_client import RedisCache


class CachedSearchService:
    def __init__(
        self,
        searxng: SearXNGClient,
        cache: RedisCache,
        cache_ttl: int = 900,
    ) -> None:
        self._searxng = searxng
        self._cache = cache
        self._cache_ttl = cache_ttl

    def _cache_key(self, request: SearchRequest) -> str:
        engines_str = ",".join(sorted(request.engines))
        raw = f"{request.query}:{request.num_results}:{engines_str}:{request.safe_search}"
        return hashlib.sha256(raw.encode()).hexdigest()

    async def search(self, request: SearchRequest) -> SearchResponse:
        start = time.monotonic()

        if request.use_cache:
            query_hash = self._cache_key(request)
            cached = await self._cache.get_cached_search(query_hash)
            if cached is not None:
                return SearchResponse(
                    query=cached.query,
                    results=cached.results,
                    total=cached.total,
                    cached=True,
                    elapsed_ms=(time.monotonic() - start) * 1000,
                )

        results = await self._searxng.search(
            query=request.query,
            num_results=request.num_results,
            engines=request.engines or None,
            safe_search=request.safe_search,
        )

        elapsed_ms = (time.monotonic() - start) * 1000
        response = SearchResponse(
            query=request.query,
            results=results,
            total=len(results),
            cached=False,
            elapsed_ms=elapsed_ms,
        )

        if request.use_cache:
            await self._cache.set_cached_search(query_hash, response, self._cache_ttl)

        return response
