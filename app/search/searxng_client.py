
import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.exceptions import SearchError
from app.core.logging import get_logger
from app.models.responses import SearchResult

logger = get_logger(__name__)


class SearXNGClient:
    def __init__(self, base_url: str, timeout: int = 30) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout),
            headers={"User-Agent": "RetrievalBot/1.0"},
            follow_redirects=True,
        )

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _client_or_raise(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("SearXNGClient not started — call start() first")
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        reraise=True,
    )
    async def search(
        self,
        query: str,
        num_results: int = 10,
        engines: list[str] | None = None,
        safe_search: int = 1,
    ) -> list[SearchResult]:
        params: dict[str, str | int] = {
            "q": query,
            "format": "json",
            "pageno": 1,
            "safesearch": safe_search,
        }
        if engines:
            params["engines"] = ",".join(engines)

        try:
            response = await self._client_or_raise().get(
                f"{self._base_url}/search",
                params=params,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            raise SearchError(f"SearXNG returned {e.response.status_code}: {e.response.text}") from e
        except Exception as e:
            raise SearchError(f"SearXNG request failed: {e}") from e

        results: list[SearchResult] = []
        for item in data.get("results", [])[:num_results]:
            results.append(
                SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    snippet=item.get("content", ""),
                    engine=item.get("engine", "unknown"),
                    score=item.get("score"),
                )
            )

        logger.info("searxng_search", query=query, count=len(results))
        return results

    async def health_check(self) -> bool:
        try:
            response = await self._client_or_raise().get(
                f"{self._base_url}/search",
                params={"q": "test", "format": "json"},
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception:
            return False
