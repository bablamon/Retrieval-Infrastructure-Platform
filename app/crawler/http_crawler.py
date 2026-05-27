import time

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.exceptions import CrawlError
from app.core.logging import get_logger
from app.models.responses import CrawlResult

logger = get_logger(__name__)

_USER_AGENT = "RetrievalBot/1.0 (open-source retrieval platform; +https://github.com/retrieval-platform)"


class HTTPCrawler:
    def __init__(self, timeout: int = 30, max_redirects: int = 5) -> None:
        self._timeout = timeout
        self._max_redirects = max_redirects
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._timeout),
            max_redirects=self._max_redirects,
            headers={"User-Agent": _USER_AGENT},
            follow_redirects=True,
            http2=True,
        )

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _client_or_raise(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("HTTPCrawler not started — call start() first")
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        reraise=True,
    )
    async def fetch(self, url: str) -> CrawlResult:
        start = time.monotonic()
        try:
            response = await self._client_or_raise().get(url)
            elapsed_ms = (time.monotonic() - start) * 1000
            html = response.text if response.content else None
            logger.info(
                "http_crawl",
                url=url,
                status=response.status_code,
                elapsed_ms=round(elapsed_ms, 1),
            )
            return CrawlResult(
                url=url,
                status_code=response.status_code,
                html=html,
                final_url=str(response.url),
                content_type=response.headers.get("content-type", ""),
                elapsed_ms=elapsed_ms,
                used_playwright=False,
                cached=False,
            )
        except (httpx.TransportError, httpx.TimeoutException):
            raise
        except Exception as exc:
            raise CrawlError(f"HTTP crawl failed for {url}: {exc}") from exc
