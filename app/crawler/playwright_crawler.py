import asyncio
import time

from app.core.exceptions import CrawlError
from app.core.logging import get_logger
from app.models.responses import CrawlResult

logger = get_logger(__name__)

_USER_AGENT = "RetrievalBot/1.0 (open-source retrieval platform)"
_LAUNCH_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--disable-setuid-sandbox",
    "--no-first-run",
    "--no-zygote",
]


class PlaywrightCrawler:
    """JS-rendering crawler. The Chromium browser is launched lazily on first use
    so instances that never crawl JS pages don't hold ~300MB of browser memory."""

    def __init__(self, semaphore: asyncio.Semaphore, timeout: int = 30000) -> None:
        self._semaphore = semaphore
        self._timeout = timeout
        self._playwright = None
        self._playwright_ctx = None
        self._browser = None
        self._launch_lock = asyncio.Lock()

    async def start(self) -> None:
        # No-op: browser is launched lazily in _ensure_browser(). Kept for
        # lifecycle symmetry with the other services.
        return None

    async def _ensure_browser(self):
        if self._browser is not None:
            return self._browser
        async with self._launch_lock:
            if self._browser is None:
                from playwright.async_api import async_playwright

                self._playwright_ctx = async_playwright()
                self._playwright = await self._playwright_ctx.__aenter__()
                self._browser = await self._playwright.chromium.launch(
                    headless=True,
                    args=_LAUNCH_ARGS,
                )
                logger.info("playwright_launched")
        return self._browser

    async def stop(self) -> None:
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright_ctx:
            await self._playwright_ctx.__aexit__(None, None, None)
            self._playwright = None
            self._playwright_ctx = None
        logger.info("playwright_stopped")

    async def fetch(self, url: str) -> CrawlResult:
        browser = await self._ensure_browser()

        async with self._semaphore:
            start = time.monotonic()
            page = None
            try:
                page = await browser.new_page()
                await page.set_extra_http_headers({"User-Agent": _USER_AGENT})

                response = await page.goto(
                    url,
                    wait_until="networkidle",
                    timeout=self._timeout,
                )
                status_code = response.status if response else 0
                html = await page.content()
                final_url = page.url
                elapsed_ms = (time.monotonic() - start) * 1000

                logger.info(
                    "playwright_crawl",
                    url=url,
                    status=status_code,
                    elapsed_ms=round(elapsed_ms, 1),
                )
                return CrawlResult(
                    url=url,
                    status_code=status_code,
                    html=html,
                    final_url=final_url,
                    content_type="text/html",
                    elapsed_ms=elapsed_ms,
                    used_playwright=True,
                    cached=False,
                )
            except Exception as exc:
                raise CrawlError(f"Playwright crawl failed for {url}: {exc}") from exc
            finally:
                if page:
                    await page.close()
