import asyncio

from app.core.exceptions import CrawlError, URLBlockedError
from app.core.logging import get_logger
from app.crawler.http_crawler import HTTPCrawler
from app.crawler.playwright_crawler import PlaywrightCrawler
from app.crawler.robots import RobotsCache
from app.extractor.bs4_extractor import BS4Extractor
from app.models.responses import CrawlResult
from app.storage.postgres_client import PostgresClient
from app.storage.redis_client import RedisCache
from app.utils.url_utils import deduplicate_urls, normalize_url

logger = get_logger(__name__)

_PLAYWRIGHT_TRIGGER_STATUSES = {403, 406, 429}
_MIN_CONTENT_LENGTH = 500
_MAX_RECURSIVE_PAGES = 100
_MAX_LINKS_PER_PAGE = 25


class AsyncCrawler:
    def __init__(
        self,
        http_crawler: HTTPCrawler,
        playwright_crawler: PlaywrightCrawler,
        robots_cache: RobotsCache,
        redis_cache: RedisCache,
        postgres: PostgresClient,
        crawl_semaphore: asyncio.Semaphore,
        cache_ttl: int = 3600,
    ) -> None:
        self._http = http_crawler
        self._playwright = playwright_crawler
        self._robots = robots_cache
        self._redis = redis_cache
        self._postgres = postgres
        self._semaphore = crawl_semaphore
        self._cache_ttl = cache_ttl

    async def crawl_many(
        self,
        urls: list[str],
        use_playwright: bool = False,
        respect_robots: bool = True,
        use_cache: bool = True,
    ) -> tuple[list[CrawlResult], list[dict[str, str]]]:
        results: list[CrawlResult] = []
        failures: list[dict[str, str]] = []

        async def _task(url: str) -> None:
            async with self._semaphore:
                try:
                    result = await self.crawl_one(url, use_playwright, respect_robots, use_cache)
                    results.append(result)
                except URLBlockedError as exc:
                    logger.info("url_blocked", url=url)
                    failures.append({"url": url, "error": str(exc)})
                except CrawlError as exc:
                    logger.warning("crawl_error", url=url, error=str(exc))
                    failures.append({"url": url, "error": str(exc)})
                except Exception as exc:
                    logger.error("crawl_unexpected", url=url, error=str(exc))
                    failures.append({"url": url, "error": str(exc)})

        async with asyncio.TaskGroup() as tg:
            for url in urls:
                tg.create_task(_task(url))

        return results, failures

    async def crawl_recursive(
        self,
        urls: list[str],
        max_depth: int = 1,
        use_playwright: bool = False,
        respect_robots: bool = True,
        use_cache: bool = True,
        max_pages: int = _MAX_RECURSIVE_PAGES,
    ) -> tuple[list[CrawlResult], list[dict[str, str]]]:
        """Breadth-first recursive crawl. Follows same-domain links up to
        ``max_depth`` levels, bounded by ``max_pages``. With max_depth == 1 this
        is equivalent to crawl_many (single level)."""
        bs4 = BS4Extractor()
        all_results: list[CrawlResult] = []
        all_failures: list[dict[str, str]] = []
        seen: set[str] = set()

        frontier = deduplicate_urls(urls)
        for url in frontier:
            seen.add(normalize_url(url))

        depth = 1
        while frontier and depth <= max_depth and len(all_results) < max_pages:
            budget = max_pages - len(all_results)
            level_results, level_failures = await self.crawl_many(
                frontier[:budget],
                use_playwright=use_playwright,
                respect_robots=respect_robots,
                use_cache=use_cache,
            )
            all_results.extend(level_results)
            all_failures.extend(level_failures)

            if depth == max_depth:
                break

            # Discover next level from the HTML we just fetched
            next_frontier: list[str] = []
            for result in level_results:
                if not result.html:
                    continue
                links = await bs4.extract_links(result.html, result.final_url)
                for link in links[:_MAX_LINKS_PER_PAGE]:
                    norm = normalize_url(link)
                    if norm not in seen:
                        seen.add(norm)
                        next_frontier.append(link)

            frontier = next_frontier
            depth += 1

        return all_results, all_failures

    async def crawl_one(
        self,
        url: str,
        use_playwright: bool = False,
        respect_robots: bool = True,
        use_cache: bool = True,
    ) -> CrawlResult:
        normalized = normalize_url(url)

        # Cache hit
        if use_cache:
            cached = await self._redis.get_cached_page(normalized)
            if cached is not None:
                return cached.model_copy(update={"cached": True})

        # robots.txt check
        if respect_robots:
            allowed = await self._robots.is_allowed(normalized)
            if not allowed:
                raise URLBlockedError(f"robots.txt disallows: {normalized}")

        # Try HTTP first
        result = await self._try_http(normalized, use_playwright)

        # Cache and record
        if use_cache:
            await self._redis.set_cached_page(normalized, result, self._cache_ttl)
        await self._redis.mark_url_crawled(normalized)
        await self._postgres.upsert_crawl_state(
            url=normalized,
            status="crawled",
            http_status=result.status_code,
        )

        return result

    async def _try_http(self, url: str, use_playwright: bool) -> CrawlResult:
        try:
            result = await self._http.fetch(url)
            needs_playwright = (
                use_playwright
                and (
                    result.status_code in _PLAYWRIGHT_TRIGGER_STATUSES
                    or (result.html and len(result.html) < _MIN_CONTENT_LENGTH)
                    or result.status_code == 0
                )
            )
            if needs_playwright:
                logger.info("playwright_fallback", url=url, reason=f"status={result.status_code}")
                return await self._playwright.fetch(url)
            return result
        except CrawlError:
            if use_playwright:
                logger.info("playwright_fallback", url=url, reason="http_error")
                return await self._playwright.fetch(url)
            raise
