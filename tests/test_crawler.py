import asyncio

import httpx
import pytest
import respx


@pytest.mark.asyncio
async def test_http_crawler_success():
    from app.crawler.http_crawler import HTTPCrawler

    with respx.mock:
        respx.get("https://example.com/page").mock(
            return_value=httpx.Response(200, text="<html><body><p>Hello</p></body></html>")
        )
        crawler = HTTPCrawler()
        await crawler.start()
        try:
            result = await crawler.fetch("https://example.com/page")
            assert result.status_code == 200
            assert result.html is not None
            assert "Hello" in result.html
            assert result.used_playwright is False
            assert result.cached is False
        finally:
            await crawler.stop()


@pytest.mark.asyncio
async def test_robots_allows_by_default(redis_cache):
    from app.crawler.robots import RobotsCache

    with respx.mock:
        respx.get("https://example.com/robots.txt").mock(
            return_value=httpx.Response(200, text="User-agent: *\nAllow: /\n")
        )
        robots = RobotsCache(redis_cache)
        allowed = await robots.is_allowed("https://example.com/page")
        assert allowed is True


@pytest.mark.asyncio
async def test_robots_disallow(redis_cache):
    from app.crawler.robots import RobotsCache

    with respx.mock:
        respx.get("https://example.com/robots.txt").mock(
            return_value=httpx.Response(
                200, text="User-agent: *\nDisallow: /private/\n"
            )
        )
        robots = RobotsCache(redis_cache)
        blocked = await robots.is_allowed("https://example.com/private/secret")
        assert blocked is False


@pytest.mark.asyncio
async def test_robots_fetch_fail_allows(redis_cache):
    from app.crawler.robots import RobotsCache

    with respx.mock:
        respx.get("https://unreachable.example/robots.txt").mock(
            side_effect=httpx.TransportError("connection refused")
        )
        robots = RobotsCache(redis_cache)
        # Should fail open (allow) when robots.txt is unreachable
        allowed = await robots.is_allowed("https://unreachable.example/page")
        assert allowed is True


@pytest.mark.asyncio
async def test_cache_hit_on_second_crawl(redis_cache, mock_postgres):
    from unittest.mock import AsyncMock, MagicMock

    from app.crawler.async_crawler import AsyncCrawler
    from app.crawler.http_crawler import HTTPCrawler
    from app.crawler.robots import RobotsCache

    with respx.mock:
        respx.get("https://example.com/page").mock(
            return_value=httpx.Response(200, text="<html><body>Content</body></html>")
        )
        respx.get("https://example.com/robots.txt").mock(
            return_value=httpx.Response(200, text="User-agent: *\nAllow: /\n")
        )

        http_crawler = HTTPCrawler()
        await http_crawler.start()

        playwright_mock = MagicMock()
        playwright_mock.fetch = AsyncMock()

        robots = RobotsCache(redis_cache)
        semaphore = asyncio.Semaphore(5)

        crawler = AsyncCrawler(
            http_crawler=http_crawler,
            playwright_crawler=playwright_mock,
            robots_cache=robots,
            redis_cache=redis_cache,
            postgres=mock_postgres,
            crawl_semaphore=semaphore,
        )

        # First crawl
        r1 = await crawler.crawl_one("https://example.com/page")
        assert r1.cached is False

        # Second crawl — should come from cache
        r2 = await crawler.crawl_one("https://example.com/page")
        assert r2.cached is True

        await http_crawler.stop()
