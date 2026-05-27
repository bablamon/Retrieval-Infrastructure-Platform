from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.logging import get_logger
from app.storage.redis_client import RedisCache

logger = get_logger(__name__)


class RobotsCache:
    def __init__(
        self,
        cache: RedisCache,
        user_agent: str = "RetrievalBot/1.0",
        timeout: int = 10,
        cache_ttl: int = 86400,
    ) -> None:
        self._cache = cache
        self._user_agent = user_agent
        self._timeout = timeout
        self._cache_ttl = cache_ttl

    async def is_allowed(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            netloc = parsed.netloc.lower()
            scheme = parsed.scheme

            content = await self._cache.get_robots_txt(netloc)
            if content is None:
                content = await self._fetch_robots(scheme, netloc)
                await self._cache.set_robots_txt(netloc, content, self._cache_ttl)

            return self._can_fetch(content, url)
        except Exception as exc:
            logger.warning("robots_check_failed", url=url, error=str(exc))
            return True  # fail open

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException)),
        reraise=False,
    )
    async def _fetch_robots(self, scheme: str, netloc: str) -> str:
        robots_url = f"{scheme}://{netloc}/robots.txt"
        try:
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
                response = await client.get(robots_url)
                if response.status_code == 200:
                    return response.text
                return ""
        except Exception:
            return ""

    def _can_fetch(self, robots_content: str, url: str) -> bool:
        if not robots_content.strip():
            return True

        try:
            parser = RobotFileParser()
            parser.parse(robots_content.splitlines())
            return parser.can_fetch(self._user_agent, url)
        except Exception:
            return True
