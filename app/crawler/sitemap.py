from urllib.parse import urlparse

import httpx
from lxml import etree

from app.core.logging import get_logger

logger = get_logger(__name__)

_SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"


class SitemapDiscovery:
    def __init__(self, timeout: int = 15) -> None:
        self._timeout = timeout

    async def discover_urls(self, base_url: str, max_urls: int = 500) -> list[str]:
        parsed = urlparse(base_url)
        root = f"{parsed.scheme}://{parsed.netloc}"

        candidates = [
            f"{root}/sitemap.xml",
            f"{root}/sitemap_index.xml",
            f"{root}/sitemap-index.xml",
        ]

        all_urls: list[str] = []
        seen_sitemaps: set[str] = set()

        async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
            for sitemap_url in candidates:
                if len(all_urls) >= max_urls:
                    break
                urls = await self._process_sitemap(client, sitemap_url, seen_sitemaps, depth=0)
                all_urls.extend(urls)

        deduped = list(dict.fromkeys(all_urls))
        return deduped[:max_urls]

    async def _process_sitemap(
        self,
        client: httpx.AsyncClient,
        sitemap_url: str,
        seen: set[str],
        depth: int,
    ) -> list[str]:
        if sitemap_url in seen or depth > 2:
            return []
        seen.add(sitemap_url)

        content = await self._fetch_sitemap(client, sitemap_url)
        if not content:
            return []

        # Check if sitemap index
        index_urls = self._parse_sitemap_index(content)
        if index_urls:
            all_urls: list[str] = []
            for sub_url in index_urls[:20]:
                urls = await self._process_sitemap(client, sub_url, seen, depth + 1)
                all_urls.extend(urls)
            return all_urls

        return self._parse_sitemap_urls(content)

    async def _fetch_sitemap(self, client: httpx.AsyncClient, url: str) -> str | None:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                return response.text
            return None
        except Exception as exc:
            logger.debug("sitemap_fetch_failed", url=url, error=str(exc))
            return None

    def _parse_sitemap_urls(self, xml_content: str) -> list[str]:
        try:
            root = etree.fromstring(xml_content.encode())
            urls: list[str] = []
            for loc in root.iter(f"{{{_SITEMAP_NS}}}loc"):
                if loc.text:
                    urls.append(loc.text.strip())
            return urls
        except Exception:
            return []

    def _parse_sitemap_index(self, xml_content: str) -> list[str]:
        try:
            root = etree.fromstring(xml_content.encode())
            # A sitemap index contains <sitemap> elements
            sitemaps = root.findall(f"{{{_SITEMAP_NS}}}sitemap")
            if not sitemaps:
                return []
            urls: list[str] = []
            for sm in sitemaps:
                loc = sm.find(f"{{{_SITEMAP_NS}}}loc")
                if loc is not None and loc.text:
                    urls.append(loc.text.strip())
            return urls
        except Exception:
            return []
