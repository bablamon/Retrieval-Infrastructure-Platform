from urllib.parse import urljoin, urlparse

from app.core.logging import get_logger
from app.models.responses import ExtractResult
from app.utils.concurrency import run_in_executor
from app.utils.text_utils import count_words
from app.utils.url_utils import is_same_domain

logger = get_logger(__name__)

_NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]


def _extract_sync(html: str, url: str | None) -> ExtractResult:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")

    # Remove noise
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else None

    # Prefer <main> > <article> > <body>
    content_tag = soup.find("main") or soup.find("article") or soup.body
    if content_tag is None:
        content_tag = soup

    text = content_tag.get_text(separator="\n", strip=True)

    return ExtractResult(
        url=url,
        title=title,
        text=text,
        author=None,
        date=None,
        language=None,
        word_count=count_words(text),
    )


def _extract_links_sync(html: str, base_url: str) -> list[str]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    links: list[str] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        full = urljoin(base_url, href)
        # Normalize: strip fragment
        parsed = urlparse(full)
        clean = parsed._replace(fragment="").geturl()
        if clean not in seen and is_same_domain(clean, base_url):
            seen.add(clean)
            links.append(clean)

    return links


class BS4Extractor:
    async def extract(self, html: str, url: str | None = None) -> ExtractResult:
        return await run_in_executor(_extract_sync, html, url)

    async def extract_links(self, html: str, base_url: str) -> list[str]:
        return await run_in_executor(_extract_links_sync, html, base_url)
