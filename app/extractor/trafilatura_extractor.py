import itertools

from app.core.logging import get_logger
from app.models.responses import ExtractResult
from app.utils.concurrency import run_in_executor
from app.utils.text_utils import count_words

logger = get_logger(__name__)

# Trafilatura keeps internal LRU caches that grow unbounded in long-running
# processes. Reset them every N extractions to keep memory flat.
_RESET_CACHE_EVERY = 200
_extract_counter = itertools.count(1)


def _extract_sync(html: str, url: str | None, output_format: str) -> tuple[str, dict] | None:
    import trafilatura

    if next(_extract_counter) % _RESET_CACHE_EVERY == 0:
        try:
            trafilatura.reset_caches()
        except Exception:
            pass

    text = trafilatura.extract(
        html,
        output_format=output_format,
        with_metadata=True,
        favor_recall=True,
        include_links=False,
        include_images=False,
        no_fallback=False,
        url=url,
    )
    if not text:
        return None

    meta = trafilatura.extract_metadata(html, default_url=url)
    metadata: dict = {}
    if meta:
        metadata = {
            "title": meta.title,
            "author": meta.author,
            "date": meta.date,
            "language": meta.language,
        }

    return text, metadata


class TrafilaturaExtractor:
    async def extract(
        self,
        html: str,
        url: str | None = None,
        output_format: str = "markdown",
    ) -> ExtractResult | None:
        result = await run_in_executor(_extract_sync, html, url, output_format)
        if result is None:
            logger.debug("trafilatura_no_content", url=url)
            return None

        text, metadata = result
        return ExtractResult(
            url=url,
            title=metadata.get("title"),
            text=text,
            author=metadata.get("author"),
            date=metadata.get("date"),
            language=metadata.get("language"),
            word_count=count_words(text),
        )
