from typing import Any

from app.core.logging import get_logger
from app.extractor.bs4_extractor import BS4Extractor
from app.extractor.chunker import SemanticChunker
from app.extractor.markdown_cleaner import MarkdownCleaner
from app.extractor.trafilatura_extractor import TrafilaturaExtractor

logger = get_logger(__name__)


class ExtractionPipeline:
    def __init__(
        self,
        trafilatura: TrafilaturaExtractor,
        bs4: BS4Extractor,
        cleaner: MarkdownCleaner,
    ) -> None:
        self._trafilatura = trafilatura
        self._bs4 = bs4
        self._cleaner = cleaner

    async def extract_and_chunk(
        self,
        html: str,
        url: str | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        output_format: str = "markdown",
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        # Try Trafilatura first, fall back to BS4
        result = await self._trafilatura.extract(html, url, output_format)
        if result is None or not result.text.strip():
            logger.debug("bs4_fallback", url=url)
            result = await self._bs4.extract(html, url)

        if not result or not result.text.strip():
            logger.warning("extraction_empty", url=url)
            return []

        cleaned = self._cleaner.clean(result.text)
        if not cleaned.strip():
            return []

        metadata: dict[str, Any] = {
            "url": url or "",
            "title": result.title,
            "author": result.author,
            "date": result.date,
            "language": result.language,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        chunker = SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk(cleaned, metadata)

        logger.info("extraction_complete", url=url, chunks=len(chunks))
        return chunks
