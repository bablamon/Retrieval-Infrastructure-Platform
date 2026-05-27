import time

from fastapi import APIRouter, Depends, HTTPException

from app.core.dependencies import get_crawler, get_extraction_pipeline
from app.crawler.async_crawler import AsyncCrawler
from app.extractor.pipeline import ExtractionPipeline
from app.models.requests import ExtractRequest
from app.models.responses import ExtractResponse, ExtractResult

router = APIRouter(prefix="/extract", tags=["extract"])


@router.post("", response_model=ExtractResponse)
async def extract(
    request: ExtractRequest,
    crawler: AsyncCrawler = Depends(get_crawler),
    pipeline: ExtractionPipeline = Depends(get_extraction_pipeline),
) -> ExtractResponse:
    start = time.monotonic()

    html = request.html
    url_str = str(request.url) if request.url else None

    if html is None and url_str is None:
        raise HTTPException(status_code=422, detail="Provide either 'html' or 'url'")

    if html is None and url_str:
        crawl_result = await crawler.crawl_one(url_str)
        html = crawl_result.html or ""

    if not html:
        raise HTTPException(status_code=422, detail="No HTML content available for extraction")

    chunks = await pipeline.extract_and_chunk(
        html=html,
        url=url_str,
        output_format=request.output_format,
    )

    if not chunks:
        raise HTTPException(status_code=422, detail="No content could be extracted")

    # Return the full joined text as a single extraction result
    full_text = "\n\n".join(c["text"] for c in chunks)
    first = chunks[0]

    result = ExtractResult(
        url=url_str,
        title=first.get("title"),
        text=full_text,
        author=first.get("author"),
        date=first.get("date"),
        language=first.get("language"),
        word_count=sum(c.get("word_count", 0) for c in chunks),
    )

    return ExtractResponse(
        result=result,
        elapsed_ms=(time.monotonic() - start) * 1000,
    )
