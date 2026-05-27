import time

from fastapi import APIRouter, Depends

from app.core.dependencies import get_crawler
from app.crawler.async_crawler import AsyncCrawler
from app.models.requests import CrawlRequest
from app.models.responses import CrawlResponse

router = APIRouter(prefix="/crawl", tags=["crawl"])


@router.post("", response_model=CrawlResponse)
async def crawl(
    request: CrawlRequest,
    crawler: AsyncCrawler = Depends(get_crawler),
) -> CrawlResponse:
    start = time.monotonic()
    urls = [str(u) for u in request.urls]

    if request.max_depth > 1:
        results, failures = await crawler.crawl_recursive(
            urls=urls,
            max_depth=request.max_depth,
            use_playwright=request.use_playwright,
            respect_robots=request.respect_robots,
            use_cache=request.use_cache,
        )
    else:
        results, failures = await crawler.crawl_many(
            urls=urls,
            use_playwright=request.use_playwright,
            respect_robots=request.respect_robots,
            use_cache=request.use_cache,
        )

    return CrawlResponse(
        results=results,
        failed=failures,
        total=len(results) + len(failures),
        elapsed_ms=(time.monotonic() - start) * 1000,
    )
