from typing import Any

from arq.connections import RedisSettings

from app.core.config import settings
from app.core.logging import configure_logging, get_logger

logger = get_logger("worker")


async def ingest_task(ctx: dict[str, Any], ingest_payload: dict[str, Any]) -> dict[str, Any]:
    from app.models.requests import IngestRequest

    pipeline = ctx["retrieval_pipeline"]
    request = IngestRequest(**ingest_payload)
    result = await pipeline.ingest(request)
    return result.model_dump()


async def crawl_task(
    ctx: dict[str, Any],
    urls: list[str],
    use_playwright: bool = False,
) -> dict[str, Any]:
    crawler = ctx["crawler"]
    results, failures = await crawler.crawl_many(urls, use_playwright=use_playwright)
    return {
        "crawled": len(results),
        "failed": len(failures),
        "failures": failures,
    }


async def on_startup(ctx: dict[str, Any]) -> None:
    configure_logging(settings.app_log_level)
    logger.info("worker_startup")

    from app.core.dependencies import (
        get_crawler,
        get_retrieval_pipeline,
        initialize_dependencies,
    )

    # The worker only ingests and crawls — it never reranks, so skip loading
    # the reranker model and save its memory footprint.
    await initialize_dependencies(load_reranker=False)

    ctx["crawler"] = get_crawler()
    ctx["retrieval_pipeline"] = get_retrieval_pipeline()
    logger.info("worker_ready")


async def on_shutdown(ctx: dict[str, Any]) -> None:
    from app.core.dependencies import disconnect_all_dependencies
    await disconnect_all_dependencies()
    logger.info("worker_shutdown")


class WorkerSettings:
    functions = [ingest_task, crawl_task]
    on_startup = on_startup
    on_shutdown = on_shutdown
    redis_settings = RedisSettings.from_dsn(settings.arq_redis_url)
    max_jobs = settings.arq_max_jobs
    job_timeout = settings.arq_job_timeout
    keep_result = 3600
