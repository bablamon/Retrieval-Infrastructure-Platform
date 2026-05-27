import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import crawl, extract, health, ingest, query, retrieve, search
from app.core.config import settings
from app.core.dependencies import disconnect_all_dependencies, initialize_dependencies
from app.core.exceptions import RetrievalBaseException, retrieval_exception_handler
from app.core.logging import configure_logging, get_logger

logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging(settings.app_log_level)
    logger.info("startup", env=settings.app_env, version=settings.app_version)

    await initialize_dependencies()

    logger.info("startup_complete")
    yield

    logger.info("shutdown_started")
    await disconnect_all_dependencies()
    logger.info("shutdown_complete")


app = FastAPI(
    title="Retrieval Infrastructure Platform",
    description="Open-source alternative to Tavily + Firecrawl, optimized for autonomous AI agents.",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        elapsed_ms=round(elapsed_ms, 1),
    )
    return response


app.add_exception_handler(RetrievalBaseException, retrieval_exception_handler)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "type": type(exc).__name__},
    )


# Register routers
app.include_router(health.router)
app.include_router(query.router)
app.include_router(search.router)
app.include_router(crawl.router)
app.include_router(extract.router)
app.include_router(ingest.router)
app.include_router(retrieve.router)
