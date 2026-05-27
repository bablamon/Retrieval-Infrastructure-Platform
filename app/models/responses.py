from typing import Any, Literal

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str
    engine: str
    score: float | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total: int
    cached: bool
    elapsed_ms: float


class CrawlResult(BaseModel):
    url: str
    status_code: int
    html: str | None
    final_url: str
    content_type: str
    elapsed_ms: float
    used_playwright: bool
    cached: bool


class CrawlResponse(BaseModel):
    results: list[CrawlResult]
    failed: list[dict[str, str]]
    total: int
    elapsed_ms: float


class ExtractResult(BaseModel):
    url: str | None = None
    title: str | None = None
    text: str
    author: str | None = None
    date: str | None = None
    language: str | None = None
    word_count: int


class ExtractResponse(BaseModel):
    result: ExtractResult
    elapsed_ms: float


class ChunkMetadata(BaseModel):
    url: str
    title: str | None = None
    chunk_index: int
    total_chunks: int
    word_count: int
    extra: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    ingested_urls: list[str]
    failed_urls: list[dict[str, str]]
    total_chunks: int
    collection: str
    elapsed_ms: float


class RetrievedChunk(BaseModel):
    text: str
    score: float
    rerank_score: float | None = None
    metadata: ChunkMetadata


class RetrieveResponse(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    total_retrieved: int
    total_reranked: int
    elapsed_ms: float


class HealthStatus(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    services: dict[str, bool]
    version: str
    uptime_seconds: float


class QueryGenerateResponse(BaseModel):
    original: str
    queries: list[str]
    elapsed_ms: float
