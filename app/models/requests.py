from typing import Any, Literal

from pydantic import AnyHttpUrl, BaseModel, Field


class QueryGenerateRequest(BaseModel):
    query: str
    num_queries: int = Field(default=5, ge=1, le=20)
    include_original: bool = True


class SearchRequest(BaseModel):
    query: str
    num_results: int = Field(default=10, ge=1, le=50)
    engines: list[str] = Field(default_factory=list)
    safe_search: int = Field(default=1, ge=0, le=2)
    use_cache: bool = True


class CrawlRequest(BaseModel):
    urls: list[AnyHttpUrl]
    use_playwright: bool = False
    respect_robots: bool = True
    # max_depth > 1 enables breadth-first recursive crawling of same-domain links.
    max_depth: int = Field(default=1, ge=1, le=5)
    use_cache: bool = True


class ExtractRequest(BaseModel):
    html: str | None = None
    url: AnyHttpUrl | None = None
    output_format: Literal["markdown", "text"] = "markdown"
    with_metadata: bool = True


class IngestRequest(BaseModel):
    urls: list[AnyHttpUrl]
    collection: str | None = None
    chunk_size: int = Field(default=512, ge=64, le=2048)
    chunk_overlap: int = Field(default=64, ge=0, le=512)
    use_playwright: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    # When true, all existing chunks for each ingested URL are deleted before
    # the new chunks are upserted. This is the idempotent default — repeated
    # ingest of the same URL yields the same final state instead of growing
    # the collection with stale chunks from prior crawls.
    replace: bool = True


class RetrieveRequest(BaseModel):
    query: str
    collection: str | None = None
    top_k: int = Field(default=20, ge=1, le=100)
    rerank_top_k: int = Field(default=2, ge=1, le=50)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    filters: dict[str, Any] | None = None
    # When true, the response includes a citation-formatted context pack.
    include_context: bool = True
    # ---- Retrieval-quality knobs ----
    # Hybrid search: fuse dense (semantic) and BM25 (lexical) results via RRF.
    # No effect on legacy (dense-only) collections.
    use_hybrid: bool = True
    # Multi-query: when > 1, the query is expanded into N retrieval variants
    # (declarative restatement, HyDE hypothetical answers, synonym variants)
    # and the results are fused via RRF before reranking.
    num_queries: int = Field(default=3, ge=1, le=8)
    # HyDE: prepend a hypothetical-answer passage to the expansion set.
    use_hyde: bool = True
    # MMR: re-rank the reranker's output to suppress near-duplicate chunks.
    use_mmr: bool = False
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    # ---- No-hallucination guardrail ----
    # Minimum cross-encoder rerank score (normalized to [0, 1]) required for
    # a chunk to be returned. Chunks below this floor are dropped; if NO
    # chunk passes, the response is a structured refusal (refused=True,
    # empty chunks/citations, answer_context=None) so the consuming agent
    # can decline to answer instead of fabricating from low-signal context.
    # 0.0 disables the gate entirely (legacy behavior).
    min_relevance: float = Field(default=0.3, ge=0.0, le=1.0)
