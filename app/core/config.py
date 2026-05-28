from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    # App
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: str = "INFO"
    app_max_crawl_concurrency: int = 10
    app_max_playwright_pages: int = 5
    app_max_crawl_depth: int = 3
    app_request_timeout: int = 30
    app_version: str = "1.0.0"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl: int = 3600
    redis_search_cache_ttl: int = 900
    redis_robots_cache_ttl: int = 86400

    # PostgreSQL
    postgres_dsn: str = "postgresql://user:password@localhost:5432/retrieval_db"
    postgres_pool_min: int = 2
    postgres_pool_max: int = 10

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "retrieval_chunks"
    qdrant_vector_size: int = 384

    # SearXNG
    searxng_url: str = "http://localhost:8080"
    searxng_max_results: int = 10

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 64
    embedding_device: str = "cpu"

    # Reranker
    # bge-reranker-base (~1.1GB) is the memory-optimal default. For higher
    # quality / multilingual support set RERANKER_MODEL=BAAI/bge-reranker-v2-m3.
    reranker_model: str = "BAAI/bge-reranker-base"
    # fp16 is only beneficial on CUDA. On CPU it falls back to slow float ops,
    # so it is disabled by default and auto-disabled on CPU in RerankerService.
    reranker_use_fp16: bool = False
    reranker_top_k: int = 2

    # ARQ Worker
    arq_redis_url: str = "redis://localhost:6379/1"
    arq_max_jobs: int = 20
    arq_job_timeout: int = 300

    # Retrieval quality
    # Hybrid (BM25 + dense) is enabled by default; can be turned off globally
    # to fall back to legacy dense-only retrieval for benchmarking.
    retrieval_hybrid_default: bool = True
    retrieval_num_queries_default: int = 3
    retrieval_use_hyde_default: bool = True
    retrieval_use_mmr_default: bool = False
    retrieval_mmr_lambda_default: float = 0.7
    # Minimum normalized cross-encoder rerank score required to surface a
    # chunk. Below this the chunk is dropped; if no chunk passes, the
    # request is refused. 0.3 is a conservative default — empirically, BGE
    # rerank scores above ~0.3 mean "answers the question", below means
    # "loosely related at best".
    retrieval_min_relevance_default: float = 0.3
    # RRF dampening constant; 60 is the Cormack et al. default and rarely
    # needs tuning. Exposed for benchmarking.
    retrieval_rrf_k: int = 60
    # When N queries are run in multi-query mode, each searches for
    # top_k * this factor candidates so the post-fusion top_k still has
    # diverse coverage. Higher = more recall, more Qdrant load.
    retrieval_multi_query_overfetch: float = 1.5

    # HyDE LLM (optional). If unset, the heuristic HyDE generator is used.
    # Any OpenAI-compatible chat-completions endpoint works (Ollama, vLLM,
    # llama.cpp server, OpenAI, Together, Groq, ...).
    hyde_llm_url: str = ""
    hyde_llm_model: str = ""
    hyde_llm_api_key: str = ""
    hyde_llm_timeout: float = 8.0


settings = Settings()
