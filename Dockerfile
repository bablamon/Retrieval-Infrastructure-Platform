FROM python:3.12-slim-bookworm

# Limit BLAS/OpenMP thread pools so CPU inference stays predictable and
# memory-bounded under concurrency (overridable at runtime).
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libxml2-dev \
    libxslt-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU-only torch wheel FIRST so the default CUDA build (which drags
# in ~5GB of unused nvidia-* libraries) is never pulled. requirements.txt then
# sees torch==2.12.0 already satisfied and skips it.
RUN pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Install Playwright chromium browser
RUN playwright install chromium --with-deps

# Pre-download ML models at build time
COPY scripts/download_models.py scripts/
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANKER_MODEL=BAAI/bge-reranker-base
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
ENV RERANKER_MODEL=${RERANKER_MODEL}
RUN python scripts/download_models.py

COPY . .

EXPOSE 8000

# Single worker: Playwright browser processes are not fork-safe.
# Scale horizontally via container replicas instead.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
