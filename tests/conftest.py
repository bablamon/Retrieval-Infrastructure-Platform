from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head><title>Test Article: Understanding Transformers</title></head>
<body>
  <nav><a href="/">Home</a></nav>
  <main>
    <article>
      <h1>Understanding Transformer Architecture</h1>
      <p>The Transformer model was introduced in the paper "Attention is All You Need" by Vaswani et al. in 2017.
      It revolutionized natural language processing by replacing recurrent neural networks with self-attention mechanisms.</p>
      <p>The core innovation is the multi-head attention mechanism, which allows the model to attend to different
      positions of the input sequence simultaneously. This enables parallelization during training, making
      Transformers significantly faster than RNNs on modern hardware.</p>
      <p>The architecture consists of an encoder and decoder stack. Each encoder layer has two sub-layers:
      a multi-head self-attention mechanism and a position-wise fully connected feed-forward network.
      Residual connections are employed around each sub-layer, followed by layer normalization.</p>
    </article>
  </main>
  <footer><p>Cookie policy. Subscribe to our newsletter. All rights reserved.</p></footer>
</body>
</html>
"""


@pytest_asyncio.fixture
async def redis_cache():
    import fakeredis.aioredis

    from app.storage.redis_client import RedisCache

    cache = RedisCache.__new__(RedisCache)
    cache._url = "redis://localhost:6379/0"
    cache._default_ttl = 3600
    cache._client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    yield cache
    await cache._client.aclose()


@pytest.fixture
def sample_html() -> str:
    return SAMPLE_HTML


@pytest.fixture
def mock_postgres():
    mock = MagicMock()
    mock.is_url_crawled = AsyncMock(return_value=False)
    mock.upsert_crawl_state = AsyncMock(return_value=None)
    mock.health_check = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def mock_qdrant():
    mock = MagicMock()
    mock.health_check = AsyncMock(return_value=True)
    mock.ensure_collection = AsyncMock(return_value=None)
    mock.upsert_chunks = AsyncMock(return_value=5)
    mock.search = AsyncMock(return_value=[])
    return mock
