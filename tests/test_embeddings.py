from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.mark.asyncio
async def test_embed_texts_returns_correct_dimensions(redis_cache):
    from app.embeddings.embedding_service import EmbeddingService

    service = EmbeddingService(
        model_name="BAAI/bge-small-en-v1.5",
        device="cpu",
        batch_size=32,
        cache=redis_cache,
    )

    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(3, 384).astype(np.float32)
    service._model = mock_model

    texts = ["hello world", "how does attention work", "python async programming"]
    vectors = await service.embed_texts(texts)

    assert len(vectors) == 3
    assert all(len(v) == 384 for v in vectors)


@pytest.mark.asyncio
async def test_cache_hit_skips_model(redis_cache):
    from app.embeddings.embedding_service import EmbeddingService

    service = EmbeddingService(
        model_name="BAAI/bge-small-en-v1.5",
        device="cpu",
        batch_size=32,
        cache=redis_cache,
    )

    mock_model = MagicMock()
    mock_model.encode.return_value = np.random.rand(1, 384).astype(np.float32)
    service._model = mock_model

    texts = ["cached text example"]

    # First call — model should be invoked
    vectors1 = await service.embed_texts(texts)
    assert mock_model.encode.call_count == 1

    # Second call — should be served from cache
    vectors2 = await service.embed_texts(texts)
    assert mock_model.encode.call_count == 1  # unchanged
    assert len(vectors2[0]) == len(vectors1[0])


@pytest.mark.asyncio
async def test_embed_query_adds_prefix(redis_cache):
    from app.embeddings.embedding_service import EmbeddingService

    service = EmbeddingService(
        model_name="BAAI/bge-small-en-v1.5",
        device="cpu",
        batch_size=32,
        cache=redis_cache,
    )

    encoded_texts = []

    def mock_encode(texts, **kwargs):
        encoded_texts.extend(texts)
        return np.random.rand(len(texts), 384).astype(np.float32)

    mock_model = MagicMock()
    mock_model.encode.side_effect = mock_encode
    service._model = mock_model

    await service.embed_query("how does attention work")
    assert any(t.startswith("query: ") for t in encoded_texts)


@pytest.mark.asyncio
async def test_embed_empty_list(redis_cache):
    from app.embeddings.embedding_service import EmbeddingService

    service = EmbeddingService(
        model_name="BAAI/bge-small-en-v1.5",
        device="cpu",
        batch_size=32,
        cache=redis_cache,
    )
    service._model = MagicMock()

    result = await service.embed_texts([])
    assert result == []
    service._model.encode.assert_not_called()
