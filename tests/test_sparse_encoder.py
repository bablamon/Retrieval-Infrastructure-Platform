"""Tests for the BM25 sparse encoder.

The encoder is stateful (writes IDF stats to Redis), so each test uses the
``redis_cache`` fixture (fakeredis) to isolate state.
"""
import math

import pytest

from app.search.sparse_encoder import (
    BM25SparseEncoder,
    SparseVec,
    _hash_token,
    tokenize,
)


def test_tokenize_lowercases_and_filters():
    tokens = tokenize("Hello, WORLD! Python3 is the best.")
    assert "hello" in tokens
    assert "world" in tokens
    assert "python3" in tokens
    # stopwords gone
    assert "is" not in tokens
    assert "the" not in tokens
    # short tokens (< min_len) gone — "a" is also a stopword anyway
    assert all(len(t) >= 2 for t in tokens)


def test_tokenize_preserves_internal_punct():
    tokens = tokenize("don't use C++ here")
    assert "don't" in tokens


def test_hash_token_stable_across_calls():
    a = _hash_token("transformer")
    b = _hash_token("transformer")
    assert a == b
    assert 0 <= a < 2**32


def test_hash_token_distinct_for_different_inputs():
    assert _hash_token("transformer") != _hash_token("Transformer")  # case-sensitive


@pytest.mark.asyncio
async def test_encode_documents_populates_redis_stats(redis_cache):
    enc = BM25SparseEncoder(redis_cache)
    docs = [
        "Transformer models use self-attention to process sequences.",
        "Self-attention computes weighted sums across all positions.",
        "Convolutional networks use kernels instead of attention.",
    ]
    sparse_vecs = await enc.encode_documents(docs, collection="test_coll")
    assert len(sparse_vecs) == 3
    for sv in sparse_vecs:
        assert isinstance(sv, SparseVec)
        assert len(sv.indices) == len(sv.values)
        assert all(v > 0.0 for v in sv.values)

    # corpus stats were written
    raw_n = await redis_cache.get("bm25:N:test_coll")
    assert raw_n is not None
    assert int(raw_n) == 3
    raw_avgdl = await redis_cache.get("bm25:avgdl:test_coll")
    assert raw_avgdl is not None
    assert float(raw_avgdl) > 0.0


@pytest.mark.asyncio
async def test_encode_query_returns_empty_on_empty_collection(redis_cache):
    enc = BM25SparseEncoder(redis_cache)
    sv = await enc.encode_query("transformer", collection="empty_coll")
    assert sv.indices == []
    assert sv.values == []


@pytest.mark.asyncio
async def test_encode_query_uses_idf(redis_cache):
    enc = BM25SparseEncoder(redis_cache)
    docs = [
        "transformer transformer transformer",  # 'transformer' is common
        "transformer is everywhere",
        "transformer documentation transformer",
        "rare_unique_marker_token shows up only once",
    ]
    await enc.encode_documents(docs, collection="idf_test")

    # 'rare_unique_marker_token' (df=1) should outweigh 'transformer' (df=3)
    sv = await enc.encode_query(
        "transformer rare_unique_marker_token", collection="idf_test"
    )
    assert len(sv.indices) == 2
    weights = dict(zip(sv.indices, sv.values, strict=False))
    rare_id = _hash_token("rare_unique_marker_token")
    common_id = _hash_token("transformer")
    assert weights[rare_id] > weights[common_id]


@pytest.mark.asyncio
async def test_query_drops_unknown_tokens(redis_cache):
    enc = BM25SparseEncoder(redis_cache)
    await enc.encode_documents(
        ["one document about pizza"], collection="known"
    )
    sv = await enc.encode_query("pizza never_seen_word", collection="known")
    # only 'pizza' was seen at ingest; never_seen_word is dropped.
    assert len(sv.indices) == 1
    assert sv.indices[0] == _hash_token("pizza")


@pytest.mark.asyncio
async def test_bm25_weight_finite_and_positive(redis_cache):
    """Spot-check that no math edge case (log of 0, division by 0) leaks
    into the output."""
    enc = BM25SparseEncoder(redis_cache)
    docs = ["the quick brown fox jumps over the lazy dog"] * 3
    sparse_vecs = await enc.encode_documents(docs, collection="finite")
    for sv in sparse_vecs:
        for v in sv.values:
            assert math.isfinite(v)
            assert v > 0.0
