"""Tests for the HyDE expander and its integration in QueryExpander."""
import pytest

from app.search.hyde import HeuristicHyDE, _detect_topic_phrase
from app.search.query_expander import QueryExpander


def test_topic_phrase_strips_question_lead():
    assert _detect_topic_phrase("What is a transformer?") == "a transformer"
    assert _detect_topic_phrase("How does self-attention work?") == "self-attention work"
    assert _detect_topic_phrase("Tell me about LLMs") == "Tell me about LLMs"


@pytest.mark.asyncio
async def test_heuristic_hyde_generates_variants():
    hyde = HeuristicHyDE()
    variants = await hyde.generate("What is gradient descent?", n=3)
    assert len(variants) == 3
    # Each variant should reference the topic
    for v in variants:
        assert "gradient descent" in v.lower()


@pytest.mark.asyncio
async def test_heuristic_hyde_zero_request():
    hyde = HeuristicHyDE()
    assert await hyde.generate("anything", n=0) == []


@pytest.mark.asyncio
async def test_query_expander_async_path_includes_original_and_hyde():
    expander = QueryExpander(hyde=HeuristicHyDE())
    variants = await expander.expand_for_retrieval(
        "What is attention?", num_queries=3, use_hyde=True
    )
    assert variants[0] == "What is attention?"  # original always first
    assert len(variants) <= 3
    assert all(isinstance(v, str) and v for v in variants)


@pytest.mark.asyncio
async def test_query_expander_async_path_falls_back_on_hyde_failure():
    """If the HyDE generator raises, the expander should still return the
    original query and synonym variants — HyDE must never break retrieval."""

    class _BoomHyDE:
        async def generate(self, query, n=1):
            raise RuntimeError("simulated failure")

    expander = QueryExpander(hyde=_BoomHyDE())
    variants = await expander.expand_for_retrieval(
        "how to use docker", num_queries=4
    )
    assert variants[0] == "how to use docker"
    assert len(variants) >= 1


@pytest.mark.asyncio
async def test_query_expander_num_queries_one_is_passthrough():
    expander = QueryExpander(hyde=HeuristicHyDE())
    variants = await expander.expand_for_retrieval("anything", num_queries=1)
    assert variants == ["anything"]


def test_query_expander_sync_path_still_works():
    """The synchronous expander is used by /query/generate; ensure it
    didn't break when we added the async path."""
    expander = QueryExpander()
    variants = expander.expand("how to use docker", num_queries=4)
    assert variants[0] == "how to use docker"
    assert len(variants) <= 4
