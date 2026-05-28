"""Unit tests for the rank-fusion + MMR helpers in app.search.fusion."""
import math

from app.search.fusion import (
    _cosine,
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
)

# ============================================================================
# Reciprocal Rank Fusion
# ============================================================================


def test_rrf_single_list_is_identity():
    """One ranked list → output order matches input order."""
    result = reciprocal_rank_fusion([["a", "b", "c"]])
    ids = [item for item, _ in result]
    assert ids == ["a", "b", "c"]


def test_rrf_combines_two_lists():
    """RRF should reward items that appear in multiple lists. 'b' shows up
    at rank 1 in both lists, so it must dominate the items that appear only
    once or appear lower."""
    fused = reciprocal_rank_fusion([["b", "a", "c"], ["b", "d", "e"]])
    ids = [item for item, _ in fused]
    assert ids[0] == "b"
    # all five distinct items present
    assert set(ids) == {"a", "b", "c", "d", "e"}


def test_rrf_score_formula():
    """Manually verify the 1/(k+rank) formula for a known case."""
    fused = dict(reciprocal_rank_fusion([["x", "y"]], k=60))
    assert math.isclose(fused["x"], 1 / 61, rel_tol=1e-9)
    assert math.isclose(fused["y"], 1 / 62, rel_tol=1e-9)


def test_rrf_weights_shift_results():
    """Weights should be honored — a heavily-weighted list dominates."""
    fused_balanced = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    fused_skewed = reciprocal_rank_fusion(
        [["a", "b"], ["b", "a"]], weights=[10.0, 0.1]
    )
    skewed_order = [item for item, _ in fused_skewed]
    assert skewed_order[0] == "a"  # first list dominates → 'a' wins
    # balanced fusion is symmetric so order is not meaningful, just ensure no crash
    assert len(fused_balanced) == 2


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([]) == []


def test_rrf_mismatched_weights_raises():
    import pytest
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])


# ============================================================================
# MMR
# ============================================================================


def test_cosine_helper_unit_vectors():
    assert math.isclose(_cosine([1, 0], [1, 0]), 1.0)
    assert math.isclose(_cosine([1, 0], [0, 1]), 0.0)
    assert math.isclose(_cosine([1, 0], [-1, 0]), -1.0)
    # Zero vector → 0 (no NaN)
    assert _cosine([0, 0], [1, 0]) == 0.0


def test_mmr_picks_relevant_first():
    """With λ=1.0 MMR degenerates to pure relevance."""
    q = [1.0, 0.0]
    candidates = [
        ("a", [0.1, 0.9]),  # not relevant
        ("b", [0.9, 0.1]),  # very relevant
        ("c", [0.5, 0.5]),  # medium
    ]
    picked = maximal_marginal_relevance(q, candidates, top_k=3, lambda_=1.0)
    assert picked[0] == "b"


def test_mmr_diversity_at_low_lambda():
    """Two near-duplicate top results: at λ=0.5 MMR should pick a diverse
    third option over a third near-duplicate."""
    q = [1.0, 0.0]
    candidates = [
        ("dup1", [1.0, 0.0]),
        ("dup2", [0.99, 0.01]),  # almost identical to dup1
        ("diff", [0.6, 0.8]),  # less relevant but very different
    ]
    picked = maximal_marginal_relevance(q, candidates, top_k=2, lambda_=0.3)
    assert picked[0] == "dup1"  # most relevant pick first
    # At low λ the diversity bonus pushes 'diff' above 'dup2'
    assert picked[1] == "diff"


def test_mmr_empty_inputs():
    assert maximal_marginal_relevance([1.0], [], top_k=5) == []
    assert maximal_marginal_relevance([1.0], [("a", [1.0])], top_k=0) == []


def test_mmr_invalid_lambda_raises():
    import pytest
    with pytest.raises(ValueError):
        maximal_marginal_relevance([1.0], [("a", [1.0])], top_k=1, lambda_=2.0)
