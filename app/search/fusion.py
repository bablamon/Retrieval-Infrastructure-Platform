"""Rank-fusion and diversity utilities used by the retrieval pipeline.

Two algorithms live here:

* ``reciprocal_rank_fusion`` — combines multiple ranked lists (e.g. dense vs.
  sparse, or several expanded-query results) into a single list. RRF is
  rank-based, so it is robust to score scales that differ across retrievers.
* ``maximal_marginal_relevance`` — re-ranks a candidate list to balance
  relevance against intra-list redundancy, suppressing near-duplicate chunks
  that often dominate top-k from the same source document.

Both are pure functions with no I/O so they are trivially unit-testable.
"""
from __future__ import annotations

import math
from collections.abc import Hashable, Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Hashable]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[Hashable, float]]:
    """Fuse N ranked lists of opaque IDs using Reciprocal Rank Fusion.

    score(doc) = sum_i w_i * (1 / (k + rank_i(doc)))

    Items absent from a given list contribute 0 for that list. The constant
    ``k`` (60 per Cormack et al.) dampens the contribution of low ranks so the
    head of each list dominates fusion. Returns a list of ``(id, score)`` in
    descending score order.
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match number of ranked lists")

    scores: dict[Hashable, float] = {}
    for w, ranked in zip(weights, ranked_lists, strict=False):
        for rank, item in enumerate(ranked, start=1):
            scores[item] = scores.get(item, 0.0) + w * (1.0 / (k + rank))

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity without numpy — vectors are usually normalized
    already (we set ``normalize_embeddings=True`` in the embedder), but we do
    not assume it here so the helper is reusable."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def maximal_marginal_relevance(
    query_vector: Sequence[float],
    candidates: Sequence[tuple[Hashable, Sequence[float]]],
    top_k: int,
    lambda_: float = 0.7,
) -> list[Hashable]:
    """Select ``top_k`` items from ``candidates`` to maximize MMR.

    mmr = λ * sim(d, q) - (1 - λ) * max_{d' in selected} sim(d, d')

    Higher λ → more relevance, lower λ → more diversity. ``candidates`` is
    ``(id, vector)`` pairs; only IDs are returned, in MMR-selection order.
    """
    if top_k <= 0 or not candidates:
        return []
    if not 0.0 <= lambda_ <= 1.0:
        raise ValueError("lambda_ must be in [0, 1]")

    remaining: list[tuple[Hashable, Sequence[float]]] = list(candidates)
    relevance = {item_id: _cosine(query_vector, vec) for item_id, vec in remaining}

    selected: list[Hashable] = []
    selected_vecs: list[Sequence[float]] = []

    while remaining and len(selected) < top_k:
        best_score = -math.inf
        best_idx = 0
        for idx, (item_id, vec) in enumerate(remaining):
            if not selected_vecs:
                score = lambda_ * relevance[item_id]
            else:
                max_sim_to_selected = max(_cosine(vec, sv) for sv in selected_vecs)
                score = lambda_ * relevance[item_id] - (1.0 - lambda_) * max_sim_to_selected
            if score > best_score:
                best_score = score
                best_idx = idx
        picked_id, picked_vec = remaining.pop(best_idx)
        selected.append(picked_id)
        selected_vecs.append(picked_vec)

    return selected
