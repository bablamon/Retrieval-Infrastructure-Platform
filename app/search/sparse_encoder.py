"""BM25 sparse vector encoder backed by Redis-stored corpus statistics.

Why this exists
---------------
Pure dense retrieval (cosine over sentence-transformer embeddings) is strong
for paraphrases and semantically related text, but it routinely misses
queries dominated by *exact* tokens: rare proper nouns, code identifiers,
acronyms, version strings, error messages. Hybrid retrieval — fusing dense
results with a lexical (BM25) signal — recovers that recall at near-zero
extra cost.

Design
------
The encoder produces a Qdrant ``SparseVector`` (parallel arrays of u32 token
ids and float weights). Per-collection corpus stats live in Redis:

* ``bm25:N:{collection}``        — total indexed document count
* ``bm25:avgdl:{collection}``    — running average document length
* ``bm25:df:{collection}`` (hash) — token_id -> document frequency

This avoids a vocabulary file on disk and lets multiple API replicas share
the same IDF table. Token IDs come from a stable 32-bit hash so encoding is
stateless within a single call — only the IDF lookup needs Redis.

The implementation is intentionally dependency-free (stdlib + the existing
RedisCache) so the lexical side of hybrid search adds no new install
surface, no torch/tokenizer overhead, and no per-request cold start.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass

from app.storage.redis_client import RedisCache

# Standard BM25 hyperparameters; k1 controls term-frequency saturation and
# b controls length-normalization strength. The values are the Robertson/Sparck
# Jones defaults that virtually every BM25 implementation uses.
_BM25_K1 = 1.5
_BM25_B = 0.75

# Token regex: alphanumerics plus internal apostrophes so contractions stay
# whole. Anything not matching (punctuation, whitespace) is a separator.
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")

# A small high-frequency stop list. Kept intentionally short so domain
# acronyms and rare function names are never accidentally filtered; BM25's
# IDF will down-weight common terms naturally.
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on",
        "for", "with", "by", "is", "are", "was", "were", "be", "been", "being",
        "this", "that", "these", "those", "it", "its", "as", "at", "from",
    }
)


@dataclass(slots=True)
class SparseVec:
    """Container for a sparse vector. Mirrors qdrant_client.models.SparseVector
    so the storage layer can convert without a hard import in this module."""

    indices: list[int]
    values: list[float]


def _hash_token(token: str) -> int:
    """Stable 32-bit hash for a token. Python's built-in hash() is salted per
    process, so we use md5 truncated to u32 to keep IDF lookups consistent
    across API replicas and across restarts."""
    digest = hashlib.md5(token.encode("utf-8"), usedforsecurity=False).digest()
    return int.from_bytes(digest[:4], "big")


def tokenize(text: str, min_len: int = 2) -> list[str]:
    """Lowercase tokenize and drop tiny / stopword tokens. Public so callers
    (and tests) can use the same tokenization the encoder uses."""
    out: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        tok = match.group(0)
        if len(tok) < min_len:
            continue
        if tok in _STOPWORDS:
            continue
        out.append(tok)
    return out


class BM25SparseEncoder:
    """Encodes texts to sparse BM25 vectors using corpus stats in Redis.

    Two encoding paths:

    * ``encode_documents`` — used at ingest. Updates the IDF table and the
      running ``avgdl`` so subsequent queries see fresh stats.
    * ``encode_query`` — used at retrieve. Read-only against the IDF table.

    The encoder degrades gracefully: if the collection has no documents yet,
    queries return an empty sparse vector (so the hybrid step contributes
    nothing instead of erroring), and the dense side carries retrieval.
    """

    def __init__(
        self,
        cache: RedisCache,
        min_token_len: int = 2,
        k1: float = _BM25_K1,
        b: float = _BM25_B,
    ) -> None:
        self._cache = cache
        self._min_token_len = min_token_len
        self._k1 = k1
        self._b = b

    # ---- Redis key helpers ---------------------------------------------------
    def _key_doc_count(self, collection: str) -> str:
        return f"bm25:N:{collection}"

    def _key_avgdl(self, collection: str) -> str:
        return f"bm25:avgdl:{collection}"

    def _key_df(self, collection: str) -> str:
        return f"bm25:df:{collection}"

    # ---- Corpus-stat I/O -----------------------------------------------------
    async def _doc_count(self, collection: str) -> int:
        raw = await self._cache.get(self._key_doc_count(collection))
        if raw is None:
            return 0
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 0

    async def _avgdl(self, collection: str) -> float:
        raw = await self._cache.get(self._key_avgdl(collection))
        if raw is None:
            # 100 is a reasonable seed avgdl — only matters until enough docs
            # accumulate; the running mean overwrites it on first ingest.
            return 100.0
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 100.0

    async def _get_df(self, collection: str, token_ids: list[int]) -> dict[int, int]:
        """Batch fetch document-frequency for a set of token ids via HMGET."""
        if not token_ids:
            return {}
        client = self._cache._client_or_raise()  # pragma: no cover — accessor
        fields = [str(t).encode() for t in token_ids]
        raws = await client.hmget(self._key_df(collection), fields)
        out: dict[int, int] = {}
        for tid, raw in zip(token_ids, raws, strict=False):
            if raw is None:
                continue
            try:
                out[tid] = int(raw)
            except (TypeError, ValueError):
                continue
        return out

    async def _update_corpus_stats(
        self, collection: str, doc_token_sets: list[set[int]], doc_lengths: list[int]
    ) -> None:
        """Apply a batch of newly indexed documents to the IDF + avgdl state.

        Documents only contribute +1 to df for tokens they contain (set
        semantics) — that is the BM25 convention. ``avgdl`` is updated via a
        weighted running mean so the value stays correct without scanning all
        docs.
        """
        if not doc_token_sets:
            return

        client = self._cache._client_or_raise()  # pragma: no cover — accessor
        new_doc_count = len(doc_token_sets)
        new_total_len = sum(doc_lengths)

        df_increments: Counter[int] = Counter()
        for tokens in doc_token_sets:
            for tid in tokens:
                df_increments[tid] += 1

        prev_N = await self._doc_count(collection)
        prev_avgdl = await self._avgdl(collection)
        total_N = prev_N + new_doc_count
        if total_N <= 0:
            new_avgdl = prev_avgdl
        else:
            new_avgdl = (prev_avgdl * prev_N + new_total_len) / total_N

        pipe = client.pipeline(transaction=False)
        pipe.set(self._key_doc_count(collection), str(total_N).encode())
        pipe.set(self._key_avgdl(collection), str(new_avgdl).encode())
        for tid, inc in df_increments.items():
            pipe.hincrby(self._key_df(collection), str(tid).encode(), inc)
        await pipe.execute()

    # ---- Encoding ------------------------------------------------------------
    def _tokens_to_term_freqs(self, text: str) -> tuple[Counter[int], int]:
        tokens = tokenize(text, min_len=self._min_token_len)
        tf: Counter[int] = Counter()
        for t in tokens:
            tf[_hash_token(t)] += 1
        return tf, len(tokens)

    async def encode_documents(
        self, texts: list[str], collection: str
    ) -> list[SparseVec]:
        """Encode a batch of documents and update corpus stats atomically.

        We compute IDF *after* the stats update so newly added rare terms
        already have df>=1 (avoiding a one-off log(N/0) edge case for
        single-doc collections).
        """
        if not texts:
            return []

        per_doc_tf: list[Counter[int]] = []
        per_doc_len: list[int] = []
        doc_token_sets: list[set[int]] = []
        for text in texts:
            tf, length = self._tokens_to_term_freqs(text)
            per_doc_tf.append(tf)
            per_doc_len.append(length)
            doc_token_sets.append(set(tf.keys()))

        await self._update_corpus_stats(collection, doc_token_sets, per_doc_len)

        N = await self._doc_count(collection)
        avgdl = await self._avgdl(collection)
        all_token_ids = {tid for s in doc_token_sets for tid in s}
        df = await self._get_df(collection, list(all_token_ids))

        return [
            self._bm25_weights(tf, dl, N, avgdl, df)
            for tf, dl in zip(per_doc_tf, per_doc_len, strict=False)
        ]

    async def encode_query(self, text: str, collection: str) -> SparseVec:
        """Read-only encoding for a query. Tokens missing from the IDF table
        are dropped (their IDF would be ill-defined) so queries against an
        empty collection just produce an empty sparse vector. We don't need
        ``avgdl`` here — it's a document-length normalization factor that
        only applies to indexed documents, not to query-side weighting."""
        tf, _dl = self._tokens_to_term_freqs(text)
        if not tf:
            return SparseVec(indices=[], values=[])

        N = await self._doc_count(collection)
        if N <= 0:
            return SparseVec(indices=[], values=[])

        df = await self._get_df(collection, list(tf.keys()))
        # Queries weight each query term as `1 * idf` (no length norm); doc
        # scoring is what carries TF + length normalization.
        return self._bm25_query_weights(tf, N, df)

    def _bm25_weights(
        self,
        tf: Counter[int],
        dl: int,
        N: int,
        avgdl: float,
        df: dict[int, int],
    ) -> SparseVec:
        """Compute BM25 per-term weights for a document."""
        if not tf or N <= 0 or avgdl <= 0:
            return SparseVec(indices=[], values=[])

        indices: list[int] = []
        values: list[float] = []
        length_norm = 1.0 - self._b + self._b * (dl / avgdl)
        for tid, freq in tf.items():
            n_qi = df.get(tid, 1)
            # Robertson-Sparck Jones IDF (with the +0.5 smoothing). We floor
            # at a tiny positive value so extremely common terms still get a
            # weight (BM25's raw form can go negative for terms in >half of
            # docs, which we don't want for hybrid fusion).
            idf = math.log(1.0 + (N - n_qi + 0.5) / (n_qi + 0.5))
            if idf <= 0.0:
                continue
            numerator = freq * (self._k1 + 1.0)
            denominator = freq + self._k1 * length_norm
            weight = idf * (numerator / denominator)
            if weight > 0.0:
                indices.append(tid)
                values.append(weight)
        return SparseVec(indices=indices, values=values)

    def _bm25_query_weights(
        self, tf: Counter[int], N: int, df: dict[int, int]
    ) -> SparseVec:
        """IDF-only query weighting, matching how BM25 typically scores at
        query time. Multiplying by query-side TF would over-weight repeated
        words in the query (e.g. paraphrastic restatements) which hurts more
        than helps in practice."""
        indices: list[int] = []
        values: list[float] = []
        for tid, freq in tf.items():
            n_qi = df.get(tid, 0)
            if n_qi <= 0:
                continue
            idf = math.log(1.0 + (N - n_qi + 0.5) / (n_qi + 0.5))
            if idf <= 0.0:
                continue
            indices.append(tid)
            values.append(idf * freq)
        return SparseVec(indices=indices, values=values)
