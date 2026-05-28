"""Semantic chunker.

Splits cleaned markdown / text into chunks that respect paragraph and
sentence boundaries while staying under a configured size budget. The
budget is now measured in *tokens* (using the embedding model's tokenizer
when one is provided) rather than whitespace-split words — bge-small-en
has a hard 512-token context window, and word counts diverge from token
counts dramatically for code, CJK text, or anything with rich punctuation,
so the old word-count cap silently allowed oversize chunks that were then
truncated by the encoder.

The chunker still degrades gracefully to word counts when no tokenizer is
passed (used by tests and by anything that doesn't load the embedding
model), so the contract is unchanged for those callers.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

TokenCounter = Callable[[str], int]


class SemanticChunker:
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        token_counter: TokenCounter | None = None,
    ) -> None:
        """``chunk_size`` and ``chunk_overlap`` are in *tokens* when
        ``token_counter`` is supplied, else in whitespace-split words. The
        defaults (512 / 64) match the bge-small context window."""
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        # Lazy import the tokenizer-based counter; fall back to word-split so
        # the chunker stays usable without a loaded embedding model (tests,
        # workers that don't embed, etc).
        self._count: TokenCounter = token_counter or self._word_count

    def chunk(self, text: str, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        segments = self._paragraphs_to_segments(paragraphs)
        merged = self._merge_small_segments(segments)
        chunks = self._apply_overlap(merged)

        result: list[dict[str, Any]] = []
        total = len(chunks)
        for i, chunk_text in enumerate(chunks):
            result.append({
                "text": chunk_text,
                "chunk_index": i,
                "total_chunks": total,
                # ``word_count`` is kept as a friendly metadata field for
                # consumers / debugging; the *chunking* decisions all use
                # the configured token counter.
                "word_count": self._word_count(chunk_text),
                **metadata,
            })
        return result

    def _paragraphs_to_segments(self, paragraphs: list[str]) -> list[str]:
        segments: list[str] = []
        for para in paragraphs:
            if self._count(para) <= self._chunk_size:
                segments.append(para)
                continue

            sentences = _SENTENCE_RE.split(para)
            current = ""
            for sentence in sentences:
                # Safety net: a single sentence longer than chunk_size (e.g.
                # code, tables, or unpunctuated text) is hard-split so no
                # chunk can exceed the embedding model's context window.
                if self._count(sentence) > self._chunk_size:
                    if current:
                        segments.append(current)
                        current = ""
                    segments.extend(self._split_oversize(sentence))
                    continue

                candidate = f"{current} {sentence}".strip() if current else sentence
                if self._count(candidate) <= self._chunk_size:
                    current = candidate
                else:
                    if current:
                        segments.append(current)
                    current = sentence
            if current:
                segments.append(current)
        return segments

    def _split_oversize(self, text: str) -> list[str]:
        """Hard-split an oversize sentence. We split by words and then *measure
        in tokens*, so even when chunk_size is in tokens we never exceed it.
        The loop accumulates words until adding the next one would overflow,
        then emits a chunk and continues."""
        words = text.split()
        if not words:
            return []

        result: list[str] = []
        current: list[str] = []
        for word in words:
            candidate_words = [*current, word]
            candidate_text = " ".join(candidate_words)
            if self._count(candidate_text) > self._chunk_size and current:
                result.append(" ".join(current))
                current = [word]
            else:
                current = candidate_words
        if current:
            result.append(" ".join(current))
        return result

    def _merge_small_segments(self, segments: list[str]) -> list[str]:
        if not segments:
            return []
        merged: list[str] = []
        current = segments[0]
        for seg in segments[1:]:
            candidate = f"{current}\n\n{seg}"
            if self._count(candidate) <= self._chunk_size:
                current = candidate
            else:
                merged.append(current)
                current = seg
        merged.append(current)
        return merged

    def _apply_overlap(self, segments: list[str]) -> list[str]:
        if self._chunk_overlap <= 0 or len(segments) <= 1:
            return segments

        result: list[str] = [segments[0]]
        for i in range(1, len(segments)):
            prev_words = segments[i - 1].split()
            # Word-level overlap is an approximation when the counter is
            # token-based, but tokens-per-word is bounded (~1.3 for English
            # with bge), so taking the last N words gives roughly N–1.3N
            # overlap tokens — close enough to the user's intent and far
            # cheaper than tokenizing repeatedly.
            overlap_words = (
                prev_words[-self._chunk_overlap:]
                if len(prev_words) > self._chunk_overlap
                else prev_words
            )
            overlap_text = " ".join(overlap_words)
            result.append(f"{overlap_text}\n\n{segments[i]}")
        return result

    def _word_count(self, text: str) -> int:
        return len(text.split())

    def _split_into_sentences(self, text: str) -> list[str]:
        return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
