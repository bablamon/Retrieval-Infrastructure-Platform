import re
from typing import Any

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


class SemanticChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

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
                "word_count": self._word_count(chunk_text),
                **metadata,
            })
        return result

    def _paragraphs_to_segments(self, paragraphs: list[str]) -> list[str]:
        segments: list[str] = []
        for para in paragraphs:
            if self._word_count(para) <= self._chunk_size:
                segments.append(para)
                continue

            sentences = _SENTENCE_RE.split(para)
            current = ""
            for sentence in sentences:
                # Safety net: a single sentence longer than chunk_size (e.g. code,
                # tables, or unpunctuated text) is hard-split by word count so no
                # chunk can exceed the embedding model's context window.
                if self._word_count(sentence) > self._chunk_size:
                    if current:
                        segments.append(current)
                        current = ""
                    segments.extend(self._split_by_words(sentence))
                    continue

                candidate = f"{current} {sentence}".strip() if current else sentence
                if self._word_count(candidate) <= self._chunk_size:
                    current = candidate
                else:
                    if current:
                        segments.append(current)
                    current = sentence
            if current:
                segments.append(current)
        return segments

    def _split_by_words(self, text: str) -> list[str]:
        words = text.split()
        return [
            " ".join(words[i : i + self._chunk_size])
            for i in range(0, len(words), self._chunk_size)
        ]

    def _merge_small_segments(self, segments: list[str]) -> list[str]:
        if not segments:
            return []
        merged: list[str] = []
        current = segments[0]
        for seg in segments[1:]:
            candidate = f"{current}\n\n{seg}"
            if self._word_count(candidate) <= self._chunk_size:
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
            overlap_words = prev_words[-self._chunk_overlap :] if len(prev_words) > self._chunk_overlap else prev_words
            overlap_text = " ".join(overlap_words)
            result.append(f"{overlap_text}\n\n{segments[i]}")
        return result

    def _word_count(self, text: str) -> int:
        return len(text.split())

    def _split_into_sentences(self, text: str) -> list[str]:
        return [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
