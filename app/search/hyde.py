"""HyDE (Hypothetical Document Embeddings) query expansion.

HyDE rewrites a query into one or more *answer-shaped* texts before
embedding. The idea (Gao et al., 2022) is that the embedding of a plausible
answer sits closer to real answer passages in the index than the embedding
of the question itself, so dense retrieval recall improves — often by a
significant margin on long-form queries.

Two implementations live here:

* ``HeuristicHyDE`` — zero-dependency, deterministic template rewriter.
  Converts interrogatives into declarative answer leads and adds genre
  scaffolding ("In summary, ...", "The key idea is ..."). Always available.
* ``LLMHyDE`` — optional. Calls an external OpenAI-compatible chat endpoint
  (only used when ``HYDE_LLM_URL`` is configured). Gives much higher quality
  hypothetical documents but adds latency and a network dep.

The ``QueryExpander`` composes these with the rule-based expander; the
retrieval pipeline embeds all variants and fuses results via RRF.
"""
from __future__ import annotations

import re
from typing import Protocol


class HyDEGenerator(Protocol):
    """Anything that can turn a query into one or more hypothetical answers."""

    async def generate(self, query: str, n: int = 1) -> list[str]: ...


# ----------------------------------------------------------------------------
# Heuristic (no external calls). Cheap, deterministic, and good enough to
# noticeably move recall on factoid + how-to queries.
# ----------------------------------------------------------------------------

_QUESTION_LEADS = re.compile(
    r"^(what\s+(?:is|are|does|do)|how\s+(?:does|do|to|can|should)|"
    r"why\s+(?:is|are|does|do)|when\s+(?:is|are|does|do|to)|"
    r"where\s+(?:is|are|does|do|to)|who\s+(?:is|are))\s+",
    re.IGNORECASE,
)


def _strip_trailing_punct(s: str) -> str:
    return re.sub(r"[?!.\s]+$", "", s).strip()


def _detect_topic_phrase(query: str) -> str:
    """Pull a plausible topic phrase out of the query so we can build a
    declarative lead. Greedy: drop the question prefix, the trailing
    punctuation, and collapse whitespace."""
    stripped = _strip_trailing_punct(query)
    m = _QUESTION_LEADS.match(stripped)
    if m:
        stripped = stripped[m.end():].strip()
    return re.sub(r"\s+", " ", stripped)


class HeuristicHyDE:
    """Template-based hypothetical-answer generation.

    Generates up to ``n`` paraphrases of an answer-shaped passage. We keep
    the templates short — bge embeddings are most useful when the
    hypothetical is *passage-like*, not essay-like, so 1–2 sentences per
    variant works better than a paragraph.
    """

    _ANSWER_TEMPLATES: tuple[str, ...] = (
        "{topic} refers to {topic_lower}, which is described as follows.",
        "In summary, {topic} is a concept that involves {topic_lower}.",
        "The key idea behind {topic} is {topic_lower}; here is how it works.",
        "To answer the question about {topic}: {topic} is best understood as {topic_lower}.",
    )

    async def generate(self, query: str, n: int = 1) -> list[str]:
        if n <= 0:
            return []
        topic = _detect_topic_phrase(query) or query.strip()
        topic_clean = _strip_trailing_punct(topic) or query.strip()
        out: list[str] = []
        for template in self._ANSWER_TEMPLATES[:n]:
            out.append(
                template.format(topic=topic_clean, topic_lower=topic_clean.lower())
            )
        return out


# ----------------------------------------------------------------------------
# Optional LLM-backed HyDE. Only wired up when HYDE_LLM_URL is configured.
# Uses httpx so it inherits our existing async client / timeout discipline.
# ----------------------------------------------------------------------------


class LLMHyDE:
    """HyDE via an OpenAI-compatible chat completions endpoint.

    The expectation: any provider that exposes ``POST /v1/chat/completions``
    with an OpenAI-shaped JSON body (Ollama, vLLM, llama.cpp's server,
    OpenAI itself, Together, Groq, etc.). Failures fall through to the
    heuristic implementation in the caller — HyDE should never break the
    request path.
    """

    _PROMPT_TEMPLATE = (
        "Write a single short factual passage (2–3 sentences) that would "
        "plausibly contain the answer to the question below. Do not "
        "include a preamble; output only the passage.\n\nQuestion: {query}"
    )

    def __init__(
        self,
        url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> None:
        self._url = url.rstrip("/") + "/v1/chat/completions"
        self._model = model
        self._api_key = api_key
        self._timeout = timeout

    async def generate(self, query: str, n: int = 1) -> list[str]:
        if n <= 0:
            return []

        import httpx  # local import — only LLM path needs it

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        body = {
            "model": self._model,
            "messages": [
                {"role": "user", "content": self._PROMPT_TEMPLATE.format(query=query)}
            ],
            "n": n,
            "temperature": 0.7,
            "max_tokens": 200,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._url, headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            # Soft failure: HyDE should never break retrieval. The caller
            # falls back to the heuristic generator (or to no expansion).
            return []

        out: list[str] = []
        for choice in data.get("choices", []):
            message = choice.get("message") or {}
            content = (message.get("content") or "").strip()
            if content:
                out.append(content)
        return out[:n]
