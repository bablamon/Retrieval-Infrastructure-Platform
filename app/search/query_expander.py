"""Query expansion: produce N retrieval variants for one user query.

The original implementation was a synonym / qualifier rewriter. It still
lives here (``RuleExpander.expand``) because it is cheap, deterministic, and
useful for SearXNG-style keyword search. For dense retrieval we layer in
HyDE-style hypothetical-answer expansion, which moves embeddings closer to
real answer passages and lifts recall on factoid + how-to queries.

``QueryExpander.expand_async`` is the new entry point used by the retrieval
pipeline; the synchronous ``expand`` is preserved for the existing
``/query/generate`` endpoint and its callers.
"""
from __future__ import annotations

import re

from app.search.hyde import HeuristicHyDE, HyDEGenerator


class QueryExpander:
    _synonym_map: dict[str, list[str]] = {
        "how to": ["guide", "tutorial", "implement"],
        "best": ["top", "recommended", "optimal"],
        "fast": ["high-performance", "efficient", "optimized"],
        "example": ["sample", "demo", "use case"],
        "python": ["python3", "py"],
        "javascript": ["js", "node.js", "nodejs"],
        "database": ["db", "datastore", "storage"],
        "machine learning": ["ml", "deep learning", "neural network"],
        "api": ["rest api", "web api", "http api", "endpoint"],
        "docker": ["container", "containerized", "docker container"],
        "error": ["bug", "issue", "exception", "problem"],
        "install": ["setup", "configure", "deployment"],
        "search": ["query", "retrieval", "lookup", "find"],
        "vector": ["embedding", "dense vector", "semantic"],
        "async": ["asynchronous", "concurrent", "non-blocking"],
    }

    _technical_qualifiers = [
        "implementation",
        "example",
        "tutorial",
        "best practices",
        "how to",
        "guide",
    ]

    _question_words = re.compile(
        r"^(what is|what are|how does|how do|why does|why is|when to|where to)\s+",
        re.IGNORECASE,
    )

    def __init__(self, hyde: HyDEGenerator | None = None) -> None:
        # Heuristic HyDE is always-on; LLMHyDE can be injected via DI when
        # ``HYDE_LLM_URL`` is configured (see core.dependencies).
        self._hyde: HyDEGenerator = hyde or HeuristicHyDE()

    # ------------------------------------------------------------------
    # Synchronous expansion (used by /query/generate and SearXNG search).
    # Pure keyword rewriting; no HyDE here because hypothetical passages
    # aren't useful as web-search queries.
    # ------------------------------------------------------------------
    def expand(
        self,
        query: str,
        num_queries: int = 5,
        include_original: bool = True,
    ) -> list[str]:
        candidates: list[str] = []

        if include_original:
            candidates.append(query)

        declarative = self._restructure_question(query)
        if declarative and declarative not in candidates:
            candidates.append(declarative)

        for variant in self._apply_synonyms(query):
            if variant not in candidates:
                candidates.append(variant)

        base = declarative or query
        for qualifier in self._technical_qualifiers:
            variant = f"{base} {qualifier}"
            if variant not in candidates:
                candidates.append(variant)
                if len(candidates) >= num_queries:
                    break

        return candidates[:num_queries]

    # ------------------------------------------------------------------
    # Async expansion for dense retrieval. Layers HyDE-style hypothetical
    # passages on top of the rule-based variants. The retrieval pipeline
    # embeds every returned variant and fuses results via RRF.
    # ------------------------------------------------------------------
    async def expand_for_retrieval(
        self,
        query: str,
        num_queries: int = 3,
        use_hyde: bool = True,
    ) -> list[str]:
        if num_queries <= 1:
            return [query]

        variants: list[str] = [query]

        # Slot 2: a declarative restatement, which usually embeds better than
        # the literal question form because corpus passages are declarative.
        declarative = self._restructure_question(query)
        if declarative and declarative not in variants:
            variants.append(declarative)

        # Remaining slots: HyDE hypothetical answers, then synonym variants
        # as a fallback if HyDE produces too few.
        remaining = num_queries - len(variants)
        if use_hyde and remaining > 0:
            try:
                hyde_variants = await self._hyde.generate(query, n=remaining)
            except Exception:
                hyde_variants = []
            for hv in hyde_variants:
                if hv and hv not in variants:
                    variants.append(hv)
                if len(variants) >= num_queries:
                    break

        if len(variants) < num_queries:
            for synonym_variant in self._apply_synonyms(query):
                if synonym_variant not in variants:
                    variants.append(synonym_variant)
                if len(variants) >= num_queries:
                    break

        return variants[:num_queries]

    # ------------------------------------------------------------------
    def _restructure_question(self, query: str) -> str | None:
        match = self._question_words.match(query)
        if match:
            return query[match.end():].strip()
        return None

    def _apply_synonyms(self, query: str) -> list[str]:
        variants: list[str] = []
        lower = query.lower()
        for term, synonyms in self._synonym_map.items():
            if term in lower:
                for syn in synonyms[:2]:
                    variant = re.sub(re.escape(term), syn, lower, flags=re.IGNORECASE)
                    if variant != lower:
                        variants.append(variant)
        return variants
