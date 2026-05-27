import re


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

    def expand(
        self,
        query: str,
        num_queries: int = 5,
        include_original: bool = True,
    ) -> list[str]:
        candidates: list[str] = []

        if include_original:
            candidates.append(query)

        # Question → declarative
        declarative = self._restructure_question(query)
        if declarative and declarative not in candidates:
            candidates.append(declarative)

        # Synonym variants
        for variant in self._apply_synonyms(query):
            if variant not in candidates:
                candidates.append(variant)

        # Technical qualifier variants
        base = declarative or query
        for qualifier in self._technical_qualifiers:
            variant = f"{base} {qualifier}"
            if variant not in candidates:
                candidates.append(variant)
                if len(candidates) >= num_queries:
                    break

        return candidates[:num_queries]

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
