import re
import unicodedata

_BOILERPLATE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"subscribe\s+to\s+(our\s+)?newsletter",
        r"accept\s+all\s+cookies",
        r"we\s+use\s+cookies",
        r"cookie\s+policy",
        r"privacy\s+policy",
        r"terms\s+(of\s+)?service",
        r"share\s+this\s+article",
        r"follow\s+us\s+on",
        r"click\s+here\s+to\s+subscribe",
        r"sign\s+up\s+for\s+(our\s+)?newsletter",
        r"advertisement",
        r"sponsored\s+content",
        r"all\s+rights\s+reserved",
        r"copyright\s+©",
    ]
]

_SMART_QUOTES = str.maketrans({
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "--",
    "…": "...",
})

_ZERO_WIDTH = re.compile(r"[​‌‍﻿­]")
_REPEATED_PUNCT = re.compile(r"([!?,;])\1{2,}")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


class MarkdownCleaner:
    def clean(self, text: str) -> str:
        # Normalize Unicode
        text = unicodedata.normalize("NFKC", text)

        # Remove zero-width and invisible characters
        text = _ZERO_WIDTH.sub("", text)

        # Smart quotes → ASCII
        text = text.translate(_SMART_QUOTES)

        # Remove boilerplate lines
        lines = text.splitlines()
        cleaned_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if any(p.search(stripped) for p in _BOILERPLATE_PATTERNS):
                continue
            cleaned_lines.append(stripped)

        text = "\n".join(cleaned_lines)

        # Collapse repeated punctuation
        text = _REPEATED_PUNCT.sub(r"\1", text)

        # Max 2 consecutive blank lines
        text = _EXCESS_BLANK_LINES.sub("\n\n", text)

        return text.strip()

    def estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)
