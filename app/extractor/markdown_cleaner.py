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

# Wiki / markdown extraction artifacts
_FRONTMATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
_EDIT_MARKER = re.compile(r"\[edit\]", re.IGNORECASE)
_REF_MARKERS = re.compile(
    r"\[(?:citation needed|when\?|who\?|why\?|update|clarification needed|"
    r"vague|dubious[^\]]*|note \d+|further explanation needed)\]",
    re.IGNORECASE,
)
# Markdown table separator rows (---|---|, :---:, | --- | --- |) and empty pipe rows
_TABLE_DIVIDER = re.compile(r"^[|\s:]*-{2,}[|\s:-]*$")
_EMPTY_PIPES = re.compile(r"^[|\s]+$")


class MarkdownCleaner:
    def clean(self, text: str) -> str:
        # Normalize Unicode
        text = unicodedata.normalize("NFKC", text)

        # Remove zero-width and invisible characters
        text = _ZERO_WIDTH.sub("", text)

        # Smart quotes → ASCII
        text = text.translate(_SMART_QUOTES)

        # Strip leading YAML front matter that the markdown extractor prepends
        text = _FRONTMATTER.sub("", text)

        # Remove boilerplate + wiki-markup noise, line by line
        lines = text.splitlines()
        cleaned_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if any(p.search(stripped) for p in _BOILERPLATE_PATTERNS):
                continue
            # Drop markdown table separator rows and empty pipe rows
            if _TABLE_DIVIDER.match(stripped) or _EMPTY_PIPES.match(stripped):
                continue
            # Strip [edit] links and named wiki reference markers inline.
            # (Bare numeric refs like [7] are left intact — they could be code
            # array indices, which matters for the coding-copilot use case.)
            stripped = _EDIT_MARKER.sub("", stripped)
            stripped = _REF_MARKERS.sub("", stripped)
            cleaned_lines.append(stripped.strip())

        text = "\n".join(cleaned_lines)

        # Collapse repeated punctuation
        text = _REPEATED_PUNCT.sub(r"\1", text)

        # Max 2 consecutive blank lines
        text = _EXCESS_BLANK_LINES.sub("\n\n", text)

        return text.strip()

    def estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)
