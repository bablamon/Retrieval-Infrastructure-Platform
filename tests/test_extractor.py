import pytest

from app.extractor.bs4_extractor import BS4Extractor
from app.extractor.chunker import SemanticChunker
from app.extractor.markdown_cleaner import MarkdownCleaner
from app.extractor.trafilatura_extractor import TrafilaturaExtractor


@pytest.mark.asyncio
async def test_trafilatura_extracts_article(sample_html):
    extractor = TrafilaturaExtractor()
    result = await extractor.extract(sample_html, url="https://example.com/article")
    assert result is not None
    assert len(result.text) > 100
    assert "Transformer" in result.text


@pytest.mark.asyncio
async def test_bs4_fallback_extracts_text(sample_html):
    extractor = BS4Extractor()
    result = await extractor.extract(sample_html, url="https://example.com/article")
    assert result is not None
    assert "Transformer" in result.text
    assert "Cookie policy" not in result.text  # footer removed


@pytest.mark.asyncio
async def test_bs4_extract_links(sample_html):
    extractor = BS4Extractor()
    links = await extractor.extract_links(sample_html, base_url="https://example.com/article")
    # same-domain only: '/' resolves to https://example.com/
    assert all("example.com" in link for link in links)


def test_markdown_cleaner_removes_boilerplate():
    cleaner = MarkdownCleaner()
    text = "Real content here.\n\nSubscribe to our newsletter\n\nMore real content."
    cleaned = cleaner.clean(text)
    assert "Subscribe to our newsletter" not in cleaned
    assert "Real content here." in cleaned


def test_markdown_cleaner_collapses_blank_lines():
    cleaner = MarkdownCleaner()
    text = "Line 1\n\n\n\n\nLine 2"
    cleaned = cleaner.clean(text)
    assert "\n\n\n" not in cleaned


def test_markdown_cleaner_strips_wiki_artifacts():
    cleaner = MarkdownCleaner()
    text = (
        "## Data types\n\n"
        "[edit]ABAP provides built-in types.[citation needed]\n\n"
        "Type | Description |\n"
        "---|---|\n"
        "I | Integer |"
    )
    cleaned = cleaner.clean(text)
    assert "[edit]" not in cleaned
    assert "[citation needed]" not in cleaned
    assert "---|---|" not in cleaned          # table separator removed
    assert "I | Integer |" in cleaned         # table data rows preserved
    assert "ABAP provides built-in types." in cleaned


def test_markdown_cleaner_strips_frontmatter():
    cleaner = MarkdownCleaner()
    text = "---\ntitle: ABAP - Wikipedia\nurl: https://x\n---\n# ABAP\n\nReal content here."
    cleaned = cleaner.clean(text)
    assert "title: ABAP" not in cleaned
    assert "Real content here." in cleaned


def test_chunker_basic():
    chunker = SemanticChunker(chunk_size=50, chunk_overlap=10)
    text = " ".join(["word"] * 200)
    chunks = chunker.chunk(text, {"url": "https://example.com", "title": "Test"})
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["text"]
        assert "url" in chunk
        assert "chunk_index" in chunk
        assert "total_chunks" in chunk


def test_chunker_overlap():
    chunker = SemanticChunker(chunk_size=20, chunk_overlap=5)
    # Two paragraphs, each 25 words, will be split with overlap
    para1 = " ".join([f"word{i}" for i in range(25)])
    para2 = " ".join([f"other{i}" for i in range(25)])
    text = f"{para1}\n\n{para2}"
    chunks = chunker.chunk(text, {"url": "https://example.com"})
    if len(chunks) >= 2:
        # Last 5 words of chunk 0 should appear at start of chunk 1
        last_words_of_0 = chunks[0]["text"].split()[-5:]
        start_of_1 = chunks[1]["text"].split()[:5]
        assert any(w in start_of_1 for w in last_words_of_0)


def test_chunker_metadata_preserved():
    chunker = SemanticChunker(chunk_size=50)
    text = " ".join(["word"] * 100)
    metadata = {"url": "https://example.com", "title": "My Title", "extra_key": "extra_value"}
    chunks = chunker.chunk(text, metadata)
    for chunk in chunks:
        assert chunk["url"] == "https://example.com"
        assert chunk["title"] == "My Title"


def test_estimate_tokens():
    cleaner = MarkdownCleaner()
    text = "word " * 100
    tokens = cleaner.estimate_tokens(text)
    assert 120 <= tokens <= 140  # ~1.3x word count


def test_chunker_respects_token_counter():
    """With a token_counter that reports 3x the word count, the chunker
    should produce roughly 3x as many splits for the same chunk_size."""
    def fake_counter(text: str) -> int:
        return len(text.split()) * 3

    baseline = SemanticChunker(chunk_size=30, chunk_overlap=0)
    token_aware = SemanticChunker(chunk_size=30, chunk_overlap=0, token_counter=fake_counter)

    text = " ".join(["word"] * 300)
    baseline_chunks = baseline.chunk(text, {"url": "https://example.com"})
    token_chunks = token_aware.chunk(text, {"url": "https://example.com"})

    # With 3x token inflation, the token-aware chunker needs roughly 3x as
    # many chunks to stay under the same chunk_size cap.
    assert len(token_chunks) >= 2 * len(baseline_chunks)


def test_chunker_oversize_sentence_hard_split():
    """A single sentence longer than chunk_size must still be split — no
    chunk should exceed the token budget."""
    def word_counter(text: str) -> int:
        return len(text.split())

    chunker = SemanticChunker(chunk_size=10, chunk_overlap=0, token_counter=word_counter)
    # No sentence boundaries, all one long stream of words
    text = " ".join([f"w{i}" for i in range(50)])
    chunks = chunker.chunk(text, {"url": "https://example.com"})
    for chunk in chunks:
        assert word_counter(chunk["text"]) <= 10
