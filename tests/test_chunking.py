"""Tests for content chunking in the knowledge store."""

from pathlib import Path

from corc.knowledge import (
    KnowledgeStore,
    chunk_markdown,
    estimate_tokens,
    _split_by_headings,
    _split_long_text,
    TARGET_TOKENS,
)


# --- Unit tests for estimate_tokens ---


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_single_word():
    assert estimate_tokens("hello") == 1  # int(1 * 1.3) = 1


def test_estimate_tokens_sentence():
    text = "The quick brown fox jumps over the lazy dog"
    # 9 words * 1.3 = 11.7 -> 11
    assert estimate_tokens(text) == 11


def test_estimate_tokens_whitespace_only():
    assert estimate_tokens("   \n\t  ") == 0


# --- Unit tests for _split_by_headings ---


def test_split_no_headings():
    body = "Just some plain text without any headings.\n\nAnother paragraph."
    sections = _split_by_headings(body)
    assert len(sections) == 1
    assert sections[0]["heading"] is None
    assert sections[0]["level"] == 0
    assert "plain text" in sections[0]["content"]


def test_split_single_heading():
    body = "# My Title\n\nSome content here."
    sections = _split_by_headings(body)
    assert len(sections) == 1
    assert sections[0]["heading"] == "My Title"
    assert sections[0]["level"] == 1
    assert "Some content here." in sections[0]["content"]


def test_split_multiple_headings():
    body = "# Title\n\nIntro text.\n\n## Section One\n\nFirst section.\n\n## Section Two\n\nSecond section."
    sections = _split_by_headings(body)
    assert len(sections) == 3
    assert sections[0]["heading"] == "Title"
    assert sections[0]["level"] == 1
    assert sections[1]["heading"] == "Section One"
    assert sections[1]["level"] == 2
    assert sections[2]["heading"] == "Section Two"
    assert sections[2]["level"] == 2


def test_split_content_before_first_heading():
    body = "Some preamble text.\n\n# First Heading\n\nBody content."
    sections = _split_by_headings(body)
    assert len(sections) == 2
    assert sections[0]["heading"] is None
    assert "preamble" in sections[0]["content"]
    assert sections[1]["heading"] == "First Heading"


def test_split_nested_headings():
    body = (
        "# Top Level\n\nIntro.\n\n"
        "## Sub Section\n\nSub content.\n\n"
        "### Sub Sub\n\nDeep content.\n\n"
        "## Another Sub\n\nMore content."
    )
    sections = _split_by_headings(body)
    assert len(sections) == 4
    assert sections[0]["level"] == 1
    assert sections[1]["level"] == 2
    assert sections[2]["level"] == 3
    assert sections[3]["level"] == 2


def test_split_heading_only_no_content():
    body = "# Just a Heading"
    sections = _split_by_headings(body)
    assert len(sections) == 1
    assert sections[0]["heading"] == "Just a Heading"
    assert sections[0]["content"] == ""


# --- Unit tests for _split_long_text ---


def test_split_long_text_short_input():
    text = "Short text."
    pieces = _split_long_text(text, target=500)
    assert len(pieces) == 1
    assert pieces[0] == "Short text."


def test_split_long_text_by_paragraphs():
    # Create text with multiple paragraphs that together exceed target
    para = "word " * 200  # ~260 tokens per paragraph
    text = f"{para.strip()}\n\n{para.strip()}\n\n{para.strip()}"
    pieces = _split_long_text(text, target=500)
    assert len(pieces) >= 2
    for piece in pieces:
        assert estimate_tokens(piece) <= 600  # some tolerance


def test_split_long_text_single_huge_paragraph():
    # One paragraph with many sentences
    sentences = ["This is sentence number %d." % i for i in range(200)]
    text = " ".join(sentences)
    pieces = _split_long_text(text, target=500)
    assert len(pieces) > 1
    for piece in pieces:
        # Each piece should be roughly at or below target
        assert estimate_tokens(piece) <= 600


# --- Unit tests for chunk_markdown ---


def test_chunk_simple_document():
    body = "# My Document\n\nSome content about the document."
    chunks = chunk_markdown(body)
    assert len(chunks) >= 1
    assert chunks[0]["heading"] == "My Document"
    assert "My Document" in chunks[0]["content"]
    assert "Some content" in chunks[0]["content"]
    assert chunks[0]["token_estimate"] > 0


def test_chunk_no_headings():
    body = "Just plain text without any headings.\n\nAnother paragraph of plain text."
    chunks = chunk_markdown(body)
    assert len(chunks) == 1
    assert chunks[0]["heading"] is None
    assert "plain text" in chunks[0]["content"]


def test_chunk_multiple_sections():
    body = (
        "# Introduction\n\nThis is the intro.\n\n"
        "## Background\n\nSome background info.\n\n"
        "## Methodology\n\nHow we did things.\n\n"
        "## Results\n\nWhat we found."
    )
    chunks = chunk_markdown(body)
    assert len(chunks) == 4
    headings = [c["heading"] for c in chunks]
    assert "Introduction" in headings
    assert "Background" in headings
    assert "Methodology" in headings
    assert "Results" in headings


def test_chunk_nested_headings():
    body = (
        "# Top\n\nTop level content.\n\n"
        "## Sub A\n\nSub A content.\n\n"
        "### Sub A Detail\n\nDetailed content.\n\n"
        "## Sub B\n\nSub B content."
    )
    chunks = chunk_markdown(body)
    assert len(chunks) == 4
    # Each heading section becomes its own chunk
    assert chunks[0]["heading"] == "Top"
    assert chunks[1]["heading"] == "Sub A"
    assert chunks[2]["heading"] == "Sub A Detail"
    assert chunks[3]["heading"] == "Sub B"


def test_chunk_very_long_section():
    """A single heading section with content exceeding 500 tokens should be split."""
    long_content = " ".join(["word"] * 1000)  # ~1300 tokens
    body = f"# Long Section\n\n{long_content}"
    chunks = chunk_markdown(body, target_tokens=500)
    assert len(chunks) > 1
    # First chunk should have the heading
    assert chunks[0]["heading"] == "Long Section"
    assert "# Long Section" in chunks[0]["content"]
    # Subsequent chunks should reference continuation
    for chunk in chunks[1:]:
        assert "continued" in chunk["content"]


def test_chunk_empty_body():
    chunks = chunk_markdown("")
    assert len(chunks) == 0


def test_chunk_whitespace_only():
    chunks = chunk_markdown("   \n\n  \t  ")
    assert len(chunks) == 0


def test_chunk_heading_only():
    body = "# Just a Title"
    chunks = chunk_markdown(body)
    assert len(chunks) == 1
    assert chunks[0]["heading"] == "Just a Title"
    assert "# Just a Title" in chunks[0]["content"]


def test_chunk_content_before_heading():
    body = "Preamble text here.\n\n# First Section\n\nSection content."
    chunks = chunk_markdown(body)
    assert len(chunks) == 2
    assert chunks[0]["heading"] is None
    assert "Preamble" in chunks[0]["content"]
    assert chunks[1]["heading"] == "First Section"


def test_chunk_token_estimates_reasonable():
    """Each chunk's token_estimate should match estimate_tokens on its content."""
    body = (
        "# Section One\n\nContent for section one.\n\n"
        "## Section Two\n\nContent for section two.\n\n"
        "## Section Three\n\nContent for section three."
    )
    chunks = chunk_markdown(body)
    for chunk in chunks:
        assert chunk["token_estimate"] == estimate_tokens(chunk["content"])


def test_chunk_preserves_heading_level():
    body = "### H3 Heading\n\nSome content."
    chunks = chunk_markdown(body)
    assert len(chunks) == 1
    assert chunks[0]["content"].startswith("### H3 Heading")


# --- Integration tests: chunks in KnowledgeStore ---


def test_add_populates_chunks(tmp_path):
    ks = KnowledgeStore(tmp_path / "knowledge", tmp_path / "knowledge.db")
    doc_id = ks.add(
        content="# My Doc\n\nSome content.\n\n## Section\n\nMore content.",
        doc_type="note",
    )

    chunks = ks.get_chunks(doc_id)
    assert len(chunks) == 2
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["chunk_index"] == 1
    assert chunks[0]["document_id"] == doc_id
    assert chunks[1]["document_id"] == doc_id


def test_add_from_file_populates_chunks(tmp_path):
    ks = KnowledgeStore(tmp_path / "knowledge", tmp_path / "knowledge.db")

    doc_path = tmp_path / "knowledge" / "test.md"
    doc_path.write_text(
        "---\nid: chunk-test\ntype: note\ntags: [test]\n---\n\n"
        "# Document Title\n\nIntro paragraph.\n\n"
        "## Details\n\nDetail content.\n\n"
        "## Conclusion\n\nFinal thoughts."
    )

    doc_id = ks.add(file_path=doc_path)
    assert doc_id == "chunk-test"

    chunks = ks.get_chunks(doc_id)
    assert len(chunks) == 3
    headings = [c["heading"] for c in chunks]
    assert "Document Title" in headings
    assert "Details" in headings
    assert "Conclusion" in headings


def test_reindex_repopulates_chunks(tmp_path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()

    (knowledge_dir / "doc1.md").write_text(
        "---\nid: d1\ntype: note\n---\n# Doc One\n\nContent one.\n\n## Sub\n\nSub content."
    )
    (knowledge_dir / "doc2.md").write_text(
        "---\nid: d2\ntype: note\n---\n# Doc Two\n\nContent two."
    )

    ks = KnowledgeStore(knowledge_dir, tmp_path / "knowledge.db")
    ks.reindex()

    chunks_d1 = ks.get_chunks("d1")
    chunks_d2 = ks.get_chunks("d2")
    assert len(chunks_d1) == 2  # title section + sub section
    assert len(chunks_d2) == 1  # single section


def test_readd_replaces_chunks(tmp_path):
    """Re-adding a document should replace its chunks."""
    ks = KnowledgeStore(tmp_path / "knowledge", tmp_path / "knowledge.db")

    doc_path = tmp_path / "knowledge" / "evolving.md"
    doc_path.write_text("---\nid: evolve\ntype: note\n---\n# Version 1\n\nOriginal.")
    ks.add(file_path=doc_path)

    chunks_v1 = ks.get_chunks("evolve")
    assert len(chunks_v1) == 1

    # Update the document
    doc_path.write_text(
        "---\nid: evolve\ntype: note\n---\n# Version 2\n\nUpdated.\n\n## New Section\n\nNew content."
    )
    ks.add(file_path=doc_path)

    chunks_v2 = ks.get_chunks("evolve")
    assert len(chunks_v2) == 2
    assert "Version 2" in chunks_v2[0]["content"]


def test_chunks_for_document_without_headings(tmp_path):
    ks = KnowledgeStore(tmp_path / "knowledge", tmp_path / "knowledge.db")
    doc_id = ks.add(
        content="Just plain text without any markdown headings.",
        doc_type="note",
    )

    chunks = ks.get_chunks(doc_id)
    assert len(chunks) == 1
    assert chunks[0]["heading"] is None
    assert "plain text" in chunks[0]["content"]


def test_chunks_for_long_document(tmp_path):
    """A document with a very long section should produce multiple chunks."""
    ks = KnowledgeStore(tmp_path / "knowledge", tmp_path / "knowledge.db")

    long_content = "\n\n".join(
        [f"This is paragraph {i} with enough words to contribute to the token count significantly." for i in range(80)]
    )
    content = f"# Long Doc\n\n{long_content}"
    doc_id = ks.add(content=content, doc_type="note")

    chunks = ks.get_chunks(doc_id)
    assert len(chunks) > 1
    # All chunks should have reasonable token estimates
    for chunk in chunks:
        assert chunk["token_estimate"] > 0
        assert chunk["token_estimate"] <= 700  # allow some margin


def test_get_chunks_nonexistent_doc(tmp_path):
    ks = KnowledgeStore(tmp_path / "knowledge", tmp_path / "knowledge.db")
    chunks = ks.get_chunks("nonexistent")
    assert chunks == []


def test_chunks_indexes_sequential(tmp_path):
    """Chunk indexes should be sequential starting from 0."""
    ks = KnowledgeStore(tmp_path / "knowledge", tmp_path / "knowledge.db")
    content = (
        "# Section 1\n\nContent 1.\n\n"
        "## Section 2\n\nContent 2.\n\n"
        "## Section 3\n\nContent 3.\n\n"
        "### Section 4\n\nContent 4."
    )
    doc_id = ks.add(content=content, doc_type="note")
    chunks = ks.get_chunks(doc_id)
    indexes = [c["chunk_index"] for c in chunks]
    assert indexes == list(range(len(chunks)))


def test_existing_tests_still_pass_with_chunks(tmp_path):
    """Ensure adding chunks doesn't break existing add/search workflow."""
    ks = KnowledgeStore(tmp_path / "knowledge", tmp_path / "knowledge.db")

    doc_path = tmp_path / "knowledge" / "test.md"
    doc_path.write_text("""---
id: compat-1
type: decision
tags: [sqlite, search]
---

# SQLite FTS5 Decision

We chose SQLite with FTS5 for full-text search because it's zero-ops and fast.
""")
    doc_id = ks.add(file_path=doc_path)
    assert doc_id == "compat-1"

    results = ks.search("SQLite FTS5")
    assert len(results) > 0
    assert results[0]["id"] == "compat-1"

    # Chunks should also be present
    chunks = ks.get_chunks(doc_id)
    assert len(chunks) >= 1
