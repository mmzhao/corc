"""Tests for knowledge store (FTS5)."""

from pathlib import Path

from corc.knowledge import KnowledgeStore


def test_add_and_search(tmp_path):
    ks = KnowledgeStore(tmp_path / "knowledge", tmp_path / "knowledge.db")

    # Create a document
    doc_path = tmp_path / "knowledge" / "test.md"
    doc_path.write_text("""---
id: test-1
type: decision
tags: [sqlite, search]
---

# SQLite FTS5 Decision

We chose SQLite with FTS5 for full-text search because it's zero-ops and fast.
""")
    doc_id = ks.add(file_path=doc_path)
    assert doc_id == "test-1"

    results = ks.search("SQLite FTS5")
    assert len(results) > 0
    assert results[0]["id"] == "test-1"


def test_add_from_content(tmp_path):
    ks = KnowledgeStore(tmp_path / "knowledge", tmp_path / "knowledge.db")
    doc_id = ks.add(content="# Test Note\n\nSome content about testing.", doc_type="note")
    assert doc_id is not None

    doc = ks.get(doc_id)
    assert doc is not None
    assert doc["title"] == "Test Note"


def test_search_no_results(tmp_path):
    ks = KnowledgeStore(tmp_path / "knowledge", tmp_path / "knowledge.db")
    results = ks.search("nonexistent query")
    assert results == []


def test_reindex(tmp_path):
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "doc1.md").write_text("---\nid: d1\ntype: note\n---\n# Doc One\nFirst document.")
    (knowledge_dir / "doc2.md").write_text("---\nid: d2\ntype: note\n---\n# Doc Two\nSecond document.")

    ks = KnowledgeStore(knowledge_dir, tmp_path / "knowledge.db")
    ks.reindex()

    stats = ks.stats()
    assert stats["total"] == 2


def test_stats(tmp_path):
    ks = KnowledgeStore(tmp_path / "knowledge", tmp_path / "knowledge.db")
    ks.add(content="# Note\nA note.", doc_type="note")
    ks.add(content="# Decision\nA decision.", doc_type="decision")

    stats = ks.stats()
    assert stats["total"] == 2


def test_list_docs(tmp_path):
    ks = KnowledgeStore(tmp_path / "knowledge", tmp_path / "knowledge.db")
    ks.add(content="# Note\nContent.", doc_type="note")
    ks.add(content="# Decision\nContent.", doc_type="decision")

    all_docs = ks.list_docs()
    assert len(all_docs) == 2

    notes = ks.list_docs(doc_type="note")
    assert len(notes) == 1
