"""Tests for index freshness checking.

Verifies that search results are always up to date by detecting stale
documents (via file mtime comparison) and reindexing them before returning
search results.
"""

import os
import time
from pathlib import Path

from corc.knowledge import KnowledgeStore, _content_hash


def _make_doc(knowledge_dir: Path, filename: str, doc_id: str, body: str) -> Path:
    """Helper to create a markdown file with frontmatter."""
    path = knowledge_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"---\nid: {doc_id}\ntype: note\n---\n\n{body}\n"
    path.write_text(content)
    return path


def test_mtime_stored_on_add(tmp_path):
    """file_mtime is stored in the documents table when a doc is added."""
    ks = KnowledgeStore(tmp_path / "knowledge", tmp_path / "knowledge.db")
    doc_path = _make_doc(tmp_path / "knowledge", "test.md", "mt-1", "# Mtime Test\n\nContent.")
    ks.add(file_path=doc_path)

    row = ks.conn.execute("SELECT file_mtime FROM documents WHERE id='mt-1'").fetchone()
    assert row is not None
    assert row[0] is not None
    assert isinstance(row[0], float)
    # Stored mtime should be close to the actual file mtime
    actual_mtime = doc_path.stat().st_mtime
    assert abs(row[0] - actual_mtime) < 1.0


def test_stale_doc_reindexed_on_search(tmp_path):
    """Modified file is automatically reindexed when search is called."""
    knowledge_dir = tmp_path / "knowledge"
    ks = KnowledgeStore(knowledge_dir, tmp_path / "knowledge.db")

    # Create and index a document about cats
    doc_path = _make_doc(knowledge_dir, "animal.md", "stale-1",
                         "# Animal Info\n\nCats are wonderful pets.")
    ks.add(file_path=doc_path)

    # Verify initial search works
    results = ks.search("cats")
    assert len(results) == 1
    assert results[0]["id"] == "stale-1"

    # Search for dogs should return nothing
    results = ks.search("dogs")
    assert len(results) == 0

    # Modify the file on disk (change content from cats to dogs)
    # Ensure mtime advances (some filesystems have 1-second resolution)
    time.sleep(0.05)
    doc_path.write_text("---\nid: stale-1\ntype: note\n---\n\n# Animal Info\n\nDogs are loyal companions.\n")
    # Force mtime to be clearly in the future
    future_mtime = time.time() + 10
    os.utime(doc_path, (future_mtime, future_mtime))

    # Search for dogs — freshness check should detect and reindex the stale file
    results = ks.search("dogs")
    assert len(results) == 1
    assert results[0]["id"] == "stale-1"

    # Old content should no longer match
    results = ks.search("cats")
    assert len(results) == 0


def test_new_file_detected_on_search(tmp_path):
    """A new file added to knowledge_dir is auto-indexed on search."""
    knowledge_dir = tmp_path / "knowledge"
    ks = KnowledgeStore(knowledge_dir, tmp_path / "knowledge.db")

    # Index is empty
    assert ks.stats()["total"] == 0

    # Drop a new file into the knowledge directory
    _make_doc(knowledge_dir, "new.md", "new-1", "# New Document\n\nBrand new content about elephants.")

    # Search should find the new doc without explicit add()
    results = ks.search("elephants")
    assert len(results) == 1
    assert results[0]["id"] == "new-1"


def test_deleted_file_removed_on_search(tmp_path):
    """A file removed from disk is cleaned from the index on search."""
    knowledge_dir = tmp_path / "knowledge"
    ks = KnowledgeStore(knowledge_dir, tmp_path / "knowledge.db")

    doc_path = _make_doc(knowledge_dir, "ephemeral.md", "del-1",
                         "# Ephemeral\n\nThis document will be deleted.")
    ks.add(file_path=doc_path)

    # Verify it's indexed
    assert ks.stats()["total"] == 1

    # Delete the file
    doc_path.unlink()

    # Search triggers freshness check which should remove the deleted doc
    results = ks.search("ephemeral")
    assert len(results) == 0
    assert ks.stats()["total"] == 0


def test_unchanged_content_not_reindexed(tmp_path):
    """File with newer mtime but identical content is not fully reindexed."""
    knowledge_dir = tmp_path / "knowledge"
    ks = KnowledgeStore(knowledge_dir, tmp_path / "knowledge.db")

    doc_path = _make_doc(knowledge_dir, "stable.md", "stable-1",
                         "# Stable Doc\n\nThis content stays the same.")
    ks.add(file_path=doc_path)

    # Record the content hash
    row = ks.conn.execute(
        "SELECT content_hash, file_mtime FROM documents WHERE id='stable-1'"
    ).fetchone()
    original_hash = row[0]
    original_mtime = row[1]

    # Touch the file to change mtime without changing content
    time.sleep(0.05)
    content = doc_path.read_text()
    doc_path.write_text(content)
    future_mtime = time.time() + 10
    os.utime(doc_path, (future_mtime, future_mtime))

    # Trigger freshness check
    refreshed = ks._refresh_stale_docs()
    assert refreshed == 0  # Content didn't change, so no reindex

    # Mtime should be updated though
    row = ks.conn.execute(
        "SELECT content_hash, file_mtime FROM documents WHERE id='stable-1'"
    ).fetchone()
    assert row[0] == original_hash  # Hash unchanged
    assert row[1] > original_mtime  # Mtime updated


def test_refresh_stale_docs_incremental(tmp_path):
    """Only stale docs are reindexed, not all docs."""
    knowledge_dir = tmp_path / "knowledge"
    ks = KnowledgeStore(knowledge_dir, tmp_path / "knowledge.db")

    # Create two documents
    doc1 = _make_doc(knowledge_dir, "doc1.md", "inc-1", "# Doc One\n\nFirst document about alpha.")
    doc2 = _make_doc(knowledge_dir, "doc2.md", "inc-2", "# Doc Two\n\nSecond document about beta.")
    ks.add(file_path=doc1)
    ks.add(file_path=doc2)

    # Modify only doc1
    time.sleep(0.05)
    doc1.write_text("---\nid: inc-1\ntype: note\n---\n\n# Doc One\n\nFirst document about gamma.\n")
    future_mtime = time.time() + 10
    os.utime(doc1, (future_mtime, future_mtime))

    refreshed = ks._refresh_stale_docs()
    assert refreshed == 1  # Only doc1 was reindexed

    # Verify doc1 has updated content
    results = ks.search("gamma")
    assert len(results) == 1
    assert results[0]["id"] == "inc-1"

    # doc2 is unchanged
    results = ks.search("beta")
    assert len(results) == 1
    assert results[0]["id"] == "inc-2"


def test_content_hash_comparison(tmp_path):
    """Stale detection uses content_hash to confirm actual content change."""
    knowledge_dir = tmp_path / "knowledge"
    ks = KnowledgeStore(knowledge_dir, tmp_path / "knowledge.db")

    content = "---\nid: hash-1\ntype: note\n---\n\n# Hash Test\n\nOriginal content.\n"
    doc_path = knowledge_dir / "hashtest.md"
    doc_path.write_text(content)
    ks.add(file_path=doc_path)

    stored_hash = ks.conn.execute(
        "SELECT content_hash FROM documents WHERE id='hash-1'"
    ).fetchone()[0]
    assert stored_hash == _content_hash(content)

    # Write different content
    new_content = "---\nid: hash-1\ntype: note\n---\n\n# Hash Test\n\nUpdated content.\n"
    doc_path.write_text(new_content)
    future_mtime = time.time() + 10
    os.utime(doc_path, (future_mtime, future_mtime))

    ks._refresh_stale_docs()

    new_hash = ks.conn.execute(
        "SELECT content_hash FROM documents WHERE id='hash-1'"
    ).fetchone()[0]
    assert new_hash == _content_hash(new_content)
    assert new_hash != stored_hash


def test_mtime_stored_for_content_based_add(tmp_path):
    """file_mtime is also stored when adding via content string (not file_path)."""
    ks = KnowledgeStore(tmp_path / "knowledge", tmp_path / "knowledge.db")
    doc_id = ks.add(content="# Content Add\n\nAdded via content.", doc_type="note")

    row = ks.conn.execute(
        "SELECT file_mtime FROM documents WHERE id=?", (doc_id,)
    ).fetchone()
    assert row is not None
    assert row[0] is not None
    assert isinstance(row[0], float)
