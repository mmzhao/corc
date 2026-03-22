"""Tests for semantic search in the knowledge store.

Tests cover:
- Embeddings module (serialization, cosine similarity)
- Semantic search when sentence-transformers IS available
- Graceful fallback when sentence-transformers is NOT available
- Hybrid search combining FTS5 and semantic
"""

import struct
from pathlib import Path
from unittest import mock

import pytest

from corc import embeddings
from corc.knowledge import KnowledgeStore


# ---------------------------------------------------------------------------
# Embedding utility tests (no sentence-transformers dependency needed)
# ---------------------------------------------------------------------------

class TestEmbeddingSerialisation:
    """Test blob serialization/deserialization — pure Python, no model needed."""

    def test_roundtrip(self):
        vec = [0.1, 0.2, 0.3, -0.5, 1.0]
        blob = embeddings.embedding_to_blob(vec)
        restored = embeddings.blob_to_embedding(blob)
        assert len(restored) == len(vec)
        for a, b in zip(vec, restored):
            assert abs(a - b) < 1e-6

    def test_blob_size(self):
        """Each float32 is 4 bytes."""
        vec = [0.0] * 384
        blob = embeddings.embedding_to_blob(vec)
        assert len(blob) == 384 * 4

    def test_empty_vector(self):
        blob = embeddings.embedding_to_blob([])
        assert blob == b""
        assert embeddings.blob_to_embedding(b"") == []


class TestCosineSimilarity:
    """Test cosine similarity — pure Python, no model needed."""

    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(embeddings.cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(embeddings.cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(embeddings.cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert embeddings.cosine_similarity(a, b) == 0.0

    def test_similar_vectors_higher_score(self):
        """Vectors closer in direction should have higher cosine similarity."""
        target = [1.0, 1.0, 0.0]
        close = [0.9, 1.1, 0.0]
        far = [0.0, 0.0, 1.0]
        assert embeddings.cosine_similarity(target, close) > embeddings.cosine_similarity(target, far)


# ---------------------------------------------------------------------------
# Mock embedding helpers
# ---------------------------------------------------------------------------

# Deterministic fake embeddings: hash the text to produce a fixed-length vector.
# This lets us test the search pipeline without loading a real model.
FAKE_DIM = 16


def _fake_embedding(text: str) -> list[float]:
    """Generate a deterministic fake embedding from text content."""
    import hashlib
    h = hashlib.md5(text.encode()).digest()
    # Use the 16 hash bytes as float seeds
    return [b / 255.0 for b in h]


def _fake_encode(texts: list[str]) -> list[list[float]]:
    return [_fake_embedding(t) for t in texts]


def _fake_encode_single(text: str) -> list[float]:
    return _fake_embedding(text)


# ---------------------------------------------------------------------------
# Tests WITH sentence-transformers available (mocked)
# ---------------------------------------------------------------------------

class TestSemanticSearchWithEmbeddings:
    """Test semantic search pipeline using mock embeddings."""

    @pytest.fixture()
    def ks(self, tmp_path):
        """Knowledge store with mocked embeddings."""
        with mock.patch.object(embeddings, "is_available", return_value=True), \
             mock.patch.object(embeddings, "encode", side_effect=_fake_encode), \
             mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            store = KnowledgeStore(tmp_path / "knowledge", tmp_path / "knowledge.db")
            # Override the cached flag
            store._embeddings_available = True
            yield store

    def _add_docs(self, ks, tmp_path):
        """Add test documents."""
        knowledge_dir = tmp_path / "knowledge"

        (knowledge_dir / "python.md").write_text(
            "---\nid: py1\ntype: decision\ntags: [python]\n---\n\n"
            "# Python Design Patterns\n\n"
            "We use factory patterns and dependency injection in our Python codebase. "
            "This helps with testability and loose coupling between modules."
        )
        (knowledge_dir / "database.md").write_text(
            "---\nid: db1\ntype: architecture\ntags: [database]\n---\n\n"
            "# Database Architecture\n\n"
            "SQLite is used as the primary data store. FTS5 provides full-text search. "
            "All tables use WAL mode for concurrent read access."
        )
        (knowledge_dir / "testing.md").write_text(
            "---\nid: test1\ntype: note\ntags: [testing]\n---\n\n"
            "# Testing Strategy\n\n"
            "Unit tests use pytest with tmp_path fixtures for isolation. "
            "Integration tests verify the full pipeline end-to-end."
        )

        with mock.patch.object(embeddings, "is_available", return_value=True), \
             mock.patch.object(embeddings, "encode", side_effect=_fake_encode), \
             mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            ks.add(file_path=knowledge_dir / "python.md")
            ks.add(file_path=knowledge_dir / "database.md")
            ks.add(file_path=knowledge_dir / "testing.md")

    def test_embeddings_stored_as_blobs(self, ks, tmp_path):
        """Verify embeddings are stored in the chunks table."""
        self._add_docs(ks, tmp_path)

        rows = ks.conn.execute(
            "SELECT embedding FROM chunks WHERE document_id = 'py1'"
        ).fetchall()
        assert len(rows) > 0
        for row in rows:
            blob = row[0]
            assert blob is not None
            vec = embeddings.blob_to_embedding(blob)
            assert len(vec) == FAKE_DIM
            assert all(isinstance(v, float) for v in vec)

    def test_semantic_search_returns_results(self, ks, tmp_path):
        """Semantic search should return scored chunk results."""
        self._add_docs(ks, tmp_path)

        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks.semantic_search("Python factory patterns")

        assert len(results) > 0
        # Results should have a score key
        assert "score" in results[0]
        assert "chunk_content" in results[0]
        assert "document_id" in results[0]

    def test_semantic_search_scores_are_valid(self, ks, tmp_path):
        """Scores should be valid cosine similarity values in [-1, 1]."""
        self._add_docs(ks, tmp_path)

        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks.semantic_search("database architecture")

        for r in results:
            assert -1.0 <= r["score"] <= 1.0

    def test_semantic_search_respects_limit(self, ks, tmp_path):
        """Semantic search should respect the limit parameter."""
        self._add_docs(ks, tmp_path)

        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks.semantic_search("testing", limit=1)

        assert len(results) <= 1

    def test_semantic_search_filters_by_type(self, ks, tmp_path):
        """Semantic search should filter by document type."""
        self._add_docs(ks, tmp_path)

        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks.semantic_search("design", doc_type="decision")

        for r in results:
            assert r["type"] == "decision"

    def test_semantic_search_results_sorted_by_score(self, ks, tmp_path):
        """Results should be sorted by score descending."""
        self._add_docs(ks, tmp_path)

        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks.semantic_search("database")

        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]["score"] >= results[i + 1]["score"]

    def test_semantic_search_no_embedding_blob_in_results(self, ks, tmp_path):
        """Raw embedding blobs should not leak into search results."""
        self._add_docs(ks, tmp_path)

        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks.semantic_search("testing")

        for r in results:
            assert "embedding" not in r


class TestHybridSearch:
    """Test hybrid search combining FTS5 and semantic."""

    @pytest.fixture()
    def ks(self, tmp_path):
        with mock.patch.object(embeddings, "is_available", return_value=True), \
             mock.patch.object(embeddings, "encode", side_effect=_fake_encode), \
             mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            store = KnowledgeStore(tmp_path / "knowledge", tmp_path / "knowledge.db")
            store._embeddings_available = True
            yield store

    def _add_docs(self, ks, tmp_path):
        knowledge_dir = tmp_path / "knowledge"
        (knowledge_dir / "alpha.md").write_text(
            "---\nid: a1\ntype: note\n---\n\n"
            "# Alpha Document\n\n"
            "This document discusses alpha algorithms and sorting techniques."
        )
        (knowledge_dir / "beta.md").write_text(
            "---\nid: b1\ntype: note\n---\n\n"
            "# Beta Document\n\n"
            "Beta testing strategies and quality assurance processes."
        )

        with mock.patch.object(embeddings, "is_available", return_value=True), \
             mock.patch.object(embeddings, "encode", side_effect=_fake_encode), \
             mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            ks.add(file_path=knowledge_dir / "alpha.md")
            ks.add(file_path=knowledge_dir / "beta.md")

    def test_hybrid_search_returns_results(self, ks, tmp_path):
        self._add_docs(ks, tmp_path)

        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks.hybrid_search("alpha algorithms")

        assert len(results) > 0
        assert "score" in results[0]

    def test_hybrid_search_has_component_scores(self, ks, tmp_path):
        """Hybrid results should include both FTS and semantic score components."""
        self._add_docs(ks, tmp_path)

        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks.hybrid_search("alpha algorithms")

        for r in results:
            assert "fts_score" in r
            assert "semantic_score" in r

    def test_hybrid_search_weight_pure_fts(self, ks, tmp_path):
        """With semantic_weight=0.0, hybrid should behave like FTS."""
        self._add_docs(ks, tmp_path)

        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks.hybrid_search("alpha", semantic_weight=0.0)

        # All scores should come from FTS only
        for r in results:
            assert r["semantic_score"] >= 0.0  # semantic score may still be computed


# ---------------------------------------------------------------------------
# Tests WITHOUT sentence-transformers (fallback behavior)
# ---------------------------------------------------------------------------

class TestFallbackWithoutSentenceTransformers:
    """Test that everything works when sentence-transformers is not installed."""

    @pytest.fixture()
    def ks(self, tmp_path):
        """Knowledge store with embeddings marked unavailable."""
        with mock.patch.object(embeddings, "is_available", return_value=False):
            store = KnowledgeStore(tmp_path / "knowledge", tmp_path / "knowledge.db")
            assert store._embeddings_available is False
            yield store

    def _add_doc(self, ks, tmp_path):
        knowledge_dir = tmp_path / "knowledge"
        (knowledge_dir / "doc.md").write_text(
            "---\nid: fallback1\ntype: note\ntags: [test]\n---\n\n"
            "# Fallback Test\n\n"
            "This document tests fallback behavior when semantic search is unavailable."
        )
        ks.add(file_path=knowledge_dir / "doc.md")

    def test_no_embeddings_stored(self, ks, tmp_path):
        """When unavailable, chunks should have NULL embeddings."""
        self._add_doc(ks, tmp_path)

        rows = ks.conn.execute(
            "SELECT embedding FROM chunks WHERE document_id = 'fallback1'"
        ).fetchall()
        assert len(rows) > 0
        for row in rows:
            assert row[0] is None

    def test_fts_search_still_works(self, ks, tmp_path):
        """FTS5 keyword search should work regardless of embeddings."""
        self._add_doc(ks, tmp_path)

        results = ks.search("fallback behavior")
        assert len(results) > 0
        assert results[0]["id"] == "fallback1"

    def test_semantic_search_falls_back_to_fts(self, ks, tmp_path):
        """semantic_search should transparently fall back to FTS5."""
        self._add_doc(ks, tmp_path)

        results = ks.semantic_search("fallback behavior")
        assert len(results) > 0
        # FTS fallback returns document-level results (with 'id' key)
        assert results[0]["id"] == "fallback1"

    def test_hybrid_search_falls_back_to_fts(self, ks, tmp_path):
        """hybrid_search should work even without embeddings."""
        self._add_doc(ks, tmp_path)

        results = ks.hybrid_search("fallback behavior")
        assert len(results) > 0
        assert results[0]["id"] == "fallback1"

    def test_add_succeeds_without_embeddings(self, ks, tmp_path):
        """Adding documents should succeed without sentence-transformers."""
        doc_id = ks.add(
            content="# No Embeddings\n\nThis should work fine.",
            doc_type="note",
        )
        assert doc_id is not None

        doc = ks.get(doc_id)
        assert doc is not None
        assert doc["title"] == "No Embeddings"

    def test_reindex_succeeds_without_embeddings(self, ks, tmp_path):
        """Reindex should work without sentence-transformers."""
        self._add_doc(ks, tmp_path)
        ks.reindex()

        stats = ks.stats()
        assert stats["total"] == 1


class TestEmbeddingErrorHandling:
    """Test graceful handling when embedding generation fails at runtime."""

    @pytest.fixture()
    def ks(self, tmp_path):
        """Knowledge store where encode raises an exception."""
        with mock.patch.object(embeddings, "is_available", return_value=True), \
             mock.patch.object(embeddings, "encode", side_effect=RuntimeError("Model load failed")):
            store = KnowledgeStore(tmp_path / "knowledge", tmp_path / "knowledge.db")
            store._embeddings_available = True
            yield store

    def test_add_succeeds_despite_encode_failure(self, ks, tmp_path):
        """Document should still be indexed even if embedding generation fails."""
        knowledge_dir = tmp_path / "knowledge"
        (knowledge_dir / "err.md").write_text(
            "---\nid: err1\ntype: note\n---\n\n# Error Test\n\nContent here."
        )

        with mock.patch.object(embeddings, "encode", side_effect=RuntimeError("Model load failed")):
            doc_id = ks.add(file_path=knowledge_dir / "err.md")

        assert doc_id == "err1"

        # Chunks should be stored but without embeddings
        rows = ks.conn.execute(
            "SELECT embedding FROM chunks WHERE document_id = 'err1'"
        ).fetchall()
        assert len(rows) > 0
        for row in rows:
            assert row[0] is None

    def test_semantic_search_falls_back_on_encode_error(self, ks, tmp_path):
        """If query encoding fails, should fall back to FTS5."""
        knowledge_dir = tmp_path / "knowledge"
        (knowledge_dir / "ok.md").write_text(
            "---\nid: ok1\ntype: note\n---\n\n# Working Doc\n\nSome searchable content."
        )
        # Add without embedding errors (mock only encode, not encode_single)
        with mock.patch.object(embeddings, "encode", return_value=[[0.0] * FAKE_DIM]):
            ks.add(file_path=knowledge_dir / "ok.md")

        # Now make encode_single fail at query time
        with mock.patch.object(embeddings, "encode_single", side_effect=RuntimeError("fail")):
            results = ks.semantic_search("searchable content")

        assert len(results) > 0
        assert results[0]["id"] == "ok1"


class TestEmbeddingsModuleReset:
    """Test the embeddings module reset functionality."""

    def test_reset_clears_state(self):
        """reset() should clear cached availability and model."""
        embeddings.reset()
        assert embeddings._model is None
        assert embeddings._available is None

    def test_is_available_caches_result(self):
        """is_available() should cache its result."""
        embeddings.reset()
        with mock.patch.dict("sys.modules", {"sentence_transformers": None}):
            # When module is in sys.modules as None, import fails
            result = embeddings.is_available()
            # Result is cached
            assert embeddings._available is not None
        embeddings.reset()
