"""Tests for seeded knowledge store documents and template exclusion."""

import sqlite3
from pathlib import Path

import pytest

from corc.knowledge import KnowledgeStore


@pytest.fixture
def real_knowledge_store():
    """Use the actual knowledge store from the project."""
    from corc.config import get_paths
    paths = get_paths()
    ks = KnowledgeStore(paths["knowledge_dir"], paths["knowledge_db"])
    # Ensure index is fresh
    ks.reindex()
    return ks


class TestSeedDocumentsExist:
    """Verify all required seed documents are indexed."""

    def test_stats_shows_five_plus_documents(self, real_knowledge_store):
        stats = real_knowledge_store.stats()
        assert stats["total"] >= 5, f"Expected 5+ documents, got {stats['total']}"

    def test_architecture_doc_exists(self, real_knowledge_store):
        doc = real_knowledge_store.get("corc-spec-v05")
        assert doc is not None
        assert doc["type"] == "architecture"
        assert "Orchestration System Specification" in doc["title"]

    def test_language_choice_decision_exists(self, real_knowledge_store):
        doc = real_knowledge_store.get("decision-language-python")
        assert doc is not None
        assert doc["type"] == "decision"
        assert "Python" in doc["title"]

    def test_data_architecture_decision_exists(self, real_knowledge_store):
        doc = real_knowledge_store.get("decision-three-layer-data")
        assert doc is not None
        assert doc["type"] == "decision"
        assert "Data Architecture" in doc["title"]

    def test_determinism_decision_exists(self, real_knowledge_store):
        doc = real_knowledge_store.get("decision-determinism-model")
        assert doc is not None
        assert doc["type"] == "decision"
        assert "Determinism" in doc["title"]

    def test_phase0_learnings_exists(self, real_knowledge_store):
        doc = real_knowledge_store.get("research-phase-0-learnings")
        assert doc is not None
        assert doc["type"] == "research"
        assert "Phase 0" in doc["title"]

    def test_document_type_counts(self, real_knowledge_store):
        stats = real_knowledge_store.stats()
        assert stats["by_type"].get("architecture", 0) >= 1
        assert stats["by_type"].get("decision", 0) >= 3
        assert stats["by_type"].get("research", 0) >= 1


class TestSeedDocumentSearch:
    """Verify search returns relevant results for key queries."""

    def test_search_python_language(self, real_knowledge_store):
        results = real_knowledge_store.search("Python language choice")
        assert len(results) > 0
        ids = [r["id"] for r in results]
        assert "decision-language-python" in ids

    def test_search_data_layers(self, real_knowledge_store):
        results = real_knowledge_store.search("data architecture layers")
        assert len(results) > 0
        ids = [r["id"] for r in results]
        assert "decision-three-layer-data" in ids

    def test_search_determinism(self, real_knowledge_store):
        results = real_knowledge_store.search("determinism hooks DAG")
        assert len(results) > 0
        ids = [r["id"] for r in results]
        assert "decision-determinism-model" in ids

    def test_search_phase0(self, real_knowledge_store):
        results = real_knowledge_store.search("Phase 0 learnings bootstrap")
        assert len(results) > 0
        ids = [r["id"] for r in results]
        assert "research-phase-0-learnings" in ids

    def test_search_knowledge_store(self, real_knowledge_store):
        results = real_knowledge_store.search("knowledge store")
        assert len(results) > 0
        # Should find the spec and the data architecture decision
        ids = [r["id"] for r in results]
        assert any(doc_id in ids for doc_id in ["corc-spec-v05", "decision-three-layer-data"])

    def test_search_mutation_log(self, real_knowledge_store):
        results = real_knowledge_store.search("mutation log source of truth")
        assert len(results) > 0
        ids = [r["id"] for r in results]
        assert "decision-three-layer-data" in ids

    def test_search_sqlite_fts5(self, real_knowledge_store):
        results = real_knowledge_store.search("FTS5 sqlite search")
        assert len(results) > 0

    def test_search_by_type_filter(self, real_knowledge_store):
        results = real_knowledge_store.search("Python", doc_type="decision")
        assert len(results) > 0
        for r in results:
            assert r["type"] == "decision"


class TestTemplateExclusion:
    """Verify _templates/ directory files are not indexed."""

    def test_templates_not_indexed_on_reindex(self, tmp_path):
        """Templates in _templates/ should be skipped during reindex."""
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()

        # Create a real document
        decisions_dir = knowledge_dir / "decisions"
        decisions_dir.mkdir()
        (decisions_dir / "real-doc.md").write_text(
            "---\nid: real-1\ntype: decision\n---\n# Real Decision\nActual content."
        )

        # Create a template in _templates/
        templates_dir = knowledge_dir / "_templates"
        templates_dir.mkdir()
        (templates_dir / "decision.md").write_text(
            "---\nid: ${id}\ntype: decision\n---\n# ${title}\nTemplate content."
        )

        ks = KnowledgeStore(knowledge_dir, tmp_path / "knowledge.db")
        ks.reindex()

        stats = ks.stats()
        assert stats["total"] == 1, f"Expected 1 doc (template excluded), got {stats['total']}"
        assert ks.get("real-1") is not None
        assert ks.get("${id}") is None

    def test_templates_not_indexed_on_refresh(self, tmp_path):
        """Templates should be skipped during incremental refresh."""
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()

        # Create a template
        templates_dir = knowledge_dir / "_templates"
        templates_dir.mkdir()
        (templates_dir / "research.md").write_text(
            "---\nid: ${id}\ntype: research\n---\n# ${title}\nTemplate."
        )

        ks = KnowledgeStore(knowledge_dir, tmp_path / "knowledge.db")

        # Search triggers refresh — template should not be picked up
        results = ks.search("template")
        assert len(results) == 0

    def test_nested_underscore_dirs_excluded(self, tmp_path):
        """Any directory starting with _ should be excluded."""
        knowledge_dir = tmp_path / "knowledge"
        knowledge_dir.mkdir()

        # Document in _hidden/subdir/
        hidden_dir = knowledge_dir / "_hidden" / "subdir"
        hidden_dir.mkdir(parents=True)
        (hidden_dir / "secret.md").write_text(
            "---\nid: hidden-1\ntype: note\n---\n# Hidden\nShould not be indexed."
        )

        # Real document
        (knowledge_dir / "visible.md").write_text(
            "---\nid: visible-1\ntype: note\n---\n# Visible\nShould be indexed."
        )

        ks = KnowledgeStore(knowledge_dir, tmp_path / "knowledge.db")
        ks.reindex()

        stats = ks.stats()
        assert stats["total"] == 1
        assert ks.get("visible-1") is not None
        assert ks.get("hidden-1") is None


class TestSeedDocumentFrontmatter:
    """Verify seed documents have proper YAML frontmatter."""

    @pytest.fixture
    def knowledge_dir(self):
        from corc.config import get_paths
        return Path(get_paths()["knowledge_dir"])

    @pytest.mark.parametrize("rel_path,expected_id,expected_type", [
        ("architecture/corc-spec.md", "corc-spec-v05", "architecture"),
        ("decisions/language-choice-python.md", "decision-language-python", "decision"),
        ("decisions/three-layer-data-architecture.md", "decision-three-layer-data", "decision"),
        ("decisions/determinism-model.md", "decision-determinism-model", "decision"),
        ("research/phase-0-learnings.md", "research-phase-0-learnings", "research"),
    ])
    def test_frontmatter_fields(self, knowledge_dir, rel_path, expected_id, expected_type):
        import yaml

        doc_path = knowledge_dir / rel_path
        assert doc_path.exists(), f"Document not found: {doc_path}"

        content = doc_path.read_text()
        assert content.startswith("---"), f"No YAML frontmatter in {rel_path}"

        # Parse frontmatter
        end = content.find("---", 3)
        fm = yaml.safe_load(content[3:end])

        assert fm["id"] == expected_id, f"Wrong id in {rel_path}"
        assert fm["type"] == expected_type, f"Wrong type in {rel_path}"
        assert fm["project"] == "corc", f"Missing project in {rel_path}"
        assert fm["status"] == "active", f"Wrong status in {rel_path}"
        assert fm["source"] == "human", f"Wrong source in {rel_path}"
        assert isinstance(fm["tags"], list), f"Tags should be a list in {rel_path}"
        assert len(fm["tags"]) > 0, f"No tags in {rel_path}"
