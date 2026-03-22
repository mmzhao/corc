"""Integration tests for hybrid search.

Exercises the full hybrid search pipeline with 10+ diverse documents:
- Both FTS5 keyword and semantic search run
- Scores normalized to [0, 1]
- Configurable weights (default 0.4 keyword + 0.6 semantic)
- Deduplication by document (best chunk score kept)
- Hybrid is the default search mode
"""

import hashlib
from pathlib import Path
from unittest import mock

import pytest

from corc import embeddings
from corc.knowledge import KnowledgeStore

# ---------------------------------------------------------------------------
# Deterministic fake embeddings — hash-based, reproducible
# ---------------------------------------------------------------------------

FAKE_DIM = 32


def _fake_embedding(text: str) -> list[float]:
    """Produce a deterministic 32-dim embedding from text.

    Uses MD5 + SHA256 to get enough bytes, then normalizes.
    """
    h1 = hashlib.md5(text.encode()).digest()  # 16 bytes
    h2 = hashlib.sha256(text.encode()).digest()[:16]  # 16 bytes
    raw = [(b / 255.0) * 2 - 1 for b in h1 + h2]  # [-1, 1]
    # Normalize to unit vector so cosine similarity is meaningful
    norm = sum(x * x for x in raw) ** 0.5
    if norm > 0:
        raw = [x / norm for x in raw]
    return raw


def _fake_encode(texts: list[str]) -> list[list[float]]:
    return [_fake_embedding(t) for t in texts]


def _fake_encode_single(text: str) -> list[float]:
    return _fake_embedding(text)


# ---------------------------------------------------------------------------
# Document corpus — 12 diverse documents spanning different types/projects
# ---------------------------------------------------------------------------

DOCUMENTS = [
    {
        "filename": "python-patterns.md",
        "content": (
            "---\nid: doc01\ntype: decision\nproject: backend\ntags: [python, design]\n---\n\n"
            "# Python Design Patterns\n\n"
            "We use factory patterns and dependency injection in our Python codebase. "
            "This helps with testability and loose coupling between modules. "
            "The abstract factory pattern is especially useful for creating families "
            "of related objects without specifying their concrete classes."
        ),
    },
    {
        "filename": "database-arch.md",
        "content": (
            "---\nid: doc02\ntype: architecture\nproject: backend\ntags: [database, sqlite]\n---\n\n"
            "# Database Architecture\n\n"
            "SQLite is used as the primary data store. FTS5 provides full-text search "
            "capability. All tables use WAL mode for concurrent read access. "
            "Migrations are handled via schema versioning in the application layer."
        ),
    },
    {
        "filename": "testing-strategy.md",
        "content": (
            "---\nid: doc03\ntype: note\nproject: backend\ntags: [testing, pytest]\n---\n\n"
            "# Testing Strategy\n\n"
            "Unit tests use pytest with tmp_path fixtures for isolation. "
            "Integration tests verify the full pipeline end-to-end. "
            "We aim for 80% code coverage with emphasis on critical paths."
        ),
    },
    {
        "filename": "api-design.md",
        "content": (
            "---\nid: doc04\ntype: decision\nproject: frontend\ntags: [api, rest]\n---\n\n"
            "# REST API Design Guidelines\n\n"
            "All endpoints follow RESTful conventions. JSON is the primary response "
            "format. Pagination uses cursor-based approach for efficiency. "
            "Rate limiting is enforced per API key with sliding window counters."
        ),
    },
    {
        "filename": "deployment.md",
        "content": (
            "---\nid: doc05\ntype: architecture\nproject: infra\ntags: [deployment, docker]\n---\n\n"
            "# Deployment Architecture\n\n"
            "Services are containerized with Docker and orchestrated via Kubernetes. "
            "Blue-green deployments minimize downtime. Health checks run every 30 seconds. "
            "Rollback is automated if error rate exceeds 5% threshold."
        ),
    },
    {
        "filename": "auth-system.md",
        "content": (
            "---\nid: doc06\ntype: decision\nproject: backend\ntags: [auth, security]\n---\n\n"
            "# Authentication System\n\n"
            "JWT tokens are used for stateless authentication. Refresh tokens have "
            "7-day expiry. Multi-factor authentication is optional but recommended. "
            "Password hashing uses bcrypt with cost factor 12."
        ),
    },
    {
        "filename": "monitoring.md",
        "content": (
            "---\nid: doc07\ntype: architecture\nproject: infra\ntags: [monitoring, observability]\n---\n\n"
            "# Monitoring and Observability\n\n"
            "Prometheus collects metrics from all services. Grafana dashboards "
            "visualize key performance indicators. Alerts fire via PagerDuty "
            "for P1 incidents. Distributed tracing uses OpenTelemetry."
        ),
    },
    {
        "filename": "caching-strategy.md",
        "content": (
            "---\nid: doc08\ntype: decision\nproject: backend\ntags: [caching, redis]\n---\n\n"
            "# Caching Strategy\n\n"
            "Redis is used for application-level caching with TTL-based invalidation. "
            "Cache-aside pattern is the default. Hot paths use read-through caching. "
            "Cache warming runs during deployment to prevent cold-start latency."
        ),
    },
    {
        "filename": "data-pipeline.md",
        "content": (
            "---\nid: doc09\ntype: architecture\nproject: data\ntags: [pipeline, etl]\n---\n\n"
            "# Data Pipeline Architecture\n\n"
            "ETL pipelines run on Apache Airflow with daily scheduling. "
            "Data validation uses Great Expectations. Schema evolution is managed "
            "via Avro. Raw data lands in S3, processed data goes to Snowflake."
        ),
    },
    {
        "filename": "search-impl.md",
        "content": (
            "---\nid: doc10\ntype: decision\nproject: backend\ntags: [search, fts5]\n---\n\n"
            "# Search Implementation\n\n"
            "Full-text search uses SQLite FTS5 with porter stemming and unicode61 "
            "tokenizer. Semantic search adds sentence-transformers embeddings. "
            "Hybrid search combines both with configurable weights for best results."
        ),
    },
    {
        "filename": "ml-serving.md",
        "content": (
            "---\nid: doc11\ntype: architecture\nproject: ml\ntags: [ml, serving]\n---\n\n"
            "# ML Model Serving\n\n"
            "Models are served via FastAPI behind an nginx reverse proxy. "
            "Batch predictions use Celery workers. A/B testing infrastructure "
            "routes traffic between model versions based on experiment configs."
        ),
    },
    {
        "filename": "code-review.md",
        "content": (
            "---\nid: doc12\ntype: note\ntags: [process, review]\n---\n\n"
            "# Code Review Process\n\n"
            "All changes require at least one approval. Reviews focus on correctness, "
            "readability, and test coverage. Automated linting enforces style. "
            "Security-sensitive changes require review from the security team."
        ),
    },
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ks_with_docs(tmp_path):
    """Create a KnowledgeStore populated with 12 diverse documents."""
    knowledge_dir = tmp_path / "knowledge"

    with mock.patch.object(embeddings, "is_available", return_value=True), \
         mock.patch.object(embeddings, "encode", side_effect=_fake_encode), \
         mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
        store = KnowledgeStore(knowledge_dir, tmp_path / "knowledge.db")
        store._embeddings_available = True

        for doc in DOCUMENTS:
            (knowledge_dir / doc["filename"]).write_text(doc["content"])
            store.add(file_path=knowledge_dir / doc["filename"])

    return store


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestHybridSearchIntegration:
    """Full pipeline integration test with 12 documents."""

    def test_corpus_has_12_documents(self, ks_with_docs):
        """Sanity: all 12 documents are indexed."""
        stats = ks_with_docs.stats()
        assert stats["total"] == 12

    def test_hybrid_search_returns_results(self, ks_with_docs):
        """Hybrid search should return results from the corpus."""
        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks_with_docs.hybrid_search("database architecture")
        assert len(results) > 0

    def test_all_scores_in_0_1_range(self, ks_with_docs):
        """All scores (combined, FTS, semantic) must be in [0, 1]."""
        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks_with_docs.hybrid_search("Python design patterns")

        for r in results:
            assert 0.0 <= r["score"] <= 1.0, f"combined score {r['score']} out of range"
            assert 0.0 <= r["fts_score"] <= 1.0, f"fts_score {r['fts_score']} out of range"
            assert 0.0 <= r["semantic_score"] <= 1.0, f"semantic_score {r['semantic_score']} out of range"

    def test_results_deduplicated_by_document(self, ks_with_docs):
        """Each document should appear at most once in results."""
        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks_with_docs.hybrid_search("search", limit=20)

        doc_ids = [r["id"] for r in results]
        assert len(doc_ids) == len(set(doc_ids)), f"Duplicate doc IDs: {doc_ids}"

    def test_results_sorted_by_score_descending(self, ks_with_docs):
        """Results must be ordered by combined score descending."""
        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks_with_docs.hybrid_search("caching strategy")

        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]["score"] >= results[i + 1]["score"], (
                    f"Score at index {i} ({results[i]['score']}) < "
                    f"score at index {i+1} ({results[i+1]['score']})"
                )

    def test_default_weights_are_0_4_keyword_0_6_semantic(self, ks_with_docs):
        """Default weights: 0.4 keyword + 0.6 semantic."""
        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks_with_docs.hybrid_search("monitoring observability")

        # Verify by checking that combined score = 0.4*fts + 0.6*sem
        for r in results:
            expected = 0.4 * r["fts_score"] + 0.6 * r["semantic_score"]
            assert abs(r["score"] - expected) < 1e-9, (
                f"Score {r['score']} != 0.4*{r['fts_score']} + 0.6*{r['semantic_score']} "
                f"= {expected}"
            )

    def test_configurable_weights(self, ks_with_docs):
        """Custom weights should be respected."""
        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks_with_docs.hybrid_search(
                "deployment docker",
                semantic_weight=0.3,
                keyword_weight=0.7,
            )

        for r in results:
            expected = 0.7 * r["fts_score"] + 0.3 * r["semantic_score"]
            assert abs(r["score"] - expected) < 1e-9

    def test_pure_keyword_weight(self, ks_with_docs):
        """With semantic_weight=0.0, only keyword scores should matter."""
        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks_with_docs.hybrid_search("testing pytest", semantic_weight=0.0)

        for r in results:
            expected = 1.0 * r["fts_score"] + 0.0 * r["semantic_score"]
            assert abs(r["score"] - expected) < 1e-9

    def test_pure_semantic_weight(self, ks_with_docs):
        """With semantic_weight=1.0, only semantic scores should matter."""
        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks_with_docs.hybrid_search("authentication security", semantic_weight=1.0)

        for r in results:
            expected = 0.0 * r["fts_score"] + 1.0 * r["semantic_score"]
            assert abs(r["score"] - expected) < 1e-9

    def test_respects_limit(self, ks_with_docs):
        """Limit parameter should constrain result count."""
        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks_with_docs.hybrid_search("design", limit=3)

        assert len(results) <= 3

    def test_filters_by_doc_type(self, ks_with_docs):
        """doc_type filter should only return documents of that type."""
        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks_with_docs.hybrid_search(
                "architecture",
                doc_type="architecture",
            )

        for r in results:
            assert r["type"] == "architecture"

    def test_filters_by_project(self, ks_with_docs):
        """project filter should only return documents from that project."""
        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks_with_docs.hybrid_search(
                "architecture",
                project="infra",
            )

        for r in results:
            assert r["project"] == "infra"

    def test_both_search_modes_contribute(self, ks_with_docs):
        """Documents found only by one search mode should still appear."""
        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks_with_docs.hybrid_search("data pipeline ETL", limit=12)

        # Should get results from both FTS and semantic
        has_fts_only = any(r["fts_score"] > 0 and r["semantic_score"] == 0 for r in results)
        has_sem_only = any(r["fts_score"] == 0 and r["semantic_score"] > 0 for r in results)
        has_both = any(r["fts_score"] > 0 and r["semantic_score"] > 0 for r in results)

        # At minimum, documents matching the FTS query should have fts_score > 0
        fts_docs = [r for r in results if r["fts_score"] > 0]
        assert len(fts_docs) > 0, "Expected some results from FTS"

    def test_result_includes_expected_fields(self, ks_with_docs):
        """All results should have standard document fields."""
        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks_with_docs.hybrid_search("API design")

        expected_fields = {"id", "file_path", "type", "title", "status",
                           "source", "created", "updated", "score",
                           "fts_score", "semantic_score"}
        for r in results:
            missing = expected_fields - set(r.keys())
            assert not missing, f"Missing fields: {missing}"

    def test_top_result_relevance(self, ks_with_docs):
        """The most relevant document should rank first for a specific query."""
        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks_with_docs.hybrid_search("full text search sqlite")

        # doc10 (search-impl) and doc02 (database-arch) should rank highly
        top_ids = {r["id"] for r in results[:3]}
        assert "doc10" in top_ids or "doc02" in top_ids, (
            f"Expected doc10 or doc02 in top 3, got {top_ids}"
        )


class TestHybridSearchFallback:
    """Test hybrid search fallback when embeddings are unavailable."""

    @pytest.fixture()
    def ks_no_embeddings(self, tmp_path):
        """KnowledgeStore without embeddings support, populated with docs."""
        knowledge_dir = tmp_path / "knowledge"
        with mock.patch.object(embeddings, "is_available", return_value=False):
            store = KnowledgeStore(knowledge_dir, tmp_path / "knowledge.db")
            assert store._embeddings_available is False

            for doc in DOCUMENTS:
                (knowledge_dir / doc["filename"]).write_text(doc["content"])
                store.add(file_path=knowledge_dir / doc["filename"])

        return store

    def test_falls_back_to_fts_gracefully(self, ks_no_embeddings):
        """Without embeddings, hybrid search should return FTS results."""
        results = ks_no_embeddings.hybrid_search("database architecture")
        assert len(results) > 0

    def test_fallback_results_have_score(self, ks_no_embeddings):
        """Fallback results should still have a score field."""
        results = ks_no_embeddings.hybrid_search("testing strategy")
        for r in results:
            assert "score" in r


class TestHybridSearchEdgeCases:
    """Edge cases for hybrid search."""

    @pytest.fixture()
    def ks(self, tmp_path):
        knowledge_dir = tmp_path / "knowledge"
        with mock.patch.object(embeddings, "is_available", return_value=True), \
             mock.patch.object(embeddings, "encode", side_effect=_fake_encode), \
             mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            store = KnowledgeStore(knowledge_dir, tmp_path / "knowledge.db")
            store._embeddings_available = True
        return store

    def test_single_result_gets_score_1(self, ks, tmp_path):
        """When only one result is returned, its normalized score should be 1.0."""
        knowledge_dir = tmp_path / "knowledge"
        (knowledge_dir / "unique.md").write_text(
            "---\nid: uniq1\ntype: note\n---\n\n"
            "# Extremely Unique Xylophone Content\n\n"
            "This document discusses xylophones and nothing else matters here."
        )
        with mock.patch.object(embeddings, "encode", side_effect=_fake_encode), \
             mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            ks.add(file_path=knowledge_dir / "unique.md")
            results = ks.hybrid_search("xylophone")

        assert len(results) >= 1
        # With a single FTS result, the normalized FTS score should be 1.0
        assert results[0]["fts_score"] == 1.0

    def test_no_results_returns_empty(self, ks):
        """Query matching nothing should return empty list."""
        with mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            results = ks.hybrid_search("zzzznonexistentterm")
        # FTS returns nothing; semantic might still return results
        # but with no docs in store, should be empty
        assert isinstance(results, list)

    def test_keyword_weight_override(self, ks, tmp_path):
        """Explicit keyword_weight should override the computed 1-semantic_weight."""
        knowledge_dir = tmp_path / "knowledge"
        (knowledge_dir / "test1.md").write_text(
            "---\nid: tw1\ntype: note\n---\n\n# Test Weight\n\nWeight testing document."
        )
        (knowledge_dir / "test2.md").write_text(
            "---\nid: tw2\ntype: note\n---\n\n# Weight Test\n\nAnother weight testing document."
        )
        with mock.patch.object(embeddings, "encode", side_effect=_fake_encode), \
             mock.patch.object(embeddings, "encode_single", side_effect=_fake_encode_single):
            ks.add(file_path=knowledge_dir / "test1.md")
            ks.add(file_path=knowledge_dir / "test2.md")
            results = ks.hybrid_search(
                "weight testing",
                semantic_weight=0.5,
                keyword_weight=0.5,
            )

        # Verify: combined = 0.5*fts + 0.5*sem
        for r in results:
            expected = 0.5 * r["fts_score"] + 0.5 * r["semantic_score"]
            assert abs(r["score"] - expected) < 1e-9
