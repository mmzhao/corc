"""Tests for pattern analysis — correlations, prompt versions, planning patterns."""

import json
import os
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from corc.rating import Rating, RatingStore
from corc.patterns import (
    GroupStats,
    Correlation,
    Recommendation,
    PatternReport,
    PromptVersionStats,
    PromptReport,
    PlanningStats,
    PlanningReport,
    analyze_patterns,
    analyze_prompts,
    analyze_planning,
    format_pattern_report,
    format_prompt_report,
    format_planning_report,
    LOW_SCORE_THRESHOLD,
    HIGH_SCORE_THRESHOLD,
    FLAG_THRESHOLD,
    MIN_SAMPLE_SIZE,
    TRUST_MIN_SAMPLE,
    _compute_dimension_avgs,
    _compute_avg_overall,
    _group_ratings,
    _compute_group_stats,
    _bucket_checklist_size,
    _bucket_dependency_count,
    _bucket_context_size,
    _bucket_done_when_specificity,
)
from corc.audit import AuditLog


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ratings_dir(tmp_path):
    """Create a temporary ratings directory."""
    d = tmp_path / "ratings"
    d.mkdir()
    return d


@pytest.fixture
def rating_store(ratings_dir):
    """Create a RatingStore pointing at a temp directory."""
    return RatingStore(ratings_dir)


def _make_rating(
    task_id: str = "t1",
    task_name: str = "test task",
    scores: dict | None = None,
    overall: float = 7.0,
    flags: list | None = None,
    method: str = "heuristic",
    timestamp: str = "2026-03-22T10:00:00Z",
    metadata: dict | None = None,
) -> Rating:
    """Helper to create a Rating with sensible defaults."""
    return Rating(
        task_id=task_id,
        task_name=task_name,
        scores=scores or {"correctness": 7, "completeness": 7, "code-quality": 7},
        overall=overall,
        flags=flags or [],
        method=method,
        timestamp=timestamp,
        metadata=metadata or {},
    )


def _make_sample_ratings(
    n: int = 10, role: str = "implementer", **kwargs
) -> list[Rating]:
    """Create N sample ratings for a role with some variance."""
    ratings = []
    for i in range(n):
        base_score = 5 + (i % 6)  # 5..10 cycling
        scores = {
            "correctness": min(10, base_score + 1),
            "completeness": base_score,
            "code-quality": max(1, base_score - 1),
            "efficiency": base_score,
            "determinism": min(10, base_score + 2),
            "resilience": base_score,
            "human-intervention": min(10, base_score + 3),
        }
        overall = sum(scores.values()) / len(scores)
        meta = {"role": role, **kwargs}
        ratings.append(
            _make_rating(
                task_id=f"t{i}",
                task_name=f"task-{i}",
                scores=scores,
                overall=round(overall, 1),
                timestamp=f"2026-03-{20 + (i % 3):02d}T{10 + i:02d}:00:00Z",
                metadata=meta,
            )
        )
    return ratings


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_compute_dimension_avgs_empty(self):
        avgs = _compute_dimension_avgs([])
        assert avgs == {}

    def test_compute_dimension_avgs_single(self):
        r = _make_rating(scores={"correctness": 8, "completeness": 6})
        avgs = _compute_dimension_avgs([r])
        assert avgs["correctness"] == 8.0
        assert avgs["completeness"] == 6.0

    def test_compute_dimension_avgs_multiple(self):
        r1 = _make_rating(scores={"correctness": 8})
        r2 = _make_rating(scores={"correctness": 6})
        avgs = _compute_dimension_avgs([r1, r2])
        assert avgs["correctness"] == 7.0

    def test_compute_avg_overall_empty(self):
        assert _compute_avg_overall([]) == 0.0

    def test_compute_avg_overall(self):
        r1 = _make_rating(overall=8.0)
        r2 = _make_rating(overall=6.0)
        assert _compute_avg_overall([r1, r2]) == 7.0

    def test_group_ratings_by_role(self):
        r1 = _make_rating(metadata={"role": "implementer"})
        r2 = _make_rating(metadata={"role": "reviewer"})
        r3 = _make_rating(metadata={"role": "implementer"})
        groups = _group_ratings([r1, r2, r3], lambda r: r.metadata.get("role"))
        assert len(groups["implementer"]) == 2
        assert len(groups["reviewer"]) == 1

    def test_group_ratings_skips_empty_keys(self):
        r1 = _make_rating(metadata={"role": "implementer"})
        r2 = _make_rating(metadata={})  # no role -> key_fn returns None
        groups = _group_ratings([r1, r2], lambda r: r.metadata.get("role"))
        assert "implementer" in groups
        assert len(groups) == 1

    def test_compute_group_stats_min_sample(self):
        """Groups below MIN_SAMPLE_SIZE should be excluded."""
        r1 = _make_rating(metadata={"role": "implementer"}, overall=8.0)
        r2 = _make_rating(metadata={"role": "implementer"}, overall=6.0)
        groups = {"implementer": [r1, r2]}
        # MIN_SAMPLE_SIZE is 3, so 2 ratings should be excluded
        stats = _compute_group_stats(groups)
        assert len(stats) == 0

    def test_compute_group_stats_sufficient_sample(self):
        ratings = _make_sample_ratings(5, role="implementer")
        groups = {"implementer": ratings}
        stats = _compute_group_stats(groups)
        assert len(stats) == 1
        assert stats[0].key == "implementer"
        assert stats[0].count == 5
        assert stats[0].avg_overall > 0

    def test_bucket_checklist_size(self):
        assert _bucket_checklist_size(0) == "none (0)"
        assert _bucket_checklist_size(1) == "small (1-3)"
        assert _bucket_checklist_size(3) == "small (1-3)"
        assert _bucket_checklist_size(4) == "medium (4-7)"
        assert _bucket_checklist_size(7) == "medium (4-7)"
        assert _bucket_checklist_size(8) == "large (8+)"
        assert _bucket_checklist_size(20) == "large (8+)"

    def test_bucket_dependency_count(self):
        assert _bucket_dependency_count(0) == "independent (0)"
        assert _bucket_dependency_count(1) == "few (1-2)"
        assert _bucket_dependency_count(2) == "few (1-2)"
        assert _bucket_dependency_count(3) == "many (3+)"
        assert _bucket_dependency_count(10) == "many (3+)"

    def test_bucket_context_size(self):
        assert _bucket_context_size(0) == "none (0)"
        assert _bucket_context_size(2) == "small (1-3)"
        assert _bucket_context_size(5) == "medium (4-7)"
        assert _bucket_context_size(10) == "large (8+)"

    def test_bucket_done_when_specificity(self):
        assert _bucket_done_when_specificity("") == "absent"
        assert _bucket_done_when_specificity("tests pass") == "vague (<30 chars)"
        assert _bucket_done_when_specificity("a" * 50) == "moderate (30-100 chars)"
        assert _bucket_done_when_specificity("a" * 150) == "specific (100+ chars)"


# ---------------------------------------------------------------------------
# Pattern analysis tests
# ---------------------------------------------------------------------------


class TestAnalyzePatterns:
    def test_empty_ratings(self):
        report = analyze_patterns([])
        assert report.total_ratings == 0
        assert report.by_role == []
        assert report.by_task_type == []
        assert report.correlations == []
        assert report.recommendations == []

    def test_single_role_group(self):
        ratings = _make_sample_ratings(5, role="implementer")
        report = analyze_patterns(ratings)
        assert report.total_ratings == 5
        assert len(report.by_role) == 1
        assert report.by_role[0].key == "implementer"
        assert report.by_role[0].count == 5

    def test_multiple_role_groups(self):
        ratings = _make_sample_ratings(5, role="implementer") + _make_sample_ratings(
            4, role="reviewer"
        )
        report = analyze_patterns(ratings)
        assert report.total_ratings == 9
        # reviewer has only 4 < MIN_SAMPLE_SIZE? No, MIN_SAMPLE_SIZE is 3
        role_keys = {s.key for s in report.by_role}
        assert "implementer" in role_keys
        assert "reviewer" in role_keys

    def test_low_score_correlations(self):
        """Roles with dimension averages below LOW_SCORE_THRESHOLD should produce correlations."""
        low_ratings = []
        for i in range(5):
            low_ratings.append(
                _make_rating(
                    task_id=f"low-{i}",
                    scores={
                        "correctness": 3,  # well below 5.0
                        "completeness": 4,
                        "code-quality": 3,
                    },
                    overall=3.3,
                    metadata={"role": "bad-agent"},
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
            )
        report = analyze_patterns(low_ratings)
        low_corr = [c for c in report.correlations if c.direction == "low"]
        assert len(low_corr) > 0
        # correctness should be flagged
        dims_flagged = {c.dimension for c in low_corr}
        assert "correctness" in dims_flagged

    def test_high_score_correlations(self):
        """Roles with dimension averages above HIGH_SCORE_THRESHOLD produce high correlations."""
        high_ratings = []
        for i in range(5):
            high_ratings.append(
                _make_rating(
                    task_id=f"high-{i}",
                    scores={
                        "correctness": 10,
                        "completeness": 9,
                        "code-quality": 10,
                    },
                    overall=9.7,
                    metadata={"role": "star-agent"},
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
            )
        report = analyze_patterns(high_ratings)
        high_corr = [c for c in report.correlations if c.direction == "high"]
        assert len(high_corr) > 0
        dims_flagged = {c.dimension for c in high_corr}
        assert "correctness" in dims_flagged

    def test_task_type_grouping(self):
        ratings = []
        for i in range(4):
            ratings.append(
                _make_rating(
                    task_id=f"sql-{i}",
                    scores={"correctness": 4, "completeness": 5},
                    overall=4.5,
                    metadata={"role": "implementer", "task_type": "sql-migration"},
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
            )
        for i in range(4):
            ratings.append(
                _make_rating(
                    task_id=f"api-{i}",
                    scores={"correctness": 9, "completeness": 8},
                    overall=8.5,
                    metadata={"role": "implementer", "task_type": "api-endpoint"},
                    timestamp=f"2026-03-22T{14 + i}:00:00Z",
                )
            )
        report = analyze_patterns(ratings)
        type_keys = {s.key for s in report.by_task_type}
        assert "sql-migration" in type_keys
        assert "api-endpoint" in type_keys

    def test_context_bundle_grouping(self):
        ratings = []
        for i in range(4):
            ratings.append(
                _make_rating(
                    task_id=f"small-{i}",
                    scores={"correctness": 8},
                    overall=8.0,
                    metadata={
                        "role": "implementer",
                        "context_bundle": ["f1.py", "f2.py"],
                    },
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
            )
        for i in range(4):
            ratings.append(
                _make_rating(
                    task_id=f"large-{i}",
                    scores={"correctness": 5},
                    overall=5.0,
                    metadata={
                        "role": "implementer",
                        "context_bundle": [f"f{j}.py" for j in range(10)],
                    },
                    timestamp=f"2026-03-22T{14 + i}:00:00Z",
                )
            )
        report = analyze_patterns(ratings)
        bundle_keys = {s.key for s in report.by_context_bundle}
        assert "small (1-3 files)" in bundle_keys
        assert "large (8+ files)" in bundle_keys

    def test_recommendations_for_low_scoring_role(self):
        """Low overall score should produce critical recommendation."""
        low_ratings = []
        for i in range(5):
            low_ratings.append(
                _make_rating(
                    task_id=f"low-{i}",
                    scores={"correctness": 3, "completeness": 4, "code-quality": 3},
                    overall=3.3,
                    metadata={"role": "struggling-agent"},
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
            )
        report = analyze_patterns(low_ratings)
        critical_recs = [r for r in report.recommendations if r.severity == "critical"]
        assert len(critical_recs) > 0
        # Should mention the role
        assert any("struggling-agent" in r.message for r in critical_recs)

    def test_recommendations_for_low_dimensions(self):
        """Role with some low dimensions should get warning recommendation."""
        ratings = []
        for i in range(5):
            ratings.append(
                _make_rating(
                    task_id=f"mid-{i}",
                    scores={
                        "correctness": 8,
                        "completeness": 8,
                        "code-quality": 5,  # below FLAG_THRESHOLD (7)
                        "efficiency": 6,  # below FLAG_THRESHOLD
                    },
                    overall=6.8,
                    metadata={"role": "mixed-quality"},
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
            )
        report = analyze_patterns(ratings)
        warning_recs = [r for r in report.recommendations if r.severity == "warning"]
        assert len(warning_recs) > 0

    def test_trust_suggestions_raise(self):
        """Consistently high scores across 20+ tasks should suggest raising trust."""
        ratings = []
        for i in range(25):
            ratings.append(
                _make_rating(
                    task_id=f"great-{i}",
                    scores={"correctness": 10, "completeness": 9, "code-quality": 9},
                    overall=9.3,
                    metadata={"role": "trusted-agent"},
                    timestamp=f"2026-03-{(i % 28) + 1:02d}T10:00:00Z",
                )
            )
        report = analyze_patterns(ratings)
        trust_recs = [r for r in report.trust_suggestions if "raising" in r.message]
        assert len(trust_recs) > 0

    def test_trust_suggestions_lower(self):
        """Consistently low scores across 20+ tasks should suggest lowering trust."""
        ratings = []
        for i in range(25):
            ratings.append(
                _make_rating(
                    task_id=f"bad-{i}",
                    scores={"correctness": 3, "completeness": 4, "code-quality": 3},
                    overall=3.3,
                    metadata={"role": "untrusted-agent"},
                    timestamp=f"2026-03-{(i % 28) + 1:02d}T10:00:00Z",
                )
            )
        report = analyze_patterns(ratings)
        trust_recs = [r for r in report.trust_suggestions if "lowering" in r.message]
        assert len(trust_recs) > 0

    def test_trust_below_minimum_sample(self):
        """Groups below TRUST_MIN_SAMPLE should not produce trust suggestions."""
        ratings = _make_sample_ratings(5, role="small-sample")
        report = analyze_patterns(ratings)
        assert report.trust_suggestions == []


# ---------------------------------------------------------------------------
# Prompt version analysis tests
# ---------------------------------------------------------------------------


class TestAnalyzePrompts:
    def test_empty_ratings(self):
        report = analyze_prompts([], "implementer")
        assert report.role == "implementer"
        assert report.total_ratings == 0
        assert report.versions == []

    def test_no_matching_role(self):
        ratings = _make_sample_ratings(5, role="reviewer")
        report = analyze_prompts(ratings, "implementer")
        assert report.total_ratings == 0
        assert report.versions == []

    def test_single_version(self):
        ratings = []
        for i in range(5):
            ratings.append(
                _make_rating(
                    task_id=f"t{i}",
                    scores={"correctness": 8, "completeness": 7},
                    overall=7.5,
                    metadata={"role": "implementer", "prompt_version": "v1.0"},
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
            )
        report = analyze_prompts(ratings, "implementer")
        assert report.total_ratings == 5
        assert len(report.versions) == 1
        assert report.versions[0].version == "v1.0"
        assert report.versions[0].count == 5

    def test_multiple_versions(self):
        ratings = []
        for i in range(5):
            ratings.append(
                _make_rating(
                    task_id=f"v1-{i}",
                    scores={"correctness": 8, "completeness": 7},
                    overall=7.5,
                    metadata={"role": "implementer", "prompt_version": "v1.0"},
                    timestamp=f"2026-03-20T{10 + i}:00:00Z",
                )
            )
        for i in range(5):
            ratings.append(
                _make_rating(
                    task_id=f"v2-{i}",
                    scores={"correctness": 5, "completeness": 4},
                    overall=4.5,
                    metadata={"role": "implementer", "prompt_version": "v2.0"},
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
            )
        report = analyze_prompts(ratings, "implementer")
        assert report.total_ratings == 10
        assert len(report.versions) == 2
        versions_by_name = {v.version: v for v in report.versions}
        assert versions_by_name["v1.0"].avg_overall == 7.5
        assert versions_by_name["v2.0"].avg_overall == 4.5

    def test_recommendation_version_regression(self):
        """Significant score difference between versions should produce recommendation."""
        ratings = []
        for i in range(5):
            ratings.append(
                _make_rating(
                    task_id=f"v1-{i}",
                    scores={"correctness": 9},
                    overall=9.0,
                    metadata={"role": "implementer", "prompt_version": "v1.0"},
                    timestamp=f"2026-03-20T{10 + i}:00:00Z",
                )
            )
        for i in range(5):
            ratings.append(
                _make_rating(
                    task_id=f"v2-{i}",
                    scores={"correctness": 5},
                    overall=5.0,
                    metadata={"role": "implementer", "prompt_version": "v2.0"},
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
            )
        report = analyze_prompts(ratings, "implementer")
        assert len(report.recommendations) > 0
        # Should reference the regression
        assert any(
            "v1.0" in r.message and "v2.0" in r.message for r in report.recommendations
        )

    def test_recommendation_retire_bad_version(self):
        """Version with very low scores should get retirement recommendation."""
        ratings = []
        for i in range(5):
            ratings.append(
                _make_rating(
                    task_id=f"v1-{i}",
                    scores={"correctness": 8},
                    overall=8.0,
                    metadata={"role": "implementer", "prompt_version": "v1.0"},
                    timestamp=f"2026-03-20T{10 + i}:00:00Z",
                )
            )
        for i in range(5):
            ratings.append(
                _make_rating(
                    task_id=f"bad-{i}",
                    scores={"correctness": 3},
                    overall=3.0,
                    metadata={"role": "implementer", "prompt_version": "v-bad"},
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
            )
        report = analyze_prompts(ratings, "implementer")
        critical_recs = [r for r in report.recommendations if r.severity == "critical"]
        assert len(critical_recs) > 0
        assert any("v-bad" in r.message for r in critical_recs)

    def test_dimension_avgs_per_version(self):
        ratings = []
        for i in range(3):
            ratings.append(
                _make_rating(
                    task_id=f"t{i}",
                    scores={"correctness": 9, "completeness": 7, "code-quality": 8},
                    overall=8.0,
                    metadata={"role": "implementer", "prompt_version": "v1.0"},
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
            )
        report = analyze_prompts(ratings, "implementer")
        assert len(report.versions) == 1
        v = report.versions[0]
        assert v.dimension_avgs["correctness"] == 9.0
        assert v.dimension_avgs["completeness"] == 7.0
        assert v.dimension_avgs["code-quality"] == 8.0


# ---------------------------------------------------------------------------
# Planning pattern analysis tests
# ---------------------------------------------------------------------------


class TestAnalyzePlanning:
    def test_empty_ratings(self):
        report = analyze_planning([])
        assert report.total_ratings == 0
        assert report.by_checklist_size == []

    def test_checklist_size_impact(self):
        ratings = []
        # Small checklists → high scores
        for i in range(5):
            ratings.append(
                _make_rating(
                    task_id=f"small-{i}",
                    overall=8.5,
                    metadata={
                        "role": "implementer",
                        "checklist": [
                            {"item": "a", "done": True},
                            {"item": "b", "done": True},
                        ],
                    },
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
            )
        # Large checklists → low scores
        for i in range(5):
            ratings.append(
                _make_rating(
                    task_id=f"large-{i}",
                    overall=4.0,
                    metadata={
                        "role": "implementer",
                        "checklist": [
                            {"item": f"x{j}", "done": False} for j in range(10)
                        ],
                    },
                    timestamp=f"2026-03-22T{15 + i}:00:00Z",
                )
            )
        report = analyze_planning(ratings)
        assert report.total_ratings == 10
        assert len(report.by_checklist_size) >= 2
        # Small checklists should have higher success rate
        buckets = {s.bucket: s for s in report.by_checklist_size}
        assert buckets["small (1-3)"].avg_overall > buckets["large (8+)"].avg_overall

    def test_dependency_count_impact(self):
        ratings = []
        # Independent tasks → high scores
        for i in range(5):
            ratings.append(
                _make_rating(
                    task_id=f"indep-{i}",
                    overall=9.0,
                    metadata={"role": "implementer", "depends_on": []},
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
            )
        # Many dependencies → lower scores
        for i in range(5):
            ratings.append(
                _make_rating(
                    task_id=f"dep-{i}",
                    overall=5.0,
                    metadata={
                        "role": "implementer",
                        "depends_on": ["dep-a", "dep-b", "dep-c", "dep-d"],
                    },
                    timestamp=f"2026-03-22T{15 + i}:00:00Z",
                )
            )
        report = analyze_planning(ratings)
        assert len(report.by_dependency_count) >= 2
        buckets = {s.bucket: s for s in report.by_dependency_count}
        assert buckets["independent (0)"].avg_overall > buckets["many (3+)"].avg_overall

    def test_done_when_specificity_impact(self):
        ratings = []
        # Specific done_when → high scores
        for i in range(5):
            ratings.append(
                _make_rating(
                    task_id=f"spec-{i}",
                    overall=8.5,
                    metadata={
                        "role": "implementer",
                        "done_when": "All tests pass, PR created with coverage >80%, "
                        "linter shows zero warnings, and the endpoint returns correct "
                        "JSON for all three test cases in the spec.",
                    },
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
            )
        # Vague done_when → low scores
        for i in range(5):
            ratings.append(
                _make_rating(
                    task_id=f"vague-{i}",
                    overall=4.5,
                    metadata={"role": "implementer", "done_when": "it works"},
                    timestamp=f"2026-03-22T{15 + i}:00:00Z",
                )
            )
        report = analyze_planning(ratings)
        assert len(report.by_done_when_specificity) >= 2
        buckets = {s.bucket: s for s in report.by_done_when_specificity}
        assert "specific (100+ chars)" in buckets
        assert "vague (<30 chars)" in buckets
        assert (
            buckets["specific (100+ chars)"].avg_overall
            > buckets["vague (<30 chars)"].avg_overall
        )

    def test_context_size_impact(self):
        ratings = []
        for i in range(5):
            ratings.append(
                _make_rating(
                    task_id=f"small-ctx-{i}",
                    overall=8.0,
                    metadata={
                        "role": "implementer",
                        "context_bundle": ["a.py", "b.py"],
                    },
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
            )
        for i in range(5):
            ratings.append(
                _make_rating(
                    task_id=f"big-ctx-{i}",
                    overall=5.0,
                    metadata={
                        "role": "implementer",
                        "context_bundle": [f"f{j}.py" for j in range(12)],
                    },
                    timestamp=f"2026-03-22T{15 + i}:00:00Z",
                )
            )
        report = analyze_planning(ratings)
        assert len(report.by_context_size) >= 2

    def test_success_rate_calculation(self):
        ratings = []
        # 3 successes (overall >= 7), 2 failures
        for i, score in enumerate([8.0, 9.0, 7.0, 4.0, 3.0]):
            ratings.append(
                _make_rating(
                    task_id=f"t{i}",
                    overall=score,
                    metadata={
                        "role": "implementer",
                        "checklist": [{"item": "a", "done": True}],
                    },
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
            )
        report = analyze_planning(ratings)
        small_bucket = [
            s for s in report.by_checklist_size if s.bucket == "small (1-3)"
        ]
        assert len(small_bucket) == 1
        assert small_bucket[0].success_rate == 0.6  # 3/5

    def test_planning_recommendations_factor_diff(self):
        """Significant difference between buckets should produce recommendation."""
        ratings = []
        for i in range(5):
            ratings.append(
                _make_rating(
                    task_id=f"good-{i}",
                    overall=9.0,
                    metadata={
                        "role": "implementer",
                        "checklist": [{"item": "a", "done": True}],
                    },
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
            )
        for i in range(5):
            ratings.append(
                _make_rating(
                    task_id=f"bad-{i}",
                    overall=4.0,
                    metadata={
                        "role": "implementer",
                        "checklist": [
                            {"item": f"x{j}", "done": False} for j in range(10)
                        ],
                    },
                    timestamp=f"2026-03-22T{15 + i}:00:00Z",
                )
            )
        report = analyze_planning(ratings)
        assert len(report.recommendations) > 0
        # Should reference checklist size
        assert any("Checklist size" in r.message for r in report.recommendations)

    def test_planning_low_success_warning(self):
        """Buckets with very low success rate should get a warning."""
        ratings = []
        for i in range(5):
            ratings.append(
                _make_rating(
                    task_id=f"fail-{i}",
                    overall=3.0,  # all below 7.0
                    metadata={
                        "role": "implementer",
                        "checklist": [
                            {"item": f"x{j}", "done": False} for j in range(10)
                        ],
                    },
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
            )
        report = analyze_planning(ratings)
        warning_recs = [r for r in report.recommendations if r.severity == "warning"]
        assert len(warning_recs) > 0
        assert any("success rate" in r.message for r in warning_recs)

    def test_string_json_parsing_in_metadata(self):
        """Metadata with JSON strings should be parsed correctly."""
        ratings = []
        for i in range(4):
            ratings.append(
                _make_rating(
                    task_id=f"json-{i}",
                    overall=7.0,
                    metadata={
                        "role": "implementer",
                        "checklist": json.dumps([{"item": "a", "done": True}]),
                        "depends_on": json.dumps(["dep-1"]),
                        "context_bundle": json.dumps(["f1.py", "f2.py"]),
                    },
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
            )
        report = analyze_planning(ratings)
        # Should still parse and bucket correctly
        assert report.total_ratings == 4


# ---------------------------------------------------------------------------
# Formatting tests
# ---------------------------------------------------------------------------


class TestFormatPatternReport:
    def test_format_empty(self):
        report = PatternReport()
        result = format_pattern_report(report)
        assert "No rating data found" in result

    def test_format_with_roles(self):
        report = PatternReport(
            total_ratings=10,
            by_role=[
                GroupStats(
                    key="implementer",
                    count=5,
                    avg_overall=7.5,
                    dimension_avgs={"correctness": 8.0},
                    low_dimensions=[],
                ),
                GroupStats(
                    key="reviewer",
                    count=5,
                    avg_overall=4.0,
                    dimension_avgs={"correctness": 3.5},
                    low_dimensions=["correctness"],
                ),
            ],
        )
        result = format_pattern_report(report)
        assert "Scores by Role:" in result
        assert "implementer" in result
        assert "reviewer" in result

    def test_format_with_correlations(self):
        report = PatternReport(
            total_ratings=10,
            correlations=[
                Correlation("role", "bad-agent", "correctness", 3.5, 5, "low"),
                Correlation("role", "good-agent", "correctness", 9.5, 5, "high"),
            ],
        )
        result = format_pattern_report(report)
        assert "Low-scoring correlations:" in result
        assert "High-scoring correlations:" in result
        assert "bad-agent" in result
        assert "good-agent" in result

    def test_format_with_recommendations(self):
        report = PatternReport(
            total_ratings=10,
            recommendations=[
                Recommendation(
                    "prompt", "critical", "Fix the bad prompt", "evidence here"
                ),
                Recommendation(
                    "workflow", "warning", "Adjust workflow", "more evidence"
                ),
            ],
        )
        result = format_pattern_report(report)
        assert "Recommendations:" in result
        assert "Fix the bad prompt" in result
        assert "Adjust workflow" in result


class TestFormatPromptReport:
    def test_format_empty(self):
        report = PromptReport(role="implementer")
        result = format_prompt_report(report)
        assert "No ratings found" in result
        assert "implementer" in result

    def test_format_with_versions(self):
        report = PromptReport(
            role="implementer",
            total_ratings=10,
            versions=[
                PromptVersionStats("v1.0", 5, 8.0, {"correctness": 9.0}),
                PromptVersionStats("v2.0", 5, 5.0, {"correctness": 4.0}),
            ],
            recommendations=[
                Recommendation(
                    "prompt", "warning", "v1.0 is better than v2.0", "evidence"
                )
            ],
        )
        result = format_prompt_report(report)
        assert "implementer" in result
        assert "v1.0" in result
        assert "v2.0" in result
        assert "Total ratings: 10" in result


class TestFormatPlanningReport:
    def test_format_empty(self):
        report = PlanningReport()
        result = format_planning_report(report)
        assert "No rating data found" in result

    def test_format_with_data(self):
        report = PlanningReport(
            total_ratings=20,
            by_checklist_size=[
                PlanningStats("checklist_size", "small (1-3)", 10, 8.0, 0.9),
                PlanningStats("checklist_size", "large (8+)", 10, 4.0, 0.2),
            ],
            recommendations=[
                Recommendation(
                    "planning", "info", "Small checklists work better", "evidence"
                )
            ],
        )
        result = format_planning_report(report)
        assert "Planning Pattern Analysis" in result
        assert "Checklist Size vs Outcome:" in result
        assert "small (1-3)" in result
        assert "large (8+)" in result
        assert "Recommendations:" in result


# ---------------------------------------------------------------------------
# Integration tests with RatingStore
# ---------------------------------------------------------------------------


class TestPatternAnalysisIntegration:
    def test_full_pipeline_with_store(self, rating_store):
        """End-to-end: save ratings, read them, analyze patterns."""
        for i in range(5):
            rating_store.save(
                _make_rating(
                    task_id=f"impl-{i}",
                    scores={"correctness": 8, "completeness": 7, "efficiency": 6},
                    overall=7.0,
                    metadata={
                        "role": "implementer",
                        "prompt_version": "v1.0",
                        "task_type": "feature",
                        "checklist": [{"item": "a", "done": True}],
                        "depends_on": [],
                        "context_bundle": ["src/main.py"],
                        "done_when": "Tests pass and PR approved",
                    },
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
            )
        for i in range(5):
            rating_store.save(
                _make_rating(
                    task_id=f"rev-{i}",
                    scores={"correctness": 9, "completeness": 9, "efficiency": 8},
                    overall=8.7,
                    metadata={
                        "role": "reviewer",
                        "prompt_version": "v1.0",
                        "task_type": "review",
                        "checklist": [{"item": "check", "done": True}],
                        "depends_on": [f"impl-{i}"],
                        "context_bundle": ["src/main.py", "tests/test_main.py"],
                        "done_when": "Review complete with actionable feedback.",
                    },
                    timestamp=f"2026-03-22T{15 + i}:00:00Z",
                )
            )

        ratings = rating_store.read_all()
        assert len(ratings) == 10

        # Test patterns
        pattern_report = analyze_patterns(ratings)
        assert pattern_report.total_ratings == 10
        assert len(pattern_report.by_role) >= 2
        output = format_pattern_report(pattern_report)
        assert "implementer" in output
        assert "reviewer" in output

        # Test prompts
        prompt_report = analyze_prompts(ratings, "implementer")
        assert prompt_report.total_ratings == 5
        output = format_prompt_report(prompt_report)
        assert "v1.0" in output

        # Test planning
        planning_report = analyze_planning(ratings)
        assert planning_report.total_ratings == 10
        output = format_planning_report(planning_report)
        assert "Planning Pattern Analysis" in output

    def test_realistic_multi_version_scenario(self, rating_store):
        """Realistic scenario: role with improving prompt versions."""
        # v1.0 — mediocre
        for i in range(5):
            rating_store.save(
                _make_rating(
                    task_id=f"v1-{i}",
                    scores={
                        "correctness": 5,
                        "completeness": 5,
                        "code-quality": 4,
                        "efficiency": 6,
                    },
                    overall=5.0,
                    metadata={"role": "implementer", "prompt_version": "v1.0"},
                    timestamp=f"2026-03-15T{10 + i}:00:00Z",
                )
            )
        # v2.0 — better
        for i in range(5):
            rating_store.save(
                _make_rating(
                    task_id=f"v2-{i}",
                    scores={
                        "correctness": 8,
                        "completeness": 7,
                        "code-quality": 7,
                        "efficiency": 8,
                    },
                    overall=7.5,
                    metadata={"role": "implementer", "prompt_version": "v2.0"},
                    timestamp=f"2026-03-20T{10 + i}:00:00Z",
                )
            )
        # v3.0 — regression
        for i in range(5):
            rating_store.save(
                _make_rating(
                    task_id=f"v3-{i}",
                    scores={
                        "correctness": 4,
                        "completeness": 3,
                        "code-quality": 3,
                        "efficiency": 5,
                    },
                    overall=3.8,
                    metadata={"role": "implementer", "prompt_version": "v3.0"},
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
            )

        ratings = rating_store.read_all()
        report = analyze_prompts(ratings, "implementer")

        assert report.total_ratings == 15
        assert len(report.versions) == 3

        # v2.0 should be best
        versions_by_name = {v.version: v for v in report.versions}
        assert (
            versions_by_name["v2.0"].avg_overall > versions_by_name["v1.0"].avg_overall
        )
        assert (
            versions_by_name["v2.0"].avg_overall > versions_by_name["v3.0"].avg_overall
        )

        # Should recommend rolling back from v3.0
        assert len(report.recommendations) > 0
        recs_text = " ".join(r.message for r in report.recommendations)
        assert "v3.0" in recs_text or "retired" in recs_text


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestAnalyzeCLI:
    """Test the new analyze CLI commands via Click test runner."""

    def _setup_project(self, tmp_path):
        """Set up minimal project structure for CLI tests."""
        events_dir = tmp_path / "data" / "events"
        events_dir.mkdir(parents=True)
        ratings_dir = tmp_path / "data" / "ratings"
        ratings_dir.mkdir(parents=True)
        (tmp_path / "data").mkdir(exist_ok=True)
        (tmp_path / ".corc").mkdir(exist_ok=True)
        (tmp_path / ".git").mkdir(exist_ok=True)
        (tmp_path / "knowledge").mkdir(exist_ok=True)
        (tmp_path / "data" / "mutations.jsonl").touch()
        return ratings_dir

    def _write_ratings(self, ratings_dir, ratings):
        """Write ratings directly to the JSONL file."""
        rs = RatingStore(ratings_dir)
        for r in ratings:
            rs.save(r)

    def test_analyze_patterns(self, tmp_path):
        from corc.cli import cli

        ratings_dir = self._setup_project(tmp_path)
        self._write_ratings(
            ratings_dir,
            [
                _make_rating(
                    task_id=f"t{i}",
                    scores={"correctness": 7, "completeness": 6},
                    overall=6.5,
                    metadata={"role": "implementer"},
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
                for i in range(5)
            ],
        )

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path.parent):
            os.chdir(str(tmp_path))
            result = runner.invoke(cli, ["analyze", "patterns"])
            assert result.exit_code == 0
            assert "Pattern Analysis" in result.output

    def test_analyze_patterns_empty(self, tmp_path):
        from corc.cli import cli

        self._setup_project(tmp_path)

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path.parent):
            os.chdir(str(tmp_path))
            result = runner.invoke(cli, ["analyze", "patterns"])
            assert result.exit_code == 0
            assert "No rating data found" in result.output

    def test_analyze_prompts(self, tmp_path):
        from corc.cli import cli

        ratings_dir = self._setup_project(tmp_path)
        self._write_ratings(
            ratings_dir,
            [
                _make_rating(
                    task_id=f"t{i}",
                    scores={"correctness": 8},
                    overall=8.0,
                    metadata={"role": "implementer", "prompt_version": "v1.0"},
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
                for i in range(5)
            ],
        )

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path.parent):
            os.chdir(str(tmp_path))
            result = runner.invoke(cli, ["analyze", "prompts", "--role", "implementer"])
            assert result.exit_code == 0
            assert "implementer" in result.output
            assert "v1.0" in result.output

    def test_analyze_prompts_no_role(self, tmp_path):
        from corc.cli import cli

        self._setup_project(tmp_path)

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path.parent):
            os.chdir(str(tmp_path))
            result = runner.invoke(cli, ["analyze", "prompts", "--role", "nonexistent"])
            assert result.exit_code == 0
            assert "No ratings found" in result.output

    def test_analyze_planning(self, tmp_path):
        from corc.cli import cli

        ratings_dir = self._setup_project(tmp_path)
        self._write_ratings(
            ratings_dir,
            [
                _make_rating(
                    task_id=f"t{i}",
                    overall=7.0,
                    metadata={
                        "role": "implementer",
                        "checklist": [{"item": "a", "done": True}],
                        "depends_on": [],
                        "done_when": "Tests pass",
                    },
                    timestamp=f"2026-03-22T{10 + i}:00:00Z",
                )
                for i in range(5)
            ],
        )

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path.parent):
            os.chdir(str(tmp_path))
            result = runner.invoke(cli, ["analyze", "planning"])
            assert result.exit_code == 0
            assert "Planning Pattern Analysis" in result.output

    def test_analyze_planning_empty(self, tmp_path):
        from corc.cli import cli

        self._setup_project(tmp_path)

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path.parent):
            os.chdir(str(tmp_path))
            result = runner.invoke(cli, ["analyze", "planning"])
            assert result.exit_code == 0
            assert "No rating data found" in result.output

    def test_analyze_prompts_requires_role(self, tmp_path):
        from corc.cli import cli

        self._setup_project(tmp_path)

        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path.parent):
            os.chdir(str(tmp_path))
            result = runner.invoke(cli, ["analyze", "prompts"])
            # Should fail because --role is required
            assert result.exit_code != 0
