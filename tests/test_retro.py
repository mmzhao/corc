"""Tests for project retrospective generation."""

import json
import time
from pathlib import Path

import pytest

from corc.audit import AuditLog
from corc.knowledge import KnowledgeStore
from corc.mutations import MutationLog
from corc.rating import Rating, RatingStore
from corc.state import WorkState
from corc.retro import (
    Retrospective,
    generate_retrospective,
    format_retrospective,
    retrospective_to_markdown,
    save_retrospective,
    _collect_project_tasks,
    _collect_project_ratings,
    _collect_project_events,
    _compute_quality_trend,
    _extract_findings,
    _identify_what_went_well,
    _identify_what_didnt_go_well,
    _generate_recommendations,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_env(tmp_path):
    """Full temporary environment with all data layers."""
    mutations_path = tmp_path / "data" / "mutations.jsonl"
    mutations_path.parent.mkdir(parents=True, exist_ok=True)
    state_db = tmp_path / "data" / "state.db"
    events_dir = tmp_path / "data" / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    ratings_dir = tmp_path / "data" / "ratings"
    ratings_dir.mkdir(parents=True, exist_ok=True)
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    knowledge_db = tmp_path / "data" / "knowledge.db"

    ml = MutationLog(mutations_path)
    ws = WorkState(state_db, ml)
    al = AuditLog(events_dir)
    rs = RatingStore(ratings_dir)
    ks = KnowledgeStore(knowledge_dir, knowledge_db)

    return {
        "ml": ml,
        "ws": ws,
        "al": al,
        "rs": rs,
        "ks": ks,
        "tmp_path": tmp_path,
    }


def _create_task(
    ml,
    task_id,
    name,
    status="completed",
    role="implementer",
    attempt_count=1,
    findings=None,
):
    """Helper to create a task via mutation log."""
    ml.append(
        "task_created",
        {
            "id": task_id,
            "name": name,
            "description": f"Description for {name}",
            "role": role,
            "depends_on": [],
            "done_when": f"{name} is done",
            "checklist": [],
            "context_bundle": [],
            "context_bundle_mtimes": {},
            "attempt_count": attempt_count,
        },
        reason="test task creation",
    )
    if status == "completed":
        ml.append(
            "task_completed",
            {
                "pr_url": f"https://github.com/test/pr/{task_id}",
                "findings": findings or [],
            },
            task_id=task_id,
            reason="test task completion",
        )
    elif status == "failed":
        ml.append(
            "task_failed",
            {
                "findings": findings or [],
                "attempt_count": attempt_count,
            },
            task_id=task_id,
            reason="test task failure",
        )
    elif status == "escalated":
        ml.append(
            "task_escalated",
            {
                "attempt_count": attempt_count,
            },
            task_id=task_id,
            reason="test task escalation",
        )


def _create_rating(rs, task_id, task_name, scores, overall=None, method="heuristic"):
    """Helper to create a rating."""
    from corc.rating import weighted_score, flagged_dimensions

    if overall is None:
        overall = weighted_score(scores)
    flags = flagged_dimensions(scores)
    rating = Rating(
        task_id=task_id,
        task_name=task_name,
        scores=scores,
        overall=round(overall, 2),
        flags=flags,
        method=method,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        metadata={"role": "implementer"},
    )
    rs.save(rating)
    return rating


def _write_events(events_dir, date_str, events):
    """Write events directly to a specific date file."""
    path = events_dir / f"{date_str}.jsonl"
    with open(path, "a") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def _build_sample_project(tmp_env):
    """Build a sample project with multiple tasks, ratings, and events."""
    ml = tmp_env["ml"]
    rs = tmp_env["rs"]
    al = tmp_env["al"]

    # Create tasks
    _create_task(ml, "t001", "myproject-setup", status="completed", attempt_count=1)
    _create_task(
        ml,
        "t002",
        "myproject-core",
        status="completed",
        attempt_count=1,
        findings=["Found that async is needed"],
    )
    _create_task(ml, "t003", "myproject-api", status="completed", attempt_count=2)
    _create_task(ml, "t004", "myproject-tests", status="failed", attempt_count=3)
    _create_task(ml, "t005", "myproject-docs", status="escalated", attempt_count=3)

    # Rebuild work state to pick up mutations
    tmp_env["ws"].rebuild()

    # Create ratings for completed tasks
    _create_rating(
        rs,
        "t001",
        "myproject-setup",
        {
            "correctness": 9,
            "completeness": 9,
            "code-quality": 8,
            "efficiency": 9,
            "determinism": 9,
            "resilience": 9,
            "human-intervention": 10,
        },
    )
    _create_rating(
        rs,
        "t002",
        "myproject-core",
        {
            "correctness": 8,
            "completeness": 7,
            "code-quality": 7,
            "efficiency": 7,
            "determinism": 8,
            "resilience": 8,
            "human-intervention": 10,
        },
    )
    _create_rating(
        rs,
        "t003",
        "myproject-api",
        {
            "correctness": 6,
            "completeness": 6,
            "code-quality": 5,
            "efficiency": 5,
            "determinism": 7,
            "resilience": 6,
            "human-intervention": 8,
        },
    )

    # Log cost and duration events
    today = time.strftime("%Y-%m-%d", time.gmtime())
    _write_events(
        al.base_dir,
        today,
        [
            {
                "timestamp": "2026-03-22T10:00:00.000Z",
                "event_type": "task_dispatch_complete",
                "task_id": "t001",
                "cost_usd": 0.50,
                "duration_s": 120.0,
                "role": "implementer",
                "project": "myproject",
            },
            {
                "timestamp": "2026-03-22T11:00:00.000Z",
                "event_type": "task_dispatch_complete",
                "task_id": "t002",
                "cost_usd": 1.20,
                "duration_s": 300.0,
                "role": "implementer",
                "project": "myproject",
            },
            {
                "timestamp": "2026-03-22T12:00:00.000Z",
                "event_type": "task_dispatch_complete",
                "task_id": "t003",
                "cost_usd": 2.00,
                "duration_s": 450.0,
                "role": "implementer",
                "project": "myproject",
            },
            {
                "timestamp": "2026-03-22T12:30:00.000Z",
                "event_type": "task_dispatch_complete",
                "task_id": "t003",
                "cost_usd": 1.50,
                "duration_s": 350.0,
                "role": "implementer",
                "project": "myproject",
                "attempt": 2,
            },
            {
                "timestamp": "2026-03-22T13:00:00.000Z",
                "event_type": "task_failed",
                "task_id": "t004",
                "cost_usd": 0.80,
                "role": "implementer",
                "project": "myproject",
            },
            {
                "timestamp": "2026-03-22T14:00:00.000Z",
                "event_type": "task_escalated",
                "task_id": "t005",
                "role": "implementer",
                "project": "myproject",
            },
        ],
    )


# ---------------------------------------------------------------------------
# Tests: _collect_project_tasks
# ---------------------------------------------------------------------------


class TestCollectProjectTasks:
    def test_matches_by_name(self, tmp_env):
        ml = tmp_env["ml"]
        _create_task(ml, "t001", "myproject-setup")
        _create_task(ml, "t002", "otherproject-core")
        tmp_env["ws"].rebuild()

        tasks = _collect_project_tasks(tmp_env["ws"], "myproject")
        assert len(tasks) == 1
        assert tasks[0]["id"] == "t001"

    def test_matches_by_description(self, tmp_env):
        ml = tmp_env["ml"]
        _create_task(ml, "t001", "setup-task")
        tmp_env["ws"].rebuild()
        # The description includes "setup-task" not "myproject",
        # so let's test falling through to all tasks
        tasks = _collect_project_tasks(tmp_env["ws"], "myproject")
        # Should return all tasks since none match
        assert len(tasks) == 1

    def test_returns_all_when_no_match(self, tmp_env):
        ml = tmp_env["ml"]
        _create_task(ml, "t001", "task-one")
        _create_task(ml, "t002", "task-two")
        tmp_env["ws"].rebuild()

        tasks = _collect_project_tasks(tmp_env["ws"], "nonexistent")
        assert len(tasks) == 2

    def test_empty_project(self, tmp_env):
        tasks = _collect_project_tasks(tmp_env["ws"], "myproject")
        assert tasks == []


# ---------------------------------------------------------------------------
# Tests: _collect_project_ratings
# ---------------------------------------------------------------------------


class TestCollectProjectRatings:
    def test_filters_by_task_ids(self, tmp_env):
        rs = tmp_env["rs"]
        _create_rating(rs, "t001", "task-one", {"correctness": 8})
        _create_rating(rs, "t002", "task-two", {"correctness": 7})
        _create_rating(rs, "t003", "task-three", {"correctness": 6})

        ratings = _collect_project_ratings(rs, {"t001", "t003"})
        assert len(ratings) == 2
        task_ids = {r.task_id for r in ratings}
        assert task_ids == {"t001", "t003"}

    def test_empty_ratings(self, tmp_env):
        ratings = _collect_project_ratings(tmp_env["rs"], {"t001"})
        assert ratings == []


# ---------------------------------------------------------------------------
# Tests: _collect_project_events
# ---------------------------------------------------------------------------


class TestCollectProjectEvents:
    def test_filters_by_task_ids(self, tmp_env):
        al = tmp_env["al"]
        today = time.strftime("%Y-%m-%d", time.gmtime())
        _write_events(
            al.base_dir,
            today,
            [
                {"event_type": "cost", "task_id": "t001", "cost_usd": 1.0},
                {"event_type": "cost", "task_id": "t002", "cost_usd": 2.0},
                {"event_type": "cost", "task_id": "t003", "cost_usd": 3.0},
            ],
        )

        events = _collect_project_events(al, {"t001", "t003"})
        assert len(events) == 2

    def test_empty_events(self, tmp_env):
        events = _collect_project_events(tmp_env["al"], {"t001"})
        assert events == []


# ---------------------------------------------------------------------------
# Tests: _compute_quality_trend
# ---------------------------------------------------------------------------


class TestComputeQualityTrend:
    def test_insufficient_data(self):
        ratings = [
            Rating(
                "t1", "task1", {"correctness": 8}, 8.0, [], "h", "2026-01-01T00:00:00Z"
            ),
            Rating(
                "t2", "task2", {"correctness": 7}, 7.0, [], "h", "2026-01-02T00:00:00Z"
            ),
        ]
        assert _compute_quality_trend(ratings) == "insufficient_data"

    def test_improving_trend(self):
        ratings = [
            Rating("t1", "task1", {}, 5.0, [], "h", "2026-01-01T00:00:00Z"),
            Rating("t2", "task2", {}, 5.0, [], "h", "2026-01-02T00:00:00Z"),
            Rating("t3", "task3", {}, 8.0, [], "h", "2026-01-03T00:00:00Z"),
            Rating("t4", "task4", {}, 9.0, [], "h", "2026-01-04T00:00:00Z"),
        ]
        assert _compute_quality_trend(ratings) == "improving"

    def test_declining_trend(self):
        ratings = [
            Rating("t1", "task1", {}, 9.0, [], "h", "2026-01-01T00:00:00Z"),
            Rating("t2", "task2", {}, 8.5, [], "h", "2026-01-02T00:00:00Z"),
            Rating("t3", "task3", {}, 5.0, [], "h", "2026-01-03T00:00:00Z"),
            Rating("t4", "task4", {}, 4.0, [], "h", "2026-01-04T00:00:00Z"),
        ]
        assert _compute_quality_trend(ratings) == "declining"

    def test_stable_trend(self):
        ratings = [
            Rating("t1", "task1", {}, 7.0, [], "h", "2026-01-01T00:00:00Z"),
            Rating("t2", "task2", {}, 7.2, [], "h", "2026-01-02T00:00:00Z"),
            Rating("t3", "task3", {}, 7.1, [], "h", "2026-01-03T00:00:00Z"),
            Rating("t4", "task4", {}, 7.3, [], "h", "2026-01-04T00:00:00Z"),
        ]
        assert _compute_quality_trend(ratings) == "stable"


# ---------------------------------------------------------------------------
# Tests: _extract_findings
# ---------------------------------------------------------------------------


class TestExtractFindings:
    def test_string_findings(self):
        tasks = [
            {"findings": ["finding one", "finding two"]},
            {"findings": ["finding three"]},
        ]
        result = _extract_findings(tasks)
        assert result == ["finding one", "finding two", "finding three"]

    def test_dict_findings(self):
        tasks = [
            {"findings": [{"content": "important finding"}]},
        ]
        result = _extract_findings(tasks)
        assert result == ["important finding"]

    def test_deduplication(self):
        tasks = [
            {"findings": ["same finding"]},
            {"findings": ["same finding"]},
        ]
        result = _extract_findings(tasks)
        assert result == ["same finding"]

    def test_empty_findings(self):
        tasks = [{"findings": []}]
        result = _extract_findings(tasks)
        assert result == []

    def test_json_string_findings(self):
        tasks = [{"findings": '["a", "b"]'}]
        result = _extract_findings(tasks)
        assert result == ["a", "b"]


# ---------------------------------------------------------------------------
# Tests: _identify_what_went_well
# ---------------------------------------------------------------------------


class TestIdentifyWhatWentWell:
    def test_high_completion_rate(self):
        tasks = [{"status": "completed", "attempt_count": 1}] * 10
        items = _identify_what_went_well(tasks, [], {})
        assert any("completion rate" in item.lower() for item in items)

    def test_first_attempt_success(self):
        tasks = [{"status": "completed", "attempt_count": 1}] * 8 + [
            {"status": "completed", "attempt_count": 2}
        ] * 2
        items = _identify_what_went_well(tasks, [], {})
        assert any("first-attempt" in item.lower() for item in items)

    def test_high_dimension_scores(self):
        items = _identify_what_went_well(
            [], [], {"correctness": 9.0, "efficiency": 8.5}
        )
        assert any("correctness" in item for item in items)
        assert any("efficiency" in item for item in items)

    def test_no_standouts(self):
        tasks = [{"status": "pending"}]
        items = _identify_what_went_well(tasks, [], {})
        assert items == ["No standout positive patterns identified"]


# ---------------------------------------------------------------------------
# Tests: _identify_what_didnt_go_well
# ---------------------------------------------------------------------------


class TestIdentifyWhatDidntGoWell:
    def test_failed_tasks(self):
        tasks = [
            {"status": "failed", "name": "broken-task", "id": "t001"},
        ]
        items = _identify_what_didnt_go_well(tasks, [], {})
        assert any("failed" in item.lower() for item in items)

    def test_escalated_tasks(self):
        tasks = [
            {"status": "escalated", "name": "hard-task", "id": "t001"},
        ]
        items = _identify_what_didnt_go_well(tasks, [], {})
        assert any("escalation" in item.lower() for item in items)

    def test_low_dimensions(self):
        items = _identify_what_didnt_go_well(
            [], [], {"efficiency": 4.0, "correctness": 5.5}
        )
        assert any("efficiency" in item for item in items)
        assert any("correctness" in item for item in items)

    def test_no_issues(self):
        tasks = [{"status": "completed", "attempt_count": 1}]
        items = _identify_what_didnt_go_well(tasks, [], {"correctness": 8.0})
        assert items == ["No major issues identified"]


# ---------------------------------------------------------------------------
# Tests: _generate_recommendations
# ---------------------------------------------------------------------------


class TestGenerateRecommendations:
    def test_low_completion_rate(self):
        retro = Retrospective(
            project_name="test",
            generated_at="2026-03-22T00:00:00Z",
            total_tasks=10,
            completed_tasks=5,
        )
        recs = _generate_recommendations(retro, [], [])
        assert any("completion rate" in r.lower() for r in recs)

    def test_declining_quality(self):
        retro = Retrospective(
            project_name="test",
            generated_at="2026-03-22T00:00:00Z",
            quality_trend="declining",
        )
        recs = _generate_recommendations(retro, [], [])
        assert any("quality declined" in r.lower() for r in recs)

    def test_improving_quality(self):
        retro = Retrospective(
            project_name="test",
            generated_at="2026-03-22T00:00:00Z",
            quality_trend="improving",
        )
        recs = _generate_recommendations(retro, [], [])
        assert any("quality improved" in r.lower() for r in recs)

    def test_worst_dimensions(self):
        retro = Retrospective(
            project_name="test",
            generated_at="2026-03-22T00:00:00Z",
            worst_dimensions=["efficiency"],
            dimension_averages={"efficiency": 5.0},
        )
        recs = _generate_recommendations(retro, [], [])
        assert any("efficiency" in r for r in recs)

    def test_high_escalation_rate(self):
        retro = Retrospective(
            project_name="test",
            generated_at="2026-03-22T00:00:00Z",
            total_tasks=10,
            escalated_tasks=3,
        )
        recs = _generate_recommendations(retro, [], [])
        assert any("escalation rate" in r.lower() for r in recs)

    def test_over_budget(self):
        retro = Retrospective(
            project_name="test",
            generated_at="2026-03-22T00:00:00Z",
            total_cost_usd=70.0,
            cost_estimate_usd=50.0,
        )
        recs = _generate_recommendations(retro, [], [])
        assert any("over budget" in r.lower() for r in recs)

    def test_no_issues_default(self):
        retro = Retrospective(
            project_name="test",
            generated_at="2026-03-22T00:00:00Z",
            total_tasks=10,
            completed_tasks=10,
            quality_trend="stable",
        )
        recs = _generate_recommendations(retro, [], [])
        assert any("executed well" in r.lower() for r in recs)


# ---------------------------------------------------------------------------
# Tests: generate_retrospective (integration)
# ---------------------------------------------------------------------------


class TestGenerateRetrospective:
    def test_full_project_retrospective(self, tmp_env):
        _build_sample_project(tmp_env)
        ws = WorkState(
            tmp_env["tmp_path"] / "data" / "state.db",
            tmp_env["ml"],
        )

        retro = generate_retrospective(
            project_name="myproject",
            work_state=ws,
            audit_log=tmp_env["al"],
            rating_store=tmp_env["rs"],
            cost_estimate_usd=5.00,
        )

        assert retro.project_name == "myproject"
        assert retro.total_tasks == 5
        assert retro.completed_tasks == 3
        assert retro.failed_tasks == 1
        assert retro.escalated_tasks == 1
        assert retro.total_cost_usd > 0
        assert retro.total_duration_s > 0
        assert retro.avg_duration_s > 0
        assert retro.avg_overall_rating > 0
        assert len(retro.dimension_averages) > 0
        assert len(retro.best_dimensions) > 0
        assert len(retro.worst_dimensions) > 0
        assert len(retro.what_went_well) > 0
        assert len(retro.what_didnt_go_well) > 0
        assert len(retro.recommendations) > 0
        assert retro.generated_at != ""
        assert retro.cost_estimate_usd == 5.00

    def test_empty_project(self, tmp_env):
        retro = generate_retrospective(
            project_name="empty",
            work_state=tmp_env["ws"],
            audit_log=tmp_env["al"],
            rating_store=tmp_env["rs"],
        )

        assert retro.project_name == "empty"
        assert retro.total_tasks == 0
        assert retro.completed_tasks == 0
        assert retro.total_cost_usd == 0.0
        assert retro.avg_overall_rating == 0.0
        assert retro.quality_trend == "insufficient_data"

    def test_all_completed_project(self, tmp_env):
        ml = tmp_env["ml"]
        rs = tmp_env["rs"]

        for i in range(5):
            _create_task(ml, f"t{i:03d}", f"proj-task-{i}", status="completed")
            _create_rating(
                rs,
                f"t{i:03d}",
                f"proj-task-{i}",
                {
                    "correctness": 9,
                    "completeness": 8,
                    "code-quality": 8,
                    "efficiency": 8,
                    "determinism": 9,
                    "resilience": 9,
                    "human-intervention": 10,
                },
            )

        ws = WorkState(
            tmp_env["tmp_path"] / "data" / "state.db",
            tmp_env["ml"],
        )

        retro = generate_retrospective(
            project_name="proj",
            work_state=ws,
            audit_log=tmp_env["al"],
            rating_store=rs,
        )

        assert retro.total_tasks == 5
        assert retro.completed_tasks == 5
        assert retro.failed_tasks == 0
        assert retro.escalated_tasks == 0
        assert any("completion rate" in item.lower() for item in retro.what_went_well)

    def test_cost_vs_estimate(self, tmp_env):
        ml = tmp_env["ml"]
        al = tmp_env["al"]
        _create_task(ml, "t001", "budget-task", status="completed")
        ws = WorkState(
            tmp_env["tmp_path"] / "data" / "state.db",
            tmp_env["ml"],
        )

        today = time.strftime("%Y-%m-%d", time.gmtime())
        _write_events(
            al.base_dir,
            today,
            [
                {
                    "event_type": "task_dispatch_complete",
                    "task_id": "t001",
                    "cost_usd": 15.0,
                    "duration_s": 100.0,
                    "timestamp": "2026-03-22T10:00:00.000Z",
                },
            ],
        )

        retro = generate_retrospective(
            project_name="budget",
            work_state=ws,
            audit_log=al,
            rating_store=tmp_env["rs"],
            cost_estimate_usd=10.0,
        )

        assert retro.total_cost_usd == 15.0
        assert retro.cost_estimate_usd == 10.0
        # Should recommend reviewing cost estimation
        assert any("budget" in r.lower() for r in retro.recommendations)

    def test_findings_aggregation(self, tmp_env):
        ml = tmp_env["ml"]
        _create_task(
            ml,
            "t001",
            "findings-task",
            status="completed",
            findings=["Found X", "Found Y"],
        )
        _create_task(
            ml, "t002", "findings-task-2", status="completed", findings=["Found Z"]
        )
        ws = WorkState(
            tmp_env["tmp_path"] / "data" / "state.db",
            tmp_env["ml"],
        )

        retro = generate_retrospective(
            project_name="findings",
            work_state=ws,
            audit_log=tmp_env["al"],
            rating_store=tmp_env["rs"],
        )

        assert "Found X" in retro.top_findings
        assert "Found Y" in retro.top_findings
        assert "Found Z" in retro.top_findings


# ---------------------------------------------------------------------------
# Tests: format_retrospective
# ---------------------------------------------------------------------------


class TestFormatRetrospective:
    def test_basic_formatting(self):
        retro = Retrospective(
            project_name="testproj",
            generated_at="2026-03-22T00:00:00Z",
            total_tasks=10,
            completed_tasks=8,
            failed_tasks=1,
            escalated_tasks=1,
            total_cost_usd=25.50,
            cost_estimate_usd=20.00,
            total_duration_s=1200.0,
            avg_duration_s=150.0,
            avg_overall_rating=7.5,
            dimension_averages={"correctness": 8.0, "efficiency": 6.5},
            best_dimensions=["correctness"],
            worst_dimensions=["efficiency"],
            quality_trend="stable",
            what_went_well=["High quality code"],
            what_didnt_go_well=["Some tasks failed"],
            top_findings=["Finding 1"],
            recommendations=["Improve testing"],
        )

        output = format_retrospective(retro)

        assert "testproj" in output
        assert "Task Summary" in output
        assert "10" in output  # total tasks
        assert "8" in output  # completed
        assert "Cost & Duration" in output
        assert "$25.50" in output
        assert "$20.00" in output
        assert "over" in output  # cost variance direction
        assert "Quality" in output
        assert "7.5" in output
        assert "stable" in output
        assert "What Went Well" in output
        assert "High quality code" in output
        assert "What Didn't Go Well" in output
        assert "Some tasks failed" in output
        assert "Top Findings" in output
        assert "Finding 1" in output
        assert "Recommendations" in output
        assert "Improve testing" in output

    def test_no_cost_estimate(self):
        retro = Retrospective(
            project_name="nocost",
            generated_at="2026-03-22T00:00:00Z",
            total_cost_usd=10.0,
        )
        output = format_retrospective(retro)
        assert "Cost estimate" not in output
        assert "Variance" not in output

    def test_under_budget(self):
        retro = Retrospective(
            project_name="under",
            generated_at="2026-03-22T00:00:00Z",
            total_cost_usd=8.0,
            cost_estimate_usd=20.0,
        )
        output = format_retrospective(retro)
        assert "under" in output

    def test_no_findings(self):
        retro = Retrospective(
            project_name="nofind",
            generated_at="2026-03-22T00:00:00Z",
        )
        output = format_retrospective(retro)
        assert "Top Findings" not in output


# ---------------------------------------------------------------------------
# Tests: retrospective_to_markdown
# ---------------------------------------------------------------------------


class TestRetrospectiveToMarkdown:
    def test_markdown_has_frontmatter(self):
        retro = Retrospective(
            project_name="mdtest",
            generated_at="2026-03-22T00:00:00Z",
            total_tasks=5,
            completed_tasks=4,
            failed_tasks=1,
            total_cost_usd=10.0,
            avg_overall_rating=7.0,
            what_went_well=["Good stuff"],
            what_didnt_go_well=["Bad stuff"],
            recommendations=["Do better"],
        )

        md = retrospective_to_markdown(retro)

        # Check YAML frontmatter
        assert md.startswith("---")
        assert "type: task-outcome" in md
        assert "project: mdtest" in md
        assert "retrospective" in md
        assert "project-review" in md
        assert "source: system" in md

    def test_markdown_sections(self):
        retro = Retrospective(
            project_name="mdtest",
            generated_at="2026-03-22T00:00:00Z",
            total_tasks=5,
            completed_tasks=4,
            dimension_averages={"correctness": 8.0},
            best_dimensions=["correctness"],
            worst_dimensions=["efficiency"],
            what_went_well=["Good"],
            what_didnt_go_well=["Bad"],
            top_findings=["Finding"],
            recommendations=["Rec"],
            task_ratings=[
                {
                    "task_id": "t001",
                    "task_name": "test-task",
                    "overall": 8.0,
                    "scores": {"correctness": 8},
                    "method": "heuristic",
                    "flags": [],
                }
            ],
        )

        md = retrospective_to_markdown(retro)

        assert "# Retrospective: mdtest" in md
        assert "## Task Summary" in md
        assert "## Cost & Duration" in md
        assert "## Quality" in md
        assert "## What Went Well" in md
        assert "## What Didn't Go Well" in md
        assert "## Top Findings" in md
        assert "## Recommendations" in md
        assert "## Task Ratings Detail" in md

    def test_markdown_cost_variance(self):
        retro = Retrospective(
            project_name="costtest",
            generated_at="2026-03-22T00:00:00Z",
            total_cost_usd=15.0,
            cost_estimate_usd=10.0,
            what_went_well=["X"],
            what_didnt_go_well=["Y"],
            recommendations=["Z"],
        )

        md = retrospective_to_markdown(retro)

        assert "**Cost estimate**: $10.00" in md
        assert "**Variance**" in md
        assert "$5.00 over" in md


# ---------------------------------------------------------------------------
# Tests: save_retrospective
# ---------------------------------------------------------------------------


class TestSaveRetrospective:
    def test_saves_to_knowledge_store(self, tmp_env):
        retro = Retrospective(
            project_name="savetest",
            generated_at="2026-03-22T00:00:00Z",
            total_tasks=3,
            completed_tasks=2,
            what_went_well=["Good"],
            what_didnt_go_well=["Bad"],
            recommendations=["Improve"],
        )

        doc_id = save_retrospective(retro, tmp_env["ks"])

        assert doc_id is not None
        assert doc_id != ""

        # Verify it's in the knowledge store
        doc = tmp_env["ks"].get(doc_id)
        assert doc is not None
        assert "savetest" in doc["content"]
        assert doc["type"] == "task-outcome"
        assert doc["project"] == "savetest"

    def test_saved_document_is_searchable(self, tmp_env):
        retro = Retrospective(
            project_name="searchtest",
            generated_at="2026-03-22T00:00:00Z",
            total_tasks=1,
            what_went_well=["Unique finding XYZ"],
            what_didnt_go_well=["Issue ABC"],
            recommendations=["Fix it"],
        )

        save_retrospective(retro, tmp_env["ks"])

        # Should be findable via search
        results = tmp_env["ks"].search("searchtest retrospective")
        assert len(results) > 0
        assert any("searchtest" in r.get("title", "") for r in results)


# ---------------------------------------------------------------------------
# Tests: CLI integration
# ---------------------------------------------------------------------------


class TestRetroCLI:
    def test_retro_command_basic(self, tmp_env, monkeypatch):
        """Test the retro CLI command with sample data."""
        from click.testing import CliRunner
        from corc.cli import cli

        _build_sample_project(tmp_env)

        # Monkeypatch get_paths to point to our tmp dirs
        def mock_get_paths(root=None):
            return {
                "root": tmp_env["tmp_path"],
                "mutations": tmp_env["tmp_path"] / "data" / "mutations.jsonl",
                "state_db": tmp_env["tmp_path"] / "data" / "state.db",
                "events_dir": tmp_env["tmp_path"] / "data" / "events",
                "sessions_dir": tmp_env["tmp_path"] / "data" / "sessions",
                "knowledge_dir": tmp_env["tmp_path"] / "knowledge",
                "knowledge_db": tmp_env["tmp_path"] / "data" / "knowledge.db",
                "corc_dir": tmp_env["tmp_path"] / ".corc",
                "ratings_dir": tmp_env["tmp_path"] / "data" / "ratings",
            }

        monkeypatch.setattr("corc.cli.get_paths", mock_get_paths)

        runner = CliRunner()
        result = runner.invoke(cli, ["retro", "myproject"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "myproject" in result.output
        assert "Task Summary" in result.output
        assert "Cost & Duration" in result.output
        assert "Quality" in result.output
        assert "What Went Well" in result.output
        assert "What Didn't Go Well" in result.output
        assert "Recommendations" in result.output
        assert "knowledge store" in result.output.lower()

    def test_retro_command_with_estimate(self, tmp_env, monkeypatch):
        """Test the retro CLI command with --estimate flag."""
        from click.testing import CliRunner
        from corc.cli import cli

        _build_sample_project(tmp_env)

        def mock_get_paths(root=None):
            return {
                "root": tmp_env["tmp_path"],
                "mutations": tmp_env["tmp_path"] / "data" / "mutations.jsonl",
                "state_db": tmp_env["tmp_path"] / "data" / "state.db",
                "events_dir": tmp_env["tmp_path"] / "data" / "events",
                "sessions_dir": tmp_env["tmp_path"] / "data" / "sessions",
                "knowledge_dir": tmp_env["tmp_path"] / "knowledge",
                "knowledge_db": tmp_env["tmp_path"] / "data" / "knowledge.db",
                "corc_dir": tmp_env["tmp_path"] / ".corc",
                "ratings_dir": tmp_env["tmp_path"] / "data" / "ratings",
            }

        monkeypatch.setattr("corc.cli.get_paths", mock_get_paths)

        runner = CliRunner()
        result = runner.invoke(cli, ["retro", "myproject", "--estimate", "3.00"])

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert "$3.00" in result.output
        assert "over" in result.output  # cost exceeded estimate

    def test_retro_command_empty_project(self, tmp_env, monkeypatch):
        """Test retro on a project with no tasks."""
        from click.testing import CliRunner
        from corc.cli import cli

        def mock_get_paths(root=None):
            return {
                "root": tmp_env["tmp_path"],
                "mutations": tmp_env["tmp_path"] / "data" / "mutations.jsonl",
                "state_db": tmp_env["tmp_path"] / "data" / "state.db",
                "events_dir": tmp_env["tmp_path"] / "data" / "events",
                "sessions_dir": tmp_env["tmp_path"] / "data" / "sessions",
                "knowledge_dir": tmp_env["tmp_path"] / "knowledge",
                "knowledge_db": tmp_env["tmp_path"] / "data" / "knowledge.db",
                "corc_dir": tmp_env["tmp_path"] / ".corc",
                "ratings_dir": tmp_env["tmp_path"] / "data" / "ratings",
            }

        monkeypatch.setattr("corc.cli.get_paths", mock_get_paths)

        runner = CliRunner()
        result = runner.invoke(cli, ["retro", "nonexistent"])

        assert result.exit_code == 0
        assert "nonexistent" in result.output


# ---------------------------------------------------------------------------
# Tests: Retrospective dataclass
# ---------------------------------------------------------------------------


class TestRetrospectiveDataclass:
    def test_default_values(self):
        retro = Retrospective(
            project_name="test",
            generated_at="2026-03-22T00:00:00Z",
        )

        assert retro.total_tasks == 0
        assert retro.completed_tasks == 0
        assert retro.failed_tasks == 0
        assert retro.escalated_tasks == 0
        assert retro.total_cost_usd == 0.0
        assert retro.cost_estimate_usd is None
        assert retro.total_duration_s == 0.0
        assert retro.avg_duration_s == 0.0
        assert retro.avg_overall_rating == 0.0
        assert retro.dimension_averages == {}
        assert retro.best_dimensions == []
        assert retro.worst_dimensions == []
        assert retro.quality_trend == ""
        assert retro.what_went_well == []
        assert retro.what_didnt_go_well == []
        assert retro.top_findings == []
        assert retro.recommendations == []
        assert retro.task_ratings == []


# ---------------------------------------------------------------------------
# Tests: end-to-end with full sample data
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_retrospective_flow(self, tmp_env):
        """End-to-end: create project data -> generate retro -> save -> verify."""
        _build_sample_project(tmp_env)

        # Must create a fresh WorkState to pick up mutations
        ws = WorkState(
            tmp_env["tmp_path"] / "data" / "state.db",
            tmp_env["ml"],
        )

        # Generate retrospective
        retro = generate_retrospective(
            project_name="myproject",
            work_state=ws,
            audit_log=tmp_env["al"],
            rating_store=tmp_env["rs"],
            cost_estimate_usd=5.00,
        )

        # Verify structure
        assert retro.project_name == "myproject"
        assert retro.total_tasks == 5
        assert retro.completed_tasks == 3
        assert retro.failed_tasks == 1
        assert retro.escalated_tasks == 1

        # Cost should sum correctly
        assert retro.total_cost_usd == pytest.approx(6.0, abs=0.01)

        # Duration should be from dispatch_complete events only
        assert retro.total_duration_s > 0

        # Ratings averaged
        assert retro.avg_overall_rating > 0
        assert "correctness" in retro.dimension_averages
        assert "efficiency" in retro.dimension_averages

        # Quality trend needs 4+ ratings; we have 3 so should be insufficient
        assert retro.quality_trend == "insufficient_data"

        # Format it
        output = format_retrospective(retro)
        assert "myproject" in output
        assert "5" in output  # total tasks

        # Convert to markdown
        md = retrospective_to_markdown(retro)
        assert "type: task-outcome" in md
        assert "project: myproject" in md

        # Save to knowledge store
        doc_id = save_retrospective(retro, tmp_env["ks"])
        assert doc_id is not None

        # Verify persistence
        doc = tmp_env["ks"].get(doc_id)
        assert doc is not None
        assert doc["type"] == "task-outcome"
        assert doc["project"] == "myproject"
        assert "Retrospective" in doc["title"]

    def test_retrospective_with_quality_trend(self, tmp_env):
        """Test with enough ratings to compute a quality trend."""
        ml = tmp_env["ml"]
        rs = tmp_env["rs"]

        # Create 6 tasks with ratings that show an improving trend
        for i in range(6):
            _create_task(ml, f"t{i:03d}", f"trendproj-task-{i}", status="completed")
            base_score = 5 + i  # Scores improve: 5, 6, 7, 8, 9, 10
            base_score = min(base_score, 10)
            _create_rating(
                rs,
                f"t{i:03d}",
                f"trendproj-task-{i}",
                {
                    "correctness": base_score,
                    "completeness": base_score,
                    "code-quality": base_score,
                    "efficiency": base_score,
                    "determinism": base_score,
                    "resilience": base_score,
                    "human-intervention": base_score,
                },
            )

        ws = WorkState(
            tmp_env["tmp_path"] / "data" / "state.db",
            tmp_env["ml"],
        )

        retro = generate_retrospective(
            project_name="trendproj",
            work_state=ws,
            audit_log=tmp_env["al"],
            rating_store=rs,
        )

        assert retro.quality_trend == "improving"
