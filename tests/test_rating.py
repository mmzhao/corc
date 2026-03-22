"""Tests for rating engine — scoring, storage, heuristics, evaluator parsing, CLI."""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from corc.rating import (
    DIMENSIONS,
    DIMENSION_NAMES,
    Rating,
    RatingEngine,
    RatingStore,
    build_evaluator_prompt,
    flagged_dimensions,
    format_dimension_drilldown,
    format_rating,
    format_trend,
    heuristic_scores,
    parse_evaluator_response,
    weighted_score,
)
from corc.audit import AuditLog
from corc.mutations import MutationLog
from corc.sessions import SessionLogger
from corc.state import WorkState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_env(tmp_path):
    """Set up a full temporary environment with all data layers."""
    mutations_path = tmp_path / "data" / "mutations.jsonl"
    mutations_path.parent.mkdir(parents=True)
    state_db = tmp_path / "data" / "state.db"
    events_dir = tmp_path / "data" / "events"
    events_dir.mkdir(parents=True)
    sessions_dir = tmp_path / "data" / "sessions"
    sessions_dir.mkdir(parents=True)
    ratings_dir = tmp_path / "data" / "ratings"
    ratings_dir.mkdir(parents=True)

    ml = MutationLog(mutations_path)
    ws = WorkState(state_db, ml)
    al = AuditLog(events_dir)
    sl = SessionLogger(sessions_dir)
    rs = RatingStore(ratings_dir)

    return {
        "tmp_path": tmp_path,
        "ml": ml,
        "ws": ws,
        "al": al,
        "sl": sl,
        "rs": rs,
    }


@pytest.fixture
def completed_task(tmp_env):
    """Create a completed task and return the env + task_id."""
    ml, ws = tmp_env["ml"], tmp_env["ws"]
    task_id = "test-001"
    ml.append(
        "task_created",
        {
            "id": task_id,
            "name": "test-task",
            "description": "A test task",
            "role": "implementer",
            "depends_on": [],
            "done_when": "Tests pass",
            "checklist": [
                {"item": "Write code", "done": True},
                {"item": "Write tests", "done": True},
                {"item": "Run lint", "done": False},
            ],
        },
        reason="test setup",
    )
    ml.append(
        "task_completed",
        {"pr_url": "https://github.com/test/pr/1", "findings": ["found a thing"]},
        reason="test complete",
        task_id=task_id,
    )
    ws.refresh()
    tmp_env["task_id"] = task_id
    return tmp_env


# ---------------------------------------------------------------------------
# Dimension definitions
# ---------------------------------------------------------------------------


class TestDimensions:
    def test_seven_dimensions(self):
        assert len(DIMENSIONS) == 7

    def test_dimension_names(self):
        expected = [
            "correctness",
            "completeness",
            "code-quality",
            "efficiency",
            "determinism",
            "resilience",
            "human-intervention",
        ]
        assert DIMENSION_NAMES == expected

    def test_weights_sum_to_one(self):
        total = sum(info["weight"] for info in DIMENSIONS.values())
        assert abs(total - 1.0) < 1e-9

    def test_each_dimension_has_description(self):
        for dim, info in DIMENSIONS.items():
            assert "description" in info
            assert len(info["description"]) > 0

    def test_each_dimension_has_weight(self):
        for dim, info in DIMENSIONS.items():
            assert 0 < info["weight"] <= 1.0


# ---------------------------------------------------------------------------
# Weighted score computation
# ---------------------------------------------------------------------------


class TestWeightedScore:
    def test_perfect_scores(self):
        scores = {dim: 10 for dim in DIMENSION_NAMES}
        assert weighted_score(scores) == 10.0

    def test_minimum_scores(self):
        scores = {dim: 1 for dim in DIMENSION_NAMES}
        assert weighted_score(scores) == 1.0

    def test_mixed_scores(self):
        scores = {dim: 5 for dim in DIMENSION_NAMES}
        assert abs(weighted_score(scores) - 5.0) < 1e-9

    def test_empty_scores(self):
        assert weighted_score({}) == 0.0

    def test_partial_scores(self):
        # Only correctness (weight 0.25)
        scores = {"correctness": 8}
        # Should normalize: 8 * 0.25 / 0.25 = 8.0
        assert abs(weighted_score(scores) - 8.0) < 1e-9

    def test_weighted_correctly(self):
        scores = {
            "correctness": 10,  # 0.25
            "completeness": 10,  # 0.15
            "code-quality": 10,  # 0.15
            "efficiency": 10,  # 0.15
            "determinism": 10,  # 0.10
            "resilience": 10,  # 0.10
            "human-intervention": 0,  # 0.10 — but clamped to 1 in real use
        }
        # Overall = (10*0.25 + 10*0.15 + 10*0.15 + 10*0.15 + 10*0.10 + 10*0.10 + 0*0.10) / 1.0
        expected = 2.5 + 1.5 + 1.5 + 1.5 + 1.0 + 1.0 + 0.0
        assert abs(weighted_score(scores) - expected) < 1e-9


# ---------------------------------------------------------------------------
# Flagged dimensions
# ---------------------------------------------------------------------------


class TestFlaggedDimensions:
    def test_no_flags_all_above(self):
        scores = {dim: 8 for dim in DIMENSION_NAMES}
        assert flagged_dimensions(scores) == []

    def test_flags_below_threshold(self):
        scores = {dim: 8 for dim in DIMENSION_NAMES}
        scores["correctness"] = 6
        scores["efficiency"] = 5
        flags = flagged_dimensions(scores)
        assert "correctness" in flags
        assert "efficiency" in flags
        assert len(flags) == 2

    def test_threshold_boundary(self):
        # Score of exactly 7 should NOT be flagged (< 7 is flagged)
        scores = {"correctness": 7, "completeness": 6}
        flags = flagged_dimensions(scores)
        assert "correctness" not in flags
        assert "completeness" in flags

    def test_custom_threshold(self):
        scores = {"correctness": 8}
        flags = flagged_dimensions(scores, threshold=9)
        assert "correctness" in flags


# ---------------------------------------------------------------------------
# Rating dataclass
# ---------------------------------------------------------------------------


class TestRating:
    def test_roundtrip(self):
        rating = Rating(
            task_id="t1",
            task_name="test",
            scores={"correctness": 8, "completeness": 7},
            overall=7.5,
            flags=[],
            method="heuristic",
            timestamp="2026-03-22T10:00:00Z",
            metadata={"role": "implementer"},
        )
        d = rating.to_dict()
        restored = Rating.from_dict(d)
        assert restored.task_id == rating.task_id
        assert restored.scores == rating.scores
        assert restored.overall == rating.overall
        assert restored.method == rating.method
        assert restored.metadata == rating.metadata

    def test_to_dict_json_serializable(self):
        rating = Rating(
            task_id="t1",
            task_name="test",
            scores={"correctness": 8},
            overall=8.0,
            flags=[],
            method="heuristic",
            timestamp="2026-03-22T10:00:00Z",
        )
        # Should not raise
        json.dumps(rating.to_dict())


# ---------------------------------------------------------------------------
# RatingStore
# ---------------------------------------------------------------------------


class TestRatingStore:
    def test_save_and_read(self, tmp_path):
        store = RatingStore(tmp_path / "ratings")
        rating = Rating(
            task_id="t1",
            task_name="task-one",
            scores={dim: 7 for dim in DIMENSION_NAMES},
            overall=7.0,
            flags=[],
            method="heuristic",
            timestamp="2026-03-22T10:00:00Z",
        )
        store.save(rating)
        ratings = store.read_all()
        assert len(ratings) == 1
        assert ratings[0].task_id == "t1"

    def test_append_only(self, tmp_path):
        store = RatingStore(tmp_path / "ratings")
        for i in range(3):
            store.save(
                Rating(
                    task_id=f"t{i}",
                    task_name=f"task-{i}",
                    scores={"correctness": 5 + i},
                    overall=float(5 + i),
                    flags=[],
                    method="heuristic",
                    timestamp=f"2026-03-22T10:0{i}:00Z",
                )
            )
        assert len(store.read_all()) == 3

    def test_get_for_task(self, tmp_path):
        store = RatingStore(tmp_path / "ratings")
        store.save(
            Rating(
                task_id="t1",
                task_name="first",
                scores={"correctness": 5},
                overall=5.0,
                flags=[],
                method="heuristic",
                timestamp="2026-03-22T10:00:00Z",
            )
        )
        store.save(
            Rating(
                task_id="t1",
                task_name="first-rerated",
                scores={"correctness": 8},
                overall=8.0,
                flags=[],
                method="evaluator",
                timestamp="2026-03-22T11:00:00Z",
            )
        )
        r = store.get_for_task("t1")
        assert r is not None
        assert r.scores["correctness"] == 8  # Latest

    def test_get_for_task_not_found(self, tmp_path):
        store = RatingStore(tmp_path / "ratings")
        assert store.get_for_task("nonexistent") is None

    def test_is_rated(self, tmp_path):
        store = RatingStore(tmp_path / "ratings")
        store.save(
            Rating(
                task_id="t1",
                task_name="test",
                scores={"correctness": 7},
                overall=7.0,
                flags=[],
                method="heuristic",
                timestamp="2026-03-22T10:00:00Z",
            )
        )
        assert store.is_rated("t1") is True
        assert store.is_rated("t2") is False

    def test_trend(self, tmp_path):
        store = RatingStore(tmp_path / "ratings")
        for i in range(5):
            store.save(
                Rating(
                    task_id=f"t{i}",
                    task_name=f"task-{i}",
                    scores={"correctness": 5 + i},
                    overall=float(5 + i),
                    flags=[],
                    method="heuristic",
                    timestamp=f"2026-03-22T10:0{i}:00Z",
                )
            )
        trend = store.get_trend(last_n=3)
        assert len(trend) == 3
        # Sorted by timestamp, last 3
        assert trend[0].task_id == "t2"
        assert trend[2].task_id == "t4"

    def test_get_by_dimension(self, tmp_path):
        store = RatingStore(tmp_path / "ratings")
        store.save(
            Rating(
                task_id="t1",
                task_name="task-one",
                scores={"correctness": 9, "efficiency": 6},
                overall=7.5,
                flags=[],
                method="heuristic",
                timestamp="2026-03-22T10:00:00Z",
            )
        )
        store.save(
            Rating(
                task_id="t2",
                task_name="task-two",
                scores={"correctness": 5},
                overall=5.0,
                flags=["correctness"],
                method="heuristic",
                timestamp="2026-03-22T11:00:00Z",
            )
        )
        results = store.get_by_dimension("correctness")
        assert len(results) == 2
        assert results[0]["score"] == 9
        assert results[1]["score"] == 5

    def test_get_by_dimension_invalid(self, tmp_path):
        store = RatingStore(tmp_path / "ratings")
        with pytest.raises(ValueError, match="Unknown dimension"):
            store.get_by_dimension("nonexistent")

    def test_empty_store(self, tmp_path):
        store = RatingStore(tmp_path / "ratings")
        assert store.read_all() == []
        assert store.get_trend() == []
        assert store.get_for_task("t1") is None
        assert store.is_rated("t1") is False

    def test_jsonl_file_created_in_data_ratings(self, tmp_path):
        ratings_dir = tmp_path / "data" / "ratings"
        store = RatingStore(ratings_dir)
        store.save(
            Rating(
                task_id="t1",
                task_name="test",
                scores={"correctness": 7},
                overall=7.0,
                flags=[],
                method="heuristic",
                timestamp="2026-03-22T10:00:00Z",
            )
        )
        assert (ratings_dir / "ratings.jsonl").exists()
        content = (ratings_dir / "ratings.jsonl").read_text()
        assert "t1" in content
        # Verify it's valid JSONL
        parsed = json.loads(content.strip())
        assert parsed["task_id"] == "t1"


# ---------------------------------------------------------------------------
# Heuristic scoring
# ---------------------------------------------------------------------------


class TestHeuristicScores:
    def test_completed_task(self):
        task = {"status": "completed", "checklist": [], "attempt_count": 1}
        scores = heuristic_scores(task, [], [])
        assert scores["correctness"] == 8
        assert scores["completeness"] == 8
        assert scores["efficiency"] == 9
        assert scores["human-intervention"] == 10

    def test_failed_task(self):
        task = {"status": "failed", "checklist": [], "attempt_count": 1}
        scores = heuristic_scores(task, [], [])
        assert scores["correctness"] == 3

    def test_checklist_progress(self):
        task = {
            "status": "completed",
            "checklist": [
                {"item": "a", "done": True},
                {"item": "b", "done": True},
                {"item": "c", "done": False},
                {"item": "d", "done": False},
            ],
            "attempt_count": 1,
        }
        scores = heuristic_scores(task, [], [])
        # 2/4 = 50% → score 5
        assert scores["completeness"] == 5

    def test_checklist_all_done(self):
        task = {
            "status": "completed",
            "checklist": [
                {"item": "a", "done": True},
                {"item": "b", "done": True},
            ],
            "attempt_count": 1,
        }
        scores = heuristic_scores(task, [], [])
        assert scores["completeness"] == 10

    def test_high_attempt_count_lowers_efficiency(self):
        task = {"status": "completed", "checklist": [], "attempt_count": 4}
        scores = heuristic_scores(task, [], [])
        assert scores["efficiency"] == 3

    def test_micro_deviations_lower_determinism(self):
        task = {
            "status": "completed",
            "checklist": [],
            "attempt_count": 1,
            "micro_deviations": ["fix1", "fix2", "fix3"],
        }
        scores = heuristic_scores(task, [], [])
        assert scores["determinism"] == 5

    def test_failure_events_lower_resilience(self):
        events = [
            {"event_type": "task_failed", "timestamp": "2026-03-22T10:00:00Z"},
            {"event_type": "task_failed", "timestamp": "2026-03-22T10:01:00Z"},
        ]
        task = {"status": "completed", "checklist": [], "attempt_count": 1}
        scores = heuristic_scores(task, events, [])
        assert scores["resilience"] == 5

    def test_escalation_events_lower_human_intervention(self):
        events = [
            {"event_type": "task_escalated", "timestamp": "2026-03-22T10:00:00Z"},
        ]
        task = {"status": "completed", "checklist": [], "attempt_count": 1}
        scores = heuristic_scores(task, events, [])
        assert scores["human-intervention"] == 6

    def test_all_dimensions_present(self):
        task = {"status": "completed", "checklist": [], "attempt_count": 1}
        scores = heuristic_scores(task, [], [])
        for dim in DIMENSION_NAMES:
            assert dim in scores
            assert 1 <= scores[dim] <= 10

    def test_checklist_as_string(self):
        """Checklist stored as JSON string should be parsed."""
        task = {
            "status": "completed",
            "checklist": json.dumps([{"item": "a", "done": True}]),
            "attempt_count": 1,
        }
        scores = heuristic_scores(task, [], [])
        assert scores["completeness"] == 10


# ---------------------------------------------------------------------------
# Evaluator prompt building
# ---------------------------------------------------------------------------


class TestBuildEvaluatorPrompt:
    def test_basic_prompt(self):
        task = {
            "name": "test-task",
            "id": "t1",
            "status": "completed",
            "done_when": "Tests pass",
            "role": "implementer",
            "attempt_count": 1,
        }
        prompt = build_evaluator_prompt(task, [], "")
        assert "test-task" in prompt
        assert "t1" in prompt
        assert "Tests pass" in prompt
        assert "Score this task" in prompt

    def test_includes_events(self):
        task = {
            "name": "t",
            "id": "t1",
            "status": "completed",
            "done_when": "x",
            "role": "impl",
            "attempt_count": 1,
        }
        events = [
            {"event_type": "task_failed", "timestamp": "2026-03-22T10:00:00.000Z"},
            {"event_type": "task_completed", "timestamp": "2026-03-22T10:01:00.000Z"},
        ]
        prompt = build_evaluator_prompt(task, events, "")
        assert "Event History" in prompt
        assert "Failures: 1" in prompt

    def test_includes_checklist(self):
        task = {
            "name": "t",
            "id": "t1",
            "status": "completed",
            "done_when": "x",
            "role": "impl",
            "attempt_count": 1,
            "checklist": [
                {"item": "Write code", "done": True},
                {"item": "Tests", "done": False},
            ],
        }
        prompt = build_evaluator_prompt(task, [], "")
        assert "Checklist" in prompt
        assert "Write code" in prompt

    def test_includes_spec_excerpt(self):
        task = {
            "name": "t",
            "id": "t1",
            "status": "completed",
            "done_when": "x",
            "role": "impl",
            "attempt_count": 1,
        }
        prompt = build_evaluator_prompt(task, [], "", spec_excerpt="Rating rubric here")
        assert "Rating rubric here" in prompt


# ---------------------------------------------------------------------------
# Evaluator response parsing
# ---------------------------------------------------------------------------


class TestParseEvaluatorResponse:
    def test_clean_json(self):
        response = json.dumps(
            {
                "correctness": 8,
                "completeness": 7,
                "code-quality": 6,
                "efficiency": 9,
                "determinism": 8,
                "resilience": 7,
                "human-intervention": 10,
            }
        )
        scores = parse_evaluator_response(response)
        assert scores is not None
        assert scores["correctness"] == 8
        assert len(scores) == 7

    def test_wrapped_in_code_fence(self):
        response = '```json\n{"correctness": 8, "completeness": 7}\n```'
        scores = parse_evaluator_response(response)
        assert scores is not None
        assert scores["correctness"] == 8

    def test_with_surrounding_text(self):
        response = 'Here are the scores:\n{"correctness": 9, "completeness": 8}\nDone.'
        scores = parse_evaluator_response(response)
        assert scores is not None
        assert scores["correctness"] == 9

    def test_clamps_to_range(self):
        response = '{"correctness": 15, "completeness": -1}'
        scores = parse_evaluator_response(response)
        assert scores is not None
        assert scores["correctness"] == 10  # clamped
        assert scores["completeness"] == 1  # clamped

    def test_invalid_json_returns_none(self):
        assert parse_evaluator_response("not json at all") is None

    def test_empty_response_returns_none(self):
        assert parse_evaluator_response("") is None

    def test_partial_dimensions(self):
        response = '{"correctness": 8}'
        scores = parse_evaluator_response(response)
        assert scores is not None
        assert len(scores) == 1

    def test_non_numeric_values_skipped(self):
        response = '{"correctness": "high", "completeness": 7}'
        scores = parse_evaluator_response(response)
        assert scores is not None
        assert "correctness" not in scores
        assert scores["completeness"] == 7


# ---------------------------------------------------------------------------
# RatingEngine
# ---------------------------------------------------------------------------


class TestRatingEngine:
    def test_rate_completed_task_heuristic(self, completed_task):
        env = completed_task
        engine = RatingEngine(
            store=env["rs"],
            work_state=env["ws"],
            audit_log=env["al"],
            session_logger=env["sl"],
        )
        rating = engine.rate_task(env["task_id"], use_claude=False)
        assert rating.task_id == env["task_id"]
        assert rating.method == "heuristic"
        assert rating.overall > 0
        for dim in DIMENSION_NAMES:
            assert dim in rating.scores
            assert 1 <= rating.scores[dim] <= 10

    def test_rate_non_completed_task_raises(self, tmp_env):
        ml, ws = tmp_env["ml"], tmp_env["ws"]
        ml.append(
            "task_created",
            {"id": "t1", "name": "pending-task", "done_when": "x", "depends_on": []},
            reason="test",
        )
        ws.refresh()
        engine = RatingEngine(
            store=tmp_env["rs"],
            work_state=ws,
            audit_log=tmp_env["al"],
            session_logger=tmp_env["sl"],
        )
        with pytest.raises(ValueError, match="not 'completed'"):
            engine.rate_task("t1", use_claude=False)

    def test_rate_nonexistent_task_raises(self, tmp_env):
        engine = RatingEngine(
            store=tmp_env["rs"],
            work_state=tmp_env["ws"],
            audit_log=tmp_env["al"],
            session_logger=tmp_env["sl"],
        )
        with pytest.raises(ValueError, match="not found"):
            engine.rate_task("nonexistent", use_claude=False)

    def test_rate_stores_result(self, completed_task):
        env = completed_task
        engine = RatingEngine(
            store=env["rs"],
            work_state=env["ws"],
            audit_log=env["al"],
            session_logger=env["sl"],
        )
        engine.rate_task(env["task_id"], use_claude=False)
        assert env["rs"].is_rated(env["task_id"])

    def test_rate_auto_scores_unscored(self, completed_task):
        env = completed_task
        engine = RatingEngine(
            store=env["rs"],
            work_state=env["ws"],
            audit_log=env["al"],
            session_logger=env["sl"],
        )
        ratings = engine.rate_auto()
        assert len(ratings) == 1
        assert ratings[0].task_id == env["task_id"]

    def test_rate_auto_skips_already_rated(self, completed_task):
        env = completed_task
        engine = RatingEngine(
            store=env["rs"],
            work_state=env["ws"],
            audit_log=env["al"],
            session_logger=env["sl"],
        )
        # Rate first
        engine.rate_task(env["task_id"], use_claude=False)
        # Auto should find nothing new
        ratings = engine.rate_auto()
        assert len(ratings) == 0

    def test_rate_with_claude_fallback(self, completed_task):
        """When claude -p fails, engine falls back to heuristic scoring."""
        env = completed_task
        engine = RatingEngine(
            store=env["rs"],
            work_state=env["ws"],
            audit_log=env["al"],
            session_logger=env["sl"],
        )
        # Mock subprocess.run to simulate claude failure
        with patch("corc.rating.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("claude not found")
            rating = engine.rate_task(env["task_id"], use_claude=True)
            assert rating.method == "heuristic"  # Fell back

    def test_rate_with_claude_success(self, completed_task):
        """When claude -p succeeds, scores are parsed from response."""
        env = completed_task
        engine = RatingEngine(
            store=env["rs"],
            work_state=env["ws"],
            audit_log=env["al"],
            session_logger=env["sl"],
        )
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(
            {
                "correctness": 9,
                "completeness": 8,
                "code-quality": 7,
                "efficiency": 8,
                "determinism": 9,
                "resilience": 8,
                "human-intervention": 10,
            }
        )
        with patch("corc.rating.subprocess.run", return_value=mock_result):
            rating = engine.rate_task(env["task_id"], use_claude=True)
            assert rating.method == "evaluator"
            assert rating.scores["correctness"] == 9

    def test_rate_auto_multiple_tasks(self, tmp_env):
        """Auto scores all unscored completed tasks."""
        ml, ws = tmp_env["ml"], tmp_env["ws"]
        for i in range(3):
            tid = f"t{i}"
            ml.append(
                "task_created",
                {"id": tid, "name": f"task-{i}", "done_when": "x", "depends_on": []},
                reason="test",
            )
            ml.append(
                "task_completed",
                {"findings": []},
                reason="done",
                task_id=tid,
            )
        ws.refresh()

        engine = RatingEngine(
            store=tmp_env["rs"],
            work_state=ws,
            audit_log=tmp_env["al"],
            session_logger=tmp_env["sl"],
        )
        # Use heuristics (no claude) via fallback
        with patch("corc.rating.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("no claude")
            ratings = engine.rate_auto()
        assert len(ratings) == 3

    def test_spec_path_loading(self, tmp_env):
        """Engine loads spec excerpt for evaluator prompt."""
        spec_path = tmp_env["tmp_path"] / "SPEC.md"
        spec_path.write_text(
            "# Spec\n## 10. Rating & Continuous Improvement\nRating content here.\n## 11. Next"
        )

        engine = RatingEngine(
            store=tmp_env["rs"],
            work_state=tmp_env["ws"],
            audit_log=tmp_env["al"],
            session_logger=tmp_env["sl"],
            spec_path=spec_path,
        )
        excerpt = engine._load_spec_excerpt()
        assert "Rating content here" in excerpt

    def test_session_summary_built(self, completed_task):
        """Engine builds session summary from session logs."""
        env = completed_task
        sl = env["sl"]
        # Write some session data
        sl.log_dispatch(env["task_id"], 1, "prompt", "system", ["Read"], 3.0)
        sl.log_output(env["task_id"], 1, "done", 0, 10.5)

        engine = RatingEngine(
            store=env["rs"],
            work_state=env["ws"],
            audit_log=env["al"],
            session_logger=sl,
        )
        summary = engine._get_session_summary(env["task_id"])
        assert "attempt 1" in summary
        assert "Tool call events" in summary or "entries" in summary


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


class TestFormatting:
    def test_format_rating(self):
        rating = Rating(
            task_id="t1",
            task_name="test-task",
            scores={dim: 7 for dim in DIMENSION_NAMES},
            overall=7.0,
            flags=[],
            method="heuristic",
            timestamp="2026-03-22T10:00:00Z",
        )
        output = format_rating(rating)
        assert "test-task" in output
        assert "7.0/10.0" in output
        assert "heuristic" in output

    def test_format_rating_with_flags(self):
        rating = Rating(
            task_id="t1",
            task_name="test-task",
            scores={"correctness": 5, "completeness": 8},
            overall=6.0,
            flags=["correctness"],
            method="evaluator",
            timestamp="2026-03-22T10:00:00Z",
        )
        output = format_rating(rating)
        assert "⚠" in output
        assert "correctness" in output

    def test_format_trend_empty(self):
        output = format_trend([])
        assert "No ratings found" in output

    def test_format_trend(self):
        ratings = [
            Rating(
                task_id=f"t{i}",
                task_name=f"task-{i}",
                scores={"correctness": 5 + i},
                overall=float(5 + i),
                flags=[],
                method="heuristic",
                timestamp=f"2026-03-22T10:0{i}:00Z",
            )
            for i in range(3)
        ]
        output = format_trend(ratings)
        assert "Rating Trend" in output
        assert "Avg overall" in output
        assert "task-0" in output

    def test_format_dimension_drilldown(self):
        entries = [
            {
                "task_id": "t1",
                "task_name": "task-one",
                "score": 8,
                "overall": 7.5,
                "timestamp": "2026-03-22T10:00:00Z",
            },
            {
                "task_id": "t2",
                "task_name": "task-two",
                "score": 6,
                "overall": 6.0,
                "timestamp": "2026-03-22T11:00:00Z",
            },
        ]
        output = format_dimension_drilldown("correctness", entries)
        assert "correctness" in output
        assert "Avg: 7.0" in output
        assert "task-one" in output

    def test_format_dimension_drilldown_empty(self):
        output = format_dimension_drilldown("correctness", [])
        assert "No data" in output


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestCLI:
    def _setup_cli_env(self, tmp_path):
        """Create a minimal environment for CLI testing."""
        (tmp_path / "data").mkdir(exist_ok=True)
        (tmp_path / "data" / "events").mkdir(exist_ok=True)
        (tmp_path / "data" / "sessions").mkdir(exist_ok=True)
        (tmp_path / "data" / "ratings").mkdir(exist_ok=True)
        (tmp_path / "knowledge").mkdir(exist_ok=True)
        (tmp_path / ".corc").mkdir(exist_ok=True)
        (tmp_path / ".git").mkdir(exist_ok=True)

        # Create a completed task
        ml = MutationLog(tmp_path / "data" / "mutations.jsonl")
        ws = WorkState(tmp_path / "data" / "state.db", ml)
        ml.append(
            "task_created",
            {
                "id": "cli-01",
                "name": "cli-test-task",
                "done_when": "x",
                "depends_on": [],
            },
            reason="test",
        )
        ml.append(
            "task_completed",
            {"findings": []},
            reason="done",
            task_id="cli-01",
        )
        ws.refresh()
        return tmp_path

    def test_rate_command(self, tmp_path):
        root = self._setup_cli_env(tmp_path)
        runner = CliRunner()
        from corc.cli import cli

        with patch("corc.config.get_project_root", return_value=root):
            with patch("corc.rating.subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError("no claude")
                result = runner.invoke(cli, ["rate", "cli-01", "--no-claude"])
        assert result.exit_code == 0, result.output
        assert "cli-test-task" in result.output or "cli-01" in result.output

    def test_rate_auto_command(self, tmp_path):
        root = self._setup_cli_env(tmp_path)
        runner = CliRunner()
        from corc.cli import cli

        with patch("corc.config.get_project_root", return_value=root):
            with patch("corc.rating.subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError("no claude")
                result = runner.invoke(cli, ["rate", "--auto", "--no-claude"])
        assert result.exit_code == 0, result.output
        assert "Rated 1 task" in result.output or "cli-01" in result.output

    def test_rate_nonexistent_task(self, tmp_path):
        root = self._setup_cli_env(tmp_path)
        runner = CliRunner()
        from corc.cli import cli

        with patch("corc.config.get_project_root", return_value=root):
            result = runner.invoke(cli, ["rate", "nonexistent", "--no-claude"])
        assert result.exit_code != 0 or "not found" in result.output

    def test_ratings_trend_command(self, tmp_path):
        root = self._setup_cli_env(tmp_path)
        runner = CliRunner()
        from corc.cli import cli

        # First create a rating
        with patch("corc.config.get_project_root", return_value=root):
            with patch("corc.rating.subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError("no claude")
                runner.invoke(cli, ["rate", "cli-01", "--no-claude"])
                result = runner.invoke(cli, ["ratings", "trend"])
        assert result.exit_code == 0, result.output
        assert "Rating Trend" in result.output

    def test_ratings_dimension_command(self, tmp_path):
        root = self._setup_cli_env(tmp_path)
        runner = CliRunner()
        from corc.cli import cli

        with patch("corc.config.get_project_root", return_value=root):
            with patch("corc.rating.subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError("no claude")
                runner.invoke(cli, ["rate", "cli-01", "--no-claude"])
                result = runner.invoke(cli, ["ratings", "dimension", "correctness"])
        assert result.exit_code == 0, result.output
        assert "correctness" in result.output
