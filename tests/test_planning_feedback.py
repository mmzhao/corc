"""Tests for planning feedback — self-improvement loop.

Tests cover:
- PlanningOutcome dataclass serialization
- PlanningFeedbackStore JSONL persistence
- record_planning_outcome from task + rating
- build_planning_feedback_section digest builder
- Truncation at ~3000 chars
- Planning lessons loading
- Integration with build_system_prompt
"""

import json
import time
from pathlib import Path

import pytest

from corc.planning_feedback import (
    MAX_FEEDBACK_CHARS,
    PlanningFeedbackStore,
    PlanningOutcome,
    build_planning_feedback_section,
    load_planning_lessons,
    record_planning_outcome,
    _context_bundle_effectiveness,
    _done_when_calibration,
    _pattern_highlights,
    _recent_failures,
    _retro_highlights,
)
from corc.rating import Rating


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_env(tmp_path):
    """Temporary environment with feedback store and corc dir."""
    feedback_path = tmp_path / "data" / "planning_feedback.jsonl"
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    corc_dir = tmp_path / ".corc"
    corc_dir.mkdir(parents=True, exist_ok=True)

    store = PlanningFeedbackStore(feedback_path)
    return {
        "store": store,
        "feedback_path": feedback_path,
        "corc_dir": corc_dir,
        "tmp_path": tmp_path,
    }


def _make_outcome(
    task_id: str = "t1",
    task_name: str = "test-task",
    role: str = "implementer",
    task_type: str = "implementation",
    done_when: str = "Tests pass and code is clean",
    checklist_size: int = 3,
    dependency_count: int = 1,
    context_bundle_size: int = 2,
    overall_score: float = 7.5,
    flags: list | None = None,
    attempt_count: int = 1,
    timestamp: str | None = None,
) -> PlanningOutcome:
    """Helper to create a PlanningOutcome for testing."""
    return PlanningOutcome(
        task_id=task_id,
        task_name=task_name,
        timestamp=timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        role=role,
        task_type=task_type,
        done_when=done_when,
        done_when_length=len(done_when),
        checklist_size=checklist_size,
        dependency_count=dependency_count,
        context_bundle_size=context_bundle_size,
        context_bundle_files=[f"file{i}.py" for i in range(context_bundle_size)],
        overall_score=overall_score,
        dimension_scores={"correctness": 8, "completeness": 7},
        flags=flags or [],
        attempt_count=attempt_count,
        status="completed",
    )


def _make_rating(
    task_id: str = "t1",
    task_name: str = "test-task",
    overall: float = 7.5,
    scores: dict | None = None,
    flags: list | None = None,
) -> Rating:
    """Helper to create a Rating for testing."""
    return Rating(
        task_id=task_id,
        task_name=task_name,
        scores=scores or {"correctness": 8, "completeness": 7},
        overall=overall,
        flags=flags or [],
        method="heuristic",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        metadata={"role": "implementer", "attempt_count": 1},
    )


# ---------------------------------------------------------------------------
# PlanningOutcome serialization
# ---------------------------------------------------------------------------


class TestPlanningOutcome:
    def test_to_dict_roundtrip(self):
        outcome = _make_outcome()
        d = outcome.to_dict()
        restored = PlanningOutcome.from_dict(d)
        assert restored.task_id == outcome.task_id
        assert restored.task_name == outcome.task_name
        assert restored.role == outcome.role
        assert restored.overall_score == outcome.overall_score
        assert restored.done_when_length == outcome.done_when_length

    def test_to_dict_json_roundtrip(self):
        outcome = _make_outcome()
        json_str = json.dumps(outcome.to_dict())
        d = json.loads(json_str)
        restored = PlanningOutcome.from_dict(d)
        assert restored.task_id == outcome.task_id
        assert restored.context_bundle_files == outcome.context_bundle_files

    def test_from_dict_with_missing_fields(self):
        """from_dict should handle missing optional fields gracefully."""
        d = {
            "task_id": "t1",
            "task_name": "test",
            "timestamp": "2026-01-01T00:00:00Z",
        }
        outcome = PlanningOutcome.from_dict(d)
        assert outcome.task_id == "t1"
        assert outcome.role == ""
        assert outcome.checklist_size == 0
        assert outcome.flags == []


# ---------------------------------------------------------------------------
# PlanningFeedbackStore
# ---------------------------------------------------------------------------


class TestPlanningFeedbackStore:
    def test_save_and_read(self, tmp_env):
        store = tmp_env["store"]
        outcome = _make_outcome()
        store.save(outcome)

        outcomes = store.read_all()
        assert len(outcomes) == 1
        assert outcomes[0].task_id == "t1"
        assert outcomes[0].overall_score == 7.5

    def test_multiple_saves(self, tmp_env):
        store = tmp_env["store"]
        store.save(_make_outcome(task_id="t1"))
        store.save(_make_outcome(task_id="t2", task_name="task-two"))
        store.save(_make_outcome(task_id="t3", task_name="task-three"))

        outcomes = store.read_all()
        assert len(outcomes) == 3
        assert {o.task_id for o in outcomes} == {"t1", "t2", "t3"}

    def test_read_empty(self, tmp_env):
        store = tmp_env["store"]
        assert store.read_all() == []

    def test_read_recent(self, tmp_env):
        store = tmp_env["store"]
        for i in range(10):
            store.save(
                _make_outcome(
                    task_id=f"t{i}",
                    timestamp=f"2026-01-{i + 1:02d}T00:00:00Z",
                )
            )

        recent = store.read_recent(3)
        assert len(recent) == 3
        # Should be the last 3 by timestamp
        assert recent[0].task_id == "t7"
        assert recent[2].task_id == "t9"

    def test_jsonl_format(self, tmp_env):
        store = tmp_env["store"]
        store.save(_make_outcome())

        content = tmp_env["feedback_path"].read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["task_id"] == "t1"

    def test_corrupted_lines_skipped(self, tmp_env):
        """Corrupted JSONL lines should be skipped without error."""
        store = tmp_env["store"]
        store.save(_make_outcome(task_id="t1"))
        # Write a corrupted line
        with open(tmp_env["feedback_path"], "a") as f:
            f.write("not valid json\n")
        store.save(_make_outcome(task_id="t2"))

        outcomes = store.read_all()
        assert len(outcomes) == 2
        assert outcomes[0].task_id == "t1"
        assert outcomes[1].task_id == "t2"


# ---------------------------------------------------------------------------
# record_planning_outcome
# ---------------------------------------------------------------------------


class TestRecordPlanningOutcome:
    def test_records_outcome_from_task_and_rating(self, tmp_env):
        store = tmp_env["store"]
        task = {
            "id": "abc123",
            "name": "implement-feature",
            "role": "implementer",
            "task_type": "implementation",
            "done_when": "Tests pass and feature is complete with all edge cases handled",
            "checklist": ["write code", "write tests", "run lint"],
            "depends_on": ["dep1"],
            "context_bundle": ["src/main.py", "tests/test_main.py"],
            "attempt_count": 1,
            "status": "completed",
        }
        rating = _make_rating(
            task_id="abc123",
            task_name="implement-feature",
            overall=8.0,
            scores={"correctness": 9, "completeness": 8},
            flags=[],
        )

        outcome = record_planning_outcome(task, rating, store)

        assert outcome.task_id == "abc123"
        assert outcome.task_name == "implement-feature"
        assert outcome.role == "implementer"
        assert outcome.done_when_length == len(task["done_when"])
        assert outcome.checklist_size == 3
        assert outcome.dependency_count == 1
        assert outcome.context_bundle_size == 2
        assert outcome.overall_score == 8.0
        assert outcome.dimension_scores == {"correctness": 9, "completeness": 8}

        # Verify it was persisted
        outcomes = store.read_all()
        assert len(outcomes) == 1
        assert outcomes[0].task_id == "abc123"

    def test_handles_string_fields(self, tmp_env):
        """Should handle checklist/deps/bundle as JSON strings."""
        store = tmp_env["store"]
        task = {
            "id": "t2",
            "name": "task-two",
            "role": "scout",
            "task_type": "investigation",
            "done_when": "Found the answer",
            "checklist": '["a", "b"]',
            "depends_on": '["dep1"]',
            "context_bundle": '["f1.py"]',
            "status": "completed",
        }
        rating = _make_rating(task_id="t2", task_name="task-two")
        outcome = record_planning_outcome(task, rating, store)
        assert outcome.checklist_size == 2
        assert outcome.dependency_count == 1
        assert outcome.context_bundle_size == 1


# ---------------------------------------------------------------------------
# Digest builder sub-functions
# ---------------------------------------------------------------------------


class TestDoneWhenCalibration:
    def test_buckets_by_length(self):
        outcomes = [
            _make_outcome(done_when="short", overall_score=5.0),  # vague
            _make_outcome(
                done_when="medium length done when criteria here exactly",
                overall_score=7.0,
            ),  # moderate
            _make_outcome(done_when="a" * 120, overall_score=9.0),  # specific
        ]
        lines = _done_when_calibration(outcomes)
        assert len(lines) >= 1
        # Should have entries for the populated buckets
        text = "\n".join(lines)
        assert "vague" in text or "moderate" in text or "specific" in text

    def test_empty_outcomes(self):
        assert _done_when_calibration([]) == []


class TestContextBundleEffectiveness:
    def test_buckets(self):
        outcomes = [
            _make_outcome(context_bundle_size=0, overall_score=6.0),
            _make_outcome(context_bundle_size=2, overall_score=7.5),
            _make_outcome(context_bundle_size=5, overall_score=8.0),
            _make_outcome(context_bundle_size=10, overall_score=7.0),
        ]
        lines = _context_bundle_effectiveness(outcomes)
        assert len(lines) >= 1
        text = "\n".join(lines)
        assert "none" in text or "small" in text or "medium" in text or "large" in text

    def test_empty_outcomes(self):
        assert _context_bundle_effectiveness([]) == []


class TestRecentFailures:
    def test_extracts_low_scores(self):
        outcomes = [
            _make_outcome(task_name="good-task", overall_score=8.0),
            _make_outcome(
                task_name="bad-task",
                overall_score=4.0,
                flags=["correctness", "completeness"],
            ),
            _make_outcome(task_name="ok-task", overall_score=6.5),
        ]
        lines = _recent_failures(outcomes)
        assert len(lines) == 1
        assert "bad-task" in lines[0]

    def test_max_items(self):
        outcomes = [
            _make_outcome(task_id=f"t{i}", task_name=f"fail-{i}", overall_score=3.0)
            for i in range(10)
        ]
        lines = _recent_failures(outcomes, max_items=3)
        assert len(lines) == 3


class TestPatternHighlights:
    def test_detects_strong_role(self):
        outcomes = [
            _make_outcome(role="implementer", overall_score=9.0),
            _make_outcome(role="implementer", overall_score=8.5),
            _make_outcome(role="implementer", overall_score=8.0),
        ]
        lines = _pattern_highlights(outcomes)
        text = "\n".join(lines)
        assert "strong performer" in text

    def test_detects_weak_role(self):
        outcomes = [
            _make_outcome(role="reviewer", overall_score=4.0),
            _make_outcome(role="reviewer", overall_score=5.0),
            _make_outcome(role="reviewer", overall_score=4.5),
        ]
        lines = _pattern_highlights(outcomes)
        text = "\n".join(lines)
        assert "underperforming" in text

    def test_skips_small_samples(self):
        outcomes = [
            _make_outcome(role="scout", overall_score=9.0),
            _make_outcome(role="scout", overall_score=9.5),
        ]
        lines = _pattern_highlights(outcomes)
        assert len(lines) == 0  # Not enough samples


class TestRetroHighlights:
    def test_computes_trend(self):
        outcomes = [
            _make_outcome(
                task_id=f"t{i}",
                timestamp=f"2026-01-{i + 1:02d}T00:00:00Z",
                overall_score=5.0 + i * 0.5,
            )
            for i in range(6)
        ]
        lines = _retro_highlights(outcomes)
        text = "\n".join(lines)
        assert "IMPROVING" in text or "stable" in text or "DECLINING" in text

    def test_includes_stats(self):
        outcomes = [_make_outcome(overall_score=7.5)]
        lines = _retro_highlights(outcomes)
        text = "\n".join(lines)
        assert "Recent avg" in text

    def test_empty_outcomes(self):
        assert _retro_highlights([]) == []


# ---------------------------------------------------------------------------
# build_planning_feedback_section
# ---------------------------------------------------------------------------


class TestBuildPlanningFeedbackSection:
    def test_empty_store_no_lessons(self, tmp_env):
        section = build_planning_feedback_section(tmp_env["store"], tmp_env["corc_dir"])
        assert "Planning Feedback" in section
        assert "No planning outcomes recorded yet" in section

    def test_with_outcomes(self, tmp_env):
        store = tmp_env["store"]
        for i in range(5):
            store.save(
                _make_outcome(
                    task_id=f"t{i}",
                    timestamp=f"2026-01-{i + 1:02d}T00:00:00Z",
                    overall_score=6.0 + i,
                )
            )

        section = build_planning_feedback_section(store, tmp_env["corc_dir"])
        assert "Planning Feedback" in section
        assert "Retrospective Highlights" in section
        assert "Done-When Calibration" in section

    def test_with_planning_lessons(self, tmp_env):
        lessons_path = tmp_env["corc_dir"] / "planning_lessons.md"
        lessons_path.write_text("# Lessons\n- Always write tests first")

        section = build_planning_feedback_section(tmp_env["store"], tmp_env["corc_dir"])
        assert "Operator Planning Lessons" in section
        assert "Always write tests first" in section

    def test_truncation(self, tmp_env):
        store = tmp_env["store"]
        # Create many outcomes to generate a large section
        for i in range(100):
            store.save(
                _make_outcome(
                    task_id=f"t{i}",
                    task_name=f"a-very-long-task-name-{i}-padding" * 3,
                    timestamp=f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}T00:00:00Z",
                    overall_score=3.0 + (i % 8),
                    role="implementer" if i % 2 == 0 else "reviewer",
                    flags=["correctness", "completeness", "efficiency"]
                    if i % 3 == 0
                    else [],
                )
            )

        # Also add a very long planning lessons file
        lessons_path = tmp_env["corc_dir"] / "planning_lessons.md"
        lessons_path.write_text("Long lesson content. " * 200)

        section = build_planning_feedback_section(store, tmp_env["corc_dir"])
        assert len(section) <= MAX_FEEDBACK_CHARS + 100  # Allow small margin
        assert "Planning Feedback" in section

    def test_cap_parameter(self, tmp_env):
        store = tmp_env["store"]
        for i in range(20):
            store.save(
                _make_outcome(
                    task_id=f"t{i}",
                    timestamp=f"2026-01-{i + 1:02d}T00:00:00Z",
                    overall_score=6.0,
                )
            )

        section = build_planning_feedback_section(
            store, tmp_env["corc_dir"], max_chars=500
        )
        assert len(section) <= 550  # Allow small margin for truncation text


# ---------------------------------------------------------------------------
# load_planning_lessons
# ---------------------------------------------------------------------------


class TestLoadPlanningLessons:
    def test_file_exists(self, tmp_env):
        lessons_path = tmp_env["corc_dir"] / "planning_lessons.md"
        lessons_path.write_text("# My Lessons\n- Lesson 1\n- Lesson 2")

        content = load_planning_lessons(tmp_env["corc_dir"])
        assert "Lesson 1" in content
        assert "Lesson 2" in content

    def test_file_missing(self, tmp_env):
        content = load_planning_lessons(tmp_env["corc_dir"])
        assert content == ""

    def test_empty_file(self, tmp_env):
        lessons_path = tmp_env["corc_dir"] / "planning_lessons.md"
        lessons_path.write_text("")

        content = load_planning_lessons(tmp_env["corc_dir"])
        assert content == ""


# ---------------------------------------------------------------------------
# Integration: build_system_prompt includes feedback
# ---------------------------------------------------------------------------


class TestBuildSystemPromptIntegration:
    def test_feedback_section_present_in_system_prompt(self, tmp_env):
        """build_system_prompt should include the Planning Feedback section."""
        from unittest.mock import MagicMock

        from corc.plan import build_system_prompt

        # Set up paths dict
        paths = {
            "root": tmp_env["tmp_path"],
            "corc_dir": tmp_env["corc_dir"],
            "planning_feedback": tmp_env["feedback_path"],
        }

        # Create some outcomes so feedback section has content
        store = tmp_env["store"]
        store.save(_make_outcome(timestamp="2026-01-01T00:00:00Z"))
        store.save(_make_outcome(task_id="t2", timestamp="2026-01-02T00:00:00Z"))

        # Mock ws and ks
        ws = MagicMock()
        ws.list_tasks.return_value = []
        ws.get_ready_tasks.return_value = []

        ks = MagicMock()
        ks.list_docs.return_value = []

        prompt = build_system_prompt(paths, ws, ks)

        assert "Planning Feedback" in prompt
        assert "Self-Improvement" in prompt

    def test_no_feedback_when_no_data(self, tmp_env):
        """With no outcomes or lessons, the feedback section should be minimal."""
        from unittest.mock import MagicMock

        from corc.plan import build_system_prompt

        paths = {
            "root": tmp_env["tmp_path"],
            "corc_dir": tmp_env["corc_dir"],
            "planning_feedback": tmp_env["feedback_path"],
        }

        ws = MagicMock()
        ws.list_tasks.return_value = []
        ws.get_ready_tasks.return_value = []

        ks = MagicMock()
        ks.list_docs.return_value = []

        prompt = build_system_prompt(paths, ws, ks)

        # Should still have the stub
        assert "Planning Feedback" in prompt
        assert "No planning outcomes recorded yet" in prompt

    def test_planner_role_has_feedback_instructions(self):
        """PLANNER_ROLE should include feedback-awareness instructions."""
        from corc.plan import PLANNER_ROLE

        assert "Feedback-Aware Planning" in PLANNER_ROLE
        assert "done-when specificity" in PLANNER_ROLE
