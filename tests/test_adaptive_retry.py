"""Tests for adaptive retry policy.

Tests the full lifecycle: recording outcomes, computing stats, adapting
retry counts, audit logging of adaptations, CLI output, and integration
with the processor. Includes simulated runs of 20+ and 25+ outcomes to
verify adaptive behavior emerges from accumulated data.
"""

import json
from pathlib import Path

import pytest

from corc.adaptive_retry import (
    DEFAULT_RETRIES,
    HIGH_SUCCESS_THRESHOLD,
    INCREASED_RETRIES,
    LOW_SUCCESS_THRESHOLD,
    MIN_SAMPLES,
    REDUCED_RETRIES,
    AdaptiveRetryConfig,
    AdaptiveRetryTracker,
    RetryStats,
    TaskOutcome,
    compute_retry_statistics,
    format_retry_statistics,
)
from corc.audit import AuditLog


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_tracker(tmp_path):
    """Tracker backed by a temp JSONL file."""
    return AdaptiveRetryTracker(tmp_path / "retry_outcomes.jsonl")


@pytest.fixture
def tmp_audit(tmp_path):
    """Audit log backed by a temp directory."""
    return AuditLog(tmp_path / "events")


# ---------------------------------------------------------------------------
# TaskOutcome
# ---------------------------------------------------------------------------


class TestTaskOutcome:
    def test_round_trip(self):
        o = TaskOutcome(
            task_type="implement",
            role="implementer",
            attempt=1,
            success=True,
            task_id="t-1",
            timestamp="2026-03-22T00:00:00.000Z",
        )
        d = o.to_dict()
        o2 = TaskOutcome.from_dict(d)
        assert o2.task_type == "implement"
        assert o2.role == "implementer"
        assert o2.attempt == 1
        assert o2.success is True
        assert o2.task_id == "t-1"

    def test_from_dict_defaults(self):
        o = TaskOutcome.from_dict({})
        assert o.task_type == "general"
        assert o.role == "unknown"
        assert o.attempt == 1
        assert o.success is False


# ---------------------------------------------------------------------------
# RetryStats
# ---------------------------------------------------------------------------


class TestRetryStats:
    def test_success_rate_zero_attempts(self):
        s = RetryStats(task_type="x", role="y")
        assert s.first_attempt_success_rate == 0.0

    def test_success_rate_computation(self):
        s = RetryStats(
            task_type="x",
            role="y",
            total_first_attempts=10,
            first_attempt_successes=9,
        )
        assert s.first_attempt_success_rate == pytest.approx(0.9)

    def test_adapted_retries_below_min_samples(self):
        s = RetryStats(
            task_type="x",
            role="y",
            total_first_attempts=MIN_SAMPLES - 1,
            first_attempt_successes=MIN_SAMPLES - 1,
        )
        assert s.adapted_retries == DEFAULT_RETRIES

    def test_adapted_retries_high_success(self):
        s = RetryStats(
            task_type="x",
            role="y",
            total_first_attempts=10,
            first_attempt_successes=10,  # 100%
        )
        assert s.adapted_retries == REDUCED_RETRIES

    def test_adapted_retries_low_success(self):
        s = RetryStats(
            task_type="x",
            role="y",
            total_first_attempts=10,
            first_attempt_successes=3,  # 30%
        )
        assert s.adapted_retries == INCREASED_RETRIES

    def test_adapted_retries_middle_range(self):
        s = RetryStats(
            task_type="x",
            role="y",
            total_first_attempts=10,
            first_attempt_successes=7,  # 70%
        )
        assert s.adapted_retries == DEFAULT_RETRIES

    def test_flagged_below_min_samples(self):
        s = RetryStats(
            task_type="x",
            role="y",
            total_first_attempts=3,
            first_attempt_successes=0,
        )
        assert s.flagged is False

    def test_flagged_low_success(self):
        s = RetryStats(
            task_type="x",
            role="y",
            total_first_attempts=10,
            first_attempt_successes=2,  # 20%
        )
        assert s.flagged is True

    def test_not_flagged_high_success(self):
        s = RetryStats(
            task_type="x",
            role="y",
            total_first_attempts=10,
            first_attempt_successes=10,
        )
        assert s.flagged is False


# ---------------------------------------------------------------------------
# AdaptiveRetryTracker - basic operations
# ---------------------------------------------------------------------------


class TestTrackerBasic:
    def test_record_and_read(self, tmp_tracker):
        outcome = TaskOutcome(
            task_type="implement",
            role="implementer",
            attempt=1,
            success=True,
            task_id="t-1",
        )
        tmp_tracker.record_outcome(outcome)
        outcomes = tmp_tracker.read_outcomes()
        assert len(outcomes) == 1
        assert outcomes[0].task_type == "implement"
        assert outcomes[0].success is True

    def test_auto_timestamp(self, tmp_tracker):
        outcome = TaskOutcome(task_type="test", role="tester", attempt=1, success=True)
        assert outcome.timestamp == ""
        tmp_tracker.record_outcome(outcome)
        stored = tmp_tracker.read_outcomes()
        assert stored[0].timestamp != ""

    def test_read_empty(self, tmp_tracker):
        assert tmp_tracker.read_outcomes() == []

    def test_multiple_outcomes(self, tmp_tracker):
        for i in range(5):
            tmp_tracker.record_outcome(
                TaskOutcome(
                    task_type="t",
                    role="r",
                    attempt=1,
                    success=(i % 2 == 0),
                    task_id=f"t-{i}",
                )
            )
        assert len(tmp_tracker.read_outcomes()) == 5

    def test_jsonl_format(self, tmp_tracker):
        tmp_tracker.record_outcome(
            TaskOutcome(task_type="a", role="b", attempt=1, success=True)
        )
        raw = tmp_tracker.data_path.read_text()
        lines = [l for l in raw.strip().split("\n") if l]
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["task_type"] == "a"
        assert parsed["success"] is True


# ---------------------------------------------------------------------------
# AdaptiveRetryTracker - compute_stats
# ---------------------------------------------------------------------------


class TestTrackerStats:
    def test_empty_stats(self, tmp_tracker):
        stats = tmp_tracker.compute_stats()
        assert stats == {}

    def test_single_group(self, tmp_tracker):
        for i in range(6):
            tmp_tracker.record_outcome(
                TaskOutcome(
                    task_type="implement",
                    role="implementer",
                    attempt=1,
                    success=True,
                    task_id=f"t-{i}",
                )
            )
        stats = tmp_tracker.compute_stats()
        key = ("implement", "implementer")
        assert key in stats
        s = stats[key]
        assert s.total_first_attempts == 6
        assert s.first_attempt_successes == 6
        assert s.first_attempt_success_rate == 1.0

    def test_multiple_groups(self, tmp_tracker):
        # Group A: high success
        for i in range(6):
            tmp_tracker.record_outcome(
                TaskOutcome(
                    task_type="review",
                    role="reviewer",
                    attempt=1,
                    success=True,
                )
            )
        # Group B: low success
        for i in range(6):
            tmp_tracker.record_outcome(
                TaskOutcome(
                    task_type="debug",
                    role="implementer",
                    attempt=1,
                    success=(i == 0),  # only 1/6 success
                )
            )
        stats = tmp_tracker.compute_stats()
        assert len(stats) == 2
        assert stats[("review", "reviewer")].first_attempt_success_rate == 1.0
        assert stats[
            ("debug", "implementer")
        ].first_attempt_success_rate == pytest.approx(1 / 6)

    def test_only_first_attempts_counted(self, tmp_tracker):
        """Only attempt=1 should count toward first-attempt success rate."""
        tmp_tracker.record_outcome(
            TaskOutcome(task_type="x", role="y", attempt=1, success=False)
        )
        tmp_tracker.record_outcome(
            TaskOutcome(task_type="x", role="y", attempt=2, success=True)
        )
        stats = tmp_tracker.compute_stats()
        s = stats[("x", "y")]
        assert s.total_first_attempts == 1
        assert s.first_attempt_successes == 0
        assert s.total_attempts == 2
        assert s.total_successes == 1


# ---------------------------------------------------------------------------
# AdaptiveRetryTracker - get_adaptive_max_retries
# ---------------------------------------------------------------------------


class TestAdaptiveRetries:
    def test_default_when_no_data(self, tmp_tracker):
        retries = tmp_tracker.get_adaptive_max_retries("foo", "bar")
        assert retries == DEFAULT_RETRIES

    def test_default_when_insufficient_samples(self, tmp_tracker):
        for i in range(MIN_SAMPLES - 1):
            tmp_tracker.record_outcome(
                TaskOutcome(task_type="x", role="y", attempt=1, success=True)
            )
        retries = tmp_tracker.get_adaptive_max_retries("x", "y")
        assert retries == DEFAULT_RETRIES

    def test_reduced_for_high_success(self, tmp_tracker):
        """>90% first-attempt success => retries reduced to 1."""
        # 10 successes, 0 failures = 100%
        for i in range(10):
            tmp_tracker.record_outcome(
                TaskOutcome(task_type="easy", role="expert", attempt=1, success=True)
            )
        retries = tmp_tracker.get_adaptive_max_retries("easy", "expert")
        assert retries == REDUCED_RETRIES

    def test_increased_for_low_success(self, tmp_tracker):
        """<50% first-attempt success => retries increased to 3."""
        # 2 successes, 8 failures = 20%
        for i in range(10):
            tmp_tracker.record_outcome(
                TaskOutcome(
                    task_type="hard",
                    role="novice",
                    attempt=1,
                    success=(i < 2),
                )
            )
        retries = tmp_tracker.get_adaptive_max_retries("hard", "novice")
        assert retries == INCREASED_RETRIES

    def test_default_for_middle_range(self, tmp_tracker):
        """50-90% first-attempt success => default retries."""
        # 7 successes, 3 failures = 70%
        for i in range(10):
            tmp_tracker.record_outcome(
                TaskOutcome(
                    task_type="medium",
                    role="dev",
                    attempt=1,
                    success=(i < 7),
                )
            )
        retries = tmp_tracker.get_adaptive_max_retries("medium", "dev")
        assert retries == DEFAULT_RETRIES

    def test_exactly_90_percent_is_default(self, tmp_tracker):
        """Exactly 90% is NOT >90%, so it should return default."""
        # 9 successes out of 10 = 90%
        for i in range(10):
            tmp_tracker.record_outcome(
                TaskOutcome(
                    task_type="edge",
                    role="r",
                    attempt=1,
                    success=(i < 9),
                )
            )
        retries = tmp_tracker.get_adaptive_max_retries("edge", "r")
        assert retries == DEFAULT_RETRIES

    def test_exactly_50_percent_is_default(self, tmp_tracker):
        """Exactly 50% is NOT <50%, so it should return default."""
        for i in range(10):
            tmp_tracker.record_outcome(
                TaskOutcome(
                    task_type="half",
                    role="r",
                    attempt=1,
                    success=(i < 5),
                )
            )
        retries = tmp_tracker.get_adaptive_max_retries("half", "r")
        assert retries == DEFAULT_RETRIES


# ---------------------------------------------------------------------------
# Audit log integration
# ---------------------------------------------------------------------------


class TestAuditLogging:
    def test_adaptation_logged_on_reduction(self, tmp_tracker, tmp_audit):
        for i in range(10):
            tmp_tracker.record_outcome(
                TaskOutcome(task_type="easy", role="pro", attempt=1, success=True)
            )
        retries = tmp_tracker.get_adaptive_max_retries(
            "easy", "pro", audit_log=tmp_audit
        )
        assert retries == REDUCED_RETRIES

        events = tmp_audit.read_all()
        adaptation_events = [e for e in events if e["event_type"] == "retry_adaptation"]
        assert len(adaptation_events) == 1
        ev = adaptation_events[0]
        assert ev["task_type"] == "easy"
        assert ev["role"] == "pro"
        assert ev["adapted_retries"] == REDUCED_RETRIES
        assert ev["sample_count"] == 10
        assert "High success rate" in ev["reason"]

    def test_adaptation_logged_on_increase(self, tmp_tracker, tmp_audit):
        for i in range(10):
            tmp_tracker.record_outcome(
                TaskOutcome(task_type="hard", role="jr", attempt=1, success=False)
            )
        retries = tmp_tracker.get_adaptive_max_retries(
            "hard", "jr", audit_log=tmp_audit
        )
        assert retries == INCREASED_RETRIES

        events = tmp_audit.read_all()
        adaptation_events = [e for e in events if e["event_type"] == "retry_adaptation"]
        assert len(adaptation_events) == 1
        ev = adaptation_events[0]
        assert ev["adapted_retries"] == INCREASED_RETRIES
        assert "Low success rate" in ev["reason"]

    def test_no_log_for_default(self, tmp_tracker, tmp_audit):
        for i in range(10):
            tmp_tracker.record_outcome(
                TaskOutcome(task_type="mid", role="dev", attempt=1, success=(i < 7))
            )
        retries = tmp_tracker.get_adaptive_max_retries(
            "mid", "dev", audit_log=tmp_audit
        )
        assert retries == DEFAULT_RETRIES
        events = tmp_audit.read_all()
        assert len(events) == 0

    def test_no_log_when_no_audit(self, tmp_tracker):
        """No crash when audit_log is None."""
        for i in range(10):
            tmp_tracker.record_outcome(
                TaskOutcome(task_type="x", role="y", attempt=1, success=True)
            )
        retries = tmp_tracker.get_adaptive_max_retries("x", "y", audit_log=None)
        assert retries == REDUCED_RETRIES


# ---------------------------------------------------------------------------
# Flagged task types
# ---------------------------------------------------------------------------


class TestFlaggedTaskTypes:
    def test_no_flagged_when_empty(self, tmp_tracker):
        assert tmp_tracker.get_flagged_task_types() == []

    def test_flagged_low_success(self, tmp_tracker):
        for i in range(10):
            tmp_tracker.record_outcome(
                TaskOutcome(
                    task_type="troublesome",
                    role="dev",
                    attempt=1,
                    success=False,
                )
            )
        flagged = tmp_tracker.get_flagged_task_types()
        assert len(flagged) == 1
        assert flagged[0].task_type == "troublesome"

    def test_not_flagged_insufficient_samples(self, tmp_tracker):
        for i in range(MIN_SAMPLES - 1):
            tmp_tracker.record_outcome(
                TaskOutcome(task_type="x", role="y", attempt=1, success=False)
            )
        assert tmp_tracker.get_flagged_task_types() == []


# ---------------------------------------------------------------------------
# Custom config
# ---------------------------------------------------------------------------


class TestCustomConfig:
    def test_custom_thresholds(self, tmp_path):
        config = AdaptiveRetryConfig(
            min_samples=3,
            high_success_threshold=0.80,
            low_success_threshold=0.30,
            reduced_retries=0,
            increased_retries=5,
        )
        tracker = AdaptiveRetryTracker(tmp_path / "outcomes.jsonl", config=config)
        # 3 out of 4 successes = 75%, between 30-80 => default
        for i in range(4):
            tracker.record_outcome(
                TaskOutcome(task_type="a", role="b", attempt=1, success=(i < 3))
            )
        assert tracker.get_adaptive_max_retries("a", "b") == config.default_retries

        # All success = 100% > 80% => reduced to 0
        for i in range(4):
            tracker.record_outcome(
                TaskOutcome(task_type="c", role="d", attempt=1, success=True)
            )
        assert tracker.get_adaptive_max_retries("c", "d") == 0

        # All fail = 0% < 30% => increased to 5
        for i in range(4):
            tracker.record_outcome(
                TaskOutcome(task_type="e", role="f", attempt=1, success=False)
            )
        assert tracker.get_adaptive_max_retries("e", "f") == 5


# ---------------------------------------------------------------------------
# Statistics and formatting
# ---------------------------------------------------------------------------


class TestStatisticsAndFormatting:
    def test_compute_retry_statistics_empty(self, tmp_tracker):
        report = compute_retry_statistics(tmp_tracker)
        assert report["stats"] == []
        assert report["flagged"] == []
        assert report["total_outcomes"] == 0
        assert report["total_first_attempts"] == 0

    def test_compute_retry_statistics_with_data(self, tmp_tracker):
        for i in range(8):
            tmp_tracker.record_outcome(
                TaskOutcome(task_type="impl", role="dev", attempt=1, success=True)
            )
        for i in range(2):
            tmp_tracker.record_outcome(
                TaskOutcome(task_type="impl", role="dev", attempt=1, success=False)
            )
        report = compute_retry_statistics(tmp_tracker)
        assert report["total_outcomes"] == 10
        assert report["total_first_attempts"] == 10
        assert len(report["stats"]) == 1

    def test_format_empty(self, tmp_tracker):
        report = compute_retry_statistics(tmp_tracker)
        text = format_retry_statistics(report)
        assert "No retry outcome data found" in text

    def test_format_with_data(self, tmp_tracker):
        for i in range(10):
            tmp_tracker.record_outcome(
                TaskOutcome(task_type="impl", role="dev", attempt=1, success=True)
            )
        report = compute_retry_statistics(tmp_tracker)
        text = format_retry_statistics(report)
        assert "impl" in text
        assert "dev" in text
        assert "100.0%" in text

    def test_format_shows_flagged(self, tmp_tracker):
        for i in range(10):
            tmp_tracker.record_outcome(
                TaskOutcome(task_type="bad", role="jr", attempt=1, success=False)
            )
        report = compute_retry_statistics(tmp_tracker)
        text = format_retry_statistics(report)
        assert "FLAGGED" in text
        assert "Flagged for investigation" in text
        assert "bad" in text


# ---------------------------------------------------------------------------
# Simulated runs: 20+ outcomes
# ---------------------------------------------------------------------------


class TestSimulated20PlusRuns:
    """Verify adaptive behavior emerges after accumulating 20+ outcomes."""

    def test_25_high_success_reduces_retries(self, tmp_tracker):
        """25 runs with 96% first-attempt success => retries reduced."""
        # 24 successes + 1 failure = 96%
        for i in range(25):
            tmp_tracker.record_outcome(
                TaskOutcome(
                    task_type="stable-task",
                    role="senior",
                    attempt=1,
                    success=(i != 12),  # one failure at i=12
                    task_id=f"run-{i}",
                )
            )
        retries = tmp_tracker.get_adaptive_max_retries("stable-task", "senior")
        assert retries == REDUCED_RETRIES

        stats = tmp_tracker.compute_stats()
        s = stats[("stable-task", "senior")]
        assert s.total_first_attempts == 25
        assert s.first_attempt_successes == 24
        assert s.first_attempt_success_rate == pytest.approx(0.96)
        assert not s.flagged

    def test_25_low_success_increases_retries(self, tmp_tracker):
        """25 runs with 20% first-attempt success => retries increased & flagged."""
        # 5 successes + 20 failures = 20%
        for i in range(25):
            tmp_tracker.record_outcome(
                TaskOutcome(
                    task_type="flaky-task",
                    role="junior",
                    attempt=1,
                    success=(i < 5),
                    task_id=f"run-{i}",
                )
            )
        retries = tmp_tracker.get_adaptive_max_retries("flaky-task", "junior")
        assert retries == INCREASED_RETRIES

        stats = tmp_tracker.compute_stats()
        s = stats[("flaky-task", "junior")]
        assert s.total_first_attempts == 25
        assert s.first_attempt_successes == 5
        assert s.first_attempt_success_rate == pytest.approx(0.2)
        assert s.flagged

    def test_20_mixed_groups_adaptive(self, tmp_tracker):
        """20+ outcomes across multiple groups show different adaptations."""
        # Group 1: implement/senior - 95% success (19/20) => REDUCED
        for i in range(20):
            tmp_tracker.record_outcome(
                TaskOutcome(
                    task_type="implement",
                    role="senior",
                    attempt=1,
                    success=(i != 5),
                )
            )

        # Group 2: debug/junior - 25% success (5/20) => INCREASED
        for i in range(20):
            tmp_tracker.record_outcome(
                TaskOutcome(
                    task_type="debug",
                    role="junior",
                    attempt=1,
                    success=(i < 5),
                )
            )

        # Group 3: review/senior - 70% success (14/20) => DEFAULT
        for i in range(20):
            tmp_tracker.record_outcome(
                TaskOutcome(
                    task_type="review",
                    role="senior",
                    attempt=1,
                    success=(i < 14),
                )
            )

        assert (
            tmp_tracker.get_adaptive_max_retries("implement", "senior")
            == REDUCED_RETRIES
        )
        assert (
            tmp_tracker.get_adaptive_max_retries("debug", "junior") == INCREASED_RETRIES
        )
        assert (
            tmp_tracker.get_adaptive_max_retries("review", "senior") == DEFAULT_RETRIES
        )

        flagged = tmp_tracker.get_flagged_task_types()
        assert len(flagged) == 1
        assert flagged[0].task_type == "debug"

    def test_30_runs_with_retry_attempts(self, tmp_tracker):
        """30 tasks, some needing retries. Only first attempts count for rate."""
        for i in range(30):
            # First attempt: 40% success rate
            first_success = i < 12
            tmp_tracker.record_outcome(
                TaskOutcome(
                    task_type="mixed",
                    role="dev",
                    attempt=1,
                    success=first_success,
                    task_id=f"t-{i}",
                )
            )
            # If first attempt failed, record retry (attempt=2)
            if not first_success:
                tmp_tracker.record_outcome(
                    TaskOutcome(
                        task_type="mixed",
                        role="dev",
                        attempt=2,
                        success=True,  # retry succeeds
                        task_id=f"t-{i}",
                    )
                )

        retries = tmp_tracker.get_adaptive_max_retries("mixed", "dev")
        assert retries == INCREASED_RETRIES  # 40% < 50%

        stats = tmp_tracker.compute_stats()
        s = stats[("mixed", "dev")]
        # Only 30 first attempts counted
        assert s.total_first_attempts == 30
        assert s.first_attempt_successes == 12
        # Total includes retries
        assert s.total_attempts == 30 + 18  # 18 retries
        assert s.flagged

    def test_gradual_improvement_tracked(self, tmp_tracker):
        """Track improvement: start bad, get better, verify adaptation changes."""
        # Phase 1: first 10 runs, 2/10 success = 20%
        for i in range(10):
            tmp_tracker.record_outcome(
                TaskOutcome(
                    task_type="evolving",
                    role="learner",
                    attempt=1,
                    success=(i < 2),
                )
            )

        stats = tmp_tracker.compute_stats()
        s = stats[("evolving", "learner")]
        assert s.first_attempt_success_rate == pytest.approx(0.2)
        assert (
            tmp_tracker.get_adaptive_max_retries("evolving", "learner")
            == INCREASED_RETRIES
        )

        # Phase 2: next 20 runs, 19/20 success
        for i in range(20):
            tmp_tracker.record_outcome(
                TaskOutcome(
                    task_type="evolving",
                    role="learner",
                    attempt=1,
                    success=(i != 10),
                )
            )

        # Overall: 2 + 19 = 21 successes / 30 total = 70%
        stats = tmp_tracker.compute_stats()
        s = stats[("evolving", "learner")]
        assert s.total_first_attempts == 30
        assert s.first_attempt_successes == 21
        # 70% is between 50-90 => default
        assert (
            tmp_tracker.get_adaptive_max_retries("evolving", "learner")
            == DEFAULT_RETRIES
        )


# ---------------------------------------------------------------------------
# Processor integration
# ---------------------------------------------------------------------------


class TestProcessorIntegration:
    """Test that the processor records outcomes to the adaptive tracker."""

    @pytest.fixture
    def processor_deps(self, tmp_path):
        """Create all dependencies needed by process_completed."""
        from corc.audit import AuditLog
        from corc.dispatch import AgentResult
        from corc.mutations import MutationLog
        from corc.sessions import SessionLogger
        from corc.state import WorkState

        (tmp_path / "data").mkdir()
        ml = MutationLog(tmp_path / "data" / "mutations.jsonl")
        ws = WorkState(tmp_path / "data" / "state.db", ml)
        al = AuditLog(tmp_path / "data" / "events")
        sl = SessionLogger(tmp_path / "data" / "sessions")
        tracker = AdaptiveRetryTracker(tmp_path / "data" / "retry_outcomes.jsonl")

        return {
            "tmp_path": tmp_path,
            "mutation_log": ml,
            "state": ws,
            "audit_log": al,
            "session_logger": sl,
            "tracker": tracker,
        }

    def _create_task(
        self, state, task_id="t-1", task_type="implement", role="implementer"
    ):
        """Create a pending task in the work state."""
        state.mutation_log.append(
            "task_created",
            {
                "id": task_id,
                "name": f"Test task {task_id}",
                "task_type": task_type,
                "role": role,
                "status": "running",
                "deps": [],
                "done_when": "",
                "max_retries": 3,
            },
            reason="test",
            task_id=task_id,
        )
        state.refresh()

    def test_successful_task_records_outcome(self, processor_deps):
        from corc.dispatch import AgentResult
        from corc.processor import process_completed

        deps = processor_deps
        task = {
            "id": "t-1",
            "name": "test",
            "task_type": "implement",
            "role": "implementer",
            "done_when": "",
            "max_retries": 3,
        }

        self._create_task(deps["state"], "t-1")

        result = AgentResult(output="Success", exit_code=0, duration_s=1.0)
        process_completed(
            task=task,
            result=result,
            attempt=1,
            mutation_log=deps["mutation_log"],
            state=deps["state"],
            audit_log=deps["audit_log"],
            session_logger=deps["session_logger"],
            project_root=deps["tmp_path"],
            adaptive_tracker=deps["tracker"],
        )

        outcomes = deps["tracker"].read_outcomes()
        assert len(outcomes) == 1
        assert outcomes[0].task_type == "implement"
        assert outcomes[0].role == "implementer"
        assert outcomes[0].attempt == 1
        assert outcomes[0].success is True

    def test_failed_task_records_outcome(self, processor_deps):
        from corc.dispatch import AgentResult
        from corc.processor import process_completed

        deps = processor_deps
        task = {
            "id": "t-2",
            "name": "test",
            "task_type": "debug",
            "role": "debugger",
            "done_when": "",
            "max_retries": 3,
        }

        self._create_task(deps["state"], "t-2", task_type="debug", role="debugger")

        result = AgentResult(output="Error", exit_code=1, duration_s=0.5)
        process_completed(
            task=task,
            result=result,
            attempt=1,
            mutation_log=deps["mutation_log"],
            state=deps["state"],
            audit_log=deps["audit_log"],
            session_logger=deps["session_logger"],
            project_root=deps["tmp_path"],
            adaptive_tracker=deps["tracker"],
        )

        outcomes = deps["tracker"].read_outcomes()
        assert len(outcomes) == 1
        assert outcomes[0].success is False
        assert outcomes[0].task_type == "debug"

    def test_no_tracker_no_crash(self, processor_deps):
        """Processor works fine without adaptive_tracker (backward compat)."""
        from corc.dispatch import AgentResult
        from corc.processor import process_completed

        deps = processor_deps
        task = {
            "id": "t-3",
            "name": "test",
            "done_when": "",
            "max_retries": 3,
        }

        self._create_task(deps["state"], "t-3")

        result = AgentResult(output="OK", exit_code=0, duration_s=1.0)
        pr = process_completed(
            task=task,
            result=result,
            attempt=1,
            mutation_log=deps["mutation_log"],
            state=deps["state"],
            audit_log=deps["audit_log"],
            session_logger=deps["session_logger"],
            project_root=deps["tmp_path"],
            # adaptive_tracker not passed => None default
        )
        assert pr.passed is True

    def test_task_type_defaults_to_general(self, processor_deps):
        """Tasks without task_type get 'general'."""
        from corc.dispatch import AgentResult
        from corc.processor import process_completed

        deps = processor_deps
        task = {
            "id": "t-4",
            "name": "test",
            "done_when": "",
            "max_retries": 3,
            # no task_type or type
        }

        self._create_task(deps["state"], "t-4")

        result = AgentResult(output="OK", exit_code=0, duration_s=1.0)
        process_completed(
            task=task,
            result=result,
            attempt=1,
            mutation_log=deps["mutation_log"],
            state=deps["state"],
            audit_log=deps["audit_log"],
            session_logger=deps["session_logger"],
            project_root=deps["tmp_path"],
            adaptive_tracker=deps["tracker"],
        )

        outcomes = deps["tracker"].read_outcomes()
        assert outcomes[0].task_type == "general"
        assert outcomes[0].role == "unknown"

    def test_simulated_20_runs_through_processor(self, processor_deps):
        """Run 20+ tasks through processor and verify adaptive behavior."""
        from corc.dispatch import AgentResult
        from corc.processor import process_completed

        deps = processor_deps
        tracker = deps["tracker"]

        # Simulate 22 tasks: 21 successes, 1 failure = ~95.5%
        for i in range(22):
            task_id = f"sim-{i}"
            task = {
                "id": task_id,
                "name": f"sim task {i}",
                "task_type": "implement",
                "role": "senior",
                "done_when": "",
                "max_retries": 3,
            }

            self._create_task(deps["state"], task_id, "implement", "senior")

            exit_code = 1 if i == 10 else 0
            result = AgentResult(output="output", exit_code=exit_code, duration_s=1.0)
            process_completed(
                task=task,
                result=result,
                attempt=1,
                mutation_log=deps["mutation_log"],
                state=deps["state"],
                audit_log=deps["audit_log"],
                session_logger=deps["session_logger"],
                project_root=deps["tmp_path"],
                adaptive_tracker=tracker,
            )

        outcomes = tracker.read_outcomes()
        assert len(outcomes) == 22

        stats = tracker.compute_stats()
        s = stats[("implement", "senior")]
        assert s.total_first_attempts == 22
        assert s.first_attempt_successes == 21

        # 95.5% > 90% => retries should be reduced
        retries = tracker.get_adaptive_max_retries("implement", "senior")
        assert retries == REDUCED_RETRIES


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLI:
    """Test the CLI command invocation."""

    def test_analyze_retries_no_data(self, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.setattr("corc.config.get_project_root", lambda: tmp_path)
        (tmp_path / "data").mkdir(exist_ok=True)
        (tmp_path / "data" / "events").mkdir(exist_ok=True)
        (tmp_path / "data" / "sessions").mkdir(exist_ok=True)
        (tmp_path / "data" / "ratings").mkdir(exist_ok=True)

        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", "retries"])
        assert result.exit_code == 0
        assert "No retry outcome data found" in result.output

    def test_analyze_retries_with_data(self, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.setattr("corc.config.get_project_root", lambda: tmp_path)
        (tmp_path / "data").mkdir(exist_ok=True)
        (tmp_path / "data" / "events").mkdir(exist_ok=True)
        (tmp_path / "data" / "sessions").mkdir(exist_ok=True)
        (tmp_path / "data" / "ratings").mkdir(exist_ok=True)

        # Write some outcomes
        tracker = AdaptiveRetryTracker(tmp_path / "data" / "retry_outcomes.jsonl")
        for i in range(10):
            tracker.record_outcome(
                TaskOutcome(
                    task_type="implement",
                    role="dev",
                    attempt=1,
                    success=True,
                )
            )

        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", "retries"])
        assert result.exit_code == 0
        assert "implement" in result.output
        assert "dev" in result.output

    def test_analyze_retries_flagged_only(self, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.setattr("corc.config.get_project_root", lambda: tmp_path)
        (tmp_path / "data").mkdir(exist_ok=True)
        (tmp_path / "data" / "events").mkdir(exist_ok=True)
        (tmp_path / "data" / "sessions").mkdir(exist_ok=True)
        (tmp_path / "data" / "ratings").mkdir(exist_ok=True)

        tracker = AdaptiveRetryTracker(tmp_path / "data" / "retry_outcomes.jsonl")

        # High success group (not flagged)
        for i in range(10):
            tracker.record_outcome(
                TaskOutcome(task_type="good", role="dev", attempt=1, success=True)
            )
        # Low success group (flagged)
        for i in range(10):
            tracker.record_outcome(
                TaskOutcome(task_type="bad", role="dev", attempt=1, success=False)
            )

        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", "retries", "--flagged-only"])
        assert result.exit_code == 0
        assert "bad" in result.output
        # "good" should not appear in flagged-only output stats section
        # (it might appear in header stats)

    def test_analyze_retries_flagged_only_none(self, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.setattr("corc.config.get_project_root", lambda: tmp_path)
        (tmp_path / "data").mkdir(exist_ok=True)
        (tmp_path / "data" / "events").mkdir(exist_ok=True)
        (tmp_path / "data" / "sessions").mkdir(exist_ok=True)
        (tmp_path / "data" / "ratings").mkdir(exist_ok=True)

        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", "retries", "--flagged-only"])
        assert result.exit_code == 0
        assert "No task types flagged" in result.output
