"""Tests for retry policies and structured escalation.

Tests the full retry chain: dispatch → fail → retry with enriched context → escalate.
Uses mock dispatchers that fail N times then succeed.
"""

import json
import threading
import time
from pathlib import Path

import pytest

from corc.audit import AuditLog
from corc.daemon import Daemon
from corc.dispatch import AgentDispatcher, AgentResult, Constraints
from corc.executor import Executor
from corc.mutations import MutationLog
from corc.processor import process_completed
from corc.retry import (
    RetryPolicy,
    create_escalation,
    get_retry_context,
    resolve_escalation,
)
from corc.sessions import SessionLogger
from corc.state import WorkState


# ---------------------------------------------------------------------------
# Mock dispatchers
# ---------------------------------------------------------------------------


class FailNTimesDispatcher(AgentDispatcher):
    """Dispatcher that fails N times, then succeeds."""

    def __init__(
        self, fail_count: int = 2, fail_output: str = "Error: something went wrong"
    ):
        self.fail_count = fail_count
        self.fail_output = fail_output
        self.call_count = 0
        self.dispatched: list[tuple[str, str]] = []

    def dispatch(
        self,
        prompt: str,
        system_prompt: str,
        constraints: Constraints,
        pid_callback=None,
        event_callback=None,
        cwd=None,
    ) -> AgentResult:
        self.call_count += 1
        self.dispatched.append((prompt, system_prompt))
        if self.call_count <= self.fail_count:
            return AgentResult(
                output=self.fail_output,
                exit_code=1,
                duration_s=0.1,
            )
        return AgentResult(
            output="Success: task completed.",
            exit_code=0,
            duration_s=0.1,
        )


class AlwaysFailDispatcher(AgentDispatcher):
    """Dispatcher that always fails."""

    def __init__(self, fail_output: str = "Error: persistent failure"):
        self.fail_output = fail_output
        self.call_count = 0
        self.dispatched: list[tuple[str, str]] = []

    def dispatch(
        self,
        prompt: str,
        system_prompt: str,
        constraints: Constraints,
        pid_callback=None,
        event_callback=None,
        cwd=None,
    ) -> AgentResult:
        self.call_count += 1
        self.dispatched.append((prompt, system_prompt))
        return AgentResult(
            output=self.fail_output,
            exit_code=1,
            duration_s=0.1,
        )


class TimeoutDispatcher(AgentDispatcher):
    """Dispatcher that simulates timeout failures."""

    def __init__(self):
        self.call_count = 0
        self.dispatched: list[tuple[str, str]] = []

    def dispatch(
        self,
        prompt: str,
        system_prompt: str,
        constraints: Constraints,
        pid_callback=None,
        event_callback=None,
        cwd=None,
    ) -> AgentResult:
        self.call_count += 1
        self.dispatched.append((prompt, system_prompt))
        return AgentResult(
            output="[TIMEOUT: agent exceeded 1800s limit]",
            exit_code=-1,
            duration_s=1800.0,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project structure for testing."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "events").mkdir()
    (tmp_path / "data" / "sessions").mkdir()
    (tmp_path / ".corc").mkdir()
    return tmp_path


@pytest.fixture
def mutation_log(tmp_project):
    return MutationLog(tmp_project / "data" / "mutations.jsonl")


@pytest.fixture
def work_state(tmp_project, mutation_log):
    return WorkState(tmp_project / "data" / "state.db", mutation_log)


@pytest.fixture
def audit_log(tmp_project):
    return AuditLog(tmp_project / "data" / "events")


@pytest.fixture
def session_logger(tmp_project):
    return SessionLogger(tmp_project / "data" / "sessions")


def _create_task(
    mutation_log,
    task_id,
    name,
    done_when="do the thing",
    depends_on=None,
    role="implementer",
    max_retries=3,
):
    """Helper to create a task via mutation log."""
    mutation_log.append(
        "task_created",
        {
            "id": task_id,
            "name": name,
            "description": f"Test task: {name}",
            "role": role,
            "depends_on": depends_on or [],
            "done_when": done_when,
            "checklist": [],
            "context_bundle": [],
            "max_retries": max_retries,
        },
        reason="Test setup",
    )


# ===========================================================================
# RetryPolicy tests
# ===========================================================================


class TestRetryPolicy:
    def test_default_max_retries(self):
        """Default policy allows 3 retries (4 total attempts)."""
        policy = RetryPolicy()
        assert policy.max_retries == 3

    def test_should_retry_after_first_attempt(self):
        """After first attempt (attempt=1), should retry."""
        policy = RetryPolicy(max_retries=3)
        assert policy.should_retry(1) is True

    def test_should_retry_after_second_attempt(self):
        """After second attempt (attempt=2), should retry."""
        policy = RetryPolicy(max_retries=3)
        assert policy.should_retry(2) is True

    def test_should_retry_after_third_attempt(self):
        """After third attempt (attempt=3), should retry."""
        policy = RetryPolicy(max_retries=3)
        assert policy.should_retry(3) is True

    def test_should_not_retry_after_fourth_attempt(self):
        """After fourth attempt (attempt=4), retries exhausted."""
        policy = RetryPolicy(max_retries=3)
        assert policy.should_retry(4) is False

    def test_retries_exhausted(self):
        """retries_exhausted returns True when max attempts reached."""
        policy = RetryPolicy(max_retries=3)
        assert policy.retries_exhausted(1) is False
        assert policy.retries_exhausted(2) is False
        assert policy.retries_exhausted(3) is False
        assert policy.retries_exhausted(4) is True

    def test_custom_max_retries(self):
        """Custom max_retries changes the limit."""
        policy = RetryPolicy(max_retries=5)
        assert policy.should_retry(5) is True
        assert policy.should_retry(6) is False

    def test_zero_retries(self):
        """max_retries=0 means no retries at all."""
        policy = RetryPolicy(max_retries=0)
        assert policy.should_retry(1) is False
        assert policy.retries_exhausted(1) is True


# ===========================================================================
# Retry context enrichment tests
# ===========================================================================


class TestRetryContext:
    def test_no_context_for_first_attempt(self, session_logger):
        """First attempt has no previous context."""
        context = get_retry_context("t1", 1, session_logger)
        assert context == ""

    def test_context_includes_previous_session(self, session_logger):
        """Retry context includes the previous attempt's session log."""
        # Simulate a failed first attempt
        session_logger.log_dispatch(
            "t1", 1, "Do the task", "You are an implementer", ["Read"], 3.0
        )
        session_logger.log_output(
            "t1", 1, "Error: file not found", exit_code=1, duration_s=5.0
        )
        session_logger.log_validation("t1", 1, False, "Agent exited with error")

        context = get_retry_context("t1", 2, session_logger)
        assert "PREVIOUS ATTEMPT (1) SESSION LOG" in context
        assert "retry attempt 2" in context
        assert "Error: file not found" in context
        assert "exit_code=1" in context

    def test_context_includes_validation_details(self, session_logger):
        """Retry context includes validation results from previous attempt."""
        session_logger.log_dispatch("t1", 1, "Do task", "system", ["Read"], 3.0)
        session_logger.log_output("t1", 1, "Done!", exit_code=0, duration_s=2.0)
        session_logger.log_validation("t1", 1, False, "file_exists check failed")

        context = get_retry_context("t1", 2, session_logger)
        assert "file_exists check failed" in context
        assert "passed=False" in context

    def test_context_truncates_long_output(self, session_logger):
        """Very long output is truncated to keep context budget manageable."""
        long_output = "x" * 10000
        session_logger.log_dispatch("t1", 1, "Do task", "system", ["Read"], 3.0)
        session_logger.log_output("t1", 1, long_output, exit_code=1, duration_s=1.0)

        context = get_retry_context("t1", 2, session_logger)
        assert "truncated" in context
        # Should not contain the full 10000 chars
        assert len(context) < 10000

    def test_no_context_for_missing_session(self, session_logger):
        """If previous session doesn't exist, returns empty string."""
        context = get_retry_context("nonexistent", 2, session_logger)
        assert context == ""


# ===========================================================================
# Escalation creation tests
# ===========================================================================


class TestEscalation:
    def test_create_escalation(self, mutation_log, work_state, session_logger):
        """Escalation record is created with all required fields."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        session_logger.log_output("t1", 3, "Error: crash", exit_code=1, duration_s=1.0)

        esc = create_escalation(
            task=task,
            attempt=3,
            error="Agent exited with code 1",
            session_logger=session_logger,
            mutation_log=mutation_log,
        )

        assert esc["escalation_id"].startswith("esc-")
        assert esc["task_id"] == "t1"
        assert esc["task_name"] == "Task 1"
        assert esc["attempts"] == 3
        assert "Agent exited with code 1" in esc["error"]
        assert len(esc["suggested_actions"]) > 0
        assert esc["session_log_path"]

    def test_escalation_in_mutation_log(self, mutation_log, work_state, session_logger):
        """Escalation is recorded in the mutation log."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        create_escalation(
            task=task,
            attempt=3,
            error="error",
            session_logger=session_logger,
            mutation_log=mutation_log,
        )

        entries = mutation_log.read_all()
        esc_entries = [e for e in entries if e["type"] == "escalation_created"]
        assert len(esc_entries) == 1
        assert esc_entries[0]["data"]["task_id"] == "t1"

    def test_escalation_in_work_state(self, mutation_log, work_state, session_logger):
        """Escalation appears in work state after refresh."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        esc = create_escalation(
            task=task,
            attempt=3,
            error="persistent failure",
            session_logger=session_logger,
            mutation_log=mutation_log,
        )

        work_state.refresh()
        escs = work_state.list_escalations()
        assert len(escs) == 1
        assert escs[0]["id"] == esc["escalation_id"]
        assert escs[0]["status"] == "pending"
        assert escs[0]["task_id"] == "t1"

    def test_resolve_escalation(self, mutation_log, work_state, session_logger):
        """Resolving an escalation updates its status."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        esc = create_escalation(
            task=task,
            attempt=3,
            error="error",
            session_logger=session_logger,
            mutation_log=mutation_log,
        )

        resolve_escalation(
            esc["escalation_id"], mutation_log, resolution="Fixed manually"
        )

        work_state.refresh()
        resolved = work_state.get_escalation(esc["escalation_id"])
        assert resolved["status"] == "resolved"
        assert resolved["resolution"] == "Fixed manually"

    def test_list_pending_escalations(self, mutation_log, work_state, session_logger):
        """list_escalations(status='pending') filters correctly."""
        _create_task(mutation_log, "t1", "Task 1")
        _create_task(mutation_log, "t2", "Task 2")
        work_state.refresh()

        esc1 = create_escalation(
            task=work_state.get_task("t1"),
            attempt=3,
            error="err1",
            session_logger=session_logger,
            mutation_log=mutation_log,
        )
        create_escalation(
            task=work_state.get_task("t2"),
            attempt=3,
            error="err2",
            session_logger=session_logger,
            mutation_log=mutation_log,
        )

        # Resolve one
        resolve_escalation(esc1["escalation_id"], mutation_log, resolution="fixed")

        work_state.refresh()
        pending = work_state.list_escalations(status="pending")
        assert len(pending) == 1
        assert pending[0]["task_id"] == "t2"

        all_escs = work_state.list_escalations()
        assert len(all_escs) == 2

    def test_suggested_actions_timeout(self, mutation_log, work_state, session_logger):
        """Suggested actions include timeout-specific advice."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        esc = create_escalation(
            task=task,
            attempt=3,
            error="TIMEOUT: agent exceeded 1800s limit",
            session_logger=session_logger,
            mutation_log=mutation_log,
        )

        assert any(
            "timeout" in a.lower() or "scope" in a.lower()
            for a in esc["suggested_actions"]
        )

    def test_suggested_actions_validation(
        self, mutation_log, work_state, session_logger
    ):
        """Suggested actions include validation-specific advice."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        esc = create_escalation(
            task=task,
            attempt=3,
            error="Validation failed: file_exists check",
            session_logger=session_logger,
            mutation_log=mutation_log,
        )

        assert any(
            "done_when" in a.lower() or "validation" in a.lower()
            for a in esc["suggested_actions"]
        )


# ===========================================================================
# Task attempt tracking tests
# ===========================================================================


class TestAttemptTracking:
    def test_task_starts_with_zero_attempts(self, mutation_log, work_state):
        """New task has attempt_count=0 and max_retries=3 by default."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        assert task["attempt_count"] == 0
        assert task["max_retries"] == 3

    def test_custom_max_retries_on_task(self, mutation_log, work_state):
        """Task can be created with custom max_retries."""
        _create_task(mutation_log, "t1", "Task 1", max_retries=5)
        work_state.refresh()
        task = work_state.get_task("t1")

        assert task["max_retries"] == 5

    def test_attempt_count_incremented_on_failure(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Processor increments attempt_count when task fails."""
        _create_task(mutation_log, "t1", "Task 1")
        mutation_log.append("task_started", {"attempt": 1}, reason="test", task_id="t1")
        work_state.refresh()
        task = work_state.get_task("t1")

        result = AgentResult(output="Error!", exit_code=1, duration_s=1.0)
        process_completed(
            task=task,
            result=result,
            attempt=1,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        task = work_state.get_task("t1")
        assert task["attempt_count"] == 1
        assert task["status"] == "failed"

    def test_attempt_count_incremented_each_failure(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """attempt_count increases with each failure."""
        _create_task(mutation_log, "t1", "Task 1")

        for attempt_num in range(1, 4):
            mutation_log.append(
                "task_started", {"attempt": attempt_num}, reason="test", task_id="t1"
            )
            work_state.refresh()
            task = work_state.get_task("t1")

            result = AgentResult(output="Error!", exit_code=1, duration_s=1.0)
            process_completed(
                task=task,
                result=result,
                attempt=attempt_num,
                mutation_log=mutation_log,
                state=work_state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=tmp_project,
            )

            task = work_state.get_task("t1")
            assert task["attempt_count"] == attempt_num

    def test_escalated_after_max_retries(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Task status becomes 'escalated' after exceeding max_retries."""
        _create_task(mutation_log, "t1", "Task 1", max_retries=2)

        # Attempts 1 and 2 should result in 'failed' (retriable)
        for attempt_num in range(1, 3):
            mutation_log.append(
                "task_started", {"attempt": attempt_num}, reason="test", task_id="t1"
            )
            work_state.refresh()
            task = work_state.get_task("t1")

            result = AgentResult(output="Error!", exit_code=1, duration_s=1.0)
            process_completed(
                task=task,
                result=result,
                attempt=attempt_num,
                mutation_log=mutation_log,
                state=work_state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=tmp_project,
            )

            task = work_state.get_task("t1")
            assert task["status"] == "failed"

        # Attempt 3 (> max_retries=2) should escalate
        mutation_log.append("task_started", {"attempt": 3}, reason="test", task_id="t1")
        work_state.refresh()
        task = work_state.get_task("t1")

        result = AgentResult(output="Error!", exit_code=1, duration_s=1.0)
        process_completed(
            task=task,
            result=result,
            attempt=3,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        task = work_state.get_task("t1")
        assert task["status"] == "escalated"
        assert task["attempt_count"] == 3

    def test_failed_task_included_in_ready_tasks(self, mutation_log, work_state):
        """Failed tasks with retries remaining appear in get_ready_tasks."""
        _create_task(mutation_log, "t1", "Task 1", max_retries=3)
        mutation_log.append("task_started", {"attempt": 1}, reason="test", task_id="t1")
        mutation_log.append(
            "task_failed",
            {"attempt": 1, "attempt_count": 1},
            reason="test failure",
            task_id="t1",
        )
        work_state.refresh()

        ready = work_state.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0]["id"] == "t1"

    def test_escalated_task_not_in_ready_tasks(self, mutation_log, work_state):
        """Escalated tasks are not returned by get_ready_tasks."""
        _create_task(mutation_log, "t1", "Task 1", max_retries=1)
        mutation_log.append("task_started", {"attempt": 1}, reason="test", task_id="t1")
        mutation_log.append(
            "task_escalated",
            {"attempt": 2, "attempt_count": 2},
            reason="max retries exhausted",
            task_id="t1",
        )
        work_state.refresh()

        ready = work_state.get_ready_tasks()
        assert len(ready) == 0

    def test_max_retries_zero_escalates_immediately(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """With max_retries=0, first failure escalates immediately."""
        _create_task(mutation_log, "t1", "Task 1", max_retries=0)
        mutation_log.append("task_started", {"attempt": 1}, reason="test", task_id="t1")
        work_state.refresh()
        task = work_state.get_task("t1")

        result = AgentResult(output="Error!", exit_code=1, duration_s=1.0)
        process_completed(
            task=task,
            result=result,
            attempt=1,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        task = work_state.get_task("t1")
        assert task["status"] == "escalated"

        # Escalation record should exist
        escs = work_state.list_escalations()
        assert len(escs) == 1

    def test_attempt_count_survives_rebuild(self, mutation_log, work_state):
        """attempt_count is preserved through state rebuild."""
        _create_task(mutation_log, "t1", "Task 1", max_retries=3)
        mutation_log.append("task_started", {"attempt": 1}, reason="test", task_id="t1")
        mutation_log.append(
            "task_failed",
            {"attempt": 1, "attempt_count": 1},
            reason="test failure",
            task_id="t1",
        )

        work_state.rebuild()
        task = work_state.get_task("t1")
        assert task["attempt_count"] == 1
        assert task["max_retries"] == 3


# ===========================================================================
# Infrastructure failure tagging tests
# ===========================================================================


class TestInfrastructureFailureTagging:
    def test_sigterm_failure_has_infrastructure_flag(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """SIGTERM (exit_code 143) is tagged as infrastructure failure."""
        _create_task(mutation_log, "t1", "Task 1")
        mutation_log.append("task_started", {"attempt": 1}, reason="test", task_id="t1")
        work_state.refresh()
        task = work_state.get_task("t1")

        result = AgentResult(output="Terminated", exit_code=143, duration_s=1.0)
        process_completed(
            task=task,
            result=result,
            attempt=1,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        # Check mutation log has infrastructure: True
        entries = mutation_log.read_all()
        fail_entries = [e for e in entries if e["type"] == "task_failed"]
        assert len(fail_entries) == 1
        assert fail_entries[0]["data"]["infrastructure"] is True
        assert fail_entries[0]["data"]["exit_code"] == 143

    def test_sigterm_failure_does_not_increment_attempt_count(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """SIGTERM failure does not increment attempt_count in state."""
        _create_task(mutation_log, "t1", "Task 1")
        mutation_log.append("task_started", {"attempt": 1}, reason="test", task_id="t1")
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["attempt_count"] == 0

        result = AgentResult(output="Terminated", exit_code=143, duration_s=1.0)
        process_completed(
            task=task,
            result=result,
            attempt=1,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        task = work_state.get_task("t1")
        assert task["status"] == "failed"
        assert task["attempt_count"] == 0  # Not incremented for infrastructure failure

    def test_normal_failure_still_increments_attempt_count(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Normal task failure (exit_code 1) still increments attempt_count."""
        _create_task(mutation_log, "t1", "Task 1")
        mutation_log.append("task_started", {"attempt": 1}, reason="test", task_id="t1")
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["attempt_count"] == 0

        result = AgentResult(output="Error!", exit_code=1, duration_s=1.0)
        process_completed(
            task=task,
            result=result,
            attempt=1,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        # Check mutation log does NOT have infrastructure flag
        entries = mutation_log.read_all()
        fail_entries = [e for e in entries if e["type"] == "task_failed"]
        assert len(fail_entries) == 1
        assert "infrastructure" not in fail_entries[0]["data"]

        task = work_state.get_task("t1")
        assert task["status"] == "failed"
        assert task["attempt_count"] == 1  # Incremented for normal failure

    def test_infrastructure_failure_does_not_burn_retry_budget(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Multiple infrastructure failures don't exhaust retry budget."""
        _create_task(mutation_log, "t1", "Task 1", max_retries=3)

        # Simulate 3 infrastructure failures (SIGTERM)
        for attempt_num in range(1, 4):
            mutation_log.append(
                "task_started", {"attempt": attempt_num}, reason="test", task_id="t1"
            )
            work_state.refresh()
            task = work_state.get_task("t1")

            result = AgentResult(output="Terminated", exit_code=143, duration_s=1.0)
            process_completed(
                task=task,
                result=result,
                attempt=attempt_num,
                mutation_log=mutation_log,
                state=work_state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=tmp_project,
            )

        # After 3 infrastructure failures, attempt_count should still be 0
        task = work_state.get_task("t1")
        assert task["attempt_count"] == 0
        assert task["status"] == "failed"  # Still retriable, not escalated


# ===========================================================================
# Daemon retry integration tests
# ===========================================================================


class TestDaemonRetry:
    def test_retry_on_failure_then_succeed(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Task fails once, retries, and succeeds on second attempt."""
        dispatcher = FailNTimesDispatcher(fail_count=1)

        _create_task(
            mutation_log, "t1", "Task 1", done_when="do the thing", max_retries=3
        )
        work_state.refresh()

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(1.5)
        daemon.stop()
        thread.join(timeout=3)

        # Dispatcher should have been called at least twice
        assert dispatcher.call_count >= 2

        # Task should be completed
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "completed"

        # No escalations should exist
        escs = work_state.list_escalations()
        assert len(escs) == 0

    def test_retry_enriched_context_injected(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Retry attempt includes previous session log in the prompt."""
        dispatcher = FailNTimesDispatcher(
            fail_count=1, fail_output="Error: missing file.txt"
        )

        _create_task(
            mutation_log, "t1", "Task 1", done_when="do the thing", max_retries=3
        )
        work_state.refresh()

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(1.5)
        daemon.stop()
        thread.join(timeout=3)

        # The second dispatch should contain enriched context from the first failure
        assert dispatcher.call_count >= 2
        # Check the retry prompt contains previous session info
        retry_prompt = dispatcher.dispatched[1][0]  # (prompt, system_prompt)
        assert "PREVIOUS ATTEMPT" in retry_prompt
        assert "missing file.txt" in retry_prompt

    def test_escalation_after_retries_exhausted(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Escalation is created when all retries are exhausted, status is 'escalated'."""
        dispatcher = AlwaysFailDispatcher(fail_output="Error: persistent failure")

        _create_task(
            mutation_log, "t1", "Task 1", done_when="do the thing", max_retries=2
        )
        work_state.refresh()

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(2.5)
        daemon.stop()
        thread.join(timeout=3)

        # Dispatcher should have been called 3 times (1 original + 2 retries)
        assert dispatcher.call_count == 3

        # Task should be escalated (not failed)
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "escalated"
        assert task["attempt_count"] == 3

        # An escalation should exist
        escs = work_state.list_escalations()
        assert len(escs) == 1
        assert escs[0]["task_id"] == "t1"
        assert escs[0]["status"] == "pending"
        assert "persistent failure" in escs[0]["error"]
        assert escs[0]["attempts"] == 3

    def test_no_retry_when_max_retries_zero(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """With max_retries=0, task escalates immediately and escalation is created."""
        dispatcher = AlwaysFailDispatcher()

        _create_task(
            mutation_log, "t1", "Task 1", done_when="do the thing", max_retries=0
        )
        work_state.refresh()

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(0.8)
        daemon.stop()
        thread.join(timeout=3)

        # Only dispatched once
        assert dispatcher.call_count == 1

        # Task should be escalated
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "escalated"

        # Escalation should exist
        escs = work_state.list_escalations()
        assert len(escs) == 1

    def test_retry_fails_twice_then_succeeds(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Task fails twice, succeeds on third attempt (exactly at the retry limit)."""
        dispatcher = FailNTimesDispatcher(fail_count=2)

        _create_task(
            mutation_log, "t1", "Task 1", done_when="do the thing", max_retries=2
        )
        work_state.refresh()

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(2.5)
        daemon.stop()
        thread.join(timeout=3)

        assert dispatcher.call_count == 3

        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "completed"

        # No escalation
        escs = work_state.list_escalations()
        assert len(escs) == 0

    def test_retry_with_custom_max_retries(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Custom max_retries=1 allows only one retry."""
        dispatcher = AlwaysFailDispatcher()

        _create_task(
            mutation_log, "t1", "Task 1", done_when="do the thing", max_retries=1
        )
        work_state.refresh()

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(1.5)
        daemon.stop()
        thread.join(timeout=3)

        # 1 original + 1 retry = 2 total
        assert dispatcher.call_count == 2

        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "escalated"

        escs = work_state.list_escalations()
        assert len(escs) == 1
        assert escs[0]["attempts"] == 2

    def test_escalation_includes_session_log_path(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Escalation record includes the path to the session log."""
        dispatcher = AlwaysFailDispatcher()

        _create_task(
            mutation_log, "t1", "Task 1", done_when="do the thing", max_retries=0
        )
        work_state.refresh()

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(0.8)
        daemon.stop()
        thread.join(timeout=3)

        work_state.refresh()
        escs = work_state.list_escalations()
        assert len(escs) == 1
        assert "t1-attempt-1" in escs[0]["session_log_path"]

    def test_escalation_suggested_actions_present(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Escalation record has suggested actions."""
        dispatcher = AlwaysFailDispatcher()

        _create_task(
            mutation_log, "t1", "Task 1", done_when="do the thing", max_retries=0
        )
        work_state.refresh()

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(0.8)
        daemon.stop()
        thread.join(timeout=3)

        work_state.refresh()
        escs = work_state.list_escalations()
        assert len(escs) == 1
        assert isinstance(escs[0]["suggested_actions"], list)
        assert len(escs[0]["suggested_actions"]) > 0

    def test_timeout_failure_retried_same_as_other_failures(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Timeout failures (exit_code=-1) are retried like any other failure."""
        dispatcher = TimeoutDispatcher()

        _create_task(
            mutation_log, "t1", "Task 1", done_when="do the thing", max_retries=2
        )
        work_state.refresh()

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(2.5)
        daemon.stop()
        thread.join(timeout=3)

        # Should have been retried: 1 original + 2 retries = 3 total
        assert dispatcher.call_count == 3

        # Task should be escalated (all retries exhausted)
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "escalated"
        assert task["attempt_count"] == 3

        # Escalation should mention timeout
        escs = work_state.list_escalations()
        assert len(escs) == 1
        assert "TIMEOUT" in escs[0]["error"]

    def test_default_max_retries_is_three(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Default max_retries=3 allows 4 total attempts before escalation."""
        dispatcher = AlwaysFailDispatcher()

        # Use default max_retries (3)
        _create_task(mutation_log, "t1", "Task 1", done_when="do the thing")
        work_state.refresh()

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(3.5)
        daemon.stop()
        thread.join(timeout=3)

        # 1 original + 3 retries = 4 total
        assert dispatcher.call_count == 4

        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "escalated"
        assert task["attempt_count"] == 4


# ===========================================================================
# CLI escalation tests
# ===========================================================================


class TestEscalationCLI:
    def test_escalations_list_command(
        self, mutation_log, work_state, session_logger, tmp_project
    ):
        """corc escalations lists pending escalations."""
        from click.testing import CliRunner
        from corc.cli import cli

        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        create_escalation(
            task=task,
            attempt=3,
            error="test error",
            session_logger=session_logger,
            mutation_log=mutation_log,
        )

        runner = CliRunner()
        # Override get_paths for the CLI
        import corc.cli as cli_module

        original_get_all = cli_module._get_all

        def mock_get_all():
            return (
                {
                    "root": tmp_project,
                    "mutations": tmp_project / "data" / "mutations.jsonl",
                    "state_db": tmp_project / "data" / "state.db",
                    "events_dir": tmp_project / "data" / "events",
                    "sessions_dir": tmp_project / "data" / "sessions",
                    "knowledge_dir": tmp_project / "knowledge",
                    "knowledge_db": tmp_project / "data" / "knowledge.db",
                    "corc_dir": tmp_project / ".corc",
                },
                mutation_log,
                work_state,
                AuditLog(tmp_project / "data" / "events"),
                session_logger,
                None,  # knowledge store not needed
            )

        cli_module._get_all = mock_get_all
        try:
            result = runner.invoke(cli, ["escalations"])
            assert result.exit_code == 0
            assert "t1" in result.output
            assert "Task 1" in result.output
        finally:
            cli_module._get_all = original_get_all

    def test_escalation_show_command(
        self, mutation_log, work_state, session_logger, tmp_project
    ):
        """corc escalation show ESC_ID shows full detail."""
        from click.testing import CliRunner
        from corc.cli import cli

        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        esc = create_escalation(
            task=task,
            attempt=3,
            error="test error details",
            session_logger=session_logger,
            mutation_log=mutation_log,
        )

        import corc.cli as cli_module

        original_get_all = cli_module._get_all

        def mock_get_all():
            return (
                {
                    "root": tmp_project,
                    "mutations": tmp_project / "data" / "mutations.jsonl",
                    "state_db": tmp_project / "data" / "state.db",
                    "events_dir": tmp_project / "data" / "events",
                    "sessions_dir": tmp_project / "data" / "sessions",
                    "knowledge_dir": tmp_project / "knowledge",
                    "knowledge_db": tmp_project / "data" / "knowledge.db",
                    "corc_dir": tmp_project / ".corc",
                },
                mutation_log,
                work_state,
                AuditLog(tmp_project / "data" / "events"),
                session_logger,
                None,
            )

        cli_module._get_all = mock_get_all
        try:
            work_state.refresh()
            result = runner = CliRunner()
            result = runner.invoke(cli, ["escalation", "show", esc["escalation_id"]])
            assert result.exit_code == 0
            assert esc["escalation_id"] in result.output
            assert "test error details" in result.output
            assert "Suggested actions" in result.output
            assert "Task 1" in result.output
        finally:
            cli_module._get_all = original_get_all

    def test_escalation_resolve_command(
        self, mutation_log, work_state, session_logger, tmp_project
    ):
        """corc escalation resolve ESC_ID marks it resolved."""
        from click.testing import CliRunner
        from corc.cli import cli

        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        esc = create_escalation(
            task=task,
            attempt=3,
            error="test error",
            session_logger=session_logger,
            mutation_log=mutation_log,
        )

        import corc.cli as cli_module

        original_get_all = cli_module._get_all

        def mock_get_all():
            return (
                {
                    "root": tmp_project,
                    "mutations": tmp_project / "data" / "mutations.jsonl",
                    "state_db": tmp_project / "data" / "state.db",
                    "events_dir": tmp_project / "data" / "events",
                    "sessions_dir": tmp_project / "data" / "sessions",
                    "knowledge_dir": tmp_project / "knowledge",
                    "knowledge_db": tmp_project / "data" / "knowledge.db",
                    "corc_dir": tmp_project / ".corc",
                },
                mutation_log,
                work_state,
                AuditLog(tmp_project / "data" / "events"),
                session_logger,
                None,
            )

        cli_module._get_all = mock_get_all
        try:
            work_state.refresh()
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "escalation",
                    "resolve",
                    esc["escalation_id"],
                    "--resolution",
                    "Fixed manually",
                ],
            )
            assert result.exit_code == 0
            assert "resolved" in result.output

            work_state.refresh()
            resolved = work_state.get_escalation(esc["escalation_id"])
            assert resolved["status"] == "resolved"
            assert resolved["resolution"] == "Fixed manually"
        finally:
            cli_module._get_all = original_get_all

    def test_escalation_resolve_with_unblock(
        self, mutation_log, work_state, session_logger, tmp_project
    ):
        """corc escalation resolve --unblock resets the task to pending."""
        from click.testing import CliRunner
        from corc.cli import cli

        _create_task(mutation_log, "t1", "Task 1")
        # Mark task as escalated
        mutation_log.append("task_started", {"attempt": 1}, reason="test", task_id="t1")
        mutation_log.append(
            "task_escalated",
            {"attempt": 1, "attempt_count": 1},
            reason="test escalation",
            task_id="t1",
        )
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "escalated"

        esc = create_escalation(
            task=task,
            attempt=3,
            error="test error",
            session_logger=session_logger,
            mutation_log=mutation_log,
        )

        import corc.cli as cli_module

        original_get_all = cli_module._get_all

        def mock_get_all():
            return (
                {
                    "root": tmp_project,
                    "mutations": tmp_project / "data" / "mutations.jsonl",
                    "state_db": tmp_project / "data" / "state.db",
                    "events_dir": tmp_project / "data" / "events",
                    "sessions_dir": tmp_project / "data" / "sessions",
                    "knowledge_dir": tmp_project / "knowledge",
                    "knowledge_db": tmp_project / "data" / "knowledge.db",
                    "corc_dir": tmp_project / ".corc",
                },
                mutation_log,
                work_state,
                AuditLog(tmp_project / "data" / "events"),
                session_logger,
                None,
            )

        cli_module._get_all = mock_get_all
        try:
            work_state.refresh()
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "escalation",
                    "resolve",
                    esc["escalation_id"],
                    "--resolution",
                    "Fixed",
                    "--unblock",
                ],
            )
            assert result.exit_code == 0
            assert "pending" in result.output

            work_state.refresh()
            task = work_state.get_task("t1")
            assert task["status"] == "pending"
        finally:
            cli_module._get_all = original_get_all

    def test_escalation_show_not_found(
        self, mutation_log, work_state, session_logger, tmp_project
    ):
        """corc escalation show with bad ID returns error."""
        from click.testing import CliRunner
        from corc.cli import cli

        import corc.cli as cli_module

        original_get_all = cli_module._get_all

        def mock_get_all():
            return (
                {
                    "root": tmp_project,
                    "mutations": tmp_project / "data" / "mutations.jsonl",
                    "state_db": tmp_project / "data" / "state.db",
                    "events_dir": tmp_project / "data" / "events",
                    "sessions_dir": tmp_project / "data" / "sessions",
                    "knowledge_dir": tmp_project / "knowledge",
                    "knowledge_db": tmp_project / "data" / "knowledge.db",
                    "corc_dir": tmp_project / ".corc",
                },
                mutation_log,
                work_state,
                AuditLog(tmp_project / "data" / "events"),
                session_logger,
                None,
            )

        cli_module._get_all = mock_get_all
        try:
            runner = CliRunner()
            result = runner.invoke(cli, ["escalation", "show", "esc-nonexistent"])
            assert result.exit_code != 0
            assert "not found" in result.output
        finally:
            cli_module._get_all = original_get_all


# ===========================================================================
# State rebuild tests
# ===========================================================================


class TestStateRebuild:
    def test_escalations_survive_rebuild(
        self, mutation_log, work_state, session_logger
    ):
        """Escalations are correctly rebuilt from mutation log."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        esc = create_escalation(
            task=task,
            attempt=3,
            error="test error",
            session_logger=session_logger,
            mutation_log=mutation_log,
        )

        # Resolve it
        resolve_escalation(esc["escalation_id"], mutation_log, resolution="fixed")

        # Full rebuild
        work_state.rebuild()

        escs = work_state.list_escalations()
        assert len(escs) == 1
        assert escs[0]["status"] == "resolved"
        assert escs[0]["resolution"] == "fixed"

    def test_multiple_escalations_rebuild(
        self, mutation_log, work_state, session_logger
    ):
        """Multiple escalations survive state rebuild."""
        _create_task(mutation_log, "t1", "Task 1")
        _create_task(mutation_log, "t2", "Task 2")
        work_state.refresh()

        create_escalation(
            task=work_state.get_task("t1"),
            attempt=3,
            error="err1",
            session_logger=session_logger,
            mutation_log=mutation_log,
        )
        create_escalation(
            task=work_state.get_task("t2"),
            attempt=2,
            error="err2",
            session_logger=session_logger,
            mutation_log=mutation_log,
        )

        work_state.rebuild()
        escs = work_state.list_escalations()
        assert len(escs) == 2

    def test_escalated_status_survives_rebuild(
        self, mutation_log, work_state, session_logger
    ):
        """Task with 'escalated' status survives state rebuild."""
        _create_task(mutation_log, "t1", "Task 1", max_retries=1)
        mutation_log.append("task_started", {"attempt": 1}, reason="test", task_id="t1")
        mutation_log.append(
            "task_failed",
            {"attempt": 1, "attempt_count": 1},
            reason="test failure",
            task_id="t1",
        )
        mutation_log.append(
            "task_started", {"attempt": 2}, reason="retry", task_id="t1"
        )
        mutation_log.append(
            "task_escalated",
            {"attempt": 2, "attempt_count": 2},
            reason="max retries exhausted",
            task_id="t1",
        )

        work_state.rebuild()
        task = work_state.get_task("t1")
        assert task["status"] == "escalated"
        assert task["attempt_count"] == 2
        assert task["max_retries"] == 1
