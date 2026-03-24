"""Tests for the daemon core: scheduler, executor, processor, and daemon loop.

Uses a MockDispatcher to test the full pipeline without calling real LLMs.
"""

import json
import os
import signal
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from corc.audit import AuditLog
from corc.daemon import Daemon, stop_daemon
from corc.dispatch import AgentDispatcher, AgentResult, Constraints
from corc.executor import CompletedTask, Executor
from corc.mutations import MutationLog
from corc.processor import (
    ProcessResult,
    _extract_findings,
    _parse_done_when,
    process_completed,
)
from corc.scheduler import get_ready_tasks
from corc.sessions import SessionLogger
from corc.state import WorkState


# ---------------------------------------------------------------------------
# Mock dispatcher
# ---------------------------------------------------------------------------


class MockDispatcher(AgentDispatcher):
    """A dispatcher that returns configurable results without calling any LLM."""

    def __init__(self, default_result: AgentResult | None = None, delay: float = 0):
        self.default_result = default_result or AgentResult(
            output="Mock output: task completed successfully.",
            exit_code=0,
            duration_s=0.1,
        )
        self.delay = delay
        self.dispatched: list[tuple[str, str, Constraints]] = []
        self._results: dict[str, AgentResult] = {}  # keyed by substring in prompt

    def set_result_for(self, prompt_substring: str, result: AgentResult):
        """Configure a specific result for prompts containing the given substring."""
        self._results[prompt_substring] = result

    def dispatch(
        self,
        prompt: str,
        system_prompt: str,
        constraints: Constraints,
        pid_callback=None,
        event_callback=None,
        cwd=None,
    ) -> AgentResult:
        self.dispatched.append((prompt, system_prompt, constraints))
        if self.delay:
            time.sleep(self.delay)
        # Check for prompt-specific results
        for substring, result in self._results.items():
            if substring in prompt:
                return result
        return self.default_result


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


@pytest.fixture
def mock_dispatcher():
    return MockDispatcher()


def _create_task(
    mutation_log,
    task_id,
    name,
    done_when="tests_pass",
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
# Scheduler tests
# ===========================================================================


class TestScheduler:
    def test_get_ready_tasks_empty(self, work_state):
        """No tasks means no ready tasks."""
        result = get_ready_tasks(work_state, parallel_limit=1)
        assert result == []

    def test_get_ready_tasks_one_pending(self, mutation_log, work_state):
        """A pending task with no deps is ready."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()

        result = get_ready_tasks(work_state, parallel_limit=1)
        assert len(result) == 1
        assert result[0]["id"] == "t1"

    def test_get_ready_tasks_respects_parallel_limit(self, mutation_log, work_state):
        """Returns at most parallel_limit tasks."""
        _create_task(mutation_log, "t1", "Task 1")
        _create_task(mutation_log, "t2", "Task 2")
        _create_task(mutation_log, "t3", "Task 3")
        work_state.refresh()

        result = get_ready_tasks(work_state, parallel_limit=2)
        assert len(result) == 2

    def test_get_ready_tasks_accounts_for_running(self, mutation_log, work_state):
        """Running tasks count against the parallel limit."""
        _create_task(mutation_log, "t1", "Task 1")
        _create_task(mutation_log, "t2", "Task 2")
        mutation_log.append("task_started", {"attempt": 1}, reason="test", task_id="t1")
        work_state.refresh()

        # parallel=2, 1 running, so 1 slot available
        result = get_ready_tasks(work_state, parallel_limit=2)
        assert len(result) == 1
        assert result[0]["id"] == "t2"

    def test_get_ready_tasks_no_slots(self, mutation_log, work_state):
        """All slots occupied returns empty."""
        _create_task(mutation_log, "t1", "Task 1")
        _create_task(mutation_log, "t2", "Task 2")
        mutation_log.append("task_started", {"attempt": 1}, reason="test", task_id="t1")
        work_state.refresh()

        result = get_ready_tasks(work_state, parallel_limit=1)
        assert len(result) == 0

    def test_get_ready_tasks_respects_dependencies(self, mutation_log, work_state):
        """Tasks with unmet deps are not ready."""
        _create_task(mutation_log, "t1", "Task 1")
        _create_task(mutation_log, "t2", "Task 2", depends_on=["t1"])
        work_state.refresh()

        result = get_ready_tasks(work_state, parallel_limit=5)
        assert len(result) == 1
        assert result[0]["id"] == "t1"

    def test_get_ready_tasks_dep_completed_unblocks(self, mutation_log, work_state):
        """Completing a dep unblocks the dependent task."""
        _create_task(mutation_log, "t1", "Task 1")
        _create_task(mutation_log, "t2", "Task 2", depends_on=["t1"])
        mutation_log.append("task_started", {"attempt": 1}, reason="test", task_id="t1")
        mutation_log.append(
            "task_completed", {"findings": []}, reason="test", task_id="t1"
        )
        work_state.refresh()

        result = get_ready_tasks(work_state, parallel_limit=5)
        assert len(result) == 1
        assert result[0]["id"] == "t2"


# ===========================================================================
# Executor tests
# ===========================================================================


class TestExecutor:
    def test_dispatch_marks_running(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        mock_dispatcher,
        tmp_project,
    ):
        """Dispatching a task marks it as running in the mutation log."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = Executor(
            dispatcher=mock_dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        executor.dispatch(task)
        # Task should be marked as running in the log
        work_state.refresh()
        updated = work_state.get_task("t1")
        assert updated["status"] == "running"
        executor.shutdown()

    def test_dispatch_and_poll(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        mock_dispatcher,
        tmp_project,
    ):
        """Dispatch a task and poll for completion."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = Executor(
            dispatcher=mock_dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        executor.dispatch(task)
        assert executor.in_flight_count == 1

        # Wait for completion
        time.sleep(0.3)
        completed = executor.poll_completed()
        assert len(completed) == 1
        assert completed[0].task["id"] == "t1"
        assert completed[0].result.exit_code == 0
        assert executor.in_flight_count == 0
        executor.shutdown()

    def test_dispatch_records_session(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        mock_dispatcher,
        tmp_project,
    ):
        """Dispatch logs the prompt and output to session logger."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = Executor(
            dispatcher=mock_dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        executor.dispatch(task)
        time.sleep(0.3)
        executor.poll_completed()

        # Session should have dispatch and output entries
        session = session_logger.read_session("t1", 1)
        types = [e["type"] for e in session]
        assert "dispatch" in types
        assert "output" in types
        executor.shutdown()

    def test_parallel_dispatch(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Multiple tasks can be dispatched in parallel."""
        dispatcher = MockDispatcher(delay=0.2)

        _create_task(mutation_log, "t1", "Task 1")
        _create_task(mutation_log, "t2", "Task 2")
        _create_task(mutation_log, "t3", "Task 3")
        work_state.refresh()

        executor = Executor(
            dispatcher=dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
            max_workers=3,
        )

        tasks = [work_state.get_task(tid) for tid in ("t1", "t2", "t3")]
        for task in tasks:
            executor.dispatch(task)

        assert executor.in_flight_count == 3

        # Wait for all to complete
        time.sleep(0.5)
        completed = executor.poll_completed()
        assert len(completed) == 3
        assert executor.in_flight_count == 0
        executor.shutdown()

    def test_dispatch_error_handling(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Dispatcher errors are caught and returned as failed results."""

        class FailingDispatcher(AgentDispatcher):
            def dispatch(
                self,
                prompt,
                system_prompt,
                constraints,
                pid_callback=None,
                event_callback=None,
                cwd=None,
            ):
                raise RuntimeError("Connection failed")

        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = Executor(
            dispatcher=FailingDispatcher(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        executor.dispatch(task)
        time.sleep(0.3)
        completed = executor.poll_completed()
        assert len(completed) == 1
        assert completed[0].result.exit_code == -1
        assert "Connection failed" in completed[0].result.output
        executor.shutdown()


# ===========================================================================
# Processor tests
# ===========================================================================


class TestProcessor:
    def test_process_successful_no_rules(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Agent exit 0 with no validation rules → completed."""
        _create_task(mutation_log, "t1", "Task 1", done_when="implement the feature")
        mutation_log.append("task_started", {"attempt": 1}, reason="test", task_id="t1")
        work_state.refresh()
        task = work_state.get_task("t1")

        result = AgentResult(output="Done!", exit_code=0, duration_s=1.0)
        pr = process_completed(
            task=task,
            result=result,
            attempt=1,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        assert pr.passed is True
        assert pr.task_id == "t1"

        # State should be updated
        updated = work_state.get_task("t1")
        assert updated["status"] == "completed"

    def test_process_agent_failure(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Agent exit != 0 → task failed."""
        _create_task(mutation_log, "t1", "Task 1")
        mutation_log.append("task_started", {"attempt": 1}, reason="test", task_id="t1")
        work_state.refresh()
        task = work_state.get_task("t1")

        result = AgentResult(output="Error!", exit_code=1, duration_s=1.0)
        pr = process_completed(
            task=task,
            result=result,
            attempt=1,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        assert pr.passed is False
        work_state.refresh()
        updated = work_state.get_task("t1")
        assert updated["status"] == "failed"

    def test_process_with_file_exists_rule(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Validation rule file_exists succeeds when file is present."""
        (tmp_project / "output.txt").write_text("hello")
        rules = json.dumps([{"file_exists": "output.txt"}])
        _create_task(mutation_log, "t1", "Task 1", done_when=rules)
        mutation_log.append("task_started", {"attempt": 1}, reason="test", task_id="t1")
        work_state.refresh()
        task = work_state.get_task("t1")

        result = AgentResult(output="Created output.txt", exit_code=0, duration_s=1.0)
        pr = process_completed(
            task=task,
            result=result,
            attempt=1,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        assert pr.passed is True

    def test_process_with_file_exists_rule_fails(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Validation rule file_exists fails when file is missing."""
        rules = json.dumps([{"file_exists": "missing.txt"}])
        _create_task(mutation_log, "t1", "Task 1", done_when=rules)
        mutation_log.append("task_started", {"attempt": 1}, reason="test", task_id="t1")
        work_state.refresh()
        task = work_state.get_task("t1")

        result = AgentResult(output="Tried but failed", exit_code=0, duration_s=1.0)
        pr = process_completed(
            task=task,
            result=result,
            attempt=1,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        assert pr.passed is False
        work_state.refresh()
        updated = work_state.get_task("t1")
        assert updated["status"] == "failed"

    def test_extract_findings(self):
        """Findings are extracted from lines prefixed with FINDING:."""
        output = "Some output\nFINDING: The API uses REST\nMore output\nFINDING: Uses Python 3.11\n"
        findings = _extract_findings(output)
        assert findings == ["The API uses REST", "Uses Python 3.11"]

    def test_extract_findings_empty(self):
        """No FINDING: lines means empty findings."""
        assert _extract_findings("just normal output") == []

    def test_parse_done_when_json_list(self):
        """JSON list of rules is parsed correctly."""
        rules = _parse_done_when('[{"file_exists": "foo.py"}, "tests_pass"]')
        assert len(rules) == 2
        assert rules[0] == {"file_exists": "foo.py"}
        assert rules[1] == "tests_pass"

    def test_parse_done_when_plain_text(self):
        """Plain text done_when returns empty (no auto-validation)."""
        rules = _parse_done_when("All tests pass and code is clean")
        assert rules == []

    def test_parse_done_when_empty(self):
        """Empty done_when returns empty."""
        assert _parse_done_when("") == []
        assert _parse_done_when(None) == []

    def test_process_extracts_findings(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Findings from agent output are stored in the task."""
        _create_task(mutation_log, "t1", "Task 1", done_when="do the thing")
        mutation_log.append("task_started", {"attempt": 1}, reason="test", task_id="t1")
        work_state.refresh()
        task = work_state.get_task("t1")

        result = AgentResult(
            output="Done!\nFINDING: Uses SQLite WAL mode\nFINDING: Has 3 tables",
            exit_code=0,
            duration_s=1.0,
        )
        pr = process_completed(
            task=task,
            result=result,
            attempt=1,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        assert pr.passed is True
        assert len(pr.findings) == 2
        assert "Uses SQLite WAL mode" in pr.findings

    def test_process_rejects_double_completion(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Processor skips tasks that are already in completed state."""
        _create_task(mutation_log, "t1", "Task 1", done_when="do the thing")
        mutation_log.append("task_started", {"attempt": 1}, reason="test", task_id="t1")
        work_state.refresh()
        task = work_state.get_task("t1")

        result = AgentResult(output="Done!", exit_code=0, duration_s=1.0)

        # First completion — should succeed
        pr1 = process_completed(
            task=task,
            result=result,
            attempt=1,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )
        assert pr1.passed is True
        work_state.refresh()
        assert work_state.get_task("t1")["status"] == "completed"

        # Count mutations before second attempt
        mutations_before = len(mutation_log.read_all())

        # Second completion — should be skipped without writing a mutation
        pr2 = process_completed(
            task=task,
            result=result,
            attempt=2,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )
        assert pr2.passed is True
        assert "already completed" in pr2.details[0][1]

        # No new mutations should have been written
        mutations_after = len(mutation_log.read_all())
        assert mutations_after == mutations_before


# ===========================================================================
# Daemon loop tests
# ===========================================================================


class TestDaemon:
    def test_daemon_starts_and_stops(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        mock_dispatcher,
        tmp_project,
    ):
        """Daemon starts, runs at least one tick, and stops gracefully."""
        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=mock_dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
        )

        # Run daemon in a thread, stop after a short delay
        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(0.3)
        daemon.stop()
        thread.join(timeout=2)
        assert not thread.is_alive()

    def test_daemon_pid_file(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        mock_dispatcher,
        tmp_project,
    ):
        """Daemon writes and cleans up PID file."""
        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=mock_dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
        )

        pid_file = tmp_project / ".corc" / "daemon.pid"

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(0.2)

        # PID file should exist while running
        assert pid_file.exists()
        pid = int(pid_file.read_text().strip())
        assert pid == os.getpid()

        daemon.stop()
        thread.join(timeout=2)

        # PID file should be cleaned up
        assert not pid_file.exists()

    def test_daemon_dispatches_ready_tasks(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        mock_dispatcher,
        tmp_project,
    ):
        """Daemon automatically dispatches ready tasks."""
        _create_task(mutation_log, "t1", "Task 1", done_when="do the thing")
        work_state.refresh()

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=mock_dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(0.5)
        daemon.stop()
        thread.join(timeout=2)

        # Task should have been dispatched
        assert len(mock_dispatcher.dispatched) >= 1

        # Task should be completed (exit 0, no validation rules)
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "completed"

    def test_daemon_handles_dag_dependencies(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        mock_dispatcher,
        tmp_project,
    ):
        """Daemon processes DAG: t2 only dispatched after t1 completes."""
        _create_task(mutation_log, "t1", "Task 1", done_when="do thing 1")
        _create_task(
            mutation_log, "t2", "Task 2", done_when="do thing 2", depends_on=["t1"]
        )
        work_state.refresh()

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=mock_dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(1.0)
        daemon.stop()
        thread.join(timeout=2)

        # Both tasks should be dispatched (t1 first, then t2)
        assert len(mock_dispatcher.dispatched) >= 2

        work_state.refresh()
        assert work_state.get_task("t1")["status"] == "completed"
        assert work_state.get_task("t2")["status"] == "completed"

    def test_daemon_parallel_dispatch(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Daemon dispatches up to --parallel tasks concurrently."""
        dispatcher = MockDispatcher(delay=0.2)

        _create_task(mutation_log, "t1", "Task 1", done_when="thing 1")
        _create_task(mutation_log, "t2", "Task 2", done_when="thing 2")
        _create_task(mutation_log, "t3", "Task 3", done_when="thing 3")
        work_state.refresh()

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=tmp_project,
            parallel=3,
            poll_interval=0.1,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(1.0)
        daemon.stop()
        thread.join(timeout=2)

        # All 3 should have been dispatched
        assert len(dispatcher.dispatched) == 3

        work_state.refresh()
        for tid in ("t1", "t2", "t3"):
            assert work_state.get_task(tid)["status"] == "completed"

    def test_daemon_parallel_limit_enforced(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Daemon doesn't exceed --parallel limit."""
        dispatcher = MockDispatcher(delay=0.3)

        _create_task(mutation_log, "t1", "Task 1", done_when="thing 1")
        _create_task(mutation_log, "t2", "Task 2", done_when="thing 2")
        _create_task(mutation_log, "t3", "Task 3", done_when="thing 3")
        work_state.refresh()

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=tmp_project,
            parallel=1,  # Only 1 at a time
            poll_interval=0.1,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(2.0)
        daemon.stop()
        thread.join(timeout=2)

        work_state.refresh()
        # All should eventually complete, but only 1 at a time
        completed = [t for t in work_state.list_tasks() if t["status"] == "completed"]
        assert len(completed) == 3

    def test_daemon_once_mode(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        mock_dispatcher,
        tmp_project,
    ):
        """--once mode: daemon processes one task and stops."""
        _create_task(mutation_log, "t1", "Task 1", done_when="thing 1")
        _create_task(mutation_log, "t2", "Task 2", done_when="thing 2")
        work_state.refresh()

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=mock_dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
            once=True,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        thread.join(timeout=3)

        # Daemon should have stopped on its own
        assert not thread.is_alive()

        # Only one task should have been dispatched
        assert len(mock_dispatcher.dispatched) == 1

    def test_daemon_target_task(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        mock_dispatcher,
        tmp_project,
    ):
        """--task mode: daemon dispatches only the specified task."""
        _create_task(mutation_log, "t1", "Task 1", done_when="thing 1")
        _create_task(mutation_log, "t2", "Task 2", done_when="thing 2")
        work_state.refresh()

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=mock_dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
            task_id="t2",
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(0.5)
        daemon.stop()
        thread.join(timeout=2)

        # Only t2 should have been dispatched
        assert len(mock_dispatcher.dispatched) == 1
        assert (
            "Task 2" in mock_dispatcher.dispatched[0][1]
        )  # system prompt contains task name

    def test_daemon_failed_task_not_retried_when_disabled(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """A failed task with max_retries=0 is not re-dispatched (escalated immediately)."""
        fail_result = AgentResult(output="Error!", exit_code=1, duration_s=0.1)
        dispatcher = MockDispatcher(default_result=fail_result)

        _create_task(mutation_log, "t1", "Task 1", done_when="thing 1", max_retries=0)
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
        time.sleep(0.5)
        daemon.stop()
        thread.join(timeout=2)

        # Task should be dispatched only once
        assert len(dispatcher.dispatched) == 1

        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "escalated"

    def test_stop_daemon_via_pid(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        mock_dispatcher,
        tmp_project,
    ):
        """stop_daemon sends SIGTERM via PID file.

        Since the daemon runs in the same process during tests (in a thread),
        we can't send SIGTERM without killing the test. Instead we test that
        stop_daemon correctly reads the PID file and returns True, and that
        daemon.stop() works for graceful shutdown.
        """
        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=mock_dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(0.2)

        # Verify PID file exists and contains our PID
        pid_file = tmp_project / ".corc" / "daemon.pid"
        assert pid_file.exists()
        pid = int(pid_file.read_text().strip())
        assert pid == os.getpid()

        # Use daemon.stop() directly (stop_daemon would SIGTERM our own process)
        daemon.stop()
        thread.join(timeout=3)
        assert not thread.is_alive()

    def test_stop_daemon_no_running(self, tmp_project):
        """stop_daemon returns False when no daemon is running."""
        (tmp_project / ".corc").mkdir(exist_ok=True)
        result = stop_daemon(tmp_project)
        assert result is False

    def test_cleanup_returns_fast_with_inflight_futures(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        tmp_project,
    ):
        """_cleanup returns in under 2 seconds even with slow in-flight futures.

        The daemon uses executor.shutdown(wait=False) so that `corc stop`
        returns immediately. In-flight agents become orphans and are
        reconciled on the next `corc start`.
        """
        # Use a dispatcher that blocks for a long time to simulate in-flight agents
        slow_dispatcher = MockDispatcher(delay=10.0)

        _create_task(mutation_log, "slow-1", "Slow Task", done_when="do it")
        work_state.refresh()
        task = work_state.get_task("slow-1")

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=slow_dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
        )

        # Manually set up the daemon state (bypass start() to test _cleanup directly)
        daemon._write_pid()
        daemon.executor.dispatch(task)
        assert daemon.executor.in_flight_count == 1

        # _cleanup must return quickly despite the in-flight future
        start_time = time.monotonic()
        daemon._cleanup()
        elapsed = time.monotonic() - start_time

        assert elapsed < 2.0, f"_cleanup took {elapsed:.1f}s, expected < 2s"
        # PID file must still be removed
        assert not daemon._pid_file.exists()

    def test_cleanup_removes_pid_file(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        mock_dispatcher,
        tmp_project,
    ):
        """_cleanup removes the PID file even with wait=False."""
        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=mock_dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
        )

        daemon._write_pid()
        assert daemon._pid_file.exists()

        daemon._cleanup()

        assert not daemon._pid_file.exists()

    def test_signal_handler_sets_running_false(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        mock_dispatcher,
        tmp_project,
    ):
        """Signal handler sets _running=False so the loop exits."""
        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=mock_dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
        )

        daemon._running = True
        daemon._handle_signal(signal.SIGTERM, None)
        assert daemon._running is False

    def test_daemon_full_pipeline(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Full pipeline: 3-task DAG with parallel=2, mock dispatch, validation."""
        # Create a file so validation passes for t3
        (tmp_project / "result.txt").write_text("final output")

        dispatcher = MockDispatcher(delay=0.1)

        # t1 and t2 are independent, t3 depends on both
        _create_task(mutation_log, "t1", "Task 1", done_when="do thing 1")
        _create_task(mutation_log, "t2", "Task 2", done_when="do thing 2")
        rules = json.dumps([{"file_exists": "result.txt"}])
        _create_task(
            mutation_log, "t3", "Task 3", done_when=rules, depends_on=["t1", "t2"]
        )
        work_state.refresh()

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=tmp_project,
            parallel=2,
            poll_interval=0.1,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(1.5)
        daemon.stop()
        thread.join(timeout=3)

        work_state.refresh()
        assert work_state.get_task("t1")["status"] == "completed"
        assert work_state.get_task("t2")["status"] == "completed"
        assert work_state.get_task("t3")["status"] == "completed"

        # All 3 should have been dispatched
        assert len(dispatcher.dispatched) == 3

    def test_daemon_picks_up_new_tasks(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        mock_dispatcher,
        tmp_project,
    ):
        """Daemon picks up tasks added after it starts (no restart needed)."""
        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=mock_dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()

        # Add a task while daemon is running
        time.sleep(0.2)
        _create_task(mutation_log, "t1", "Late Task", done_when="do it")

        time.sleep(0.5)
        daemon.stop()
        thread.join(timeout=2)

        # Task should have been picked up and dispatched
        assert len(mock_dispatcher.dispatched) >= 1
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "completed"


# ===========================================================================
# External dispatch reconciliation tests
# ===========================================================================


def _start_task(mutation_log, task_id, attempt=1):
    """Helper to mark a task as started (simulating external dispatch)."""
    mutation_log.append(
        "task_started",
        {"attempt": attempt},
        reason="Externally dispatched",
        task_id=task_id,
    )


def _create_agent(
    mutation_log, agent_id, task_id, role="implementer", pid=None, worktree_path=None
):
    """Helper to create an agent record."""
    mutation_log.append(
        "agent_created",
        {
            "id": agent_id,
            "role": role,
            "task_id": task_id,
            "pid": pid,
            "worktree_path": worktree_path,
        },
        reason="Test setup",
    )


class TestExternalDispatchReconciliation:
    """Test that the daemon reconciles externally-dispatched tasks during its loop.

    Tasks dispatched via 'corc dispatch' are in 'running' state but have no
    executor handle. The daemon should detect these and reconcile them.

    Tests simulate external dispatch happening WHILE the daemon is running
    (by adding mutations after daemon start) to exercise the in-loop
    reconciliation path rather than the startup reconciliation.
    """

    def test_daemon_completes_externally_dispatched_task(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        mock_dispatcher,
        tmp_project,
    ):
        """Daemon picks up a task dispatched externally with successful output."""
        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=mock_dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
            pid_checker=lambda pid: False,  # Agent process is dead
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()

        # Simulate external dispatch while daemon is running
        time.sleep(0.3)
        _create_task(mutation_log, "ext-1", "External Task", done_when="do it")
        _start_task(mutation_log, "ext-1")
        session_logger.log_dispatch("ext-1", 1, "prompt", "system", ["Read"], 3.0)
        session_logger.log_output("ext-1", 1, "Task completed successfully!", 0, 5.0)

        time.sleep(0.8)
        daemon.stop()
        thread.join(timeout=3)

        work_state.refresh()
        task = work_state.get_task("ext-1")
        assert task["status"] == "completed"

        # Daemon should NOT have re-dispatched the task — it was reconciled
        ext_dispatches = [
            d for d in mock_dispatcher.dispatched if "External Task" in d[1]
        ]
        assert len(ext_dispatches) == 0

    def test_daemon_fails_externally_dispatched_task_no_output(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        mock_dispatcher,
        tmp_project,
    ):
        """Daemon marks externally-dispatched task as failed when agent died without output."""
        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=mock_dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
            pid_checker=lambda pid: False,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()

        # Simulate external dispatch while daemon is running — agent dies without output
        time.sleep(0.3)
        _create_task(
            mutation_log, "ext-2", "Dead External", done_when="do it", max_retries=0
        )
        _start_task(mutation_log, "ext-2")

        time.sleep(0.8)
        daemon.stop()
        thread.join(timeout=3)

        work_state.refresh()
        task = work_state.get_task("ext-2")
        # Reconciliation marks it as failed; with max_retries=0 the scheduler
        # won't retry it, so it stays failed.
        assert task["status"] == "failed"

        # Verify the reconciled flag is set in the mutation log
        entries = mutation_log.read_all()
        fail_entries = [e for e in entries if e["type"] == "task_failed"]
        assert len(fail_entries) >= 1
        assert any(e["data"].get("reconciled") for e in fail_entries)

    def test_daemon_leaves_alive_external_task_alone(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        mock_dispatcher,
        tmp_project,
    ):
        """Daemon leaves externally-dispatched task alone if agent PID is alive."""
        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=mock_dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
            pid_checker=lambda pid: True,  # Agent is alive
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()

        # Simulate external dispatch with a live agent
        time.sleep(0.3)
        _create_task(mutation_log, "ext-3", "Alive External", done_when="do it")
        _start_task(mutation_log, "ext-3")
        _create_agent(mutation_log, "agent-ext-3", "ext-3", pid=os.getpid())

        time.sleep(0.8)
        daemon.stop()
        thread.join(timeout=3)

        work_state.refresh()
        task = work_state.get_task("ext-3")
        assert task["status"] == "running"

    def test_daemon_reconciles_external_task_failed_exit(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        mock_dispatcher,
        tmp_project,
    ):
        """Daemon processes externally-dispatched task that exited non-zero."""
        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=mock_dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
            pid_checker=lambda pid: False,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()

        # Simulate external dispatch with error exit
        time.sleep(0.3)
        _create_task(
            mutation_log, "ext-4", "Failed External", done_when="do it", max_retries=0
        )
        _start_task(mutation_log, "ext-4")
        session_logger.log_dispatch("ext-4", 1, "prompt", "system", ["Read"], 3.0)
        session_logger.log_output("ext-4", 1, "Error: something broke", 1, 5.0)

        time.sleep(0.8)
        daemon.stop()
        thread.join(timeout=3)

        work_state.refresh()
        task = work_state.get_task("ext-4")
        # exit_code=1 with max_retries=0: processor marks as failed, then
        # attempt 1 > max_retries 0, so escalated
        assert task["status"] == "escalated"

    def test_daemon_retries_external_failed_task(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        mock_dispatcher,
        tmp_project,
    ):
        """Daemon retries externally-dispatched task after reconciling it as failed."""
        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=mock_dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
            pid_checker=lambda pid: False,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()

        # Simulate external dispatch — agent died without output
        time.sleep(0.3)
        _create_task(
            mutation_log, "ext-5", "Retry External", done_when="do it", max_retries=3
        )
        _start_task(mutation_log, "ext-5")

        time.sleep(1.5)
        daemon.stop()
        thread.join(timeout=3)

        work_state.refresh()
        task = work_state.get_task("ext-5")
        # Should have been retried and completed (mock dispatcher returns exit 0)
        assert task["status"] == "completed"
        # Daemon should have dispatched a retry
        assert len(mock_dispatcher.dispatched) >= 1

    def test_daemon_external_task_unblocks_downstream(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        mock_dispatcher,
        tmp_project,
    ):
        """Completing an externally-dispatched task unblocks downstream tasks."""
        # Create tasks before starting daemon so they're visible
        _create_task(mutation_log, "ext-6", "External Blocker", done_when="do it")
        _create_task(
            mutation_log,
            "t-downstream",
            "Downstream Task",
            done_when="do it too",
            depends_on=["ext-6"],
        )
        work_state.refresh()

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=mock_dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
            pid_checker=lambda pid: False,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()

        # Simulate external dispatch of ext-6 while daemon is running
        time.sleep(0.3)
        _start_task(mutation_log, "ext-6")
        session_logger.log_dispatch("ext-6", 1, "prompt", "system", ["Read"], 3.0)
        session_logger.log_output("ext-6", 1, "Done!", 0, 5.0)

        time.sleep(1.5)
        daemon.stop()
        thread.join(timeout=3)

        work_state.refresh()
        # External task completed via reconciliation
        assert work_state.get_task("ext-6")["status"] == "completed"
        # Downstream task was unblocked and completed
        assert work_state.get_task("t-downstream")["status"] == "completed"
        # Downstream task was dispatched by daemon
        assert len(mock_dispatcher.dispatched) >= 1

    def test_daemon_mixed_internal_and_external_tasks(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        mock_dispatcher,
        tmp_project,
    ):
        """Daemon handles both internally-dispatched and externally-dispatched tasks."""
        # Internal task: pending, will be dispatched by daemon
        _create_task(mutation_log, "internal-1", "Internal Task", done_when="do it")
        work_state.refresh()

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=mock_dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
            pid_checker=lambda pid: False,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()

        # Simulate external dispatch while daemon is running
        time.sleep(0.3)
        _create_task(mutation_log, "ext-7", "External Task", done_when="do it")
        _start_task(mutation_log, "ext-7")
        session_logger.log_dispatch("ext-7", 1, "prompt", "system", ["Read"], 3.0)
        session_logger.log_output("ext-7", 1, "External done!", 0, 5.0)

        time.sleep(1.0)
        daemon.stop()
        thread.join(timeout=3)

        work_state.refresh()
        # Both should be completed
        assert work_state.get_task("internal-1")["status"] == "completed"
        assert work_state.get_task("ext-7")["status"] == "completed"

    def test_daemon_once_mode_with_external_task(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        mock_dispatcher,
        tmp_project,
    ):
        """Daemon in --once mode exits after reconciling an external task."""
        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=mock_dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
            once=True,
            pid_checker=lambda pid: False,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()

        # Simulate external dispatch
        time.sleep(0.3)
        _create_task(mutation_log, "ext-8", "Once External", done_when="do it")
        _start_task(mutation_log, "ext-8")
        session_logger.log_dispatch("ext-8", 1, "prompt", "system", ["Read"], 3.0)
        session_logger.log_output("ext-8", 1, "Done!", 0, 5.0)

        thread.join(timeout=5)

        # Daemon should have stopped on its own after reconciling the task
        assert not thread.is_alive()

        work_state.refresh()
        assert work_state.get_task("ext-8")["status"] == "completed"


# ===========================================================================
# CLI integration tests
# ===========================================================================


class TestCLI:
    def test_start_stop_commands_exist(self):
        """The start and stop CLI commands are registered."""
        from click.testing import CliRunner
        from corc.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["start", "--help"])
        assert result.exit_code == 0
        assert "--parallel" in result.output
        assert "--task" in result.output
        assert "--once" in result.output

        result = runner.invoke(cli, ["stop", "--help"])
        assert result.exit_code == 0
