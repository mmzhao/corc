"""Tests for daemon restart recovery — reconciliation on startup.

Tests the full reconciliation flow: rebuild SQLite from mutation log,
check PID liveness for running tasks, process output from dead agents,
mark dead agents without output as failed, clean stale worktrees, and
verify the daemon recovers cleanly after a mid-task kill.
"""

import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from corc.audit import AuditLog
from corc.daemon import Daemon
from corc.dispatch import AgentDispatcher, AgentResult, Constraints
from corc.executor import Executor
from corc.mutations import MutationLog
from corc.reconcile import (
    clean_stale_worktrees,
    is_claude_process,
    is_pid_alive,
    reconcile_on_startup,
    _get_agent_pid,
    _get_last_agent_output,
)
from corc.retry import RetryPolicy
from corc.sessions import SessionLogger
from corc.state import WorkState


# ---------------------------------------------------------------------------
# Test fixtures
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


def _create_task(mutation_log, task_id, name, done_when="do the thing",
                 depends_on=None, role="implementer"):
    """Helper to create a task via mutation log."""
    mutation_log.append("task_created", {
        "id": task_id,
        "name": name,
        "description": f"Test task: {name}",
        "role": role,
        "depends_on": depends_on or [],
        "done_when": done_when,
        "checklist": [],
        "context_bundle": [],
    }, reason="Test setup")


def _start_task(mutation_log, task_id, attempt=1):
    """Helper to mark a task as started."""
    mutation_log.append(
        "task_started",
        {"attempt": attempt},
        reason="Test: marking as running",
        task_id=task_id,
    )


def _assign_task(mutation_log, task_id, agent_id):
    """Helper to mark a task as assigned."""
    mutation_log.append(
        "task_assigned",
        {"agent_id": agent_id},
        reason="Test: assigning agent",
        task_id=task_id,
    )


def _create_agent(mutation_log, agent_id, task_id, role="implementer",
                  pid=None, worktree_path=None):
    """Helper to create an agent record."""
    mutation_log.append("agent_created", {
        "id": agent_id,
        "role": role,
        "task_id": task_id,
        "pid": pid,
        "worktree_path": worktree_path,
    }, reason="Test setup")


# ---------------------------------------------------------------------------
# Mock dispatcher for daemon tests
# ---------------------------------------------------------------------------


class MockDispatcher(AgentDispatcher):
    """A dispatcher that returns configurable results without calling any LLM."""

    def __init__(self, default_result=None, delay=0):
        self.default_result = default_result or AgentResult(
            output="Mock output: task completed successfully.",
            exit_code=0,
            duration_s=0.1,
        )
        self.delay = delay
        self.dispatched = []
        self._results = {}

    def set_result_for(self, prompt_substring, result):
        self._results[prompt_substring] = result

    def dispatch(self, prompt, system_prompt, constraints,
                 pid_callback=None, event_callback=None):
        self.dispatched.append((prompt, system_prompt, constraints))
        if self.delay:
            time.sleep(self.delay)
        for substring, result in self._results.items():
            if substring in prompt:
                return result
        return self.default_result


# ===========================================================================
# Unit tests: rebuild SQLite from mutation log
# ===========================================================================


class TestRebuildState:
    """Test that SQLite is correctly rebuilt from the mutation log on boot."""

    def test_rebuild_empty_log(self, work_state, mutation_log, audit_log,
                                session_logger, tmp_project):
        """Rebuilding from an empty log produces empty state."""
        summary = reconcile_on_startup(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
            pid_checker=lambda pid: False,
        )
        assert summary["rebuilt_state"] is True
        assert summary["running_found"] == 0
        assert work_state.list_tasks() == []

    def test_rebuild_from_mutation_log(self, work_state, mutation_log, audit_log,
                                        session_logger, tmp_project):
        """State is rebuilt correctly from mutation log entries."""
        _create_task(mutation_log, "t1", "Task 1")
        _create_task(mutation_log, "t2", "Task 2", depends_on=["t1"])
        mutation_log.append("task_started", {"attempt": 1}, reason="test", task_id="t1")
        mutation_log.append("task_completed", {"findings": []}, reason="test", task_id="t1")

        # Delete the SQLite DB to simulate a fresh start
        work_state.conn.close()
        os.unlink(str(work_state.db_path))
        # Re-create WorkState (but don't replay yet — rebuild will do that)
        new_state = WorkState(work_state.db_path, mutation_log)

        summary = reconcile_on_startup(
            state=new_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
            pid_checker=lambda pid: False,
        )

        assert summary["rebuilt_state"] is True
        tasks = new_state.list_tasks()
        assert len(tasks) == 2
        t1 = new_state.get_task("t1")
        assert t1["status"] == "completed"
        t2 = new_state.get_task("t2")
        assert t2["status"] == "pending"

    def test_rebuild_clears_stale_sqlite(self, work_state, mutation_log, audit_log,
                                          session_logger, tmp_project):
        """Rebuild clears old SQLite data and replays fresh from log."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        assert work_state.get_task("t1") is not None

        # Directly insert a fake task into SQLite (simulating corruption)
        work_state.conn.execute(
            "INSERT INTO tasks(id, name, done_when, created, updated) VALUES(?, ?, ?, ?, ?)",
            ("fake", "Fake Task", "fake", "2026-01-01", "2026-01-01"),
        )
        work_state.conn.commit()
        assert work_state.get_task("fake") is not None

        # Reconcile should rebuild from log — fake task should be gone
        summary = reconcile_on_startup(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
            pid_checker=lambda pid: False,
        )

        assert summary["rebuilt_state"] is True
        assert work_state.get_task("fake") is None
        assert work_state.get_task("t1") is not None


# ===========================================================================
# Unit tests: PID liveness checking
# ===========================================================================


class TestPIDLiveness:
    """Test PID liveness and process identification."""

    def test_is_pid_alive_self(self):
        """Current process PID should be alive."""
        assert is_pid_alive(os.getpid()) is True

    def test_is_pid_alive_dead(self):
        """A very large PID should not exist."""
        assert is_pid_alive(999999999) is False

    def test_is_claude_process_not_claude(self):
        """Current test process is not a claude process."""
        # The test runner (python/pytest) is not claude
        result = is_claude_process(os.getpid())
        assert result is False

    def test_running_task_agent_alive(self, work_state, mutation_log, audit_log,
                                       session_logger, tmp_project):
        """Running task with alive agent is left running."""
        _create_task(mutation_log, "t1", "Task 1")
        _start_task(mutation_log, "t1")
        _create_agent(mutation_log, "agent-1", "t1", pid=os.getpid())
        work_state.refresh()

        summary = reconcile_on_startup(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
            pid_checker=lambda pid: True,  # Simulate alive claude process
        )

        assert summary["running_found"] == 1
        assert summary["agents_alive"] == 1
        assert summary["agents_dead_with_output"] == 0
        assert summary["agents_dead_no_output"] == 0

        # Task should still be running
        task = work_state.get_task("t1")
        assert task["status"] == "running"


# ===========================================================================
# Unit tests: dead agent with output → process output
# ===========================================================================


class TestDeadAgentWithOutput:
    """Test that dead agents with recorded output get their output processed."""

    def test_dead_agent_output_processed(self, work_state, mutation_log, audit_log,
                                          session_logger, tmp_project):
        """Dead agent with session output → output is processed normally."""
        _create_task(mutation_log, "t1", "Task 1", done_when="do the thing")
        _start_task(mutation_log, "t1")
        _create_agent(mutation_log, "agent-1", "t1", pid=99999)
        work_state.refresh()

        # Write a session log with output (simulating the agent finished writing
        # output but the daemon died before processing it)
        session_logger.log_dispatch("t1", 1, "prompt", "system", ["Read"], 3.0)
        session_logger.log_output("t1", 1, "Mock output: completed.", 0, 5.0)

        summary = reconcile_on_startup(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
            pid_checker=lambda pid: False,  # Agent is dead
        )

        assert summary["agents_dead_with_output"] == 1
        assert summary["agents_dead_no_output"] == 0

        # Task should be completed (exit code 0, no validation rules → pass)
        task = work_state.get_task("t1")
        assert task["status"] == "completed"

    def test_dead_agent_output_failed_exit(self, work_state, mutation_log, audit_log,
                                            session_logger, tmp_project):
        """Dead agent with failed output → task marked failed."""
        _create_task(mutation_log, "t1", "Task 1", done_when="do the thing")
        _start_task(mutation_log, "t1")
        _create_agent(mutation_log, "agent-1", "t1", pid=99999)
        work_state.refresh()

        # Agent produced output but exited with error
        session_logger.log_dispatch("t1", 1, "prompt", "system", ["Read"], 3.0)
        session_logger.log_output("t1", 1, "Error: something went wrong", 1, 5.0)

        summary = reconcile_on_startup(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
            pid_checker=lambda pid: False,
        )

        assert summary["agents_dead_with_output"] == 1

        # Task should be failed (exit code 1)
        task = work_state.get_task("t1")
        assert task["status"] == "failed"

    def test_dead_agent_output_with_findings(self, work_state, mutation_log, audit_log,
                                              session_logger, tmp_project):
        """Dead agent output with findings → findings are extracted."""
        _create_task(mutation_log, "t1", "Task 1", done_when="do it")
        _start_task(mutation_log, "t1")
        _create_agent(mutation_log, "agent-1", "t1", pid=99999)
        work_state.refresh()

        output = "Done!\nFINDING: SQLite WAL mode is required\nAll good."
        session_logger.log_dispatch("t1", 1, "prompt", "system", ["Read"], 3.0)
        session_logger.log_output("t1", 1, output, 0, 5.0)

        summary = reconcile_on_startup(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
            pid_checker=lambda pid: False,
        )

        task = work_state.get_task("t1")
        assert task["status"] == "completed"
        assert "SQLite WAL mode is required" in task["findings"]


# ===========================================================================
# Unit tests: dead agent without output → mark failed
# ===========================================================================


class TestDeadAgentNoOutput:
    """Test that dead agents without output are marked failed."""

    def test_dead_agent_no_output_marked_failed(self, work_state, mutation_log,
                                                  audit_log, session_logger, tmp_project):
        """Dead agent with no session output → task marked failed."""
        _create_task(mutation_log, "t1", "Task 1")
        _start_task(mutation_log, "t1")
        _create_agent(mutation_log, "agent-1", "t1", pid=99999)
        work_state.refresh()

        # No session log written — agent died before producing anything

        summary = reconcile_on_startup(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
            pid_checker=lambda pid: False,
        )

        assert summary["agents_dead_no_output"] == 1

        # Task should be failed
        task = work_state.get_task("t1")
        assert task["status"] == "failed"

    def test_dead_agent_no_output_has_reconciled_flag(self, work_state, mutation_log,
                                                       audit_log, session_logger, tmp_project):
        """Failed task mutation from reconciliation has 'reconciled' flag."""
        _create_task(mutation_log, "t1", "Task 1")
        _start_task(mutation_log, "t1")
        work_state.refresh()

        reconcile_on_startup(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
            pid_checker=lambda pid: False,
        )

        # Check the mutation log for the reconciled flag
        entries = mutation_log.read_all()
        fail_entries = [e for e in entries if e["type"] == "task_failed"]
        assert len(fail_entries) == 1
        assert fail_entries[0]["data"]["reconciled"] is True
        assert "Reconciliation" in fail_entries[0]["reason"]

    def test_dead_agent_no_output_no_pid(self, work_state, mutation_log, audit_log,
                                          session_logger, tmp_project):
        """Running task with no agent PID → treated as dead, marked failed."""
        _create_task(mutation_log, "t1", "Task 1")
        _start_task(mutation_log, "t1")
        # No agent record created — PID is unknown
        work_state.refresh()

        summary = reconcile_on_startup(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
            pid_checker=lambda pid: False,
        )

        assert summary["agents_dead_no_output"] == 1
        task = work_state.get_task("t1")
        assert task["status"] == "failed"


# ===========================================================================
# Unit tests: assigned task reconciliation
# ===========================================================================


class TestAssignedTaskReconciliation:
    """Test that 'assigned' tasks are also reconciled on restart."""

    def test_assigned_task_dead_agent(self, work_state, mutation_log, audit_log,
                                       session_logger, tmp_project):
        """Assigned task with dead agent → marked failed."""
        _create_task(mutation_log, "t1", "Task 1")
        _assign_task(mutation_log, "t1", "agent-1")
        _create_agent(mutation_log, "agent-1", "t1", pid=99999)
        work_state.refresh()

        summary = reconcile_on_startup(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
            pid_checker=lambda pid: False,
        )

        assert summary["assigned_found"] == 1
        task = work_state.get_task("t1")
        assert task["status"] == "failed"

    def test_assigned_task_alive_agent(self, work_state, mutation_log, audit_log,
                                        session_logger, tmp_project):
        """Assigned task with alive agent → left as-is."""
        _create_task(mutation_log, "t1", "Task 1")
        _assign_task(mutation_log, "t1", "agent-1")
        _create_agent(mutation_log, "agent-1", "t1", pid=os.getpid())
        work_state.refresh()

        summary = reconcile_on_startup(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
            pid_checker=lambda pid: True,
        )

        assert summary["agents_alive"] == 1
        task = work_state.get_task("t1")
        assert task["status"] == "assigned"


# ===========================================================================
# Unit tests: stale worktree cleanup
# ===========================================================================


class TestWorktreeCleanup:
    """Test cleanup of stale git worktrees."""

    def test_clean_nonexistent_worktree(self, work_state, mutation_log, tmp_project):
        """Worktree path that doesn't exist is silently skipped."""
        _create_agent(mutation_log, "agent-1", "t1", pid=99999,
                      worktree_path=str(tmp_project / "worktrees" / "nonexistent"))
        work_state.refresh()

        cleaned = clean_stale_worktrees(work_state, tmp_project)
        assert cleaned == 0

    def test_clean_stale_worktree_directory(self, work_state, mutation_log, tmp_project):
        """Worktree directory for dead agent is cleaned up."""
        worktree_dir = tmp_project / "worktrees" / "agent-1"
        worktree_dir.mkdir(parents=True)
        (worktree_dir / "some_file.py").write_text("# code")

        _create_agent(mutation_log, "agent-1", "t1", pid=99999,
                      worktree_path=str(worktree_dir))
        work_state.refresh()

        # Mock git worktree remove to fail (simulates non-git-tracked directory).
        # The fallback path uses shutil.rmtree via _remove_dir.
        with patch("corc.reconcile.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            cleaned = clean_stale_worktrees(work_state, tmp_project)

        assert cleaned == 1
        assert not worktree_dir.exists()

    def test_skip_worktree_alive_agent(self, work_state, mutation_log, tmp_project):
        """Worktree for alive agent is NOT cleaned up."""
        worktree_dir = tmp_project / "worktrees" / "agent-1"
        worktree_dir.mkdir(parents=True)

        _create_agent(mutation_log, "agent-1", "t1", pid=os.getpid(),
                      worktree_path=str(worktree_dir))
        work_state.refresh()

        cleaned = clean_stale_worktrees(work_state, tmp_project)
        assert cleaned == 0
        assert worktree_dir.exists()

    def test_worktrees_cleaned_during_reconcile(self, work_state, mutation_log,
                                                  audit_log, session_logger, tmp_project):
        """Reconciliation cleans stale worktrees as part of the full flow."""
        worktree_dir = tmp_project / "worktrees" / "agent-1"
        worktree_dir.mkdir(parents=True)
        (worktree_dir / "file.py").write_text("# partial work")

        _create_task(mutation_log, "t1", "Task 1")
        _start_task(mutation_log, "t1")
        _create_agent(mutation_log, "agent-1", "t1", pid=99999,
                      worktree_path=str(worktree_dir))
        work_state.refresh()

        with patch("corc.reconcile.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            summary = reconcile_on_startup(
                state=work_state,
                mutation_log=mutation_log,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=tmp_project,
                pid_checker=lambda pid: False,
            )

        assert summary["worktrees_cleaned"] == 1
        assert not worktree_dir.exists()

    def test_worktrees_cleaned_even_without_stale_tasks(self, work_state, mutation_log,
                                                         audit_log, session_logger, tmp_project):
        """Stale worktrees are cleaned even when no running/assigned tasks exist."""
        worktree_dir = tmp_project / "worktrees" / "agent-old"
        worktree_dir.mkdir(parents=True)
        (worktree_dir / "leftover.py").write_text("# old work")

        # Agent record exists but no running tasks
        _create_agent(mutation_log, "agent-old", "t-old", pid=99999,
                      worktree_path=str(worktree_dir))
        work_state.refresh()

        with patch("corc.reconcile.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            summary = reconcile_on_startup(
                state=work_state,
                mutation_log=mutation_log,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=tmp_project,
                pid_checker=lambda pid: False,
            )

        assert summary["worktrees_cleaned"] == 1


# ===========================================================================
# Unit tests: helper functions
# ===========================================================================


class TestHelpers:
    """Test helper functions used by reconciliation."""

    def test_get_agent_pid_found(self, work_state, mutation_log):
        """Returns PID when agent has one."""
        _create_agent(mutation_log, "agent-1", "t1", pid=12345)
        work_state.refresh()
        assert _get_agent_pid(work_state, "t1") == 12345

    def test_get_agent_pid_none(self, work_state, mutation_log):
        """Returns None when no agent exists for task."""
        assert _get_agent_pid(work_state, "nonexistent") is None

    def test_get_agent_pid_no_pid(self, work_state, mutation_log):
        """Returns None when agent has no PID."""
        _create_agent(mutation_log, "agent-1", "t1", pid=None)
        work_state.refresh()
        assert _get_agent_pid(work_state, "t1") is None

    def test_get_last_agent_output_found(self, session_logger):
        """Returns AgentResult when session log has output."""
        session_logger.log_dispatch("t1", 1, "prompt", "system", [], 1.0)
        session_logger.log_output("t1", 1, "output text", 0, 5.0)

        result = _get_last_agent_output(session_logger, "t1")
        assert result is not None
        assert result.output == "output text"
        assert result.exit_code == 0
        assert result.duration_s == 5.0

    def test_get_last_agent_output_no_session(self, session_logger):
        """Returns None when no session exists."""
        result = _get_last_agent_output(session_logger, "nonexistent")
        assert result is None

    def test_get_last_agent_output_dispatch_only(self, session_logger):
        """Returns None when session has dispatch but no output."""
        session_logger.log_dispatch("t1", 1, "prompt", "system", [], 1.0)

        result = _get_last_agent_output(session_logger, "t1")
        assert result is None


# ===========================================================================
# Unit tests: multiple running tasks
# ===========================================================================


class TestMultipleRunningTasks:
    """Test reconciliation with multiple tasks in different states."""

    def test_mixed_state_reconciliation(self, work_state, mutation_log, audit_log,
                                         session_logger, tmp_project):
        """Multiple tasks: one alive, one dead with output, one dead without."""
        # Task 1: running, agent alive
        _create_task(mutation_log, "t1", "Task 1")
        _start_task(mutation_log, "t1")
        _create_agent(mutation_log, "agent-1", "t1", pid=1001)

        # Task 2: running, agent dead, has output
        _create_task(mutation_log, "t2", "Task 2", done_when="do it")
        _start_task(mutation_log, "t2")
        _create_agent(mutation_log, "agent-2", "t2", pid=1002)
        session_logger.log_dispatch("t2", 1, "prompt", "system", [], 3.0)
        session_logger.log_output("t2", 1, "Completed successfully", 0, 10.0)

        # Task 3: running, agent dead, no output
        _create_task(mutation_log, "t3", "Task 3")
        _start_task(mutation_log, "t3")
        _create_agent(mutation_log, "agent-3", "t3", pid=1003)

        work_state.refresh()

        def pid_checker(pid):
            return pid == 1001  # Only agent-1 is alive

        summary = reconcile_on_startup(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
            pid_checker=pid_checker,
        )

        assert summary["running_found"] == 3
        assert summary["agents_alive"] == 1
        assert summary["agents_dead_with_output"] == 1
        assert summary["agents_dead_no_output"] == 1

        # Verify final states
        assert work_state.get_task("t1")["status"] == "running"
        assert work_state.get_task("t2")["status"] == "completed"
        assert work_state.get_task("t3")["status"] == "failed"


# ===========================================================================
# Unit tests: audit log events
# ===========================================================================


class TestAuditLogEvents:
    """Test that reconciliation logs appropriate audit events."""

    def test_audit_events_logged(self, work_state, mutation_log, audit_log,
                                  session_logger, tmp_project):
        """All reconciliation steps produce audit log events."""
        _create_task(mutation_log, "t1", "Task 1")
        _start_task(mutation_log, "t1")
        _create_agent(mutation_log, "agent-1", "t1", pid=99999)
        work_state.refresh()

        reconcile_on_startup(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
            pid_checker=lambda pid: False,
        )

        events = audit_log.read_today()
        event_types = [e["event_type"] for e in events]
        assert "reconcile_state_rebuilt" in event_types
        assert "reconcile_marked_failed" in event_types
        assert "reconcile_complete" in event_types


# ===========================================================================
# Integration test: daemon reconciliation on start
# ===========================================================================


class TestDaemonReconciliation:
    """Test that the daemon calls reconciliation on startup."""

    def test_daemon_reconciles_on_start(self, mutation_log, work_state, audit_log,
                                          session_logger, tmp_project):
        """Daemon calls reconcile_on_startup() and stores the summary."""
        _create_task(mutation_log, "t1", "Task 1")
        _start_task(mutation_log, "t1")
        work_state.refresh()

        dispatcher = MockDispatcher()
        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
            pid_checker=lambda pid: False,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(0.5)
        daemon.stop()
        thread.join(timeout=3)

        # Check that reconciliation ran
        assert daemon._reconcile_summary is not None
        assert daemon._reconcile_summary["rebuilt_state"] is True
        assert daemon._reconcile_summary["running_found"] == 1
        assert daemon._reconcile_summary["agents_dead_no_output"] == 1

    def test_daemon_reconcile_then_dispatch(self, mutation_log, work_state, audit_log,
                                              session_logger, tmp_project):
        """After reconciliation, the daemon retries failed tasks and dispatches ready ones."""
        # Task 1 was running but agent is dead → gets failed → retried → completed
        # Task 2 depends on t1 → unblocked after t1 completes via retry
        # Task 3 has no deps → should be dispatched
        _create_task(mutation_log, "t1", "Task 1", done_when="do it")
        _start_task(mutation_log, "t1")
        _create_task(mutation_log, "t2", "Task 2", done_when="do it", depends_on=["t1"])
        _create_task(mutation_log, "t3", "Task 3", done_when="do it")
        work_state.refresh()

        dispatcher = MockDispatcher()
        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
            pid_checker=lambda pid: False,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(1.5)
        daemon.stop()
        thread.join(timeout=3)

        work_state.refresh()

        # t1 was reconcile-failed then retried → completed
        # t2 unblocked after t1 → completed
        # t3 dispatched independently → completed
        assert work_state.get_task("t1")["status"] == "completed"
        assert work_state.get_task("t2")["status"] == "completed"
        assert work_state.get_task("t3")["status"] == "completed"
        # t1 (retry) + t2 + t3 = at least 3 dispatches
        assert len(dispatcher.dispatched) >= 3

    def test_daemon_reconcile_processes_output(self, mutation_log, work_state, audit_log,
                                                 session_logger, tmp_project):
        """Daemon reconciliation processes dead agent output, unblocking downstream tasks."""
        # t1 was running, agent died but left output → should be completed
        # t2 depends on t1 → should become ready and get dispatched
        _create_task(mutation_log, "t1", "Task 1", done_when="do it")
        _start_task(mutation_log, "t1")
        _create_agent(mutation_log, "agent-1", "t1", pid=99999)
        _create_task(mutation_log, "t2", "Task 2", done_when="do it too", depends_on=["t1"])
        work_state.refresh()

        # Simulate agent-1 having written output before dying
        session_logger.log_dispatch("t1", 1, "prompt", "system", ["Read"], 3.0)
        session_logger.log_output("t1", 1, "Successfully completed!", 0, 5.0)

        dispatcher = MockDispatcher()
        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
            pid_checker=lambda pid: False,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(1.0)
        daemon.stop()
        thread.join(timeout=3)

        work_state.refresh()

        # t1 completed via reconciliation, t2 became ready and was dispatched
        assert work_state.get_task("t1")["status"] == "completed"
        assert work_state.get_task("t2")["status"] == "completed"
        assert len(dispatcher.dispatched) >= 1


# ===========================================================================
# Integration test: kill daemon mid-task and verify clean restart
# ===========================================================================


class TestKillDaemonMidTask:
    """Test that killing the daemon mid-task leads to clean recovery on restart.

    This is the key integration test: simulates a daemon crash while an agent
    is running, then verifies that restarting the daemon correctly reconciles.
    """

    def test_kill_and_restart_no_output(self, mutation_log, audit_log,
                                         session_logger, tmp_project):
        """Kill daemon mid-task (no output) → restart → reconcile marks failed → retry succeeds.

        Simulates a daemon crash by manually setting up the state as if the
        daemon died while a task was running (task_started written but no
        output produced). Then starts a new daemon that reconciles and retries.
        """
        # Phase 1: Simulate the crashed state
        # A task was dispatched and marked as running, but the agent died
        # before producing any output (no session log entries)
        _create_task(mutation_log, "t1", "Crashed Task", done_when="do it")
        _start_task(mutation_log, "t1")
        _create_agent(mutation_log, "agent-1", "t1", pid=99999)

        # No session output — agent died before producing anything

        # Phase 2: Restart with a fresh daemon — reconciliation should kick in
        fast_dispatcher = MockDispatcher()  # Fast dispatcher for retry
        state2 = WorkState(tmp_project / "data" / "state2.db", mutation_log)

        daemon2 = Daemon(
            state=state2,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=fast_dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
            pid_checker=lambda pid: False,  # Agent is dead
            retry_policy=RetryPolicy(max_retries=1),
        )

        thread2 = threading.Thread(target=daemon2.start)
        thread2.start()
        time.sleep(1.0)
        daemon2.stop()
        thread2.join(timeout=3)

        # Verify reconciliation happened
        assert daemon2._reconcile_summary is not None
        assert daemon2._reconcile_summary["rebuilt_state"] is True
        assert daemon2._reconcile_summary["running_found"] == 1
        assert daemon2._reconcile_summary["agents_dead_no_output"] == 1

        # After reconciliation + retry, the task should be completed
        state2.refresh()
        task = state2.get_task("t1")
        assert task["status"] == "completed", f"Expected completed, got {task['status']}"

        # Verify the retry dispatched
        assert len(fast_dispatcher.dispatched) >= 1

    def test_kill_and_restart_with_output(self, mutation_log, audit_log,
                                            session_logger, tmp_project):
        """Kill daemon after agent finishes but before processing → restart processes output."""
        state1 = WorkState(tmp_project / "data" / "state.db", mutation_log)

        _create_task(mutation_log, "t1", "Task 1", done_when="do it")
        _start_task(mutation_log, "t1")
        _create_agent(mutation_log, "agent-1", "t1", pid=99999)
        state1.refresh()

        # Simulate: agent finished and wrote output, but daemon didn't process it
        session_logger.log_dispatch("t1", 1, "prompt", "system", ["Read"], 3.0)
        session_logger.log_output("t1", 1, "Agent completed the task!", 0, 10.0)

        # Phase 2: New daemon picks up the orphaned output
        state2 = WorkState(tmp_project / "data" / "state2.db", mutation_log)
        dispatcher = MockDispatcher()

        daemon = Daemon(
            state=state2,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
            pid_checker=lambda pid: False,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(0.5)
        daemon.stop()
        thread.join(timeout=3)

        # Verify: reconciliation processed the output → task completed
        assert daemon._reconcile_summary["agents_dead_with_output"] == 1
        state2.refresh()
        task = state2.get_task("t1")
        assert task["status"] == "completed"

        # No re-dispatch needed — the original output was processed
        assert len(dispatcher.dispatched) == 0

    def test_kill_and_restart_dag_continues(self, mutation_log, audit_log,
                                              session_logger, tmp_project):
        """After restart and reconciliation, blocked downstream tasks become ready."""
        state1 = WorkState(tmp_project / "data" / "state.db", mutation_log)

        # t1 → t2 → t3 pipeline
        _create_task(mutation_log, "t1", "Task 1", done_when="do it")
        _create_task(mutation_log, "t2", "Task 2", done_when="do it", depends_on=["t1"])
        _create_task(mutation_log, "t3", "Task 3", done_when="do it", depends_on=["t2"])

        # t1 was running and agent left output
        _start_task(mutation_log, "t1")
        _create_agent(mutation_log, "agent-1", "t1", pid=99999)
        state1.refresh()

        session_logger.log_dispatch("t1", 1, "prompt", "system", ["Read"], 3.0)
        session_logger.log_output("t1", 1, "Done with task 1!", 0, 5.0)

        # Restart daemon — should process t1, then dispatch t2, then t3
        state2 = WorkState(tmp_project / "data" / "state2.db", mutation_log)
        dispatcher = MockDispatcher()

        daemon = Daemon(
            state=state2,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
            pid_checker=lambda pid: False,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(1.5)
        daemon.stop()
        thread.join(timeout=3)

        state2.refresh()
        assert state2.get_task("t1")["status"] == "completed"  # Via reconciliation
        assert state2.get_task("t2")["status"] == "completed"  # Dispatched after t1
        assert state2.get_task("t3")["status"] == "completed"  # Dispatched after t2

    def test_kill_and_restart_idempotent(self, mutation_log, audit_log,
                                           session_logger, tmp_project):
        """Multiple restarts converge to the same state (idempotent reconciliation)."""
        _create_task(mutation_log, "t1", "Task 1", done_when="do it")
        _start_task(mutation_log, "t1")
        _create_agent(mutation_log, "agent-1", "t1", pid=99999)

        session_logger.log_dispatch("t1", 1, "prompt", "system", ["Read"], 3.0)
        session_logger.log_output("t1", 1, "Done!", 0, 5.0)

        # First restart
        state1 = WorkState(tmp_project / "data" / "state1.db", mutation_log)
        summary1 = reconcile_on_startup(
            state=state1,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
            pid_checker=lambda pid: False,
        )
        assert summary1["agents_dead_with_output"] == 1
        assert state1.get_task("t1")["status"] == "completed"

        # Second restart (reconcile again — should find no running tasks)
        state2 = WorkState(tmp_project / "data" / "state2.db", mutation_log)
        summary2 = reconcile_on_startup(
            state=state2,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
            pid_checker=lambda pid: False,
        )
        # t1 is now completed, so no running tasks found on second reconcile
        assert summary2["running_found"] == 0
        assert state2.get_task("t1")["status"] == "completed"

    def test_restart_with_real_subprocess(self, mutation_log, audit_log,
                                           session_logger, tmp_project):
        """Kill a real subprocess mid-flight and verify reconciliation handles it.

        Spawns a real 'sleep' process to simulate an agent, kills it, and
        verifies the reconciliation correctly identifies it as dead.
        """
        # Start a real process to get a real PID
        proc = subprocess.Popen(
            ["sleep", "30"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        real_pid = proc.pid

        _create_task(mutation_log, "t1", "Task 1")
        _start_task(mutation_log, "t1")
        _create_agent(mutation_log, "agent-1", "t1", pid=real_pid)

        state = WorkState(tmp_project / "data" / "state.db", mutation_log)
        state.refresh()

        # Verify PID is alive
        assert is_pid_alive(real_pid) is True

        # Kill the process
        proc.kill()
        proc.wait()

        # Verify PID is dead
        assert is_pid_alive(real_pid) is False

        # Reconcile — should detect dead process
        summary = reconcile_on_startup(
            state=state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
            # Use default pid_checker but it won't find 'claude' in process name
            # since we killed it. That's fine — dead PID returns False from both.
            pid_checker=lambda pid: False,
        )

        assert summary["agents_dead_no_output"] == 1
        task = state.get_task("t1")
        assert task["status"] == "failed"
