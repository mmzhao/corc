"""Tests for daemon agent recovery — re-attaching monitoring on restart.

Tests that when the daemon restarts, it:
1. Stores PID and worktree_path when agents are dispatched
2. Scans for running claude processes matching task IDs
3. Re-attaches monitoring for live agents
4. Polls re-attached agents for completion
5. Does not duplicate-dispatch tasks with live agents
6. Handles agents that finished while daemon was down
"""

import os
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
    _get_agent_pid,
    _get_last_agent_output,
    is_pid_alive,
    reconcile_on_startup,
    scan_claude_processes,
)
from corc.retry import RetryPolicy
from corc.sessions import SessionLogger
from corc.state import WorkState


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_task(
    mutation_log, task_id, name, done_when="do the thing", depends_on=None
):
    """Helper to create a task via mutation log."""
    mutation_log.append(
        "task_created",
        {
            "id": task_id,
            "name": name,
            "description": f"Test: {name}",
            "role": "implementer",
            "depends_on": depends_on or [],
            "done_when": done_when,
            "checklist": [],
            "context_bundle": [],
        },
        reason="Test setup",
    )


def _start_task(mutation_log, task_id, attempt=1):
    """Helper to mark a task as started."""
    mutation_log.append(
        "task_started",
        {"attempt": attempt},
        reason="Test: marking as running",
        task_id=task_id,
    )


def _create_agent(mutation_log, agent_id, task_id, pid=None, worktree_path=None):
    """Helper to create an agent record."""
    mutation_log.append(
        "agent_created",
        {
            "id": agent_id,
            "role": "implementer",
            "task_id": task_id,
            "pid": pid,
            "worktree_path": worktree_path,
        },
        reason="Test setup",
    )


# ---------------------------------------------------------------------------
# Mock dispatcher that supports pid_callback
# ---------------------------------------------------------------------------


class MockDispatcher(AgentDispatcher):
    """Dispatcher that returns configurable results and optionally calls pid_callback."""

    def __init__(self, default_result=None, delay=0, mock_pid=None):
        self.default_result = default_result or AgentResult(
            output="Mock output: task completed successfully.",
            exit_code=0,
            duration_s=0.1,
        )
        self.delay = delay
        self.dispatched = []
        self._results = {}
        self.mock_pid = mock_pid

    def set_result_for(self, prompt_substring, result):
        self._results[prompt_substring] = result

    def dispatch(
        self,
        prompt,
        system_prompt,
        constraints,
        pid_callback=None,
        event_callback=None,
        cwd=None,
    ):
        self.dispatched.append((prompt, system_prompt, constraints))
        if pid_callback and self.mock_pid is not None:
            pid_callback(self.mock_pid)
        if self.delay:
            time.sleep(self.delay)
        for substring, result in self._results.items():
            if substring in prompt:
                return result
        return self.default_result


# ===========================================================================
# Test: Mutation log records PID when agent is dispatched
# ===========================================================================


class TestPidCapture:
    """Test that executor captures agent PID via pid_callback during dispatch."""

    def test_pid_stored_in_mutation_log(
        self, work_state, mutation_log, audit_log, session_logger, tmp_project
    ):
        """When agent dispatched, PID is recorded via agent_updated mutation."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()

        dispatcher = MockDispatcher(mock_pid=12345)
        executor = Executor(
            dispatcher=dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        task = work_state.get_task("t1")
        executor.dispatch(task)
        time.sleep(0.5)
        executor.poll_completed()

        # Check mutation log for agent_updated with PID
        entries = mutation_log.read_all()
        pid_entries = [
            e
            for e in entries
            if e["type"] == "agent_updated" and e.get("data", {}).get("pid") == 12345
        ]
        assert len(pid_entries) == 1

        # Verify PID is materialized in state
        work_state.refresh()
        pid = _get_agent_pid(work_state, "t1")
        assert pid == 12345

        executor.shutdown()

    def test_prompt_includes_task_id(
        self, work_state, mutation_log, audit_log, session_logger, tmp_project
    ):
        """Dispatched prompt contains the task ID for process scanning."""
        _create_task(mutation_log, "my-task-123", "Task 1")
        work_state.refresh()

        dispatcher = MockDispatcher()
        executor = Executor(
            dispatcher=dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        task = work_state.get_task("my-task-123")
        executor.dispatch(task)
        time.sleep(0.5)
        executor.poll_completed()

        assert len(dispatcher.dispatched) == 1
        prompt = dispatcher.dispatched[0][0]
        assert "my-task-123" in prompt

        executor.shutdown()


# ===========================================================================
# Test: scan_claude_processes finds running claude processes
# ===========================================================================


class TestScanClaudeProcesses:
    """Test scanning ps output for claude -p processes matching task IDs."""

    def test_scan_finds_matching_process(self):
        """Finds claude -p process whose command line contains a known task_id."""
        mock_ps_output = (
            "  PID ARGS\n"
            "12345 /usr/bin/claude -p Task ID: task-abc\\n\\nComplete...\n"
            "67890 /usr/bin/python test.py\n"
        )
        with patch("corc.reconcile.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=mock_ps_output, stderr=""
            )
            result = scan_claude_processes({"task-abc", "task-def"})

        assert result == {"task-abc": 12345}

    def test_scan_no_matches(self):
        """Returns empty dict when no claude processes match."""
        mock_ps_output = "  PID ARGS\n12345 /usr/bin/python test.py\n"
        with patch("corc.reconcile.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=mock_ps_output, stderr=""
            )
            result = scan_claude_processes({"task-abc"})

        assert result == {}

    def test_scan_empty_task_ids(self):
        """Returns empty dict immediately for empty task_ids set."""
        result = scan_claude_processes(set())
        assert result == {}

    def test_scan_handles_ps_failure(self):
        """Handles ps command failure gracefully."""
        with patch("corc.reconcile.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("ps not found")
            result = scan_claude_processes({"task-abc"})
        assert result == {}

    def test_scan_handles_timeout(self):
        """Handles ps timeout gracefully."""
        with patch("corc.reconcile.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="ps", timeout=10)
            result = scan_claude_processes({"task-abc"})
        assert result == {}

    def test_scan_multiple_matches(self):
        """Finds multiple claude processes for different tasks."""
        mock_ps_output = (
            "  PID ARGS\n"
            "111 /usr/bin/claude -p Task ID: task-one\\nDo it\n"
            "222 /usr/bin/claude -p Task ID: task-two\\nDo more\n"
            "333 /usr/bin/python test.py\n"
        )
        with patch("corc.reconcile.subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=mock_ps_output, stderr=""
            )
            result = scan_claude_processes({"task-one", "task-two"})

        assert result == {"task-one": 111, "task-two": 222}


# ===========================================================================
# Test: Reconciliation returns alive agent info for re-attachment
# ===========================================================================


class TestReconcileAliveAgentInfo:
    """Test that reconcile_on_startup returns info needed to re-attach monitoring."""

    def test_alive_agent_info_returned(
        self, work_state, mutation_log, audit_log, session_logger, tmp_project
    ):
        """Alive agents include task, pid, attempt, worktree_path, agent_id."""
        _create_task(mutation_log, "t1", "Task 1")
        _start_task(mutation_log, "t1")
        _create_agent(
            mutation_log,
            "agent-1",
            "t1",
            pid=os.getpid(),
            worktree_path="/tmp/wt/t1",
        )
        work_state.refresh()

        # Write a session to set attempt number
        session_logger.log_dispatch("t1", 2, "prompt", "system", [], 1.0)

        summary = reconcile_on_startup(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
            pid_checker=lambda pid: True,
        )

        assert summary["agents_alive"] == 1
        assert len(summary["alive_agents"]) == 1

        info = summary["alive_agents"][0]
        assert info["task"]["id"] == "t1"
        assert info["pid"] == os.getpid()
        assert info["attempt"] == 2
        assert info["worktree_path"] == "/tmp/wt/t1"
        assert info["agent_id"] == "agent-1"

    def test_dead_agent_not_in_alive_list(
        self, work_state, mutation_log, audit_log, session_logger, tmp_project
    ):
        """Dead agents are NOT in the alive_agents list."""
        _create_task(mutation_log, "t1", "Task 1")
        _start_task(mutation_log, "t1")
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

        assert summary["agents_alive"] == 0
        assert len(summary["alive_agents"]) == 0

    def test_process_scan_fallback(
        self, work_state, mutation_log, audit_log, session_logger, tmp_project
    ):
        """When PID is not in agent record, process scan is used as fallback."""
        _create_task(mutation_log, "t1", "Task 1")
        _start_task(mutation_log, "t1")
        # Agent has NO PID recorded
        _create_agent(mutation_log, "agent-1", "t1", pid=None)
        work_state.refresh()

        # Simulate scan finding a process
        with patch("corc.reconcile.scan_claude_processes") as mock_scan:
            mock_scan.return_value = {"t1": 54321}
            summary = reconcile_on_startup(
                state=work_state,
                mutation_log=mutation_log,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=tmp_project,
                pid_checker=lambda pid: True,
            )

        assert summary["agents_alive"] == 1
        assert len(summary["alive_agents"]) == 1
        assert summary["alive_agents"][0]["pid"] == 54321


# ===========================================================================
# Test: Re-attach monitoring via executor
# ===========================================================================


class TestReattachMonitoring:
    """Test re-attaching monitoring for live agents via executor.reattach()."""

    def test_reattach_tracks_in_flight(
        self, work_state, mutation_log, audit_log, session_logger, tmp_project
    ):
        """Reattached task appears in executor.in_flight_task_ids."""
        _create_task(mutation_log, "t1", "Task 1")
        _start_task(mutation_log, "t1")
        work_state.refresh()

        # Start a real process to get a real PID
        proc = subprocess.Popen(["sleep", "30"], stdout=subprocess.PIPE)

        try:
            executor = Executor(
                dispatcher=MockDispatcher(),
                mutation_log=mutation_log,
                state=work_state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=tmp_project,
            )

            task = work_state.get_task("t1")
            executor.reattach(
                task=task,
                pid=proc.pid,
                attempt=1,
                worktree_path=None,
                agent_id="agent-1",
            )

            assert "t1" in executor.in_flight_task_ids
            assert executor.in_flight_count == 1

            executor.shutdown(wait=False)
        finally:
            proc.kill()
            proc.wait()

    def test_reattach_captures_output_on_completion(
        self, work_state, mutation_log, audit_log, session_logger, tmp_project
    ):
        """When reattached agent exits, session output is captured via poll_completed."""
        _create_task(mutation_log, "t1", "Task 1", done_when="do it")
        _start_task(mutation_log, "t1")
        work_state.refresh()

        # Write session output (captured before daemon died)
        session_logger.log_dispatch("t1", 1, "prompt", "system", ["Read"], 3.0)
        session_logger.log_output("t1", 1, "Agent completed successfully!", 0, 10.0)

        # Start a very short-lived process
        proc = subprocess.Popen(["sleep", "0.1"], stdout=subprocess.PIPE)
        real_pid = proc.pid

        executor = Executor(
            dispatcher=MockDispatcher(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        task = work_state.get_task("t1")
        executor.reattach(
            task=task,
            pid=real_pid,
            attempt=1,
            worktree_path=None,
            agent_id="agent-1",
        )

        # Wait for the process to finish and _wait_for_pid to detect it
        proc.wait()
        time.sleep(3)

        completed = executor.poll_completed()
        assert len(completed) == 1
        assert completed[0].task["id"] == "t1"
        assert completed[0].result.exit_code == 0
        assert "completed successfully" in completed[0].result.output

        # No longer in flight
        assert "t1" not in executor.in_flight_task_ids

        executor.shutdown()

    def test_reattach_no_output_returns_failure(
        self, work_state, mutation_log, audit_log, session_logger, tmp_project
    ):
        """When reattached agent exits with no session output, result is failure."""
        _create_task(mutation_log, "t1", "Task 1")
        _start_task(mutation_log, "t1")
        work_state.refresh()

        # No session output written — simulates daemon crash before any events

        proc = subprocess.Popen(["sleep", "0.1"], stdout=subprocess.PIPE)
        real_pid = proc.pid

        executor = Executor(
            dispatcher=MockDispatcher(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        task = work_state.get_task("t1")
        executor.reattach(
            task=task,
            pid=real_pid,
            attempt=1,
            worktree_path=None,
            agent_id="agent-1",
        )

        proc.wait()
        time.sleep(3)

        completed = executor.poll_completed()
        assert len(completed) == 1
        assert completed[0].result.exit_code == -1

        executor.shutdown()


# ===========================================================================
# Test: No duplicate dispatches for tasks with live agents
# ===========================================================================


class TestNoDuplicateDispatch:
    """Test that tasks with live agents are not re-dispatched after daemon restart."""

    def test_alive_agent_prevents_dispatch(
        self, mutation_log, audit_log, session_logger, tmp_project
    ):
        """Running task with alive agent: daemon does NOT re-dispatch it."""
        _create_task(mutation_log, "t1", "Task 1", done_when="do it")
        _start_task(mutation_log, "t1")

        # Start a long-running process to simulate the running agent
        proc = subprocess.Popen(["sleep", "30"], stdout=subprocess.PIPE)
        real_pid = proc.pid
        _create_agent(mutation_log, "agent-1", "t1", pid=real_pid)

        try:
            work_state = WorkState(tmp_project / "data" / "state.db", mutation_log)
            dispatcher = MockDispatcher()

            daemon = Daemon(
                state=work_state,
                mutation_log=mutation_log,
                audit_log=audit_log,
                session_logger=session_logger,
                dispatcher=dispatcher,
                project_root=tmp_project,
                poll_interval=0.1,
                pid_checker=lambda pid: is_pid_alive(pid),
            )

            thread = threading.Thread(target=daemon.start)
            thread.start()
            time.sleep(1.0)
            daemon.stop()
            thread.join(timeout=3)

            # Reconciliation found the alive agent
            assert daemon._reconcile_summary["agents_alive"] == 1
            assert len(daemon._reconcile_summary["alive_agents"]) == 1

            # No dispatches — task is running with live agent
            assert len(dispatcher.dispatched) == 0

            # Task still running
            work_state.refresh()
            assert work_state.get_task("t1")["status"] == "running"
        finally:
            proc.kill()
            proc.wait()

    def test_alive_agent_tracked_by_executor(
        self, mutation_log, audit_log, session_logger, tmp_project
    ):
        """After reattach, alive agent is tracked in executor.in_flight_task_ids,
        preventing _reconcile_external_tasks from touching it."""
        _create_task(mutation_log, "t1", "Task 1", done_when="do it")
        _start_task(mutation_log, "t1")

        proc = subprocess.Popen(["sleep", "30"], stdout=subprocess.PIPE)
        real_pid = proc.pid
        _create_agent(mutation_log, "agent-1", "t1", pid=real_pid)

        try:
            work_state = WorkState(tmp_project / "data" / "state.db", mutation_log)
            dispatcher = MockDispatcher()

            daemon = Daemon(
                state=work_state,
                mutation_log=mutation_log,
                audit_log=audit_log,
                session_logger=session_logger,
                dispatcher=dispatcher,
                project_root=tmp_project,
                poll_interval=0.1,
                pid_checker=lambda pid: is_pid_alive(pid),
            )

            thread = threading.Thread(target=daemon.start)
            thread.start()
            time.sleep(0.5)

            # The executor should be tracking this task
            assert "t1" in daemon.executor.in_flight_task_ids

            daemon.stop()
            thread.join(timeout=3)
        finally:
            proc.kill()
            proc.wait()


# ===========================================================================
# Test: Agent finished while daemon was down
# ===========================================================================


class TestAgentFinishedWhileDaemonDown:
    """Test handling agents that completed while the daemon was down."""

    def test_dead_agent_with_output_completed_on_restart(
        self, mutation_log, audit_log, session_logger, tmp_project
    ):
        """Agent finished (dead) with output → output processed on restart."""
        _create_task(mutation_log, "t1", "Task 1", done_when="do it")
        _start_task(mutation_log, "t1")
        _create_agent(mutation_log, "agent-1", "t1", pid=99999)

        # Agent produced output before daemon could process it
        session_logger.log_dispatch("t1", 1, "prompt", "system", ["Read"], 3.0)
        session_logger.log_output("t1", 1, "Task done!", 0, 5.0)

        # New daemon starts — should reconcile and process output
        work_state = WorkState(tmp_project / "data" / "state.db", mutation_log)
        dispatcher = MockDispatcher()

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
            pid_checker=lambda pid: False,  # Agent is dead
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        time.sleep(0.5)
        daemon.stop()
        thread.join(timeout=3)

        # Task completed via reconciliation
        work_state.refresh()
        assert work_state.get_task("t1")["status"] == "completed"

        # No re-dispatch needed
        assert len(dispatcher.dispatched) == 0

    def test_dead_agent_no_output_marked_failed_then_retried(
        self, mutation_log, audit_log, session_logger, tmp_project
    ):
        """Agent died with no output → marked failed → retried successfully."""
        _create_task(mutation_log, "t1", "Task 1", done_when="do it")
        _start_task(mutation_log, "t1")
        _create_agent(mutation_log, "agent-1", "t1", pid=99999)
        # No session output — agent died immediately

        work_state = WorkState(tmp_project / "data" / "state.db", mutation_log)
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

        # Reconciliation marked failed, then scheduler retried, and mock completed
        work_state.refresh()
        assert work_state.get_task("t1")["status"] == "completed"
        assert len(dispatcher.dispatched) >= 1


# ===========================================================================
# Integration test: dispatch → kill daemon → restart → recover
# ===========================================================================


class TestFullRecoveryFlow:
    """End-to-end: simulate daemon crash and restart with running agent."""

    def test_dispatch_kill_restart_reattach_completes(
        self, mutation_log, audit_log, session_logger, tmp_project
    ):
        """Full flow: task dispatched → daemon killed → restart → agent finishes → output captured.

        Simulates a daemon crash by setting up mutation log state as if an
        agent was dispatched and running.  Starts a real sleep process as the
        "agent", restarts the daemon, and verifies the daemon reattaches
        monitoring.  When the process exits, the daemon captures the output
        and marks the task completed.
        """
        _create_task(mutation_log, "t1", "Recovery Task", done_when="do it")
        _start_task(mutation_log, "t1")

        # Start a short-lived process to simulate agent that will finish soon
        proc = subprocess.Popen(["sleep", "1.5"], stdout=subprocess.PIPE)
        real_pid = proc.pid
        _create_agent(mutation_log, "agent-1", "t1", pid=real_pid)

        # Session output captured before daemon "died"
        session_logger.log_dispatch("t1", 1, "prompt", "system", ["Read"], 3.0)
        session_logger.log_output("t1", 1, "Task completed!", 0, 5.0)

        # "Restart" daemon with fresh state
        work_state = WorkState(tmp_project / "data" / "state.db", mutation_log)
        dispatcher = MockDispatcher()

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
            pid_checker=lambda pid: is_pid_alive(pid),
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()

        # Wait for agent to finish and daemon to process it
        proc.wait()
        time.sleep(4)
        daemon.stop()
        thread.join(timeout=5)

        # Task should be completed
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "completed", (
            f"Expected completed, got {task['status']}"
        )

        # No duplicate dispatches
        assert len(dispatcher.dispatched) == 0

    def test_dispatch_kill_restart_no_duplicate_while_alive(
        self, mutation_log, audit_log, session_logger, tmp_project
    ):
        """Restart with alive agent: no dispatch until agent finishes."""
        _create_task(mutation_log, "t1", "Task 1", done_when="do it")
        _start_task(mutation_log, "t1")

        # Long-running "agent"
        proc = subprocess.Popen(["sleep", "30"], stdout=subprocess.PIPE)
        real_pid = proc.pid
        _create_agent(mutation_log, "agent-1", "t1", pid=real_pid)

        try:
            work_state = WorkState(tmp_project / "data" / "state.db", mutation_log)
            dispatcher = MockDispatcher()

            daemon = Daemon(
                state=work_state,
                mutation_log=mutation_log,
                audit_log=audit_log,
                session_logger=session_logger,
                dispatcher=dispatcher,
                project_root=tmp_project,
                poll_interval=0.1,
                pid_checker=lambda pid: is_pid_alive(pid),
            )

            thread = threading.Thread(target=daemon.start)
            thread.start()
            time.sleep(1.0)
            daemon.stop()
            thread.join(timeout=3)

            # Zero dispatches — agent is still alive
            assert len(dispatcher.dispatched) == 0

            # Task still running
            work_state.refresh()
            assert work_state.get_task("t1")["status"] == "running"
        finally:
            proc.kill()
            proc.wait()

    def test_dispatch_kill_restart_downstream_continues(
        self, mutation_log, audit_log, session_logger, tmp_project
    ):
        """After restart: dead agent output processed, downstream tasks unblocked."""
        # t1 → t2 pipeline
        _create_task(mutation_log, "t1", "Task 1", done_when="do it")
        _start_task(mutation_log, "t1")
        _create_agent(mutation_log, "agent-1", "t1", pid=99999)
        _create_task(
            mutation_log, "t2", "Task 2", done_when="do more", depends_on=["t1"]
        )

        # t1 agent left output
        session_logger.log_dispatch("t1", 1, "prompt", "system", [], 1.0)
        session_logger.log_output("t1", 1, "Done!", 0, 5.0)

        work_state = WorkState(tmp_project / "data" / "state.db", mutation_log)
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
        assert work_state.get_task("t1")["status"] == "completed"
        assert work_state.get_task("t2")["status"] == "completed"
        # t2 was dispatched after t1 was reconciled
        assert len(dispatcher.dispatched) >= 1
