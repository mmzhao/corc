"""Tests for dispatch timeout reliability.

Tests that agent timeout kills work reliably in all scenarios:
- Timer-based timeout fires and kills the process
- Snapshotted timeout value is resilient to module-level changes (hot-reload)
- kill_agent_process utility works correctly
- Backup timeout mechanism (via executor.kill_timed_out_agents) catches
  agents that the Timer thread missed
"""

import os
import signal
import subprocess
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from corc.dispatch import (
    AgentResult,
    ClaudeCodeDispatcher,
    Constraints,
    _DISPATCH_DEFAULTS,
    kill_agent_process,
)


# ---------------------------------------------------------------------------
# Tests: kill_agent_process
# ---------------------------------------------------------------------------


class TestKillAgentProcess:
    """Test the kill_agent_process utility function."""

    def test_kill_running_process(self):
        """kill_agent_process kills a real running process."""
        proc = subprocess.Popen(
            ["sleep", "30"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        pid = proc.pid

        # Verify alive
        assert _pid_alive(pid)

        result = kill_agent_process(pid)
        assert result is True

        # Wait for process to actually die
        proc.wait(timeout=5)
        assert not _pid_alive(pid)

    def test_kill_already_dead_process(self):
        """kill_agent_process handles already-dead processes gracefully."""
        proc = subprocess.Popen(
            ["true"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        proc.wait()
        pid = proc.pid

        # Process is already dead
        result = kill_agent_process(pid)
        assert result is True

    def test_kill_nonexistent_pid(self):
        """kill_agent_process handles nonexistent PIDs."""
        result = kill_agent_process(999999999)
        assert result is True  # ProcessLookupError → already dead


# ---------------------------------------------------------------------------
# Tests: Timer-based timeout in ClaudeCodeDispatcher
# ---------------------------------------------------------------------------


class TestDispatchTimeout:
    """Test that dispatch timeout works via threading.Timer."""

    def test_timeout_kills_slow_process(self):
        """A slow process is killed when it exceeds the timeout."""
        original_timeout = _DISPATCH_DEFAULTS["agent_timeout_s"]
        _DISPATCH_DEFAULTS["agent_timeout_s"] = 1  # 1 second

        dispatcher = ClaudeCodeDispatcher()
        pids_seen = []

        def track_pid(pid):
            pids_seen.append(pid)

        try:
            # Create the real process BEFORE the patch (patch affects
            # the shared subprocess module, not just corc.dispatch)
            slow_proc = subprocess.Popen(
                ["sleep", "30"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            with patch("corc.dispatch.subprocess.Popen", return_value=slow_proc):
                result = dispatcher.dispatch(
                    prompt="test",
                    system_prompt="test",
                    constraints=Constraints(),
                    pid_callback=track_pid,
                )
        finally:
            _DISPATCH_DEFAULTS["agent_timeout_s"] = original_timeout

        assert result.exit_code == -1
        assert "TIMEOUT" in result.output
        assert result.duration_s >= 0.5  # Must have waited some time
        assert len(pids_seen) == 1

    def test_normal_completion_no_timeout(self):
        """A fast process completes normally without timeout."""
        original_timeout = _DISPATCH_DEFAULTS["agent_timeout_s"]
        _DISPATCH_DEFAULTS["agent_timeout_s"] = 30  # 30 seconds

        try:
            # Create a process that exits immediately
            fast_proc = subprocess.Popen(
                ["echo", "done"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            with patch("corc.dispatch.subprocess.Popen", return_value=fast_proc):
                dispatcher = ClaudeCodeDispatcher()
                result = dispatcher.dispatch(
                    prompt="test",
                    system_prompt="test",
                    constraints=Constraints(),
                )
        finally:
            _DISPATCH_DEFAULTS["agent_timeout_s"] = original_timeout

        # Should complete normally (not timeout)
        assert "TIMEOUT" not in result.output

    def test_timeout_value_snapshotted(self):
        """Timeout value is captured at dispatch start, not read at fire time.

        This simulates hot-reload changing _DISPATCH_DEFAULTS mid-dispatch.
        The timer should still fire at the original timeout value.
        """
        original_timeout = _DISPATCH_DEFAULTS["agent_timeout_s"]
        _DISPATCH_DEFAULTS["agent_timeout_s"] = 1  # Short timeout

        start = time.time()
        timer_fired = threading.Event()

        try:
            slow_proc = subprocess.Popen(
                ["sleep", "30"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            with patch("corc.dispatch.subprocess.Popen", return_value=slow_proc):
                dispatcher = ClaudeCodeDispatcher()

                # Start dispatch in a thread
                result_holder = []

                def run_dispatch():
                    r = dispatcher.dispatch(
                        prompt="test",
                        system_prompt="test",
                        constraints=Constraints(),
                    )
                    result_holder.append(r)
                    timer_fired.set()

                t = threading.Thread(target=run_dispatch)
                t.start()

                # Simulate hot-reload: change the timeout to a very long value
                # AFTER dispatch started. The timer should still fire at 1s.
                time.sleep(0.2)
                _DISPATCH_DEFAULTS["agent_timeout_s"] = 9999

                timer_fired.wait(timeout=10)
                t.join(timeout=5)
        finally:
            _DISPATCH_DEFAULTS["agent_timeout_s"] = original_timeout

        elapsed = time.time() - start
        assert len(result_holder) == 1
        result = result_holder[0]
        assert "TIMEOUT" in result.output
        # Should have timed out after ~1 second, not 9999 seconds
        assert elapsed < 5


# ---------------------------------------------------------------------------
# Tests: Executor backup timeout mechanism
# ---------------------------------------------------------------------------


class TestExecutorBackupTimeout:
    """Test the backup timeout mechanism in executor.kill_timed_out_agents."""

    def test_kill_timed_out_agent(self, tmp_path):
        """Executor kills agents that exceed the timeout."""
        from corc.audit import AuditLog
        from corc.executor import Executor
        from corc.mutations import MutationLog
        from corc.sessions import SessionLogger
        from corc.state import WorkState

        # Set up infrastructure
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "events").mkdir()
        (tmp_path / "data" / "sessions").mkdir()
        (tmp_path / ".corc").mkdir()

        ml = MutationLog(tmp_path / "data" / "mutations.jsonl")
        ws = WorkState(tmp_path / "data" / "state.db", ml)
        al = AuditLog(tmp_path / "data" / "events")
        sl = SessionLogger(tmp_path / "data" / "sessions")

        dispatcher = MagicMock()
        executor = Executor(
            dispatcher=dispatcher,
            mutation_log=ml,
            state=ws,
            audit_log=al,
            session_logger=sl,
            project_root=tmp_path,
        )

        # Start a real process
        proc = subprocess.Popen(
            ["sleep", "30"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Register it as an in-flight agent
        task_id = "test-task-1"
        executor._agent_pids[task_id] = proc.pid
        executor._agent_start_times[task_id] = time.time() - 100  # Started 100s ago

        # Kill agents with a 50s timeout — our agent started 100s ago
        killed = executor.kill_timed_out_agents(50.0)

        assert task_id in killed

        # Process should be dead
        proc.wait(timeout=5)
        assert not _pid_alive(proc.pid)

    def test_no_kill_within_timeout(self, tmp_path):
        """Executor does NOT kill agents that are within the timeout."""
        from corc.audit import AuditLog
        from corc.executor import Executor
        from corc.mutations import MutationLog
        from corc.sessions import SessionLogger
        from corc.state import WorkState

        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "events").mkdir()
        (tmp_path / "data" / "sessions").mkdir()
        (tmp_path / ".corc").mkdir()

        ml = MutationLog(tmp_path / "data" / "mutations.jsonl")
        ws = WorkState(tmp_path / "data" / "state.db", ml)
        al = AuditLog(tmp_path / "data" / "events")
        sl = SessionLogger(tmp_path / "data" / "sessions")

        dispatcher = MagicMock()
        executor = Executor(
            dispatcher=dispatcher,
            mutation_log=ml,
            state=ws,
            audit_log=al,
            session_logger=sl,
            project_root=tmp_path,
        )

        # Start a real process
        proc = subprocess.Popen(
            ["sleep", "30"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Register as agent that just started
        task_id = "test-task-2"
        executor._agent_pids[task_id] = proc.pid
        executor._agent_start_times[task_id] = time.time()  # Just started

        # Kill with 1800s timeout — agent is well within limits
        killed = executor.kill_timed_out_agents(1800.0)

        assert killed == []
        assert _pid_alive(proc.pid)  # Process should still be alive

        # Cleanup
        proc.kill()
        proc.wait()

    def test_kill_multiple_timed_out(self, tmp_path):
        """Executor kills multiple timed-out agents in one call."""
        from corc.audit import AuditLog
        from corc.executor import Executor
        from corc.mutations import MutationLog
        from corc.sessions import SessionLogger
        from corc.state import WorkState

        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "events").mkdir()
        (tmp_path / "data" / "sessions").mkdir()
        (tmp_path / ".corc").mkdir()

        ml = MutationLog(tmp_path / "data" / "mutations.jsonl")
        ws = WorkState(tmp_path / "data" / "state.db", ml)
        al = AuditLog(tmp_path / "data" / "events")
        sl = SessionLogger(tmp_path / "data" / "sessions")

        dispatcher = MagicMock()
        executor = Executor(
            dispatcher=dispatcher,
            mutation_log=ml,
            state=ws,
            audit_log=al,
            session_logger=sl,
            project_root=tmp_path,
        )

        procs = []
        for i in range(3):
            proc = subprocess.Popen(
                ["sleep", "30"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            procs.append(proc)
            task_id = f"task-{i}"
            executor._agent_pids[task_id] = proc.pid
            executor._agent_start_times[task_id] = time.time() - 200  # All old

        killed = executor.kill_timed_out_agents(100.0)

        assert len(killed) == 3

        for proc in procs:
            proc.wait(timeout=5)

    def test_no_pid_skipped(self, tmp_path):
        """Agents with no PID are skipped (not killed)."""
        from corc.audit import AuditLog
        from corc.executor import Executor
        from corc.mutations import MutationLog
        from corc.sessions import SessionLogger
        from corc.state import WorkState

        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "events").mkdir()
        (tmp_path / "data" / "sessions").mkdir()
        (tmp_path / ".corc").mkdir()

        ml = MutationLog(tmp_path / "data" / "mutations.jsonl")
        ws = WorkState(tmp_path / "data" / "state.db", ml)
        al = AuditLog(tmp_path / "data" / "events")
        sl = SessionLogger(tmp_path / "data" / "sessions")

        dispatcher = MagicMock()
        executor = Executor(
            dispatcher=dispatcher,
            mutation_log=ml,
            state=ws,
            audit_log=al,
            session_logger=sl,
            project_root=tmp_path,
        )

        # Agent with start time but no PID
        executor._agent_start_times["no-pid-task"] = time.time() - 200

        killed = executor.kill_timed_out_agents(100.0)
        assert killed == []


# ---------------------------------------------------------------------------
# Tests: Timer thread resilience to hot-reload
# ---------------------------------------------------------------------------


class TestTimerHotReloadResilience:
    """Test that running Timer threads survive module-level hot-reload changes."""

    def test_timer_fires_after_globals_change(self):
        """A Timer thread created before _DISPATCH_DEFAULTS changes still fires."""
        fired = threading.Event()
        original = _DISPATCH_DEFAULTS["agent_timeout_s"]

        def on_timeout():
            fired.set()

        # Create timer using current default
        timer = threading.Timer(0.5, on_timeout)
        timer.daemon = True
        timer.start()

        # Simulate hot-reload changing the global
        _DISPATCH_DEFAULTS["agent_timeout_s"] = 99999

        try:
            # Timer should still fire at 0.5s regardless of the global change
            assert fired.wait(timeout=3), "Timer did not fire after globals changed"
        finally:
            _DISPATCH_DEFAULTS["agent_timeout_s"] = original
            timer.cancel()

    def test_closure_survives_module_reload(self):
        """Closure variables (proc, timed_out) are not affected by module reload.

        We verify that the closure captures local variables, not module-level
        globals, for the critical kill logic.
        """
        # This tests the pattern used in ClaudeCodeDispatcher.dispatch()
        timed_out = False
        proc_killed = threading.Event()

        def _kill_on_timeout():
            nonlocal timed_out
            timed_out = True
            proc_killed.set()

        timer = threading.Timer(0.3, _kill_on_timeout)
        timer.daemon = True
        timer.start()

        # Wait for the timer
        assert proc_killed.wait(timeout=3), "Timer closure did not fire"
        assert timed_out is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    """Check if PID exists."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
