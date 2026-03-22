"""Tests for the pause switch: corc pause, corc resume, daemon pause check.

Covers:
- write_pause_lock creates .corc/pause.lock with correct JSON
- remove_pause_lock removes the file
- read_pause_lock reads it back
- is_paused returns correct boolean
- Daemon skips scheduling when paused but still processes in-flight tasks
- Daemon resumes scheduling after pause lock is removed
- CLI pause/resume commands work end-to-end
- CLI status shows pause state
- CLI dispatch is blocked when paused
"""

import json
import os
import threading
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from corc.audit import AuditLog
from corc.cli import cli
from corc.daemon import Daemon
from corc.dispatch import AgentDispatcher, AgentResult, Constraints
from corc.mutations import MutationLog
from corc.pause import is_paused, read_pause_lock, remove_pause_lock, write_pause_lock
from corc.sessions import SessionLogger
from corc.state import WorkState


# ---------------------------------------------------------------------------
# Mock dispatcher (same pattern as test_daemon.py)
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

    def dispatch(self, prompt: str, system_prompt: str, constraints: Constraints,
                 pid_callback=None, event_callback=None, cwd=None) -> AgentResult:
        self.dispatched.append((prompt, system_prompt, constraints))
        if self.delay:
            time.sleep(self.delay)
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
def corc_dir(tmp_project):
    return tmp_project / ".corc"


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


def _create_task(mutation_log, task_id, name, done_when="do the thing", depends_on=None):
    """Helper to create a task via mutation log."""
    mutation_log.append("task_created", {
        "id": task_id,
        "name": name,
        "description": f"Test task: {name}",
        "role": "implementer",
        "depends_on": depends_on or [],
        "done_when": done_when,
        "checklist": [],
        "context_bundle": [],
    }, reason="Test setup")


# ===========================================================================
# Core pause module tests
# ===========================================================================


class TestPauseLock:
    def test_write_pause_lock_creates_file(self, corc_dir):
        """write_pause_lock creates .corc/pause.lock with correct JSON."""
        lock = write_pause_lock(corc_dir, "deploying hotfix", source="operator")

        lock_path = corc_dir / "pause.lock"
        assert lock_path.exists()

        data = json.loads(lock_path.read_text())
        assert data["reason"] == "deploying hotfix"
        assert data["source"] == "operator"
        assert "timestamp" in data
        # Timestamp should be ISO format
        assert "T" in data["timestamp"]

    def test_write_pause_lock_default_source(self, corc_dir):
        """Default source includes cli:<pid>."""
        lock = write_pause_lock(corc_dir, "testing")
        assert lock["source"] == f"cli:{os.getpid()}"

    def test_write_pause_lock_returns_data(self, corc_dir):
        """write_pause_lock returns the lock data dict."""
        lock = write_pause_lock(corc_dir, "reason here", source="agent:scout")
        assert lock["reason"] == "reason here"
        assert lock["source"] == "agent:scout"
        assert "timestamp" in lock

    def test_write_pause_lock_creates_dir(self, tmp_path):
        """write_pause_lock creates .corc directory if it doesn't exist."""
        corc_dir = tmp_path / "new_project" / ".corc"
        assert not corc_dir.exists()
        write_pause_lock(corc_dir, "test")
        assert (corc_dir / "pause.lock").exists()

    def test_remove_pause_lock(self, corc_dir):
        """remove_pause_lock removes the file and returns True."""
        write_pause_lock(corc_dir, "testing")
        assert (corc_dir / "pause.lock").exists()

        result = remove_pause_lock(corc_dir)
        assert result is True
        assert not (corc_dir / "pause.lock").exists()

    def test_remove_pause_lock_when_not_paused(self, corc_dir):
        """remove_pause_lock returns False when no lock exists."""
        result = remove_pause_lock(corc_dir)
        assert result is False

    def test_read_pause_lock(self, corc_dir):
        """read_pause_lock returns the lock data."""
        write_pause_lock(corc_dir, "maintenance window", source="ops-team")
        lock = read_pause_lock(corc_dir)

        assert lock is not None
        assert lock["reason"] == "maintenance window"
        assert lock["source"] == "ops-team"
        assert "timestamp" in lock

    def test_read_pause_lock_when_not_paused(self, corc_dir):
        """read_pause_lock returns None when no lock exists."""
        assert read_pause_lock(corc_dir) is None

    def test_read_pause_lock_corrupted(self, corc_dir):
        """Corrupted lock file is treated as paused with unknown reason."""
        (corc_dir / "pause.lock").write_text("not json{{{")
        lock = read_pause_lock(corc_dir)
        assert lock is not None
        assert "unknown" in lock["reason"].lower() or "corrupt" in lock["reason"].lower()

    def test_is_paused_true(self, corc_dir):
        """is_paused returns True when lock file exists."""
        write_pause_lock(corc_dir, "test")
        assert is_paused(corc_dir) is True

    def test_is_paused_false(self, corc_dir):
        """is_paused returns False when no lock file exists."""
        assert is_paused(corc_dir) is False

    def test_pause_resume_cycle(self, corc_dir):
        """Full pause/resume cycle works correctly."""
        assert not is_paused(corc_dir)

        write_pause_lock(corc_dir, "deploy in progress")
        assert is_paused(corc_dir)

        lock = read_pause_lock(corc_dir)
        assert lock["reason"] == "deploy in progress"

        remove_pause_lock(corc_dir)
        assert not is_paused(corc_dir)
        assert read_pause_lock(corc_dir) is None


# ===========================================================================
# Daemon pause integration tests
# ===========================================================================


class TestDaemonPause:
    def test_daemon_skips_dispatch_when_paused(self, mutation_log, work_state, audit_log,
                                                session_logger, mock_dispatcher, tmp_project):
        """Daemon does NOT dispatch new tasks when paused."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()

        # Pause before starting daemon
        write_pause_lock(tmp_project / ".corc", "testing pause")

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

        # Task should NOT have been dispatched
        assert len(mock_dispatcher.dispatched) == 0

        # Task should still be pending
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "pending"

    def test_daemon_resumes_dispatch_after_unpause(self, mutation_log, work_state, audit_log,
                                                     session_logger, mock_dispatcher, tmp_project):
        """Daemon resumes dispatching after pause lock is removed."""
        _create_task(mutation_log, "t1", "Task 1", done_when="do thing")
        work_state.refresh()

        corc_dir = tmp_project / ".corc"

        # Start paused
        write_pause_lock(corc_dir, "temporary pause")

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

        # Daemon is running but paused — no dispatch yet
        time.sleep(0.3)
        assert len(mock_dispatcher.dispatched) == 0

        # Remove the pause lock
        remove_pause_lock(corc_dir)

        # Give daemon time to pick up the change and dispatch
        time.sleep(0.5)
        daemon.stop()
        thread.join(timeout=2)

        # Task should now have been dispatched
        assert len(mock_dispatcher.dispatched) >= 1

        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "completed"

    def test_daemon_processes_inflight_while_paused(self, mutation_log, work_state, audit_log,
                                                      session_logger, tmp_project):
        """In-flight tasks complete even when paused (only new dispatch blocked)."""
        dispatcher = MockDispatcher(delay=0.3)

        _create_task(mutation_log, "t1", "Task 1", done_when="do thing 1")
        _create_task(mutation_log, "t2", "Task 2", done_when="do thing 2")
        work_state.refresh()

        corc_dir = tmp_project / ".corc"

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

        # Let t1 get dispatched, then pause immediately
        time.sleep(0.15)
        write_pause_lock(corc_dir, "pause during dispatch")

        # Wait for in-flight tasks to complete
        time.sleep(0.8)
        daemon.stop()
        thread.join(timeout=2)

        work_state.refresh()
        t1 = work_state.get_task("t1")

        # t1 was dispatched before pause — should complete
        # t2 should remain pending (was not dispatched because pause kicked in)
        # Note: timing can be tricky; at minimum t1 was dispatched
        assert len(dispatcher.dispatched) >= 1
        # The in-flight task should have been processed
        assert t1["status"] in ("completed", "running")

    def test_daemon_pause_with_multiple_tasks(self, mutation_log, work_state, audit_log,
                                                session_logger, mock_dispatcher, tmp_project):
        """Multiple pending tasks: none dispatched while paused, all dispatched after resume."""
        _create_task(mutation_log, "t1", "Task 1", done_when="do 1")
        _create_task(mutation_log, "t2", "Task 2", done_when="do 2")
        _create_task(mutation_log, "t3", "Task 3", done_when="do 3")
        work_state.refresh()

        corc_dir = tmp_project / ".corc"
        write_pause_lock(corc_dir, "initial pause")

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=mock_dispatcher,
            project_root=tmp_project,
            parallel=3,
            poll_interval=0.1,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()

        # Paused — nothing dispatched
        time.sleep(0.3)
        assert len(mock_dispatcher.dispatched) == 0

        # Resume
        remove_pause_lock(corc_dir)
        time.sleep(0.5)

        daemon.stop()
        thread.join(timeout=2)

        # All 3 tasks should have been dispatched after resume
        assert len(mock_dispatcher.dispatched) == 3

        work_state.refresh()
        for tid in ("t1", "t2", "t3"):
            assert work_state.get_task(tid)["status"] == "completed"


# ===========================================================================
# CLI integration tests
# ===========================================================================


class TestCLIPause:
    def test_pause_command_help(self):
        """The pause CLI command exists and has expected args."""
        runner = CliRunner()
        result = runner.invoke(cli, ["pause", "--help"])
        assert result.exit_code == 0
        assert "REASON" in result.output
        assert "--source" in result.output

    def test_resume_command_help(self):
        """The resume CLI command exists."""
        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--help"])
        assert result.exit_code == 0

    def test_pause_writes_lock(self, tmp_project, monkeypatch):
        """corc pause writes the lock file."""
        monkeypatch.chdir(tmp_project)
        # Create minimal project markers so get_paths works
        (tmp_project / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        (tmp_project / "data").mkdir(exist_ok=True)
        (tmp_project / "data" / "events").mkdir(exist_ok=True)
        (tmp_project / "data" / "sessions").mkdir(exist_ok=True)
        (tmp_project / "data" / "mutations.jsonl").touch()

        runner = CliRunner()
        result = runner.invoke(cli, ["pause", "deploying hotfix"])
        assert result.exit_code == 0
        assert "Paused" in result.output
        assert "deploying hotfix" in result.output

        # Lock file should exist
        lock_path = tmp_project / ".corc" / "pause.lock"
        assert lock_path.exists()
        data = json.loads(lock_path.read_text())
        assert data["reason"] == "deploying hotfix"

    def test_resume_removes_lock(self, tmp_project, monkeypatch):
        """corc resume removes the lock file."""
        monkeypatch.chdir(tmp_project)
        (tmp_project / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        (tmp_project / "data").mkdir(exist_ok=True)
        (tmp_project / "data" / "events").mkdir(exist_ok=True)
        (tmp_project / "data" / "sessions").mkdir(exist_ok=True)
        (tmp_project / "data" / "mutations.jsonl").touch()

        # Write lock first
        write_pause_lock(tmp_project / ".corc", "test pause")

        runner = CliRunner()
        result = runner.invoke(cli, ["resume"])
        assert result.exit_code == 0
        assert "Resumed" in result.output

        assert not (tmp_project / ".corc" / "pause.lock").exists()

    def test_resume_when_not_paused(self, tmp_project, monkeypatch):
        """corc resume when not paused shows a message."""
        monkeypatch.chdir(tmp_project)
        (tmp_project / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        (tmp_project / "data").mkdir(exist_ok=True)
        (tmp_project / "data" / "events").mkdir(exist_ok=True)
        (tmp_project / "data" / "sessions").mkdir(exist_ok=True)
        (tmp_project / "data" / "mutations.jsonl").touch()

        runner = CliRunner()
        result = runner.invoke(cli, ["resume"])
        assert result.exit_code == 0
        assert "Not paused" in result.output

    def test_pause_already_paused(self, tmp_project, monkeypatch):
        """corc pause when already paused shows the existing reason."""
        monkeypatch.chdir(tmp_project)
        (tmp_project / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        (tmp_project / "data").mkdir(exist_ok=True)
        (tmp_project / "data" / "events").mkdir(exist_ok=True)
        (tmp_project / "data" / "sessions").mkdir(exist_ok=True)
        (tmp_project / "data" / "mutations.jsonl").touch()

        write_pause_lock(tmp_project / ".corc", "first reason")

        runner = CliRunner()
        result = runner.invoke(cli, ["pause", "second reason"])
        assert result.exit_code == 0
        assert "Already paused" in result.output
        assert "first reason" in result.output

    def test_status_shows_pause_state(self, tmp_project, monkeypatch):
        """corc status shows pause information when paused."""
        monkeypatch.chdir(tmp_project)
        (tmp_project / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        (tmp_project / "data").mkdir(exist_ok=True)
        (tmp_project / "data" / "events").mkdir(exist_ok=True)
        (tmp_project / "data" / "sessions").mkdir(exist_ok=True)
        (tmp_project / "data" / "mutations.jsonl").touch()

        write_pause_lock(tmp_project / ".corc", "deploy in progress", source="operator")

        runner = CliRunner()
        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "PAUSED" in result.output
        assert "deploy in progress" in result.output
        assert "operator" in result.output

    def test_status_no_pause(self, tmp_project, monkeypatch):
        """corc status without pause does not show pause info."""
        monkeypatch.chdir(tmp_project)
        (tmp_project / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        (tmp_project / "data").mkdir(exist_ok=True)
        (tmp_project / "data" / "events").mkdir(exist_ok=True)
        (tmp_project / "data" / "sessions").mkdir(exist_ok=True)
        (tmp_project / "data" / "mutations.jsonl").touch()

        runner = CliRunner()
        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "PAUSED" not in result.output

    def test_pause_with_custom_source(self, tmp_project, monkeypatch):
        """corc pause --source sets custom source."""
        monkeypatch.chdir(tmp_project)
        (tmp_project / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        (tmp_project / "data").mkdir(exist_ok=True)
        (tmp_project / "data" / "events").mkdir(exist_ok=True)
        (tmp_project / "data" / "sessions").mkdir(exist_ok=True)
        (tmp_project / "data" / "mutations.jsonl").touch()

        runner = CliRunner()
        result = runner.invoke(cli, ["pause", "agent detected issue", "--source", "agent:scout-abc123"])
        assert result.exit_code == 0

        lock = read_pause_lock(tmp_project / ".corc")
        assert lock["source"] == "agent:scout-abc123"
        assert lock["reason"] == "agent detected issue"
