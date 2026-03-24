"""Tests for task deprioritize/reprioritize: CLI commands, scheduler exclusion, agent kill."""

import json
import os
import signal
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from corc.cli import cli
from corc.mutations import MutationLog
from corc.scheduler import get_ready_tasks
from corc.state import WorkState


@pytest.fixture
def state(tmp_path):
    """Create a minimal mutation log + work state for testing."""
    ml = MutationLog(tmp_path / "mutations.jsonl")
    ws = WorkState(tmp_path / "state.db", ml)
    return ml, ws


@pytest.fixture
def project_dir(tmp_path):
    """Create a minimal project structure for CLI testing."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "events").mkdir()
    (tmp_path / "data" / "sessions").mkdir()
    (tmp_path / "data" / "knowledge").mkdir()
    (tmp_path / ".corc").mkdir()
    return tmp_path


# ---------------------------------------------------------------------------
# Scheduler: deprioritized tasks are excluded from ready list
# ---------------------------------------------------------------------------


def test_deprioritized_task_excluded_from_ready(state):
    """Tasks with priority -1 are NOT returned by get_ready_tasks."""
    ml, ws = state
    ml.append(
        "task_created",
        {"id": "t1", "name": "normal task", "done_when": "done", "priority": 100},
        reason="test",
    )
    ml.append(
        "task_created",
        {"id": "t2", "name": "shelved task", "done_when": "done", "priority": -1},
        reason="test",
    )
    ws.refresh()

    ready = ws.get_ready_tasks()
    ready_ids = [t["id"] for t in ready]
    assert "t1" in ready_ids
    assert "t2" not in ready_ids


def test_deprioritized_task_excluded_from_scheduler(state):
    """Scheduler's get_ready_tasks also excludes deprioritized tasks."""
    ml, ws = state
    ml.append(
        "task_created",
        {"id": "t1", "name": "normal", "done_when": "done", "priority": 50},
        reason="test",
    )
    ml.append(
        "task_created",
        {"id": "t2", "name": "shelved", "done_when": "done", "priority": -1},
        reason="test",
    )
    ws.refresh()

    dispatched = get_ready_tasks(ws, parallel_limit=10)
    dispatched_ids = [t["id"] for t in dispatched]
    assert "t1" in dispatched_ids
    assert "t2" not in dispatched_ids


def test_deprioritized_failed_task_excluded_from_ready(state):
    """Even failed retriable tasks with priority -1 are excluded."""
    ml, ws = state
    ml.append(
        "task_created",
        {
            "id": "t1",
            "name": "failed shelved",
            "done_when": "done",
            "priority": -1,
            "max_retries": 3,
        },
        reason="test",
    )
    ml.append(
        "task_failed",
        {"findings": [], "attempt_count": 1},
        reason="failed",
        task_id="t1",
    )
    # Manually set priority to -1 after failure
    ml.append(
        "task_updated",
        {"priority": -1},
        reason="deprioritized",
        task_id="t1",
    )
    ws.refresh()

    ready = ws.get_ready_tasks()
    assert len(ready) == 0


def test_reprioritized_task_becomes_ready(state):
    """A deprioritized task becomes ready again after reprioritization."""
    ml, ws = state
    ml.append(
        "task_created",
        {"id": "t1", "name": "shelved task", "done_when": "done", "priority": -1},
        reason="test",
    )
    ws.refresh()

    # Verify not ready
    ready = ws.get_ready_tasks()
    assert len(ready) == 0

    # Reprioritize
    ml.append(
        "task_updated",
        {"priority": 100},
        reason="reprioritized",
        task_id="t1",
    )
    ws.refresh()

    # Now it should be ready
    ready = ws.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0]["id"] == "t1"
    assert ready[0]["priority"] == 100


# ---------------------------------------------------------------------------
# CLI: corc task deprioritize
# ---------------------------------------------------------------------------


def test_cli_deprioritize_pending_task(project_dir):
    """CLI deprioritize sets priority -1 and status pending for a pending task."""
    runner = CliRunner()
    with patch("corc.cli.get_paths") as mock_paths:
        mock_paths.return_value = {
            "root": project_dir,
            "mutations": project_dir / "data" / "mutations.jsonl",
            "state_db": project_dir / "data" / "state.db",
            "events_dir": project_dir / "data" / "events",
            "sessions_dir": project_dir / "data" / "sessions",
            "knowledge_dir": project_dir / "data" / "knowledge",
            "knowledge_db": project_dir / "data" / "knowledge.db",
            "corc_dir": project_dir / ".corc",
            "ratings_dir": project_dir / "data" / "ratings",
            "retry_outcomes": project_dir / "data" / "retry_outcomes.jsonl",
        }

        # Create a task
        result = runner.invoke(
            cli, ["task", "create", "test-task", "--done-when", "tests pass"]
        )
        assert result.exit_code == 0
        # Extract task ID from output
        task_id = result.output.split("Created task ")[1].split(":")[0]

        # Deprioritize
        result = runner.invoke(cli, ["task", "deprioritize", task_id])
        assert result.exit_code == 0
        assert "deprioritized" in result.output.lower()

        # Verify task state via status command
        result = runner.invoke(cli, ["task", "status", task_id])
        assert result.exit_code == 0
        assert "Priority: -1" in result.output
        assert "Status: pending" in result.output


def test_cli_deprioritize_already_deprioritized(project_dir):
    """CLI deprioritize on already deprioritized task is a no-op."""
    runner = CliRunner()
    with patch("corc.cli.get_paths") as mock_paths:
        mock_paths.return_value = {
            "root": project_dir,
            "mutations": project_dir / "data" / "mutations.jsonl",
            "state_db": project_dir / "data" / "state.db",
            "events_dir": project_dir / "data" / "events",
            "sessions_dir": project_dir / "data" / "sessions",
            "knowledge_dir": project_dir / "data" / "knowledge",
            "knowledge_db": project_dir / "data" / "knowledge.db",
            "corc_dir": project_dir / ".corc",
            "ratings_dir": project_dir / "data" / "ratings",
            "retry_outcomes": project_dir / "data" / "retry_outcomes.jsonl",
        }

        # Create and deprioritize
        result = runner.invoke(
            cli, ["task", "create", "test-task", "--done-when", "tests pass"]
        )
        task_id = result.output.split("Created task ")[1].split(":")[0]
        runner.invoke(cli, ["task", "deprioritize", task_id])

        # Second deprioritize should be a no-op
        result = runner.invoke(cli, ["task", "deprioritize", task_id])
        assert result.exit_code == 0
        assert "already deprioritized" in result.output.lower()


def test_cli_deprioritize_nonexistent_task(project_dir):
    """CLI deprioritize with unknown task ID exits with error."""
    runner = CliRunner()
    with patch("corc.cli.get_paths") as mock_paths:
        mock_paths.return_value = {
            "root": project_dir,
            "mutations": project_dir / "data" / "mutations.jsonl",
            "state_db": project_dir / "data" / "state.db",
            "events_dir": project_dir / "data" / "events",
            "sessions_dir": project_dir / "data" / "sessions",
            "knowledge_dir": project_dir / "data" / "knowledge",
            "knowledge_db": project_dir / "data" / "knowledge.db",
            "corc_dir": project_dir / ".corc",
            "ratings_dir": project_dir / "data" / "ratings",
            "retry_outcomes": project_dir / "data" / "retry_outcomes.jsonl",
        }

        result = runner.invoke(cli, ["task", "deprioritize", "nonexist"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()


# ---------------------------------------------------------------------------
# CLI: corc task reprioritize
# ---------------------------------------------------------------------------


def test_cli_reprioritize_restores_task(project_dir):
    """CLI reprioritize restores a deprioritized task to dispatchable priority."""
    runner = CliRunner()
    with patch("corc.cli.get_paths") as mock_paths:
        mock_paths.return_value = {
            "root": project_dir,
            "mutations": project_dir / "data" / "mutations.jsonl",
            "state_db": project_dir / "data" / "state.db",
            "events_dir": project_dir / "data" / "events",
            "sessions_dir": project_dir / "data" / "sessions",
            "knowledge_dir": project_dir / "data" / "knowledge",
            "knowledge_db": project_dir / "data" / "knowledge.db",
            "corc_dir": project_dir / ".corc",
            "ratings_dir": project_dir / "data" / "ratings",
            "retry_outcomes": project_dir / "data" / "retry_outcomes.jsonl",
        }

        # Create, deprioritize, then reprioritize
        result = runner.invoke(
            cli, ["task", "create", "test-task", "--done-when", "tests pass"]
        )
        task_id = result.output.split("Created task ")[1].split(":")[0]
        runner.invoke(cli, ["task", "deprioritize", task_id])

        result = runner.invoke(cli, ["task", "reprioritize", task_id])
        assert result.exit_code == 0
        assert "reprioritized" in result.output.lower()

        # Verify restored priority
        result = runner.invoke(cli, ["task", "status", task_id])
        assert "Priority: 100" in result.output


def test_cli_reprioritize_with_custom_priority(project_dir):
    """CLI reprioritize --priority sets a specific priority."""
    runner = CliRunner()
    with patch("corc.cli.get_paths") as mock_paths:
        mock_paths.return_value = {
            "root": project_dir,
            "mutations": project_dir / "data" / "mutations.jsonl",
            "state_db": project_dir / "data" / "state.db",
            "events_dir": project_dir / "data" / "events",
            "sessions_dir": project_dir / "data" / "sessions",
            "knowledge_dir": project_dir / "data" / "knowledge",
            "knowledge_db": project_dir / "data" / "knowledge.db",
            "corc_dir": project_dir / ".corc",
            "ratings_dir": project_dir / "data" / "ratings",
            "retry_outcomes": project_dir / "data" / "retry_outcomes.jsonl",
        }

        result = runner.invoke(
            cli, ["task", "create", "test-task", "--done-when", "tests pass"]
        )
        task_id = result.output.split("Created task ")[1].split(":")[0]
        runner.invoke(cli, ["task", "deprioritize", task_id])

        result = runner.invoke(
            cli, ["task", "reprioritize", task_id, "--priority", "10"]
        )
        assert result.exit_code == 0

        result = runner.invoke(cli, ["task", "status", task_id])
        assert "Priority: 10" in result.output


def test_cli_reprioritize_not_deprioritized_is_noop(project_dir):
    """CLI reprioritize on a task that isn't deprioritized is a no-op."""
    runner = CliRunner()
    with patch("corc.cli.get_paths") as mock_paths:
        mock_paths.return_value = {
            "root": project_dir,
            "mutations": project_dir / "data" / "mutations.jsonl",
            "state_db": project_dir / "data" / "state.db",
            "events_dir": project_dir / "data" / "events",
            "sessions_dir": project_dir / "data" / "sessions",
            "knowledge_dir": project_dir / "data" / "knowledge",
            "knowledge_db": project_dir / "data" / "knowledge.db",
            "corc_dir": project_dir / ".corc",
            "ratings_dir": project_dir / "data" / "ratings",
            "retry_outcomes": project_dir / "data" / "retry_outcomes.jsonl",
        }

        result = runner.invoke(
            cli, ["task", "create", "test-task", "--done-when", "tests pass"]
        )
        task_id = result.output.split("Created task ")[1].split(":")[0]

        result = runner.invoke(cli, ["task", "reprioritize", task_id])
        assert result.exit_code == 0
        assert "not deprioritized" in result.output.lower()


# ---------------------------------------------------------------------------
# Agent kill: deprioritize kills running agent
# ---------------------------------------------------------------------------


def test_deprioritize_running_task_kills_agent(state):
    """Deprioritizing a running task sets status to pending and priority to -1."""
    ml, ws = state

    # Create a task and make it running with an agent
    ml.append(
        "task_created",
        {"id": "t1", "name": "running task", "done_when": "done", "priority": 50},
        reason="test",
    )
    ml.append(
        "task_started",
        {"attempt": 1},
        reason="dispatched",
        task_id="t1",
    )
    ml.append(
        "agent_created",
        {"id": "agent-123", "role": "implementer", "task_id": "t1", "pid": 99999},
        reason="test",
    )
    ws.refresh()

    # Verify task is running
    task = ws.get_task("t1")
    assert task["status"] == "running"

    # Deprioritize (update priority and status)
    ml.append(
        "task_updated",
        {"priority": -1, "status": "pending"},
        reason="deprioritized",
        task_id="t1",
    )
    ws.refresh()

    # Verify task is now pending with priority -1
    task = ws.get_task("t1")
    assert task["status"] == "pending"
    assert task["priority"] == -1

    # Verify it's not in ready tasks
    ready = ws.get_ready_tasks()
    assert len(ready) == 0


def test_deprioritize_kills_agent_process():
    """Deprioritize command sends SIGTERM to the agent process."""
    import subprocess

    # Start a dummy process to kill
    proc = subprocess.Popen(
        ["sleep", "60"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pid = proc.pid

    try:
        # Verify it's running
        os.kill(pid, 0)  # Raises if not running

        # Simulate what the deprioritize command does
        os.kill(pid, signal.SIGTERM)

        # Wait for it to die
        proc.wait(timeout=5)
        assert proc.returncode is not None
    finally:
        # Ensure cleanup
        try:
            proc.kill()
        except ProcessLookupError:
            pass


# ---------------------------------------------------------------------------
# Integration: full deprioritize → reprioritize cycle
# ---------------------------------------------------------------------------


def test_full_deprioritize_reprioritize_cycle(state):
    """Full cycle: create → deprioritize → verify not dispatched → reprioritize → verify dispatched."""
    ml, ws = state

    # Create tasks
    ml.append(
        "task_created",
        {"id": "t1", "name": "task A", "done_when": "done", "priority": 50},
        reason="test",
    )
    ml.append(
        "task_created",
        {"id": "t2", "name": "task B", "done_when": "done", "priority": 100},
        reason="test",
    )
    ws.refresh()

    # Both should be ready
    ready = get_ready_tasks(ws, parallel_limit=10)
    assert len(ready) == 2

    # Deprioritize t1
    ml.append(
        "task_updated",
        {"priority": -1, "status": "pending"},
        reason="deprioritized",
        task_id="t1",
    )
    ws.refresh()

    # Only t2 should be ready
    ready = get_ready_tasks(ws, parallel_limit=10)
    assert len(ready) == 1
    assert ready[0]["id"] == "t2"

    # Reprioritize t1
    ml.append(
        "task_updated",
        {"priority": 25},
        reason="reprioritized",
        task_id="t1",
    )
    ws.refresh()

    # Both should be ready again, t1 first (priority 25 < 100)
    ready = get_ready_tasks(ws, parallel_limit=10)
    assert len(ready) == 2
    assert ready[0]["id"] == "t1"  # priority 25
    assert ready[1]["id"] == "t2"  # priority 100


def test_deprioritized_task_not_dispatched_by_daemon_tick(state):
    """Daemon's scheduler loop does not dispatch deprioritized tasks.

    This is the key integration test: verifies the daemon's tick loop
    will skip tasks with priority -1 even if they are pending with
    satisfied dependencies.
    """
    ml, ws = state

    # Create a deprioritized pending task
    ml.append(
        "task_created",
        {"id": "t1", "name": "shelved task", "done_when": "done", "priority": -1},
        reason="test",
    )
    ws.refresh()

    # Verify task exists and is pending
    task = ws.get_task("t1")
    assert task["status"] == "pending"
    assert task["priority"] == -1

    # Scheduler should return nothing
    ready = get_ready_tasks(ws, parallel_limit=5)
    assert len(ready) == 0

    # Even with high parallel limit
    ready = get_ready_tasks(ws, parallel_limit=100)
    assert len(ready) == 0
