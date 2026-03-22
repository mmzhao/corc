"""Tests for CLI task complete — specifically the double-completion guard."""

import json
import uuid

import pytest
from click.testing import CliRunner

from corc.cli import cli
from corc.mutations import MutationLog
from corc.state import WorkState


@pytest.fixture
def tmp_project(tmp_path, monkeypatch):
    """Create a minimal project structure and point corc config at it."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "events").mkdir()
    (tmp_path / "data" / "sessions").mkdir()
    (tmp_path / "data" / "knowledge").mkdir()
    (tmp_path / ".corc").mkdir()

    # Override get_paths so the CLI uses our temp directory
    monkeypatch.setattr("corc.cli.get_paths", lambda: {
        "root": tmp_path,
        "corc_dir": tmp_path / ".corc",
        "mutations": tmp_path / "data" / "mutations.jsonl",
        "state_db": tmp_path / "data" / "state.db",
        "events_dir": tmp_path / "data" / "events",
        "sessions_dir": tmp_path / "data" / "sessions",
        "knowledge_dir": tmp_path / "data" / "knowledge",
        "knowledge_db": tmp_path / "data" / "knowledge.db",
    })

    return tmp_path


@pytest.fixture
def seeded_task(tmp_project):
    """Create a task and mark it completed via mutation log, return task_id."""
    ml = MutationLog(tmp_project / "data" / "mutations.jsonl")
    task_id = "test0001"
    ml.append("task_created", {
        "id": task_id,
        "name": "Test Task",
        "description": "A task for testing",
        "role": "implementer",
        "depends_on": [],
        "done_when": "tests pass",
        "checklist": [],
        "context_bundle": [],
    }, reason="Test setup")
    return task_id, ml


class TestTaskCompleteDoubleCompletion:
    def test_complete_task_succeeds_first_time(self, tmp_project, seeded_task):
        """First completion should succeed normally."""
        task_id, ml = seeded_task
        runner = CliRunner()

        result = runner.invoke(cli, ["task", "complete", task_id])
        assert result.exit_code == 0
        assert "marked as completed" in result.output

    def test_complete_already_completed_task_returns_error(self, tmp_project, seeded_task):
        """Second completion of the same task should fail with an error."""
        task_id, ml = seeded_task
        runner = CliRunner()

        # First completion
        result1 = runner.invoke(cli, ["task", "complete", task_id])
        assert result1.exit_code == 0
        assert "marked as completed" in result1.output

        # Count mutations after first completion
        mutations_before = len(ml.read_all())

        # Second completion — should error
        result2 = runner.invoke(cli, ["task", "complete", task_id])
        assert result2.exit_code != 0
        assert "already completed" in result2.output

        # No new mutations should have been written
        mutations_after = len(ml.read_all())
        assert mutations_after == mutations_before

    def test_complete_nonexistent_task_returns_error(self, tmp_project, seeded_task):
        """Completing a nonexistent task should show 'not found'."""
        runner = CliRunner()
        result = runner.invoke(cli, ["task", "complete", "nonexistent"])
        assert "not found" in result.output
