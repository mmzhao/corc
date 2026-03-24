"""Tests for corc task set-repo and --target-repo on task create."""

import json

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

    monkeypatch.setattr(
        "corc.cli.get_paths",
        lambda: {
            "root": tmp_path,
            "corc_dir": tmp_path / ".corc",
            "mutations": tmp_path / "data" / "mutations.jsonl",
            "state_db": tmp_path / "data" / "state.db",
            "events_dir": tmp_path / "data" / "events",
            "sessions_dir": tmp_path / "data" / "sessions",
            "knowledge_dir": tmp_path / "data" / "knowledge",
            "knowledge_db": tmp_path / "data" / "knowledge.db",
        },
    )

    return tmp_path


@pytest.fixture
def seeded_task(tmp_project):
    """Create a task without target_repo, return (task_id, ml)."""
    ml = MutationLog(tmp_project / "data" / "mutations.jsonl")
    task_id = "test0001"
    ml.append(
        "task_created",
        {
            "id": task_id,
            "name": "Test Task",
            "description": "A task for testing",
            "role": "implementer",
            "depends_on": [],
            "done_when": "tests pass",
            "checklist": [],
            "context_bundle": [],
        },
        reason="Test setup",
    )
    return task_id, ml


class TestTaskSetRepo:
    """Tests for `corc task set-repo TASK_ID REPO`."""

    def test_set_repo_on_existing_task(self, tmp_project, seeded_task):
        """set-repo should set target_repo on an existing task."""
        task_id, ml = seeded_task
        runner = CliRunner()

        result = runner.invoke(cli, ["task", "set-repo", task_id, "fdp"])
        assert result.exit_code == 0
        assert "target repo set to 'fdp'" in result.output

        # Verify in state
        ws = WorkState(tmp_project / "data" / "state.db", ml)
        task = ws.get_task(task_id)
        assert task["target_repo"] == "fdp"

    def test_set_repo_shows_in_status(self, tmp_project, seeded_task):
        """After set-repo, task status should display Target repo."""
        task_id, _ml = seeded_task
        runner = CliRunner()

        runner.invoke(cli, ["task", "set-repo", task_id, "fdp"])
        result = runner.invoke(cli, ["task", "status", task_id])
        assert result.exit_code == 0
        assert "Target repo: fdp" in result.output

    def test_set_repo_idempotent(self, tmp_project, seeded_task):
        """Setting the same repo twice should be a no-op."""
        task_id, ml = seeded_task
        runner = CliRunner()

        runner.invoke(cli, ["task", "set-repo", task_id, "fdp"])
        mutations_before = len(ml.read_all())

        result = runner.invoke(cli, ["task", "set-repo", task_id, "fdp"])
        assert result.exit_code == 0
        assert "already targets repo 'fdp'" in result.output

        # No new mutation should have been appended
        assert len(ml.read_all()) == mutations_before

    def test_set_repo_nonexistent_task(self, tmp_project, seeded_task):
        """set-repo on nonexistent task should fail."""
        runner = CliRunner()

        result = runner.invoke(cli, ["task", "set-repo", "nonexist", "fdp"])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_set_repo_mutation_logged(self, tmp_project, seeded_task):
        """set-repo should write a task_updated mutation with target_repo."""
        task_id, ml = seeded_task
        runner = CliRunner()

        runner.invoke(cli, ["task", "set-repo", task_id, "fdp"])

        mutations = ml.read_all()
        update_mutations = [
            m
            for m in mutations
            if m["type"] == "task_updated" and m.get("task_id") == task_id
        ]
        assert len(update_mutations) == 1
        assert update_mutations[0]["data"]["target_repo"] == "fdp"

    def test_set_repo_overwrite(self, tmp_project, seeded_task):
        """set-repo should overwrite a previously set target_repo."""
        task_id, ml = seeded_task
        runner = CliRunner()

        runner.invoke(cli, ["task", "set-repo", task_id, "fdp"])
        result = runner.invoke(cli, ["task", "set-repo", task_id, "other-repo"])
        assert result.exit_code == 0
        assert "target repo set to 'other-repo'" in result.output

        ws = WorkState(tmp_project / "data" / "state.db", ml)
        task = ws.get_task(task_id)
        assert task["target_repo"] == "other-repo"


class TestTaskCreateTargetRepo:
    """Tests for `corc task create --target-repo`."""

    def test_create_with_target_repo(self, tmp_project):
        """task create --target-repo should store target_repo."""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "task",
                "create",
                "my-fdp-task",
                "--done-when",
                "tests pass",
                "--target-repo",
                "fdp",
            ],
        )
        assert result.exit_code == 0
        assert "[repo: fdp]" in result.output

        # Extract task_id from output like "Created task abc12345: ..."
        task_id = result.output.split("Created task ")[1].split(":")[0]

        # Verify in mutations
        ml = MutationLog(tmp_project / "data" / "mutations.jsonl")
        mutations = ml.read_all()
        created = [m for m in mutations if m["type"] == "task_created"]
        assert len(created) == 1
        assert created[0]["data"]["target_repo"] == "fdp"

    def test_create_without_target_repo(self, tmp_project):
        """task create without --target-repo should not include target_repo."""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "task",
                "create",
                "my-local-task",
                "--done-when",
                "tests pass",
            ],
        )
        assert result.exit_code == 0
        assert "[repo:" not in result.output

        # Verify in state
        ml = MutationLog(tmp_project / "data" / "mutations.jsonl")
        ws = WorkState(tmp_project / "data" / "state.db", ml)
        tasks = ws.list_tasks()
        assert len(tasks) == 1
        assert tasks[0]["target_repo"] is None

    def test_create_target_repo_shows_in_status(self, tmp_project):
        """task created with --target-repo should show it in status."""
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "task",
                "create",
                "fdp-task",
                "--done-when",
                "tests pass",
                "--target-repo",
                "fdp",
            ],
        )
        task_id = result.output.split("Created task ")[1].split(":")[0]

        result = runner.invoke(cli, ["task", "status", task_id])
        assert result.exit_code == 0
        assert "Target repo: fdp" in result.output
