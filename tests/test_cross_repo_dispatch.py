"""Tests for cross-repo dispatch (target_repo feature).

Tests cover:
- resolve_target_repo helper: fallback to project_root, resolution via RepoManager
- target_repo column in state.py: creation, update, migration
- CLI --target-repo flag: validation and context path resolution
- Integration: worktree creation in target repo
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from corc.config import CorcConfig
from corc.executor import resolve_target_repo
from corc.mutations import MutationLog
from corc.repo import RepoManager
from corc.state import WorkState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def state(tmp_path):
    """Create a fresh MutationLog + WorkState."""
    ml = MutationLog(tmp_path / "mutations.jsonl")
    ws = WorkState(tmp_path / "state.db", ml)
    return ml, ws


@pytest.fixture
def corc_project(tmp_path):
    """Create a minimal corc project structure."""
    root = tmp_path / "corc-project"
    root.mkdir()
    (root / "data").mkdir()
    (root / "data" / "events").mkdir()
    (root / "data" / "sessions").mkdir()
    (root / ".corc").mkdir()
    return root


@pytest.fixture
def target_repo_dir(tmp_path):
    """Create a minimal target repo directory."""
    repo = tmp_path / "target-repo"
    repo.mkdir()
    (repo / "src").mkdir()
    (repo / "src" / "main.py").write_text("# main module\n")
    return repo


@pytest.fixture
def config_with_repo(corc_project, target_repo_dir):
    """Create a CorcConfig with a registered target repo."""
    config_path = corc_project / ".corc" / "config.yaml"
    cfg = CorcConfig(
        data={
            "repos": {
                "my-target": {
                    "path": str(target_repo_dir),
                    "merge_policy": "auto",
                    "protected_branches": ["main"],
                    "enforcement_level": "strict",
                }
            }
        },
        config_path=config_path,
    )
    cfg.save()
    return cfg


# ---------------------------------------------------------------------------
# resolve_target_repo tests
# ---------------------------------------------------------------------------


class TestResolveTargetRepo:
    """Tests for the resolve_target_repo helper function."""

    def test_no_target_repo_returns_project_root(self, corc_project):
        """Tasks with no target_repo should return the corc project root."""
        task = {"id": "t1", "name": "test", "done_when": "done"}
        result = resolve_target_repo(task, corc_project)
        assert result == corc_project

    def test_empty_target_repo_returns_project_root(self, corc_project):
        """Tasks with empty target_repo should return the corc project root."""
        task = {"id": "t1", "name": "test", "done_when": "done", "target_repo": ""}
        result = resolve_target_repo(task, corc_project)
        assert result == corc_project

    def test_none_target_repo_returns_project_root(self, corc_project):
        """Tasks with None target_repo should return the corc project root."""
        task = {"id": "t1", "name": "test", "done_when": "done", "target_repo": None}
        result = resolve_target_repo(task, corc_project)
        assert result == corc_project

    def test_valid_target_repo_returns_repo_path(
        self, corc_project, target_repo_dir, config_with_repo
    ):
        """Tasks with a valid target_repo should resolve to the registered path."""
        task = {
            "id": "t1",
            "name": "test",
            "done_when": "done",
            "target_repo": "my-target",
        }
        with patch("corc.config.load_config", return_value=config_with_repo):
            result = resolve_target_repo(task, corc_project)
        assert result == target_repo_dir

    def test_unregistered_target_repo_raises(self, corc_project):
        """Tasks with an unregistered target_repo should raise ValueError."""
        task = {
            "id": "t1",
            "name": "test",
            "done_when": "done",
            "target_repo": "nonexistent-repo",
        }
        cfg = CorcConfig(data={"repos": {}})
        with patch("corc.config.load_config", return_value=cfg):
            with pytest.raises(ValueError, match="not registered"):
                resolve_target_repo(task, corc_project)


# ---------------------------------------------------------------------------
# State (SQLite) tests for target_repo column
# ---------------------------------------------------------------------------


class TestTargetRepoInState:
    """Tests for target_repo persistence in work state."""

    def test_task_created_without_target_repo(self, state):
        """Tasks created without target_repo should store None."""
        ml, ws = state
        ml.append(
            "task_created",
            {"id": "t1", "name": "no-target", "done_when": "done"},
            reason="test",
        )
        ws.refresh()

        task = ws.get_task("t1")
        assert task is not None
        assert task.get("target_repo") is None

    def test_task_created_with_target_repo(self, state):
        """Tasks created with target_repo should persist the value."""
        ml, ws = state
        ml.append(
            "task_created",
            {
                "id": "t1",
                "name": "cross-repo",
                "done_when": "done",
                "target_repo": "my-app",
            },
            reason="test",
        )
        ws.refresh()

        task = ws.get_task("t1")
        assert task is not None
        assert task["target_repo"] == "my-app"

    def test_task_updated_target_repo(self, state):
        """task_updated mutation should be able to set target_repo."""
        ml, ws = state
        ml.append(
            "task_created",
            {"id": "t1", "name": "task", "done_when": "done"},
            reason="test",
        )
        ws.refresh()
        assert ws.get_task("t1")["target_repo"] is None

        ml.append(
            "task_updated",
            {"target_repo": "new-repo"},
            reason="set target repo",
            task_id="t1",
        )
        ws.refresh()
        assert ws.get_task("t1")["target_repo"] == "new-repo"

    def test_rebuild_preserves_target_repo(self, state, tmp_path):
        """Rebuilding state from mutations should preserve target_repo."""
        ml, ws = state
        ml.append(
            "task_created",
            {
                "id": "t1",
                "name": "cross-repo",
                "done_when": "done",
                "target_repo": "production",
            },
            reason="test",
        )

        # Rebuild from scratch
        ws2 = WorkState(tmp_path / "state2.db", ml)
        task = ws2.get_task("t1")
        assert task["target_repo"] == "production"

    def test_migration_adds_target_repo_column(self, tmp_path):
        """Opening a DB with old schema should add target_repo column."""
        from tests.test_state import (
            OLD_SCHEMA,
            _get_column_names,
            _create_old_schema_db,
        )

        db_path = tmp_path / "state.db"
        _create_old_schema_db(db_path)

        old_columns = _get_column_names(db_path)
        assert "target_repo" not in old_columns

        ml = MutationLog(tmp_path / "mutations.jsonl")
        WorkState(db_path, ml)

        new_columns = _get_column_names(db_path)
        assert "target_repo" in new_columns

    def test_fresh_db_has_target_repo_column(self, tmp_path):
        """Fresh database should include the target_repo column."""
        from tests.test_state import _get_column_names

        ml = MutationLog(tmp_path / "mutations.jsonl")
        db_path = tmp_path / "state.db"
        WorkState(db_path, ml)

        columns = _get_column_names(db_path)
        assert "target_repo" in columns


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLITargetRepo:
    """Tests for --target-repo flag in 'corc task create'."""

    @pytest.fixture
    def cli_env(self, tmp_path, monkeypatch):
        """Set up CLI test environment."""
        from click.testing import CliRunner

        root = tmp_path / "project"
        root.mkdir()
        (root / "data").mkdir()
        (root / "data" / "events").mkdir()
        (root / "data" / "sessions").mkdir()
        (root / "knowledge").mkdir()
        (root / ".corc").mkdir()

        paths = {
            "root": root,
            "data_dir": root / "data",
            "mutations": root / "data" / "mutations.jsonl",
            "state_db": root / "data" / "state.db",
            "events_dir": root / "data" / "events",
            "sessions_dir": root / "data" / "sessions",
            "knowledge_dir": root / "knowledge",
            "knowledge_db": root / "data" / "knowledge.db",
            "corc_dir": root / ".corc",
            "ratings_dir": root / "data" / "ratings",
            "retry_outcomes": root / "data" / "retry_outcomes.jsonl",
        }
        monkeypatch.setattr("corc.cli.get_paths", lambda: paths)

        runner = CliRunner()
        return runner, root, paths

    def test_create_task_with_target_repo(self, cli_env, target_repo_dir):
        """Creating a task with --target-repo should store the value."""
        from corc.cli import cli

        runner, root, paths = cli_env

        # Register the target repo
        cfg = CorcConfig(
            data={
                "repos": {
                    "my-target": {
                        "path": str(target_repo_dir),
                        "merge_policy": "auto",
                        "protected_branches": ["main"],
                        "enforcement_level": "strict",
                    }
                }
            },
            config_path=root / ".corc" / "config.yaml",
        )
        cfg.save()

        with patch("corc.cli.load_config", return_value=cfg):
            result = runner.invoke(
                cli,
                [
                    "task",
                    "create",
                    "cross-repo-task",
                    "--done-when",
                    "tests pass",
                    "--target-repo",
                    "my-target",
                ],
            )

        assert result.exit_code == 0
        assert "[repo=my-target]" in result.output

        # Verify the task was stored with target_repo
        ml = MutationLog(paths["mutations"])
        ws = WorkState(paths["state_db"], ml)
        tasks = ws.list_tasks()
        assert len(tasks) == 1
        assert tasks[0]["target_repo"] == "my-target"

    def test_create_task_with_unregistered_target_repo(self, cli_env):
        """Creating a task with an unregistered --target-repo should fail."""
        from corc.cli import cli

        runner, root, paths = cli_env

        cfg = CorcConfig(data={"repos": {}}, config_path=root / ".corc" / "config.yaml")
        cfg.save()

        with patch("corc.cli.load_config", return_value=cfg):
            result = runner.invoke(
                cli,
                [
                    "task",
                    "create",
                    "bad-task",
                    "--done-when",
                    "tests pass",
                    "--target-repo",
                    "nonexistent",
                ],
            )

        assert result.exit_code != 0
        assert "not registered" in result.output

    def test_create_task_without_target_repo_defaults_to_none(self, cli_env):
        """Tasks created without --target-repo should have no target_repo."""
        from corc.cli import cli

        runner, root, paths = cli_env

        result = runner.invoke(
            cli,
            ["task", "create", "normal-task", "--done-when", "tests pass"],
        )

        assert result.exit_code == 0
        assert "[repo=" not in result.output

        ml = MutationLog(paths["mutations"])
        ws = WorkState(paths["state_db"], ml)
        tasks = ws.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].get("target_repo") is None

    def test_context_bundle_validated_against_target_repo(
        self, cli_env, target_repo_dir
    ):
        """Context bundle paths should be validated against target repo root."""
        from corc.cli import cli

        runner, root, paths = cli_env

        # Create a file in the target repo
        (target_repo_dir / "src" / "app.py").write_text("# app\n")

        cfg = CorcConfig(
            data={
                "repos": {
                    "my-target": {
                        "path": str(target_repo_dir),
                        "merge_policy": "auto",
                        "protected_branches": ["main"],
                        "enforcement_level": "strict",
                    }
                }
            },
            config_path=root / ".corc" / "config.yaml",
        )
        cfg.save()

        with patch("corc.cli.load_config", return_value=cfg):
            # This file exists in target repo
            result = runner.invoke(
                cli,
                [
                    "task",
                    "create",
                    "context-test",
                    "--done-when",
                    "tests pass",
                    "--target-repo",
                    "my-target",
                    "--context",
                    "src/app.py",
                ],
            )

        assert result.exit_code == 0
        assert "file not found" not in result.output.lower()

    def test_context_bundle_warns_missing_in_target_repo(
        self, cli_env, target_repo_dir
    ):
        """Missing context bundle paths in target repo should produce warnings."""
        from corc.cli import cli

        runner, root, paths = cli_env

        cfg = CorcConfig(
            data={
                "repos": {
                    "my-target": {
                        "path": str(target_repo_dir),
                        "merge_policy": "auto",
                        "protected_branches": ["main"],
                        "enforcement_level": "strict",
                    }
                }
            },
            config_path=root / ".corc" / "config.yaml",
        )
        cfg.save()

        with patch("corc.cli.load_config", return_value=cfg):
            # This file does NOT exist in target repo
            result = runner.invoke(
                cli,
                [
                    "task",
                    "create",
                    "missing-context",
                    "--done-when",
                    "tests pass",
                    "--target-repo",
                    "my-target",
                    "--context",
                    "nonexistent/file.py",
                ],
            )

        # Should still create (non-strict), but warn
        assert result.exit_code == 0
        assert "file not found" in result.output.lower()


# ---------------------------------------------------------------------------
# Integration test: worktree creation in target repo
# ---------------------------------------------------------------------------


class TestCrossRepoWorktree:
    """Integration test verifying worktrees are created in the target repo."""

    @pytest.fixture
    def git_target_repo(self, tmp_path):
        """Create a real git repo to serve as the target."""
        repo = tmp_path / "git-target"
        repo.mkdir()
        subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=str(repo),
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=str(repo),
            capture_output=True,
        )
        # Create initial commit
        (repo / "README.md").write_text("# Target Repo\n")
        subprocess.run(
            ["git", "add", "."], cwd=str(repo), capture_output=True, check=True
        )
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=str(repo),
            capture_output=True,
            check=True,
        )
        return repo

    def test_worktree_created_in_target_repo(self, git_target_repo, corc_project):
        """Worktrees for cross-repo tasks should be created in the target repo."""
        from corc.worktree import create_worktree, remove_worktree

        # Create worktree in the target repo (not corc project)
        worktree_path, branch_name = create_worktree(
            git_target_repo, "cross-task", attempt=1
        )

        try:
            # Worktree should be under the target repo's .corc/worktrees
            assert str(worktree_path).startswith(str(git_target_repo))
            assert worktree_path.exists()
            # The README from the target repo should be present
            assert (worktree_path / "README.md").exists()
            assert (worktree_path / "README.md").read_text() == "# Target Repo\n"
            # Branch should be corc-prefixed
            assert branch_name.startswith("corc/")
        finally:
            remove_worktree(git_target_repo, worktree_path)

    def test_resolve_target_repo_used_in_dispatch_flow(
        self, corc_project, git_target_repo
    ):
        """End-to-end: resolve_target_repo returns target repo for worktree creation."""
        task = {
            "id": "t1",
            "name": "cross-repo-task",
            "done_when": "done",
            "target_repo": "my-git-target",
        }

        cfg = CorcConfig(
            data={
                "repos": {
                    "my-git-target": {
                        "path": str(git_target_repo),
                        "merge_policy": "auto",
                        "protected_branches": ["main"],
                        "enforcement_level": "strict",
                    }
                }
            }
        )

        with patch("corc.config.load_config", return_value=cfg):
            repo_root = resolve_target_repo(task, corc_project)

        assert repo_root == git_target_repo
        # Verify we can actually create a worktree here
        from corc.worktree import create_worktree, remove_worktree

        wt_path, branch = create_worktree(repo_root, "t1", attempt=1)
        try:
            assert wt_path.exists()
            assert str(wt_path).startswith(str(git_target_repo))
        finally:
            remove_worktree(repo_root, wt_path)
