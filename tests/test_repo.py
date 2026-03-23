"""Tests for multi-repo registration with per-repo settings.

Verifies:
- Adding repos with default and custom settings
- Listing all repos with their settings
- Showing full config for a single repo
- Removing repos
- Updating repo settings
- Validation of merge_policy, enforcement_level, protected_branches
- Duplicate repo detection
- Settings persistence via config save/reload
- CLI commands: corc repo add/list/show/remove/update
"""

from pathlib import Path

import pytest
import yaml

from corc.config import CorcConfig, load_config
from corc.repo import (
    RepoManager,
    RepoAlreadyExistsError,
    RepoNotFoundError,
    RepoValidationError,
    DEFAULT_MERGE_POLICY,
    DEFAULT_ENFORCEMENT_LEVEL,
    DEFAULT_PROTECTED_BRANCHES,
    VALID_MERGE_POLICIES,
    VALID_ENFORCEMENT_LEVELS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project directory with .corc/ and .git marker."""
    corc_dir = tmp_path / ".corc"
    corc_dir.mkdir()
    (tmp_path / ".git").mkdir()
    # Initialize required data directories for CLI tests
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "events").mkdir()
    (data_dir / "sessions").mkdir()
    (data_dir / "ratings").mkdir()
    (tmp_path / "knowledge").mkdir()
    return tmp_path


@pytest.fixture
def config(tmp_project):
    """CorcConfig backed by a tmp project."""
    config_path = tmp_project / ".corc" / "config.yaml"
    return CorcConfig(config_path=config_path)


@pytest.fixture
def mgr(config):
    """RepoManager instance backed by tmp config."""
    return RepoManager(config)


# ---------------------------------------------------------------------------
# RepoManager: add
# ---------------------------------------------------------------------------


class TestRepoAdd:
    """Adding repos with default and custom settings."""

    def test_add_with_defaults(self, mgr):
        result = mgr.add("my-app", "/code/my-app")
        assert result["path"] == "/code/my-app"
        assert result["merge_policy"] == DEFAULT_MERGE_POLICY
        assert result["protected_branches"] == DEFAULT_PROTECTED_BRANCHES
        assert result["enforcement_level"] == DEFAULT_ENFORCEMENT_LEVEL

    def test_add_with_custom_settings(self, mgr):
        result = mgr.add(
            "prod-app",
            "/code/prod",
            merge_policy="human-only",
            protected_branches=["main", "staging"],
            enforcement_level="relaxed",
        )
        assert result["path"] == "/code/prod"
        assert result["merge_policy"] == "human-only"
        assert result["protected_branches"] == ["main", "staging"]
        assert result["enforcement_level"] == "relaxed"

    def test_add_auto_merge_policy(self, mgr):
        result = mgr.add("tool", "/code/tool", merge_policy="auto")
        assert result["merge_policy"] == "auto"

    def test_add_human_only_merge_policy(self, mgr):
        result = mgr.add("tool", "/code/tool", merge_policy="human-only")
        assert result["merge_policy"] == "human-only"

    def test_add_strict_enforcement(self, mgr):
        result = mgr.add("tool", "/code/tool", enforcement_level="strict")
        assert result["enforcement_level"] == "strict"

    def test_add_relaxed_enforcement(self, mgr):
        result = mgr.add("tool", "/code/tool", enforcement_level="relaxed")
        assert result["enforcement_level"] == "relaxed"

    def test_add_multiple_protected_branches(self, mgr):
        result = mgr.add(
            "app", "/code/app", protected_branches=["main", "staging", "prod"]
        )
        assert result["protected_branches"] == ["main", "staging", "prod"]

    def test_add_duplicate_raises(self, mgr):
        mgr.add("app", "/code/app")
        with pytest.raises(RepoAlreadyExistsError, match="already exists"):
            mgr.add("app", "/code/other")

    def test_add_invalid_merge_policy(self, mgr):
        with pytest.raises(RepoValidationError, match="Invalid merge_policy"):
            mgr.add("app", "/code/app", merge_policy="invalid")

    def test_add_invalid_enforcement_level(self, mgr):
        with pytest.raises(RepoValidationError, match="Invalid enforcement_level"):
            mgr.add("app", "/code/app", enforcement_level="invalid")

    def test_add_empty_name_raises(self, mgr):
        with pytest.raises(RepoValidationError, match="non-empty"):
            mgr.add("", "/code/app")

    def test_add_empty_path_raises(self, mgr):
        with pytest.raises(RepoValidationError, match="non-empty"):
            mgr.add("app", "")

    def test_add_invalid_protected_branches_type(self, mgr):
        with pytest.raises(RepoValidationError, match="must be a list"):
            mgr.add("app", "/code/app", protected_branches="main")

    def test_add_invalid_protected_branch_empty_string(self, mgr):
        with pytest.raises(RepoValidationError, match="non-empty string"):
            mgr.add("app", "/code/app", protected_branches=["main", ""])


# ---------------------------------------------------------------------------
# RepoManager: get
# ---------------------------------------------------------------------------


class TestRepoGet:
    """Getting repo configuration."""

    def test_get_existing_repo(self, mgr):
        mgr.add("app", "/code/app", merge_policy="human-only")
        result = mgr.get("app")
        assert result["path"] == "/code/app"
        assert result["merge_policy"] == "human-only"
        assert result["enforcement_level"] == "strict"
        assert result["protected_branches"] == ["main"]

    def test_get_nonexistent_raises(self, mgr):
        with pytest.raises(RepoNotFoundError, match="not found"):
            mgr.get("nonexistent")

    def test_get_returns_copy(self, mgr):
        """Mutating the returned dict should not affect the stored config."""
        mgr.add("app", "/code/app")
        result = mgr.get("app")
        result["merge_policy"] = "TAMPERED"
        # Re-fetch should be unchanged
        assert mgr.get("app")["merge_policy"] == "auto"


# ---------------------------------------------------------------------------
# RepoManager: list
# ---------------------------------------------------------------------------


class TestRepoList:
    """Listing repos."""

    def test_list_empty(self, mgr):
        result = mgr.list_repos()
        assert result == []

    def test_list_one_repo(self, mgr):
        mgr.add("app", "/code/app")
        result = mgr.list_repos()
        assert len(result) == 1
        assert result[0]["name"] == "app"
        assert result[0]["path"] == "/code/app"
        assert result[0]["merge_policy"] == "auto"

    def test_list_multiple_repos(self, mgr):
        mgr.add("app1", "/code/app1", merge_policy="auto")
        mgr.add("app2", "/code/app2", merge_policy="human-only")
        mgr.add("app3", "/code/app3", enforcement_level="relaxed")
        result = mgr.list_repos()
        assert len(result) == 3
        names = {r["name"] for r in result}
        assert names == {"app1", "app2", "app3"}

    def test_list_includes_all_settings(self, mgr):
        mgr.add(
            "prod",
            "/code/prod",
            merge_policy="human-only",
            protected_branches=["main", "staging"],
            enforcement_level="strict",
        )
        result = mgr.list_repos()
        repo = result[0]
        assert repo["name"] == "prod"
        assert repo["path"] == "/code/prod"
        assert repo["merge_policy"] == "human-only"
        assert repo["protected_branches"] == ["main", "staging"]
        assert repo["enforcement_level"] == "strict"


# ---------------------------------------------------------------------------
# RepoManager: remove
# ---------------------------------------------------------------------------


class TestRepoRemove:
    """Removing repos."""

    def test_remove_existing(self, mgr):
        mgr.add("app", "/code/app")
        mgr.remove("app")
        assert mgr.list_repos() == []

    def test_remove_nonexistent_raises(self, mgr):
        with pytest.raises(RepoNotFoundError, match="not found"):
            mgr.remove("nonexistent")

    def test_remove_one_of_many(self, mgr):
        mgr.add("app1", "/code/app1")
        mgr.add("app2", "/code/app2")
        mgr.add("app3", "/code/app3")
        mgr.remove("app2")
        result = mgr.list_repos()
        names = {r["name"] for r in result}
        assert names == {"app1", "app3"}


# ---------------------------------------------------------------------------
# RepoManager: update
# ---------------------------------------------------------------------------


class TestRepoUpdate:
    """Updating repo settings."""

    def test_update_merge_policy(self, mgr):
        mgr.add("app", "/code/app", merge_policy="auto")
        result = mgr.update("app", merge_policy="human-only")
        assert result["merge_policy"] == "human-only"
        # Verify it persisted in the manager
        assert mgr.get("app")["merge_policy"] == "human-only"

    def test_update_enforcement_level(self, mgr):
        mgr.add("app", "/code/app", enforcement_level="strict")
        result = mgr.update("app", enforcement_level="relaxed")
        assert result["enforcement_level"] == "relaxed"

    def test_update_protected_branches(self, mgr):
        mgr.add("app", "/code/app", protected_branches=["main"])
        result = mgr.update("app", protected_branches=["main", "staging", "prod"])
        assert result["protected_branches"] == ["main", "staging", "prod"]

    def test_update_path(self, mgr):
        mgr.add("app", "/code/old-path")
        result = mgr.update("app", path="/code/new-path")
        assert result["path"] == "/code/new-path"

    def test_update_multiple_fields(self, mgr):
        mgr.add("app", "/code/app", merge_policy="auto", enforcement_level="strict")
        result = mgr.update(
            "app", merge_policy="human-only", enforcement_level="relaxed"
        )
        assert result["merge_policy"] == "human-only"
        assert result["enforcement_level"] == "relaxed"

    def test_update_nonexistent_raises(self, mgr):
        with pytest.raises(RepoNotFoundError, match="not found"):
            mgr.update("nonexistent", merge_policy="auto")

    def test_update_invalid_merge_policy(self, mgr):
        mgr.add("app", "/code/app")
        with pytest.raises(RepoValidationError, match="Invalid merge_policy"):
            mgr.update("app", merge_policy="invalid")

    def test_update_invalid_enforcement_level(self, mgr):
        mgr.add("app", "/code/app")
        with pytest.raises(RepoValidationError, match="Invalid enforcement_level"):
            mgr.update("app", enforcement_level="invalid")

    def test_update_preserves_unchanged_fields(self, mgr):
        mgr.add(
            "app",
            "/code/app",
            merge_policy="auto",
            protected_branches=["main"],
            enforcement_level="strict",
        )
        mgr.update("app", merge_policy="human-only")
        result = mgr.get("app")
        assert result["merge_policy"] == "human-only"
        assert result["path"] == "/code/app"
        assert result["protected_branches"] == ["main"]
        assert result["enforcement_level"] == "strict"


# ---------------------------------------------------------------------------
# Persistence: save and reload
# ---------------------------------------------------------------------------


class TestPersistence:
    """Settings persist across save/reload cycles."""

    def test_save_and_reload(self, tmp_project):
        """Repos survive a config save + reload cycle."""
        config_path = tmp_project / ".corc" / "config.yaml"
        cfg = CorcConfig(config_path=config_path)
        mgr = RepoManager(cfg)

        mgr.add(
            "prod",
            "/code/prod",
            merge_policy="human-only",
            protected_branches=["main", "staging"],
            enforcement_level="strict",
        )
        mgr.add("tool", "/code/tool", merge_policy="auto", enforcement_level="relaxed")
        cfg.save()

        # Reload from disk
        cfg2 = load_config(tmp_project)
        mgr2 = RepoManager(cfg2)

        repos = mgr2.list_repos()
        assert len(repos) == 2

        prod = mgr2.get("prod")
        assert prod["path"] == "/code/prod"
        assert prod["merge_policy"] == "human-only"
        assert prod["protected_branches"] == ["main", "staging"]
        assert prod["enforcement_level"] == "strict"

        tool = mgr2.get("tool")
        assert tool["path"] == "/code/tool"
        assert tool["merge_policy"] == "auto"
        assert tool["enforcement_level"] == "relaxed"

    def test_remove_persists(self, tmp_project):
        """Removed repos stay removed after reload."""
        config_path = tmp_project / ".corc" / "config.yaml"
        cfg = CorcConfig(config_path=config_path)
        mgr = RepoManager(cfg)

        mgr.add("app1", "/code/app1")
        mgr.add("app2", "/code/app2")
        mgr.remove("app1")
        cfg.save()

        cfg2 = load_config(tmp_project)
        mgr2 = RepoManager(cfg2)
        repos = mgr2.list_repos()
        assert len(repos) == 1
        assert repos[0]["name"] == "app2"

    def test_update_persists(self, tmp_project):
        """Updated settings persist after reload."""
        config_path = tmp_project / ".corc" / "config.yaml"
        cfg = CorcConfig(config_path=config_path)
        mgr = RepoManager(cfg)

        mgr.add("app", "/code/app", merge_policy="auto")
        mgr.update("app", merge_policy="human-only")
        cfg.save()

        cfg2 = load_config(tmp_project)
        mgr2 = RepoManager(cfg2)
        assert mgr2.get("app")["merge_policy"] == "human-only"

    def test_config_yaml_contains_repos(self, tmp_project):
        """Verify the raw YAML file contains repos data."""
        config_path = tmp_project / ".corc" / "config.yaml"
        cfg = CorcConfig(config_path=config_path)
        mgr = RepoManager(cfg)

        mgr.add("my-app", "/code/my-app", merge_policy="human-only")
        cfg.save()

        with open(config_path) as f:
            raw = yaml.safe_load(f)

        assert "repos" in raw
        assert "my-app" in raw["repos"]
        assert raw["repos"]["my-app"]["merge_policy"] == "human-only"
        assert raw["repos"]["my-app"]["path"] == "/code/my-app"


# ---------------------------------------------------------------------------
# CLI: corc repo add/list/show/remove/update
# ---------------------------------------------------------------------------


class TestRepoCLI:
    """Test CLI commands for repo management."""

    def test_repo_add(self, tmp_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()
        result = runner.invoke(cli, ["repo", "add", "my-app", "--path", "/code/my-app"])
        assert result.exit_code == 0
        assert "Registered repo 'my-app'" in result.output
        assert "merge_policy: auto" in result.output
        assert "enforcement_level: strict" in result.output

    def test_repo_add_with_options(self, tmp_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "repo",
                "add",
                "prod",
                "--path",
                "/code/prod",
                "--merge-policy",
                "human-only",
                "--protected-branches",
                "main,staging",
                "--enforcement-level",
                "relaxed",
            ],
        )
        assert result.exit_code == 0
        assert "Registered repo 'prod'" in result.output
        assert "merge_policy: human-only" in result.output
        assert "main, staging" in result.output
        assert "enforcement_level: relaxed" in result.output

    def test_repo_add_duplicate(self, tmp_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()
        runner.invoke(cli, ["repo", "add", "app", "--path", "/code/app"])
        result = runner.invoke(cli, ["repo", "add", "app", "--path", "/code/other"])
        assert result.exit_code != 0
        assert "already exists" in result.output

    def test_repo_list_empty(self, tmp_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()
        result = runner.invoke(cli, ["repo", "list"])
        assert result.exit_code == 0
        assert "No repos registered" in result.output

    def test_repo_list_with_repos(self, tmp_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()
        runner.invoke(cli, ["repo", "add", "app1", "--path", "/code/app1"])
        runner.invoke(
            cli,
            [
                "repo",
                "add",
                "app2",
                "--path",
                "/code/app2",
                "--merge-policy",
                "human-only",
            ],
        )

        result = runner.invoke(cli, ["repo", "list"])
        assert result.exit_code == 0
        assert "app1" in result.output
        assert "app2" in result.output
        assert "auto" in result.output
        assert "human-only" in result.output

    def test_repo_show(self, tmp_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()
        runner.invoke(
            cli,
            [
                "repo",
                "add",
                "prod",
                "--path",
                "/code/prod",
                "--merge-policy",
                "human-only",
                "--protected-branches",
                "main,staging",
            ],
        )

        result = runner.invoke(cli, ["repo", "show", "prod"])
        assert result.exit_code == 0
        assert "Repo: prod" in result.output
        assert "path: /code/prod" in result.output
        assert "merge_policy: human-only" in result.output
        assert "main, staging" in result.output
        assert "enforcement_level: strict" in result.output

    def test_repo_show_nonexistent(self, tmp_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()
        result = runner.invoke(cli, ["repo", "show", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_repo_remove(self, tmp_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()
        runner.invoke(cli, ["repo", "add", "app", "--path", "/code/app"])
        result = runner.invoke(cli, ["repo", "remove", "app"])
        assert result.exit_code == 0
        assert "Removed repo 'app'" in result.output

        # Verify it's gone
        result = runner.invoke(cli, ["repo", "list"])
        assert "No repos registered" in result.output

    def test_repo_remove_nonexistent(self, tmp_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()
        result = runner.invoke(cli, ["repo", "remove", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_repo_update(self, tmp_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()
        runner.invoke(cli, ["repo", "add", "app", "--path", "/code/app"])

        result = runner.invoke(
            cli, ["repo", "update", "app", "--merge-policy", "human-only"]
        )
        assert result.exit_code == 0
        assert "Updated repo 'app'" in result.output
        assert "merge_policy: human-only" in result.output

    def test_repo_update_multiple_fields(self, tmp_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()
        runner.invoke(cli, ["repo", "add", "app", "--path", "/code/app"])

        result = runner.invoke(
            cli,
            [
                "repo",
                "update",
                "app",
                "--merge-policy",
                "human-only",
                "--enforcement-level",
                "relaxed",
                "--protected-branches",
                "main,staging,prod",
            ],
        )
        assert result.exit_code == 0
        assert "merge_policy: human-only" in result.output
        assert "enforcement_level: relaxed" in result.output
        assert "main, staging, prod" in result.output

    def test_repo_update_no_options(self, tmp_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()
        runner.invoke(cli, ["repo", "add", "app", "--path", "/code/app"])

        result = runner.invoke(cli, ["repo", "update", "app"])
        assert result.exit_code != 0
        assert "Nothing to update" in result.output

    def test_repo_update_nonexistent(self, tmp_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["repo", "update", "nonexistent", "--merge-policy", "auto"]
        )
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_repo_add_persists_to_config(self, tmp_project, monkeypatch):
        """Verify that repo add writes to .corc/config.yaml."""
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()
        runner.invoke(
            cli,
            [
                "repo",
                "add",
                "my-app",
                "--path",
                "/code/my-app",
                "--merge-policy",
                "human-only",
            ],
        )

        # Read the config file directly
        config_path = tmp_project / ".corc" / "config.yaml"
        with open(config_path) as f:
            raw = yaml.safe_load(f)

        assert "repos" in raw
        assert "my-app" in raw["repos"]
        assert raw["repos"]["my-app"]["merge_policy"] == "human-only"

    def test_repo_full_lifecycle(self, tmp_project, monkeypatch):
        """Full CRUD lifecycle via CLI."""
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()

        # Add
        result = runner.invoke(cli, ["repo", "add", "app", "--path", "/code/app"])
        assert result.exit_code == 0

        # List
        result = runner.invoke(cli, ["repo", "list"])
        assert "app" in result.output

        # Show
        result = runner.invoke(cli, ["repo", "show", "app"])
        assert "Repo: app" in result.output

        # Update
        result = runner.invoke(
            cli, ["repo", "update", "app", "--merge-policy", "human-only"]
        )
        assert result.exit_code == 0

        # Verify update
        result = runner.invoke(cli, ["repo", "show", "app"])
        assert "merge_policy: human-only" in result.output

        # Remove
        result = runner.invoke(cli, ["repo", "remove", "app"])
        assert result.exit_code == 0

        # Verify removed
        result = runner.invoke(cli, ["repo", "list"])
        assert "No repos registered" in result.output


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module constants are correct."""

    def test_valid_merge_policies(self):
        assert "auto" in VALID_MERGE_POLICIES
        assert "human-only" in VALID_MERGE_POLICIES

    def test_valid_enforcement_levels(self):
        assert "strict" in VALID_ENFORCEMENT_LEVELS
        assert "relaxed" in VALID_ENFORCEMENT_LEVELS

    def test_defaults(self):
        assert DEFAULT_MERGE_POLICY == "auto"
        assert DEFAULT_ENFORCEMENT_LEVEL == "strict"
        assert DEFAULT_PROTECTED_BRANCHES == ["main"]
