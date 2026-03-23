"""Tests for registering corc and fdp repos with correct per-repo settings.

Task: register-corc-and-fdp-repos
Done when:
  - Both repos registered in .corc/config.yaml
  - corc: path set, auto-merge, relaxed enforcement
  - fdp: path set, human-only merge, strict enforcement, protected_branches=[main]
  - Enforcement hooks auto-generated for both
  - corc repo list shows both repos with settings
"""

import json
from pathlib import Path

import pytest
import yaml

from corc.config import CorcConfig, load_config
from corc.hook_gen import sync_hooks, generate_settings
from corc.repo import RepoManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project directory with .corc/ and .git marker."""
    corc_dir = tmp_path / ".corc"
    corc_dir.mkdir()
    (tmp_path / ".git").mkdir()
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


@pytest.fixture
def corc_repo_path(tmp_path):
    """Create a temporary directory simulating the corc repo."""
    repo = tmp_path / "corc"
    repo.mkdir()
    (repo / ".git").mkdir()
    return str(repo)


@pytest.fixture
def fdp_repo_path(tmp_path):
    """Create a temporary directory simulating the fdp repo."""
    repo = tmp_path / "fdp"
    repo.mkdir()
    (repo / ".git").mkdir()
    return str(repo)


# ---------------------------------------------------------------------------
# Registration: corc repo settings
# ---------------------------------------------------------------------------


class TestCorcRepoRegistration:
    """Verify corc repo: auto-merge, relaxed enforcement."""

    def test_corc_registered_with_correct_merge_policy(self, mgr, corc_repo_path):
        mgr.add(
            "corc", corc_repo_path, merge_policy="auto", enforcement_level="relaxed"
        )
        repo = mgr.get("corc")
        assert repo["merge_policy"] == "auto"

    def test_corc_registered_with_relaxed_enforcement(self, mgr, corc_repo_path):
        mgr.add(
            "corc", corc_repo_path, merge_policy="auto", enforcement_level="relaxed"
        )
        repo = mgr.get("corc")
        assert repo["enforcement_level"] == "relaxed"

    def test_corc_path_set(self, mgr, corc_repo_path):
        mgr.add(
            "corc", corc_repo_path, merge_policy="auto", enforcement_level="relaxed"
        )
        repo = mgr.get("corc")
        assert repo["path"] == corc_repo_path

    def test_corc_default_protected_branches(self, mgr, corc_repo_path):
        mgr.add(
            "corc", corc_repo_path, merge_policy="auto", enforcement_level="relaxed"
        )
        repo = mgr.get("corc")
        assert repo["protected_branches"] == ["main"]


# ---------------------------------------------------------------------------
# Registration: fdp repo settings
# ---------------------------------------------------------------------------


class TestFdpRepoRegistration:
    """Verify fdp repo: human-only merge, strict enforcement, protected_branches=[main]."""

    def test_fdp_registered_with_human_only_merge(self, mgr, fdp_repo_path):
        mgr.add(
            "fdp",
            fdp_repo_path,
            merge_policy="human-only",
            protected_branches=["main"],
            enforcement_level="strict",
        )
        repo = mgr.get("fdp")
        assert repo["merge_policy"] == "human-only"

    def test_fdp_registered_with_strict_enforcement(self, mgr, fdp_repo_path):
        mgr.add(
            "fdp",
            fdp_repo_path,
            merge_policy="human-only",
            protected_branches=["main"],
            enforcement_level="strict",
        )
        repo = mgr.get("fdp")
        assert repo["enforcement_level"] == "strict"

    def test_fdp_protected_branches_main(self, mgr, fdp_repo_path):
        mgr.add(
            "fdp",
            fdp_repo_path,
            merge_policy="human-only",
            protected_branches=["main"],
            enforcement_level="strict",
        )
        repo = mgr.get("fdp")
        assert repo["protected_branches"] == ["main"]

    def test_fdp_path_set(self, mgr, fdp_repo_path):
        mgr.add(
            "fdp",
            fdp_repo_path,
            merge_policy="human-only",
            protected_branches=["main"],
            enforcement_level="strict",
        )
        repo = mgr.get("fdp")
        assert repo["path"] == fdp_repo_path


# ---------------------------------------------------------------------------
# Both repos registered together
# ---------------------------------------------------------------------------


class TestBothReposRegistered:
    """Verify both repos coexist with correct settings."""

    def _register_both(self, mgr, corc_path, fdp_path):
        mgr.add("corc", corc_path, merge_policy="auto", enforcement_level="relaxed")
        mgr.add(
            "fdp",
            fdp_path,
            merge_policy="human-only",
            protected_branches=["main"],
            enforcement_level="strict",
        )

    def test_both_listed(self, mgr, corc_repo_path, fdp_repo_path):
        self._register_both(mgr, corc_repo_path, fdp_repo_path)
        repos = mgr.list_repos()
        assert len(repos) == 2
        names = {r["name"] for r in repos}
        assert names == {"corc", "fdp"}

    def test_corc_settings_in_list(self, mgr, corc_repo_path, fdp_repo_path):
        self._register_both(mgr, corc_repo_path, fdp_repo_path)
        repos = mgr.list_repos()
        corc = next(r for r in repos if r["name"] == "corc")
        assert corc["merge_policy"] == "auto"
        assert corc["enforcement_level"] == "relaxed"
        assert corc["path"] == corc_repo_path

    def test_fdp_settings_in_list(self, mgr, corc_repo_path, fdp_repo_path):
        self._register_both(mgr, corc_repo_path, fdp_repo_path)
        repos = mgr.list_repos()
        fdp = next(r for r in repos if r["name"] == "fdp")
        assert fdp["merge_policy"] == "human-only"
        assert fdp["enforcement_level"] == "strict"
        assert fdp["protected_branches"] == ["main"]
        assert fdp["path"] == fdp_repo_path

    def test_persist_and_reload(self, tmp_project, corc_repo_path, fdp_repo_path):
        """Both repos survive save/reload cycle."""
        config_path = tmp_project / ".corc" / "config.yaml"
        cfg = CorcConfig(config_path=config_path)
        mgr = RepoManager(cfg)

        self._register_both(mgr, corc_repo_path, fdp_repo_path)
        cfg.save()

        # Reload from disk
        cfg2 = load_config(tmp_project)
        mgr2 = RepoManager(cfg2)

        repos = mgr2.list_repos()
        assert len(repos) == 2

        corc = mgr2.get("corc")
        assert corc["merge_policy"] == "auto"
        assert corc["enforcement_level"] == "relaxed"
        assert corc["path"] == corc_repo_path

        fdp = mgr2.get("fdp")
        assert fdp["merge_policy"] == "human-only"
        assert fdp["enforcement_level"] == "strict"
        assert fdp["protected_branches"] == ["main"]

    def test_config_yaml_contains_both(
        self, tmp_project, corc_repo_path, fdp_repo_path
    ):
        """Verify raw YAML file has both repos."""
        config_path = tmp_project / ".corc" / "config.yaml"
        cfg = CorcConfig(config_path=config_path)
        mgr = RepoManager(cfg)

        self._register_both(mgr, corc_repo_path, fdp_repo_path)
        cfg.save()

        with open(config_path) as f:
            raw = yaml.safe_load(f)

        assert "repos" in raw
        assert "corc" in raw["repos"]
        assert "fdp" in raw["repos"]
        assert raw["repos"]["corc"]["merge_policy"] == "auto"
        assert raw["repos"]["corc"]["enforcement_level"] == "relaxed"
        assert raw["repos"]["fdp"]["merge_policy"] == "human-only"
        assert raw["repos"]["fdp"]["enforcement_level"] == "strict"
        assert raw["repos"]["fdp"]["protected_branches"] == ["main"]


# ---------------------------------------------------------------------------
# Hook generation for both repos
# ---------------------------------------------------------------------------


class TestHookGeneration:
    """Verify enforcement hooks are correctly generated for both repos."""

    def test_corc_relaxed_no_enforce_policy_script(self, corc_repo_path):
        """Relaxed enforcement should NOT generate enforce-policy.sh."""
        repo_config = {
            "enforcement_level": "relaxed",
            "protected_branches": ["main"],
            "merge_policy": "auto",
        }
        written = sync_hooks(corc_repo_path, repo_config)

        filenames = [Path(f).name for f in written]
        assert "settings.json" in filenames
        assert "format-python.sh" in filenames
        assert "enforce-policy.sh" not in filenames

    def test_corc_relaxed_settings_no_pretooluse(self, corc_repo_path):
        """Relaxed enforcement settings should not have PreToolUse hooks."""
        settings = generate_settings("relaxed", ["main"], "auto")
        assert "PreToolUse" not in settings.get("hooks", {})
        assert "PostToolUse" in settings["hooks"]

    def test_fdp_strict_has_enforce_policy_script(self, fdp_repo_path):
        """Strict enforcement should generate enforce-policy.sh."""
        repo_config = {
            "enforcement_level": "strict",
            "protected_branches": ["main"],
            "merge_policy": "human-only",
        }
        written = sync_hooks(fdp_repo_path, repo_config)

        filenames = [Path(f).name for f in written]
        assert "settings.json" in filenames
        assert "format-python.sh" in filenames
        assert "enforce-policy.sh" in filenames

    def test_fdp_strict_settings_has_pretooluse(self, fdp_repo_path):
        """Strict enforcement settings should have PreToolUse hooks."""
        settings = generate_settings("strict", ["main"], "human-only")
        assert "PreToolUse" in settings["hooks"]
        assert "PostToolUse" in settings["hooks"]

    def test_fdp_enforce_policy_script_blocks_main(self, fdp_repo_path):
        """Enforce-policy.sh should reference the 'main' branch."""
        repo_config = {
            "enforcement_level": "strict",
            "protected_branches": ["main"],
            "merge_policy": "human-only",
        }
        sync_hooks(fdp_repo_path, repo_config)

        enforce_path = Path(fdp_repo_path) / ".claude" / "hooks" / "enforce-policy.sh"
        assert enforce_path.exists()
        content = enforce_path.read_text()
        assert '"main"' in content
        assert "BLOCKED" in content

    def test_fdp_enforce_policy_blocks_auto_merge(self, fdp_repo_path):
        """Enforce-policy.sh should block gh pr merge --auto."""
        repo_config = {
            "enforcement_level": "strict",
            "protected_branches": ["main"],
            "merge_policy": "human-only",
        }
        sync_hooks(fdp_repo_path, repo_config)

        enforce_path = Path(fdp_repo_path) / ".claude" / "hooks" / "enforce-policy.sh"
        content = enforce_path.read_text()
        assert "gh" in content and "merge" in content and "--auto" in content

    def test_corc_has_format_hook(self, corc_repo_path):
        """Both repos get the format-python.sh PostToolUse hook."""
        repo_config = {
            "enforcement_level": "relaxed",
            "protected_branches": ["main"],
            "merge_policy": "auto",
        }
        sync_hooks(corc_repo_path, repo_config)

        format_path = Path(corc_repo_path) / ".claude" / "hooks" / "format-python.sh"
        assert format_path.exists()
        assert format_path.stat().st_mode & 0o111  # executable

    def test_fdp_has_format_hook(self, fdp_repo_path):
        """Both repos get the format-python.sh PostToolUse hook."""
        repo_config = {
            "enforcement_level": "strict",
            "protected_branches": ["main"],
            "merge_policy": "human-only",
        }
        sync_hooks(fdp_repo_path, repo_config)

        format_path = Path(fdp_repo_path) / ".claude" / "hooks" / "format-python.sh"
        assert format_path.exists()
        assert format_path.stat().st_mode & 0o111  # executable

    def test_fdp_settings_json_structure(self, fdp_repo_path):
        """Settings.json should have correct structure for strict enforcement."""
        repo_config = {
            "enforcement_level": "strict",
            "protected_branches": ["main"],
            "merge_policy": "human-only",
        }
        sync_hooks(fdp_repo_path, repo_config)

        settings_path = Path(fdp_repo_path) / ".claude" / "settings.json"
        assert settings_path.exists()
        settings = json.loads(settings_path.read_text())
        assert "hooks" in settings
        assert "PreToolUse" in settings["hooks"]
        assert "PostToolUse" in settings["hooks"]

    def test_corc_settings_json_structure(self, corc_repo_path):
        """Settings.json should have correct structure for relaxed enforcement."""
        repo_config = {
            "enforcement_level": "relaxed",
            "protected_branches": ["main"],
            "merge_policy": "auto",
        }
        sync_hooks(corc_repo_path, repo_config)

        settings_path = Path(corc_repo_path) / ".claude" / "settings.json"
        assert settings_path.exists()
        settings = json.loads(settings_path.read_text())
        assert "hooks" in settings
        assert "PreToolUse" not in settings["hooks"]
        assert "PostToolUse" in settings["hooks"]


# ---------------------------------------------------------------------------
# CLI: corc repo list shows both repos
# ---------------------------------------------------------------------------


class TestCLIRepoList:
    """Verify corc repo list output shows both repos with correct settings."""

    def test_repo_list_shows_both(
        self, tmp_project, monkeypatch, corc_repo_path, fdp_repo_path
    ):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()

        # Register corc
        result = runner.invoke(
            cli,
            [
                "repo",
                "add",
                "corc",
                "--path",
                corc_repo_path,
                "--merge-policy",
                "auto",
                "--enforcement-level",
                "relaxed",
            ],
        )
        assert result.exit_code == 0, f"Failed to add corc: {result.output}"

        # Register fdp
        result = runner.invoke(
            cli,
            [
                "repo",
                "add",
                "fdp",
                "--path",
                fdp_repo_path,
                "--merge-policy",
                "human-only",
                "--protected-branches",
                "main",
                "--enforcement-level",
                "strict",
            ],
        )
        assert result.exit_code == 0, f"Failed to add fdp: {result.output}"

        # List repos
        result = runner.invoke(cli, ["repo", "list"])
        assert result.exit_code == 0
        assert "corc" in result.output
        assert "fdp" in result.output
        assert "auto" in result.output
        assert "human-only" in result.output
        assert "relaxed" in result.output
        assert "strict" in result.output

    def test_repo_list_corc_details(
        self, tmp_project, monkeypatch, corc_repo_path, fdp_repo_path
    ):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()

        runner.invoke(
            cli,
            [
                "repo",
                "add",
                "corc",
                "--path",
                corc_repo_path,
                "--merge-policy",
                "auto",
                "--enforcement-level",
                "relaxed",
            ],
        )
        runner.invoke(
            cli,
            [
                "repo",
                "add",
                "fdp",
                "--path",
                fdp_repo_path,
                "--merge-policy",
                "human-only",
                "--protected-branches",
                "main",
                "--enforcement-level",
                "strict",
            ],
        )

        # Show corc details
        result = runner.invoke(cli, ["repo", "show", "corc"])
        assert result.exit_code == 0
        assert "merge_policy: auto" in result.output
        assert "enforcement_level: relaxed" in result.output

    def test_repo_list_fdp_details(
        self, tmp_project, monkeypatch, corc_repo_path, fdp_repo_path
    ):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()

        runner.invoke(
            cli,
            [
                "repo",
                "add",
                "corc",
                "--path",
                corc_repo_path,
                "--merge-policy",
                "auto",
                "--enforcement-level",
                "relaxed",
            ],
        )
        runner.invoke(
            cli,
            [
                "repo",
                "add",
                "fdp",
                "--path",
                fdp_repo_path,
                "--merge-policy",
                "human-only",
                "--protected-branches",
                "main",
                "--enforcement-level",
                "strict",
            ],
        )

        # Show fdp details
        result = runner.invoke(cli, ["repo", "show", "fdp"])
        assert result.exit_code == 0
        assert "merge_policy: human-only" in result.output
        assert "enforcement_level: strict" in result.output
        assert "main" in result.output

    def test_hook_files_generated_via_cli(
        self, tmp_project, monkeypatch, corc_repo_path, fdp_repo_path
    ):
        """CLI repo add should auto-generate hook files."""
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()

        result = runner.invoke(
            cli,
            [
                "repo",
                "add",
                "corc",
                "--path",
                corc_repo_path,
                "--merge-policy",
                "auto",
                "--enforcement-level",
                "relaxed",
            ],
        )
        assert "Generated" in result.output
        assert "hook file" in result.output

        result = runner.invoke(
            cli,
            [
                "repo",
                "add",
                "fdp",
                "--path",
                fdp_repo_path,
                "--merge-policy",
                "human-only",
                "--protected-branches",
                "main",
                "--enforcement-level",
                "strict",
            ],
        )
        assert "Generated" in result.output
        assert "hook file" in result.output

        # Verify files on disk
        corc_settings = Path(corc_repo_path) / ".claude" / "settings.json"
        fdp_settings = Path(fdp_repo_path) / ".claude" / "settings.json"
        assert corc_settings.exists()
        assert fdp_settings.exists()

        # fdp (strict) should have enforce-policy.sh
        fdp_enforce = Path(fdp_repo_path) / ".claude" / "hooks" / "enforce-policy.sh"
        assert fdp_enforce.exists()

        # corc (relaxed) should NOT have enforce-policy.sh
        corc_enforce = Path(corc_repo_path) / ".claude" / "hooks" / "enforce-policy.sh"
        assert not corc_enforce.exists()
