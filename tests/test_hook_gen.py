"""Tests for auto-generation of Claude Code hooks from repo config.

Verifies:
- generate_settings() produces correct hook structure for strict vs relaxed
- generate_enforce_policy_script() embeds protected branches correctly
- generate_format_script() produces valid formatting hook
- sync_hooks() writes files to correct locations with correct permissions
- Hook generation is deterministic (same input → same output)
- Hook generation is idempotent (running twice produces identical files)
- CLI integration: repo add/update automatically generate hooks
"""

import json
import os
import stat
from pathlib import Path

import pytest

from corc.hook_gen import (
    generate_enforce_policy_script,
    generate_format_script,
    generate_settings,
    sync_hooks,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_repo(tmp_path):
    """Create a minimal repo directory structure."""
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
def strict_config():
    """Repo config for strict enforcement."""
    return {
        "path": "/code/prod",
        "merge_policy": "human-only",
        "protected_branches": ["main", "staging"],
        "enforcement_level": "strict",
    }


@pytest.fixture
def relaxed_config():
    """Repo config for relaxed enforcement."""
    return {
        "path": "/code/tool",
        "merge_policy": "auto",
        "protected_branches": ["main"],
        "enforcement_level": "relaxed",
    }


# ---------------------------------------------------------------------------
# generate_settings
# ---------------------------------------------------------------------------


class TestGenerateSettings:
    """Settings dict generation for strict and relaxed repos."""

    def test_strict_has_pre_tool_use(self):
        settings = generate_settings("strict", ["main"])
        assert "PreToolUse" in settings["hooks"]

    def test_strict_has_post_tool_use(self):
        settings = generate_settings("strict", ["main"])
        assert "PostToolUse" in settings["hooks"]

    def test_strict_pre_tool_use_matches_bash(self):
        settings = generate_settings("strict", ["main"])
        pre = settings["hooks"]["PreToolUse"]
        assert len(pre) == 1
        assert pre[0]["matcher"] == "Bash"

    def test_strict_pre_tool_use_runs_enforce_policy(self):
        settings = generate_settings("strict", ["main"])
        pre = settings["hooks"]["PreToolUse"]
        hook = pre[0]["hooks"][0]
        assert hook["type"] == "command"
        assert hook["command"] == ".claude/hooks/enforce-policy.sh"

    def test_relaxed_has_no_pre_tool_use(self):
        settings = generate_settings("relaxed", ["main"])
        assert "PreToolUse" not in settings["hooks"]

    def test_relaxed_has_post_tool_use(self):
        settings = generate_settings("relaxed", ["main"])
        assert "PostToolUse" in settings["hooks"]

    def test_post_tool_use_matches_write_edit(self):
        settings = generate_settings("strict", ["main"])
        post = settings["hooks"]["PostToolUse"]
        assert len(post) == 1
        assert post[0]["matcher"] == "Write|Edit"

    def test_post_tool_use_runs_format_python(self):
        settings = generate_settings("strict", ["main"])
        post = settings["hooks"]["PostToolUse"]
        hook = post[0]["hooks"][0]
        assert hook["type"] == "command"
        assert hook["command"] == ".claude/hooks/format-python.sh"

    def test_relaxed_post_tool_use_format_python(self):
        settings = generate_settings("relaxed", ["main"])
        post = settings["hooks"]["PostToolUse"]
        hook = post[0]["hooks"][0]
        assert hook["command"] == ".claude/hooks/format-python.sh"

    def test_default_branches_when_none(self):
        """Should use ['main'] when no branches specified."""
        settings = generate_settings("strict")
        assert "PreToolUse" in settings["hooks"]

    def test_deterministic_strict(self):
        """Same input always produces the same output."""
        s1 = generate_settings("strict", ["main", "staging"])
        s2 = generate_settings("strict", ["main", "staging"])
        assert s1 == s2

    def test_deterministic_relaxed(self):
        s1 = generate_settings("relaxed", ["main"])
        s2 = generate_settings("relaxed", ["main"])
        assert s1 == s2

    def test_strict_vs_relaxed_differ(self):
        strict = generate_settings("strict", ["main"])
        relaxed = generate_settings("relaxed", ["main"])
        assert strict != relaxed
        assert "PreToolUse" in strict["hooks"]
        assert "PreToolUse" not in relaxed["hooks"]


# ---------------------------------------------------------------------------
# generate_enforce_policy_script
# ---------------------------------------------------------------------------


class TestGenerateEnforcePolicyScript:
    """Enforce-policy.sh script generation."""

    def test_contains_shebang(self):
        script = generate_enforce_policy_script(["main"])
        assert script.startswith("#!/usr/bin/env bash\n")

    def test_contains_set_euo_pipefail(self):
        script = generate_enforce_policy_script(["main"])
        assert "set -euo pipefail" in script

    def test_contains_protected_branch(self):
        script = generate_enforce_policy_script(["main"])
        assert '"main"' in script

    def test_contains_multiple_branches(self):
        script = generate_enforce_policy_script(["main", "staging", "prod"])
        assert '"main"' in script
        assert '"prod"' in script
        assert '"staging"' in script

    def test_branches_are_sorted(self):
        """Branches in the script should be sorted for determinism."""
        script = generate_enforce_policy_script(["staging", "main", "prod"])
        # Find the PROTECTED_BRANCHES line
        for line in script.splitlines():
            if "PROTECTED_BRANCHES=" in line:
                assert '"main" "prod" "staging"' in line
                break
        else:
            pytest.fail("PROTECTED_BRANCHES line not found")

    def test_blocks_git_push(self):
        script = generate_enforce_policy_script(["main"])
        assert "git" in script
        assert "push" in script
        assert "BLOCKED" in script

    def test_blocks_auto_merge(self):
        script = generate_enforce_policy_script(["main"])
        assert "gh" in script
        assert "pr" in script
        assert "merge" in script
        assert "--auto" in script

    def test_exits_with_code_2_on_block(self):
        script = generate_enforce_policy_script(["main"])
        assert "exit 2" in script

    def test_exits_with_code_0_on_allow(self):
        script = generate_enforce_policy_script(["main"])
        assert "exit 0" in script

    def test_reads_json_from_stdin(self):
        script = generate_enforce_policy_script(["main"])
        assert "INPUT=$(cat)" in script
        assert "jq" in script
        assert ".tool_input.command" in script

    def test_deterministic(self):
        s1 = generate_enforce_policy_script(["main", "staging"])
        s2 = generate_enforce_policy_script(["main", "staging"])
        assert s1 == s2

    def test_deterministic_regardless_of_input_order(self):
        """Branch order in input should not affect output (sorted)."""
        s1 = generate_enforce_policy_script(["staging", "main"])
        s2 = generate_enforce_policy_script(["main", "staging"])
        assert s1 == s2

    def test_generated_by_corc_comment(self):
        script = generate_enforce_policy_script(["main"])
        assert "Generated by corc" in script


# ---------------------------------------------------------------------------
# generate_format_script
# ---------------------------------------------------------------------------


class TestGenerateFormatScript:
    """Format-python.sh script generation."""

    def test_contains_shebang(self):
        script = generate_format_script()
        assert script.startswith("#!/usr/bin/env bash\n")

    def test_contains_ruff_format(self):
        script = generate_format_script()
        assert "ruff format" in script

    def test_checks_python_extension(self):
        script = generate_format_script()
        assert "*.py" in script

    def test_reads_file_path_from_json(self):
        script = generate_format_script()
        assert "file_path" in script
        assert "jq" in script

    def test_exits_0(self):
        script = generate_format_script()
        assert "exit 0" in script

    def test_deterministic(self):
        s1 = generate_format_script()
        s2 = generate_format_script()
        assert s1 == s2

    def test_generated_by_corc_comment(self):
        script = generate_format_script()
        assert "Generated by corc" in script


# ---------------------------------------------------------------------------
# sync_hooks
# ---------------------------------------------------------------------------


class TestSyncHooks:
    """Writing hooks to disk."""

    def test_strict_creates_settings_json(self, tmp_repo, strict_config):
        strict_config["path"] = str(tmp_repo)
        sync_hooks(str(tmp_repo), strict_config)
        settings_path = tmp_repo / ".claude" / "settings.json"
        assert settings_path.exists()

    def test_strict_creates_enforce_policy_sh(self, tmp_repo, strict_config):
        strict_config["path"] = str(tmp_repo)
        sync_hooks(str(tmp_repo), strict_config)
        script_path = tmp_repo / ".claude" / "hooks" / "enforce-policy.sh"
        assert script_path.exists()

    def test_strict_creates_format_python_sh(self, tmp_repo, strict_config):
        strict_config["path"] = str(tmp_repo)
        sync_hooks(str(tmp_repo), strict_config)
        script_path = tmp_repo / ".claude" / "hooks" / "format-python.sh"
        assert script_path.exists()

    def test_strict_returns_three_files(self, tmp_repo, strict_config):
        strict_config["path"] = str(tmp_repo)
        written = sync_hooks(str(tmp_repo), strict_config)
        assert len(written) == 3

    def test_relaxed_creates_settings_json(self, tmp_repo, relaxed_config):
        relaxed_config["path"] = str(tmp_repo)
        sync_hooks(str(tmp_repo), relaxed_config)
        settings_path = tmp_repo / ".claude" / "settings.json"
        assert settings_path.exists()

    def test_relaxed_creates_format_python_sh(self, tmp_repo, relaxed_config):
        relaxed_config["path"] = str(tmp_repo)
        sync_hooks(str(tmp_repo), relaxed_config)
        script_path = tmp_repo / ".claude" / "hooks" / "format-python.sh"
        assert script_path.exists()

    def test_relaxed_does_not_create_enforce_policy_sh(self, tmp_repo, relaxed_config):
        relaxed_config["path"] = str(tmp_repo)
        sync_hooks(str(tmp_repo), relaxed_config)
        script_path = tmp_repo / ".claude" / "hooks" / "enforce-policy.sh"
        assert not script_path.exists()

    def test_relaxed_returns_two_files(self, tmp_repo, relaxed_config):
        relaxed_config["path"] = str(tmp_repo)
        written = sync_hooks(str(tmp_repo), relaxed_config)
        assert len(written) == 2

    def test_settings_json_is_valid_json(self, tmp_repo, strict_config):
        strict_config["path"] = str(tmp_repo)
        sync_hooks(str(tmp_repo), strict_config)
        settings_path = tmp_repo / ".claude" / "settings.json"
        content = json.loads(settings_path.read_text())
        assert "hooks" in content

    def test_strict_settings_json_has_pre_tool_use(self, tmp_repo, strict_config):
        strict_config["path"] = str(tmp_repo)
        sync_hooks(str(tmp_repo), strict_config)
        settings_path = tmp_repo / ".claude" / "settings.json"
        content = json.loads(settings_path.read_text())
        assert "PreToolUse" in content["hooks"]

    def test_relaxed_settings_json_no_pre_tool_use(self, tmp_repo, relaxed_config):
        relaxed_config["path"] = str(tmp_repo)
        sync_hooks(str(tmp_repo), relaxed_config)
        settings_path = tmp_repo / ".claude" / "settings.json"
        content = json.loads(settings_path.read_text())
        assert "PreToolUse" not in content["hooks"]

    def test_enforce_script_has_correct_branches(self, tmp_repo, strict_config):
        strict_config["path"] = str(tmp_repo)
        sync_hooks(str(tmp_repo), strict_config)
        script_path = tmp_repo / ".claude" / "hooks" / "enforce-policy.sh"
        content = script_path.read_text()
        assert '"main"' in content
        assert '"staging"' in content

    def test_scripts_are_executable(self, tmp_repo, strict_config):
        strict_config["path"] = str(tmp_repo)
        sync_hooks(str(tmp_repo), strict_config)
        for name in ["format-python.sh", "enforce-policy.sh"]:
            script_path = tmp_repo / ".claude" / "hooks" / name
            mode = os.stat(script_path).st_mode
            assert mode & stat.S_IXUSR, f"{name} should be executable"

    def test_creates_claude_directory(self, tmp_repo, strict_config):
        """Should create .claude/ and .claude/hooks/ if they don't exist."""
        strict_config["path"] = str(tmp_repo)
        assert not (tmp_repo / ".claude").exists()
        sync_hooks(str(tmp_repo), strict_config)
        assert (tmp_repo / ".claude").is_dir()
        assert (tmp_repo / ".claude" / "hooks").is_dir()

    def test_idempotent_strict(self, tmp_repo, strict_config):
        """Running sync_hooks twice produces identical files."""
        strict_config["path"] = str(tmp_repo)
        sync_hooks(str(tmp_repo), strict_config)

        # Read all files after first sync
        files_first = {}
        for f in (tmp_repo / ".claude").rglob("*"):
            if f.is_file():
                files_first[str(f.relative_to(tmp_repo))] = f.read_text()

        # Run sync again
        sync_hooks(str(tmp_repo), strict_config)

        # Read all files after second sync
        files_second = {}
        for f in (tmp_repo / ".claude").rglob("*"):
            if f.is_file():
                files_second[str(f.relative_to(tmp_repo))] = f.read_text()

        assert files_first == files_second

    def test_idempotent_relaxed(self, tmp_repo, relaxed_config):
        """Running sync_hooks twice for relaxed produces identical files."""
        relaxed_config["path"] = str(tmp_repo)
        sync_hooks(str(tmp_repo), relaxed_config)

        files_first = {}
        for f in (tmp_repo / ".claude").rglob("*"):
            if f.is_file():
                files_first[str(f.relative_to(tmp_repo))] = f.read_text()

        sync_hooks(str(tmp_repo), relaxed_config)

        files_second = {}
        for f in (tmp_repo / ".claude").rglob("*"):
            if f.is_file():
                files_second[str(f.relative_to(tmp_repo))] = f.read_text()

        assert files_first == files_second

    def test_idempotent_byte_identical(self, tmp_repo, strict_config):
        """Files should be byte-identical after repeated runs."""
        strict_config["path"] = str(tmp_repo)
        sync_hooks(str(tmp_repo), strict_config)
        settings1 = (tmp_repo / ".claude" / "settings.json").read_bytes()
        enforce1 = (tmp_repo / ".claude" / "hooks" / "enforce-policy.sh").read_bytes()
        format1 = (tmp_repo / ".claude" / "hooks" / "format-python.sh").read_bytes()

        sync_hooks(str(tmp_repo), strict_config)
        settings2 = (tmp_repo / ".claude" / "settings.json").read_bytes()
        enforce2 = (tmp_repo / ".claude" / "hooks" / "enforce-policy.sh").read_bytes()
        format2 = (tmp_repo / ".claude" / "hooks" / "format-python.sh").read_bytes()

        assert settings1 == settings2
        assert enforce1 == enforce2
        assert format1 == format2

    def test_defaults_when_config_missing_keys(self, tmp_repo):
        """Should use defaults when config keys are missing."""
        config = {"path": str(tmp_repo)}
        written = sync_hooks(str(tmp_repo), config)
        # Default is strict, so should have 3 files
        assert len(written) == 3

    def test_returned_paths_are_absolute(self, tmp_repo, strict_config):
        strict_config["path"] = str(tmp_repo)
        written = sync_hooks(str(tmp_repo), strict_config)
        for path_str in written:
            assert Path(path_str).is_absolute()


# ---------------------------------------------------------------------------
# Determinism across different branch orderings
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Verify deterministic output regardless of input ordering."""

    def test_settings_deterministic_across_branch_order(self):
        s1 = generate_settings("strict", ["staging", "main", "prod"])
        s2 = generate_settings("strict", ["main", "prod", "staging"])
        # Settings don't embed branches, so should always be equal
        assert s1 == s2

    def test_enforce_script_deterministic_across_branch_order(self):
        s1 = generate_enforce_policy_script(["staging", "main", "prod"])
        s2 = generate_enforce_policy_script(["prod", "main", "staging"])
        assert s1 == s2

    def test_full_sync_deterministic(self, tmp_path):
        """Full sync produces identical output regardless of branch order."""
        repo1 = tmp_path / "repo1"
        repo2 = tmp_path / "repo2"
        repo1.mkdir()
        repo2.mkdir()

        config1 = {
            "enforcement_level": "strict",
            "protected_branches": ["staging", "main", "prod"],
            "merge_policy": "human-only",
        }
        config2 = {
            "enforcement_level": "strict",
            "protected_branches": ["prod", "main", "staging"],
            "merge_policy": "human-only",
        }

        sync_hooks(str(repo1), config1)
        sync_hooks(str(repo2), config2)

        # Compare all generated files
        for rel_path in [
            ".claude/settings.json",
            ".claude/hooks/enforce-policy.sh",
            ".claude/hooks/format-python.sh",
        ]:
            content1 = (repo1 / rel_path).read_text()
            content2 = (repo2 / rel_path).read_text()
            assert content1 == content2, f"Files differ: {rel_path}"


# ---------------------------------------------------------------------------
# CLI Integration: repo add/update generates hooks
# ---------------------------------------------------------------------------


class TestCLIHookGeneration:
    """Test that CLI commands automatically generate hooks."""

    @pytest.fixture
    def tmp_project(self, tmp_path):
        """Create a minimal project directory for CLI tests."""
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

    def test_repo_add_generates_hooks(self, tmp_project, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()

        # Create a target repo directory
        repo_dir = tmp_path / "my-app"
        repo_dir.mkdir()

        result = runner.invoke(cli, ["repo", "add", "my-app", "--path", str(repo_dir)])
        assert result.exit_code == 0
        assert "Generated" in result.output
        assert "hook file(s)" in result.output

        # Verify hooks were generated
        settings_path = repo_dir / ".claude" / "settings.json"
        assert settings_path.exists()
        content = json.loads(settings_path.read_text())
        # Default enforcement is strict
        assert "PreToolUse" in content["hooks"]
        assert "PostToolUse" in content["hooks"]

    def test_repo_add_strict_generates_enforce_policy(
        self, tmp_project, tmp_path, monkeypatch
    ):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()

        repo_dir = tmp_path / "prod-app"
        repo_dir.mkdir()

        result = runner.invoke(
            cli,
            [
                "repo",
                "add",
                "prod",
                "--path",
                str(repo_dir),
                "--enforcement-level",
                "strict",
                "--protected-branches",
                "main,staging",
            ],
        )
        assert result.exit_code == 0

        enforce_path = repo_dir / ".claude" / "hooks" / "enforce-policy.sh"
        assert enforce_path.exists()
        content = enforce_path.read_text()
        assert '"main"' in content
        assert '"staging"' in content

    def test_repo_add_relaxed_no_enforce_policy(
        self, tmp_project, tmp_path, monkeypatch
    ):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()

        repo_dir = tmp_path / "tool"
        repo_dir.mkdir()

        result = runner.invoke(
            cli,
            [
                "repo",
                "add",
                "tool",
                "--path",
                str(repo_dir),
                "--enforcement-level",
                "relaxed",
            ],
        )
        assert result.exit_code == 0

        # Relaxed should have format but NOT enforce-policy
        format_path = repo_dir / ".claude" / "hooks" / "format-python.sh"
        enforce_path = repo_dir / ".claude" / "hooks" / "enforce-policy.sh"
        assert format_path.exists()
        assert not enforce_path.exists()

        # Settings should not have PreToolUse
        settings_path = repo_dir / ".claude" / "settings.json"
        content = json.loads(settings_path.read_text())
        assert "PreToolUse" not in content["hooks"]
        assert "PostToolUse" in content["hooks"]

    def test_repo_update_regenerates_hooks(self, tmp_project, tmp_path, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()

        repo_dir = tmp_path / "app"
        repo_dir.mkdir()

        # Add as strict
        runner.invoke(
            cli,
            [
                "repo",
                "add",
                "app",
                "--path",
                str(repo_dir),
                "--enforcement-level",
                "strict",
            ],
        )
        enforce_path = repo_dir / ".claude" / "hooks" / "enforce-policy.sh"
        assert enforce_path.exists()

        # Update to relaxed
        result = runner.invoke(
            cli, ["repo", "update", "app", "--enforcement-level", "relaxed"]
        )
        assert result.exit_code == 0
        assert "Regenerated" in result.output

        # Settings should now be relaxed (no PreToolUse)
        settings_path = repo_dir / ".claude" / "settings.json"
        content = json.loads(settings_path.read_text())
        assert "PreToolUse" not in content["hooks"]

    def test_repo_update_branches_updates_enforce_script(
        self, tmp_project, tmp_path, monkeypatch
    ):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()

        repo_dir = tmp_path / "app"
        repo_dir.mkdir()

        # Add with main only
        runner.invoke(cli, ["repo", "add", "app", "--path", str(repo_dir)])
        enforce_path = repo_dir / ".claude" / "hooks" / "enforce-policy.sh"
        content_before = enforce_path.read_text()
        assert '"main"' in content_before
        assert '"staging"' not in content_before

        # Update branches to include staging
        result = runner.invoke(
            cli,
            ["repo", "update", "app", "--protected-branches", "main,staging"],
        )
        assert result.exit_code == 0

        content_after = enforce_path.read_text()
        assert '"main"' in content_after
        assert '"staging"' in content_after

    def test_repo_add_generates_output_message(
        self, tmp_project, tmp_path, monkeypatch
    ):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(tmp_project)
        runner = CliRunner()

        repo_dir = tmp_path / "app"
        repo_dir.mkdir()

        result = runner.invoke(cli, ["repo", "add", "app", "--path", str(repo_dir)])
        assert result.exit_code == 0
        assert "Generated 3 hook file(s)" in result.output
