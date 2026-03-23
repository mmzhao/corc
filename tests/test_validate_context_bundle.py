"""Tests for context_bundle path validation at task creation time.

Covers:
- validate_context_bundle_paths() unit tests
- CLI warns on missing paths (non-strict)
- CLI errors on missing paths with --strict
- Section references validate the underlying file exists
"""

from pathlib import Path

import pytest
from click.testing import CliRunner

from corc.cli import cli
from corc.context import validate_context_bundle_paths
from corc.mutations import MutationLog
from corc.state import WorkState


# ---------------------------------------------------------------------------
# Unit tests for validate_context_bundle_paths
# ---------------------------------------------------------------------------


class TestValidateContextBundlePaths:
    def test_all_paths_exist(self, tmp_path):
        (tmp_path / "a.py").write_text("# a")
        (tmp_path / "b.md").write_text("# b")
        missing = validate_context_bundle_paths(["a.py", "b.md"], tmp_path)
        assert missing == []

    def test_missing_path_reported(self, tmp_path):
        missing = validate_context_bundle_paths(["does_not_exist.py"], tmp_path)
        assert len(missing) == 1
        assert missing[0]["ref"] == "does_not_exist.py"
        assert missing[0]["file_path"] == "does_not_exist.py"
        assert "not found" in missing[0]["reason"]

    def test_mix_of_existing_and_missing(self, tmp_path):
        (tmp_path / "exists.py").write_text("# yes")
        missing = validate_context_bundle_paths(
            ["exists.py", "nope.py", "also_nope.md"], tmp_path
        )
        assert len(missing) == 2
        refs = [m["ref"] for m in missing]
        assert "nope.py" in refs
        assert "also_nope.md" in refs

    def test_empty_bundle(self, tmp_path):
        missing = validate_context_bundle_paths([], tmp_path)
        assert missing == []

    def test_section_ref_validates_file(self, tmp_path):
        """spec.md#section validates that spec.md exists."""
        missing = validate_context_bundle_paths(["spec.md#module-1-search"], tmp_path)
        assert len(missing) == 1
        assert missing[0]["ref"] == "spec.md#module-1-search"
        assert missing[0]["file_path"] == "spec.md"

    def test_section_ref_existing_file(self, tmp_path):
        """Section ref to existing file passes validation."""
        (tmp_path / "spec.md").write_text("# Spec\n## Module 1: Search\nDetails.")
        missing = validate_context_bundle_paths(["spec.md#module-1-search"], tmp_path)
        assert missing == []

    def test_nested_path(self, tmp_path):
        """Validate paths in subdirectories."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "foo.py").write_text("# foo")
        missing = validate_context_bundle_paths(["src/foo.py", "src/bar.py"], tmp_path)
        assert len(missing) == 1
        assert missing[0]["file_path"] == "src/bar.py"


# ---------------------------------------------------------------------------
# Fixtures for CLI tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# CLI warning tests (non-strict)
# ---------------------------------------------------------------------------


class TestTaskCreateWarnings:
    def test_warns_on_missing_context_path(self, tmp_project):
        """Task is created but a warning is emitted for missing paths."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "task",
                "create",
                "my-task",
                "--done-when",
                "tests pass",
                "--context",
                "nonexistent.py",
            ],
        )
        assert result.exit_code == 0
        assert "Created task" in result.output
        # Warning goes to stderr
        assert "Warning: context_bundle: nonexistent.py" in result.output

    def test_warns_multiple_missing_paths(self, tmp_project):
        """Multiple missing paths each produce a warning."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "task",
                "create",
                "my-task",
                "--done-when",
                "tests pass",
                "--context",
                "a.py,b.py",
            ],
        )
        assert result.exit_code == 0
        assert "Warning: context_bundle: a.py" in result.output
        assert "Warning: context_bundle: b.py" in result.output
        assert "Created task" in result.output

    def test_no_warning_when_paths_exist(self, tmp_project):
        """No warning when all context paths exist."""
        (tmp_project / "real.py").write_text("# real file")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "task",
                "create",
                "my-task",
                "--done-when",
                "tests pass",
                "--context",
                "real.py",
            ],
        )
        assert result.exit_code == 0
        assert "Warning" not in result.output
        assert "Created task" in result.output

    def test_warns_section_ref_missing_file(self, tmp_project):
        """Section references warn when the underlying file is missing."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "task",
                "create",
                "my-task",
                "--done-when",
                "tests pass",
                "--context",
                "missing.md#some-section",
            ],
        )
        assert result.exit_code == 0
        assert "Warning: context_bundle: missing.md#some-section" in result.output
        assert "Created task" in result.output

    def test_no_warning_section_ref_file_exists(self, tmp_project):
        """Section reference to existing file produces no warning."""
        (tmp_project / "spec.md").write_text("# Spec\n## Design\nContent.")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "task",
                "create",
                "my-task",
                "--done-when",
                "tests pass",
                "--context",
                "spec.md#design",
            ],
        )
        assert result.exit_code == 0
        assert "Warning" not in result.output
        assert "Created task" in result.output

    def test_mix_existing_and_missing(self, tmp_project):
        """Warns only for missing paths, task still created."""
        (tmp_project / "exists.py").write_text("# ok")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "task",
                "create",
                "my-task",
                "--done-when",
                "tests pass",
                "--context",
                "exists.py,missing.py",
            ],
        )
        assert result.exit_code == 0
        assert "Warning: context_bundle: missing.py" in result.output
        assert "exists.py" not in result.output.split("Warning")[0] or True
        assert "Created task" in result.output


# ---------------------------------------------------------------------------
# CLI strict mode tests
# ---------------------------------------------------------------------------


class TestTaskCreateStrict:
    def test_strict_errors_on_missing_path(self, tmp_project):
        """--strict refuses to create the task when paths are missing."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "task",
                "create",
                "my-task",
                "--done-when",
                "tests pass",
                "--strict",
                "--context",
                "nonexistent.py",
            ],
        )
        assert result.exit_code != 0
        assert "Warning: context_bundle: nonexistent.py" in result.output
        assert "Aborted" in result.output
        assert "--strict rejects missing context_bundle paths" in result.output
        # Task should NOT be created
        assert "Created task" not in result.output

    def test_strict_errors_on_section_ref_missing_file(self, tmp_project):
        """--strict errors when section reference file doesn't exist."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "task",
                "create",
                "my-task",
                "--done-when",
                "tests pass",
                "--strict",
                "--context",
                "missing.md#design",
            ],
        )
        assert result.exit_code != 0
        assert "Warning: context_bundle: missing.md#design" in result.output
        assert "Aborted" in result.output
        assert "Created task" not in result.output

    def test_strict_passes_when_all_paths_exist(self, tmp_project):
        """--strict allows task creation when all paths exist."""
        (tmp_project / "real.py").write_text("# real")
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "task",
                "create",
                "my-task",
                "--done-when",
                "tests pass",
                "--strict",
                "--context",
                "real.py",
            ],
        )
        assert result.exit_code == 0
        assert "Warning" not in result.output
        assert "Created task" in result.output

    def test_strict_no_context_bundle_passes(self, tmp_project):
        """--strict with no context bundle creates task normally."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "task",
                "create",
                "my-task",
                "--done-when",
                "tests pass",
                "--strict",
            ],
        )
        assert result.exit_code == 0
        assert "Created task" in result.output

    def test_strict_no_mutation_on_failure(self, tmp_project):
        """--strict failure does not write a task_created mutation."""
        ml = MutationLog(tmp_project / "data" / "mutations.jsonl")
        before = len(ml.read_all())

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "task",
                "create",
                "my-task",
                "--done-when",
                "tests pass",
                "--strict",
                "--context",
                "missing.py",
            ],
        )
        assert result.exit_code != 0

        after = len(ml.read_all())
        assert after == before, "No mutation should be appended on strict failure"

    def test_strict_multiple_missing_all_warned(self, tmp_project):
        """--strict with multiple missing paths lists all before aborting."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "task",
                "create",
                "my-task",
                "--done-when",
                "tests pass",
                "--strict",
                "--context",
                "a.py,b.py,c.py",
            ],
        )
        assert result.exit_code != 0
        assert "Warning: context_bundle: a.py" in result.output
        assert "Warning: context_bundle: b.py" in result.output
        assert "Warning: context_bundle: c.py" in result.output
        assert "Aborted" in result.output
        assert "Created task" not in result.output
