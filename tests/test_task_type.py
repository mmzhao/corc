"""Tests for task_type field: differentiation between implementation, investigation, and bugfix.

Covers:
- Type-specific done_when linting rules
- CLI --type flag on task create
- task list showing type
- State storage and migration (backwards compatibility)
- Default type for existing tasks
"""

import json
import sqlite3

import pytest
from click.testing import CliRunner

from corc.lint_done_when import (
    lint_done_when,
    LintResult,
    VALID_TASK_TYPES,
    TYPE_SPECIFIC_PATTERNS,
)
from corc.mutations import MutationLog
from corc.state import WorkState


# ---------------------------------------------------------------------------
# Type-specific linting: investigation
# ---------------------------------------------------------------------------


class TestInvestigationLinting:
    """Investigation tasks must mention 'root cause' or 'documented'."""

    def test_investigation_with_root_cause_passes(self):
        result = lint_done_when(
            "Root cause found and documented in issue tracker",
            task_type="investigation",
        )
        assert result.passed, f"Expected pass, got warnings: {result.warnings}"

    def test_investigation_with_documented_passes(self):
        result = lint_done_when(
            "Issue documented with error logs and posted to tracker",
            task_type="investigation",
        )
        assert result.passed, f"Expected pass, got warnings: {result.warnings}"

    def test_investigation_without_required_patterns_warns(self):
        result = lint_done_when(
            "All tests pass and file exists",
            task_type="investigation",
        )
        assert not result.passed
        assert any("investigation" in w.lower() for w in result.warnings)
        assert any(
            "root cause" in w.lower() or "documented" in w.lower()
            for w in result.warnings
        )

    def test_investigation_root_cause_case_insensitive(self):
        result = lint_done_when(
            "ROOT CAUSE found and tests pass",
            task_type="investigation",
        )
        assert result.passed, f"Expected pass, got warnings: {result.warnings}"

    def test_investigation_documented_case_insensitive(self):
        result = lint_done_when(
            "Issue DOCUMENTED with error logs",
            task_type="investigation",
        )
        assert result.passed, f"Expected pass, got warnings: {result.warnings}"


# ---------------------------------------------------------------------------
# Type-specific linting: bugfix
# ---------------------------------------------------------------------------


class TestBugfixLinting:
    """Bugfix tasks must mention 'regression test' or 'reproduced'."""

    def test_bugfix_with_regression_test_passes(self):
        result = lint_done_when(
            "Regression test added and all tests pass",
            task_type="bugfix",
        )
        assert result.passed, f"Expected pass, got warnings: {result.warnings}"

    def test_bugfix_with_reproduced_passes(self):
        result = lint_done_when(
            "Bug reproduced and fix verified with tests",
            task_type="bugfix",
        )
        assert result.passed, f"Expected pass, got warnings: {result.warnings}"

    def test_bugfix_without_required_patterns_warns(self):
        result = lint_done_when(
            "All tests pass and file exists",
            task_type="bugfix",
        )
        assert not result.passed
        assert any("bugfix" in w.lower() for w in result.warnings)
        assert any(
            "regression test" in w.lower() or "reproduced" in w.lower()
            for w in result.warnings
        )

    def test_bugfix_regression_test_case_insensitive(self):
        result = lint_done_when(
            "REGRESSION TEST covers the edge case",
            task_type="bugfix",
        )
        assert result.passed, f"Expected pass, got warnings: {result.warnings}"

    def test_bugfix_reproduced_case_insensitive(self):
        result = lint_done_when(
            "Bug REPRODUCED in test environment and fix applied",
            task_type="bugfix",
        )
        assert result.passed, f"Expected pass, got warnings: {result.warnings}"


# ---------------------------------------------------------------------------
# Type-specific linting: implementation (default)
# ---------------------------------------------------------------------------


class TestImplementationLinting:
    """Implementation type has no extra requirements beyond the base linter."""

    def test_implementation_no_extra_rules(self):
        result = lint_done_when("All tests pass", task_type="implementation")
        assert result.passed

    def test_default_type_is_implementation(self):
        """Calling lint_done_when without task_type defaults to implementation."""
        result = lint_done_when("All tests pass")
        assert result.passed

    def test_implementation_still_checks_subjective(self):
        result = lint_done_when("Code is clean", task_type="implementation")
        assert not result.passed
        assert "clean" in result.subjective_words


# ---------------------------------------------------------------------------
# Type-specific linting: edge cases
# ---------------------------------------------------------------------------


class TestTypeSpecificEdgeCases:
    def test_type_specific_plus_subjective(self):
        """Type-specific warnings combine with subjective word warnings."""
        result = lint_done_when(
            "Good investigation completed",
            task_type="investigation",
        )
        assert not result.passed
        # Should have both subjective and type-specific warnings
        assert "good" in result.subjective_words
        assert any("investigation" in w.lower() for w in result.warnings)

    def test_valid_task_types_constant(self):
        """VALID_TASK_TYPES contains the three expected types."""
        assert "implementation" in VALID_TASK_TYPES
        assert "investigation" in VALID_TASK_TYPES
        assert "bugfix" in VALID_TASK_TYPES

    def test_type_specific_patterns_defined(self):
        """TYPE_SPECIFIC_PATTERNS has entries for investigation and bugfix."""
        assert "investigation" in TYPE_SPECIFIC_PATTERNS
        assert "bugfix" in TYPE_SPECIFIC_PATTERNS
        # implementation has no type-specific patterns
        assert "implementation" not in TYPE_SPECIFIC_PATTERNS


# ---------------------------------------------------------------------------
# State: task_type storage
# ---------------------------------------------------------------------------


@pytest.fixture
def state(tmp_path):
    ml = MutationLog(tmp_path / "mutations.jsonl")
    ws = WorkState(tmp_path / "state.db", ml)
    return ml, ws


class TestTaskTypeState:
    def test_task_created_stores_task_type(self, state):
        ml, ws = state
        ml.append(
            "task_created",
            {
                "id": "t1",
                "name": "investigation task",
                "done_when": "root cause documented",
                "task_type": "investigation",
            },
            reason="test",
        )
        ws.refresh()

        task = ws.get_task("t1")
        assert task is not None
        assert task["task_type"] == "investigation"

    def test_task_created_defaults_to_implementation(self, state):
        ml, ws = state
        ml.append(
            "task_created",
            {
                "id": "t1",
                "name": "regular task",
                "done_when": "tests pass",
            },
            reason="test",
        )
        ws.refresh()

        task = ws.get_task("t1")
        assert task is not None
        assert task["task_type"] == "implementation"

    def test_task_type_bugfix(self, state):
        ml, ws = state
        ml.append(
            "task_created",
            {
                "id": "t1",
                "name": "fix the bug",
                "done_when": "regression test passes",
                "task_type": "bugfix",
            },
            reason="test",
        )
        ws.refresh()

        task = ws.get_task("t1")
        assert task["task_type"] == "bugfix"

    def test_task_type_survives_rebuild(self, state, tmp_path):
        ml, ws = state
        ml.append(
            "task_created",
            {
                "id": "t1",
                "name": "investigation",
                "done_when": "root cause found",
                "task_type": "investigation",
            },
            reason="test",
        )

        # Rebuild from scratch (simulates restart)
        ws2 = WorkState(tmp_path / "state2.db", ml)
        task = ws2.get_task("t1")
        assert task["task_type"] == "investigation"

    def test_task_updated_can_change_type(self, state):
        ml, ws = state
        ml.append(
            "task_created",
            {
                "id": "t1",
                "name": "task",
                "done_when": "tests pass",
                "task_type": "implementation",
            },
            reason="test",
        )
        ml.append(
            "task_updated",
            {"task_type": "bugfix"},
            reason="reclassified",
            task_id="t1",
        )
        ws.refresh()

        task = ws.get_task("t1")
        assert task["task_type"] == "bugfix"

    def test_list_tasks_includes_task_type(self, state):
        ml, ws = state
        ml.append(
            "task_created",
            {
                "id": "t1",
                "name": "task a",
                "done_when": "tests pass",
                "task_type": "implementation",
            },
            reason="test",
        )
        ml.append(
            "task_created",
            {
                "id": "t2",
                "name": "task b",
                "done_when": "root cause documented",
                "task_type": "investigation",
            },
            reason="test",
        )
        ws.refresh()

        tasks = ws.list_tasks()
        types = {t["id"]: t["task_type"] for t in tasks}
        assert types["t1"] == "implementation"
        assert types["t2"] == "investigation"


# ---------------------------------------------------------------------------
# State: migration / backwards compatibility
# ---------------------------------------------------------------------------


# Schema without task_type column (simulates pre-existing database)
OLD_SCHEMA_NO_TASK_TYPE = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'pending',
    role TEXT,
    agent_id TEXT,
    depends_on TEXT DEFAULT '[]',
    done_when TEXT NOT NULL,
    checklist TEXT DEFAULT '[]',
    context_bundle TEXT DEFAULT '[]',
    context_bundle_mtimes TEXT DEFAULT '{}',
    pr_url TEXT,
    proof_of_work TEXT,
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    completed TEXT,
    findings TEXT DEFAULT '[]',
    micro_deviations TEXT DEFAULT '[]',
    attempt_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    merge_status TEXT,
    priority INTEGER DEFAULT 100
);

CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    task_id TEXT,
    status TEXT DEFAULT 'idle',
    worktree_path TEXT,
    pid INTEGER,
    started TEXT,
    last_activity TEXT
);

CREATE TABLE IF NOT EXISTS escalations (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    task_name TEXT,
    error TEXT,
    attempts INTEGER,
    session_log_path TEXT,
    suggested_actions TEXT DEFAULT '[]',
    done_when TEXT,
    status TEXT DEFAULT 'pending',
    resolution TEXT,
    created TEXT NOT NULL,
    resolved TEXT
);

CREATE TABLE IF NOT EXISTS finding_rejections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    finding_index INTEGER NOT NULL,
    finding_type TEXT NOT NULL DEFAULT 'general',
    finding_content TEXT NOT NULL,
    rejection_reason TEXT NOT NULL,
    ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_escalations_status ON escalations(status);
"""


def _get_column_names(db_path, table="tasks"):
    conn = sqlite3.connect(str(db_path))
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    conn.close()
    return columns


class TestTaskTypeMigration:
    def test_fresh_db_has_task_type_column(self, tmp_path):
        """Fresh database should include task_type column."""
        ml = MutationLog(tmp_path / "mutations.jsonl")
        db_path = tmp_path / "state.db"
        WorkState(db_path, ml)

        columns = _get_column_names(db_path)
        assert "task_type" in columns

    def test_migration_adds_task_type_column(self, tmp_path):
        """Opening a database without task_type column should add it."""
        db_path = tmp_path / "state.db"

        # Create DB with old schema (no task_type column)
        conn = sqlite3.connect(str(db_path))
        conn.executescript(OLD_SCHEMA_NO_TASK_TYPE)
        conn.commit()
        conn.close()

        old_columns = _get_column_names(db_path)
        assert "task_type" not in old_columns

        # Open with WorkState — migration should add the column
        ml = MutationLog(tmp_path / "mutations.jsonl")
        WorkState(db_path, ml)

        new_columns = _get_column_names(db_path)
        assert "task_type" in new_columns

    def test_migration_preserves_existing_tasks(self, tmp_path):
        """Existing tasks without task_type get 'implementation' default."""
        db_path = tmp_path / "state.db"

        # Create DB with old schema and insert a task
        conn = sqlite3.connect(str(db_path))
        conn.executescript(OLD_SCHEMA_NO_TASK_TYPE)
        conn.execute(
            """INSERT INTO tasks(id, name, description, status, role, depends_on,
               done_when, checklist, context_bundle, context_bundle_mtimes,
               created, updated)
               VALUES(?, ?, ?, 'pending', ?, '[]', ?, '[]', '[]', '{}', ?, ?)""",
            (
                "t-old",
                "old task",
                "desc",
                "implementer",
                "tests pass",
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:00:00Z",
            ),
        )
        conn.commit()
        conn.close()

        # Open with WorkState — migration runs
        ml = MutationLog(tmp_path / "mutations.jsonl")
        ws = WorkState(db_path, ml)

        task = ws.get_task("t-old")
        assert task is not None
        assert task["name"] == "old task"
        # Default value from migration
        assert task["task_type"] == "implementation"


# ---------------------------------------------------------------------------
# CLI integration tests
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


class TestCLITaskType:
    def test_create_with_default_type(self, tmp_project):
        """Without --type, task defaults to implementation."""
        runner = CliRunner()
        result = runner.invoke(
            __import__("corc.cli", fromlist=["cli"]).cli,
            ["task", "create", "my-task", "--done-when", "All tests pass"],
        )
        assert result.exit_code == 0
        assert "Created task" in result.output

    def test_create_with_investigation_type(self, tmp_project):
        runner = CliRunner()
        from corc.cli import cli

        result = runner.invoke(
            cli,
            [
                "task",
                "create",
                "investigate-bug",
                "--done-when",
                "Root cause documented in findings",
                "--type",
                "investigation",
            ],
        )
        assert result.exit_code == 0
        assert "Created task" in result.output
        assert "[investigation]" in result.output

    def test_create_with_bugfix_type(self, tmp_project):
        runner = CliRunner()
        from corc.cli import cli

        result = runner.invoke(
            cli,
            [
                "task",
                "create",
                "fix-crash",
                "--done-when",
                "Bug reproduced and regression test added",
                "--type",
                "bugfix",
            ],
        )
        assert result.exit_code == 0
        assert "Created task" in result.output
        assert "[bugfix]" in result.output

    def test_create_implementation_type_no_bracket_in_output(self, tmp_project):
        """Implementation type (default) should not show [implementation] in output."""
        runner = CliRunner()
        from corc.cli import cli

        result = runner.invoke(
            cli,
            [
                "task",
                "create",
                "build-feature",
                "--done-when",
                "All tests pass",
                "--type",
                "implementation",
            ],
        )
        assert result.exit_code == 0
        assert "[implementation]" not in result.output

    def test_create_with_invalid_type_rejected(self, tmp_project):
        """Invalid type should be rejected by Click's Choice validator."""
        runner = CliRunner()
        from corc.cli import cli

        result = runner.invoke(
            cli,
            [
                "task",
                "create",
                "task",
                "--done-when",
                "tests pass",
                "--type",
                "invalid",
            ],
        )
        assert result.exit_code != 0

    def test_create_investigation_strict_warns_without_required(self, tmp_project):
        """--strict + investigation type without 'root cause'/'documented' should fail."""
        runner = CliRunner()
        from corc.cli import cli

        result = runner.invoke(
            cli,
            [
                "task",
                "create",
                "investigate",
                "--done-when",
                "All tests pass",
                "--type",
                "investigation",
                "--strict",
            ],
        )
        assert result.exit_code != 0

    def test_create_bugfix_strict_warns_without_required(self, tmp_project):
        """--strict + bugfix type without 'regression test'/'reproduced' should fail."""
        runner = CliRunner()
        from corc.cli import cli

        result = runner.invoke(
            cli,
            [
                "task",
                "create",
                "fix-bug",
                "--done-when",
                "All tests pass",
                "--type",
                "bugfix",
                "--strict",
            ],
        )
        assert result.exit_code != 0

    def test_list_shows_type_for_non_default(self, tmp_project):
        """task list should show type for investigation/bugfix tasks."""
        runner = CliRunner()
        from corc.cli import cli

        # Create an investigation task
        runner.invoke(
            cli,
            [
                "task",
                "create",
                "probe-issue",
                "--done-when",
                "Root cause documented",
                "--type",
                "investigation",
            ],
        )
        # Create a default task
        runner.invoke(
            cli,
            [
                "task",
                "create",
                "build-thing",
                "--done-when",
                "All tests pass",
            ],
        )

        result = runner.invoke(cli, ["task", "list"])
        assert result.exit_code == 0
        assert "(investigation)" in result.output
        # Default implementation type should not show
        assert "(implementation)" not in result.output

    def test_status_shows_type(self, tmp_project):
        """task status should show the task type."""
        runner = CliRunner()
        from corc.cli import cli

        result = runner.invoke(
            cli,
            [
                "task",
                "create",
                "my-bugfix",
                "--done-when",
                "Bug reproduced and regression test added",
                "--type",
                "bugfix",
            ],
        )
        assert result.exit_code == 0
        # Extract task ID from output
        task_id = result.output.split("Created task ")[1].split(":")[0]

        result = runner.invoke(cli, ["task", "status", task_id])
        assert result.exit_code == 0
        assert "Type: bugfix" in result.output

    def test_backwards_compatible_no_type_flag(self, tmp_project):
        """Tasks created without --type should still work and default to implementation."""
        runner = CliRunner()
        from corc.cli import cli

        result = runner.invoke(
            cli,
            ["task", "create", "old-style-task", "--done-when", "All tests pass"],
        )
        assert result.exit_code == 0
        assert "Created task" in result.output

        # Extract task ID
        task_id = result.output.split("Created task ")[1].split(":")[0]

        # Status should show implementation type
        result = runner.invoke(cli, ["task", "status", task_id])
        assert "Type: implementation" in result.output
