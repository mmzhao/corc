"""Tests for work state (SQLite materialized from mutation log)."""

import json
import sqlite3
from pathlib import Path

import pytest

from corc.mutations import MutationLog
from corc.state import SCHEMA, WorkState


@pytest.fixture
def state(tmp_path):
    ml = MutationLog(tmp_path / "mutations.jsonl")
    ws = WorkState(tmp_path / "state.db", ml)
    return ml, ws


def test_create_and_get_task(state):
    ml, ws = state
    ml.append(
        "task_created",
        {
            "id": "t1",
            "name": "test task",
            "done_when": "tests pass",
            "role": "implementer",
        },
        reason="test",
    )
    ws.refresh()

    task = ws.get_task("t1")
    assert task is not None
    assert task["name"] == "test task"
    assert task["status"] == "pending"
    assert task["done_when"] == "tests pass"


def test_task_lifecycle(state):
    ml, ws = state
    ml.append(
        "task_created",
        {
            "id": "t1",
            "name": "task",
            "done_when": "done",
        },
        reason="test",
    )
    ml.append("task_assigned", {"agent_id": "a1"}, reason="test", task_id="t1")
    ml.append("task_started", {"attempt": 1}, reason="test", task_id="t1")
    ml.append(
        "task_completed",
        {"pr_url": "http://pr", "findings": ["found thing"]},
        reason="test",
        task_id="t1",
    )
    ws.refresh()

    task = ws.get_task("t1")
    assert task["status"] == "completed"
    assert task["pr_url"] == "http://pr"


def test_ready_tasks_with_deps(state):
    ml, ws = state
    ml.append(
        "task_created",
        {
            "id": "t1",
            "name": "first",
            "done_when": "done",
            "depends_on": [],
        },
        reason="test",
    )
    ml.append(
        "task_created",
        {
            "id": "t2",
            "name": "second",
            "done_when": "done",
            "depends_on": ["t1"],
        },
        reason="test",
    )
    ws.refresh()

    ready = ws.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0]["id"] == "t1"

    # Complete t1
    ml.append("task_completed", {}, reason="test", task_id="t1")
    ws.refresh()

    ready = ws.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0]["id"] == "t2"


def test_rebuild_from_mutation_log(state, tmp_path):
    ml, ws = state
    ml.append(
        "task_created",
        {
            "id": "t1",
            "name": "task",
            "done_when": "done",
        },
        reason="test",
    )
    ml.append("task_completed", {"pr_url": "http://pr"}, reason="test", task_id="t1")

    # Create a fresh WorkState from the same mutation log (simulates restart)
    ws2 = WorkState(tmp_path / "state2.db", ml)
    task = ws2.get_task("t1")
    assert task["status"] == "completed"
    assert task["pr_url"] == "http://pr"


def test_list_tasks_by_status(state):
    ml, ws = state
    ml.append(
        "task_created", {"id": "t1", "name": "a", "done_when": "d"}, reason="test"
    )
    ml.append(
        "task_created", {"id": "t2", "name": "b", "done_when": "d"}, reason="test"
    )
    ml.append("task_completed", {}, reason="test", task_id="t1")
    ws.refresh()

    completed = ws.list_tasks(status="completed")
    assert len(completed) == 1
    pending = ws.list_tasks(status="pending")
    assert len(pending) == 1


# --- Schema migration tests ---

# Old schema without attempt_count, max_retries, merge_status columns
OLD_SCHEMA = """
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
    micro_deviations TEXT DEFAULT '[]'
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


def _create_old_schema_db(db_path):
    """Create a database with the old schema (no retry columns)."""
    conn = sqlite3.connect(str(db_path))
    conn.executescript(OLD_SCHEMA)
    conn.commit()
    conn.close()


def _get_column_names(db_path, table="tasks"):
    """Get column names for a table from PRAGMA table_info."""
    conn = sqlite3.connect(str(db_path))
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    conn.close()
    return columns


def test_fresh_db_has_retry_columns(tmp_path):
    """Fresh database should include attempt_count and max_retries columns."""
    ml = MutationLog(tmp_path / "mutations.jsonl")
    db_path = tmp_path / "state.db"
    WorkState(db_path, ml)

    columns = _get_column_names(db_path)
    assert "attempt_count" in columns
    assert "max_retries" in columns
    assert "merge_status" in columns


def test_migration_adds_retry_columns(tmp_path):
    """Opening a database with old schema should add the missing columns."""
    db_path = tmp_path / "state.db"

    # Create DB with old schema (no retry columns)
    _create_old_schema_db(db_path)
    old_columns = _get_column_names(db_path)
    assert "attempt_count" not in old_columns
    assert "max_retries" not in old_columns

    # Now open with WorkState — migration should add columns
    ml = MutationLog(tmp_path / "mutations.jsonl")
    WorkState(db_path, ml)

    new_columns = _get_column_names(db_path)
    assert "attempt_count" in new_columns
    assert "max_retries" in new_columns
    assert "merge_status" in new_columns


def test_migration_preserves_existing_data(tmp_path):
    """Migration should not lose data in existing rows."""
    db_path = tmp_path / "state.db"

    # Create DB with old schema and insert a task row
    conn = sqlite3.connect(str(db_path))
    conn.executescript(OLD_SCHEMA)
    conn.execute(
        """INSERT INTO tasks(id, name, description, status, role, depends_on,
           done_when, checklist, context_bundle, context_bundle_mtimes,
           created, updated)
           VALUES(?, ?, ?, 'pending', ?, '[]', ?, '[]', '[]', '{}', ?, ?)""",
        (
            "t1",
            "existing task",
            "desc",
            "implementer",
            "done",
            "2025-01-01T00:00:00Z",
            "2025-01-01T00:00:00Z",
        ),
    )
    conn.commit()
    conn.close()

    # Open with WorkState — migration runs, data should survive
    ml = MutationLog(tmp_path / "mutations.jsonl")
    ws = WorkState(db_path, ml)

    task = ws.get_task("t1")
    assert task is not None
    assert task["name"] == "existing task"
    assert task["status"] == "pending"
    # New columns should have defaults
    assert task["attempt_count"] == 0
    assert task["max_retries"] == 3


def test_old_schema_replays_retry_mutations(tmp_path):
    """Mutations with retry fields should replay correctly after migration."""
    db_path = tmp_path / "state.db"

    # Create DB with old schema (no retry columns)
    _create_old_schema_db(db_path)

    # Create mutation log with retry-aware mutations
    ml = MutationLog(tmp_path / "mutations.jsonl")
    ml.append(
        "task_created",
        {
            "id": "t1",
            "name": "retry task",
            "done_when": "tests pass",
            "max_retries": 5,
        },
        reason="test",
    )
    ml.append(
        "task_failed",
        {"findings": ["error occurred"], "attempt_count": 1},
        reason="first attempt failed",
        task_id="t1",
    )

    # Open WorkState — should migrate schema then replay mutations
    ws = WorkState(db_path, ml)

    task = ws.get_task("t1")
    assert task is not None
    assert task["status"] == "failed"
    assert task["max_retries"] == 5
    assert task["attempt_count"] == 1


def test_task_created_stores_attempt_count(state):
    """task_created mutation should explicitly store attempt_count."""
    ml, ws = state
    ml.append(
        "task_created",
        {
            "id": "t1",
            "name": "task with retries",
            "done_when": "done",
            "max_retries": 2,
            "attempt_count": 0,
        },
        reason="test",
    )
    ws.refresh()

    task = ws.get_task("t1")
    assert task["attempt_count"] == 0
    assert task["max_retries"] == 2


def test_retry_lifecycle(state):
    """Full retry lifecycle: create → fail → retry → complete."""
    ml, ws = state

    ml.append(
        "task_created",
        {"id": "t1", "name": "retry task", "done_when": "done", "max_retries": 3},
        reason="test",
    )
    ws.refresh()

    task = ws.get_task("t1")
    assert task["attempt_count"] == 0
    assert task["max_retries"] == 3

    # First attempt fails
    ml.append(
        "task_failed",
        {"findings": ["error 1"], "attempt_count": 1},
        reason="failed",
        task_id="t1",
    )
    ws.refresh()

    task = ws.get_task("t1")
    assert task["status"] == "failed"
    assert task["attempt_count"] == 1

    # Should still be retriable
    ready = ws.get_ready_tasks()
    assert any(t["id"] == "t1" for t in ready)

    # Escalate after too many failures
    ml.append(
        "task_escalated",
        {"attempt_count": 4},
        reason="exceeded retries",
        task_id="t1",
    )
    ws.refresh()

    task = ws.get_task("t1")
    assert task["status"] == "escalated"
    assert task["attempt_count"] == 4


def test_migration_is_idempotent(tmp_path):
    """Running migration multiple times should not error or change data."""
    db_path = tmp_path / "state.db"
    ml = MutationLog(tmp_path / "mutations.jsonl")
    ml.append(
        "task_created",
        {"id": "t1", "name": "task", "done_when": "done", "max_retries": 5},
        reason="test",
    )

    # Create WorkState twice on the same DB (simulates restart)
    ws1 = WorkState(db_path, ml)
    task1 = ws1.get_task("t1")
    ws1.conn.close()

    ws2 = WorkState(db_path, ml)
    task2 = ws2.get_task("t1")

    assert task1["max_retries"] == task2["max_retries"] == 5
    assert task1["attempt_count"] == task2["attempt_count"] == 0


def test_task_updated_depends_on(state):
    """task_updated mutation should update depends_on in SQLite."""
    ml, ws = state
    ml.append(
        "task_created",
        {
            "id": "t1",
            "name": "task",
            "done_when": "done",
            "depends_on": [],
        },
        reason="test",
    )
    ws.refresh()

    task = ws.get_task("t1")
    assert task["depends_on"] == []

    # Update depends_on via task_updated mutation
    ml.append(
        "task_updated",
        {"depends_on": ["t0", "t-1"]},
        reason="add deps",
        task_id="t1",
    )
    ws.refresh()

    task = ws.get_task("t1")
    assert task["depends_on"] == ["t0", "t-1"]


def test_task_updated_all_fields(state):
    """task_updated mutation should handle all task fields."""
    ml, ws = state
    ml.append(
        "task_created",
        {
            "id": "t1",
            "name": "original name",
            "done_when": "original done_when",
            "description": "original desc",
            "role": "implementer",
            "depends_on": [],
            "checklist": [],
            "context_bundle": [],
            "priority": 100,
        },
        reason="test",
    )
    ws.refresh()

    # Update every field that task_updated should support
    ml.append(
        "task_updated",
        {
            "name": "new name",
            "description": "new desc",
            "status": "running",
            "role": "scout",
            "depends_on": ["dep1"],
            "done_when": "new done_when",
            "checklist": [{"item": "check1", "done": False}],
            "context_bundle": ["file1.md", "file2.md"],
            "context_bundle_mtimes": {"file1.md": "2025-01-01"},
            "priority": 10,
            "pr_url": "http://example.com/pr/1",
            "proof_of_work": {"commit": "abc123"},
            "findings": ["finding1"],
            "micro_deviations": ["deviation1"],
            "attempt_count": 2,
            "max_retries": 5,
            "merge_status": "merged",
        },
        reason="update all fields",
        task_id="t1",
    )
    ws.refresh()

    task = ws.get_task("t1")
    assert task["name"] == "new name"
    assert task["description"] == "new desc"
    assert task["status"] == "running"
    assert task["role"] == "scout"
    assert task["depends_on"] == ["dep1"]
    assert task["done_when"] == "new done_when"
    assert task["checklist"] == [{"item": "check1", "done": False}]
    assert task["context_bundle"] == ["file1.md", "file2.md"]
    assert task["context_bundle_mtimes"] == {"file1.md": "2025-01-01"}
    assert task["priority"] == 10
    assert task["pr_url"] == "http://example.com/pr/1"
    assert task["proof_of_work"] == {"commit": "abc123"}
    assert task["findings"] == ["finding1"]
    assert task["micro_deviations"] == ["deviation1"]
    assert task["attempt_count"] == 2
    assert task["max_retries"] == 5
    assert task["merge_status"] == "merged"
