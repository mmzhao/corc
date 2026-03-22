"""Tests for task priority: field storage, scheduler ordering, CLI commands, backwards compat."""

import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from corc.mutations import MutationLog
from corc.scheduler import get_ready_tasks
from corc.state import WorkState


@pytest.fixture
def state(tmp_path):
    ml = MutationLog(tmp_path / "mutations.jsonl")
    ws = WorkState(tmp_path / "state.db", ml)
    return ml, ws


# ---------------------------------------------------------------------------
# Basic priority field
# ---------------------------------------------------------------------------


def test_task_created_with_default_priority(state):
    """Tasks created without explicit priority get default 100."""
    ml, ws = state
    ml.append(
        "task_created",
        {"id": "t1", "name": "default task", "done_when": "done"},
        reason="test",
    )
    ws.refresh()

    task = ws.get_task("t1")
    assert task is not None
    assert task["priority"] == 100


def test_task_created_with_explicit_priority(state):
    """Tasks created with explicit priority store it correctly."""
    ml, ws = state
    ml.append(
        "task_created",
        {"id": "t1", "name": "urgent task", "done_when": "done", "priority": 10},
        reason="test",
    )
    ws.refresh()

    task = ws.get_task("t1")
    assert task["priority"] == 10


def test_task_priority_updated_via_task_updated(state):
    """Priority can be updated via task_updated mutation."""
    ml, ws = state
    ml.append(
        "task_created",
        {"id": "t1", "name": "task", "done_when": "done", "priority": 100},
        reason="test",
    )
    ml.append(
        "task_updated",
        {"priority": 5},
        reason="bumped priority",
        task_id="t1",
    )
    ws.refresh()

    task = ws.get_task("t1")
    assert task["priority"] == 5


# ---------------------------------------------------------------------------
# Priority ordering in get_ready_tasks
# ---------------------------------------------------------------------------


def test_ready_tasks_sorted_by_priority(state):
    """get_ready_tasks returns tasks sorted by priority ascending."""
    ml, ws = state
    # Create tasks with various priorities (not in sorted order)
    ml.append(
        "task_created",
        {"id": "t1", "name": "low priority", "done_when": "done", "priority": 200},
        reason="test",
    )
    ml.append(
        "task_created",
        {"id": "t2", "name": "high priority", "done_when": "done", "priority": 10},
        reason="test",
    )
    ml.append(
        "task_created",
        {"id": "t3", "name": "medium priority", "done_when": "done", "priority": 50},
        reason="test",
    )
    ws.refresh()

    ready = ws.get_ready_tasks()
    assert len(ready) == 3
    assert ready[0]["id"] == "t2"  # priority 10 (highest)
    assert ready[1]["id"] == "t3"  # priority 50
    assert ready[2]["id"] == "t1"  # priority 200 (lowest)


def test_scheduler_dispatches_highest_priority_first(state):
    """Scheduler's get_ready_tasks picks the highest-priority task when slots are limited."""
    ml, ws = state
    ml.append(
        "task_created",
        {"id": "t1", "name": "normal task", "done_when": "done", "priority": 100},
        reason="test",
    )
    ml.append(
        "task_created",
        {"id": "t2", "name": "urgent bugfix", "done_when": "done", "priority": 1},
        reason="test",
    )
    ml.append(
        "task_created",
        {"id": "t3", "name": "nice to have", "done_when": "done", "priority": 500},
        reason="test",
    )
    ws.refresh()

    # With only 1 slot, the highest-priority (lowest number) task should be picked
    dispatched = get_ready_tasks(ws, parallel_limit=1)
    assert len(dispatched) == 1
    assert dispatched[0]["id"] == "t2"

    # With 2 slots, the top 2 should be returned in priority order
    dispatched = get_ready_tasks(ws, parallel_limit=2)
    assert len(dispatched) == 2
    assert dispatched[0]["id"] == "t2"
    assert dispatched[1]["id"] == "t1"


def test_priority_ordering_with_dependencies(state):
    """Priority ordering respects dependency constraints."""
    ml, ws = state
    # t1 has no deps (priority 100)
    ml.append(
        "task_created",
        {"id": "t1", "name": "first", "done_when": "done", "priority": 100, "depends_on": []},
        reason="test",
    )
    # t2 depends on t1 (priority 1 — very urgent, but blocked)
    ml.append(
        "task_created",
        {"id": "t2", "name": "urgent but blocked", "done_when": "done", "priority": 1, "depends_on": ["t1"]},
        reason="test",
    )
    # t3 has no deps (priority 50)
    ml.append(
        "task_created",
        {"id": "t3", "name": "medium", "done_when": "done", "priority": 50, "depends_on": []},
        reason="test",
    )
    ws.refresh()

    ready = ws.get_ready_tasks()
    # t2 should NOT be in ready (blocked by t1)
    ready_ids = [t["id"] for t in ready]
    assert "t2" not in ready_ids
    assert ready[0]["id"] == "t3"  # priority 50, ready
    assert ready[1]["id"] == "t1"  # priority 100, ready


def test_priority_bumped_task_moves_to_front(state):
    """Updating a task's priority via task_updated changes its dispatch order."""
    ml, ws = state
    ml.append(
        "task_created",
        {"id": "t1", "name": "task A", "done_when": "done", "priority": 100},
        reason="test",
    )
    ml.append(
        "task_created",
        {"id": "t2", "name": "task B", "done_when": "done", "priority": 100},
        reason="test",
    )
    ws.refresh()

    # Both have same priority — check initial order
    ready = ws.get_ready_tasks()
    assert len(ready) == 2

    # Now bump t2 to high priority
    ml.append(
        "task_updated",
        {"priority": 1},
        reason="urgent",
        task_id="t2",
    )
    ws.refresh()

    ready = ws.get_ready_tasks()
    assert ready[0]["id"] == "t2"  # now highest priority
    assert ready[1]["id"] == "t1"


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------


def test_backwards_compat_old_tasks_get_default_priority(state):
    """Tasks created from old mutations (no priority field) get default 100."""
    ml, ws = state
    # Simulate an old mutation without priority field
    ml.append(
        "task_created",
        {"id": "t1", "name": "old task", "done_when": "done"},
        reason="test",
    )
    ws.refresh()

    task = ws.get_task("t1")
    assert task["priority"] == 100


def test_backwards_compat_mixed_old_and_new_tasks(state):
    """Mix of old (no priority) and new (with priority) tasks sort correctly."""
    ml, ws = state
    # Old task without priority
    ml.append(
        "task_created",
        {"id": "t1", "name": "old task", "done_when": "done"},
        reason="test",
    )
    # New task with explicit high priority
    ml.append(
        "task_created",
        {"id": "t2", "name": "urgent new task", "done_when": "done", "priority": 10},
        reason="test",
    )
    # New task with explicit low priority
    ml.append(
        "task_created",
        {"id": "t3", "name": "low priority task", "done_when": "done", "priority": 500},
        reason="test",
    )
    ws.refresh()

    ready = ws.get_ready_tasks()
    assert len(ready) == 3
    assert ready[0]["id"] == "t2"  # priority 10
    assert ready[1]["id"] == "t1"  # priority 100 (default)
    assert ready[2]["id"] == "t3"  # priority 500


def test_schema_migration_adds_priority_column(tmp_path):
    """Opening an old database (no priority column) should add it via migration."""
    db_path = tmp_path / "state.db"

    # Create DB with old schema (no priority column)
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
        micro_deviations TEXT DEFAULT '[]',
        attempt_count INTEGER DEFAULT 0,
        max_retries INTEGER DEFAULT 3,
        merge_status TEXT
    );
    CREATE TABLE IF NOT EXISTS agents (
        id TEXT PRIMARY KEY, role TEXT NOT NULL, task_id TEXT,
        status TEXT DEFAULT 'idle', worktree_path TEXT, pid INTEGER,
        started TEXT, last_activity TEXT
    );
    CREATE TABLE IF NOT EXISTS escalations (
        id TEXT PRIMARY KEY, task_id TEXT NOT NULL, task_name TEXT,
        error TEXT, attempts INTEGER, session_log_path TEXT,
        suggested_actions TEXT DEFAULT '[]', done_when TEXT,
        status TEXT DEFAULT 'pending', resolution TEXT,
        created TEXT NOT NULL, resolved TEXT
    );
    CREATE TABLE IF NOT EXISTS finding_rejections (
        id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
        finding_index INTEGER NOT NULL, finding_type TEXT NOT NULL DEFAULT 'general',
        finding_content TEXT NOT NULL, rejection_reason TEXT NOT NULL, ts TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
    CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
    CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
    CREATE INDEX IF NOT EXISTS idx_escalations_status ON escalations(status);
    """
    conn = sqlite3.connect(str(db_path))
    conn.executescript(OLD_SCHEMA)
    # Insert a task with old schema (no priority column)
    conn.execute(
        """INSERT INTO tasks(id, name, description, status, role, depends_on,
           done_when, checklist, context_bundle, context_bundle_mtimes,
           created, updated)
           VALUES(?, ?, ?, 'pending', ?, '[]', ?, '[]', '[]', '{}', ?, ?)""",
        ("t1", "old task", "desc", "implementer", "done",
         "2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
    )
    conn.commit()
    conn.close()

    # Verify priority column doesn't exist yet
    conn = sqlite3.connect(str(db_path))
    columns = {row[1] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
    assert "priority" not in columns
    conn.close()

    # Open with WorkState — migration should add priority column
    ml = MutationLog(tmp_path / "mutations.jsonl")
    ws = WorkState(db_path, ml)

    # Verify column was added
    columns = {row[1] for row in ws.conn.execute("PRAGMA table_info(tasks)").fetchall()}
    assert "priority" in columns

    # Old task should get default priority
    task = ws.get_task("t1")
    assert task is not None
    assert task["priority"] == 100


def test_rebuild_preserves_priority(state, tmp_path):
    """Full rebuild from mutation log preserves priority values."""
    ml, ws = state
    ml.append(
        "task_created",
        {"id": "t1", "name": "urgent", "done_when": "done", "priority": 5},
        reason="test",
    )
    ml.append(
        "task_created",
        {"id": "t2", "name": "normal", "done_when": "done"},
        reason="test",
    )
    ws.refresh()

    # Create fresh WorkState (simulates rebuild on restart)
    ws2 = WorkState(tmp_path / "state2.db", ml)
    t1 = ws2.get_task("t1")
    t2 = ws2.get_task("t2")
    assert t1["priority"] == 5
    assert t2["priority"] == 100


# ---------------------------------------------------------------------------
# Failed tasks with priority
# ---------------------------------------------------------------------------


def test_failed_retriable_tasks_sorted_by_priority(state):
    """Failed retriable tasks are also sorted by priority."""
    ml, ws = state
    ml.append(
        "task_created",
        {"id": "t1", "name": "low pri failed", "done_when": "done", "priority": 200, "max_retries": 3},
        reason="test",
    )
    ml.append(
        "task_created",
        {"id": "t2", "name": "high pri failed", "done_when": "done", "priority": 5, "max_retries": 3},
        reason="test",
    )
    # Fail both with 1 attempt (still retriable)
    ml.append(
        "task_failed",
        {"findings": [], "attempt_count": 1},
        reason="failed",
        task_id="t1",
    )
    ml.append(
        "task_failed",
        {"findings": [], "attempt_count": 1},
        reason="failed",
        task_id="t2",
    )
    ws.refresh()

    ready = ws.get_ready_tasks()
    assert len(ready) == 2
    assert ready[0]["id"] == "t2"  # priority 5, dispatched first
    assert ready[1]["id"] == "t1"  # priority 200
