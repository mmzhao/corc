"""Tests for work state (SQLite materialized from mutation log)."""

import json
from pathlib import Path

import pytest

from corc.mutations import MutationLog
from corc.state import WorkState


@pytest.fixture
def state(tmp_path):
    ml = MutationLog(tmp_path / "mutations.jsonl")
    ws = WorkState(tmp_path / "state.db", ml)
    return ml, ws


def test_create_and_get_task(state):
    ml, ws = state
    ml.append("task_created", {
        "id": "t1",
        "name": "test task",
        "done_when": "tests pass",
        "role": "implementer",
    }, reason="test")
    ws.refresh()

    task = ws.get_task("t1")
    assert task is not None
    assert task["name"] == "test task"
    assert task["status"] == "pending"
    assert task["done_when"] == "tests pass"


def test_task_lifecycle(state):
    ml, ws = state
    ml.append("task_created", {
        "id": "t1", "name": "task", "done_when": "done",
    }, reason="test")
    ml.append("task_assigned", {"agent_id": "a1"}, reason="test", task_id="t1")
    ml.append("task_started", {"attempt": 1}, reason="test", task_id="t1")
    ml.append("task_completed", {"pr_url": "http://pr", "findings": ["found thing"]},
              reason="test", task_id="t1")
    ws.refresh()

    task = ws.get_task("t1")
    assert task["status"] == "completed"
    assert task["pr_url"] == "http://pr"


def test_ready_tasks_with_deps(state):
    ml, ws = state
    ml.append("task_created", {
        "id": "t1", "name": "first", "done_when": "done", "depends_on": [],
    }, reason="test")
    ml.append("task_created", {
        "id": "t2", "name": "second", "done_when": "done", "depends_on": ["t1"],
    }, reason="test")
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
    ml.append("task_created", {
        "id": "t1", "name": "task", "done_when": "done",
    }, reason="test")
    ml.append("task_completed", {"pr_url": "http://pr"}, reason="test", task_id="t1")

    # Create a fresh WorkState from the same mutation log (simulates restart)
    ws2 = WorkState(tmp_path / "state2.db", ml)
    task = ws2.get_task("t1")
    assert task["status"] == "completed"
    assert task["pr_url"] == "http://pr"


def test_list_tasks_by_status(state):
    ml, ws = state
    ml.append("task_created", {"id": "t1", "name": "a", "done_when": "d"}, reason="test")
    ml.append("task_created", {"id": "t2", "name": "b", "done_when": "d"}, reason="test")
    ml.append("task_completed", {}, reason="test", task_id="t1")
    ws.refresh()

    completed = ws.list_tasks(status="completed")
    assert len(completed) == 1
    pending = ws.list_tasks(status="pending")
    assert len(pending) == 1
