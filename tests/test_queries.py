"""Tests for the query API — data layer for TUI and future web GUI."""

import json
from pathlib import Path

import pytest

from corc.audit import AuditLog
from corc.mutations import MutationLog
from corc.queries import QueryAPI
from corc.sessions import SessionLogger
from corc.state import WorkState


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def data_layers(tmp_path):
    """Create all three data layers and return (mutation_log, work_state, audit_log, session_logger, query_api)."""
    ml = MutationLog(tmp_path / "mutations.jsonl")
    ws = WorkState(tmp_path / "state.db", ml)
    al = AuditLog(tmp_path / "events")
    sl = SessionLogger(tmp_path / "sessions")
    qa = QueryAPI(ws, al, sl)
    return ml, ws, al, sl, qa


def _create_task(
    ml, task_id, name, depends_on=None, priority=100, task_type="implementation"
):
    """Helper to create a task via mutation."""
    ml.append(
        "task_created",
        {
            "id": task_id,
            "name": name,
            "done_when": f"{name} is complete",
            "role": "implementer",
            "depends_on": depends_on or [],
            "priority": priority,
            "task_type": task_type,
        },
        reason="test",
    )


def _assign_agent(ml, task_id, agent_id):
    """Helper to create and assign an agent to a task."""
    ml.append(
        "agent_created",
        {
            "id": agent_id,
            "role": "implementer",
            "task_id": task_id,
            "pid": 12345,
        },
        reason="test",
    )
    ml.append("task_assigned", {"agent_id": agent_id}, reason="test", task_id=task_id)


def _start_task(ml, task_id):
    ml.append("task_started", {"attempt": 1}, reason="test", task_id=task_id)


def _complete_task(ml, task_id):
    ml.append(
        "task_completed",
        {"pr_url": f"http://pr/{task_id}"},
        reason="test",
        task_id=task_id,
    )


def _fail_task(ml, task_id, attempt_count=1):
    ml.append(
        "task_failed",
        {"findings": ["error"], "attempt_count": attempt_count},
        reason="test",
        task_id=task_id,
    )


# ------------------------------------------------------------------
# get_active_plan_tasks
# ------------------------------------------------------------------


class TestGetActivePlanTasks:
    def test_returns_non_completed_tasks(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        _create_task(ml, "t1", "pending task")
        _create_task(ml, "t2", "completed task")
        _complete_task(ml, "t2")
        _create_task(ml, "t3", "running task")
        _assign_agent(ml, "t3", "a1")
        _start_task(ml, "t3")
        ws.refresh()

        active = qa.get_active_plan_tasks()
        active_ids = {t["id"] for t in active}

        assert "t1" in active_ids  # pending
        assert "t2" not in active_ids  # completed — excluded
        assert "t3" in active_ids  # running

    def test_includes_all_active_statuses(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        # Create tasks in various active statuses
        _create_task(ml, "t-pending", "pending")
        _create_task(ml, "t-assigned", "assigned")
        _assign_agent(ml, "t-assigned", "a1")
        _create_task(ml, "t-running", "running")
        _assign_agent(ml, "t-running", "a2")
        _start_task(ml, "t-running")
        _create_task(ml, "t-failed", "failed")
        _fail_task(ml, "t-failed")
        _create_task(ml, "t-escalated", "escalated")
        ml.append(
            "task_escalated", {"attempt_count": 4}, reason="test", task_id="t-escalated"
        )
        _create_task(ml, "t-handed-off", "handed off")
        ml.append("task_handed_off", {}, reason="test", task_id="t-handed-off")
        _create_task(ml, "t-pending-merge", "pending merge")
        ml.append(
            "task_pending_merge",
            {"proof_of_work": {"commit": "abc"}, "findings": []},
            reason="test",
            task_id="t-pending-merge",
        )
        ws.refresh()

        active = qa.get_active_plan_tasks()
        active_ids = {t["id"] for t in active}

        assert "t-pending" in active_ids
        assert "t-assigned" in active_ids
        assert "t-running" in active_ids
        assert "t-failed" in active_ids
        assert "t-escalated" in active_ids
        assert "t-handed-off" in active_ids
        assert "t-pending-merge" in active_ids

    def test_sorted_by_priority(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        _create_task(ml, "t-low", "low priority", priority=200)
        _create_task(ml, "t-high", "high priority", priority=10)
        _create_task(ml, "t-mid", "mid priority", priority=100)
        ws.refresh()

        active = qa.get_active_plan_tasks()
        assert [t["id"] for t in active] == ["t-high", "t-mid", "t-low"]

    def test_empty_when_all_completed(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        _create_task(ml, "t1", "task")
        _complete_task(ml, "t1")
        ws.refresh()

        assert qa.get_active_plan_tasks() == []

    def test_empty_when_no_tasks(self, data_layers):
        _, _, _, _, qa = data_layers
        assert qa.get_active_plan_tasks() == []


# ------------------------------------------------------------------
# get_running_tasks_with_agents
# ------------------------------------------------------------------


class TestGetRunningTasksWithAgents:
    def test_returns_running_tasks_with_agent_info(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        _create_task(ml, "t1", "running task")
        _assign_agent(ml, "t1", "agent-1")
        _start_task(ml, "t1")
        ws.refresh()

        result = qa.get_running_tasks_with_agents()
        assert len(result) == 1
        assert result[0]["id"] == "t1"
        assert result[0]["status"] == "running"
        assert len(result[0]["agents"]) == 1
        assert result[0]["agents"][0]["id"] == "agent-1"
        assert result[0]["agents"][0]["role"] == "implementer"

    def test_excludes_non_running_tasks(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        _create_task(ml, "t1", "pending task")
        _create_task(ml, "t2", "running task")
        _assign_agent(ml, "t2", "a1")
        _start_task(ml, "t2")
        _create_task(ml, "t3", "completed task")
        _complete_task(ml, "t3")
        ws.refresh()

        result = qa.get_running_tasks_with_agents()
        assert len(result) == 1
        assert result[0]["id"] == "t2"

    def test_task_with_no_agent(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        # Edge case: task marked running but no agent record
        _create_task(ml, "t1", "orphan running task")
        _start_task(ml, "t1")
        ws.refresh()

        result = qa.get_running_tasks_with_agents()
        assert len(result) == 1
        assert result[0]["agents"] == []

    def test_empty_when_nothing_running(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        _create_task(ml, "t1", "pending task")
        ws.refresh()

        assert qa.get_running_tasks_with_agents() == []


# ------------------------------------------------------------------
# get_ready_tasks
# ------------------------------------------------------------------


class TestGetReadyTasks:
    def test_returns_tasks_with_satisfied_deps(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        _create_task(ml, "t1", "no deps")
        _create_task(ml, "t2", "depends on t1", depends_on=["t1"])
        ws.refresh()

        ready = qa.get_ready_tasks()
        ready_ids = [t["id"] for t in ready]
        assert "t1" in ready_ids
        assert "t2" not in ready_ids

    def test_dep_completed_unblocks_task(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        _create_task(ml, "t1", "first")
        _create_task(ml, "t2", "second", depends_on=["t1"])
        _complete_task(ml, "t1")
        ws.refresh()

        ready = qa.get_ready_tasks()
        ready_ids = [t["id"] for t in ready]
        assert "t2" in ready_ids

    def test_includes_retriable_failed_tasks(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        _create_task(ml, "t1", "will fail")
        _fail_task(ml, "t1", attempt_count=1)
        ws.refresh()

        ready = qa.get_ready_tasks()
        assert any(t["id"] == "t1" for t in ready)

    def test_excludes_exhausted_failed_tasks(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        ml.append(
            "task_created",
            {
                "id": "t1",
                "name": "exhausted",
                "done_when": "done",
                "max_retries": 2,
            },
            reason="test",
        )
        _fail_task(ml, "t1", attempt_count=3)
        ws.refresh()

        ready = qa.get_ready_tasks()
        assert not any(t["id"] == "t1" for t in ready)

    def test_empty_when_no_tasks(self, data_layers):
        _, _, _, _, qa = data_layers
        assert qa.get_ready_tasks() == []


# ------------------------------------------------------------------
# get_blocked_tasks_with_reasons
# ------------------------------------------------------------------


class TestGetBlockedTasksWithReasons:
    def test_returns_tasks_with_unsatisfied_deps(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        _create_task(ml, "t1", "first")
        _create_task(ml, "t2", "blocked", depends_on=["t1"])
        ws.refresh()

        blocked = qa.get_blocked_tasks_with_reasons()
        assert len(blocked) == 1
        assert blocked[0]["id"] == "t2"
        assert blocked[0]["blocked_by"] == ["t1"]
        assert "t1" in blocked[0]["reason"]

    def test_multiple_unsatisfied_deps(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        _create_task(ml, "t1", "dep1")
        _create_task(ml, "t2", "dep2")
        _create_task(ml, "t3", "blocked by both", depends_on=["t1", "t2"])
        ws.refresh()

        blocked = qa.get_blocked_tasks_with_reasons()
        assert len(blocked) == 1
        assert set(blocked[0]["blocked_by"]) == {"t1", "t2"}
        assert "2" in blocked[0]["reason"]  # "2 incomplete dependencies"

    def test_partially_satisfied_deps(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        _create_task(ml, "t1", "dep1")
        _create_task(ml, "t2", "dep2")
        _create_task(ml, "t3", "partially blocked", depends_on=["t1", "t2"])
        _complete_task(ml, "t1")
        ws.refresh()

        blocked = qa.get_blocked_tasks_with_reasons()
        assert len(blocked) == 1
        assert blocked[0]["blocked_by"] == ["t2"]
        assert "1" in blocked[0]["reason"]

    def test_not_blocked_when_deps_satisfied(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        _create_task(ml, "t1", "dep")
        _create_task(ml, "t2", "unblocked", depends_on=["t1"])
        _complete_task(ml, "t1")
        ws.refresh()

        blocked = qa.get_blocked_tasks_with_reasons()
        assert len(blocked) == 0

    def test_no_deps_not_blocked(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        _create_task(ml, "t1", "no deps")
        ws.refresh()

        blocked = qa.get_blocked_tasks_with_reasons()
        assert len(blocked) == 0

    def test_excludes_non_pending_tasks(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        # Running task with unsatisfied deps should not appear
        _create_task(ml, "t1", "dep")
        _create_task(ml, "t2", "running with dep", depends_on=["t1"])
        _assign_agent(ml, "t2", "a1")
        _start_task(ml, "t2")
        ws.refresh()

        blocked = qa.get_blocked_tasks_with_reasons()
        assert len(blocked) == 0

    def test_singular_dependency_reason(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        _create_task(ml, "t1", "dep")
        _create_task(ml, "t2", "blocked", depends_on=["t1"])
        ws.refresh()

        blocked = qa.get_blocked_tasks_with_reasons()
        assert "dependency" in blocked[0]["reason"]  # singular
        assert "dependencies" not in blocked[0]["reason"]

    def test_plural_dependencies_reason(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        _create_task(ml, "t1", "dep1")
        _create_task(ml, "t2", "dep2")
        _create_task(ml, "t3", "blocked", depends_on=["t1", "t2"])
        ws.refresh()

        blocked = qa.get_blocked_tasks_with_reasons()
        assert "dependencies" in blocked[0]["reason"]  # plural

    def test_empty_when_no_tasks(self, data_layers):
        _, _, _, _, qa = data_layers
        assert qa.get_blocked_tasks_with_reasons() == []


# ------------------------------------------------------------------
# get_recent_events
# ------------------------------------------------------------------


class TestGetRecentEvents:
    def test_returns_recent_audit_events(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        al.log("task_dispatched", task_id="t1", agent_id="a1")
        al.log("task_completed", task_id="t1")
        al.log("tool_call", task_id="t2", tool="bash")

        events = qa.get_recent_events(10)
        assert len(events) == 3
        assert events[0]["event_type"] == "task_dispatched"
        assert events[-1]["event_type"] == "tool_call"

    def test_limits_to_n_events(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        for i in range(20):
            al.log("event", task_id=f"t{i}")

        events = qa.get_recent_events(5)
        assert len(events) == 5
        # Should be the last 5 events
        assert events[0]["task_id"] == "t15"
        assert events[-1]["task_id"] == "t19"

    def test_default_n(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        for i in range(3):
            al.log("event", task_id=f"t{i}")

        # Default n=50, should return all 3
        events = qa.get_recent_events()
        assert len(events) == 3

    def test_empty_when_no_events(self, data_layers):
        _, _, _, _, qa = data_layers
        assert qa.get_recent_events(10) == []


# ------------------------------------------------------------------
# get_task_stream_events
# ------------------------------------------------------------------


class TestGetTaskStreamEvents:
    def test_returns_stream_events_from_latest_attempt(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        # Log a dispatch and some stream events for attempt 1
        sl.log_dispatch("t1", 1, "prompt", "system", ["bash"], 1.0)
        sl.log_stream_event("t1", 1, {"type": "assistant", "content": "hello"})
        sl.log_stream_event("t1", 1, {"type": "tool_use", "tool": "bash"})
        sl.log_output("t1", 1, "done", 0, 10.0)

        events = qa.get_task_stream_events("t1")
        assert len(events) == 2
        assert all(e["type"] == "stream_event" for e in events)

    def test_returns_only_latest_attempt(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        # Attempt 1
        sl.log_stream_event("t1", 1, {"type": "assistant", "content": "attempt 1"})
        # Attempt 2
        sl.log_stream_event("t1", 2, {"type": "assistant", "content": "attempt 2"})
        sl.log_stream_event("t1", 2, {"type": "tool_use", "tool": "bash"})

        events = qa.get_task_stream_events("t1")
        assert len(events) == 2
        # Should be from attempt 2 only
        content_0 = json.loads(events[0]["content"])
        assert content_0["content"] == "attempt 2"

    def test_empty_when_no_session(self, data_layers):
        _, _, _, _, qa = data_layers
        assert qa.get_task_stream_events("nonexistent") == []

    def test_filters_out_non_stream_entries(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        sl.log_dispatch("t1", 1, "prompt", "system", ["bash"], 1.0)
        sl.log_stream_event("t1", 1, {"type": "assistant", "content": "work"})
        sl.log_output("t1", 1, "done", 0, 5.0)
        sl.log_validation("t1", 1, True, "all passed")

        events = qa.get_task_stream_events("t1")
        assert len(events) == 1
        assert events[0]["type"] == "stream_event"


# ------------------------------------------------------------------
# JSON serializability
# ------------------------------------------------------------------


class TestJsonSerializable:
    """All query results must be JSON-serializable dicts."""

    def test_active_plan_tasks_serializable(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        _create_task(ml, "t1", "task")
        ws.refresh()

        result = qa.get_active_plan_tasks()
        serialized = json.dumps(result)
        assert json.loads(serialized) == result

    def test_running_tasks_with_agents_serializable(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        _create_task(ml, "t1", "running")
        _assign_agent(ml, "t1", "a1")
        _start_task(ml, "t1")
        ws.refresh()

        result = qa.get_running_tasks_with_agents()
        serialized = json.dumps(result)
        assert json.loads(serialized) == result

    def test_ready_tasks_serializable(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        _create_task(ml, "t1", "ready")
        ws.refresh()

        result = qa.get_ready_tasks()
        serialized = json.dumps(result)
        assert json.loads(serialized) == result

    def test_blocked_tasks_serializable(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        _create_task(ml, "t1", "dep")
        _create_task(ml, "t2", "blocked", depends_on=["t1"])
        ws.refresh()

        result = qa.get_blocked_tasks_with_reasons()
        serialized = json.dumps(result)
        assert json.loads(serialized) == result

    def test_recent_events_serializable(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        al.log("test_event", task_id="t1", extra="data")

        result = qa.get_recent_events(10)
        serialized = json.dumps(result)
        assert json.loads(serialized) == result

    def test_task_stream_events_serializable(self, data_layers):
        ml, ws, al, sl, qa = data_layers

        sl.log_stream_event("t1", 1, {"type": "assistant", "content": "hello"})

        result = qa.get_task_stream_events("t1")
        serialized = json.dumps(result)
        assert json.loads(serialized) == result
