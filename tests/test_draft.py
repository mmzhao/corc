"""Tests for draft task status: creation, scheduler exclusion, approval flow, CLI commands."""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from corc.mutations import MutationLog
from corc.scheduler import get_ready_tasks as scheduler_get_ready
from corc.state import WorkState


@pytest.fixture
def state(tmp_path):
    ml = MutationLog(tmp_path / "mutations.jsonl")
    ws = WorkState(tmp_path / "state.db", ml)
    return ml, ws


# ---------------------------------------------------------------------------
# State layer — draft creation and storage
# ---------------------------------------------------------------------------


def test_task_created_with_draft_status(state):
    """Task created with status='draft' stores draft status."""
    ml, ws = state
    ml.append(
        "task_created",
        {"id": "t1", "name": "draft task", "done_when": "done", "status": "draft"},
        reason="test",
    )
    ws.refresh()
    task = ws.get_task("t1")
    assert task is not None
    assert task["status"] == "draft"


def test_task_created_defaults_to_pending(state):
    """Task created without explicit status defaults to pending."""
    ml, ws = state
    ml.append(
        "task_created",
        {"id": "t1", "name": "normal task", "done_when": "done"},
        reason="test",
    )
    ws.refresh()
    task = ws.get_task("t1")
    assert task["status"] == "pending"


def test_list_tasks_by_draft_status(state):
    """list_tasks(status='draft') returns only draft tasks."""
    ml, ws = state
    ml.append(
        "task_created",
        {"id": "t1", "name": "draft", "done_when": "done", "status": "draft"},
        reason="test",
    )
    ml.append(
        "task_created",
        {"id": "t2", "name": "pending", "done_when": "done"},
        reason="test",
    )
    ws.refresh()

    drafts = ws.list_tasks(status="draft")
    assert len(drafts) == 1
    assert drafts[0]["id"] == "t1"


# ---------------------------------------------------------------------------
# Approval mutation
# ---------------------------------------------------------------------------


def test_task_approved_flips_draft_to_pending(state):
    """task_approved mutation changes status from draft to pending."""
    ml, ws = state
    ml.append(
        "task_created",
        {"id": "t1", "name": "draft task", "done_when": "done", "status": "draft"},
        reason="test",
    )
    ws.refresh()
    assert ws.get_task("t1")["status"] == "draft"

    ml.append("task_approved", {}, reason="approved", task_id="t1")
    ws.refresh()
    assert ws.get_task("t1")["status"] == "pending"


def test_approve_all_drafts(state):
    """Multiple draft tasks can be approved individually."""
    ml, ws = state
    for i in range(3):
        ml.append(
            "task_created",
            {
                "id": f"t{i}",
                "name": f"draft {i}",
                "done_when": "done",
                "status": "draft",
            },
            reason="test",
        )
    ws.refresh()
    assert len(ws.list_tasks(status="draft")) == 3

    for i in range(3):
        ml.append("task_approved", {}, reason="approved all", task_id=f"t{i}")
    ws.refresh()

    assert len(ws.list_tasks(status="draft")) == 0
    assert len(ws.list_tasks(status="pending")) == 3


# ---------------------------------------------------------------------------
# Scheduler exclusion — draft tasks are never dispatched
# ---------------------------------------------------------------------------


def test_draft_not_in_ready_tasks(state):
    """Draft tasks are excluded from get_ready_tasks."""
    ml, ws = state
    ml.append(
        "task_created",
        {"id": "t1", "name": "draft task", "done_when": "done", "status": "draft"},
        reason="test",
    )
    ws.refresh()

    ready = ws.get_ready_tasks()
    assert len(ready) == 0


def test_scheduler_never_dispatches_draft(state):
    """Scheduler get_ready_tasks never returns draft tasks."""
    ml, ws = state
    ml.append(
        "task_created",
        {"id": "t1", "name": "draft task", "done_when": "done", "status": "draft"},
        reason="test",
    )
    ws.refresh()

    ready = scheduler_get_ready(ws, parallel_limit=10)
    assert len(ready) == 0


def test_draft_becomes_schedulable_after_approval(state):
    """After approval, a former draft task appears in ready tasks."""
    ml, ws = state
    ml.append(
        "task_created",
        {"id": "t1", "name": "draft task", "done_when": "done", "status": "draft"},
        reason="test",
    )
    ws.refresh()
    assert len(ws.get_ready_tasks()) == 0

    ml.append("task_approved", {}, reason="approved", task_id="t1")
    ws.refresh()
    ready = ws.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0]["id"] == "t1"


def test_mixed_draft_and_pending_scheduling(state):
    """Only pending tasks are scheduled; drafts are excluded."""
    ml, ws = state
    ml.append(
        "task_created",
        {"id": "t1", "name": "draft", "done_when": "done", "status": "draft"},
        reason="test",
    )
    ml.append(
        "task_created",
        {"id": "t2", "name": "pending", "done_when": "done"},
        reason="test",
    )
    ws.refresh()

    ready = ws.get_ready_tasks()
    assert len(ready) == 1
    assert ready[0]["id"] == "t2"


def test_draft_with_dependencies_not_scheduled(state):
    """Draft task with met dependencies is still not scheduled."""
    ml, ws = state
    ml.append(
        "task_created",
        {"id": "t1", "name": "dep", "done_when": "done"},
        reason="test",
    )
    ml.append(
        "task_completed",
        {"pr_url": None, "findings": []},
        reason="done",
        task_id="t1",
    )
    ml.append(
        "task_created",
        {
            "id": "t2",
            "name": "draft with deps",
            "done_when": "done",
            "status": "draft",
            "depends_on": ["t1"],
        },
        reason="test",
    )
    ws.refresh()

    ready = ws.get_ready_tasks()
    assert all(t["id"] != "t2" for t in ready)


# ---------------------------------------------------------------------------
# Rebuild / replay durability
# ---------------------------------------------------------------------------


def test_draft_survives_rebuild(state):
    """Draft status is correctly replayed from mutation log after full rebuild."""
    ml, ws = state
    ml.append(
        "task_created",
        {"id": "t1", "name": "draft", "done_when": "done", "status": "draft"},
        reason="test",
    )
    ws.refresh()
    assert ws.get_task("t1")["status"] == "draft"

    ws.rebuild()
    assert ws.get_task("t1")["status"] == "draft"


def test_approval_survives_rebuild(state):
    """Approval mutation is replayed correctly after rebuild."""
    ml, ws = state
    ml.append(
        "task_created",
        {"id": "t1", "name": "draft", "done_when": "done", "status": "draft"},
        reason="test",
    )
    ml.append("task_approved", {}, reason="approved", task_id="t1")
    ws.refresh()
    assert ws.get_task("t1")["status"] == "pending"

    ws.rebuild()
    assert ws.get_task("t1")["status"] == "pending"


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    """Set up a minimal CORC environment for CLI testing."""
    corc_dir = tmp_path / ".corc"
    corc_dir.mkdir()
    (corc_dir / "events").mkdir()
    (corc_dir / "sessions").mkdir()
    knowledge_dir = corc_dir / "knowledge"
    knowledge_dir.mkdir()

    monkeypatch.setattr(
        "corc.cli.get_paths",
        lambda: {
            "root": tmp_path,
            "corc_dir": corc_dir,
            "mutations": corc_dir / "mutations.jsonl",
            "state_db": corc_dir / "state.db",
            "events_dir": corc_dir / "events",
            "sessions_dir": corc_dir / "sessions",
            "knowledge_dir": knowledge_dir,
            "knowledge_db": corc_dir / "knowledge.db",
        },
    )

    return tmp_path


def test_cli_create_draft(cli_env):
    """corc task create --draft creates task with draft status."""
    from corc.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli, ["task", "create", "my-task", "--done-when", "tests pass", "--draft"]
    )
    assert result.exit_code == 0
    assert "[draft]" in result.output

    # Verify it's actually in draft status
    result2 = runner.invoke(cli, ["task", "list", "--draft"])
    assert result2.exit_code == 0
    assert "my-task" in result2.output
    assert "draft" in result2.output


def test_cli_create_without_draft_is_pending(cli_env):
    """corc task create without --draft creates pending task."""
    from corc.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli, ["task", "create", "my-task", "--done-when", "tests pass"]
    )
    assert result.exit_code == 0
    assert "[draft]" not in result.output

    result2 = runner.invoke(cli, ["task", "list", "--draft"])
    assert result2.exit_code == 0
    assert "No tasks found." in result2.output


def test_cli_approve_single(cli_env):
    """corc task approve TASK_ID flips draft to pending."""
    from corc.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli, ["task", "create", "my-draft", "--done-when", "done", "--draft"]
    )
    assert result.exit_code == 0
    # Extract task ID from output: "Created task XXXXXXXX: ..."
    task_id = result.output.split("Created task ")[1].split(":")[0].strip()

    result2 = runner.invoke(cli, ["task", "approve", task_id])
    assert result2.exit_code == 0
    assert "Approved" in result2.output

    # Verify it's now pending, not draft
    result3 = runner.invoke(cli, ["task", "list", "--draft"])
    assert "No tasks found." in result3.output

    result4 = runner.invoke(cli, ["task", "list", "--status", "pending"])
    assert task_id in result4.output


def test_cli_approve_all(cli_env):
    """corc task approve --all approves all draft tasks."""
    from corc.cli import cli

    runner = CliRunner()
    ids = []
    for i in range(3):
        result = runner.invoke(
            cli, ["task", "create", f"draft-{i}", "--done-when", "done", "--draft"]
        )
        assert result.exit_code == 0
        tid = result.output.split("Created task ")[1].split(":")[0].strip()
        ids.append(tid)

    result2 = runner.invoke(cli, ["task", "approve", "--all"])
    assert result2.exit_code == 0
    assert "Approved 3 draft task(s)." in result2.output

    # All should now be pending
    result3 = runner.invoke(cli, ["task", "list", "--draft"])
    assert "No tasks found." in result3.output


def test_cli_approve_nonexistent_task(cli_env):
    """corc task approve with nonexistent task ID errors."""
    from corc.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["task", "approve", "nonexist"])
    assert result.exit_code != 0
    assert "not found" in result.output


def test_cli_approve_non_draft_task(cli_env):
    """corc task approve on a pending task errors."""
    from corc.cli import cli

    runner = CliRunner()
    result = runner.invoke(
        cli, ["task", "create", "pending-task", "--done-when", "done"]
    )
    task_id = result.output.split("Created task ")[1].split(":")[0].strip()

    result2 = runner.invoke(cli, ["task", "approve", task_id])
    assert result2.exit_code != 0
    assert "not 'draft'" in result2.output


def test_cli_approve_no_args(cli_env):
    """corc task approve without args or --all errors."""
    from corc.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["task", "approve"])
    assert result.exit_code != 0


def test_cli_approve_all_no_drafts(cli_env):
    """corc task approve --all with no drafts prints message."""
    from corc.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["task", "approve", "--all"])
    assert result.exit_code == 0
    assert "No draft tasks" in result.output


def test_cli_list_draft_shows_only_drafts(cli_env):
    """corc task list --draft shows only draft tasks, not pending."""
    from corc.cli import cli

    runner = CliRunner()
    runner.invoke(
        cli, ["task", "create", "draft-task", "--done-when", "done", "--draft"]
    )
    runner.invoke(cli, ["task", "create", "pending-task", "--done-when", "done"])

    result = runner.invoke(cli, ["task", "list", "--draft"])
    assert result.exit_code == 0
    assert "draft-task" in result.output
    assert "pending-task" not in result.output


# ---------------------------------------------------------------------------
# Integration: full draft → approve → dispatch flow
# ---------------------------------------------------------------------------


def test_full_draft_approve_dispatch_flow(state):
    """End-to-end: draft → not scheduled → approve → scheduled."""
    ml, ws = state

    # Create draft
    ml.append(
        "task_created",
        {"id": "t1", "name": "feature X", "done_when": "tests pass", "status": "draft"},
        reason="test",
    )
    ws.refresh()

    # Not schedulable
    assert ws.get_task("t1")["status"] == "draft"
    assert len(scheduler_get_ready(ws, parallel_limit=10)) == 0

    # Approve
    ml.append("task_approved", {}, reason="reviewed and approved", task_id="t1")
    ws.refresh()

    # Now schedulable
    assert ws.get_task("t1")["status"] == "pending"
    ready = scheduler_get_ready(ws, parallel_limit=10)
    assert len(ready) == 1
    assert ready[0]["id"] == "t1"
