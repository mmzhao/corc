"""Tests for the executor module — pull_main integration and dispatch flow.

Verifies:
1. Executor calls pull_main before creating a worktree in dispatch
2. Executor logs the pull result as an audit event
3. Dispatch continues gracefully when pull_main fails
4. pull_main is skipped when reusing a conflict worktree
5. Executor's try_merge_worktree (PR path) pulls main after successful merge
6. Daemon pulls main after a successful PR merge via _handle_pr_based_merge
"""

import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from corc.audit import AuditLog
from corc.daemon import Daemon
from corc.dispatch import AgentDispatcher, AgentResult, Constraints
from corc.executor import CompletedTask, Executor
from corc.mutations import MutationLog
from corc.pr import PRInfo
from corc.processor import ProcessResult
from corc.sessions import SessionLogger
from corc.state import WorkState
from corc.worktree import create_worktree


# ---------------------------------------------------------------------------
# Mock dispatcher
# ---------------------------------------------------------------------------


class MockDispatcher(AgentDispatcher):
    """Returns a canned result without calling any LLM."""

    def __init__(self, result: AgentResult | None = None, delay: float = 0):
        self.result = result or AgentResult(
            output="Mock output: task done.",
            exit_code=0,
            duration_s=0.1,
        )
        self.delay = delay
        self.dispatched: list[tuple[str, str, Constraints]] = []

    def dispatch(
        self,
        prompt,
        system_prompt,
        constraints,
        pid_callback=None,
        event_callback=None,
        cwd=None,
    ):
        self.dispatched.append((prompt, system_prompt, constraints))
        if self.delay:
            time.sleep(self.delay)
        return self.result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path):
    """Create a real git repository for executor tests."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )
    # Initial commit so HEAD exists
    (repo / "README.md").write_text("# Test repo\n")
    subprocess.run(
        ["git", "add", "README.md"], cwd=str(repo), capture_output=True, check=True
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )

    # Set up corc directories
    (repo / ".corc").mkdir()
    (repo / "data").mkdir()
    (repo / "data" / "events").mkdir()
    (repo / "data" / "sessions").mkdir()
    return repo


@pytest.fixture
def mutation_log(git_repo):
    return MutationLog(git_repo / "data" / "mutations.jsonl")


@pytest.fixture
def work_state(git_repo, mutation_log):
    return WorkState(git_repo / "data" / "state.db", mutation_log)


@pytest.fixture
def audit_log(git_repo):
    return AuditLog(git_repo / "data" / "events")


@pytest.fixture
def session_logger(git_repo):
    return SessionLogger(git_repo / "data" / "sessions")


def _create_task(
    mutation_log,
    task_id,
    name,
    done_when="do the thing",
    depends_on=None,
    role="implementer",
    max_retries=3,
):
    """Helper to create a task via mutation log."""
    mutation_log.append(
        "task_created",
        {
            "id": task_id,
            "name": name,
            "description": f"Test task: {name}",
            "role": role,
            "depends_on": depends_on or [],
            "done_when": done_when,
            "checklist": [],
            "context_bundle": [],
            "max_retries": max_retries,
        },
        reason="Test setup",
    )


def _make_executor(
    git_repo,
    mutation_log,
    work_state,
    audit_log,
    session_logger,
    dispatcher=None,
    defer_merge=False,
):
    """Helper to build an Executor with sensible defaults."""
    return Executor(
        dispatcher=dispatcher or MockDispatcher(),
        mutation_log=mutation_log,
        state=work_state,
        audit_log=audit_log,
        session_logger=session_logger,
        project_root=git_repo,
        defer_merge=defer_merge,
    )


# ===========================================================================
# Pull main before dispatch
# ===========================================================================


class TestPullMainBeforeDispatch:
    """Executor.dispatch() must call pull_main before creating a worktree."""

    def test_pull_main_called_before_worktree_creation(
        self,
        git_repo,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
    ):
        """pull_main is invoked exactly once with the project root."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = _make_executor(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
        )

        with patch("corc.executor.pull_main", return_value=True) as mock_pull:
            executor.dispatch(task)
            time.sleep(0.5)
            executor.poll_completed()

        mock_pull.assert_called_once_with(git_repo)
        executor.shutdown()

    def test_pull_main_called_before_create_worktree(
        self,
        git_repo,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
    ):
        """pull_main is called strictly before create_worktree."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = _make_executor(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
        )

        call_order = []

        original_create_worktree = create_worktree

        def tracking_pull(project_root):
            call_order.append("pull_main")
            return True

        def tracking_create_wt(project_root, task_id, attempt):
            call_order.append("create_worktree")
            return original_create_worktree(project_root, task_id, attempt)

        with (
            patch("corc.executor.pull_main", side_effect=tracking_pull),
            patch("corc.executor.create_worktree", side_effect=tracking_create_wt),
        ):
            executor.dispatch(task)
            time.sleep(0.5)
            executor.poll_completed()

        assert call_order.index("pull_main") < call_order.index("create_worktree"), (
            f"pull_main must be called before create_worktree; order was {call_order}"
        )
        executor.shutdown()

    def test_pull_main_audit_event_logged(
        self,
        git_repo,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
    ):
        """Executor logs a main_pulled audit event with success status."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = _make_executor(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
        )

        with patch("corc.executor.pull_main", return_value=False):
            executor.dispatch(task)
            time.sleep(0.5)
            executor.poll_completed()

        events = audit_log.read_for_task("t1")
        pull_events = [e for e in events if e["event_type"] == "main_pulled"]
        assert len(pull_events) == 1
        assert pull_events[0]["success"] is False
        executor.shutdown()

    def test_dispatch_continues_when_pull_fails(
        self,
        git_repo,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
    ):
        """Dispatch proceeds normally even when pull_main returns False."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = _make_executor(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
        )

        with patch("corc.executor.pull_main", return_value=False):
            executor.dispatch(task)
            time.sleep(0.5)
            completed = executor.poll_completed()

        # Task was dispatched and completed despite pull failure
        assert len(completed) == 1
        work_state.refresh()
        assert work_state.get_task("t1")["status"] == "running"
        executor.shutdown()

    def test_pull_main_skipped_for_conflict_worktree(
        self,
        git_repo,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
    ):
        """pull_main is NOT called when reusing a saved conflict worktree."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = _make_executor(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
        )

        # Create a worktree and register it as a conflict worktree
        wt, _branch = create_worktree(git_repo, "t1", attempt=1)
        executor.set_conflict_worktree("t1", wt)

        with patch("corc.executor.pull_main") as mock_pull:
            executor.dispatch(task)
            time.sleep(0.5)
            executor.poll_completed()

        # pull_main should NOT have been called — conflict worktree reused
        mock_pull.assert_not_called()
        executor.shutdown()


# ===========================================================================
# Pull main after PR merge in executor
# ===========================================================================


class TestPullMainAfterPRMerge:
    """Executor._try_pr_merge pulls main after a successful gh pr merge."""

    def test_try_pr_merge_pulls_main_on_success(
        self,
        git_repo,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
    ):
        """After merge_pr succeeds, pull_main is called to sync local repo."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()

        wt, branch = create_worktree(git_repo, "t1", attempt=1)

        executor = _make_executor(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
            defer_merge=True,
        )

        pr_info = PRInfo(
            url="https://github.com/org/repo/pull/42",
            number=42,
            branch=branch,
            title="[corc] Task 1 (t1)",
        )

        with (
            patch("corc.executor.merge_pr", return_value=True),
            patch("corc.executor.pull_main", return_value=True) as mock_pull,
        ):
            status = executor.try_merge_worktree("t1", wt, pr_info=pr_info)

        assert status == "merged"
        mock_pull.assert_called_once_with(git_repo)
        executor.shutdown()

    def test_try_pr_merge_no_pull_on_failure(
        self,
        git_repo,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
    ):
        """When merge_pr fails, pull_main is NOT called."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()

        wt, branch = create_worktree(git_repo, "t1", attempt=1)

        executor = _make_executor(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
            defer_merge=True,
        )

        pr_info = PRInfo(
            url="https://github.com/org/repo/pull/42",
            number=42,
            branch=branch,
            title="[corc] Task 1 (t1)",
        )

        with (
            patch("corc.executor.merge_pr", return_value=False),
            patch("corc.executor.pull_main") as mock_pull,
        ):
            status = executor.try_merge_worktree("t1", wt, pr_info=pr_info)

        assert status == "conflict"
        mock_pull.assert_not_called()
        executor.shutdown()


# ===========================================================================
# Pull main after PR merge in daemon
# ===========================================================================


class TestDaemonPullMainAfterMerge:
    """Daemon._handle_pr_based_merge pulls main after processor merges a PR."""

    def _make_daemon(
        self, git_repo, mutation_log, work_state, audit_log, session_logger, dispatcher
    ):
        return Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=git_repo,
            poll_interval=0.1,
            auto_reload=False,
        )

    def test_daemon_pulls_main_after_pr_merged(
        self,
        git_repo,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
    ):
        """When proc_result.pr_merged is True, daemon calls pull_main."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()

        wt, branch = create_worktree(git_repo, "t1", attempt=1)

        daemon = self._make_daemon(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
            MockDispatcher(),
        )

        item = CompletedTask(
            task=work_state.get_task("t1"),
            result=AgentResult(output="Done", exit_code=0, duration_s=0.1),
            attempt=1,
            worktree_path=wt,
            agent_id="agent-test",
            pr_info=PRInfo(
                url="https://github.com/org/repo/pull/10",
                number=10,
                branch=branch,
                title="[corc] Task 1 (t1)",
            ),
        )

        proc_result = ProcessResult(
            task_id="t1",
            passed=True,
            details=[(True, "All checks pass")],
            pr_merged=True,
            pr_commented=True,
        )

        with patch("corc.daemon.pull_main", return_value=True) as mock_pull:
            daemon._handle_worktree_merge(item, proc_result)

        mock_pull.assert_called_once_with(git_repo)
        daemon.executor.shutdown()

    def test_daemon_no_pull_when_pr_not_merged_by_processor(
        self,
        git_repo,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
    ):
        """When processor did not merge the PR, daemon retries via executor
        (which handles its own pull_main internally)."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()

        wt, branch = create_worktree(git_repo, "t1", attempt=1)

        daemon = self._make_daemon(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
            MockDispatcher(),
        )

        item = CompletedTask(
            task=work_state.get_task("t1"),
            result=AgentResult(output="Done", exit_code=0, duration_s=0.1),
            attempt=1,
            worktree_path=wt,
            agent_id="agent-test",
            pr_info=PRInfo(
                url="https://github.com/org/repo/pull/10",
                number=10,
                branch=branch,
                title="[corc] Task 1 (t1)",
            ),
        )

        proc_result = ProcessResult(
            task_id="t1",
            passed=True,
            details=[(True, "All checks pass")],
            pr_merged=False,
            pr_commented=True,
        )

        with (
            patch("corc.daemon.pull_main") as mock_daemon_pull,
            # Mock executor's try_merge_worktree to avoid real git ops
            patch.object(daemon.executor, "try_merge_worktree", return_value="merged"),
            patch.object(daemon.executor, "cleanup_worktree"),
        ):
            daemon._handle_worktree_merge(item, proc_result)

        # Daemon's pull_main should NOT be called directly — executor handles it
        mock_daemon_pull.assert_not_called()
        daemon.executor.shutdown()

    def test_daemon_pull_main_records_merge_status(
        self,
        git_repo,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
    ):
        """After pulling main, daemon records merge_status = 'merged' in mutation log."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()

        wt, branch = create_worktree(git_repo, "t1", attempt=1)

        daemon = self._make_daemon(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
            MockDispatcher(),
        )

        item = CompletedTask(
            task=work_state.get_task("t1"),
            result=AgentResult(output="Done", exit_code=0, duration_s=0.1),
            attempt=1,
            worktree_path=wt,
            agent_id="agent-test",
            pr_info=PRInfo(
                url="https://github.com/org/repo/pull/10",
                number=10,
                branch=branch,
                title="[corc] Task 1 (t1)",
            ),
        )

        proc_result = ProcessResult(
            task_id="t1",
            passed=True,
            details=[(True, "All checks pass")],
            pr_merged=True,
            pr_commented=True,
        )

        with patch("corc.daemon.pull_main", return_value=True):
            daemon._handle_worktree_merge(item, proc_result)

        # Check mutation log for merge_status
        entries = mutation_log.read_all()
        updated = [e for e in entries if e.get("type") == "task_updated"]
        assert any(e["data"].get("merge_status") == "merged" for e in updated), (
            f"Expected merge_status='merged' in mutations; got {updated}"
        )

        # Audit log should show pr_merge_synced
        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "pr_merge_synced" in event_types
        daemon.executor.shutdown()


# ===========================================================================
# Merged PR detection before dispatch
# ===========================================================================


class TestMergedPRDetection:
    """Executor.dispatch() checks for merged PRs and skips dispatch if found."""

    def test_dispatch_skipped_when_merged_pr_exists(
        self,
        git_repo,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
    ):
        """If a merged PR exists for the task, dispatch is skipped and task is marked completed."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        merged_pr = PRInfo(
            url="https://github.com/org/repo/pull/99",
            number=99,
            branch="corc/t1-1",
            title="[corc] Task 1 (t1)",
        )

        dispatcher = MockDispatcher()
        executor = _make_executor(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
            dispatcher=dispatcher,
        )

        with patch("corc.executor.check_for_merged_pr", return_value=merged_pr):
            executor.dispatch(task)
            time.sleep(0.3)
            completed = executor.poll_completed()

        # No agent should have been dispatched
        assert len(dispatcher.dispatched) == 0
        # No in-flight tasks
        assert executor.in_flight_count == 0

        # Task should be marked completed via mutation log
        work_state.refresh()
        updated = work_state.get_task("t1")
        assert updated["status"] == "completed"

        # Check mutation log for the completion with PR info
        entries = mutation_log.read_all()
        completions = [e for e in entries if e.get("type") == "task_completed"]
        assert len(completions) == 1
        assert completions[0]["data"]["pr_url"] == "https://github.com/org/repo/pull/99"
        assert completions[0]["data"]["pr_number"] == 99
        assert completions[0]["data"]["already_merged"] is True

        # Audit log should show dispatch_skipped_merged_pr
        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "dispatch_skipped_merged_pr" in event_types

        executor.shutdown()

    def test_dispatch_proceeds_when_no_merged_pr(
        self,
        git_repo,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
    ):
        """If no merged PR exists, dispatch proceeds normally."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        dispatcher = MockDispatcher()
        executor = _make_executor(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
            dispatcher=dispatcher,
        )

        with (
            patch("corc.executor.check_for_merged_pr", return_value=None),
            patch("corc.executor.pull_main", return_value=True),
        ):
            executor.dispatch(task)
            time.sleep(0.5)
            completed = executor.poll_completed()

        # Agent should have been dispatched
        assert len(dispatcher.dispatched) == 1
        assert len(completed) == 1

        executor.shutdown()

    def test_merged_pr_check_uses_task_id(
        self,
        git_repo,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
    ):
        """check_for_merged_pr is called with the correct project_root and task_id."""
        _create_task(mutation_log, "my-task-42", "Task 42")
        work_state.refresh()
        task = work_state.get_task("my-task-42")

        dispatcher = MockDispatcher()
        executor = _make_executor(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
            dispatcher=dispatcher,
        )

        with (
            patch("corc.executor.check_for_merged_pr", return_value=None) as mock_check,
            patch("corc.executor.pull_main", return_value=True),
        ):
            executor.dispatch(task)
            time.sleep(0.3)
            executor.poll_completed()

        mock_check.assert_called_once_with(git_repo, "my-task-42")
        executor.shutdown()
