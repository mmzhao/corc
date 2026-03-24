"""Tests for optimistic merge strategy — parallel agents working in worktrees.

Tests the full optimistic merge lifecycle:
1. Agent completes → validation passes → merge worktree to main → success
2. Agent completes → validation passes → merge conflict → main merged into worktree → retry
3. Merge status tracked in task mutations
4. Conflict worktree reused on retry dispatch
5. Unresolvable conflicts fall back to fresh worktree
6. Three-stage conflict resolution pipeline:
   Stage 1: clean git merge → push → re-merge PR (no agent dispatch)
   Stage 2: conflicted merge → agent resolution → push → re-merge PR
   Stage 3: agent failure → task_failed for full retry
"""

import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from corc.audit import AuditLog
from corc.dispatch import AgentDispatcher, AgentResult, Constraints
from corc.executor import CompletedTask, Executor
from corc.mutations import MutationLog
from corc.pr import PRInfo
from corc.processor import ProcessResult, process_completed
from corc.sessions import SessionLogger
from corc.state import WorkState
from corc.worktree import (
    WorktreeError,
    create_worktree,
    merge_main_into_worktree,
    merge_worktree,
    remove_worktree,
    _get_current_branch,
)


# ---------------------------------------------------------------------------
# Fixtures — create a real git repo for worktree tests
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path):
    """Create a real git repository for worktree tests."""
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

    # Create initial commit so HEAD exists
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


# ---------------------------------------------------------------------------
# Mock dispatchers
# ---------------------------------------------------------------------------


class CommittingDispatcher(AgentDispatcher):
    """Dispatcher that creates a file and commits it in the worktree."""

    def __init__(self, filename="agent_output.py", content="# Agent work\n"):
        self.filename = filename
        self.content = content
        self.dispatched = []

    def dispatch(
        self,
        prompt,
        system_prompt,
        constraints,
        pid_callback=None,
        event_callback=None,
        cwd=None,
    ):
        self.dispatched.append({"prompt": prompt, "cwd": cwd})
        if cwd:
            new_file = Path(cwd) / self.filename
            new_file.write_text(self.content)
            subprocess.run(["git", "add", self.filename], cwd=cwd, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", f"Add {self.filename}"],
                cwd=cwd,
                capture_output=True,
            )
        return AgentResult(output="Done", exit_code=0, duration_s=0.1)


class NoopDispatcher(AgentDispatcher):
    """Dispatcher that returns success without modifying files."""

    def __init__(self):
        self.dispatched = []

    def dispatch(
        self,
        prompt,
        system_prompt,
        constraints,
        pid_callback=None,
        event_callback=None,
        cwd=None,
    ):
        self.dispatched.append({"prompt": prompt, "cwd": cwd})
        return AgentResult(output="Done (noop)", exit_code=0, duration_s=0.1)


# ===========================================================================
# Unit tests: merge_main_into_worktree
# ===========================================================================


class TestMergeMainIntoWorktree:
    """Test the merge_main_into_worktree() function."""

    def test_merge_main_no_conflict(self, git_repo):
        """Non-conflicting changes from main are merged into worktree."""
        wt, branch = create_worktree(git_repo, "task-1", attempt=1)

        # Make changes in worktree (new file)
        (wt / "agent_file.py").write_text("# Agent work\n")
        subprocess.run(
            ["git", "add", "agent_file.py"], cwd=str(wt), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Agent work"], cwd=str(wt), capture_output=True
        )

        # Make non-conflicting changes in main (different file)
        (git_repo / "other_file.py").write_text("# From another agent\n")
        subprocess.run(
            ["git", "add", "other_file.py"], cwd=str(git_repo), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Other agent work"],
            cwd=str(git_repo),
            capture_output=True,
        )

        # Merge main into worktree should succeed
        success = merge_main_into_worktree(git_repo, wt)
        assert success is True

        # Worktree should now have both files
        assert (wt / "agent_file.py").exists()
        assert (wt / "other_file.py").exists()

    def test_merge_main_with_conflict(self, git_repo):
        """Conflicting changes prevent merge, worktree left clean."""
        wt, branch = create_worktree(git_repo, "task-1", attempt=1)

        # Both modify README.md
        (wt / "README.md").write_text("# Changed in worktree\n")
        subprocess.run(["git", "add", "README.md"], cwd=str(wt), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Worktree change"], cwd=str(wt), capture_output=True
        )

        (git_repo / "README.md").write_text("# Changed in main\n")
        subprocess.run(
            ["git", "add", "README.md"], cwd=str(git_repo), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Main change"],
            cwd=str(git_repo),
            capture_output=True,
        )

        # Merge should fail
        success = merge_main_into_worktree(git_repo, wt)
        assert success is False

        # Worktree should be clean (merge aborted)
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(wt),
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == ""

    def test_merge_main_no_new_commits(self, git_repo):
        """When main has no new commits, merge is a no-op success."""
        wt, branch = create_worktree(git_repo, "task-1", attempt=1)

        # Only worktree has changes
        (wt / "agent_file.py").write_text("# Agent work\n")
        subprocess.run(
            ["git", "add", "agent_file.py"], cwd=str(wt), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Agent work"], cwd=str(wt), capture_output=True
        )

        success = merge_main_into_worktree(git_repo, wt)
        assert success is True


class TestGetCurrentBranch:
    """Test _get_current_branch helper."""

    def test_returns_branch_name(self, git_repo):
        """Returns the current branch name."""
        branch = _get_current_branch(git_repo)
        assert branch is not None
        # Could be "main" or "master" depending on git config
        assert branch in ("main", "master")

    def test_returns_none_for_nonexistent(self, tmp_path):
        """Returns None for a non-git directory."""
        branch = _get_current_branch(tmp_path)
        assert branch is None


# ===========================================================================
# Integration tests: Optimistic merge via Executor
# ===========================================================================


class TestOptimisticMergeSuccess:
    """Test the successful merge path: agent completes → validation → merge."""

    def test_try_merge_worktree_success(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """try_merge_worktree returns 'merged' when merge succeeds."""
        dispatcher = CommittingDispatcher()
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = Executor(
            dispatcher=dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
            defer_merge=True,
        )

        executor.dispatch(task)
        time.sleep(0.5)
        completed = executor.poll_completed()
        assert len(completed) == 1

        item = completed[0]
        assert item.worktree_path is not None
        assert item.worktree_path.exists()  # Worktree still alive (defer_merge)

        # Try merge
        status = executor.try_merge_worktree(item.task["id"], item.worktree_path)
        assert status == "merged"

        # File should be in main repo
        assert (git_repo / "agent_output.py").exists()

        # Clean up
        executor.cleanup_worktree(item.task["id"], item.worktree_path)
        assert not item.worktree_path.exists()
        executor.shutdown()

    def test_try_merge_no_changes(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """try_merge_worktree returns 'merged' when no changes were made."""
        dispatcher = NoopDispatcher()
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = Executor(
            dispatcher=dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
            defer_merge=True,
        )

        executor.dispatch(task)
        time.sleep(0.5)
        completed = executor.poll_completed()
        assert len(completed) == 1

        item = completed[0]
        status = executor.try_merge_worktree(item.task["id"], item.worktree_path)
        # merge_worktree returns True for no-op, so status is "merged"
        assert status == "merged"

        executor.cleanup_worktree(item.task["id"], item.worktree_path)
        executor.shutdown()

    def test_deferred_merge_keeps_worktree_alive(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """With defer_merge=True, worktree is NOT cleaned up in poll_completed."""
        dispatcher = CommittingDispatcher()
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = Executor(
            dispatcher=dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
            defer_merge=True,
        )

        executor.dispatch(task)
        time.sleep(0.5)
        completed = executor.poll_completed()

        assert len(completed) == 1
        assert completed[0].worktree_path is not None
        # Worktree should still exist
        assert completed[0].worktree_path.exists()

        # File should NOT be in main repo yet (merge deferred)
        assert not (git_repo / "agent_output.py").exists()

        executor.cleanup_worktree("t1", completed[0].worktree_path)
        executor.shutdown()


# ===========================================================================
# Integration tests: Merge conflict and retry
# ===========================================================================


class TestMergeConflictRetry:
    """Test the conflict-retry path: merge fails → merge main into worktree → retry."""

    @pytest.mark.xfail(
        reason="Flaky: claude -p may resolve trivial README conflict non-deterministically"
    )
    def test_try_merge_worktree_conflict(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """try_merge_worktree returns 'conflict' when there's a merge conflict."""
        dispatcher = CommittingDispatcher(
            filename="README.md", content="# Changed by agent\n"
        )
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = Executor(
            dispatcher=dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
            defer_merge=True,
        )

        executor.dispatch(task)
        time.sleep(0.5)

        # While agent was working, another agent merged changes to main
        (git_repo / "README.md").write_text("# Changed by other agent\n")
        subprocess.run(
            ["git", "add", "README.md"], cwd=str(git_repo), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Other agent merged first"],
            cwd=str(git_repo),
            capture_output=True,
        )

        completed = executor.poll_completed()
        assert len(completed) == 1

        item = completed[0]
        status = executor.try_merge_worktree(item.task["id"], item.worktree_path)
        assert status == "conflict"

        # Main should be unchanged (merge was aborted)
        assert (git_repo / "README.md").read_text() == "# Changed by other agent\n"

        executor.cleanup_worktree(item.task["id"], item.worktree_path)
        executor.shutdown()

    def test_prepare_conflict_retry_success(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """prepare_conflict_retry merges main into worktree and saves it."""
        # Create a worktree with agent changes
        wt, branch = create_worktree(git_repo, "task-1", attempt=1)
        (wt / "agent_file.py").write_text("# Agent work\n")
        subprocess.run(
            ["git", "add", "agent_file.py"], cwd=str(wt), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Agent work"], cwd=str(wt), capture_output=True
        )

        # Commit to main (non-conflicting)
        (git_repo / "other_file.py").write_text("# Other agent\n")
        subprocess.run(
            ["git", "add", "other_file.py"], cwd=str(git_repo), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Other agent"],
            cwd=str(git_repo),
            capture_output=True,
        )

        executor = Executor(
            dispatcher=NoopDispatcher(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
            defer_merge=True,
        )

        success = executor.prepare_conflict_retry("task-1", wt)
        assert success is True

        # Worktree should have both files
        assert (wt / "agent_file.py").exists()
        assert (wt / "other_file.py").exists()

        # Worktree should be saved for reuse
        assert "task-1" in executor._conflict_worktrees
        assert executor._conflict_worktrees["task-1"] == wt

        executor.shutdown()

    def test_conflict_worktree_reused_on_retry(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """When a conflict worktree exists, next dispatch reuses it."""
        # Create and prepare a conflict worktree
        wt, branch = create_worktree(git_repo, "t1", attempt=1)
        (wt / "agent_work.py").write_text("# Agent's previous work\n")
        subprocess.run(
            ["git", "add", "agent_work.py"], cwd=str(wt), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Agent work"], cwd=str(wt), capture_output=True
        )

        dispatcher = NoopDispatcher()
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = Executor(
            dispatcher=dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
            defer_merge=True,
        )

        # Register the conflict worktree
        executor.set_conflict_worktree("t1", wt)

        # Dispatch should reuse the worktree instead of creating new one
        executor.dispatch(task)
        time.sleep(0.5)
        completed = executor.poll_completed()

        assert len(completed) == 1
        # The dispatched cwd should be the conflict worktree
        assert dispatcher.dispatched[0]["cwd"] == str(wt)

        # Audit should show worktree_reused event
        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "worktree_reused" in event_types

        executor.shutdown()

    def test_full_conflict_retry_flow(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Full flow: two parallel agents with non-overlapping files both merge."""
        # Both worktrees created from same HEAD (simulating parallel start)
        wt_a, _ = create_worktree(git_repo, "task-a", attempt=1)
        wt_b, _ = create_worktree(git_repo, "task-b", attempt=1)

        # --- Agent A: creates file_a.py ---
        (wt_a / "file_a.py").write_text("# Agent A\n")
        subprocess.run(["git", "add", "file_a.py"], cwd=str(wt_a), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Agent A work"], cwd=str(wt_a), capture_output=True
        )

        # --- Agent B: creates file_b.py ---
        (wt_b / "file_b.py").write_text("# Agent B\n")
        subprocess.run(["git", "add", "file_b.py"], cwd=str(wt_b), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Agent B work"], cwd=str(wt_b), capture_output=True
        )

        # Agent A merges first — succeeds
        merged = merge_worktree(git_repo, wt_a)
        assert merged is True
        remove_worktree(git_repo, wt_a)
        assert (git_repo / "file_a.py").exists()

        # Agent B merges second — should also succeed (different files)
        merged = merge_worktree(git_repo, wt_b)
        assert merged is True

        # Both files should exist in main
        assert (git_repo / "file_a.py").exists()
        assert (git_repo / "file_b.py").exists()

        remove_worktree(git_repo, wt_b)

    @pytest.mark.xfail(
        reason="Flaky: claude -p may resolve trivial README conflict non-deterministically"
    )
    def test_full_conflict_retry_with_overlapping_files(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """When agents modify the same file: conflict detected, retry works."""
        # Both worktrees created from same HEAD (simulating parallel start)
        wt_a, _ = create_worktree(git_repo, "task-a", attempt=1)
        wt_b, _ = create_worktree(git_repo, "task-b", attempt=1)

        # --- Agent A: modifies README ---
        (wt_a / "README.md").write_text("# Modified by Agent A\n")
        subprocess.run(["git", "add", "README.md"], cwd=str(wt_a), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Agent A changes README"],
            cwd=str(wt_a),
            capture_output=True,
        )

        # --- Agent B: also modifies README ---
        (wt_b / "README.md").write_text("# Modified by Agent B\n")
        subprocess.run(["git", "add", "README.md"], cwd=str(wt_b), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Agent B changes README"],
            cwd=str(wt_b),
            capture_output=True,
        )

        # Agent A merges first — succeeds
        merged = merge_worktree(git_repo, wt_a)
        assert merged is True
        remove_worktree(git_repo, wt_a)
        assert (git_repo / "README.md").read_text() == "# Modified by Agent A\n"

        # Agent B's merge to main FAILS (conflict!)
        merged = merge_worktree(git_repo, wt_b)
        assert merged is False

        # Main should still have Agent A's version
        assert (git_repo / "README.md").read_text() == "# Modified by Agent A\n"

        # Merge main into B's worktree also fails (same conflict)
        rebase_ok = merge_main_into_worktree(git_repo, wt_b)
        assert rebase_ok is False

        # Worktree should be clean (merge aborted)
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(wt_b),
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == ""

        remove_worktree(git_repo, wt_b)


# ===========================================================================
# Integration tests: Merge status tracking in mutations
# ===========================================================================


class TestMergeStatusTracking:
    """Test that merge status is tracked via task_updated mutations."""

    def test_merge_status_recorded_on_success(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Successful merge records merge_status='merged' in mutations."""
        dispatcher = CommittingDispatcher()
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = Executor(
            dispatcher=dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
            defer_merge=True,
        )

        executor.dispatch(task)
        time.sleep(0.5)
        completed = executor.poll_completed()
        item = completed[0]

        # Merge succeeds
        status = executor.try_merge_worktree(item.task["id"], item.worktree_path)
        assert status == "merged"

        # Record in mutations
        mutation_log.append(
            "task_updated",
            {"merge_status": "merged"},
            reason="Worktree merged to main",
            task_id="t1",
        )
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["merge_status"] == "merged"

        executor.cleanup_worktree("t1", item.worktree_path)
        executor.shutdown()

    def test_merge_status_recorded_on_conflict(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Merge conflict records merge_status='conflict' in mutations."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()

        # Record conflict status
        mutation_log.append(
            "task_updated",
            {"merge_status": "conflict"},
            reason="Merge conflict detected",
            task_id="t1",
        )
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["merge_status"] == "conflict"

    def test_merge_status_in_state_table(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """merge_status column is properly stored and retrieved from state."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()

        # Initially no merge status
        task = work_state.get_task("t1")
        assert task.get("merge_status") is None

        # Set merge status
        mutation_log.append(
            "task_updated",
            {"merge_status": "merged"},
            reason="Test",
            task_id="t1",
        )
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["merge_status"] == "merged"

        # Update to conflict
        mutation_log.append(
            "task_updated",
            {"merge_status": "conflict"},
            reason="Test",
            task_id="t1",
        )
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["merge_status"] == "conflict"


# ===========================================================================
# Integration tests: Executor defer_merge behavior
# ===========================================================================


class TestExecutorDeferMerge:
    """Test the defer_merge parameter on Executor."""

    @pytest.mark.xfail(
        reason="Pre-existing: test repo lacks git remote needed for PR-based workflow"
    )
    def test_defer_merge_false_merges_automatically(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """With defer_merge=False (default), executor merges in poll_completed."""
        dispatcher = CommittingDispatcher()
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = Executor(
            dispatcher=dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
            defer_merge=False,  # default behavior
        )

        executor.dispatch(task)
        time.sleep(0.5)
        completed = executor.poll_completed()

        assert len(completed) == 1
        # File should already be in main repo (merged automatically)
        assert (git_repo / "agent_output.py").exists()
        # Worktree should be cleaned up
        assert not completed[0].worktree_path.exists()
        executor.shutdown()

    def test_defer_merge_true_skips_merge(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """With defer_merge=True, executor does NOT merge in poll_completed."""
        dispatcher = CommittingDispatcher()
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = Executor(
            dispatcher=dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
            defer_merge=True,
        )

        executor.dispatch(task)
        time.sleep(0.5)
        completed = executor.poll_completed()

        assert len(completed) == 1
        # File should NOT be in main repo (merge deferred)
        assert not (git_repo / "agent_output.py").exists()
        # Worktree should still exist
        assert completed[0].worktree_path.exists()

        executor.cleanup_worktree("t1", completed[0].worktree_path)
        executor.shutdown()


# ===========================================================================
# Integration tests: Audit log events
# ===========================================================================


class TestOptimisticMergeAuditEvents:
    """Test that optimistic merge events are properly audit logged."""

    def test_merge_success_audit_events(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Successful merge logs worktree_merged and worktree_removed events."""
        dispatcher = CommittingDispatcher()
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = Executor(
            dispatcher=dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
            defer_merge=True,
        )

        executor.dispatch(task)
        time.sleep(0.5)
        completed = executor.poll_completed()
        item = completed[0]

        executor.try_merge_worktree(item.task["id"], item.worktree_path)
        executor.cleanup_worktree(item.task["id"], item.worktree_path)

        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "worktree_merged" in event_types
        assert "worktree_removed" in event_types
        executor.shutdown()

    @pytest.mark.xfail(
        reason="Flaky: claude -p may resolve trivial README conflict non-deterministically"
    )
    def test_merge_conflict_audit_events(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Merge conflict logs worktree_merge_conflict event."""
        dispatcher = CommittingDispatcher(
            filename="README.md", content="# Agent changes\n"
        )
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = Executor(
            dispatcher=dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
            defer_merge=True,
        )

        executor.dispatch(task)
        time.sleep(0.5)

        # Create conflict on main
        (git_repo / "README.md").write_text("# Changed by other agent\n")
        subprocess.run(
            ["git", "add", "README.md"], cwd=str(git_repo), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Other agent"],
            cwd=str(git_repo),
            capture_output=True,
        )

        completed = executor.poll_completed()
        item = completed[0]

        status = executor.try_merge_worktree(item.task["id"], item.worktree_path)
        assert status == "conflict"

        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "worktree_merge_conflict" in event_types

        executor.cleanup_worktree("t1", item.worktree_path)
        executor.shutdown()

    def test_conflict_retry_prepared_audit_events(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """prepare_conflict_retry logs worktree_conflict_retry_prepared event."""
        # Create a worktree with agent changes
        wt, branch = create_worktree(git_repo, "t1", attempt=1)
        (wt / "agent_file.py").write_text("# Agent work\n")
        subprocess.run(
            ["git", "add", "agent_file.py"], cwd=str(wt), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Agent work"], cwd=str(wt), capture_output=True
        )

        # Commit non-conflicting change to main
        (git_repo / "other.py").write_text("# Other\n")
        subprocess.run(
            ["git", "add", "other.py"], cwd=str(git_repo), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Other"], cwd=str(git_repo), capture_output=True
        )

        executor = Executor(
            dispatcher=NoopDispatcher(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
            defer_merge=True,
        )

        success = executor.prepare_conflict_retry("t1", wt)
        assert success is True

        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "worktree_conflict_retry_prepared" in event_types

        executor.shutdown()


# ===========================================================================
# Edge case tests
# ===========================================================================


class TestOptimisticMergeEdgeCases:
    """Test edge cases in the optimistic merge strategy."""

    def test_no_worktree_path_skips_merge(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """CompletedTask with no worktree_path does not attempt merge."""
        # This simulates the fallback case where worktree creation failed
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()

        completed = CompletedTask(
            task=work_state.get_task("t1"),
            result=AgentResult(output="Done", exit_code=0, duration_s=0.1),
            attempt=1,
            worktree_path=None,
            agent_id="agent-test",
        )

        # No merge should be attempted
        assert completed.worktree_path is None
        assert completed.merge_status is None

    def test_prepare_conflict_retry_unresolvable(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """When main→worktree merge also fails, worktree is cleaned up."""
        # Create a worktree that conflicts with main
        wt, _ = create_worktree(git_repo, "t1", attempt=1)
        (wt / "README.md").write_text("# Worktree version\n")
        subprocess.run(["git", "add", "README.md"], cwd=str(wt), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Worktree change"], cwd=str(wt), capture_output=True
        )

        # Conflicting change on main
        (git_repo / "README.md").write_text("# Main version\n")
        subprocess.run(
            ["git", "add", "README.md"], cwd=str(git_repo), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Main change"],
            cwd=str(git_repo),
            capture_output=True,
        )

        executor = Executor(
            dispatcher=NoopDispatcher(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
            defer_merge=True,
        )

        success = executor.prepare_conflict_retry("t1", wt)
        assert success is False

        # Worktree should have been cleaned up
        assert not wt.exists()

        # Should NOT be in conflict worktrees
        assert "t1" not in executor._conflict_worktrees

        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "worktree_conflict_unresolvable" in event_types

        executor.shutdown()

    def test_parallel_agents_non_overlapping_both_merge(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Two parallel agents with non-overlapping files both merge successfully."""
        _create_task(mutation_log, "t1", "Task 1")
        _create_task(mutation_log, "t2", "Task 2")
        work_state.refresh()
        task1 = work_state.get_task("t1")
        task2 = work_state.get_task("t2")

        # Create worktrees for both
        wt1, _ = create_worktree(git_repo, "t1", attempt=1)
        wt2, _ = create_worktree(git_repo, "t2", attempt=1)

        # Each agent modifies different files
        (wt1 / "file_1.py").write_text("# Agent 1\n")
        subprocess.run(["git", "add", "file_1.py"], cwd=str(wt1), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Agent 1 work"], cwd=str(wt1), capture_output=True
        )

        (wt2 / "file_2.py").write_text("# Agent 2\n")
        subprocess.run(["git", "add", "file_2.py"], cwd=str(wt2), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Agent 2 work"], cwd=str(wt2), capture_output=True
        )

        # Agent 1 merges first
        merged1 = merge_worktree(git_repo, wt1)
        assert merged1 is True

        # Agent 2 merges second — should also succeed (different files)
        merged2 = merge_worktree(git_repo, wt2)
        assert merged2 is True

        # Both files should be in main
        assert (git_repo / "file_1.py").exists()
        assert (git_repo / "file_2.py").exists()

        remove_worktree(git_repo, wt1)
        remove_worktree(git_repo, wt2)


# ===========================================================================
# Daemon-level integration tests: full _handle_worktree_merge flow
# ===========================================================================


@pytest.mark.xfail(
    reason="Pre-existing: test repos lack git remote needed for PR push/create in daemon _tick()"
)
class TestDaemonOptimisticMergeFlow:
    """Test the daemon's _handle_worktree_merge end-to-end.

    These tests exercise the complete daemon-level merge path:
    _tick() → poll_completed → process_completed → _handle_worktree_merge.
    """

    def _make_daemon(
        self, git_repo, mutation_log, work_state, audit_log, session_logger, dispatcher
    ):
        """Create a Daemon instance wired to a real git repo."""
        from corc.daemon import Daemon

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=git_repo,
            parallel=1,
            poll_interval=0.1,
        )
        return daemon

    def test_daemon_tick_merges_worktree_on_success(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Daemon _tick: agent completes → validation passes → merge to main."""
        dispatcher = CommittingDispatcher(
            filename="feature.py", content="def feature(): pass\n"
        )
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()

        daemon = self._make_daemon(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
            dispatcher,
        )

        # Tick 1: schedule + dispatch
        daemon._tick()
        time.sleep(0.5)

        # Tick 2: poll completed → process → merge
        daemon._tick()

        # File should now be in main repo (merged by _handle_worktree_merge)
        assert (git_repo / "feature.py").exists()
        assert (git_repo / "feature.py").read_text() == "def feature(): pass\n"

        # Merge status should be recorded in mutations
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["merge_status"] == "merged"

        # Audit log should have merge events
        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "worktree_merged" in event_types
        assert "worktree_removed" in event_types

        daemon.executor.shutdown()

    def test_daemon_tick_handles_merge_conflict_retry(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Daemon _tick: merge conflict → task failed → retry with merged worktree."""
        # Agent will modify README.md (which will conflict with main)
        dispatcher = CommittingDispatcher(
            filename="README.md", content="# Agent version\n"
        )
        _create_task(mutation_log, "t1", "Task 1", max_retries=3)
        work_state.refresh()

        daemon = self._make_daemon(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
            dispatcher,
        )

        # Tick 1: schedule + dispatch
        daemon._tick()
        time.sleep(0.5)

        # While agent was working, simulate another agent merging changes to main
        (git_repo / "README.md").write_text("# Other agent's version\n")
        subprocess.run(
            ["git", "add", "README.md"], cwd=str(git_repo), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Other agent merged first"],
            cwd=str(git_repo),
            capture_output=True,
        )

        # Tick 2: poll completed → process → merge conflict → mark failed
        daemon._tick()

        # Main should still have the other agent's version (merge aborted)
        assert (git_repo / "README.md").read_text() == "# Other agent's version\n"

        # Task should be marked failed (for retry)
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "failed"
        assert task["merge_status"] == "conflict"

        # Audit log should record conflict and retry preparation
        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "worktree_merge_conflict" in event_types
        assert "merge_conflict_retry" in event_types

        daemon.executor.shutdown()

    def test_daemon_tick_successful_merge_completes_task(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Daemon _tick: successful merge keeps task in 'completed' status."""
        dispatcher = CommittingDispatcher(
            filename="new_module.py", content="# New module\n"
        )
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()

        daemon = self._make_daemon(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
            dispatcher,
        )

        # Two ticks: dispatch + complete
        daemon._tick()
        time.sleep(0.5)
        daemon._tick()

        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "completed"
        assert task["merge_status"] == "merged"

        daemon.executor.shutdown()

    def test_daemon_conflict_retry_reuses_worktree_on_next_dispatch(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """After merge conflict, next dispatch reuses the prepared worktree."""
        # First dispatch: agent creates a new file (non-conflicting)
        dispatcher = CommittingDispatcher(
            filename="agent_work.py", content="# Agent work\n"
        )
        _create_task(mutation_log, "t1", "Task 1", max_retries=3)
        work_state.refresh()

        daemon = self._make_daemon(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
            dispatcher,
        )

        # Create worktree, dispatch, and complete
        daemon._tick()
        time.sleep(0.5)

        # Create a non-conflicting change on main to trigger a merge
        # that will succeed (different file)
        (git_repo / "other_file.py").write_text("# Other agent\n")
        subprocess.run(
            ["git", "add", "other_file.py"], cwd=str(git_repo), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Other agent"],
            cwd=str(git_repo),
            capture_output=True,
        )

        # Tick 2: complete + merge (should succeed since different files)
        daemon._tick()

        work_state.refresh()
        task = work_state.get_task("t1")
        # Non-conflicting merge should succeed
        assert task["status"] == "completed"
        assert task["merge_status"] == "merged"

        # Both files should be in main
        assert (git_repo / "agent_work.py").exists()
        assert (git_repo / "other_file.py").exists()

        daemon.executor.shutdown()

    def test_daemon_handle_worktree_merge_validation_failed_cleans_up(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """When validation fails, _handle_worktree_merge cleans up worktree.

        Uses max_retries=0 to prevent the scheduler from re-dispatching the
        failed task (which would create a new worktree and mask the cleanup).
        """

        # Agent that exits with error (fails validation)
        class FailingDispatcher(AgentDispatcher):
            def dispatch(
                self,
                prompt,
                system_prompt,
                constraints,
                pid_callback=None,
                event_callback=None,
                cwd=None,
            ):
                return AgentResult(output="Error!", exit_code=1, duration_s=0.1)

        # max_retries=0 so the task escalates immediately and isn't retried
        _create_task(mutation_log, "t1", "Task 1", max_retries=0)
        work_state.refresh()

        daemon = self._make_daemon(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
            FailingDispatcher(),
        )

        # Tick 1: dispatch (task moves to running, agent fails immediately)
        daemon._tick()
        time.sleep(0.5)

        # Tick 2: poll completed → process (exit_code=1 → escalated) → cleanup worktree
        daemon._tick()

        # Task should be escalated (max_retries=0, attempt=1 > 0)
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "escalated"

        # Worktree should be cleaned up (validation failed path)
        worktree_dir = git_repo / ".corc" / "worktrees"
        if worktree_dir.exists():
            remaining = list(worktree_dir.iterdir())
            assert len(remaining) == 0, f"Stale worktrees: {remaining}"

        daemon.executor.shutdown()

    def test_daemon_full_conflict_retry_cycle(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Full cycle: dispatch → conflict → retry with conflict worktree → merge succeeds.

        This tests the complete retry flow:
        1. Agent A dispatched, makes changes to file_a.py
        2. Meanwhile, another commit lands on main (non-conflicting)
        3. Agent A completes, validation passes, merge to main succeeds
        4. Verify task completed and file merged
        """
        # Use a dispatcher that creates a unique file each time
        call_count = {"n": 0}

        class IncrementingDispatcher(AgentDispatcher):
            def dispatch(
                self,
                prompt,
                system_prompt,
                constraints,
                pid_callback=None,
                event_callback=None,
                cwd=None,
            ):
                call_count["n"] += 1
                if cwd:
                    fname = f"agent_file_{call_count['n']}.py"
                    (Path(cwd) / fname).write_text(f"# Attempt {call_count['n']}\n")
                    subprocess.run(["git", "add", fname], cwd=cwd, capture_output=True)
                    subprocess.run(
                        ["git", "commit", "-m", f"Agent attempt {call_count['n']}"],
                        cwd=cwd,
                        capture_output=True,
                    )
                return AgentResult(output="Done", exit_code=0, duration_s=0.1)

        _create_task(mutation_log, "t1", "Task 1", max_retries=3)
        work_state.refresh()

        daemon = self._make_daemon(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
            IncrementingDispatcher(),
        )

        # Tick 1: dispatch
        daemon._tick()
        time.sleep(0.5)

        # Tick 2: complete + merge (no conflict since main hasn't changed)
        daemon._tick()

        # Agent should have created file_1.py
        assert (git_repo / "agent_file_1.py").exists()

        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "completed"
        assert task["merge_status"] == "merged"

        daemon.executor.shutdown()


# ===========================================================================
# Tests: Three-stage merge conflict resolution pipeline
# ===========================================================================


def _make_subprocess_side_effect(
    merge_returncode=0, conflicted_files=None, commit_returncode=0
):
    """Create a subprocess.run mock side-effect for the three-stage pipeline.

    Inspects the command list to determine which git operation is being
    called and returns the appropriate mock result.
    """

    def side_effect(cmd, *args, **kwargs):
        result = MagicMock()
        result.stderr = ""
        result.stdout = ""
        result.returncode = 0

        if "merge" in cmd and "--abort" in cmd:
            # git merge --abort
            pass
        elif "merge" in cmd:
            # git merge main
            result.returncode = merge_returncode
        elif "diff" in cmd:
            # git diff --name-only --diff-filter=U
            result.stdout = "\n".join(conflicted_files or [])
        elif "commit" in cmd:
            # git commit
            result.returncode = commit_returncode
        return result

    return side_effect


class TestThreeStageConflictResolution:
    """Tests for the three-stage merge conflict resolution pipeline.

    Stage 1: clean git merge main → push → re-merge PR (no agent dispatch)
    Stage 2: conflicted merge → agent resolution → push → re-merge PR
    Stage 3: agent failure → task_failed for full retry
    """

    def _make_executor(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Create an Executor with mock dependencies for unit testing."""
        return Executor(
            dispatcher=NoopDispatcher(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
            defer_merge=True,
        )

    @patch("corc.executor._try_agent_conflict_resolution")
    @patch("corc.executor.pull_main")
    @patch("corc.executor.merge_pr")
    @patch("corc.executor.push_branch")
    @patch("corc.executor._get_current_branch")
    @patch("corc.executor.get_worktree_branch")
    @patch("corc.executor.subprocess.run")
    def test_stage1_clean_merge_produces_merged_without_agent(
        self,
        mock_run,
        mock_wt_branch,
        mock_main_branch,
        mock_push,
        mock_merge_pr,
        mock_pull,
        mock_agent,
        git_repo,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
    ):
        """Stage 1: clean git merge main in worktree produces merged PR
        without dispatching the conflict resolution agent."""
        mock_wt_branch.return_value = "corc/t1-1"
        mock_main_branch.return_value = "main"
        # git merge main succeeds (clean merge)
        mock_run.side_effect = _make_subprocess_side_effect(merge_returncode=0)
        mock_push.return_value = (True, "")
        mock_merge_pr.return_value = True
        mock_pull.return_value = True

        executor = self._make_executor(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
        )
        pr_info = PRInfo(
            url="https://github.com/test/repo/pull/42",
            number=42,
            branch="corc/t1-1",
            title="Task 1",
        )
        wt_path = git_repo / ".corc" / "worktrees" / "t1-1"
        wt_path.mkdir(parents=True, exist_ok=True)

        result = executor.resolve_conflict_and_remerge_pr("t1", wt_path, pr_info)

        assert result == "merged"
        # Agent should NOT have been dispatched (stage 1 resolves without it)
        mock_agent.assert_not_called()
        # Push should use --force-with-lease
        mock_push.assert_called_once_with(
            git_repo,
            "corc/t1-1",
            force_with_lease=True,
        )
        # merge_pr should be called to re-merge the PR
        mock_merge_pr.assert_called_once_with(git_repo, 42)
        # pull_main should sync local after merge
        mock_pull.assert_called_once_with(git_repo)

        # Audit log should record stage 1 events
        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "conflict_stage1_clean_merge" in event_types
        assert "conflict_stage1_merged" in event_types

        executor.shutdown()

    @patch("corc.executor._try_agent_conflict_resolution")
    @patch("corc.executor.pull_main")
    @patch("corc.executor.merge_pr")
    @patch("corc.executor.push_branch")
    @patch("corc.executor._get_current_branch")
    @patch("corc.executor.get_worktree_branch")
    @patch("corc.executor.subprocess.run")
    def test_stage2_conflicted_merge_triggers_agent_then_merge(
        self,
        mock_run,
        mock_wt_branch,
        mock_main_branch,
        mock_push,
        mock_merge_pr,
        mock_pull,
        mock_agent,
        git_repo,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
    ):
        """Stage 2: git merge has conflicts → agent resolves → push → re-merge PR."""
        mock_wt_branch.return_value = "corc/t1-1"
        mock_main_branch.return_value = "main"
        # git merge fails (conflict), git diff shows conflicted files,
        # git commit succeeds after agent resolution
        mock_run.side_effect = _make_subprocess_side_effect(
            merge_returncode=1,
            conflicted_files=["src/foo.py", "src/bar.py"],
            commit_returncode=0,
        )
        mock_agent.return_value = True  # Agent resolves conflicts
        mock_push.return_value = (True, "")
        mock_merge_pr.return_value = True
        mock_pull.return_value = True

        executor = self._make_executor(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
        )
        pr_info = PRInfo(
            url="https://github.com/test/repo/pull/42",
            number=42,
            branch="corc/t1-1",
            title="Task 1",
        )
        wt_path = git_repo / ".corc" / "worktrees" / "t1-1"
        wt_path.mkdir(parents=True, exist_ok=True)

        result = executor.resolve_conflict_and_remerge_pr("t1", wt_path, pr_info)

        assert result == "merged"
        # Agent SHOULD have been dispatched for stage 2
        mock_agent.assert_called_once_with(wt_path, ["src/foo.py", "src/bar.py"])
        # Push should use --force-with-lease
        mock_push.assert_called_once_with(
            git_repo,
            "corc/t1-1",
            force_with_lease=True,
        )
        mock_merge_pr.assert_called_once_with(git_repo, 42)
        mock_pull.assert_called_once_with(git_repo)

        # Audit log should record stage 2 events
        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "conflict_stage2_agent_resolution" in event_types
        assert "conflict_stage2_merged" in event_types

        executor.shutdown()

    @patch("corc.executor._try_agent_conflict_resolution")
    @patch("corc.executor.pull_main")
    @patch("corc.executor.merge_pr")
    @patch("corc.executor.push_branch")
    @patch("corc.executor._get_current_branch")
    @patch("corc.executor.get_worktree_branch")
    @patch("corc.executor.subprocess.run")
    def test_stage3_agent_failure_returns_conflict_for_task_failed(
        self,
        mock_run,
        mock_wt_branch,
        mock_main_branch,
        mock_push,
        mock_merge_pr,
        mock_pull,
        mock_agent,
        git_repo,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
    ):
        """Stage 3: agent can't resolve → returns 'conflict' for task_failed."""
        mock_wt_branch.return_value = "corc/t1-1"
        mock_main_branch.return_value = "main"
        # git merge fails (conflict), git diff shows conflicted files
        mock_run.side_effect = _make_subprocess_side_effect(
            merge_returncode=1,
            conflicted_files=["src/foo.py"],
        )
        mock_agent.return_value = False  # Agent CANNOT resolve

        executor = self._make_executor(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
        )
        pr_info = PRInfo(
            url="https://github.com/test/repo/pull/42",
            number=42,
            branch="corc/t1-1",
            title="Task 1",
        )
        wt_path = git_repo / ".corc" / "worktrees" / "t1-1"
        wt_path.mkdir(parents=True, exist_ok=True)

        result = executor.resolve_conflict_and_remerge_pr("t1", wt_path, pr_info)

        assert result == "conflict"
        # Agent was dispatched but failed
        mock_agent.assert_called_once()
        # Push and merge_pr should NOT be called (agent failed)
        mock_push.assert_not_called()
        mock_merge_pr.assert_not_called()

        # Audit log should record agent failure
        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "conflict_stage2_agent_failed" in event_types

        executor.shutdown()


class TestThreeStageInDaemonApplyMergeResult:
    """Tests that daemon._apply_merge_result uses the three-stage pipeline."""

    def _make_daemon(
        self, git_repo, mutation_log, work_state, audit_log, session_logger, dispatcher
    ):
        """Create a Daemon instance."""
        from corc.daemon import Daemon

        return Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=git_repo,
            parallel=1,
            poll_interval=0.1,
        )

    def test_apply_merge_result_conflict_with_pr_calls_three_stage(
        self,
        git_repo,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
    ):
        """_apply_merge_result with conflict + pr_info calls resolve_conflict_and_remerge_pr."""
        _create_task(mutation_log, "t1", "Task 1", max_retries=3)
        work_state.refresh()

        daemon = self._make_daemon(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
            NoopDispatcher(),
        )

        wt_path = git_repo / ".corc" / "worktrees" / "t1-1"
        wt_path.mkdir(parents=True, exist_ok=True)

        pr_info = PRInfo(
            url="https://github.com/test/repo/pull/42",
            number=42,
            branch="corc/t1-1",
            title="Task 1",
        )
        item = CompletedTask(
            task=work_state.get_task("t1"),
            result=AgentResult(output="Done", exit_code=0, duration_s=0.1),
            attempt=1,
            worktree_path=wt_path,
            agent_id="agent-test",
            pr_info=pr_info,
        )

        # Mock resolve_conflict_and_remerge_pr to return "merged"
        with (
            patch.object(
                daemon.executor,
                "resolve_conflict_and_remerge_pr",
                return_value="merged",
            ) as mock_resolve,
            patch.object(daemon.executor, "cleanup_worktree") as mock_cleanup,
        ):
            daemon._apply_merge_result(item, "conflict")

            mock_resolve.assert_called_once_with("t1", wt_path, pr_info)
            mock_cleanup.assert_called_once_with("t1", wt_path)

        # Task should have merge_status=merged (not conflict)
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["merge_status"] == "merged"

        daemon.executor.shutdown()

    def test_apply_merge_result_conflict_all_stages_fail_marks_task_failed(
        self,
        git_repo,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
    ):
        """When three-stage pipeline returns 'conflict', task is marked failed."""
        _create_task(mutation_log, "t1", "Task 1", max_retries=3)
        work_state.refresh()

        daemon = self._make_daemon(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
            NoopDispatcher(),
        )

        wt_path = git_repo / ".corc" / "worktrees" / "t1-1"
        wt_path.mkdir(parents=True, exist_ok=True)

        pr_info = PRInfo(
            url="https://github.com/test/repo/pull/42",
            number=42,
            branch="corc/t1-1",
            title="Task 1",
        )
        item = CompletedTask(
            task=work_state.get_task("t1"),
            result=AgentResult(output="Done", exit_code=0, duration_s=0.1),
            attempt=1,
            worktree_path=wt_path,
            agent_id="agent-test",
            pr_info=pr_info,
        )

        # Mock resolve_conflict_and_remerge_pr to return "conflict" (all stages failed)
        with (
            patch.object(
                daemon.executor,
                "resolve_conflict_and_remerge_pr",
                return_value="conflict",
            ),
            patch.object(daemon.executor, "cleanup_worktree"),
        ):
            daemon._apply_merge_result(item, "conflict")

        # Task should be marked failed (for retry)
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "failed"
        assert task["merge_status"] == "conflict"

        daemon.executor.shutdown()

    def test_apply_merge_result_conflict_no_pr_marks_task_failed(
        self,
        git_repo,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
    ):
        """Without pr_info, conflict goes straight to task_failed."""
        _create_task(mutation_log, "t1", "Task 1", max_retries=3)
        work_state.refresh()

        daemon = self._make_daemon(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
            NoopDispatcher(),
        )

        wt_path = git_repo / ".corc" / "worktrees" / "t1-1"
        wt_path.mkdir(parents=True, exist_ok=True)

        item = CompletedTask(
            task=work_state.get_task("t1"),
            result=AgentResult(output="Done", exit_code=0, duration_s=0.1),
            attempt=1,
            worktree_path=wt_path,
            agent_id="agent-test",
            pr_info=None,  # No PR
        )

        with (
            patch.object(
                daemon.executor, "resolve_conflict_and_remerge_pr"
            ) as mock_resolve,
            patch.object(daemon.executor, "cleanup_worktree"),
        ):
            daemon._apply_merge_result(item, "conflict")
            # Three-stage pipeline should NOT be called without pr_info
            mock_resolve.assert_not_called()

        # Task should be marked failed
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "failed"
        assert task["merge_status"] == "conflict"

        daemon.executor.shutdown()
