"""Tests for git worktree isolation in agent dispatch.

Tests the full worktree lifecycle:
1. Worktree created before agent dispatch (git worktree add)
2. Agent runs in the worktree directory (cwd set to worktree path)
3. Worktree changes merged back to main after completion
4. Worktree cleaned up after task completion (git worktree remove)
5. Worktree path tracked in agents table
6. Executor creates and cleans up worktrees
7. CLI dispatch creates and cleans up worktrees
"""

import json
import os
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from corc.audit import AuditLog
from corc.dispatch import AgentDispatcher, AgentResult, Constraints
from corc.executor import CompletedTask, Executor
from corc.mutations import MutationLog
from corc.sessions import SessionLogger
from corc.state import WorkState
from corc.worktree import (
    WorktreeError,
    create_worktree,
    merge_worktree,
    remove_worktree,
    _get_worktree_branch,
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
    subprocess.run(["git", "config", "user.email", "test@test.com"],
                   cwd=str(repo), capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test"],
                   cwd=str(repo), capture_output=True, check=True)

    # Create initial commit so HEAD exists
    (repo / "README.md").write_text("# Test repo\n")
    subprocess.run(["git", "add", "README.md"], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"],
                   cwd=str(repo), capture_output=True, check=True)

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


def _create_task(mutation_log, task_id, name, done_when="do the thing",
                 depends_on=None, role="implementer"):
    """Helper to create a task via mutation log."""
    mutation_log.append("task_created", {
        "id": task_id,
        "name": name,
        "description": f"Test task: {name}",
        "role": role,
        "depends_on": depends_on or [],
        "done_when": done_when,
        "checklist": [],
        "context_bundle": [],
    }, reason="Test setup")


# ---------------------------------------------------------------------------
# Mock dispatcher that tracks cwd
# ---------------------------------------------------------------------------


class CwdTrackingDispatcher(AgentDispatcher):
    """Dispatcher that records the cwd it was called with."""

    def __init__(self, default_result=None):
        self.default_result = default_result or AgentResult(
            output="Mock output: task completed successfully.",
            exit_code=0,
            duration_s=0.1,
        )
        self.dispatched = []
        self.received_cwd = None

    def dispatch(self, prompt, system_prompt, constraints,
                 pid_callback=None, event_callback=None, cwd=None):
        self.dispatched.append((prompt, system_prompt, constraints))
        self.received_cwd = cwd
        return self.default_result


class CwdTrackingDispatcherMulti(AgentDispatcher):
    """Dispatcher that records all cwd values for multiple dispatches."""

    def __init__(self, default_result=None):
        self.default_result = default_result or AgentResult(
            output="Mock output: task completed successfully.",
            exit_code=0,
            duration_s=0.1,
        )
        self.dispatched = []
        self.received_cwds = []

    def dispatch(self, prompt, system_prompt, constraints,
                 pid_callback=None, event_callback=None, cwd=None):
        self.dispatched.append((prompt, system_prompt, constraints))
        self.received_cwds.append(cwd)
        return self.default_result


# ===========================================================================
# Unit tests: worktree module functions
# ===========================================================================


class TestCreateWorktree:
    """Test create_worktree() function."""

    def test_creates_worktree_directory(self, git_repo):
        """Worktree directory is created under .corc/worktrees/."""
        worktree_path, branch_name = create_worktree(git_repo, "task-1", attempt=1)

        assert worktree_path.exists()
        assert worktree_path.is_dir()
        assert str(worktree_path).endswith(".corc/worktrees/task-1-1")

    def test_creates_correct_branch(self, git_repo):
        """Worktree branch is named corc/{task_id}-{attempt}."""
        worktree_path, branch_name = create_worktree(git_repo, "task-1", attempt=1)

        assert branch_name == "corc/task-1-1"

        # Verify branch exists
        result = subprocess.run(
            ["git", "branch", "--list", branch_name],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        assert branch_name in result.stdout

    def test_worktree_has_repo_content(self, git_repo):
        """Worktree contains the same files as the main repo."""
        worktree_path, _ = create_worktree(git_repo, "task-1", attempt=1)

        assert (worktree_path / "README.md").exists()
        assert (worktree_path / "README.md").read_text() == "# Test repo\n"

    def test_multiple_worktrees(self, git_repo):
        """Multiple worktrees can be created for different tasks."""
        wt1, br1 = create_worktree(git_repo, "task-1", attempt=1)
        wt2, br2 = create_worktree(git_repo, "task-2", attempt=1)

        assert wt1.exists()
        assert wt2.exists()
        assert wt1 != wt2
        assert br1 != br2

    def test_different_attempts_different_worktrees(self, git_repo):
        """Different attempt numbers create different worktrees."""
        wt1, br1 = create_worktree(git_repo, "task-1", attempt=1)
        # Remove first to avoid stale worktree conflict
        remove_worktree(git_repo, wt1)

        wt2, br2 = create_worktree(git_repo, "task-1", attempt=2)

        assert wt2.exists()
        assert br2 == "corc/task-1-2"

    def test_replaces_stale_worktree(self, git_repo):
        """If a stale worktree exists from a crash, it's replaced."""
        wt1, _ = create_worktree(git_repo, "task-1", attempt=1)
        assert wt1.exists()

        # Create again — should not error
        wt2, _ = create_worktree(git_repo, "task-1", attempt=1)
        assert wt2.exists()
        assert wt2 == wt1


class TestRemoveWorktree:
    """Test remove_worktree() function."""

    def test_removes_worktree_directory(self, git_repo):
        """Worktree directory is removed after remove_worktree()."""
        wt, _ = create_worktree(git_repo, "task-1", attempt=1)
        assert wt.exists()

        removed = remove_worktree(git_repo, wt)
        assert removed is True
        assert not wt.exists()

    def test_removes_branch(self, git_repo):
        """Branch is removed when remove_branch=True (default)."""
        wt, branch = create_worktree(git_repo, "task-1", attempt=1)

        remove_worktree(git_repo, wt, remove_branch=True)

        result = subprocess.run(
            ["git", "branch", "--list", branch],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        assert branch not in result.stdout

    def test_keeps_branch_when_requested(self, git_repo):
        """Branch is kept when remove_branch=False."""
        wt, branch = create_worktree(git_repo, "task-1", attempt=1)

        remove_worktree(git_repo, wt, remove_branch=False)

        result = subprocess.run(
            ["git", "branch", "--list", branch],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        assert branch in result.stdout

    def test_nonexistent_worktree_returns_false(self, git_repo):
        """Removing a nonexistent worktree returns False."""
        result = remove_worktree(git_repo, git_repo / ".corc" / "worktrees" / "nonexistent")
        assert result is False


class TestMergeWorktree:
    """Test merge_worktree() function."""

    def test_merge_with_changes(self, git_repo):
        """Changes in worktree are merged back to main."""
        wt, branch = create_worktree(git_repo, "task-1", attempt=1)

        # Make changes in the worktree
        (wt / "new_file.py").write_text("# New file from agent\n")
        subprocess.run(["git", "add", "new_file.py"], cwd=str(wt), capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add new file"],
                       cwd=str(wt), capture_output=True)

        # Merge back
        merged = merge_worktree(git_repo, wt)
        assert merged is True

        # Verify file exists in main repo
        assert (git_repo / "new_file.py").exists()

    def test_merge_no_changes(self, git_repo):
        """Merge with no changes returns True (no-op)."""
        wt, _ = create_worktree(git_repo, "task-1", attempt=1)

        merged = merge_worktree(git_repo, wt)
        assert merged is True

    def test_merge_conflict_returns_false(self, git_repo):
        """Merge conflict returns False and main is not corrupted."""
        wt, _ = create_worktree(git_repo, "task-1", attempt=1)

        # Make conflicting changes in main repo
        (git_repo / "README.md").write_text("# Changed in main\n")
        subprocess.run(["git", "add", "README.md"], cwd=str(git_repo), capture_output=True)
        subprocess.run(["git", "commit", "-m", "Main change"],
                       cwd=str(git_repo), capture_output=True)

        # Make conflicting changes in worktree
        (wt / "README.md").write_text("# Changed in worktree\n")
        subprocess.run(["git", "add", "README.md"], cwd=str(wt), capture_output=True)
        subprocess.run(["git", "commit", "-m", "Worktree change"],
                       cwd=str(wt), capture_output=True)

        # Merge should fail
        merged = merge_worktree(git_repo, wt)
        assert merged is False

        # Main repo should still be clean
        assert (git_repo / "README.md").read_text() == "# Changed in main\n"


class TestGetWorktreeBranch:
    """Test _get_worktree_branch() helper."""

    def test_returns_branch_name(self, git_repo):
        """Returns the branch name for a worktree."""
        wt, expected_branch = create_worktree(git_repo, "task-1", attempt=1)
        branch = _get_worktree_branch(git_repo, wt)
        assert branch == expected_branch

    def test_returns_none_for_nonexistent(self, git_repo):
        """Returns None for a path that's not a worktree."""
        branch = _get_worktree_branch(git_repo, git_repo / "nonexistent")
        assert branch is None


# ===========================================================================
# Integration tests: Executor with worktrees
# ===========================================================================


class TestExecutorWorktreeLifecycle:
    """Test that the executor creates worktrees, tracks them, and cleans up."""

    def test_executor_creates_worktree_before_dispatch(self, git_repo, mutation_log,
                                                        work_state, audit_log,
                                                        session_logger):
        """Executor creates a git worktree before dispatching the agent."""
        dispatcher = CwdTrackingDispatcher()
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
        )

        executor.dispatch(task)
        # Wait for dispatch to complete
        time.sleep(0.5)
        completed = executor.poll_completed()

        # Agent should have received a worktree cwd
        assert dispatcher.received_cwd is not None
        assert ".corc/worktrees/" in dispatcher.received_cwd
        executor.shutdown()

    def test_executor_passes_worktree_as_cwd(self, git_repo, mutation_log,
                                              work_state, audit_log,
                                              session_logger):
        """The dispatcher is called with cwd set to the worktree path."""
        dispatcher = CwdTrackingDispatcher()
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
        )

        executor.dispatch(task)
        time.sleep(0.5)
        completed = executor.poll_completed()

        # cwd should be the worktree path
        assert dispatcher.received_cwd is not None
        wt_path = Path(dispatcher.received_cwd)
        assert "t1-1" in wt_path.name
        executor.shutdown()

    def test_executor_creates_agent_record_with_worktree(self, git_repo, mutation_log,
                                                          work_state, audit_log,
                                                          session_logger):
        """Agent record in state includes worktree_path."""
        dispatcher = CwdTrackingDispatcher()
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
        )

        executor.dispatch(task)
        time.sleep(0.5)
        executor.poll_completed()

        # Agent record should have worktree path
        work_state.refresh()
        agents = work_state.get_agents_for_task("t1")
        assert len(agents) == 1
        assert agents[0]["worktree_path"] is not None
        assert ".corc/worktrees/t1-1" in agents[0]["worktree_path"]
        executor.shutdown()

    def test_executor_cleans_up_worktree_after_completion(self, git_repo, mutation_log,
                                                           work_state, audit_log,
                                                           session_logger):
        """Worktree directory is removed after task completes."""
        dispatcher = CwdTrackingDispatcher()
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
        )

        executor.dispatch(task)
        time.sleep(0.5)
        completed = executor.poll_completed()

        assert len(completed) == 1
        # Worktree should have been cleaned up
        wt_path = completed[0].worktree_path
        assert wt_path is not None
        assert not wt_path.exists()
        executor.shutdown()

    def test_executor_merges_worktree_changes(self, git_repo, mutation_log,
                                               work_state, audit_log,
                                               session_logger):
        """Changes made in worktree are merged back to main after completion."""
        class CommittingDispatcher(AgentDispatcher):
            """Dispatcher that creates a file in the cwd (worktree)."""
            def dispatch(self, prompt, system_prompt, constraints,
                         pid_callback=None, event_callback=None, cwd=None):
                if cwd:
                    # Create a file in the worktree
                    new_file = Path(cwd) / "agent_output.py"
                    new_file.write_text("# Agent created this file\n")
                    subprocess.run(["git", "add", "agent_output.py"],
                                   cwd=cwd, capture_output=True)
                    subprocess.run(["git", "commit", "-m", "Agent work"],
                                   cwd=cwd, capture_output=True)
                return AgentResult(output="Done", exit_code=0, duration_s=0.1)

        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = Executor(
            dispatcher=CommittingDispatcher(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
        )

        executor.dispatch(task)
        time.sleep(0.5)
        completed = executor.poll_completed()

        assert len(completed) == 1
        # File should now exist in the main repo
        assert (git_repo / "agent_output.py").exists()
        executor.shutdown()

    def test_executor_completed_task_has_worktree_path(self, git_repo, mutation_log,
                                                        work_state, audit_log,
                                                        session_logger):
        """CompletedTask includes worktree_path and agent_id."""
        dispatcher = CwdTrackingDispatcher()
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
        )

        executor.dispatch(task)
        time.sleep(0.5)
        completed = executor.poll_completed()

        assert len(completed) == 1
        assert completed[0].worktree_path is not None
        assert completed[0].agent_id is not None
        assert completed[0].agent_id.startswith("agent-")
        executor.shutdown()

    def test_parallel_agents_get_separate_worktrees(self, git_repo, mutation_log,
                                                     work_state, audit_log,
                                                     session_logger):
        """Two agents dispatched in parallel get different worktrees."""
        dispatcher = CwdTrackingDispatcherMulti()
        _create_task(mutation_log, "t1", "Task 1")
        _create_task(mutation_log, "t2", "Task 2")
        work_state.refresh()
        task1 = work_state.get_task("t1")
        task2 = work_state.get_task("t2")

        executor = Executor(
            dispatcher=dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
            max_workers=2,
        )

        executor.dispatch(task1)
        executor.dispatch(task2)
        time.sleep(1.0)
        completed = executor.poll_completed()

        assert len(completed) == 2
        # Both should have different worktree paths
        assert len(dispatcher.received_cwds) == 2
        assert dispatcher.received_cwds[0] != dispatcher.received_cwds[1]
        assert all(cwd is not None for cwd in dispatcher.received_cwds)
        executor.shutdown()


# ===========================================================================
# Integration tests: worktree audit logging
# ===========================================================================


class TestWorktreeAuditEvents:
    """Test that worktree lifecycle events are logged to the audit log."""

    def test_worktree_created_event(self, git_repo, mutation_log, work_state,
                                     audit_log, session_logger):
        """Audit log records worktree_created event."""
        dispatcher = CwdTrackingDispatcher()
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
        )

        executor.dispatch(task)
        time.sleep(0.5)
        executor.poll_completed()

        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "worktree_created" in event_types
        executor.shutdown()

    def test_worktree_removed_event(self, git_repo, mutation_log, work_state,
                                     audit_log, session_logger):
        """Audit log records worktree_removed event after cleanup."""
        dispatcher = CwdTrackingDispatcher()
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
        )

        executor.dispatch(task)
        time.sleep(0.5)
        executor.poll_completed()

        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "worktree_removed" in event_types
        executor.shutdown()

    def test_worktree_merged_event(self, git_repo, mutation_log, work_state,
                                    audit_log, session_logger):
        """Audit log records worktree_merged event."""
        dispatcher = CwdTrackingDispatcher()
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
        )

        executor.dispatch(task)
        time.sleep(0.5)
        executor.poll_completed()

        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "worktree_merged" in event_types
        executor.shutdown()


# ===========================================================================
# Edge case tests
# ===========================================================================


class TestWorktreeEdgeCases:
    """Test error handling and edge cases."""

    def test_worktree_creation_failure_falls_back(self, git_repo, mutation_log,
                                                   work_state, audit_log,
                                                   session_logger):
        """If worktree creation fails, agent runs in project root (cwd=None)."""
        dispatcher = CwdTrackingDispatcher()
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
        )

        # Mock create_worktree to fail
        with patch("corc.executor.create_worktree", side_effect=WorktreeError("git broken")):
            executor.dispatch(task)
            time.sleep(0.5)
            completed = executor.poll_completed()

        assert len(completed) == 1
        assert dispatcher.received_cwd is None  # Fell back to no cwd
        executor.shutdown()

    def test_full_lifecycle_create_run_merge_cleanup(self, git_repo):
        """Full worktree lifecycle: create → edit → commit → merge → remove."""
        # 1. Create
        wt, branch = create_worktree(git_repo, "task-1", attempt=1)
        assert wt.exists()

        # 2. Edit in worktree
        (wt / "feature.py").write_text("def hello(): return 'world'\n")
        subprocess.run(["git", "add", "feature.py"], cwd=str(wt), capture_output=True)
        subprocess.run(["git", "commit", "-m", "Add feature"],
                       cwd=str(wt), capture_output=True)

        # 3. Merge
        merged = merge_worktree(git_repo, wt)
        assert merged is True
        assert (git_repo / "feature.py").exists()

        # 4. Remove
        removed = remove_worktree(git_repo, wt)
        assert removed is True
        assert not wt.exists()

        # Branch should be cleaned up
        result = subprocess.run(
            ["git", "branch", "--list", branch],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        assert branch not in result.stdout.strip()
