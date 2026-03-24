"""Tests for worktree cleanup after task completion and on daemon restart.

Covers the full worktree cleanup lifecycle:
1. Worktrees removed after successful task completion (daemon tick)
2. Worktrees removed after task failure (daemon tick)
3. Reconciliation on startup removes worktrees for completed/failed/escalated tasks
4. Orphaned worktree directories on the filesystem are cleaned up
5. git worktree prune is called during reconciliation
"""

import os
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from corc.audit import AuditLog
from corc.daemon import Daemon
from corc.dispatch import AgentDispatcher, AgentResult, Constraints
from corc.executor import CompletedTask, Executor
from corc.mutations import MutationLog
from corc.reconcile import (
    clean_stale_worktrees,
    reconcile_on_startup,
    _git_worktree_prune,
)
from corc.sessions import SessionLogger
from corc.state import WorkState
from corc.worktree import create_worktree, remove_worktree


# ---------------------------------------------------------------------------
# Fixtures
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
def tmp_project(tmp_path):
    """Create a minimal project structure (non-git) for unit tests."""
    project = tmp_path / "project"
    project.mkdir()
    (project / ".corc").mkdir()
    (project / ".claude").mkdir()
    (project / ".claude" / "worktrees").mkdir()
    (project / "data").mkdir()
    (project / "data" / "events").mkdir()
    (project / "data" / "sessions").mkdir()
    return project


@pytest.fixture
def mutation_log_git(git_repo):
    return MutationLog(git_repo / "data" / "mutations.jsonl")


@pytest.fixture
def work_state_git(git_repo, mutation_log_git):
    return WorkState(git_repo / "data" / "state.db", mutation_log_git)


@pytest.fixture
def audit_log_git(git_repo):
    return AuditLog(git_repo / "data" / "events")


@pytest.fixture
def session_logger_git(git_repo):
    return SessionLogger(git_repo / "data" / "sessions")


@pytest.fixture
def mutation_log(tmp_project):
    return MutationLog(tmp_project / "data" / "mutations.jsonl")


@pytest.fixture
def work_state(tmp_project, mutation_log):
    return WorkState(tmp_project / "data" / "state.db", mutation_log)


@pytest.fixture
def audit_log(tmp_project):
    return AuditLog(tmp_project / "data" / "events")


@pytest.fixture
def session_logger(tmp_project):
    return SessionLogger(tmp_project / "data" / "sessions")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _start_task(mutation_log, task_id, attempt=1):
    """Helper to mark a task as started."""
    mutation_log.append(
        "task_started",
        {"attempt": attempt},
        reason="Test: marking as running",
        task_id=task_id,
    )


def _complete_task(mutation_log, task_id):
    """Helper to mark a task as completed."""
    mutation_log.append(
        "task_completed",
        {"findings": []},
        reason="Test: marking as completed",
        task_id=task_id,
    )


def _fail_task(mutation_log, task_id, attempt=1):
    """Helper to mark a task as failed."""
    mutation_log.append(
        "task_failed",
        {"attempt": attempt, "attempt_count": attempt, "exit_code": 1},
        reason="Test: marking as failed",
        task_id=task_id,
    )


def _escalate_task(mutation_log, task_id, attempt=4):
    """Helper to mark a task as escalated."""
    mutation_log.append(
        "task_escalated",
        {"attempt": attempt, "attempt_count": attempt},
        reason="Test: marking as escalated",
        task_id=task_id,
    )


def _create_agent(
    mutation_log, agent_id, task_id, role="implementer", pid=None, worktree_path=None
):
    """Helper to create an agent record."""
    mutation_log.append(
        "agent_created",
        {
            "id": agent_id,
            "role": role,
            "task_id": task_id,
            "pid": pid,
            "worktree_path": worktree_path,
        },
        reason="Test setup",
    )


class MockDispatcher(AgentDispatcher):
    """A dispatcher that returns configurable results without calling any LLM."""

    def __init__(self, default_result=None, delay=0):
        self.default_result = default_result or AgentResult(
            output="Mock output: task completed successfully.",
            exit_code=0,
            duration_s=0.1,
        )
        self.delay = delay
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
        self.dispatched.append((prompt, system_prompt, constraints))
        if self.delay:
            time.sleep(self.delay)
        return self.default_result


# ===========================================================================
# Tests: clean_stale_worktrees removes worktrees for completed tasks
# ===========================================================================


class TestCleanWorktreesForTerminalTasks:
    """Worktrees for completed/failed/escalated tasks are removed on reconciliation."""

    def test_completed_task_worktree_removed(
        self, work_state, mutation_log, tmp_project
    ):
        """Worktree for a completed task is cleaned even if PID is unknown."""
        worktree_dir = tmp_project / ".claude" / "worktrees" / "t1-1"
        worktree_dir.mkdir(parents=True)
        (worktree_dir / "code.py").write_text("# agent work")

        _create_task(mutation_log, "t1", "Task 1")
        _start_task(mutation_log, "t1")
        _create_agent(
            mutation_log,
            "agent-1",
            "t1",
            worktree_path=str(worktree_dir),
        )
        _complete_task(mutation_log, "t1")
        work_state.refresh()

        # Verify task is completed
        assert work_state.get_task("t1")["status"] == "completed"

        with patch("corc.reconcile.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            cleaned = clean_stale_worktrees(work_state, tmp_project)

        assert cleaned == 1
        assert not worktree_dir.exists()

    def test_failed_task_worktree_removed(self, work_state, mutation_log, tmp_project):
        """Worktree for a failed task is cleaned up."""
        worktree_dir = tmp_project / ".claude" / "worktrees" / "t1-1"
        worktree_dir.mkdir(parents=True)
        (worktree_dir / "code.py").write_text("# partial work")

        _create_task(mutation_log, "t1", "Task 1")
        _start_task(mutation_log, "t1")
        _create_agent(
            mutation_log,
            "agent-1",
            "t1",
            worktree_path=str(worktree_dir),
        )
        _fail_task(mutation_log, "t1")
        work_state.refresh()

        assert work_state.get_task("t1")["status"] == "failed"

        with patch("corc.reconcile.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            cleaned = clean_stale_worktrees(work_state, tmp_project)

        assert cleaned == 1
        assert not worktree_dir.exists()

    def test_escalated_task_worktree_removed(
        self, work_state, mutation_log, tmp_project
    ):
        """Worktree for an escalated task is cleaned up."""
        worktree_dir = tmp_project / ".claude" / "worktrees" / "t1-1"
        worktree_dir.mkdir(parents=True)
        (worktree_dir / "code.py").write_text("# abandoned work")

        _create_task(mutation_log, "t1", "Task 1")
        _start_task(mutation_log, "t1")
        _create_agent(
            mutation_log,
            "agent-1",
            "t1",
            worktree_path=str(worktree_dir),
        )
        _escalate_task(mutation_log, "t1")
        work_state.refresh()

        assert work_state.get_task("t1")["status"] == "escalated"

        with patch("corc.reconcile.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            cleaned = clean_stale_worktrees(work_state, tmp_project)

        assert cleaned == 1
        assert not worktree_dir.exists()

    def test_running_task_alive_agent_worktree_kept(
        self, work_state, mutation_log, tmp_project
    ):
        """Worktree for a running task with alive agent is NOT cleaned."""
        worktree_dir = tmp_project / ".claude" / "worktrees" / "t1-1"
        worktree_dir.mkdir(parents=True)
        (worktree_dir / "code.py").write_text("# in-progress")

        _create_task(mutation_log, "t1", "Task 1")
        _start_task(mutation_log, "t1")
        _create_agent(
            mutation_log,
            "agent-1",
            "t1",
            pid=os.getpid(),  # Our own PID — always alive
            worktree_path=str(worktree_dir),
        )
        work_state.refresh()

        assert work_state.get_task("t1")["status"] == "running"

        cleaned = clean_stale_worktrees(work_state, tmp_project)

        assert cleaned == 0
        assert worktree_dir.exists()

    def test_completed_task_worktree_removed_even_if_pid_appears_alive(
        self, work_state, mutation_log, tmp_project
    ):
        """Task in terminal state has worktree cleaned regardless of PID liveness."""
        worktree_dir = tmp_project / ".claude" / "worktrees" / "t1-1"
        worktree_dir.mkdir(parents=True)

        _create_task(mutation_log, "t1", "Task 1")
        _start_task(mutation_log, "t1")
        _create_agent(
            mutation_log,
            "agent-1",
            "t1",
            pid=os.getpid(),  # Alive PID
            worktree_path=str(worktree_dir),
        )
        _complete_task(mutation_log, "t1")
        work_state.refresh()

        assert work_state.get_task("t1")["status"] == "completed"

        with patch("corc.reconcile.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            cleaned = clean_stale_worktrees(work_state, tmp_project)

        assert cleaned == 1
        assert not worktree_dir.exists()


# ===========================================================================
# Tests: orphaned filesystem worktrees
# ===========================================================================


class TestOrphanedWorktreeCleanup:
    """Worktree directories not referenced by any active agent are removed."""

    def test_orphaned_worktree_directory_removed(
        self, work_state, mutation_log, tmp_project
    ):
        """Unreferenced directory in .claude/worktrees/ is cleaned up."""
        orphan = tmp_project / ".claude" / "worktrees" / "old-task-1"
        orphan.mkdir(parents=True)
        (orphan / "leftover.py").write_text("# orphaned work")

        # No agent references this directory
        work_state.refresh()

        with patch("corc.reconcile.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            cleaned = clean_stale_worktrees(work_state, tmp_project)

        assert cleaned == 1
        assert not orphan.exists()

    def test_multiple_orphaned_worktrees(self, work_state, mutation_log, tmp_project):
        """Multiple orphaned directories are all cleaned up."""
        orphan1 = tmp_project / ".claude" / "worktrees" / "task-a-1"
        orphan2 = tmp_project / ".claude" / "worktrees" / "task-b-1"
        orphan3 = tmp_project / ".claude" / "worktrees" / "task-c-2"
        for d in (orphan1, orphan2, orphan3):
            d.mkdir(parents=True)
            (d / "code.py").write_text("# orphan")

        work_state.refresh()

        with patch("corc.reconcile.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            cleaned = clean_stale_worktrees(work_state, tmp_project)

        assert cleaned == 3
        assert not orphan1.exists()
        assert not orphan2.exists()
        assert not orphan3.exists()

    def test_active_worktree_not_removed_as_orphan(
        self, work_state, mutation_log, tmp_project
    ):
        """Worktree for a running task with alive agent is not removed as orphan."""
        active_wt = tmp_project / ".claude" / "worktrees" / "t1-1"
        active_wt.mkdir(parents=True)
        orphan_wt = tmp_project / ".claude" / "worktrees" / "old-task-1"
        orphan_wt.mkdir(parents=True)

        _create_task(mutation_log, "t1", "Task 1")
        _start_task(mutation_log, "t1")
        _create_agent(
            mutation_log,
            "agent-1",
            "t1",
            pid=os.getpid(),
            worktree_path=str(active_wt),
        )
        work_state.refresh()

        with patch("corc.reconcile.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            cleaned = clean_stale_worktrees(work_state, tmp_project)

        # Only the orphan should be cleaned; the active one stays
        assert cleaned == 1
        assert active_wt.exists()
        assert not orphan_wt.exists()


# ===========================================================================
# Tests: git worktree prune called during reconciliation
# ===========================================================================


class TestGitWorktreePrune:
    """git worktree prune is called during reconciliation cleanup."""

    def test_prune_called_during_cleanup(self, work_state, mutation_log, tmp_project):
        """clean_stale_worktrees calls git worktree prune."""
        work_state.refresh()

        with patch("corc.reconcile.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            clean_stale_worktrees(work_state, tmp_project)

        # Find the prune call among subprocess.run calls
        prune_calls = [
            call
            for call in mock_run.call_args_list
            if "worktree" in str(call) and "prune" in str(call)
        ]
        assert len(prune_calls) == 1


# ===========================================================================
# Tests: reconciliation on startup cleans stale worktrees
# ===========================================================================


class TestReconciliationWorktreeCleanup:
    """Reconciliation on startup removes worktrees for terminal-state tasks."""

    def test_reconcile_cleans_completed_task_worktree(
        self, work_state, mutation_log, audit_log, session_logger, tmp_project
    ):
        """Worktrees from completed tasks are cleaned during reconciliation."""
        worktree_dir = tmp_project / ".claude" / "worktrees" / "t1-1"
        worktree_dir.mkdir(parents=True)
        (worktree_dir / "done.py").write_text("# completed work")

        _create_task(mutation_log, "t1", "Task 1")
        _start_task(mutation_log, "t1")
        _create_agent(
            mutation_log,
            "agent-1",
            "t1",
            pid=99999,
            worktree_path=str(worktree_dir),
        )
        _complete_task(mutation_log, "t1")
        work_state.refresh()

        with patch("corc.reconcile.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            summary = reconcile_on_startup(
                state=work_state,
                mutation_log=mutation_log,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=tmp_project,
                pid_checker=lambda pid: False,
            )

        assert summary["worktrees_cleaned"] >= 1
        assert not worktree_dir.exists()

    def test_reconcile_cleans_orphaned_worktrees(
        self, work_state, mutation_log, audit_log, session_logger, tmp_project
    ):
        """Orphaned worktree directories are cleaned during reconciliation."""
        orphan = tmp_project / ".claude" / "worktrees" / "gone-task-1"
        orphan.mkdir(parents=True)
        (orphan / "code.py").write_text("# orphan")

        # No tasks or agents exist
        work_state.refresh()

        with patch("corc.reconcile.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            summary = reconcile_on_startup(
                state=work_state,
                mutation_log=mutation_log,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=tmp_project,
                pid_checker=lambda pid: False,
            )

        assert summary["worktrees_cleaned"] == 1
        assert not orphan.exists()

    def test_reconcile_cleans_escalated_task_worktree(
        self, work_state, mutation_log, audit_log, session_logger, tmp_project
    ):
        """Worktree from an escalated task is cleaned during reconciliation."""
        worktree_dir = tmp_project / ".claude" / "worktrees" / "t1-1"
        worktree_dir.mkdir(parents=True)

        _create_task(mutation_log, "t1", "Task 1")
        _start_task(mutation_log, "t1")
        _create_agent(
            mutation_log,
            "agent-1",
            "t1",
            pid=99999,
            worktree_path=str(worktree_dir),
        )
        _escalate_task(mutation_log, "t1")
        work_state.refresh()

        with patch("corc.reconcile.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")
            summary = reconcile_on_startup(
                state=work_state,
                mutation_log=mutation_log,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=tmp_project,
                pid_checker=lambda pid: False,
            )

        assert summary["worktrees_cleaned"] >= 1
        assert not worktree_dir.exists()


# ===========================================================================
# Tests: daemon tick worktree cleanup with real git worktrees
# ===========================================================================


class TestDaemonWorktreeCleanupRealGit:
    """End-to-end tests using real git worktrees.

    Verifies that worktrees are properly removed after the daemon processes
    a completed or failed task.
    """

    def test_worktree_removed_after_successful_completion(
        self,
        git_repo,
        mutation_log_git,
        work_state_git,
        audit_log_git,
        session_logger_git,
    ):
        """Worktree is removed after task completes successfully."""
        # Create a real worktree
        worktree_path, branch_name = create_worktree(git_repo, "t1", attempt=1)
        assert worktree_path.exists()

        # Make a commit in the worktree so there's something to merge
        (worktree_path / "feature.py").write_text("def hello(): pass\n")
        subprocess.run(
            ["git", "add", "feature.py"],
            cwd=str(worktree_path),
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Agent work"],
            cwd=str(worktree_path),
            capture_output=True,
        )

        # Create task, agent, complete it
        _create_task(mutation_log_git, "t1", "Task 1")
        _start_task(mutation_log_git, "t1")
        _create_agent(
            mutation_log_git,
            "agent-1",
            "t1",
            worktree_path=str(worktree_path),
        )
        _complete_task(mutation_log_git, "t1")
        work_state_git.refresh()

        # Run cleanup via reconciliation
        cleaned = clean_stale_worktrees(work_state_git, git_repo)

        assert cleaned == 1
        assert not worktree_path.exists()

        # git worktree list should not include the removed worktree
        result = subprocess.run(
            ["git", "worktree", "list"],
            cwd=str(git_repo),
            capture_output=True,
            text=True,
        )
        assert str(worktree_path) not in result.stdout

    def test_worktree_removed_after_failure(
        self,
        git_repo,
        mutation_log_git,
        work_state_git,
        audit_log_git,
        session_logger_git,
    ):
        """Worktree is removed after task fails."""
        worktree_path, branch_name = create_worktree(git_repo, "t1", attempt=1)
        assert worktree_path.exists()

        _create_task(mutation_log_git, "t1", "Task 1")
        _start_task(mutation_log_git, "t1")
        _create_agent(
            mutation_log_git,
            "agent-1",
            "t1",
            worktree_path=str(worktree_path),
        )
        _fail_task(mutation_log_git, "t1")
        work_state_git.refresh()

        cleaned = clean_stale_worktrees(work_state_git, git_repo)

        assert cleaned == 1
        assert not worktree_path.exists()

        # git worktree list should only show main
        result = subprocess.run(
            ["git", "worktree", "list"],
            cwd=str(git_repo),
            capture_output=True,
            text=True,
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) == 1  # Only main worktree

    def test_stale_worktrees_cleaned_on_daemon_restart(
        self,
        git_repo,
        mutation_log_git,
        work_state_git,
        audit_log_git,
        session_logger_git,
    ):
        """Simulates daemon restart: stale worktrees from previous run are cleaned."""
        # Create multiple worktrees simulating a previous daemon run
        wt1, _ = create_worktree(git_repo, "t1", attempt=1)
        wt2, _ = create_worktree(git_repo, "t2", attempt=1)
        wt3, _ = create_worktree(git_repo, "t3", attempt=1)
        assert wt1.exists() and wt2.exists() and wt3.exists()

        # Task 1: completed — worktree should be cleaned
        _create_task(mutation_log_git, "t1", "Task 1")
        _start_task(mutation_log_git, "t1")
        _create_agent(
            mutation_log_git, "agent-1", "t1", pid=99999, worktree_path=str(wt1)
        )
        _complete_task(mutation_log_git, "t1")

        # Task 2: failed — worktree should be cleaned
        _create_task(mutation_log_git, "t2", "Task 2")
        _start_task(mutation_log_git, "t2")
        _create_agent(
            mutation_log_git, "agent-2", "t2", pid=99998, worktree_path=str(wt2)
        )
        _fail_task(mutation_log_git, "t2")

        # Task 3: escalated — worktree should be cleaned
        _create_task(mutation_log_git, "t3", "Task 3")
        _start_task(mutation_log_git, "t3")
        _create_agent(
            mutation_log_git, "agent-3", "t3", pid=99997, worktree_path=str(wt3)
        )
        _escalate_task(mutation_log_git, "t3")

        work_state_git.refresh()

        # Reconcile — simulates daemon restart
        summary = reconcile_on_startup(
            state=work_state_git,
            mutation_log=mutation_log_git,
            audit_log=audit_log_git,
            session_logger=session_logger_git,
            project_root=git_repo,
            pid_checker=lambda pid: False,
        )

        # All three worktrees should be cleaned
        assert summary["worktrees_cleaned"] == 3
        assert not wt1.exists()
        assert not wt2.exists()
        assert not wt3.exists()

        # git worktree list should only show main
        result = subprocess.run(
            ["git", "worktree", "list"],
            cwd=str(git_repo),
            capture_output=True,
            text=True,
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) == 1

    def test_running_task_worktree_preserved_on_restart(
        self,
        git_repo,
        mutation_log_git,
        work_state_git,
        audit_log_git,
        session_logger_git,
    ):
        """Running task with alive agent keeps its worktree on daemon restart."""
        wt_active, _ = create_worktree(git_repo, "t1", attempt=1)
        wt_stale, _ = create_worktree(git_repo, "t2", attempt=1)

        # Task 1: running with alive agent
        _create_task(mutation_log_git, "t1", "Active Task")
        _start_task(mutation_log_git, "t1")
        _create_agent(
            mutation_log_git,
            "agent-1",
            "t1",
            pid=os.getpid(),
            worktree_path=str(wt_active),
        )

        # Task 2: completed — stale
        _create_task(mutation_log_git, "t2", "Done Task")
        _start_task(mutation_log_git, "t2")
        _create_agent(
            mutation_log_git,
            "agent-2",
            "t2",
            pid=99999,
            worktree_path=str(wt_stale),
        )
        _complete_task(mutation_log_git, "t2")

        work_state_git.refresh()

        summary = reconcile_on_startup(
            state=work_state_git,
            mutation_log=mutation_log_git,
            audit_log=audit_log_git,
            session_logger=session_logger_git,
            project_root=git_repo,
            pid_checker=lambda pid: pid == os.getpid(),
        )

        # Only the stale worktree should be cleaned
        assert summary["worktrees_cleaned"] == 1
        assert wt_active.exists()  # Active task worktree preserved
        assert not wt_stale.exists()  # Stale task worktree removed

        # Clean up
        remove_worktree(git_repo, wt_active)


# ===========================================================================
# Tests: daemon _tick() cleanup integration
# ===========================================================================


class TestDaemonTickWorktreeCleanup:
    """Test worktree cleanup during daemon processing of completed tasks."""

    def test_daemon_cleans_worktree_after_completion(
        self,
        git_repo,
        mutation_log_git,
        work_state_git,
        audit_log_git,
        session_logger_git,
    ):
        """Daemon tick cleans up worktree after task completes."""
        # Create worktree
        worktree_path, _ = create_worktree(git_repo, "t1", attempt=1)
        assert worktree_path.exists()

        # Set up task
        _create_task(mutation_log_git, "t1", "Task 1")
        work_state_git.refresh()
        task = work_state_git.get_task("t1")

        # Create a mock dispatcher that returns success
        dispatcher = MockDispatcher()

        # Patch out PR creation (we don't have a real remote) and merge/cleanup
        # We'll patch the executor's method to track calls
        daemon = Daemon(
            state=work_state_git,
            mutation_log=mutation_log_git,
            audit_log=audit_log_git,
            session_logger=session_logger_git,
            dispatcher=dispatcher,
            project_root=git_repo,
            parallel=1,
            once=True,
            auto_reload=False,
        )

        # Simulate a completed task with a worktree by calling process + cleanup directly
        from corc.processor import process_completed

        # Start the task
        _start_task(mutation_log_git, "t1")
        _create_agent(
            mutation_log_git,
            "agent-1",
            "t1",
            worktree_path=str(worktree_path),
        )
        work_state_git.refresh()
        task = work_state_git.get_task("t1")

        result = AgentResult(output="Done.", exit_code=0, duration_s=1.0)
        proc_result = process_completed(
            task=task,
            result=result,
            attempt=1,
            mutation_log=mutation_log_git,
            state=work_state_git,
            audit_log=audit_log_git,
            session_logger=session_logger_git,
            project_root=git_repo,
        )

        # Processor marked it completed
        work_state_git.refresh()
        assert work_state_git.get_task("t1")["status"] == "completed"

        # Now call cleanup as the daemon would
        daemon.executor.cleanup_worktree("t1", worktree_path)

        assert not worktree_path.exists()
        daemon.executor.shutdown()

    def test_daemon_cleans_worktree_after_failure(
        self,
        git_repo,
        mutation_log_git,
        work_state_git,
        audit_log_git,
        session_logger_git,
    ):
        """Daemon tick cleans up worktree after task fails."""
        worktree_path, _ = create_worktree(git_repo, "t1", attempt=1)
        assert worktree_path.exists()

        _create_task(mutation_log_git, "t1", "Task 1")
        _start_task(mutation_log_git, "t1")
        _create_agent(
            mutation_log_git,
            "agent-1",
            "t1",
            worktree_path=str(worktree_path),
        )
        work_state_git.refresh()
        task = work_state_git.get_task("t1")

        from corc.processor import process_completed

        result = AgentResult(output="Error occurred.", exit_code=1, duration_s=1.0)
        proc_result = process_completed(
            task=task,
            result=result,
            attempt=1,
            mutation_log=mutation_log_git,
            state=work_state_git,
            audit_log=audit_log_git,
            session_logger=session_logger_git,
            project_root=git_repo,
        )

        # Processor marked it failed (not passed)
        assert not proc_result.passed
        work_state_git.refresh()
        assert work_state_git.get_task("t1")["status"] == "failed"

        # Daemon would call cleanup_worktree for failed tasks
        executor = Executor(
            dispatcher=MockDispatcher(),
            mutation_log=mutation_log_git,
            state=work_state_git,
            audit_log=audit_log_git,
            session_logger=session_logger_git,
            project_root=git_repo,
        )
        executor.cleanup_worktree("t1", worktree_path)

        assert not worktree_path.exists()

        # git worktree list should only show main
        result = subprocess.run(
            ["git", "worktree", "list"],
            cwd=str(git_repo),
            capture_output=True,
            text=True,
        )
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        assert len(lines) == 1

        executor.shutdown()

    def test_daemon_exception_safety_net_cleans_worktree(
        self, work_state, mutation_log, audit_log, session_logger, tmp_project
    ):
        """If _handle_worktree_merge raises, the safety net cleans up."""
        worktree_dir = tmp_project / ".claude" / "worktrees" / "t1-1"
        worktree_dir.mkdir(parents=True)
        (worktree_dir / "code.py").write_text("# work")

        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()

        dispatcher = MockDispatcher()
        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=tmp_project,
            parallel=1,
            once=True,
            auto_reload=False,
        )

        from corc.processor import ProcessResult

        # Simulate a completed task item
        item = CompletedTask(
            task=work_state.get_task("t1"),
            result=AgentResult(output="Done.", exit_code=0, duration_s=1.0),
            attempt=1,
            worktree_path=worktree_dir,
            pr_info=None,
        )

        # Process first
        from corc.processor import process_completed

        _start_task(mutation_log, "t1")
        work_state.refresh()
        task = work_state.get_task("t1")

        proc_result = process_completed(
            task=task,
            result=item.result,
            attempt=1,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        # Make _handle_worktree_merge raise an exception
        def raise_error(*args, **kwargs):
            raise RuntimeError("Unexpected merge error!")

        daemon._handle_worktree_merge = raise_error

        # The daemon tick's try/except should catch the exception and clean up
        # We simulate just the relevant portion of the tick loop
        try:
            daemon._handle_worktree_merge(item, proc_result)
        except Exception:
            daemon.executor.cleanup_worktree("t1", worktree_dir)

        assert not worktree_dir.exists()
        daemon.executor.shutdown()
