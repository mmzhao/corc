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
import sys
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
    _INSTALLABLE_FILES,
    _neutralize_installable_files,
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
        },
        reason="Test setup",
    )


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
        self.received_cwds.append(cwd)
        return self.default_result


# ===========================================================================
# Unit tests: worktree module functions
# ===========================================================================


class TestCreateWorktree:
    """Test create_worktree() function."""

    def test_creates_worktree_directory(self, git_repo):
        """Worktree directory is created under .claude/worktrees/."""
        worktree_path, branch_name = create_worktree(git_repo, "task-1", attempt=1)

        assert worktree_path.exists()
        assert worktree_path.is_dir()
        assert str(worktree_path).endswith(".claude/worktrees/task-1-1")

    def test_creates_correct_branch(self, git_repo):
        """Worktree branch is named corc/{task_id}-{attempt}."""
        worktree_path, branch_name = create_worktree(git_repo, "task-1", attempt=1)

        assert branch_name == "corc/task-1-1"

        # Verify branch exists
        result = subprocess.run(
            ["git", "branch", "--list", branch_name],
            capture_output=True,
            text=True,
            cwd=str(git_repo),
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
            capture_output=True,
            text=True,
            cwd=str(git_repo),
        )
        assert branch not in result.stdout

    def test_keeps_branch_when_requested(self, git_repo):
        """Branch is kept when remove_branch=False."""
        wt, branch = create_worktree(git_repo, "task-1", attempt=1)

        remove_worktree(git_repo, wt, remove_branch=False)

        result = subprocess.run(
            ["git", "branch", "--list", branch],
            capture_output=True,
            text=True,
            cwd=str(git_repo),
        )
        assert branch in result.stdout

    def test_nonexistent_worktree_returns_false(self, git_repo):
        """Removing a nonexistent worktree returns False."""
        result = remove_worktree(
            git_repo, git_repo / ".claude" / "worktrees" / "nonexistent"
        )
        assert result is False


class TestMergeWorktree:
    """Test merge_worktree() function."""

    def test_merge_with_changes(self, git_repo):
        """Changes in worktree are merged back to main."""
        wt, branch = create_worktree(git_repo, "task-1", attempt=1)

        # Make changes in the worktree
        (wt / "new_file.py").write_text("# New file from agent\n")
        subprocess.run(["git", "add", "new_file.py"], cwd=str(wt), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Add new file"], cwd=str(wt), capture_output=True
        )

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
        subprocess.run(
            ["git", "add", "README.md"], cwd=str(git_repo), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Main change"],
            cwd=str(git_repo),
            capture_output=True,
        )

        # Make conflicting changes in worktree
        (wt / "README.md").write_text("# Changed in worktree\n")
        subprocess.run(["git", "add", "README.md"], cwd=str(wt), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Worktree change"], cwd=str(wt), capture_output=True
        )

        # Merge should fail (agent can't resolve semantic conflict)
        with patch("corc.worktree._try_agent_conflict_resolution", return_value=False):
            merged = merge_worktree(git_repo, wt)
        assert merged is False

        # Main repo should still be clean
        assert (git_repo / "README.md").read_text() == "# Changed in main\n"


class TestMergeWorktreeDataFilesUntracked:
    """Test that untracked data files survive worktree merges.

    data/ is excluded from git tracking (.gitignore). The daemon writes to
    data/mutations.jsonl in the working directory. Worktree merges must not
    disturb these untracked files.
    """

    def test_untracked_mutations_survive_merge(self, git_repo):
        """Untracked data/mutations.jsonl is not affected by worktree merge."""
        data_dir = git_repo / "data"
        data_dir.mkdir(exist_ok=True)
        (data_dir / "mutations.jsonl").write_text('{"seq":1}\n{"seq":2}\n')

        wt, branch = create_worktree(git_repo, "task-data-1", attempt=1)

        # Agent adds code in worktree
        (wt / "feature.py").write_text("def new_feature(): pass\n")
        subprocess.run(["git", "add", "feature.py"], cwd=str(wt), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Agent work"], cwd=str(wt), capture_output=True
        )

        # Daemon appends to mutations on main (untracked)
        with open(data_dir / "mutations.jsonl", "a") as f:
            f.write('{"seq":3,"event":"daemon_append"}\n')

        merged = merge_worktree(git_repo, wt)
        assert merged is True

        # Code merged
        assert (git_repo / "feature.py").exists()

        # Untracked mutations.jsonl preserved with daemon's append
        content = (data_dir / "mutations.jsonl").read_text()
        assert "daemon_append" in content
        assert content.count("\n") == 3

    def test_code_only_merge(self, git_repo):
        """Code-only changes merge normally with data/ untracked."""
        wt, branch = create_worktree(git_repo, "task-code-only", attempt=1)

        (wt / "new_module.py").write_text("# New module\ndef hello(): return 42\n")
        subprocess.run(
            ["git", "add", "new_module.py"], cwd=str(wt), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add new module"], cwd=str(wt), capture_output=True
        )

        merged = merge_worktree(git_repo, wt)
        assert merged is True
        assert (git_repo / "new_module.py").exists()


class TestAgentConflictResolution:
    """Test agent-driven merge conflict resolution.

    When non-data file conflicts remain after auto-resolving data files,
    merge_worktree() invokes a short claude -p agent to attempt resolution.
    Trivial conflicts (whitespace, formatting) are resolved by the agent;
    genuine semantic conflicts cause the agent to bail out and the merge fails.
    """

    def test_trivial_conflict_resolved_by_agent(self, git_repo):
        """Trivial whitespace/formatting conflict is resolved by the agent."""
        # Create a file and commit it
        (git_repo / "module.py").write_text("def hello():\n    return 'world'\n")
        subprocess.run(
            ["git", "add", "module.py"], cwd=str(git_repo), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add module"],
            cwd=str(git_repo),
            capture_output=True,
        )

        # Create worktree
        wt, branch = create_worktree(git_repo, "task-trivial", attempt=1)

        # Main: formatting change to same line
        (git_repo / "module.py").write_text("def  hello():\n    return 'world'\n")
        subprocess.run(
            ["git", "add", "module.py"], cwd=str(git_repo), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Main formatting"],
            cwd=str(git_repo),
            capture_output=True,
        )

        # Worktree: different formatting change to same line
        (wt / "module.py").write_text("def hello() :\n    return 'world'\n")
        subprocess.run(["git", "add", "module.py"], cwd=str(wt), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Worktree formatting"],
            cwd=str(wt),
            capture_output=True,
        )

        # Mock the agent: simulate resolving the conflict by writing clean content
        # and staging the file (what _try_agent_conflict_resolution does internally)
        def mock_agent_resolve(project_root, conflicted_files):
            for fpath in conflicted_files:
                full_path = Path(project_root) / fpath
                # Write resolved content (conflict markers removed)
                full_path.write_text("def hello():\n    return 'world'\n")
                # Stage the resolved file
                subprocess.run(
                    ["git", "add", "--", fpath],
                    cwd=str(project_root),
                    capture_output=True,
                )
            return True

        with patch(
            "corc.worktree._try_agent_conflict_resolution",
            side_effect=mock_agent_resolve,
        ):
            merged = merge_worktree(git_repo, wt)

        assert merged is True
        content = (git_repo / "module.py").read_text()
        assert "<<<<<<<" not in content
        assert "def hello():" in content

    def test_genuine_semantic_conflict_still_fails(self, git_repo):
        """Genuine semantic conflict: agent bails out, merge fails as before."""
        (git_repo / "module.py").write_text("def hello():\n    return 'world'\n")
        subprocess.run(
            ["git", "add", "module.py"], cwd=str(git_repo), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add module"],
            cwd=str(git_repo),
            capture_output=True,
        )

        wt, branch = create_worktree(git_repo, "task-semantic", attempt=1)

        # Main: completely different implementation
        (git_repo / "module.py").write_text("def goodbye():\n    return 'mars'\n")
        subprocess.run(
            ["git", "add", "module.py"], cwd=str(git_repo), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Rewrite module"],
            cwd=str(git_repo),
            capture_output=True,
        )

        # Worktree: completely different rewrite
        (wt / "module.py").write_text("def greet(name):\n    return f'hi {name}'\n")
        subprocess.run(["git", "add", "module.py"], cwd=str(wt), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Worktree rewrite"],
            cwd=str(wt),
            capture_output=True,
        )

        # Agent fails on genuine semantic conflict (returns False)
        with patch("corc.worktree._try_agent_conflict_resolution", return_value=False):
            merged = merge_worktree(git_repo, wt)

        assert merged is False
        # Main should be clean (merge aborted)
        content = (git_repo / "module.py").read_text()
        assert "goodbye" in content
        assert "<<<<<<<" not in content

    def test_agent_receives_conflicted_file_paths(self, git_repo):
        """The agent is called with the correct list of conflicted file paths."""
        (git_repo / "module.py").write_text("x = 1\n")
        subprocess.run(
            ["git", "add", "module.py"], cwd=str(git_repo), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Add module"],
            cwd=str(git_repo),
            capture_output=True,
        )

        wt, branch = create_worktree(git_repo, "task-paths", attempt=1)

        # Create conflict
        (git_repo / "module.py").write_text("x = 2\n")
        subprocess.run(
            ["git", "add", "module.py"], cwd=str(git_repo), capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "Main change"],
            cwd=str(git_repo),
            capture_output=True,
        )

        (wt / "module.py").write_text("x = 3\n")
        subprocess.run(["git", "add", "module.py"], cwd=str(wt), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Worktree change"],
            cwd=str(wt),
            capture_output=True,
        )

        # Track what the agent was called with
        captured_args = {}

        def mock_capture(project_root, conflicted_files):
            captured_args["project_root"] = project_root
            captured_args["conflicted_files"] = conflicted_files
            return False  # Don't actually resolve

        with patch(
            "corc.worktree._try_agent_conflict_resolution",
            side_effect=mock_capture,
        ):
            merge_worktree(git_repo, wt)

        assert "conflicted_files" in captured_args
        assert "module.py" in captured_args["conflicted_files"]
        assert captured_args["project_root"] == git_repo

    def test_agent_not_invoked_when_no_conflicts(self, git_repo):
        """Agent is NOT invoked when there are no conflicts to resolve."""
        wt, branch = create_worktree(git_repo, "task-clean", attempt=1)

        # Non-conflicting change in worktree
        (wt / "new_file.py").write_text("# new file\n")
        subprocess.run(["git", "add", "new_file.py"], cwd=str(wt), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Add file"],
            cwd=str(wt),
            capture_output=True,
        )

        with patch("corc.worktree._try_agent_conflict_resolution") as mock_agent:
            merged = merge_worktree(git_repo, wt)

        assert merged is True
        mock_agent.assert_not_called()


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

    def test_executor_creates_worktree_before_dispatch(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
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
        assert ".claude/worktrees/" in dispatcher.received_cwd
        executor.shutdown()

    def test_executor_passes_worktree_as_cwd(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
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

    def test_executor_creates_agent_record_with_worktree(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
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
        assert ".claude/worktrees/t1-1" in agents[0]["worktree_path"]
        executor.shutdown()

    def test_executor_cleans_up_worktree_after_completion(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
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

    def test_executor_merges_worktree_changes(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Changes made in worktree are merged back to main after completion."""
        from corc.pr import PRInfo

        class CommittingDispatcher(AgentDispatcher):
            """Dispatcher that creates a file in the cwd (worktree)."""

            def dispatch(
                self,
                prompt,
                system_prompt,
                constraints,
                pid_callback=None,
                event_callback=None,
                cwd=None,
            ):
                if cwd:
                    # Create a file in the worktree
                    new_file = Path(cwd) / "agent_output.py"
                    new_file.write_text("# Agent created this file\n")
                    subprocess.run(
                        ["git", "add", "agent_output.py"], cwd=cwd, capture_output=True
                    )
                    subprocess.run(
                        ["git", "commit", "-m", "Agent work"],
                        cwd=cwd,
                        capture_output=True,
                    )
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

        # Mock PR creation to return a PR (no real remote needed)
        mock_pr = PRInfo(
            url="https://github.com/test/repo/pull/1",
            number=1,
            branch="corc/t1-1",
            title="Test PR",
        )
        with patch.object(executor, "_create_pr_from_worktree", return_value=mock_pr):
            executor.dispatch(task)
            time.sleep(0.5)
            completed = executor.poll_completed()

        assert len(completed) == 1
        # File should now exist in the main repo (merged back)
        assert (git_repo / "agent_output.py").exists()
        executor.shutdown()

    def test_executor_completed_task_has_worktree_path(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
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

    def test_parallel_agents_get_separate_worktrees(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
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

    def test_worktree_created_event(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
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

    def test_worktree_removed_event(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Audit log records worktree_removed event after cleanup."""
        from corc.pr import PRInfo

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

        # Mock PR creation so the executor follows the merge+cleanup path
        mock_pr = PRInfo(
            url="https://github.com/test/repo/pull/1",
            number=1,
            branch="corc/t1-1",
            title="Test PR",
        )
        with patch.object(executor, "_create_pr_from_worktree", return_value=mock_pr):
            executor.dispatch(task)
            time.sleep(0.5)
            executor.poll_completed()

        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "worktree_removed" in event_types
        executor.shutdown()

    def test_worktree_merged_event(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Audit log records worktree_merged event."""
        from corc.pr import PRInfo

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

        # Mock PR creation so the executor follows the merge path
        mock_pr = PRInfo(
            url="https://github.com/test/repo/pull/1",
            number=1,
            branch="corc/t1-1",
            title="Test PR",
        )
        with patch.object(executor, "_create_pr_from_worktree", return_value=mock_pr):
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

    def test_worktree_creation_failure_falls_back(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
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
        with patch(
            "corc.executor.create_worktree", side_effect=WorktreeError("git broken")
        ):
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
        subprocess.run(
            ["git", "commit", "-m", "Add feature"], cwd=str(wt), capture_output=True
        )

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
            capture_output=True,
            text=True,
            cwd=str(git_repo),
        )
        assert branch not in result.stdout.strip()


# ===========================================================================
# Python path isolation tests (fix-worktree-python-path-conflict)
# ===========================================================================


@pytest.fixture
def git_repo_with_pyproject(tmp_path):
    """Create a git repo that has a pyproject.toml and src/ layout.

    Mirrors the real corc project structure so we can test that
    create_worktree() neutralizes the installable files.
    """
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

    # Create pyproject.toml (the file that enables `pip install -e .`)
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "fakepkg"\nversion = "0.1.0"\n'
    )

    # Create a src/ layout
    (repo / "src" / "fakepkg").mkdir(parents=True)
    (repo / "src" / "fakepkg" / "__init__.py").write_text('__version__ = "0.1.0"\n')

    # Create initial commit
    subprocess.run(["git", "add", "."], cwd=str(repo), capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=str(repo),
        capture_output=True,
        check=True,
    )

    # Set up corc directories
    (repo / ".corc").mkdir()

    return repo


class TestWorktreePythonPathIsolation:
    """Verify that worktrees cannot hijack the shared Python environment.

    Root cause: if an agent runs `pip install -e .` inside a worktree,
    the editable install's .pth file gets rewritten to point at the
    worktree's src/ instead of the main project's src/. All subsequent
    `import corc` calls then resolve to the worktree copy.

    Fix: create_worktree() removes pyproject.toml (and setup.py/setup.cfg)
    from the worktree after creation, preventing `pip install -e .`.
    """

    def test_worktree_has_no_pyproject_toml(self, git_repo_with_pyproject):
        """After create_worktree(), pyproject.toml must not exist in the worktree."""
        repo = git_repo_with_pyproject

        # Verify main repo has pyproject.toml
        assert (repo / "pyproject.toml").exists()

        wt, _ = create_worktree(repo, "task-1", attempt=1)

        # Worktree must NOT have pyproject.toml
        assert not (wt / "pyproject.toml").exists(), (
            "pyproject.toml must be removed from worktree to prevent "
            "`pip install -e .` from hijacking Python imports"
        )

    def test_worktree_has_no_setup_py(self, git_repo_with_pyproject):
        """After create_worktree(), setup.py must not exist in the worktree."""
        repo = git_repo_with_pyproject

        # Add a setup.py and commit it
        (repo / "setup.py").write_text("from setuptools import setup; setup()\n")
        subprocess.run(["git", "add", "setup.py"], cwd=str(repo), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Add setup.py"],
            cwd=str(repo),
            capture_output=True,
        )

        wt, _ = create_worktree(repo, "task-2", attempt=1)

        assert not (wt / "setup.py").exists()

    def test_worktree_has_no_setup_cfg(self, git_repo_with_pyproject):
        """After create_worktree(), setup.cfg must not exist in the worktree."""
        repo = git_repo_with_pyproject

        # Add a setup.cfg and commit it
        (repo / "setup.cfg").write_text("[metadata]\nname = fakepkg\n")
        subprocess.run(["git", "add", "setup.cfg"], cwd=str(repo), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Add setup.cfg"],
            cwd=str(repo),
            capture_output=True,
        )

        wt, _ = create_worktree(repo, "task-3", attempt=1)

        assert not (wt / "setup.cfg").exists()

    def test_worktree_source_files_still_present(self, git_repo_with_pyproject):
        """Worktree still has source code files (only install files removed)."""
        repo = git_repo_with_pyproject

        wt, _ = create_worktree(repo, "task-4", attempt=1)

        # Source files should still be there — agents need to edit them
        assert (wt / "src" / "fakepkg" / "__init__.py").exists()

    def test_main_repo_pyproject_unaffected(self, git_repo_with_pyproject):
        """Creating a worktree does not remove pyproject.toml from main repo."""
        repo = git_repo_with_pyproject

        create_worktree(repo, "task-5", attempt=1)

        # Main repo's pyproject.toml must still exist
        assert (repo / "pyproject.toml").exists()
        assert "fakepkg" in (repo / "pyproject.toml").read_text()

    def test_neutralize_installable_files_is_idempotent(self, tmp_path):
        """Calling _neutralize_installable_files on a dir without those files is safe."""
        # Should not raise even if files don't exist
        _neutralize_installable_files(tmp_path)

    def test_main_project_import_unaffected_by_worktree(self, git_repo_with_pyproject):
        """Verify that a worktree's src/ directory is not on sys.path.

        This test ensures that creating a worktree and having it exist on
        disk does not cause Python to find its src/ directory via sys.path.
        """
        repo = git_repo_with_pyproject

        wt, _ = create_worktree(repo, "task-6", attempt=1)
        worktree_src = str(wt / "src")

        # The worktree's src/ should NOT be in sys.path
        assert worktree_src not in sys.path, (
            f"Worktree src/ found in sys.path: {worktree_src}"
        )

    def test_conftest_ensures_main_src_first(self):
        """The project conftest.py puts main src/ at front of sys.path.

        This is a defense-in-depth check: even if a stale .pth file adds
        a worktree's src/, the conftest ensures the main src/ wins.
        """
        project_root = Path(__file__).resolve().parent.parent
        main_src = str(project_root / "src")

        # conftest.py should have been loaded by pytest already
        assert main_src in sys.path, (
            f"Main project src/ not found in sys.path: {main_src}"
        )
        # It should be at position 0 (highest priority)
        idx = sys.path.index(main_src)
        # Check no worktree src/ comes before it
        for i in range(idx):
            assert ".claude/worktrees/" not in sys.path[i], (
                f"Worktree path found before main src/ in sys.path: {sys.path[i]}"
            )


# ===========================================================================
# Human-only merge policy enforcement tests
# ===========================================================================


class TestHumanOnlyMergeBlocking:
    """Test that merge_worktree and git push are blocked for human-only repos.

    These tests verify deterministic enforcement in code (not hooks):
    - merge_worktree raises ProtectedBranchError for human-only repos
    - assert_not_protected raises for human-only repos on protected branches
    - Executor.try_merge_worktree returns "blocked" for human-only repos
    - Executor never calls merge_worktree for human-only repos
    - Failed PR creation for human-only repos marks the task as failed
    """

    def test_assert_not_protected_raises_for_human_only(self, git_repo):
        """assert_not_protected raises ProtectedBranchError for human-only repo on protected branch."""
        from corc.worktree import assert_not_protected, ProtectedBranchError
        from corc.repo_policy import RepoPolicy

        policy = RepoPolicy(name="test-repo", merge_policy="human-only")
        with patch("corc.repo_policy.get_repo_policy", return_value=policy):
            with pytest.raises(ProtectedBranchError, match="human-only"):
                assert_not_protected(git_repo, "main")

    def test_assert_not_protected_allows_auto_merge(self, git_repo):
        """assert_not_protected does NOT raise for auto merge repos."""
        from corc.worktree import assert_not_protected
        from corc.repo_policy import RepoPolicy

        policy = RepoPolicy(name="test-repo", merge_policy="auto")
        with patch("corc.repo_policy.get_repo_policy", return_value=policy):
            # Should not raise
            assert_not_protected(git_repo, "main")

    def test_assert_not_protected_allows_non_protected_branch(self, git_repo):
        """assert_not_protected allows human-only repos on non-protected branches."""
        from corc.worktree import assert_not_protected
        from corc.repo_policy import RepoPolicy

        policy = RepoPolicy(
            name="test-repo",
            merge_policy="human-only",
            protected_branches=["main"],
        )
        with patch("corc.repo_policy.get_repo_policy", return_value=policy):
            # Feature branch should be allowed
            assert_not_protected(git_repo, "corc/task-1-1")

    def test_merge_worktree_blocked_for_human_only(self, git_repo):
        """merge_worktree raises ProtectedBranchError for human-only repos."""
        from corc.worktree import ProtectedBranchError
        from corc.repo_policy import RepoPolicy

        wt, branch = create_worktree(git_repo, "task-block", attempt=1)

        # Make a change so there's something to merge
        (wt / "blocked.py").write_text("# should not merge\n")
        subprocess.run(["git", "add", "blocked.py"], cwd=str(wt), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Blocked change"],
            cwd=str(wt),
            capture_output=True,
        )

        policy = RepoPolicy(name="test-repo", merge_policy="human-only")
        with patch("corc.repo_policy.get_repo_policy", return_value=policy):
            with pytest.raises(ProtectedBranchError):
                merge_worktree(git_repo, wt)

        # Verify the file was NOT merged into main
        assert not (git_repo / "blocked.py").exists()

    def test_executor_try_merge_returns_blocked_for_human_only(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Executor.try_merge_worktree returns 'blocked' for human-only repos."""
        from corc.repo_policy import RepoPolicy

        dispatcher = CwdTrackingDispatcher()
        executor = Executor(
            dispatcher=dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
        )

        wt, branch = create_worktree(git_repo, "task-try-merge", attempt=1)

        policy = RepoPolicy(name="test-repo", merge_policy="human-only")
        with patch("corc.executor.get_repo_policy", return_value=policy):
            status = executor.try_merge_worktree("task-try-merge", wt)

        assert status == "blocked"
        executor.shutdown()

    def test_executor_never_calls_merge_worktree_for_human_only(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Executor.poll_completed never calls merge_worktree for human-only repos."""
        from corc.repo_policy import RepoPolicy

        dispatcher = CwdTrackingDispatcher()
        _create_task(mutation_log, "t-hm", "Human Only Task")
        work_state.refresh()
        task = work_state.get_task("t-hm")

        executor = Executor(
            dispatcher=dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
        )

        policy = RepoPolicy(name="test-repo", merge_policy="human-only")
        with (
            patch("corc.executor.get_repo_policy", return_value=policy),
            patch("corc.executor.merge_worktree") as mock_merge,
            patch("corc.executor.pull_main", return_value=True),
        ):
            executor.dispatch(task)
            time.sleep(0.5)
            completed = executor.poll_completed()

        # merge_worktree should never have been called
        mock_merge.assert_not_called()
        executor.shutdown()

    def test_failed_pr_for_human_only_results_in_task_failure(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """If PR creation fails for a human-only repo, task fails (not completes)."""
        from corc.repo_policy import RepoPolicy

        dispatcher = CwdTrackingDispatcher()
        _create_task(mutation_log, "t-pr-fail", "PR Fail Task")
        work_state.refresh()
        task = work_state.get_task("t-pr-fail")

        executor = Executor(
            dispatcher=dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
        )

        policy = RepoPolicy(name="test-repo", merge_policy="human-only")
        with (
            patch("corc.executor.get_repo_policy", return_value=policy),
            patch("corc.executor.pull_main", return_value=True),
            patch.object(executor, "_create_pr_from_worktree", return_value=None),
        ):
            executor.dispatch(task)
            time.sleep(0.5)
            completed = executor.poll_completed()

        assert len(completed) == 1
        # Task should be marked as failed (exit_code=1)
        assert completed[0].result.exit_code == 1
        assert "PR creation failed" in completed[0].result.output
        assert completed[0].pr_info is None
        executor.shutdown()

    def test_git_push_to_main_blocked_for_human_only(self, git_repo):
        """assert_not_protected blocks push-like operations to protected branches."""
        from corc.worktree import assert_not_protected, ProtectedBranchError
        from corc.repo_policy import RepoPolicy

        policy = RepoPolicy(name="test-repo", merge_policy="human-only")
        with patch("corc.repo_policy.get_repo_policy", return_value=policy):
            # Pushing to main should be blocked
            with pytest.raises(ProtectedBranchError):
                assert_not_protected(git_repo, "main")

            # Pushing to master should also be blocked if it's in protected_branches
            policy2 = RepoPolicy(
                name="test-repo",
                merge_policy="human-only",
                protected_branches=["main", "master"],
            )
        with patch("corc.repo_policy.get_repo_policy", return_value=policy2):
            with pytest.raises(ProtectedBranchError):
                assert_not_protected(git_repo, "master")
