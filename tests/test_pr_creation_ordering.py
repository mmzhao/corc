"""Tests for PR creation ordering fix.

Verifies:
1. PRs are created BEFORE worktree merge (not after)
2. Auto-merge repos: merge via gh pr merge, NOT direct git merge
3. Human-only repos: PR created, worktree NOT merged, task stays pending_merge
4. Branch is pushed before PR creation
5. PR has commits ahead of main when created
6. try_merge_worktree uses gh pr merge when pr_info is provided
7. Daemon _handle_worktree_merge respects PR workflow
"""

import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from corc.audit import AuditLog
from corc.dispatch import AgentDispatcher, AgentResult, Constraints
from corc.executor import CompletedTask, Executor
from corc.mutations import MutationLog
from corc.pr import PRInfo, merge_pr, pull_main
from corc.processor import ProcessResult, process_completed
from corc.repo_policy import RepoPolicy
from corc.sessions import SessionLogger
from corc.state import WorkState
from corc.worktree import create_worktree, remove_worktree


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path):
    """Create a real git repository for PR ordering tests."""
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
        ["git", "add", "README.md"],
        cwd=str(repo),
        capture_output=True,
        check=True,
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


def _write_repos_yaml(project_root, content):
    """Write a repos.yaml config file."""
    repos_yaml = project_root / ".corc" / "repos.yaml"
    repos_yaml.write_text(content)


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


class MockDispatcher(AgentDispatcher):
    """Dispatcher that returns a canned result."""

    def __init__(self, result=None):
        self.result = result or AgentResult(
            output="Mock output: task completed.",
            exit_code=0,
            duration_s=0.1,
        )
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
        return self.result


# ===========================================================================
# Test: try_merge_worktree uses gh pr merge when pr_info is provided
# ===========================================================================


class TestTryMergeWorktreeWithPR:
    """try_merge_worktree should use gh pr merge when pr_info is provided."""

    def test_pr_merge_used_when_pr_info_provided(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """try_merge_worktree calls merge_pr instead of merge_worktree when pr_info given."""
        wt, branch = create_worktree(git_repo, "t1", attempt=1)

        executor = Executor(
            dispatcher=MockDispatcher(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
            defer_merge=True,
        )

        pr_info = PRInfo(
            url="https://github.com/org/repo/pull/42",
            number=42,
            branch=branch,
            title="[corc] Test (t1)",
        )

        with (
            patch("corc.executor.merge_pr", return_value=True) as mock_merge_pr,
            patch("corc.executor.pull_main", return_value=True) as mock_pull,
            patch("corc.executor.merge_worktree") as mock_direct_merge,
        ):
            status = executor.try_merge_worktree("t1", wt, pr_info=pr_info)

        assert status == "merged"
        # gh pr merge should be called, NOT direct git merge
        mock_merge_pr.assert_called_once_with(git_repo, 42)
        mock_pull.assert_called_once_with(git_repo)
        mock_direct_merge.assert_not_called()

        # Audit log should record PR-based merge
        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "worktree_merged_via_pr" in event_types

        executor.shutdown()

    def test_direct_merge_when_no_pr_info(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """try_merge_worktree falls back to direct merge when no pr_info."""
        wt, branch = create_worktree(git_repo, "t1", attempt=1)
        # Make a commit so there's something to merge
        (wt / "file.py").write_text("# work\n")
        subprocess.run(["git", "add", "file.py"], cwd=str(wt), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Work"], cwd=str(wt), capture_output=True
        )

        executor = Executor(
            dispatcher=MockDispatcher(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
            defer_merge=True,
        )

        # No pr_info → should use direct merge
        status = executor.try_merge_worktree("t1", wt)

        assert status == "merged"
        # File should be merged into main
        assert (git_repo / "file.py").exists()

        executor.shutdown()

    def test_pr_merge_failure_returns_conflict(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """try_merge_worktree returns 'conflict' when gh pr merge fails."""
        wt, branch = create_worktree(git_repo, "t1", attempt=1)

        executor = Executor(
            dispatcher=MockDispatcher(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
            defer_merge=True,
        )

        pr_info = PRInfo(
            url="https://github.com/org/repo/pull/42",
            number=42,
            branch=branch,
            title="[corc] Test (t1)",
        )

        with (
            patch("corc.executor.merge_pr", return_value=False) as mock_merge_pr,
            patch("corc.executor.pull_main") as mock_pull,
        ):
            status = executor.try_merge_worktree("t1", wt, pr_info=pr_info)

        assert status == "conflict"
        mock_merge_pr.assert_called_once()
        # pull_main should NOT be called on failure
        mock_pull.assert_not_called()

        # Audit log should record failure
        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "pr_merge_failed" in event_types

        executor.shutdown()

    def test_pr_merge_exception_returns_error(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """try_merge_worktree returns 'error' on exception during PR merge."""
        wt, branch = create_worktree(git_repo, "t1", attempt=1)

        executor = Executor(
            dispatcher=MockDispatcher(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
            defer_merge=True,
        )

        pr_info = PRInfo(
            url="https://github.com/org/repo/pull/42",
            number=42,
            branch=branch,
            title="[corc] Test (t1)",
        )

        with patch("corc.executor.merge_pr", side_effect=RuntimeError("network error")):
            status = executor.try_merge_worktree("t1", wt, pr_info=pr_info)

        assert status == "error"

        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "pr_merge_error" in event_types

        executor.shutdown()


# ===========================================================================
# Test: Daemon _handle_worktree_merge with PR workflow
# ===========================================================================


class TestDaemonPRBasedMerge:
    """Test daemon's _handle_worktree_merge with PR-based workflow."""

    def _make_daemon(
        self,
        git_repo,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        dispatcher,
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

    def test_auto_merge_pr_already_merged_by_processor(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Auto-merge: when processor merged the PR, daemon pulls main and cleans up."""
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

        # pull_main should be called to sync local repo
        mock_pull.assert_called_once_with(git_repo)

        # Merge status should be recorded
        entries = mutation_log.read_all()
        updated = [e for e in entries if e.get("type") == "task_updated"]
        assert any(e["data"].get("merge_status") == "merged" for e in updated)

        # Worktree should be cleaned up
        assert not wt.exists()

        # Audit log should show PR sync event
        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "pr_merge_synced" in event_types

        daemon.executor.shutdown()

    def test_auto_merge_pr_not_merged_retries_via_executor(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Auto-merge: when processor didn't merge PR, daemon retries via try_merge_worktree."""
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

        pr_info = PRInfo(
            url="https://github.com/org/repo/pull/10",
            number=10,
            branch=branch,
            title="[corc] Task 1 (t1)",
        )

        item = CompletedTask(
            task=work_state.get_task("t1"),
            result=AgentResult(output="Done", exit_code=0, duration_s=0.1),
            attempt=1,
            worktree_path=wt,
            agent_id="agent-test",
            pr_info=pr_info,
        )

        proc_result = ProcessResult(
            task_id="t1",
            passed=True,
            details=[(True, "All checks pass")],
            pr_merged=False,  # Processor failed to merge
            pr_commented=True,
        )

        with (
            patch("corc.executor.merge_pr", return_value=True) as mock_merge_pr,
            patch("corc.executor.pull_main", return_value=True),
        ):
            daemon._handle_worktree_merge(item, proc_result)

        # Should retry via gh pr merge (through try_merge_worktree)
        mock_merge_pr.assert_called_once_with(git_repo, 10)

        # Merge status should be recorded
        entries = mutation_log.read_all()
        updated = [e for e in entries if e.get("type") == "task_updated"]
        assert any(e["data"].get("merge_status") == "merged" for e in updated)

        daemon.executor.shutdown()

    def test_human_only_does_not_merge_cleans_up_keeps_branch(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Human-only: daemon does NOT merge, cleans up worktree, keeps branch."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()

        wt, branch = create_worktree(git_repo, "t1", attempt=1)
        # Create a commit in the worktree
        (wt / "feature.py").write_text("# Feature\n")
        subprocess.run(["git", "add", "feature.py"], cwd=str(wt), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Add feature"],
            cwd=str(wt),
            capture_output=True,
        )

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
                url="https://github.com/org/repo/pull/20",
                number=20,
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

        with patch(
            "corc.daemon.get_repo_policy",
            return_value=RepoPolicy(name="repo", merge_policy="human-only"),
        ):
            daemon._handle_worktree_merge(item, proc_result)

        # Worktree should be cleaned up
        assert not wt.exists()

        # But branch should still exist (for the PR)
        result = subprocess.run(
            ["git", "branch", "--list", branch],
            capture_output=True,
            text=True,
            cwd=str(git_repo),
        )
        assert branch in result.stdout

        # Feature should NOT be in main (not merged)
        assert not (git_repo / "feature.py").exists()

        # Audit log should record human-only skip
        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "worktree_merge_skipped_human_only" in event_types

        daemon.executor.shutdown()

    def test_validation_failed_cleans_up_worktree(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Validation failure: clean up worktree regardless of PR info."""
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
            result=AgentResult(output="Failed", exit_code=0, duration_s=0.1),
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
            passed=False,
            details=[(False, "Tests failed")],
        )

        daemon._handle_worktree_merge(item, proc_result)

        # Worktree should be cleaned up even though PR exists
        assert not wt.exists()

        daemon.executor.shutdown()

    def test_no_pr_falls_back_to_direct_merge(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """When no PR exists, fall back to direct git merge."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()

        wt, branch = create_worktree(git_repo, "t1", attempt=1)
        # Create a commit in the worktree
        (wt / "feature.py").write_text("# Feature\n")
        subprocess.run(["git", "add", "feature.py"], cwd=str(wt), capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Add feature"],
            cwd=str(wt),
            capture_output=True,
        )

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
            pr_info=None,  # No PR
        )

        proc_result = ProcessResult(
            task_id="t1",
            passed=True,
            details=[(True, "All checks pass")],
        )

        daemon._handle_worktree_merge(item, proc_result)

        # Feature should be merged into main (direct git merge)
        assert (git_repo / "feature.py").exists()

        # Merge status should be recorded
        entries = mutation_log.read_all()
        updated = [e for e in entries if e.get("type") == "task_updated"]
        assert any(e["data"].get("merge_status") == "merged" for e in updated)

        daemon.executor.shutdown()


# ===========================================================================
# Test: PR is created BEFORE merge attempt
# ===========================================================================


class TestPRCreatedBeforeMerge:
    """Verify the PR is created before any merge attempt in the daemon flow."""

    def test_pr_creation_precedes_merge_in_poll_completed(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """In executor.poll_completed(), PR is created before any merge happens."""
        dispatcher = MockDispatcher()
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
            defer_merge=True,  # Daemon mode
        )

        mock_pr_info = PRInfo(
            url="https://github.com/org/repo/pull/1",
            number=1,
            branch="corc/t1-1",
            title="[corc] Task 1 (t1)",
        )

        call_order = []

        def track_push(*args, **kwargs):
            call_order.append("push_branch")
            return (True, "")

        def track_create_pr(*args, **kwargs):
            call_order.append("create_pr")
            return (mock_pr_info, "")

        with (
            patch("corc.executor.pull_main", return_value=False),
            patch("corc.executor.push_branch", side_effect=track_push),
            patch("corc.executor.create_pr", side_effect=track_create_pr),
            patch("corc.executor.get_worktree_branch", return_value="corc/t1-1"),
        ):
            executor.dispatch(task)
            time.sleep(0.5)
            completed = executor.poll_completed()

        # PR should be created in poll_completed
        assert len(completed) == 1
        assert completed[0].pr_info is not None
        assert completed[0].pr_info.number == 1

        # push_branch should be called before create_pr
        assert call_order == ["push_branch", "create_pr"]

        # In defer_merge mode, no merge happens in poll_completed
        # The daemon's _handle_worktree_merge runs AFTER this
        assert completed[0].worktree_path is not None  # Worktree still alive

        executor.shutdown()

    def test_pr_exists_with_commits_ahead_when_created(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """PR branch has commits ahead of main when PR is created."""
        dispatcher = CommittingDispatcher(
            filename="feature.py", content="def feature(): pass\n"
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

        branch_name_at_pr_creation = None

        def capture_create_pr(project_root, branch_name, task_dict, **kwargs):
            nonlocal branch_name_at_pr_creation
            branch_name_at_pr_creation = branch_name

            # Verify the branch has commits ahead of main
            result = subprocess.run(
                ["git", "log", f"HEAD..{branch_name}", "--oneline"],
                capture_output=True,
                text=True,
                cwd=str(project_root),
            )
            # Branch should have at least one commit ahead of main
            assert result.stdout.strip(), (
                f"Branch {branch_name} has NO commits ahead of main at PR creation time!"
            )

            return (
                PRInfo(
                    url="https://github.com/org/repo/pull/1",
                    number=1,
                    branch=branch_name,
                    title=f"[corc] Task 1 (t1)",
                ),
                "",
            )

        with (
            patch("corc.executor.pull_main", return_value=False),
            patch("corc.executor.push_branch", return_value=(True, "")),
            patch("corc.executor.create_pr", side_effect=capture_create_pr),
        ):
            executor.dispatch(task)
            time.sleep(0.5)
            completed = executor.poll_completed()

        assert len(completed) == 1
        assert completed[0].pr_info is not None
        assert branch_name_at_pr_creation is not None

        executor.shutdown()


# ===========================================================================
# Test: Full daemon tick with PR workflow
# ===========================================================================


class TestDaemonTickPRWorkflow:
    """End-to-end daemon tick tests with PR workflow."""

    def _make_daemon(
        self,
        git_repo,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        dispatcher,
    ):
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

    def test_auto_merge_full_tick_uses_pr_merge_not_direct(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Full daemon tick: auto-merge uses gh pr merge, not direct git merge."""
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

        mock_pr_info = PRInfo(
            url="https://github.com/org/repo/pull/10",
            number=10,
            branch="corc/t1-1",
            title="[corc] Task 1 (t1)",
        )

        with (
            patch("corc.executor.pull_main", return_value=False),
            patch("corc.executor.push_branch", return_value=(True, "")),
            patch("corc.executor.create_pr", return_value=(mock_pr_info, "")),
            patch("corc.executor.get_worktree_branch", return_value="corc/t1-1"),
            patch("corc.processor.post_review_comment", return_value=True),
            patch("corc.processor.merge_pr", return_value=True) as mock_proc_merge,
            patch("corc.daemon.pull_main", return_value=True) as mock_daemon_pull,
        ):
            # Tick 1: dispatch
            daemon._tick()
            time.sleep(0.5)
            # Tick 2: poll → process → merge
            daemon._tick()

        # Processor should merge PR via gh pr merge
        mock_proc_merge.assert_called_once_with(git_repo, 10)

        # Daemon should pull main to sync local after processor merges
        mock_daemon_pull.assert_called()

        # Task should be completed
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "completed"

        # Merge status should be "merged"
        assert task.get("merge_status") == "merged"

        daemon.executor.shutdown()

    def test_human_only_full_tick_no_merge_pending(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Full daemon tick: human-only repo creates PR but does not merge."""
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

        mock_pr_info = PRInfo(
            url="https://github.com/org/repo/pull/20",
            number=20,
            branch="corc/t1-1",
            title="[corc] Task 1 (t1)",
        )

        with (
            patch("corc.executor.pull_main", return_value=False),
            patch("corc.executor.push_branch", return_value=(True, "")),
            patch("corc.executor.create_pr", return_value=(mock_pr_info, "")),
            patch("corc.executor.get_worktree_branch", return_value="corc/t1-1"),
            patch("corc.processor.post_review_comment", return_value=True),
            patch("corc.processor.merge_pr") as mock_proc_merge,
            patch("corc.processor.get_repo_policy") as mock_proc_policy,
            patch("corc.daemon.get_repo_policy") as mock_daemon_policy,
        ):
            human_policy = RepoPolicy(name="repo", merge_policy="human-only")
            mock_proc_policy.return_value = human_policy
            mock_daemon_policy.return_value = human_policy

            # Tick 1: dispatch
            daemon._tick()
            time.sleep(0.5)
            # Tick 2: poll → process → handle merge
            daemon._tick()

        # Processor should NOT merge
        mock_proc_merge.assert_not_called()

        # Task should be pending_merge (not completed)
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "pending_merge"

        # Feature should NOT be in main
        assert not (git_repo / "feature.py").exists()

        # Audit log should show human-only skip
        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "worktree_merge_skipped_human_only" in event_types

        daemon.executor.shutdown()

    def test_no_pr_fallback_to_direct_merge(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Daemon tick without remote: falls back to direct git merge."""
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

        # No PR mocks → PR creation will fail (no remote), so item.pr_info = None
        # Tick 1: dispatch
        daemon._tick()
        time.sleep(0.5)
        # Tick 2: poll → process → direct merge (fallback)
        daemon._tick()

        # File should be in main (merged directly)
        assert (git_repo / "feature.py").exists()

        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "completed"
        assert task.get("merge_status") == "merged"

        daemon.executor.shutdown()


# ===========================================================================
# Test: cleanup_worktree with remove_branch parameter
# ===========================================================================


class TestCleanupWorktreeKeepBranch:
    """Test cleanup_worktree with remove_branch=False."""

    def test_cleanup_keeps_branch_when_requested(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """cleanup_worktree with remove_branch=False preserves the git branch."""
        wt, branch = create_worktree(git_repo, "t1", attempt=1)

        executor = Executor(
            dispatcher=MockDispatcher(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
            defer_merge=True,
        )

        executor.cleanup_worktree("t1", wt, remove_branch=False)

        # Worktree directory should be gone
        assert not wt.exists()

        # But the branch should still exist
        result = subprocess.run(
            ["git", "branch", "--list", branch],
            capture_output=True,
            text=True,
            cwd=str(git_repo),
        )
        assert branch in result.stdout

        executor.shutdown()

    def test_cleanup_removes_branch_by_default(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """cleanup_worktree removes the branch by default."""
        wt, branch = create_worktree(git_repo, "t1", attempt=1)

        executor = Executor(
            dispatcher=MockDispatcher(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
            defer_merge=True,
        )

        executor.cleanup_worktree("t1", wt)

        # Both worktree and branch should be gone
        assert not wt.exists()
        result = subprocess.run(
            ["git", "branch", "--list", branch],
            capture_output=True,
            text=True,
            cwd=str(git_repo),
        )
        assert branch not in result.stdout

        executor.shutdown()


# ===========================================================================
# Test: PR URL logged in audit events
# ===========================================================================


class TestPRURLInAuditEvents:
    """Verify PR URL is logged in audit events."""

    def test_pr_merge_synced_event_has_url(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """pr_merge_synced audit event includes the actual GitHub PR URL."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()

        wt, branch = create_worktree(git_repo, "t1", attempt=1)

        daemon = TestDaemonPRBasedMerge()._make_daemon(
            git_repo,
            mutation_log,
            work_state,
            audit_log,
            session_logger,
            MockDispatcher(),
        )

        pr_url = "https://github.com/org/repo/pull/99"
        item = CompletedTask(
            task=work_state.get_task("t1"),
            result=AgentResult(output="Done", exit_code=0, duration_s=0.1),
            attempt=1,
            worktree_path=wt,
            agent_id="agent-test",
            pr_info=PRInfo(
                url=pr_url,
                number=99,
                branch=branch,
                title="[corc] Task 1 (t1)",
            ),
        )

        proc_result = ProcessResult(
            task_id="t1",
            passed=True,
            details=[(True, "ok")],
            pr_merged=True,
        )

        with patch("corc.daemon.pull_main", return_value=True):
            daemon._handle_worktree_merge(item, proc_result)

        # Check audit event for PR URL
        events = audit_log.read_for_task("t1")
        sync_events = [e for e in events if e["event_type"] == "pr_merge_synced"]
        assert len(sync_events) == 1
        assert sync_events[0]["pr_url"] == pr_url
        assert sync_events[0]["pr_number"] == 99

        daemon.executor.shutdown()

    def test_worktree_merged_via_pr_event_has_url(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """worktree_merged_via_pr audit event includes the PR URL."""
        wt, branch = create_worktree(git_repo, "t1", attempt=1)

        executor = Executor(
            dispatcher=MockDispatcher(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
            defer_merge=True,
        )

        pr_info = PRInfo(
            url="https://github.com/org/repo/pull/77",
            number=77,
            branch=branch,
            title="test",
        )

        with (
            patch("corc.executor.merge_pr", return_value=True),
            patch("corc.executor.pull_main", return_value=True),
        ):
            status = executor.try_merge_worktree("t1", wt, pr_info=pr_info)

        assert status == "merged"

        events = audit_log.read_for_task("t1")
        pr_events = [e for e in events if e["event_type"] == "worktree_merged_via_pr"]
        assert len(pr_events) == 1
        assert pr_events[0]["pr_url"] == "https://github.com/org/repo/pull/77"
        assert pr_events[0]["pr_number"] == 77

        executor.shutdown()
