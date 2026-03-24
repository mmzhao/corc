"""Tests for PR-based workflow.

Tests cover:
1. Executor git-pulls main before creating worktree
2. Agents create PRs via gh pr create on completion
3. Processor posts validation summary as PR comment via gh pr comment
4. Auto-merge repos: processor merges PR after review comment
5. Human-only repos: PR left open and operator notified
6. No code ever pushed directly to main
7. PR creation and review comment flow for both auto and human-only
"""

import json
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from corc.audit import AuditLog
from corc.dispatch import AgentDispatcher, AgentResult, Constraints
from corc.executor import CompletedTask, Executor
from corc.mutations import MutationLog
from corc.notifications import NotificationManager, notify_pr_awaiting_human_merge
from corc.pr import (
    PRError,
    PRInfo,
    _extract_pr_number,
    _format_review_comment,
    create_pr,
    merge_pr,
    post_review_comment,
    pull_main,
    push_branch,
)
from corc.processor import ProcessResult, process_completed
from corc.sessions import SessionLogger
from corc.state import WorkState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path):
    """Create a real git repository for PR workflow tests."""
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


def _write_repos_yaml(project_root, content):
    """Write a repos.yaml config file."""
    repos_yaml = project_root / ".corc" / "repos.yaml"
    repos_yaml.write_text(content)


# ---------------------------------------------------------------------------
# Mock dispatcher
# ---------------------------------------------------------------------------


class MockDispatcher(AgentDispatcher):
    """Dispatcher that returns a canned result."""

    def __init__(self, result=None):
        self.result = result or AgentResult(
            output="Mock output: task completed.",
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
        return self.result


class CommittingDispatcher(AgentDispatcher):
    """Dispatcher that creates a commit in the worktree.

    Unlike MockDispatcher, this ensures there are commits ahead of main
    so that _create_pr_from_worktree doesn't bail early.
    """

    def __init__(self, result=None):
        self.result = result or AgentResult(
            output="Mock output: task completed.",
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
        if cwd:
            (Path(cwd) / "agent_work.py").write_text("# work\n")
            subprocess.run(
                ["git", "add", "agent_work.py"], cwd=cwd, capture_output=True
            )
            subprocess.run(
                ["git", "commit", "-m", "Agent work"], cwd=cwd, capture_output=True
            )
        return self.result


# ===========================================================================
# Unit tests: pr.py functions
# ===========================================================================


class TestPullMain:
    """Test pull_main() function."""

    def test_pull_main_no_remote(self, git_repo):
        """pull_main returns False when no remote is configured."""
        result = pull_main(git_repo)
        assert result is False

    def test_pull_main_with_remote(self, git_repo):
        """pull_main returns True when remote is configured and pull succeeds."""
        # Set up a bare repo as remote
        bare = git_repo.parent / "bare.git"
        subprocess.run(
            ["git", "clone", "--bare", str(git_repo), str(bare)],
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "remote", "add", "origin", str(bare)],
            cwd=str(git_repo),
            capture_output=True,
            check=True,
        )
        # Set upstream
        subprocess.run(
            ["git", "push", "-u", "origin", "main"],
            cwd=str(git_repo),
            capture_output=True,
        )
        # Now pull should succeed
        result = pull_main(git_repo)
        assert result is True

    @patch("corc.pr.subprocess.run")
    def test_pull_main_handles_subprocess_error(self, mock_run):
        """pull_main returns False on subprocess error."""
        mock_run.side_effect = subprocess.SubprocessError("network error")
        result = pull_main(Path("/tmp/fake"))
        assert result is False


class TestPushBranch:
    """Test push_branch() function."""

    @patch("corc.pr.subprocess.run")
    def test_push_branch_success(self, mock_run):
        """push_branch returns (True, '') on success."""
        mock_run.return_value = MagicMock(returncode=0)
        success, error = push_branch(Path("/tmp/fake"), "corc/task-1-1")
        assert success is True
        assert error == ""
        # Verify the correct command was called
        cmd = mock_run.call_args[0][0]
        assert cmd == ["git", "push", "-u", "origin", "corc/task-1-1"]

    @patch("corc.pr.subprocess.run")
    def test_push_branch_failure(self, mock_run):
        """push_branch returns (False, error_message) on failure."""
        mock_run.return_value = MagicMock(returncode=1, stderr="remote rejected")
        success, error = push_branch(Path("/tmp/fake"), "corc/task-1-1")
        assert success is False
        assert error == "remote rejected"

    @patch("corc.pr.subprocess.run")
    def test_push_branch_exception(self, mock_run):
        """push_branch returns (False, error_message) on exception."""
        mock_run.side_effect = subprocess.SubprocessError("failed")
        success, error = push_branch(Path("/tmp/fake"), "corc/task-1-1")
        assert success is False
        assert "failed" in error


class TestCreatePR:
    """Test create_pr() function."""

    @patch("corc.pr.subprocess.run")
    def test_create_pr_success(self, mock_run):
        """create_pr returns (PRInfo, '') on success."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="https://github.com/org/repo/pull/42\n",
        )
        task = {"id": "t1", "name": "Test Task", "done_when": "tests pass"}
        pr_info, error = create_pr(Path("/tmp/fake"), "corc/t1-1", task)

        assert pr_info is not None
        assert error == ""
        assert pr_info.url == "https://github.com/org/repo/pull/42"
        assert pr_info.number == 42
        assert pr_info.branch == "corc/t1-1"
        assert "Test Task" in pr_info.title

    @patch("corc.pr.subprocess.run")
    def test_create_pr_failure(self, mock_run):
        """create_pr returns (None, error_message) on failure."""
        mock_run.return_value = MagicMock(returncode=1, stderr="no remote configured")
        task = {"id": "t1", "name": "Test Task", "done_when": "tests pass"}
        pr_info, error = create_pr(Path("/tmp/fake"), "corc/t1-1", task)
        assert pr_info is None
        assert error == "no remote configured"

    @patch("corc.pr.subprocess.run")
    def test_create_pr_uses_gh_cli(self, mock_run):
        """create_pr calls gh pr create with correct args."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="https://github.com/org/repo/pull/1\n",
        )
        task = {"id": "t1", "name": "My Task", "done_when": "done"}
        create_pr(Path("/tmp/repo"), "corc/t1-1", task, base_branch="main")

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "gh"
        assert cmd[1] == "pr"
        assert cmd[2] == "create"
        assert "--head" in cmd
        assert "corc/t1-1" in cmd
        assert "--base" in cmd
        assert "main" in cmd
        assert "--title" in cmd
        assert "--body" in cmd


class TestPostReviewComment:
    """Test post_review_comment() function."""

    @patch("corc.pr.subprocess.run")
    def test_post_review_comment_success(self, mock_run):
        """post_review_comment returns True on success."""
        mock_run.return_value = MagicMock(returncode=0)
        result = post_review_comment(
            Path("/tmp/fake"),
            pr_number=42,
            passed=True,
            details=[(True, "All tests pass")],
        )
        assert result is True

    @patch("corc.pr.subprocess.run")
    def test_post_review_comment_failure(self, mock_run):
        """post_review_comment returns False on failure."""
        mock_run.return_value = MagicMock(returncode=1, stderr="not found")
        result = post_review_comment(
            Path("/tmp/fake"),
            pr_number=999,
            passed=True,
            details=[(True, "ok")],
        )
        assert result is False

    @patch("corc.pr.subprocess.run")
    def test_post_review_comment_uses_gh_cli(self, mock_run):
        """post_review_comment calls gh pr comment."""
        mock_run.return_value = MagicMock(returncode=0)
        post_review_comment(
            Path("/tmp/fake"),
            pr_number=42,
            passed=True,
            details=[(True, "Tests pass")],
            findings=["Found improvement opportunity"],
        )

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "gh"
        assert cmd[1] == "pr"
        assert cmd[2] == "comment"
        assert cmd[3] == "42"
        assert "--body" in cmd

    @patch("corc.pr.subprocess.run")
    def test_post_review_comment_includes_validation_details(self, mock_run):
        """Review comment body includes validation details."""
        mock_run.return_value = MagicMock(returncode=0)
        post_review_comment(
            Path("/tmp/fake"),
            pr_number=42,
            passed=False,
            details=[(True, "File exists"), (False, "Tests failed")],
            findings=["Important finding"],
        )

        cmd = mock_run.call_args[0][0]
        body_idx = cmd.index("--body") + 1
        body = cmd[body_idx]
        assert "Validation Failed" in body
        assert "File exists" in body
        assert "Tests failed" in body
        assert "Important finding" in body


class TestMergePR:
    """Test merge_pr() function."""

    @patch("corc.pr.subprocess.run")
    def test_merge_pr_success(self, mock_run):
        """merge_pr returns True on success."""
        mock_run.return_value = MagicMock(returncode=0)
        result = merge_pr(Path("/tmp/fake"), pr_number=42)
        assert result is True

    @patch("corc.pr.subprocess.run")
    def test_merge_pr_failure(self, mock_run):
        """merge_pr returns False on failure."""
        mock_run.return_value = MagicMock(returncode=1, stderr="merge conflict")
        result = merge_pr(Path("/tmp/fake"), pr_number=42)
        assert result is False

    @patch("corc.pr.subprocess.run")
    def test_merge_pr_uses_gh_cli(self, mock_run):
        """merge_pr calls gh pr merge with --merge and --delete-branch."""
        mock_run.return_value = MagicMock(returncode=0)
        merge_pr(Path("/tmp/fake"), pr_number=42)

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "gh"
        assert cmd[1] == "pr"
        assert cmd[2] == "merge"
        assert cmd[3] == "42"
        assert "--merge" in cmd
        assert "--delete-branch" in cmd
        # Must NOT use --auto (that would be auto-merge, blocked by policy)
        assert "--auto" not in cmd


class TestExtractPRNumber:
    """Test _extract_pr_number() helper."""

    def test_extract_from_github_url(self):
        assert _extract_pr_number("https://github.com/org/repo/pull/42") == 42

    def test_extract_from_url_with_trailing_slash(self):
        assert _extract_pr_number("https://github.com/org/repo/pull/42/") == 42

    def test_extract_returns_zero_on_invalid(self):
        assert _extract_pr_number("not-a-url") == 0

    def test_extract_returns_zero_on_empty(self):
        assert _extract_pr_number("") == 0


class TestFormatReviewComment:
    """Test _format_review_comment() helper."""

    def test_passed_comment(self):
        comment = _format_review_comment(
            passed=True,
            details=[(True, "All tests pass")],
        )
        assert "Validation Passed" in comment
        assert "All tests pass" in comment

    def test_failed_comment(self):
        comment = _format_review_comment(
            passed=False,
            details=[(True, "File exists"), (False, "Tests failed")],
        )
        assert "Validation Failed" in comment
        assert "File exists" in comment
        assert "Tests failed" in comment

    def test_comment_with_findings(self):
        comment = _format_review_comment(
            passed=True,
            details=[(True, "ok")],
            findings=["Found a bug", "Performance issue"],
        )
        assert "Findings" in comment
        assert "Found a bug" in comment
        assert "Performance issue" in comment

    def test_comment_without_findings(self):
        comment = _format_review_comment(
            passed=True,
            details=[(True, "ok")],
            findings=None,
        )
        assert "Findings" not in comment


# ===========================================================================
# Integration tests: Executor pulls main before worktree creation
# ===========================================================================


class TestExecutorPullsMain:
    """Test that the executor git-pulls main before creating worktrees."""

    def test_executor_pulls_main_before_worktree(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Executor calls pull_main before creating worktree."""
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
        )

        # patch pull_main to track that it was called
        with patch("corc.executor.pull_main", return_value=True) as mock_pull:
            executor.dispatch(task)
            time.sleep(0.5)
            executor.poll_completed()

        # pull_main should have been called before worktree creation
        mock_pull.assert_called_once_with(git_repo)
        executor.shutdown()

    def test_executor_logs_pull_event(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Executor logs a main_pulled audit event."""
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
        )

        with patch("corc.executor.pull_main", return_value=False):
            executor.dispatch(task)
            time.sleep(0.5)
            executor.poll_completed()

        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "main_pulled" in event_types

        # Verify it logged success=False (no remote configured)
        pull_events = [e for e in events if e["event_type"] == "main_pulled"]
        assert pull_events[0]["success"] is False
        executor.shutdown()

    def test_executor_continues_if_pull_fails(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Executor still creates worktree even if pull fails."""
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
        )

        with patch("corc.executor.pull_main", return_value=False):
            executor.dispatch(task)
            time.sleep(0.5)
            completed = executor.poll_completed()

        # Agent should still have run in a worktree
        assert len(completed) == 1
        assert dispatcher.received_cwd is not None
        assert ".claude/worktrees/" in dispatcher.received_cwd
        executor.shutdown()

    def test_executor_does_not_pull_for_conflict_worktree(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Executor skips pull when reusing a conflict worktree."""
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
        )

        # Create a fake conflict worktree
        fake_wt = git_repo / ".corc" / "worktrees" / "conflict-wt"
        fake_wt.mkdir(parents=True)
        executor.set_conflict_worktree("t1", fake_wt)

        with patch("corc.executor.pull_main") as mock_pull:
            executor.dispatch(task)
            time.sleep(0.5)
            executor.poll_completed()

        # pull_main should NOT have been called (reusing conflict worktree)
        mock_pull.assert_not_called()
        executor.shutdown()


# ===========================================================================
# Integration tests: Executor creates PRs on completion
# ===========================================================================


class TestExecutorCreatesPR:
    """Test that the executor creates PRs after agent completion."""

    def test_executor_creates_pr_on_success(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Executor creates a PR when agent completes successfully."""
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
        )

        mock_pr_info = PRInfo(
            url="https://github.com/org/repo/pull/1",
            number=1,
            branch="corc/t1-1",
            title="[corc] Task 1 (t1)",
        )

        # Mock _create_pr_from_worktree directly: the real method does a
        # subprocess git-log check that fails in test repos with no commits.
        with (
            patch("corc.executor.pull_main", return_value=False),
            patch.object(
                executor, "_create_pr_from_worktree", return_value=mock_pr_info
            ),
        ):
            executor.dispatch(task)
            time.sleep(0.5)
            completed = executor.poll_completed()

        assert len(completed) == 1
        assert completed[0].pr_info is not None
        assert completed[0].pr_info.url == "https://github.com/org/repo/pull/1"
        assert completed[0].pr_info.number == 1
        executor.shutdown()

    def test_executor_no_pr_on_failure(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Executor does not create a PR when agent fails."""
        dispatcher = MockDispatcher(
            result=AgentResult(output="Error", exit_code=1, duration_s=0.1)
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
        )

        with (
            patch("corc.executor.pull_main", return_value=False),
            patch("corc.executor.push_branch") as mock_push,
            patch("corc.executor.create_pr") as mock_create,
        ):
            executor.dispatch(task)
            time.sleep(0.5)
            completed = executor.poll_completed()

        assert len(completed) == 1
        assert completed[0].pr_info is None
        # push and create should not have been called
        mock_push.assert_not_called()
        mock_create.assert_not_called()
        executor.shutdown()

    def test_executor_logs_pr_created_event(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Executor logs pr_created audit event."""
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
        )

        mock_pr_info = PRInfo(
            url="https://github.com/org/repo/pull/5",
            number=5,
            branch="corc/t1-1",
            title="[corc] Task 1 (t1)",
        )

        with (
            patch("corc.executor.pull_main", return_value=False),
            patch("corc.executor.push_branch", return_value=(True, "")),
            patch("corc.executor.create_pr", return_value=(mock_pr_info, "")),
            patch("corc.executor.get_worktree_branch", return_value="corc/t1-1"),
        ):
            executor.dispatch(task)
            time.sleep(0.5)
            executor.poll_completed()

        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "pr_created" in event_types

        pr_events = [e for e in events if e["event_type"] == "pr_created"]
        assert pr_events[0]["pr_number"] == 5
        assert pr_events[0]["pr_url"] == "https://github.com/org/repo/pull/5"
        executor.shutdown()

    def test_executor_handles_push_failure(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Executor handles push failure gracefully."""
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
        )

        with (
            patch("corc.executor.pull_main", return_value=False),
            patch(
                "corc.executor.push_branch",
                return_value=(False, "remote: permission denied"),
            ),
            patch("corc.executor.get_worktree_branch", return_value="corc/t1-1"),
        ):
            executor.dispatch(task)
            time.sleep(0.5)
            completed = executor.poll_completed()

        assert len(completed) == 1
        assert completed[0].pr_info is None

        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "pr_push_failed" in event_types
        executor.shutdown()

    def test_push_failure_audit_includes_error_message(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """pr_push_failed audit event includes the error message from push_branch."""
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
        )

        with (
            patch("corc.executor.pull_main", return_value=False),
            patch(
                "corc.executor.push_branch",
                return_value=(False, "remote: Repository not found"),
            ),
            patch("corc.executor.get_worktree_branch", return_value="corc/t1-1"),
        ):
            executor.dispatch(task)
            time.sleep(0.5)
            completed = executor.poll_completed()

        assert len(completed) == 1
        assert completed[0].pr_info is None

        events = audit_log.read_for_task("t1")
        push_failed_events = [e for e in events if e["event_type"] == "pr_push_failed"]
        assert len(push_failed_events) == 1
        assert push_failed_events[0]["error"] == "remote: Repository not found"
        executor.shutdown()

    def test_pr_creation_failure_audit_includes_error_message(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """pr_creation_failed audit event includes the error message from create_pr."""
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
        )

        with (
            patch("corc.executor.pull_main", return_value=False),
            patch("corc.executor.push_branch", return_value=(True, "")),
            patch(
                "corc.executor.create_pr",
                return_value=(None, "pull request already exists for branch"),
            ),
            patch("corc.executor.get_worktree_branch", return_value="corc/t1-1"),
        ):
            executor.dispatch(task)
            time.sleep(0.5)
            completed = executor.poll_completed()

        assert len(completed) == 1
        assert completed[0].pr_info is None

        events = audit_log.read_for_task("t1")
        pr_failed_events = [
            e for e in events if e["event_type"] == "pr_creation_failed"
        ]
        assert len(pr_failed_events) == 1
        assert pr_failed_events[0]["error"] == "pull request already exists for branch"
        executor.shutdown()


# ===========================================================================
# Integration tests: Processor posts review comments
# ===========================================================================


class TestProcessorReviewComments:
    """Test processor posts validation summary as PR comment."""

    def test_processor_posts_review_comment_on_pass(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Processor posts review comment when validation passes."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        pr_info = PRInfo(
            url="https://github.com/org/repo/pull/10",
            number=10,
            branch="corc/t1-1",
            title="[corc] Task 1 (t1)",
        )
        result = AgentResult(output="Done", exit_code=0, duration_s=0.1)

        with (
            patch(
                "corc.processor.post_review_comment", return_value=True
            ) as mock_comment,
            patch("corc.processor.merge_pr", return_value=True),
        ):
            proc_result = process_completed(
                task=task,
                result=result,
                attempt=1,
                mutation_log=mutation_log,
                state=work_state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=git_repo,
                pr_info=pr_info,
            )

        # Verify review comment was posted
        mock_comment.assert_called_once()
        call_args = mock_comment.call_args
        assert call_args[0][1] == 10  # pr_number
        assert call_args[0][2] is True  # passed
        assert proc_result.pr_commented is True

    def test_processor_posts_review_comment_on_fail(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Processor posts review comment even when validation fails."""
        _create_task(
            mutation_log,
            "t1",
            "Task 1",
            done_when='[{"type": "file_exists", "path": "/nonexistent/file.py"}]',
        )
        work_state.refresh()
        task = work_state.get_task("t1")

        pr_info = PRInfo(
            url="https://github.com/org/repo/pull/11",
            number=11,
            branch="corc/t1-1",
            title="[corc] Task 1 (t1)",
        )
        result = AgentResult(output="Done", exit_code=0, duration_s=0.1)

        with (
            patch(
                "corc.processor.post_review_comment", return_value=True
            ) as mock_comment,
            patch("corc.processor.merge_pr", return_value=False),
        ):
            proc_result = process_completed(
                task=task,
                result=result,
                attempt=1,
                mutation_log=mutation_log,
                state=work_state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=git_repo,
                pr_info=pr_info,
            )

        # Review comment should still be posted (even though validation failed)
        mock_comment.assert_called_once()
        assert proc_result.pr_commented is True

    def test_processor_no_comment_without_pr_info(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Processor does not post comment when no PR info is available."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        result = AgentResult(output="Done", exit_code=0, duration_s=0.1)

        with (
            patch("corc.processor.post_review_comment") as mock_comment,
            patch("corc.processor.merge_pr"),
        ):
            proc_result = process_completed(
                task=task,
                result=result,
                attempt=1,
                mutation_log=mutation_log,
                state=work_state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=git_repo,
                pr_info=None,
            )

        # No comment should be posted
        mock_comment.assert_not_called()
        assert proc_result.pr_commented is False

    def test_processor_logs_review_comment_event(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Processor logs pr_review_comment_posted audit event."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        pr_info = PRInfo(
            url="https://github.com/org/repo/pull/10",
            number=10,
            branch="corc/t1-1",
            title="[corc] Task 1 (t1)",
        )
        result = AgentResult(output="Done", exit_code=0, duration_s=0.1)

        with (
            patch("corc.processor.post_review_comment", return_value=True),
            patch("corc.processor.merge_pr", return_value=True),
        ):
            process_completed(
                task=task,
                result=result,
                attempt=1,
                mutation_log=mutation_log,
                state=work_state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=git_repo,
                pr_info=pr_info,
            )

        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "pr_review_comment_posted" in event_types


# ===========================================================================
# Integration tests: Auto-merge repos — processor merges PR
# ===========================================================================


class TestAutoMergePRWorkflow:
    """Test auto-merge repos: processor merges PR after review comment."""

    def test_auto_merge_merges_pr_after_comment(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Auto-merge repo: processor merges PR after posting review comment."""
        # Default policy is auto
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        pr_info = PRInfo(
            url="https://github.com/org/repo/pull/10",
            number=10,
            branch="corc/t1-1",
            title="[corc] Task 1 (t1)",
        )
        result = AgentResult(output="Done", exit_code=0, duration_s=0.1)

        with (
            patch(
                "corc.processor.post_review_comment", return_value=True
            ) as mock_comment,
            patch("corc.processor.merge_pr", return_value=True) as mock_merge,
        ):
            proc_result = process_completed(
                task=task,
                result=result,
                attempt=1,
                mutation_log=mutation_log,
                state=work_state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=git_repo,
                pr_info=pr_info,
            )

        # Both comment and merge should have been called
        mock_comment.assert_called_once()
        mock_merge.assert_called_once_with(git_repo, 10)
        assert proc_result.pr_merged is True
        assert proc_result.pr_commented is True

    def test_auto_merge_logs_pr_merged_event(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Auto-merge repo: processor logs pr_merged audit event."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        pr_info = PRInfo(
            url="https://github.com/org/repo/pull/10",
            number=10,
            branch="corc/t1-1",
            title="[corc] Task 1 (t1)",
        )
        result = AgentResult(output="Done", exit_code=0, duration_s=0.1)

        with (
            patch("corc.processor.post_review_comment", return_value=True),
            patch("corc.processor.merge_pr", return_value=True),
        ):
            process_completed(
                task=task,
                result=result,
                attempt=1,
                mutation_log=mutation_log,
                state=work_state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=git_repo,
                pr_info=pr_info,
            )

        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "pr_merged" in event_types

    def test_auto_merge_marks_task_completed(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Auto-merge repo: task is marked as completed."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        pr_info = PRInfo(
            url="https://github.com/org/repo/pull/10",
            number=10,
            branch="corc/t1-1",
            title="[corc] Task 1 (t1)",
        )
        result = AgentResult(output="Done", exit_code=0, duration_s=0.1)

        with (
            patch("corc.processor.post_review_comment", return_value=True),
            patch("corc.processor.merge_pr", return_value=True),
        ):
            proc_result = process_completed(
                task=task,
                result=result,
                attempt=1,
                mutation_log=mutation_log,
                state=work_state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=git_repo,
                pr_info=pr_info,
            )

        assert proc_result.passed is True
        work_state.refresh()
        t = work_state.get_task("t1")
        assert t["status"] == "completed"

    def test_auto_merge_records_pr_info_in_mutation(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Auto-merge repo: task_completed mutation includes PR info."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        pr_info = PRInfo(
            url="https://github.com/org/repo/pull/42",
            number=42,
            branch="corc/t1-1",
            title="[corc] Task 1 (t1)",
        )
        result = AgentResult(output="Done", exit_code=0, duration_s=0.1)

        with (
            patch("corc.processor.post_review_comment", return_value=True),
            patch("corc.processor.merge_pr", return_value=True),
        ):
            process_completed(
                task=task,
                result=result,
                attempt=1,
                mutation_log=mutation_log,
                state=work_state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=git_repo,
                pr_info=pr_info,
            )

        # Check mutation log for pr_url
        entries = mutation_log.read_all()
        completed_entries = [e for e in entries if e.get("type") == "task_completed"]
        assert len(completed_entries) >= 1
        last_completed = completed_entries[-1]
        assert last_completed["data"]["pr_url"] == "https://github.com/org/repo/pull/42"
        assert last_completed["data"]["pr_number"] == 42
        assert last_completed["data"]["pr_merged"] is True


# ===========================================================================
# Integration tests: Human-only repos — PR left open, operator notified
# ===========================================================================


class TestHumanOnlyPRWorkflow:
    """Test human-only repos: PR left open and operator notified."""

    def test_human_only_does_not_merge_pr(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Human-only repo: processor does NOT merge PR."""
        _write_repos_yaml(
            git_repo,
            """
repos:
  repo:
    merge_policy: human-only
    protected_branches: [main]
""",
        )
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        pr_info = PRInfo(
            url="https://github.com/org/repo/pull/20",
            number=20,
            branch="corc/t1-1",
            title="[corc] Task 1 (t1)",
        )
        result = AgentResult(output="Done", exit_code=0, duration_s=0.1)

        with (
            patch(
                "corc.processor.post_review_comment", return_value=True
            ) as mock_comment,
            patch("corc.processor.merge_pr") as mock_merge,
            patch("corc.processor.get_repo_policy") as mock_policy,
        ):
            from corc.repo_policy import RepoPolicy

            mock_policy.return_value = RepoPolicy(
                name="repo", merge_policy="human-only"
            )
            proc_result = process_completed(
                task=task,
                result=result,
                attempt=1,
                mutation_log=mutation_log,
                state=work_state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=git_repo,
                pr_info=pr_info,
            )

        # Comment should be posted, but merge should NOT be called
        mock_comment.assert_called_once()
        mock_merge.assert_not_called()
        assert proc_result.pr_commented is True
        assert proc_result.pr_merged is False

    def test_human_only_marks_task_pending_merge(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Human-only repo: task is marked as pending_merge."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        pr_info = PRInfo(
            url="https://github.com/org/repo/pull/20",
            number=20,
            branch="corc/t1-1",
            title="[corc] Task 1 (t1)",
        )
        result = AgentResult(output="Done", exit_code=0, duration_s=0.1)

        with (
            patch("corc.processor.post_review_comment", return_value=True),
            patch("corc.processor.merge_pr"),
            patch("corc.processor.get_repo_policy") as mock_policy,
        ):
            from corc.repo_policy import RepoPolicy

            mock_policy.return_value = RepoPolicy(
                name="repo", merge_policy="human-only"
            )
            proc_result = process_completed(
                task=task,
                result=result,
                attempt=1,
                mutation_log=mutation_log,
                state=work_state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=git_repo,
                pr_info=pr_info,
            )

        assert proc_result.passed is True
        work_state.refresh()
        t = work_state.get_task("t1")
        assert t["status"] == "pending_merge"

    def test_human_only_notifies_operator(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Human-only repo: operator is notified about PR awaiting merge."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        pr_info = PRInfo(
            url="https://github.com/org/repo/pull/20",
            number=20,
            branch="corc/t1-1",
            title="[corc] Task 1 (t1)",
        )
        result = AgentResult(output="Done", exit_code=0, duration_s=0.1)

        mock_notifier = MagicMock(spec=NotificationManager)
        mock_notifier.notify.return_value = {"terminal": True}

        with (
            patch("corc.processor.post_review_comment", return_value=True),
            patch("corc.processor.merge_pr"),
            patch("corc.processor.get_repo_policy") as mock_policy,
        ):
            from corc.repo_policy import RepoPolicy

            mock_policy.return_value = RepoPolicy(
                name="repo", merge_policy="human-only"
            )
            process_completed(
                task=task,
                result=result,
                attempt=1,
                mutation_log=mutation_log,
                state=work_state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=git_repo,
                pr_info=pr_info,
                notification_manager=mock_notifier,
            )

        # Notification should have been sent
        mock_notifier.notify.assert_called()
        # Check that the notification is about PR awaiting merge
        notify_call = mock_notifier.notify.call_args
        assert (
            "awaiting" in notify_call[1].get("title", "")
            or "awaiting" in notify_call[0][1]
            if len(notify_call[0]) > 1
            else True
        )

    def test_human_only_records_pr_info_in_mutation(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Human-only repo: task_pending_merge mutation includes PR info."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        pr_info = PRInfo(
            url="https://github.com/org/repo/pull/20",
            number=20,
            branch="corc/t1-1",
            title="[corc] Task 1 (t1)",
        )
        result = AgentResult(output="Done", exit_code=0, duration_s=0.1)

        with (
            patch("corc.processor.post_review_comment", return_value=True),
            patch("corc.processor.merge_pr"),
            patch("corc.processor.get_repo_policy") as mock_policy,
        ):
            from corc.repo_policy import RepoPolicy

            mock_policy.return_value = RepoPolicy(
                name="repo", merge_policy="human-only"
            )
            process_completed(
                task=task,
                result=result,
                attempt=1,
                mutation_log=mutation_log,
                state=work_state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=git_repo,
                pr_info=pr_info,
            )

        entries = mutation_log.read_all()
        pending_entries = [e for e in entries if e.get("type") == "task_pending_merge"]
        assert len(pending_entries) >= 1
        last = pending_entries[-1]
        assert last["data"]["pr_url"] == "https://github.com/org/repo/pull/20"
        assert last["data"]["pr_number"] == 20


# ===========================================================================
# Integration tests: No code pushed directly to main
# ===========================================================================


class TestNoPushToMain:
    """Verify the PR workflow ensures no direct push to main."""

    def test_executor_pushes_branch_not_main(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Executor pushes worktree branch, never main."""
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
        )

        with (
            patch("corc.executor.pull_main", return_value=False),
            patch("corc.executor.push_branch", return_value=(True, "")) as mock_push,
            patch(
                "corc.executor.create_pr",
                return_value=(
                    PRInfo(
                        url="https://github.com/org/repo/pull/1",
                        number=1,
                        branch="corc/t1-1",
                        title="test",
                    ),
                    "",
                ),
            ),
            patch("corc.executor.get_worktree_branch", return_value="corc/t1-1"),
        ):
            executor.dispatch(task)
            time.sleep(0.5)
            executor.poll_completed()

        # push_branch should be called with the worktree branch, not main
        mock_push.assert_called_once()
        pushed_branch = mock_push.call_args[0][1]
        assert pushed_branch == "corc/t1-1"
        assert pushed_branch != "main"
        executor.shutdown()

    def test_merge_goes_through_pr_not_direct(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """For auto repos, merge is done via gh pr merge, not git merge to main."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        pr_info = PRInfo(
            url="https://github.com/org/repo/pull/10",
            number=10,
            branch="corc/t1-1",
            title="test",
        )
        result = AgentResult(output="Done", exit_code=0, duration_s=0.1)

        with (
            patch("corc.processor.post_review_comment", return_value=True),
            patch("corc.processor.merge_pr", return_value=True) as mock_merge,
        ):
            process_completed(
                task=task,
                result=result,
                attempt=1,
                mutation_log=mutation_log,
                state=work_state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=git_repo,
                pr_info=pr_info,
            )

        # Merge should go through gh pr merge (PR-based), not direct git merge
        mock_merge.assert_called_once_with(git_repo, 10)


# ===========================================================================
# Unit tests: notify_pr_awaiting_human_merge
# ===========================================================================


class TestNotifyPRAwaiting:
    """Test the PR awaiting human merge notification."""

    def test_notify_sends_notification(self):
        """notify_pr_awaiting_human_merge sends to notification manager."""
        mock_manager = MagicMock(spec=NotificationManager)
        mock_manager.notify.return_value = {"terminal": True}

        task = {"id": "t1", "name": "Task 1"}
        pr_info = PRInfo(
            url="https://github.com/org/repo/pull/42",
            number=42,
            branch="corc/t1-1",
            title="test",
        )

        result = notify_pr_awaiting_human_merge(mock_manager, task, pr_info)

        mock_manager.notify.assert_called_once()
        call_args = mock_manager.notify.call_args
        assert "pr_awaiting_merge" in call_args[0][0]  # event_type
        assert "42" in call_args[0][1]  # title contains PR number
        assert "Task 1" in call_args[0][1]  # title contains task name
        assert (
            "human-only" in call_args[0][2].lower()
            or "merge" in call_args[0][2].lower()
        )

    def test_notify_includes_pr_url(self):
        """Notification body includes PR URL."""
        mock_manager = MagicMock(spec=NotificationManager)
        mock_manager.notify.return_value = {"terminal": True}

        task = {"id": "t1", "name": "Task 1"}
        pr_info = PRInfo(
            url="https://github.com/org/repo/pull/42",
            number=42,
            branch="corc/t1-1",
            title="test",
        )

        notify_pr_awaiting_human_merge(mock_manager, task, pr_info)

        call_args = mock_manager.notify.call_args
        body = call_args[0][2]  # third positional arg is body
        assert "https://github.com/org/repo/pull/42" in body


# ===========================================================================
# End-to-end workflow tests
# ===========================================================================


class TestEndToEndPRWorkflow:
    """End-to-end tests that simulate the complete PR workflow."""

    def test_auto_repo_full_flow(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Full flow for auto repo: pull → dispatch → PR create → validate → comment → merge."""
        dispatcher = CommittingDispatcher()
        _create_task(mutation_log, "t1", "Implement feature")
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

        mock_pr_info = PRInfo(
            url="https://github.com/org/repo/pull/100",
            number=100,
            branch="corc/t1-1",
            title="[corc] Implement feature (t1)",
        )

        # Step 1: Executor dispatches (pulls main, creates worktree, runs agent)
        # Step 2: Agent completes, executor creates PR
        with (
            patch("corc.executor.pull_main", return_value=True) as mock_pull,
            patch("corc.executor.push_branch", return_value=(True, "")),
            patch("corc.executor.create_pr", return_value=(mock_pr_info, "")),
            patch("corc.executor.get_worktree_branch", return_value="corc/t1-1"),
        ):
            executor.dispatch(task)
            time.sleep(0.5)
            completed = executor.poll_completed()

        assert len(completed) == 1
        item = completed[0]
        assert item.pr_info is not None
        assert item.pr_info.number == 100

        # Step 3: Processor validates, posts comment, merges PR
        with (
            patch(
                "corc.processor.post_review_comment", return_value=True
            ) as mock_comment,
            patch("corc.processor.merge_pr", return_value=True) as mock_merge,
        ):
            proc_result = process_completed(
                task=item.task,
                result=item.result,
                attempt=item.attempt,
                mutation_log=mutation_log,
                state=work_state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=git_repo,
                pr_info=item.pr_info,
            )

        # Verify full flow
        mock_pull.assert_called_once()
        mock_comment.assert_called_once()
        mock_merge.assert_called_once_with(git_repo, 100)
        assert proc_result.passed is True
        assert proc_result.pr_commented is True
        assert proc_result.pr_merged is True

        # Task should be completed
        work_state.refresh()
        t = work_state.get_task("t1")
        assert t["status"] == "completed"
        executor.shutdown()

    def test_human_only_repo_full_flow(
        self, git_repo, mutation_log, work_state, audit_log, session_logger
    ):
        """Full flow for human-only repo: pull → dispatch → PR create → validate → comment → notify (no merge)."""
        dispatcher = CommittingDispatcher()
        _create_task(mutation_log, "t1", "Fix production bug")
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

        mock_pr_info = PRInfo(
            url="https://github.com/org/repo/pull/200",
            number=200,
            branch="corc/t1-1",
            title="[corc] Fix production bug (t1)",
        )

        # Step 1-2: Executor dispatches and creates PR
        with (
            patch("corc.executor.pull_main", return_value=True),
            patch("corc.executor.push_branch", return_value=(True, "")),
            patch("corc.executor.create_pr", return_value=(mock_pr_info, "")),
            patch("corc.executor.get_worktree_branch", return_value="corc/t1-1"),
        ):
            executor.dispatch(task)
            time.sleep(0.5)
            completed = executor.poll_completed()

        item = completed[0]
        mock_notifier = MagicMock(spec=NotificationManager)
        mock_notifier.notify.return_value = {"terminal": True}

        # Step 3: Processor validates, posts comment, does NOT merge
        with (
            patch(
                "corc.processor.post_review_comment", return_value=True
            ) as mock_comment,
            patch("corc.processor.merge_pr") as mock_merge,
            patch("corc.processor.get_repo_policy") as mock_policy,
        ):
            from corc.repo_policy import RepoPolicy

            mock_policy.return_value = RepoPolicy(
                name="repo", merge_policy="human-only"
            )
            proc_result = process_completed(
                task=item.task,
                result=item.result,
                attempt=item.attempt,
                mutation_log=mutation_log,
                state=work_state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=git_repo,
                pr_info=item.pr_info,
                notification_manager=mock_notifier,
            )

        # Verify: comment posted, merge NOT called, operator notified
        mock_comment.assert_called_once()
        mock_merge.assert_not_called()
        assert proc_result.pr_commented is True
        assert proc_result.pr_merged is False

        # Operator should be notified
        mock_notifier.notify.assert_called()

        # Task should be pending_merge (not completed)
        work_state.refresh()
        t = work_state.get_task("t1")
        assert t["status"] == "pending_merge"
        executor.shutdown()
