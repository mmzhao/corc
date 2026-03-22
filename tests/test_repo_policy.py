"""Tests for repo merge policies.

Tests cover:
1. Policy configuration loading from .corc/repos.yaml
2. Auto vs human-only merge policy enforcement
3. PreToolUse hooks blocking git push to protected branches
4. PreToolUse hooks blocking gh pr merge --auto
5. Processor respecting merge policy (creates PR but skips merge for human-only)
6. Executor respecting merge policy (skips worktree merge for human-only)
"""

import json
import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from corc.audit import AuditLog
from corc.dispatch import AgentDispatcher, AgentResult, Constraints
from corc.executor import CompletedTask, Executor
from corc.hooks import check_bash_command, pre_tool_use_hook
from corc.mutations import MutationLog
from corc.processor import ProcessResult, process_completed
from corc.repo_policy import (
    RepoPolicy,
    check_auto_merge_allowed,
    check_push_allowed,
    get_repo_policy,
    is_protected_branch,
    load_repo_policies,
)
from corc.sessions import SessionLogger
from corc.state import WorkState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project structure with .corc directory."""
    project = tmp_path / "project"
    project.mkdir()
    (project / ".corc").mkdir()
    (project / "data").mkdir()
    (project / "data" / "events").mkdir()
    (project / "data" / "sessions").mkdir()
    return project


@pytest.fixture
def git_repo(tmp_path):
    """Create a real git repository for merge policy tests."""
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


def _write_repos_yaml(project_root, content):
    """Write a repos.yaml config file."""
    repos_yaml = project_root / ".corc" / "repos.yaml"
    repos_yaml.write_text(content)


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


# ===========================================================================
# Unit tests: RepoPolicy dataclass
# ===========================================================================


class TestRepoPolicy:
    """Test RepoPolicy dataclass."""

    def test_default_policy_is_auto(self):
        """Default merge policy is auto."""
        policy = RepoPolicy(name="test")
        assert policy.merge_policy == "auto"
        assert policy.is_auto is True
        assert policy.is_human_only is False

    def test_human_only_policy(self):
        """human-only policy sets block flags automatically."""
        policy = RepoPolicy(name="test", merge_policy="human-only")
        assert policy.is_human_only is True
        assert policy.is_auto is False
        assert policy.block_auto_merge is True
        assert policy.block_direct_push is True

    def test_auto_policy_no_blocks(self):
        """Auto policy does not force block flags."""
        policy = RepoPolicy(name="test", merge_policy="auto")
        assert policy.block_auto_merge is False
        assert policy.block_direct_push is False

    def test_invalid_policy_raises(self):
        """Invalid merge policy raises ValueError."""
        with pytest.raises(ValueError, match="Invalid merge_policy"):
            RepoPolicy(name="test", merge_policy="invalid")

    def test_protected_branches_default(self):
        """Default protected branches is [main]."""
        policy = RepoPolicy(name="test")
        assert policy.protected_branches == ["main"]

    def test_custom_protected_branches(self):
        """Custom protected branches list."""
        policy = RepoPolicy(
            name="test",
            protected_branches=["main", "staging", "production"],
        )
        assert "main" in policy.protected_branches
        assert "staging" in policy.protected_branches
        assert "production" in policy.protected_branches


# ===========================================================================
# Unit tests: load_repo_policies
# ===========================================================================


class TestLoadRepoPolicies:
    """Test loading policies from repos.yaml."""

    def test_load_auto_policy(self, tmp_project):
        """Load an auto merge policy from repos.yaml."""
        _write_repos_yaml(tmp_project, """
repos:
  internal-tool:
    merge_policy: auto
    protected_branches: [main]
    require_reviewer_approval: true
""")
        policies = load_repo_policies(tmp_project)
        assert "internal-tool" in policies
        assert policies["internal-tool"].is_auto is True
        assert policies["internal-tool"].protected_branches == ["main"]

    def test_load_human_only_policy(self, tmp_project):
        """Load a human-only merge policy from repos.yaml."""
        _write_repos_yaml(tmp_project, """
repos:
  production-app:
    merge_policy: human-only
    protected_branches: [main, staging]
    require_reviewer_approval: true
    block_auto_merge: true
    block_direct_push: true
""")
        policies = load_repo_policies(tmp_project)
        assert "production-app" in policies
        p = policies["production-app"]
        assert p.is_human_only is True
        assert p.block_auto_merge is True
        assert p.block_direct_push is True
        assert "main" in p.protected_branches
        assert "staging" in p.protected_branches

    def test_load_multiple_repos(self, tmp_project):
        """Load multiple repo policies."""
        _write_repos_yaml(tmp_project, """
repos:
  repo-a:
    merge_policy: auto
    protected_branches: [main]
  repo-b:
    merge_policy: human-only
    protected_branches: [main, staging]
""")
        policies = load_repo_policies(tmp_project)
        assert len(policies) == 2
        assert policies["repo-a"].is_auto is True
        assert policies["repo-b"].is_human_only is True

    def test_missing_repos_yaml_returns_empty(self, tmp_project):
        """Missing repos.yaml returns empty dict (not an error)."""
        policies = load_repo_policies(tmp_project)
        assert policies == {}

    def test_empty_repos_yaml_returns_empty(self, tmp_project):
        """Empty repos.yaml returns empty dict."""
        _write_repos_yaml(tmp_project, "")
        policies = load_repo_policies(tmp_project)
        assert policies == {}

    def test_repos_yaml_with_no_repos_key(self, tmp_project):
        """repos.yaml without 'repos' key returns empty dict."""
        _write_repos_yaml(tmp_project, "foo: bar\n")
        policies = load_repo_policies(tmp_project)
        assert policies == {}


# ===========================================================================
# Unit tests: get_repo_policy
# ===========================================================================


class TestGetRepoPolicy:
    """Test getting a repo policy by name."""

    def test_returns_configured_policy(self, tmp_project):
        """Returns policy from config when repo name matches."""
        _write_repos_yaml(tmp_project, """
repos:
  my-repo:
    merge_policy: human-only
    protected_branches: [main, staging]
""")
        policy = get_repo_policy(tmp_project, repo_name="my-repo")
        assert policy.is_human_only is True
        assert "staging" in policy.protected_branches

    def test_returns_default_auto_for_unknown_repo(self, tmp_project):
        """Unknown repo name returns default auto policy."""
        _write_repos_yaml(tmp_project, """
repos:
  some-other-repo:
    merge_policy: human-only
""")
        policy = get_repo_policy(tmp_project, repo_name="unknown-repo")
        assert policy.is_auto is True
        assert policy.name == "unknown-repo"

    def test_returns_default_when_no_config(self, tmp_project):
        """Returns default auto policy when no repos.yaml exists."""
        policy = get_repo_policy(tmp_project, repo_name="any-repo")
        assert policy.is_auto is True


# ===========================================================================
# Unit tests: is_protected_branch
# ===========================================================================


class TestIsProtectedBranch:
    """Test protected branch checking."""

    def test_main_is_protected_by_default(self, tmp_project):
        """main is protected by default."""
        assert is_protected_branch(tmp_project, "main", repo_name="test") is True

    def test_feature_branch_not_protected(self, tmp_project):
        """Feature branches are not protected by default."""
        assert is_protected_branch(tmp_project, "feature/abc", repo_name="test") is False

    def test_custom_protected_branches(self, tmp_project):
        """Custom protected branches are respected."""
        _write_repos_yaml(tmp_project, """
repos:
  test:
    merge_policy: human-only
    protected_branches: [main, staging, production]
""")
        assert is_protected_branch(tmp_project, "main", repo_name="test") is True
        assert is_protected_branch(tmp_project, "staging", repo_name="test") is True
        assert is_protected_branch(tmp_project, "production", repo_name="test") is True
        assert is_protected_branch(tmp_project, "develop", repo_name="test") is False


# ===========================================================================
# Unit tests: check_push_allowed
# ===========================================================================


class TestCheckPushAllowed:
    """Test push blocking for protected branches."""

    def test_push_allowed_with_auto_policy(self, tmp_project):
        """Push to main is allowed with auto policy (block_direct_push=false)."""
        _write_repos_yaml(tmp_project, """
repos:
  test:
    merge_policy: auto
    protected_branches: [main]
""")
        allowed, reason = check_push_allowed(
            tmp_project, "git push origin main", repo_name="test"
        )
        assert allowed is True

    def test_push_blocked_with_human_only_policy(self, tmp_project):
        """Push to main is blocked with human-only policy."""
        _write_repos_yaml(tmp_project, """
repos:
  test:
    merge_policy: human-only
    protected_branches: [main]
""")
        allowed, reason = check_push_allowed(
            tmp_project, "git push origin main", repo_name="test"
        )
        assert allowed is False
        assert "protected branch" in reason.lower()
        assert "main" in reason

    def test_push_to_feature_branch_allowed(self, tmp_project):
        """Push to feature branch is always allowed."""
        _write_repos_yaml(tmp_project, """
repos:
  test:
    merge_policy: human-only
    protected_branches: [main]
""")
        allowed, reason = check_push_allowed(
            tmp_project, "git push origin feature/abc", repo_name="test"
        )
        assert allowed is True

    def test_push_with_refspec_blocked(self, tmp_project):
        """Push with HEAD:main refspec is blocked."""
        _write_repos_yaml(tmp_project, """
repos:
  test:
    merge_policy: human-only
    protected_branches: [main]
""")
        allowed, reason = check_push_allowed(
            tmp_project, "git push origin HEAD:main", repo_name="test"
        )
        assert allowed is False

    def test_push_to_staging_blocked(self, tmp_project):
        """Push to staging is blocked when staging is protected."""
        _write_repos_yaml(tmp_project, """
repos:
  test:
    merge_policy: human-only
    protected_branches: [main, staging]
""")
        allowed, reason = check_push_allowed(
            tmp_project, "git push origin staging", repo_name="test"
        )
        assert allowed is False
        assert "staging" in reason

    def test_push_allowed_when_no_config(self, tmp_project):
        """Push is allowed when no repos.yaml config exists."""
        allowed, reason = check_push_allowed(
            tmp_project, "git push origin main", repo_name="test"
        )
        assert allowed is True


# ===========================================================================
# Unit tests: check_auto_merge_allowed
# ===========================================================================


class TestCheckAutoMergeAllowed:
    """Test auto-merge blocking."""

    def test_auto_merge_allowed_with_auto_policy(self, tmp_project):
        """gh pr merge --auto is allowed with auto policy."""
        _write_repos_yaml(tmp_project, """
repos:
  test:
    merge_policy: auto
    protected_branches: [main]
""")
        allowed, reason = check_auto_merge_allowed(
            tmp_project, "gh pr merge 123 --auto", repo_name="test"
        )
        assert allowed is True

    def test_auto_merge_blocked_with_human_only_policy(self, tmp_project):
        """gh pr merge --auto is blocked with human-only policy."""
        _write_repos_yaml(tmp_project, """
repos:
  test:
    merge_policy: human-only
    protected_branches: [main]
""")
        allowed, reason = check_auto_merge_allowed(
            tmp_project, "gh pr merge 123 --auto", repo_name="test"
        )
        assert allowed is False
        assert "auto-merge" in reason.lower() or "block_auto_merge" in reason

    def test_regular_merge_allowed_even_human_only(self, tmp_project):
        """gh pr merge without --auto is allowed (just creates the PR)."""
        _write_repos_yaml(tmp_project, """
repos:
  test:
    merge_policy: human-only
    protected_branches: [main]
""")
        allowed, reason = check_auto_merge_allowed(
            tmp_project, "gh pr merge 123", repo_name="test"
        )
        assert allowed is True

    def test_auto_merge_blocked_with_explicit_flag(self, tmp_project):
        """block_auto_merge can be set explicitly on auto policy."""
        _write_repos_yaml(tmp_project, """
repos:
  test:
    merge_policy: auto
    protected_branches: [main]
    block_auto_merge: true
""")
        allowed, reason = check_auto_merge_allowed(
            tmp_project, "gh pr merge --auto 456", repo_name="test"
        )
        assert allowed is False

    def test_auto_merge_allowed_no_config(self, tmp_project):
        """Auto merge allowed when no config exists."""
        allowed, reason = check_auto_merge_allowed(
            tmp_project, "gh pr merge --auto 789", repo_name="test"
        )
        assert allowed is True


# ===========================================================================
# Unit tests: PreToolUse hooks
# ===========================================================================


class TestPreToolUseHook:
    """Test the PreToolUse hook entry point."""

    def test_bash_git_push_blocked(self, tmp_project):
        """Bash tool git push to protected branch is blocked."""
        _write_repos_yaml(tmp_project, """
repos:
  test:
    merge_policy: human-only
    protected_branches: [main]
""")
        allowed, reason = pre_tool_use_hook(
            "Bash",
            {"command": "git push origin main"},
            tmp_project,
            repo_name="test",
        )
        assert allowed is False
        assert "protected branch" in reason.lower()

    def test_bash_gh_auto_merge_blocked(self, tmp_project):
        """Bash tool gh pr merge --auto is blocked."""
        _write_repos_yaml(tmp_project, """
repos:
  test:
    merge_policy: human-only
    protected_branches: [main]
""")
        allowed, reason = pre_tool_use_hook(
            "Bash",
            {"command": "gh pr merge 123 --auto --squash"},
            tmp_project,
            repo_name="test",
        )
        assert allowed is False
        assert "auto-merge" in reason.lower() or "block_auto_merge" in reason

    def test_non_bash_tool_allowed(self, tmp_project):
        """Non-Bash tools are always allowed."""
        _write_repos_yaml(tmp_project, """
repos:
  test:
    merge_policy: human-only
    protected_branches: [main]
""")
        allowed, reason = pre_tool_use_hook(
            "Read",
            {"file_path": "/some/file"},
            tmp_project,
            repo_name="test",
        )
        assert allowed is True

    def test_bash_non_git_command_allowed(self, tmp_project):
        """Non-git Bash commands are always allowed."""
        _write_repos_yaml(tmp_project, """
repos:
  test:
    merge_policy: human-only
    protected_branches: [main]
""")
        allowed, reason = pre_tool_use_hook(
            "Bash",
            {"command": "npm install"},
            tmp_project,
            repo_name="test",
        )
        assert allowed is True


# ===========================================================================
# Unit tests: check_bash_command
# ===========================================================================


class TestCheckBashCommand:
    """Test bash command checking for chained commands."""

    def test_chained_git_push_blocked(self, tmp_project):
        """git push in a && chain is blocked."""
        _write_repos_yaml(tmp_project, """
repos:
  test:
    merge_policy: human-only
    protected_branches: [main]
""")
        allowed, reason = check_bash_command(
            "npm test && git push origin main",
            tmp_project,
            repo_name="test",
        )
        assert allowed is False

    def test_chained_auto_merge_blocked(self, tmp_project):
        """gh pr merge --auto in a ; chain is blocked."""
        _write_repos_yaml(tmp_project, """
repos:
  test:
    merge_policy: human-only
    protected_branches: [main]
""")
        allowed, reason = check_bash_command(
            "echo done; gh pr merge 123 --auto",
            tmp_project,
            repo_name="test",
        )
        assert allowed is False

    def test_safe_commands_allowed(self, tmp_project):
        """Safe commands are not blocked."""
        _write_repos_yaml(tmp_project, """
repos:
  test:
    merge_policy: human-only
    protected_branches: [main]
""")
        allowed, reason = check_bash_command(
            "git status && git diff",
            tmp_project,
            repo_name="test",
        )
        assert allowed is True


# ===========================================================================
# Mock dispatcher for executor/processor tests
# ===========================================================================


class MockDispatcher(AgentDispatcher):
    """Dispatcher that returns a configurable result."""

    def __init__(self, result=None):
        self.result = result or AgentResult(
            output="Mock output: task completed successfully.",
            exit_code=0,
            duration_s=0.1,
        )
        self.dispatched = []

    def dispatch(self, prompt, system_prompt, constraints,
                 pid_callback=None, event_callback=None, cwd=None):
        self.dispatched.append({"prompt": prompt, "cwd": cwd})
        return self.result


# ===========================================================================
# Integration tests: Executor respects merge policy
# ===========================================================================


class TestExecutorMergePolicy:
    """Test that the executor respects repo merge policies."""

    def test_auto_policy_merges_worktree(self, git_repo, mutation_log,
                                          work_state, audit_log, session_logger):
        """With auto policy, worktree changes are merged to main."""
        _write_repos_yaml(git_repo, """
repos:
  repo:
    merge_policy: auto
    protected_branches: [main]
""")

        class CommittingDispatcher(AgentDispatcher):
            def dispatch(self, prompt, system_prompt, constraints,
                         pid_callback=None, event_callback=None, cwd=None):
                if cwd:
                    new_file = Path(cwd) / "auto_merged.py"
                    new_file.write_text("# Auto merged\n")
                    subprocess.run(["git", "add", "auto_merged.py"],
                                   cwd=cwd, capture_output=True)
                    subprocess.run(["git", "commit", "-m", "Auto merge work"],
                                   cwd=cwd, capture_output=True)
                return AgentResult(output="Done", exit_code=0, duration_s=0.1)

        _create_task(mutation_log, "t1", "Auto merge task")
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

        with patch("corc.executor.get_repo_policy") as mock_policy:
            mock_policy.return_value = RepoPolicy(name="repo", merge_policy="auto")
            executor.dispatch(task)
            time.sleep(0.5)
            completed = executor.poll_completed()

        assert len(completed) == 1
        # File should be merged into main
        assert (git_repo / "auto_merged.py").exists()

        # Check audit log for worktree_merged event
        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "worktree_merged" in event_types
        assert "worktree_merge_skipped" not in event_types
        executor.shutdown()

    def test_human_only_policy_skips_merge(self, git_repo, mutation_log,
                                            work_state, audit_log, session_logger):
        """With human-only policy, worktree changes are NOT merged."""
        class CommittingDispatcher(AgentDispatcher):
            def dispatch(self, prompt, system_prompt, constraints,
                         pid_callback=None, event_callback=None, cwd=None):
                if cwd:
                    new_file = Path(cwd) / "human_only.py"
                    new_file.write_text("# Should not be merged\n")
                    subprocess.run(["git", "add", "human_only.py"],
                                   cwd=cwd, capture_output=True)
                    subprocess.run(["git", "commit", "-m", "Human only work"],
                                   cwd=cwd, capture_output=True)
                return AgentResult(output="Done", exit_code=0, duration_s=0.1)

        _create_task(mutation_log, "t1", "Human only task")
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

        with patch("corc.executor.get_repo_policy") as mock_policy:
            mock_policy.return_value = RepoPolicy(
                name="repo", merge_policy="human-only"
            )
            executor.dispatch(task)
            time.sleep(0.5)
            completed = executor.poll_completed()

        assert len(completed) == 1
        # File should NOT be in main repo (merge was skipped)
        assert not (git_repo / "human_only.py").exists()

        # Check audit log for worktree_merge_skipped event
        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "worktree_merge_skipped" in event_types
        assert "worktree_merged" not in event_types
        executor.shutdown()

    def test_human_only_policy_preserves_branch(self, git_repo, mutation_log,
                                                  work_state, audit_log, session_logger):
        """With human-only policy, the worktree branch is preserved (not deleted)."""
        class CommittingDispatcher(AgentDispatcher):
            def dispatch(self, prompt, system_prompt, constraints,
                         pid_callback=None, event_callback=None, cwd=None):
                if cwd:
                    new_file = Path(cwd) / "branch_preserved.py"
                    new_file.write_text("# Branch preserved\n")
                    subprocess.run(["git", "add", "branch_preserved.py"],
                                   cwd=cwd, capture_output=True)
                    subprocess.run(["git", "commit", "-m", "Preserve branch"],
                                   cwd=cwd, capture_output=True)
                return AgentResult(output="Done", exit_code=0, duration_s=0.1)

        _create_task(mutation_log, "t1", "Branch preserve task")
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

        with patch("corc.executor.get_repo_policy") as mock_policy:
            mock_policy.return_value = RepoPolicy(
                name="repo", merge_policy="human-only"
            )
            executor.dispatch(task)
            time.sleep(0.5)
            completed = executor.poll_completed()

        # Branch should still exist (for creating PR)
        result = subprocess.run(
            ["git", "branch", "--list", "corc/t1-1"],
            capture_output=True, text=True, cwd=str(git_repo),
        )
        assert "corc/t1-1" in result.stdout
        executor.shutdown()


# ===========================================================================
# Integration tests: Processor respects merge policy
# ===========================================================================


class TestProcessorMergePolicy:
    """Test that the processor respects repo merge policies."""

    def test_auto_policy_marks_completed(self, git_repo, mutation_log,
                                          work_state, audit_log, session_logger):
        """With auto policy, processor marks task as completed."""
        _create_task(mutation_log, "t1", "Auto task")
        mutation_log.append("task_started", {"attempt": 1},
                           reason="test", task_id="t1")
        work_state.refresh()
        task = work_state.get_task("t1")

        result = AgentResult(output="All done", exit_code=0, duration_s=1.0)

        with patch("corc.processor.get_repo_policy") as mock_policy:
            mock_policy.return_value = RepoPolicy(name="repo", merge_policy="auto")
            proc_result = process_completed(
                task=task,
                result=result,
                attempt=1,
                mutation_log=mutation_log,
                state=work_state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=git_repo,
            )

        assert proc_result.passed is True

        # Task should be marked as completed
        work_state.refresh()
        updated = work_state.get_task("t1")
        assert updated["status"] == "completed"

        # Audit log should have task_completed event
        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "task_completed" in event_types
        assert "task_pending_merge" not in event_types

    def test_human_only_policy_marks_pending_merge(self, git_repo, mutation_log,
                                                     work_state, audit_log,
                                                     session_logger):
        """With human-only policy, processor marks task as pending_merge."""
        _create_task(mutation_log, "t1", "Human only task")
        mutation_log.append("task_started", {"attempt": 1},
                           reason="test", task_id="t1")
        work_state.refresh()
        task = work_state.get_task("t1")

        result = AgentResult(output="All done", exit_code=0, duration_s=1.0)

        with patch("corc.processor.get_repo_policy") as mock_policy:
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
            )

        assert proc_result.passed is True

        # Audit log should have task_pending_merge event (not task_completed)
        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "task_pending_merge" in event_types

    def test_human_only_failed_task_not_affected(self, git_repo, mutation_log,
                                                   work_state, audit_log,
                                                   session_logger):
        """Failed tasks follow normal failure path regardless of merge policy."""
        _create_task(mutation_log, "t1", "Failing task")
        mutation_log.append("task_started", {"attempt": 1},
                           reason="test", task_id="t1")
        work_state.refresh()
        task = work_state.get_task("t1")

        result = AgentResult(output="Error occurred", exit_code=1, duration_s=1.0)

        with patch("corc.processor.get_repo_policy") as mock_policy:
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
            )

        assert proc_result.passed is False

        # Task should be marked as failed, not pending_merge
        work_state.refresh()
        updated = work_state.get_task("t1")
        assert updated["status"] == "failed"

    def test_auto_policy_extracts_findings(self, git_repo, mutation_log,
                                            work_state, audit_log, session_logger):
        """Findings are extracted regardless of merge policy."""
        _create_task(mutation_log, "t1", "Task with findings")
        mutation_log.append("task_started", {"attempt": 1},
                           reason="test", task_id="t1")
        work_state.refresh()
        task = work_state.get_task("t1")

        result = AgentResult(
            output="FINDING: Found a security issue\nFINDING: Missing test coverage",
            exit_code=0,
            duration_s=1.0,
        )

        with patch("corc.processor.get_repo_policy") as mock_policy:
            mock_policy.return_value = RepoPolicy(name="repo", merge_policy="auto")
            proc_result = process_completed(
                task=task,
                result=result,
                attempt=1,
                mutation_log=mutation_log,
                state=work_state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=git_repo,
            )

        assert len(proc_result.findings) == 2
        assert "Found a security issue" in proc_result.findings[0]


# ===========================================================================
# Integration tests: Full flow (executor + processor)
# ===========================================================================


class TestFullMergePolicyFlow:
    """Test the complete flow from dispatch through processing."""

    def test_auto_merge_full_flow(self, git_repo, mutation_log, work_state,
                                   audit_log, session_logger):
        """Full auto-merge flow: dispatch -> merge -> process -> completed."""
        _write_repos_yaml(git_repo, """
repos:
  repo:
    merge_policy: auto
    protected_branches: [main]
""")

        class WorkingDispatcher(AgentDispatcher):
            def dispatch(self, prompt, system_prompt, constraints,
                         pid_callback=None, event_callback=None, cwd=None):
                if cwd:
                    (Path(cwd) / "feature.py").write_text("# Feature\n")
                    subprocess.run(["git", "add", "feature.py"],
                                   cwd=cwd, capture_output=True)
                    subprocess.run(["git", "commit", "-m", "Add feature"],
                                   cwd=cwd, capture_output=True)
                return AgentResult(output="Done", exit_code=0, duration_s=0.1)

        _create_task(mutation_log, "t1", "Full auto task")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = Executor(
            dispatcher=WorkingDispatcher(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
        )

        with patch("corc.executor.get_repo_policy") as exec_mock, \
             patch("corc.processor.get_repo_policy") as proc_mock:
            exec_mock.return_value = RepoPolicy(name="repo", merge_policy="auto")
            proc_mock.return_value = RepoPolicy(name="repo", merge_policy="auto")

            executor.dispatch(task)
            time.sleep(0.5)
            completed = executor.poll_completed()

            assert len(completed) == 1
            # File should be merged
            assert (git_repo / "feature.py").exists()

            # Process the completed task
            proc_result = process_completed(
                task=completed[0].task,
                result=completed[0].result,
                attempt=completed[0].attempt,
                mutation_log=mutation_log,
                state=work_state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=git_repo,
            )

        assert proc_result.passed is True
        work_state.refresh()
        assert work_state.get_task("t1")["status"] == "completed"
        executor.shutdown()

    def test_human_only_full_flow(self, git_repo, mutation_log, work_state,
                                    audit_log, session_logger):
        """Full human-only flow: dispatch -> skip merge -> process -> pending_merge."""
        class WorkingDispatcher(AgentDispatcher):
            def dispatch(self, prompt, system_prompt, constraints,
                         pid_callback=None, event_callback=None, cwd=None):
                if cwd:
                    (Path(cwd) / "feature.py").write_text("# Feature\n")
                    subprocess.run(["git", "add", "feature.py"],
                                   cwd=cwd, capture_output=True)
                    subprocess.run(["git", "commit", "-m", "Add feature"],
                                   cwd=cwd, capture_output=True)
                return AgentResult(output="Done", exit_code=0, duration_s=0.1)

        _create_task(mutation_log, "t1", "Full human-only task")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = Executor(
            dispatcher=WorkingDispatcher(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=git_repo,
        )

        with patch("corc.executor.get_repo_policy") as exec_mock, \
             patch("corc.processor.get_repo_policy") as proc_mock:
            human_policy = RepoPolicy(name="repo", merge_policy="human-only")
            exec_mock.return_value = human_policy
            proc_mock.return_value = human_policy

            executor.dispatch(task)
            time.sleep(0.5)
            completed = executor.poll_completed()

            assert len(completed) == 1
            # File should NOT be in main
            assert not (git_repo / "feature.py").exists()
            # Branch should exist for PR
            result = subprocess.run(
                ["git", "branch", "--list", "corc/t1-1"],
                capture_output=True, text=True, cwd=str(git_repo),
            )
            assert "corc/t1-1" in result.stdout

            # Process the completed task
            proc_result = process_completed(
                task=completed[0].task,
                result=completed[0].result,
                attempt=completed[0].attempt,
                mutation_log=mutation_log,
                state=work_state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=git_repo,
            )

        assert proc_result.passed is True

        # Check audit events show the correct flow
        events = audit_log.read_for_task("t1")
        event_types = [e["event_type"] for e in events]
        assert "worktree_merge_skipped" in event_types
        assert "task_pending_merge" in event_types
        assert "worktree_merged" not in event_types
        assert "task_completed" not in event_types
        executor.shutdown()
