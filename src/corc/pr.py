"""PR-based workflow for task completion.

Every task produces a PR. The processor reviews and comments before merge
decisions. Auto-merge repos get merged after review comment; human-only
repos leave the PR open and notify the operator.

Key operations:
- pull_main(): git pull on main before creating worktrees
- push_branch(): push worktree branch to remote
- create_pr(): create a PR from a worktree branch via gh pr create
- post_review_comment(): post validation summary as PR comment
- merge_pr(): merge PR via gh pr merge (auto-merge repos only)
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _get_repo_token(project_root: Path) -> str | None:
    """Load a per-repo GH_TOKEN from .corc/secrets.yaml.

    Returns the token string if found for this repo, None otherwise.
    """
    secrets_path = project_root / ".corc" / "secrets.yaml"
    if not secrets_path.exists():
        return None
    try:
        with open(secrets_path) as f:
            data = yaml.safe_load(f) or {}
        tokens = data.get("repo_tokens", {})
        repo_name = project_root.name
        return tokens.get(repo_name)
    except (yaml.YAMLError, OSError):
        return None


def _gh_env(project_root: Path) -> dict[str, str]:
    """Build environment for gh commands, injecting per-repo token if available."""
    env = os.environ.copy()
    token = _get_repo_token(project_root)
    if token:
        env["GH_TOKEN"] = token
    return env


class PRError(Exception):
    """Raised when a PR operation fails."""

    pass


@dataclass
class PRInfo:
    """Information about a created PR."""

    url: str
    number: int
    branch: str
    title: str


def pull_main(project_root: Path, timeout: int = 60) -> bool:
    """Pull latest changes on main before creating worktrees.

    Ensures worktrees branch from the latest main. If there's no remote
    configured or pull fails (e.g., offline), returns False but does not
    raise — the workflow continues with the local state.

    Args:
        project_root: The main repository root directory.
        timeout: Timeout for the git pull command in seconds.

    Returns:
        True if pull succeeded, False otherwise.
    """
    # Check if there's a remote configured
    try:
        result = subprocess.run(
            ["git", "remote"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            logger.debug("No git remote configured, skipping pull")
            return False
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return False

    # Fetch and pull
    try:
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=timeout,
        )
        if result.returncode == 0:
            logger.info("Pulled latest main: %s", result.stdout.strip())
            return True
        else:
            logger.warning("git pull failed: %s", result.stderr.strip())
            return False
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        logger.warning("git pull error: %s", e)
        return False


def push_branch(
    project_root: Path, branch_name: str, timeout: int = 60
) -> tuple[bool, str]:
    """Push a worktree branch to the remote.

    Args:
        project_root: The main repository root directory.
        branch_name: The branch name to push.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (success, error_message). error_message is empty on success.
    """
    try:
        result = subprocess.run(
            ["git", "push", "-u", "origin", branch_name],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=timeout,
        )
        if result.returncode == 0:
            logger.info("Pushed branch %s to origin", branch_name)
            return (True, "")
        else:
            error_msg = result.stderr.strip()
            logger.warning("git push failed for %s: %s", branch_name, error_msg)
            return (False, error_msg)
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        error_msg = str(e)
        logger.warning("git push error for %s: %s", branch_name, error_msg)
        return (False, error_msg)


def create_pr(
    project_root: Path,
    branch_name: str,
    task: dict,
    base_branch: str = "main",
    timeout: int = 30,
) -> tuple[PRInfo | None, str]:
    """Create a PR from a worktree branch via gh pr create.

    Args:
        project_root: The main repository root directory.
        branch_name: The branch name for the PR head.
        task: Task dict with id, name, done_when fields.
        base_branch: The base branch to merge into.
        timeout: Timeout in seconds.

    Returns:
        Tuple of (PRInfo, error_message). PRInfo is None on failure;
        error_message is empty on success.
    """
    task_id = task.get("id", "unknown")
    task_name = task.get("name", "unknown")

    title = f"[corc] {task_name} ({task_id})"
    body = (
        f"## Task: {task_name}\n\n"
        f"**Task ID:** {task_id}\n"
        f"**Done when:** {task.get('done_when', 'N/A')}\n\n"
        f"---\n"
        f"*Automated PR created by CORC orchestrator.*"
    )

    try:
        result = subprocess.run(
            [
                "gh",
                "pr",
                "create",
                "--head",
                branch_name,
                "--base",
                base_branch,
                "--title",
                title,
                "--body",
                body,
            ],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=timeout,
            env=_gh_env(project_root),
        )
        if result.returncode != 0:
            error_msg = result.stderr.strip()
            logger.warning("gh pr create failed: %s", error_msg)
            return (None, error_msg)

        pr_url = result.stdout.strip()
        pr_number = _extract_pr_number(pr_url)

        logger.info("Created PR #%s: %s", pr_number, pr_url)
        return (
            PRInfo(
                url=pr_url,
                number=pr_number,
                branch=branch_name,
                title=title,
            ),
            "",
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        error_msg = str(e)
        logger.warning("gh pr create error: %s", error_msg)
        return (None, error_msg)


def post_review_comment(
    project_root: Path,
    pr_number: int,
    passed: bool,
    details: list[tuple[bool, str]],
    findings: list[str] | None = None,
    timeout: int = 30,
) -> bool:
    """Post a validation summary as a PR comment via gh pr comment.

    Args:
        project_root: The main repository root directory.
        pr_number: The PR number to comment on.
        passed: Whether all validations passed.
        details: List of (passed, description) tuples from validation.
        findings: Optional list of findings from the agent.
        timeout: Timeout in seconds.

    Returns:
        True if comment was posted successfully, False otherwise.
    """
    comment_body = _format_review_comment(passed, details, findings)

    try:
        result = subprocess.run(
            [
                "gh",
                "pr",
                "comment",
                str(pr_number),
                "--body",
                comment_body,
            ],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=timeout,
            env=_gh_env(project_root),
        )
        if result.returncode == 0:
            logger.info("Posted review comment on PR #%s", pr_number)
            return True
        else:
            logger.warning("gh pr comment failed: %s", result.stderr.strip())
            return False
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        logger.warning("gh pr comment error: %s", e)
        return False


def merge_pr(project_root: Path, pr_number: int, timeout: int = 60) -> bool:
    """Merge a PR via gh pr merge (for auto-merge repos only).

    Uses --merge (not --squash or --rebase) to match the existing --no-ff
    merge strategy used by worktree merges. Deletes the branch after merge.

    Args:
        project_root: The main repository root directory.
        pr_number: The PR number to merge.
        timeout: Timeout in seconds.

    Returns:
        True if merge succeeded, False otherwise.
    """
    try:
        result = subprocess.run(
            [
                "gh",
                "pr",
                "merge",
                str(pr_number),
                "--merge",
                "--delete-branch",
            ],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=timeout,
            env=_gh_env(project_root),
        )
        if result.returncode == 0:
            logger.info("Merged PR #%s", pr_number)
            return True

        # Non-zero exit — gh pr merge can fail even when the merge itself
        # succeeded (e.g. --delete-branch fails, status check race).
        # Verify the actual merge state before declaring failure.
        logger.warning(
            "gh pr merge returned non-zero for PR #%s: %s",
            pr_number,
            result.stderr.strip(),
        )
        if _check_pr_merged(project_root, pr_number, timeout):
            logger.warning(
                "PR #%s is actually MERGED despite non-zero exit from gh pr merge",
                pr_number,
            )
            return True

        return False
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        logger.warning("gh pr merge error: %s", e)
        return False


def _check_pr_merged(project_root: Path, pr_number: int, timeout: int = 30) -> bool:
    """Check whether a PR is in MERGED state via gh pr view.

    Used after a non-zero exit from gh pr merge to verify whether the
    merge actually succeeded despite the error (e.g. --delete-branch
    failed, status check race condition).

    Args:
        project_root: The main repository root directory.
        pr_number: The PR number to check.
        timeout: Timeout in seconds.

    Returns:
        True if the PR state is MERGED, False otherwise.
    """
    try:
        result = subprocess.run(
            [
                "gh",
                "pr",
                "view",
                str(pr_number),
                "--json",
                "state",
            ],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=timeout,
            env=_gh_env(project_root),
        )
        if result.returncode != 0:
            logger.warning(
                "gh pr view failed for PR #%s: %s",
                pr_number,
                result.stderr.strip(),
            )
            return False

        data = json.loads(result.stdout)
        state = data.get("state", "")
        return state == "MERGED"
    except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
        logger.warning("gh pr view error for PR #%s: %s", pr_number, e)
        return False
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("Failed to parse gh pr view output for PR #%s: %s", pr_number, e)
        return False


def get_worktree_branch(project_root: Path, worktree_path: Path) -> str | None:
    """Get the branch name for a worktree.

    Delegates to the worktree module's internal helper.

    Args:
        project_root: The main repository root directory.
        worktree_path: Path to the worktree.

    Returns:
        Branch name or None if not found.
    """
    from corc.worktree import _get_worktree_branch

    return _get_worktree_branch(project_root, worktree_path)


def _format_review_comment(
    passed: bool,
    details: list[tuple[bool, str]],
    findings: list[str] | None = None,
) -> str:
    """Format a validation summary as a markdown comment.

    Args:
        passed: Whether all validations passed.
        details: List of (passed, description) tuples.
        findings: Optional list of findings.

    Returns:
        Formatted markdown string.
    """
    status = "✅ **Validation Passed**" if passed else "❌ **Validation Failed**"
    lines = [f"## CORC Validation Summary\n", status, ""]

    if details:
        lines.append("### Checks")
        for check_passed, description in details:
            icon = "✅" if check_passed else "❌"
            lines.append(f"- {icon} {description}")
        lines.append("")

    if findings:
        lines.append("### Findings")
        for finding in findings:
            lines.append(f"- {finding}")
        lines.append("")

    lines.append("---")
    lines.append("*Automated review by CORC processor.*")

    return "\n".join(lines)


def _extract_pr_number(pr_url: str) -> int:
    """Extract PR number from a GitHub PR URL.

    Expected format: https://github.com/owner/repo/pull/123

    Args:
        pr_url: The PR URL returned by gh pr create.

    Returns:
        PR number as integer, or 0 if extraction fails.
    """
    try:
        # URL like https://github.com/owner/repo/pull/123
        parts = pr_url.rstrip("/").split("/")
        return int(parts[-1])
    except (ValueError, IndexError):
        return 0
