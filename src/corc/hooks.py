"""PreToolUse hooks for enforcing repo merge policies.

These hooks intercept tool use events from Claude Code and block
operations that violate repo merge policies:

- block_direct_push: Blocks `git push` to protected branches
- block_auto_merge: Blocks `gh pr merge --auto` commands

Hooks return (allowed, reason) tuples. If allowed is False, the tool
use should be blocked and reason displayed to the agent.
"""

from pathlib import Path

from corc.repo_policy import (
    check_auto_merge_allowed,
    check_push_allowed,
    get_repo_policy,
)


def pre_tool_use_hook(
    tool_name: str,
    tool_input: dict,
    project_root: Path,
    repo_name: str | None = None,
) -> tuple[bool, str]:
    """Main PreToolUse hook entry point.

    Checks tool use events against repo merge policies and returns
    whether the operation should be allowed.

    Args:
        tool_name: Name of the tool being used (e.g., "Bash").
        tool_input: Tool input parameters (e.g., {"command": "git push origin main"}).
        project_root: Path to the project root.
        repo_name: Optional repo name override.

    Returns:
        Tuple of (allowed, reason). If allowed is False, the operation
        should be blocked and reason explains why.
    """
    if tool_name == "Bash":
        command = tool_input.get("command", "")
        return check_bash_command(command, project_root, repo_name)

    return True, ""


def check_bash_command(
    command: str,
    project_root: Path,
    repo_name: str | None = None,
) -> tuple[bool, str]:
    """Check a Bash command against repo merge policies.

    Inspects the command (including chained commands with &&, ||, ;, |)
    for git push and gh pr merge patterns, and enforces block_direct_push
    and block_auto_merge policies.

    Args:
        command: The bash command string.
        project_root: Path to the project root.
        repo_name: Optional repo name override.

    Returns:
        Tuple of (allowed, reason).
    """
    # Check each segment of chained commands individually
    segments = [command.strip()] + _split_command_chain(command)

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        # Check for git push to protected branches
        parts = segment.split()
        if len(parts) >= 2 and parts[0] == "git" and parts[1] == "push":
            allowed, reason = check_push_allowed(project_root, segment, repo_name)
            if not allowed:
                return False, reason

        # Check for gh pr merge --auto
        if (len(parts) >= 3
                and parts[0] == "gh"
                and parts[1] == "pr"
                and parts[2] == "merge"
                and "--auto" in parts):
            allowed, reason = check_auto_merge_allowed(
                project_root, segment, repo_name
            )
            if not allowed:
                return False, reason

    return True, ""


def _looks_like_git_push(command: str) -> bool:
    """Quick check if a command looks like a git push."""
    parts = command.split()
    if len(parts) >= 2 and parts[0] == "git" and parts[1] == "push":
        return True
    # Also catch piped or chained commands containing git push
    for segment in _split_command_chain(command):
        segment_parts = segment.strip().split()
        if (len(segment_parts) >= 2
                and segment_parts[0] == "git"
                and segment_parts[1] == "push"):
            return True
    return False


def _looks_like_auto_merge(command: str) -> bool:
    """Quick check if a command looks like gh pr merge with --auto."""
    parts = command.split()
    if (len(parts) >= 3
            and parts[0] == "gh"
            and parts[1] == "pr"
            and parts[2] == "merge"
            and "--auto" in parts):
        return True
    # Also catch piped or chained commands
    for segment in _split_command_chain(command):
        segment_parts = segment.strip().split()
        if (len(segment_parts) >= 3
                and segment_parts[0] == "gh"
                and segment_parts[1] == "pr"
                and segment_parts[2] == "merge"
                and "--auto" in segment_parts):
            return True
    return False


def _split_command_chain(command: str) -> list[str]:
    """Split a command by shell operators (&&, ||, ;, |).

    Simple split — doesn't handle quoting. Good enough for detecting
    git push and gh pr merge patterns.
    """
    segments = []
    for part in command.replace("&&", "\n").replace("||", "\n").replace(";", "\n").split("\n"):
        # Also split on pipe, but keep the full command
        for sub in part.split("|"):
            sub = sub.strip()
            if sub:
                segments.append(sub)
    return segments
