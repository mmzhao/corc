"""Git worktree lifecycle management for parallel agent isolation.

Each dispatched agent runs in its own git worktree, providing filesystem
isolation so multiple agents can edit files without conflicts.

Lifecycle: create_worktree() -> agent runs -> merge_worktree() -> remove_worktree()
"""

import subprocess
import shutil
from pathlib import Path


class WorktreeError(Exception):
    """Raised when a git worktree operation fails."""
    pass


def create_worktree(project_root: Path, task_id: str, attempt: int = 1) -> tuple[Path, str]:
    """Create a git worktree for an agent to work in.

    Creates a new worktree branching from the current HEAD. The worktree
    is placed under .corc/worktrees/{task_id}-{attempt} and the branch
    is named corc/{task_id}-{attempt}.

    Args:
        project_root: The main repository root directory.
        task_id: Task identifier.
        attempt: Attempt number (for retries on fresh worktrees).

    Returns:
        Tuple of (worktree_path, branch_name).

    Raises:
        WorktreeError: If git worktree add fails.
    """
    project_root = Path(project_root)
    worktree_dir = project_root / ".corc" / "worktrees"
    worktree_dir.mkdir(parents=True, exist_ok=True)

    worktree_name = f"{task_id}-{attempt}"
    worktree_path = worktree_dir / worktree_name
    branch_name = f"corc/{worktree_name}"

    # Remove stale worktree if it exists (e.g. from a crash)
    if worktree_path.exists():
        _force_remove_worktree(project_root, worktree_path)

    # Create the worktree with a new branch from HEAD
    result = subprocess.run(
        ["git", "worktree", "add", "-b", branch_name, str(worktree_path), "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(project_root),
        timeout=30,
    )

    if result.returncode != 0:
        # Branch might already exist from a previous attempt — try without -b
        result2 = subprocess.run(
            ["git", "worktree", "add", str(worktree_path), branch_name],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=30,
        )
        if result2.returncode != 0:
            raise WorktreeError(
                f"Failed to create worktree: {result.stderr.strip()} / {result2.stderr.strip()}"
            )

    return worktree_path, branch_name


def remove_worktree(project_root: Path, worktree_path: Path, remove_branch: bool = True) -> bool:
    """Remove a git worktree and optionally its branch.

    Args:
        project_root: The main repository root directory.
        worktree_path: Path to the worktree to remove.
        remove_branch: If True, also delete the worktree's branch.

    Returns:
        True if the worktree was removed, False if it didn't exist.
    """
    project_root = Path(project_root)
    worktree_path = Path(worktree_path)

    if not worktree_path.exists():
        # Still try to prune in case git has stale references
        subprocess.run(
            ["git", "worktree", "prune"],
            capture_output=True,
            cwd=str(project_root),
            timeout=30,
        )
        return False

    # Determine branch name before removing
    branch_name = _get_worktree_branch(project_root, worktree_path)

    _force_remove_worktree(project_root, worktree_path)

    # Remove the branch if requested
    if remove_branch and branch_name:
        subprocess.run(
            ["git", "branch", "-D", branch_name],
            capture_output=True,
            cwd=str(project_root),
            timeout=30,
        )

    return True


def merge_worktree(project_root: Path, worktree_path: Path) -> bool:
    """Merge changes from a worktree branch back into the main branch.

    Performs a simple merge of the worktree's branch into the current
    HEAD of the main repo. Uses --no-ff to preserve branch history.

    Args:
        project_root: The main repository root directory.
        worktree_path: Path to the worktree whose branch to merge.

    Returns:
        True if merge succeeded, False if there was a conflict.

    Raises:
        WorktreeError: If the merge cannot be attempted.
    """
    project_root = Path(project_root)
    worktree_path = Path(worktree_path)

    branch_name = _get_worktree_branch(project_root, worktree_path)
    if not branch_name:
        raise WorktreeError(f"Cannot determine branch for worktree: {worktree_path}")

    # Check if there are any commits on the branch that differ from main
    diff_result = subprocess.run(
        ["git", "log", "HEAD.." + branch_name, "--oneline"],
        capture_output=True,
        text=True,
        cwd=str(project_root),
        timeout=30,
    )
    if not diff_result.stdout.strip():
        # No new commits — nothing to merge
        return True

    # Attempt the merge
    result = subprocess.run(
        ["git", "merge", "--no-ff", branch_name, "-m", f"Merge {branch_name}"],
        capture_output=True,
        text=True,
        cwd=str(project_root),
        timeout=60,
    )

    if result.returncode != 0:
        # Merge conflict — abort the merge
        subprocess.run(
            ["git", "merge", "--abort"],
            capture_output=True,
            cwd=str(project_root),
            timeout=30,
        )
        return False

    return True


def merge_main_into_worktree(project_root: Path, worktree_path: Path) -> bool:
    """Merge current main HEAD into a worktree branch.

    Used after an optimistic merge to main fails due to conflicts.
    Incorporates the latest main branch changes into the worktree so
    the agent can be retried with the merged state as baseline.

    The worktree will contain both the agent's changes and the latest
    main branch state, allowing the next agent to resolve any conflicts.

    Args:
        project_root: The main repository root directory.
        worktree_path: Path to the worktree to merge into.

    Returns:
        True if merge succeeded, False if conflicts prevent merge.

    Raises:
        WorktreeError: If the branch cannot be determined.
    """
    project_root = Path(project_root)
    worktree_path = Path(worktree_path)

    # Get the current branch in the main repo (main/master/etc)
    main_branch = _get_current_branch(project_root)
    if not main_branch:
        raise WorktreeError("Cannot determine main branch for merge into worktree")

    # Merge main into the worktree branch (run in worktree directory)
    result = subprocess.run(
        ["git", "merge", main_branch, "-m", f"Merge {main_branch} for conflict resolution"],
        capture_output=True,
        text=True,
        cwd=str(worktree_path),
        timeout=60,
    )

    if result.returncode != 0:
        # Conflict — abort the merge
        subprocess.run(
            ["git", "merge", "--abort"],
            capture_output=True,
            cwd=str(worktree_path),
            timeout=30,
        )
        return False

    return True


def _get_current_branch(project_root: Path) -> str | None:
    """Get the name of the current branch in the main repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _force_remove_worktree(project_root: Path, worktree_path: Path):
    """Remove a worktree, falling back to manual removal if git fails."""
    try:
        result = subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree_path)],
            capture_output=True,
            cwd=str(project_root),
            timeout=30,
        )
        if result.returncode != 0:
            _manual_remove(worktree_path, project_root)
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        _manual_remove(worktree_path, project_root)


def _manual_remove(worktree_path: Path, project_root: Path):
    """Manual fallback: remove directory and prune worktree list."""
    try:
        shutil.rmtree(str(worktree_path), ignore_errors=True)
    except Exception:
        pass
    try:
        subprocess.run(
            ["git", "worktree", "prune"],
            capture_output=True,
            cwd=str(project_root),
            timeout=30,
        )
    except Exception:
        pass


def _get_worktree_branch(project_root: Path, worktree_path: Path) -> str | None:
    """Get the branch name checked out in a worktree."""
    try:
        result = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=30,
        )
        if result.returncode != 0:
            return None

        # Parse porcelain output to find our worktree's branch
        current_path = None
        for line in result.stdout.splitlines():
            if line.startswith("worktree "):
                current_path = line[len("worktree "):]
            elif line.startswith("branch ") and current_path:
                if str(worktree_path) == current_path:
                    branch = line[len("branch "):]
                    # Strip refs/heads/ prefix
                    if branch.startswith("refs/heads/"):
                        branch = branch[len("refs/heads/"):]
                    return branch
                current_path = None

        return None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
