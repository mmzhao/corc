"""Git worktree lifecycle management for parallel agent isolation.

Each dispatched agent runs in its own git worktree, providing filesystem
isolation so multiple agents can edit files without conflicts.

Lifecycle: create_worktree() -> agent runs -> merge_worktree() -> remove_worktree()

IMPORTANT: Worktrees contain a full checkout of the repo, including
pyproject.toml and src/. To prevent agents from accidentally running
`pip install -e .` inside a worktree (which would redirect the shared
Python environment's editable install to the worktree's src/), we
neutralize the worktree's pyproject.toml after creation.
"""

import subprocess
import shutil
from pathlib import Path

# Files that enable `pip install -e .` — neutralized in worktrees to prevent
# accidental editable installs that would hijack the shared Python path.
_INSTALLABLE_FILES = ("pyproject.toml", "setup.py", "setup.cfg")

# Data files/directories that are auto-resolved during merge by keeping main's
# version (--ours).  These files are written exclusively by the daemon on main;
# agents never modify them.  This prevents the #1 source of merge conflicts
# (data/mutations.jsonl diverging because the daemon appends while agents run).
_DATA_PATHS_OURS = (
    "data/mutations.jsonl",
    "data/audit.jsonl",
    "data/sessions/",
)


class WorktreeError(Exception):
    """Raised when a git worktree operation fails."""

    pass


def create_worktree(
    project_root: Path, task_id: str, attempt: int = 1
) -> tuple[Path, str]:
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

    # Neutralize the worktree so agents can't accidentally `pip install -e .`
    # which would hijack the shared Python environment's editable install path.
    _neutralize_installable_files(worktree_path)

    return worktree_path, branch_name


def remove_worktree(
    project_root: Path, worktree_path: Path, remove_branch: bool = True
) -> bool:
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

    Uses a two-step merge to auto-resolve data file conflicts:
    1. ``git merge --no-ff --no-commit`` stages the merge without committing.
    2. ``git checkout --ours`` for data files (mutations.jsonl, audit.jsonl,
       sessions/) keeps main's version — agents never write these files.
    3. ``git commit`` finalises the merge.

    This eliminates the #1 source of merge conflicts: data/mutations.jsonl
    diverging because the daemon appends to it on main while agents run.

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

    # Step 1: Start the merge without committing
    result = subprocess.run(
        ["git", "merge", "--no-ff", "--no-commit", branch_name],
        capture_output=True,
        text=True,
        cwd=str(project_root),
        timeout=60,
    )

    # Even if there are conflicts (returncode != 0), we may be able to
    # auto-resolve them if they're only in data files.  Check below.

    # Step 2: For each data path, resolve to main's version (--ours).
    # If the path doesn't exist in the repo, git checkout --ours is a no-op
    # (it will fail silently, which is fine).
    for data_path in _DATA_PATHS_OURS:
        subprocess.run(
            ["git", "checkout", "--ours", "--", data_path],
            capture_output=True,
            cwd=str(project_root),
            timeout=10,
        )
        # Stage the resolved file so it's no longer marked as conflicted
        subprocess.run(
            ["git", "add", "--", data_path],
            capture_output=True,
            cwd=str(project_root),
            timeout=10,
        )

    # Check if there are still unresolved conflicts after auto-resolving data files
    status_result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        capture_output=True,
        text=True,
        cwd=str(project_root),
        timeout=30,
    )
    if status_result.stdout.strip():
        # Still have unresolved conflicts in non-data files — abort
        subprocess.run(
            ["git", "merge", "--abort"],
            capture_output=True,
            cwd=str(project_root),
            timeout=30,
        )
        return False

    # Step 3: Commit the merge
    commit_result = subprocess.run(
        ["git", "commit", "--no-edit", "-m", f"Merge {branch_name}"],
        capture_output=True,
        text=True,
        cwd=str(project_root),
        timeout=60,
    )

    if commit_result.returncode != 0:
        # Commit failed unexpectedly — abort
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
        [
            "git",
            "merge",
            main_branch,
            "-m",
            f"Merge {main_branch} for conflict resolution",
        ],
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
                current_path = line[len("worktree ") :]
            elif line.startswith("branch ") and current_path:
                if str(worktree_path) == current_path:
                    branch = line[len("branch ") :]
                    # Strip refs/heads/ prefix
                    if branch.startswith("refs/heads/"):
                        branch = branch[len("refs/heads/") :]
                    return branch
                current_path = None

        return None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def _neutralize_installable_files(worktree_path: Path):
    """Remove package install files from a worktree to prevent path hijacking.

    If an agent runs `pip install -e .` inside a worktree, it overwrites the
    shared Python environment's editable install (.pth file) to point at the
    worktree's src/ instead of the main project's src/. This causes all
    ``import corc`` calls — including from the main project and other worktrees —
    to resolve to the wrong source tree.

    By removing pyproject.toml, setup.py, and setup.cfg from the worktree,
    ``pip install -e .`` will fail harmlessly, preventing the hijack. The agent
    can still import corc via the existing editable install from the main project.
    """
    for filename in _INSTALLABLE_FILES:
        filepath = worktree_path / filename
        if filepath.exists():
            filepath.unlink()
