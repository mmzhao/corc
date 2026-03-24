"""Repo merge policy configuration and enforcement.

Loads per-repo merge policies from .corc/repos.yaml. Policies control
whether agents can auto-merge PRs or whether merges require human action.

Three merge policies:
- direct: Skip PRs entirely. Merge worktree directly to main. For repos where PRs aren't needed.
- auto: Create PR, auto-merge via gh pr merge after validation.
- human-only: Agent creates PR, but only humans can merge.

Additional per-repo settings:
- protected_branches: list of branch names that agents cannot push to directly
- block_auto_merge: prevent agents from enabling auto-merge on PRs
- block_direct_push: prevent agents from pushing directly to protected branches
- require_reviewer_approval: require reviewer approval before merge
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RepoPolicy:
    """Merge policy configuration for a single repository."""

    name: str
    merge_policy: str = "auto"  # "direct", "auto", or "human-only"
    protected_branches: list[str] = field(default_factory=lambda: ["main"])
    require_reviewer_approval: bool = True
    block_auto_merge: bool = False
    block_direct_push: bool = False

    def __post_init__(self):
        if self.merge_policy not in ("direct", "auto", "human-only"):
            raise ValueError(
                f"Invalid merge_policy '{self.merge_policy}': "
                f"must be 'direct', 'auto', or 'human-only'"
            )
        # human-only implies blocking auto-merge and direct push
        if self.merge_policy == "human-only":
            self.block_auto_merge = True
            self.block_direct_push = True

    @property
    def is_direct(self) -> bool:
        return self.merge_policy == "direct"

    @property
    def is_auto(self) -> bool:
        return self.merge_policy == "auto"

    @property
    def is_human_only(self) -> bool:
        return self.merge_policy == "human-only"


def load_repo_policies(project_root: Path) -> dict[str, RepoPolicy]:
    """Load repo policies from .corc/repos.yaml.

    Returns a dict mapping repo name -> RepoPolicy.
    If the file doesn't exist, returns an empty dict.
    """
    repos_yaml = Path(project_root) / ".corc" / "repos.yaml"
    if not repos_yaml.exists():
        return {}

    with open(repos_yaml) as f:
        data = yaml.safe_load(f)

    if not data or not isinstance(data, dict):
        return {}

    repos_data = data.get("repos", {})
    if not isinstance(repos_data, dict):
        return {}

    policies = {}
    for name, config in repos_data.items():
        if not isinstance(config, dict):
            continue
        policies[name] = RepoPolicy(
            name=name,
            merge_policy=config.get("merge_policy", "auto"),
            protected_branches=config.get("protected_branches", ["main"]),
            require_reviewer_approval=config.get("require_reviewer_approval", True),
            block_auto_merge=config.get("block_auto_merge", False),
            block_direct_push=config.get("block_direct_push", False),
        )

    return policies


def get_repo_name(project_root: Path) -> str:
    """Derive the repository name from git remote origin or directory name.

    Tries git remote origin URL first, falls back to directory name.
    """
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            url = result.stdout.strip()
            # Extract repo name from URL: git@github.com:org/repo.git -> repo
            # or https://github.com/org/repo.git -> repo
            name = url.rstrip("/").rsplit("/", 1)[-1]
            if name.endswith(".git"):
                name = name[:-4]
            return name
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass

    # Fallback to directory name
    return Path(project_root).name


def get_repo_policy(project_root: Path, repo_name: str | None = None) -> RepoPolicy:
    """Get the merge policy for the current repository.

    Looks up the repo name in .corc/repos.yaml. If no config exists for
    this repo, returns a default "auto" policy.

    Args:
        project_root: Path to the project root.
        repo_name: Optional repo name override. If None, derived from git.

    Returns:
        RepoPolicy for the repository.
    """
    if repo_name is None:
        repo_name = get_repo_name(project_root)

    policies = load_repo_policies(project_root)

    if repo_name in policies:
        return policies[repo_name]

    # Default: auto merge with main as protected branch
    return RepoPolicy(name=repo_name, merge_policy="auto")


def is_protected_branch(
    project_root: Path, branch: str, repo_name: str | None = None
) -> bool:
    """Check if a branch is protected for the current repo.

    Args:
        project_root: Path to the project root.
        branch: Branch name to check.
        repo_name: Optional repo name override.

    Returns:
        True if the branch is in the protected_branches list.
    """
    policy = get_repo_policy(project_root, repo_name)
    return branch in policy.protected_branches


def check_push_allowed(
    project_root: Path, command: str, repo_name: str | None = None
) -> tuple[bool, str]:
    """Check if a git push command is allowed by the repo policy.

    Parses the git push command to determine the target branch and
    checks against protected_branches + block_direct_push policy.

    Args:
        project_root: Path to the project root.
        command: The full git push command string.
        repo_name: Optional repo name override.

    Returns:
        Tuple of (allowed, reason). If allowed is False, reason explains why.
    """
    policy = get_repo_policy(project_root, repo_name)

    if not policy.block_direct_push:
        return True, ""

    # Parse the push target branch from the command
    target_branch = _parse_push_target(command, project_root)
    if target_branch and target_branch in policy.protected_branches:
        return False, (
            f"Direct push to protected branch '{target_branch}' is blocked "
            f"by repo policy '{policy.name}' (merge_policy={policy.merge_policy})"
        )

    return True, ""


def check_auto_merge_allowed(
    project_root: Path, command: str, repo_name: str | None = None
) -> tuple[bool, str]:
    """Check if a gh pr merge --auto command is allowed by the repo policy.

    Args:
        project_root: Path to the project root.
        command: The full gh pr merge command string.
        repo_name: Optional repo name override.

    Returns:
        Tuple of (allowed, reason). If allowed is False, reason explains why.
    """
    policy = get_repo_policy(project_root, repo_name)

    if not policy.block_auto_merge:
        return True, ""

    # Check if the command contains auto-merge flags
    if _is_auto_merge_command(command):
        return False, (
            f"Auto-merge is blocked by repo policy '{policy.name}' "
            f"(merge_policy={policy.merge_policy}, block_auto_merge=true)"
        )

    return True, ""


def _parse_push_target(command: str, project_root: Path) -> str | None:
    """Extract the target branch from a git push command.

    Handles common patterns:
    - git push origin main
    - git push origin HEAD:main
    - git push (uses current branch tracking)
    """
    parts = command.strip().split()

    if len(parts) < 2 or parts[0] != "git" or parts[1] != "push":
        return None

    # Filter out flags
    args = [p for p in parts[2:] if not p.startswith("-")]

    if len(args) >= 2:
        refspec = args[1]
        # Handle HEAD:branch or src:dst
        if ":" in refspec:
            return refspec.split(":")[-1]
        return refspec
    elif len(args) == 1:
        # Just remote name, pushing current branch
        return _get_current_branch(project_root)
    elif len(args) == 0:
        # Plain git push — pushes current branch to tracking remote
        return _get_current_branch(project_root)

    return None


def _get_current_branch(project_root: Path) -> str | None:
    """Get the current branch name."""
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
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    return None


def _is_auto_merge_command(command: str) -> bool:
    """Check if a command is a gh pr merge with --auto flag."""
    parts = command.strip().split()

    # Must start with 'gh pr merge'
    if len(parts) < 3:
        return False
    if parts[0] != "gh" or parts[1] != "pr" or parts[2] != "merge":
        return False

    # Check for --auto flag
    return "--auto" in parts
