"""Multi-repo registration with per-repo merge policies.

Repos are stored under the 'repos' key in .corc/config.yaml:

    repos:
      internal-tool:
        path: /path/to/internal-tool
        merge_policy: auto
        protected_branches: [main]
        enforcement_level: strict
      production-app:
        path: /path/to/production-app
        merge_policy: human-only
        protected_branches: [main, staging]
        enforcement_level: strict

Usage:
    from corc.config import load_config
    from corc.repo import RepoManager

    cfg = load_config()
    mgr = RepoManager(cfg)
    mgr.add("my-repo", "/path/to/repo", merge_policy="auto")
    mgr.list_repos()
    mgr.get("my-repo")
"""

from __future__ import annotations

from typing import Any

from corc.config import CorcConfig


VALID_MERGE_POLICIES = ("auto", "human-only")
VALID_ENFORCEMENT_LEVELS = ("strict", "relaxed")

DEFAULT_MERGE_POLICY = "auto"
DEFAULT_PROTECTED_BRANCHES: list[str] = ["main"]
DEFAULT_ENFORCEMENT_LEVEL = "strict"


class RepoError(Exception):
    """Base exception for repo operations."""


class RepoNotFoundError(RepoError):
    """Raised when a repo name is not found."""


class RepoAlreadyExistsError(RepoError):
    """Raised when trying to add a repo that already exists."""


class RepoValidationError(RepoError):
    """Raised when repo settings are invalid."""


def _validate_merge_policy(value: str) -> None:
    """Validate merge_policy is one of the allowed values."""
    if value not in VALID_MERGE_POLICIES:
        raise RepoValidationError(
            f"Invalid merge_policy '{value}'. Must be one of: {', '.join(VALID_MERGE_POLICIES)}"
        )


def _validate_enforcement_level(value: str) -> None:
    """Validate enforcement_level is one of the allowed values."""
    if value not in VALID_ENFORCEMENT_LEVELS:
        raise RepoValidationError(
            f"Invalid enforcement_level '{value}'. Must be one of: {', '.join(VALID_ENFORCEMENT_LEVELS)}"
        )


def _validate_protected_branches(value: list) -> None:
    """Validate protected_branches is a list of non-empty strings."""
    if not isinstance(value, list):
        raise RepoValidationError("protected_branches must be a list")
    for branch in value:
        if not isinstance(branch, str) or not branch.strip():
            raise RepoValidationError(
                f"Each protected branch must be a non-empty string, got: {branch!r}"
            )


class RepoManager:
    """Manages multi-repo registration with per-repo settings.

    Repos are stored in the centralized CorcConfig under the 'repos' key.
    Each repo has a name (used as the dict key) and settings including
    path, merge_policy, protected_branches, and enforcement_level.
    """

    def __init__(self, config: CorcConfig):
        self._config = config

    def _get_repos(self) -> dict[str, Any]:
        """Return the repos dict from config (or empty dict)."""
        repos = self._config.get("repos")
        if repos is None or not isinstance(repos, dict):
            return {}
        return repos

    def _set_repos(self, repos: dict[str, Any]) -> None:
        """Write the repos dict back to config."""
        self._config.set("repos", repos)

    def add(
        self,
        name: str,
        path: str,
        merge_policy: str = DEFAULT_MERGE_POLICY,
        protected_branches: list[str] | None = None,
        enforcement_level: str = DEFAULT_ENFORCEMENT_LEVEL,
    ) -> dict[str, Any]:
        """Register a new repo.

        Args:
            name: Unique repo identifier.
            path: Filesystem path to the repo.
            merge_policy: 'auto' or 'human-only'.
            protected_branches: List of protected branch names.
            enforcement_level: 'strict' or 'relaxed'.

        Returns:
            The repo config dict that was stored.

        Raises:
            RepoAlreadyExistsError: If a repo with this name already exists.
            RepoValidationError: If settings are invalid.
        """
        if not name or not name.strip():
            raise RepoValidationError("Repo name must be a non-empty string")
        if not path or not path.strip():
            raise RepoValidationError("Repo path must be a non-empty string")

        _validate_merge_policy(merge_policy)
        _validate_enforcement_level(enforcement_level)

        if protected_branches is None:
            protected_branches = list(DEFAULT_PROTECTED_BRANCHES)
        _validate_protected_branches(protected_branches)

        repos = self._get_repos()
        if name in repos:
            raise RepoAlreadyExistsError(f"Repo '{name}' already exists")

        repo_config: dict[str, Any] = {
            "path": path,
            "merge_policy": merge_policy,
            "protected_branches": protected_branches,
            "enforcement_level": enforcement_level,
        }
        repos[name] = repo_config
        self._set_repos(repos)
        return repo_config

    def get(self, name: str) -> dict[str, Any]:
        """Get a repo's full configuration.

        Args:
            name: Repo identifier.

        Returns:
            Dict with path, merge_policy, protected_branches, enforcement_level.

        Raises:
            RepoNotFoundError: If repo not found.
        """
        repos = self._get_repos()
        if name not in repos:
            raise RepoNotFoundError(f"Repo '{name}' not found")
        return dict(repos[name])

    def list_repos(self) -> list[dict[str, Any]]:
        """List all registered repos with their settings.

        Returns:
            List of dicts, each with 'name' plus all repo settings.
        """
        repos = self._get_repos()
        result = []
        for name, settings in repos.items():
            entry = {"name": name}
            entry.update(settings)
            result.append(entry)
        return result

    def remove(self, name: str) -> None:
        """Remove a repo registration.

        Args:
            name: Repo identifier.

        Raises:
            RepoNotFoundError: If repo not found.
        """
        repos = self._get_repos()
        if name not in repos:
            raise RepoNotFoundError(f"Repo '{name}' not found")
        del repos[name]
        self._set_repos(repos)

    def update(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Update settings for an existing repo.

        Args:
            name: Repo identifier.
            **kwargs: Settings to update (path, merge_policy,
                      protected_branches, enforcement_level).

        Returns:
            The updated repo config dict.

        Raises:
            RepoNotFoundError: If repo not found.
            RepoValidationError: If new settings are invalid.
        """
        repos = self._get_repos()
        if name not in repos:
            raise RepoNotFoundError(f"Repo '{name}' not found")

        repo = repos[name]

        if "merge_policy" in kwargs:
            _validate_merge_policy(kwargs["merge_policy"])
            repo["merge_policy"] = kwargs["merge_policy"]

        if "enforcement_level" in kwargs:
            _validate_enforcement_level(kwargs["enforcement_level"])
            repo["enforcement_level"] = kwargs["enforcement_level"]

        if "protected_branches" in kwargs:
            _validate_protected_branches(kwargs["protected_branches"])
            repo["protected_branches"] = kwargs["protected_branches"]

        if "path" in kwargs:
            if not kwargs["path"] or not kwargs["path"].strip():
                raise RepoValidationError("Repo path must be a non-empty string")
            repo["path"] = kwargs["path"]

        repos[name] = repo
        self._set_repos(repos)
        return dict(repo)
