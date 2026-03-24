"""Centralized configuration — single source of truth for all CORC settings.

All settings live in .corc/config.yaml. Every module reads from this config
instead of using hardcoded values. Defaults are provided for everything so the
system works with no config file at all.

Usage:
    from corc.config import load_config, get_paths

    cfg = load_config()              # loads from .corc/config.yaml (or defaults)
    cfg = load_config(root)          # loads from <root>/.corc/config.yaml
    cfg.get("dispatch.agent_timeout_s")   # => 1800
    cfg.get("retry.default_retries")      # => 2
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Project root / paths (unchanged from before)
# ---------------------------------------------------------------------------


def get_project_root() -> Path:
    """Find the project root by looking for .git or pyproject.toml."""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return cwd


def get_paths(root: Path | None = None) -> dict:
    root = root or get_project_root()
    return {
        "root": root,
        "data_dir": root / "data",
        "mutations": root / "data" / "mutations.jsonl",
        "state_db": root / "data" / "state.db",
        "events_dir": root / "data" / "events",
        "sessions_dir": root / "data" / "sessions",
        "knowledge_dir": root / "knowledge",
        "knowledge_db": root / "data" / "knowledge.db",
        "corc_dir": root / ".corc",
        "ratings_dir": root / "data" / "ratings",
        "retry_outcomes": root / "data" / "retry_outcomes.jsonl",
        "planning_feedback": root / "data" / "planning_feedback.jsonl",
    }


# ---------------------------------------------------------------------------
# Default configuration — every setting with a sensible default
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    # --- Dispatch / agent constraints ---
    "dispatch": {
        "provider": "claude-code",
        "agent_timeout_s": 1800,
        "default_allowed_tools": ["Read", "Edit", "Write", "Bash", "Grep", "Glob"],
        "max_budget_usd": 3.0,
        "max_turns": 50,
    },
    # --- Daemon ---
    "daemon": {
        "poll_interval": 5.0,
        "parallel": 1,
    },
    # --- Retry / adaptive retry ---
    "retry": {
        "default_retries": 2,
        "reduced_retries": 1,
        "increased_retries": 3,
        "min_samples": 5,
        "high_success_threshold": 0.90,
        "low_success_threshold": 0.50,
    },
    # --- Cost alerts ---
    "alerts": {
        "cost": {
            "enabled": True,
            "daily_limit_usd": 50.0,
            "project_limit_usd": 200.0,
            "task_limit_usd": 10.0,
        },
    },
    # --- Audit log backup ---
    "audit": {
        "backup_path": "~/.corc-backups/audit/",
        "backup_interval": "daily",
        "rotate_after_days": 90,
    },
    # --- Notifications ---
    "notifications": {
        "channels": {
            "terminal": {"enabled": True},
            "slack": {"enabled": False, "webhook_url": None},
            "discord": {"enabled": False, "webhook_url": None},
            "telegram": {"enabled": False, "bot_token": None, "chat_id": None},
        },
        "triggers": {
            "escalation": ["terminal"],
            "task_complete": [],
            "task_failure": ["terminal"],
            "cost_threshold": ["terminal"],
            "pause": ["terminal"],
            "daily_summary": ["terminal"],
        },
    },
    # --- Pattern analysis ---
    "patterns": {
        "low_score_threshold": 5.0,
        "high_score_threshold": 9.0,
        "flag_threshold": 7.0,
        "min_sample_size": 3,
        "trust_min_sample": 20,
    },
    # --- Curation ---
    "curation": {
        "blacklist_threshold": 3,
    },
    # --- Knowledge ---
    "knowledge": {
        "target_tokens": 500,
    },
    # --- Context assembly ---
    "context": {
        "max_tokens_warn": 25000,
    },
    # --- Webhook timeouts ---
    "webhooks": {
        "timeout": 10.0,
    },
}


# ---------------------------------------------------------------------------
# CorcConfig class
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict.

    Values from override take precedence. Nested dicts are merged recursively.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class CorcConfig:
    """Centralized configuration loaded from .corc/config.yaml.

    Provides dot-notation access to nested keys via get():
        cfg.get("dispatch.agent_timeout_s")  => 1800
        cfg.get("retry.default_retries")     => 2

    Falls back to DEFAULTS for any missing key, so the system works
    with no config file at all.
    """

    def __init__(
        self, data: dict[str, Any] | None = None, config_path: Path | None = None
    ):
        """Create a CorcConfig from merged defaults + overrides.

        Args:
            data: Override dict (merged on top of DEFAULTS).
            config_path: Path to the config.yaml file (for saving).
        """
        self._config_path = config_path
        if data is None:
            self._data = copy.deepcopy(DEFAULTS)
        else:
            self._data = _deep_merge(DEFAULTS, data)

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Get a config value using dot-separated keys.

        Examples:
            cfg.get("dispatch.agent_timeout_s")
            cfg.get("retry.default_retries")
            cfg.get("nonexistent.key", "fallback")
        """
        keys = dotted_key.split(".")
        node: Any = self._data
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node

    def set(self, dotted_key: str, value: Any) -> None:
        """Set a config value using dot-separated keys.

        Creates intermediate dicts as needed.
        """
        keys = dotted_key.split(".")
        node = self._data
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = value

    def as_dict(self) -> dict[str, Any]:
        """Return the full configuration as a plain dict."""
        return copy.deepcopy(self._data)

    @property
    def config_path(self) -> Path | None:
        """Return the path to the config file (if set)."""
        return self._config_path

    def save(self, path: Path | None = None) -> Path:
        """Write the current configuration to a YAML file.

        Args:
            path: Override path. Defaults to the path used to load.

        Returns:
            The path the config was written to.

        Raises:
            ValueError: If no path is available.
        """
        target = path or self._config_path
        if target is None:
            raise ValueError(
                "No config path specified — pass a path or load from a file first."
            )
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)

        # Only write user overrides (diff from defaults) for cleanliness
        user_data = _diff_from_defaults(self._data, DEFAULTS)
        with open(target, "w") as f:
            yaml.safe_dump(user_data, f, default_flow_style=False, sort_keys=False)
        return target


def _diff_from_defaults(data: dict, defaults: dict) -> dict:
    """Return only the keys/values in data that differ from defaults.

    Produces a minimal config file containing only user overrides.
    """
    diff: dict[str, Any] = {}
    for key, value in data.items():
        if key not in defaults:
            diff[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(defaults[key], dict):
            sub_diff = _diff_from_defaults(value, defaults[key])
            if sub_diff:
                diff[key] = sub_diff
        elif value != defaults[key]:
            diff[key] = copy.deepcopy(value)
    return diff


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(root: Path | None = None) -> CorcConfig:
    """Load configuration from .corc/config.yaml, falling back to defaults.

    Args:
        root: Project root directory. If None, auto-detected.

    Returns:
        CorcConfig instance with defaults merged with file overrides.
    """
    root = root or get_project_root()
    config_path = root / ".corc" / "config.yaml"
    user_data: dict[str, Any] = {}

    if config_path.exists():
        try:
            with open(config_path) as f:
                raw = yaml.safe_load(f)
            if isinstance(raw, dict):
                user_data = raw
        except (yaml.YAMLError, OSError):
            pass  # Stick with defaults on parse error

    return CorcConfig(data=user_data, config_path=config_path)


def _parse_value(value_str: str) -> Any:
    """Parse a string value into an appropriate Python type.

    Handles: booleans, ints, floats, null, lists (JSON-like), strings.
    """
    # Booleans
    if value_str.lower() in ("true", "yes"):
        return True
    if value_str.lower() in ("false", "no"):
        return False
    # Null
    if value_str.lower() in ("null", "none", "~"):
        return None
    # Integer
    try:
        return int(value_str)
    except ValueError:
        pass
    # Float
    try:
        return float(value_str)
    except ValueError:
        pass
    # JSON-like list
    if value_str.startswith("[") and value_str.endswith("]"):
        import json

        try:
            return json.loads(value_str)
        except json.JSONDecodeError:
            pass
    # Plain string
    return value_str
