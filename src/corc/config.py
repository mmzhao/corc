"""Project paths and configuration."""

from pathlib import Path


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
    }
