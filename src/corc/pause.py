"""Pause switch — global halt for all new dispatch.

Any agent or operator can pause the system by writing a lock file.
The daemon checks for this file before each dispatch cycle.
In-flight tasks complete normally; only new dispatches are blocked.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def write_pause_lock(corc_dir: Path, reason: str, source: str | None = None) -> dict:
    """Write the pause lock file with reason, timestamp, and source.

    Args:
        corc_dir: Path to the .corc directory.
        reason: Human-readable reason for pausing.
        source: Who/what triggered the pause (defaults to "cli:<pid>").

    Returns:
        The lock data dict that was written.
    """
    corc_dir = Path(corc_dir)
    corc_dir.mkdir(parents=True, exist_ok=True)

    lock_data = {
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source or f"cli:{os.getpid()}",
    }

    lock_path = corc_dir / "pause.lock"
    lock_path.write_text(json.dumps(lock_data, indent=2) + "\n")
    return lock_data


def remove_pause_lock(corc_dir: Path) -> bool:
    """Remove the pause lock file.

    Returns:
        True if a lock was removed, False if no lock existed.
    """
    lock_path = Path(corc_dir) / "pause.lock"
    if lock_path.exists():
        lock_path.unlink()
        return True
    return False


def read_pause_lock(corc_dir: Path) -> dict | None:
    """Read the pause lock file if it exists.

    Returns:
        The lock data dict, or None if not paused.
    """
    lock_path = Path(corc_dir) / "pause.lock"
    if not lock_path.exists():
        return None

    try:
        return json.loads(lock_path.read_text())
    except (json.JSONDecodeError, OSError):
        # Corrupted lock file — treat as paused with unknown reason
        return {"reason": "unknown (corrupted lock file)", "timestamp": "unknown", "source": "unknown"}


def is_paused(corc_dir: Path) -> bool:
    """Check if the system is paused.

    Returns:
        True if a pause lock file exists.
    """
    return (Path(corc_dir) / "pause.lock").exists()
