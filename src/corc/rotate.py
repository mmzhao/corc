"""Log rotation — move old logs to date-stamped archive directories.

Moves session and audit log files older than a configurable number of days
into archive directories organized by date: archive/YYYY-MM-DD/.

**Files are never deleted, only moved to the archive.**

Configuration (in .corc/config.yaml):
    rotation:
      rotate_after_days: 7       # Default: move files older than 7 days
      session_archive: null      # Default: data/sessions/archive/
      events_archive: null       # Default: data/events/archive/

Public API:
    - rotate_logs()          — move old session and audit logs to archive
    - rotate_session_logs()  — move old session logs only
    - rotate_event_logs()    — move old audit/event logs only
    - load_rotation_config() — read rotation settings from config.yaml
    - run_daily_rotation()   — entry point for daemon (checks if due)
"""

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


# ── Default configuration ──────────────────────────────────────────────

DEFAULT_ROTATION_CONFIG = {
    "rotate_after_days": 7,
    "session_archive": None,  # None → sessions_dir / "archive"
    "events_archive": None,  # None → events_dir / "archive"
}


def load_rotation_config(corc_dir: Path) -> dict:
    """Load rotation configuration from .corc/config.yaml.

    Falls back to defaults if the file or rotation section is missing.
    """
    config_path = Path(corc_dir) / "config.yaml"
    config = dict(DEFAULT_ROTATION_CONFIG)

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                raw = yaml.safe_load(f) or {}
            rotation = raw.get("rotation", {})
            if rotation:
                config.update({k: v for k, v in rotation.items() if v is not None})
        except (yaml.YAMLError, OSError):
            pass  # Stick with defaults on parse error

    return config


# ── Rotation state tracking ───────────────────────────────────────────


def _last_rotation_path(corc_dir: Path) -> Path:
    """Path to the file that stores the last rotation timestamp."""
    return Path(corc_dir) / "last_rotation"


def get_last_rotation_time(corc_dir: Path) -> float | None:
    """Read the timestamp of the last successful rotation.

    Returns epoch seconds or None if no rotation has been performed.
    """
    path = _last_rotation_path(corc_dir)
    if not path.exists():
        return None
    try:
        return float(path.read_text().strip())
    except (ValueError, OSError):
        return None


def _record_rotation_time(corc_dir: Path, timestamp: float | None = None):
    """Record the current time as the last rotation time."""
    ts = timestamp if timestamp is not None else time.time()
    path = _last_rotation_path(corc_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(ts))


def is_rotation_due(corc_dir: Path) -> bool:
    """Check whether a rotation is due (daily cadence).

    Returns True if at least 24 hours have elapsed since the last rotation.
    """
    last = get_last_rotation_time(corc_dir)
    if last is None:
        return True  # Never rotated

    now = time.time()
    threshold = 24 * 3600  # 1 day
    return (now - last) >= threshold


# ── Core rotation logic ───────────────────────────────────────────────


def _date_from_mtime(mtime: float) -> str:
    """Convert a file mtime to YYYY-MM-DD string (UTC)."""
    dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def _move_old_files(
    source_dir: Path,
    archive_dir: Path,
    rotate_after_days: int,
    glob_pattern: str = "*.jsonl",
) -> dict:
    """Move files older than rotate_after_days from source to archive.

    Files are organized into archive/YYYY-MM-DD/ subdirectories based
    on their modification time. Files are never deleted.

    Returns a summary dict with count of files moved and their paths.
    """
    if not source_dir.exists():
        return {"moved": 0, "files": []}

    cutoff = time.time() - (rotate_after_days * 24 * 3600)
    moved_files = []

    for f in sorted(source_dir.glob(glob_pattern)):
        # Skip the archive directory itself
        if f.is_dir():
            continue
        # Skip files inside archive subdirectories
        try:
            f.relative_to(archive_dir)
            continue
        except ValueError:
            pass  # Not inside archive dir, proceed

        if f.stat().st_mtime < cutoff:
            date_str = _date_from_mtime(f.stat().st_mtime)
            dest_dir = archive_dir / date_str
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / f.name
            f.rename(dest_path)
            moved_files.append(str(dest_path))

    return {"moved": len(moved_files), "files": moved_files}


def rotate_session_logs(
    sessions_dir: Path,
    rotate_after_days: int = 7,
    archive_dir: Path | None = None,
) -> dict:
    """Move old session log files to the session archive.

    Args:
        sessions_dir: Path to data/sessions/ directory.
        rotate_after_days: Files older than this are moved.
        archive_dir: Override archive location (default: sessions_dir/archive/).

    Returns:
        Summary dict with moved count and file paths.
    """
    sessions_dir = Path(sessions_dir)
    if archive_dir is None:
        archive_dir = sessions_dir / "archive"
    else:
        archive_dir = Path(archive_dir)

    return _move_old_files(sessions_dir, archive_dir, rotate_after_days)


def rotate_event_logs(
    events_dir: Path,
    rotate_after_days: int = 7,
    archive_dir: Path | None = None,
) -> dict:
    """Move old audit/event log files to the events archive.

    Args:
        events_dir: Path to data/events/ directory.
        rotate_after_days: Files older than this are moved.
        archive_dir: Override archive location (default: events_dir/archive/).

    Returns:
        Summary dict with moved count and file paths.
    """
    events_dir = Path(events_dir)
    if archive_dir is None:
        archive_dir = events_dir / "archive"
    else:
        archive_dir = Path(archive_dir)

    return _move_old_files(events_dir, archive_dir, rotate_after_days)


def rotate_logs(
    events_dir: Path,
    sessions_dir: Path,
    rotate_after_days: int = 7,
    events_archive: Path | None = None,
    sessions_archive: Path | None = None,
) -> dict:
    """Move old session and audit logs to their respective archives.

    This is the main entry point for log rotation. Files are moved to
    date-stamped archive subdirectories and are never deleted.

    Args:
        events_dir: Path to data/events/ directory.
        sessions_dir: Path to data/sessions/ directory.
        rotate_after_days: Files older than this many days are moved.
        events_archive: Override events archive dir.
        sessions_archive: Override sessions archive dir.

    Returns:
        Summary dict with session and event rotation results.
    """
    sessions_result = rotate_session_logs(
        sessions_dir, rotate_after_days, sessions_archive
    )
    events_result = rotate_event_logs(events_dir, rotate_after_days, events_archive)

    return {
        "sessions": sessions_result,
        "events": events_result,
        "rotate_after_days": rotate_after_days,
    }


# ── Daemon entry point ────────────────────────────────────────────────


def run_daily_rotation(
    corc_dir: Path,
    events_dir: Path,
    sessions_dir: Path,
) -> dict | None:
    """Run log rotation if due according to daily schedule.

    This is the entry point called by the daemon's daily maintenance cycle.
    Checks if 24 hours have elapsed since the last rotation. If so, reads
    config and moves old files to archive directories.

    Returns a summary dict if rotation was performed, None if not due.
    """
    if not is_rotation_due(corc_dir):
        return None

    config = load_rotation_config(corc_dir)
    rotate_days = config["rotate_after_days"]

    # Resolve archive paths from config or use defaults
    events_archive = (
        Path(config["events_archive"]) if config.get("events_archive") else None
    )
    sessions_archive = (
        Path(config["session_archive"]) if config.get("session_archive") else None
    )

    result = rotate_logs(
        events_dir=events_dir,
        sessions_dir=sessions_dir,
        rotate_after_days=rotate_days,
        events_archive=events_archive,
        sessions_archive=sessions_archive,
    )

    # Record rotation time
    _record_rotation_time(corc_dir)

    return result
