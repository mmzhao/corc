"""Audit log backup — configurable backup and rotation of event/session logs.

Copies data/events/ and data/sessions/ to a configurable backup path.
Rotates (deletes) backup files older than a configurable number of days.
Designed to run as part of the daemon's daily maintenance cycle.
"""

import shutil
import time
from pathlib import Path

import yaml


# ── Default configuration ──────────────────────────────────────────────

DEFAULT_BACKUP_CONFIG = {
    "backup_path": "~/.corc-backups/audit/",
    "backup_interval": "daily",
    "rotate_after_days": 90,
}


def load_audit_config(corc_dir: Path) -> dict:
    """Load audit backup configuration from .corc/config.yaml.

    Falls back to defaults if the file or audit section is missing.
    Expands ~ in backup_path.
    """
    config_path = Path(corc_dir) / "config.yaml"
    config = dict(DEFAULT_BACKUP_CONFIG)

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                raw = yaml.safe_load(f) or {}
            audit = raw.get("audit", {})
            if audit:
                config.update({k: v for k, v in audit.items() if v is not None})
        except (yaml.YAMLError, OSError):
            pass  # Stick with defaults on parse error

    # Expand ~ in backup_path
    config["backup_path"] = str(Path(config["backup_path"]).expanduser())
    return config


# ── Backup state tracking ─────────────────────────────────────────────


def _last_backup_path(corc_dir: Path) -> Path:
    """Path to the file that stores the last backup timestamp."""
    return Path(corc_dir) / "last_backup"


def get_last_backup_time(corc_dir: Path) -> float | None:
    """Read the timestamp of the last successful backup.

    Returns epoch seconds or None if no backup has been performed.
    """
    path = _last_backup_path(corc_dir)
    if not path.exists():
        return None
    try:
        return float(path.read_text().strip())
    except (ValueError, OSError):
        return None


def _record_backup_time(corc_dir: Path, timestamp: float | None = None):
    """Record the current time as the last backup time."""
    ts = timestamp if timestamp is not None else time.time()
    path = _last_backup_path(corc_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(ts))


def is_backup_due(corc_dir: Path, interval: str = "daily") -> bool:
    """Check whether a backup is due based on the configured interval.

    Args:
        corc_dir: Path to .corc/ directory.
        interval: "daily" or "weekly".

    Returns:
        True if enough time has elapsed since the last backup.
    """
    last = get_last_backup_time(corc_dir)
    if last is None:
        return True  # Never backed up

    now = time.time()
    if interval == "weekly":
        threshold = 7 * 24 * 3600  # 7 days
    else:
        threshold = 24 * 3600  # 1 day (default for "daily")

    return (now - last) >= threshold


# ── Backup logic ───────────────────────────────────────────────────────


def run_backup(
    events_dir: Path,
    sessions_dir: Path,
    backup_path: str | Path,
) -> dict:
    """Copy data/events/ and data/sessions/ to the backup path.

    Uses shutil.copytree with dirs_exist_ok=True to merge into
    existing backup directories (incremental-friendly).

    Returns a summary dict with counts of files copied.
    """
    backup_root = Path(backup_path)
    events_dst = backup_root / "events"
    sessions_dst = backup_root / "sessions"

    summary = {
        "events_copied": 0,
        "sessions_copied": 0,
        "backup_path": str(backup_root),
    }

    # Backup events
    if events_dir.exists() and any(events_dir.iterdir()):
        events_dst.mkdir(parents=True, exist_ok=True)
        for src_file in sorted(events_dir.glob("*.jsonl")):
            dst_file = events_dst / src_file.name
            shutil.copy2(src_file, dst_file)
            summary["events_copied"] += 1

    # Backup sessions
    if sessions_dir.exists() and any(sessions_dir.iterdir()):
        sessions_dst.mkdir(parents=True, exist_ok=True)
        for src_file in sorted(sessions_dir.glob("*.jsonl")):
            dst_file = sessions_dst / src_file.name
            shutil.copy2(src_file, dst_file)
            summary["sessions_copied"] += 1

    return summary


# ── Rotation logic ─────────────────────────────────────────────────────


def rotate_old_backups(backup_path: str | Path, rotate_after_days: int) -> dict:
    """Delete backup files older than rotate_after_days.

    Checks mtime of each .jsonl file in the backup events/ and sessions/
    subdirectories. Files older than the threshold are removed.

    Returns a summary dict with counts of files rotated.
    """
    backup_root = Path(backup_path)
    cutoff = time.time() - (rotate_after_days * 24 * 3600)

    summary = {"events_rotated": 0, "sessions_rotated": 0}

    for subdir, key in [("events", "events_rotated"), ("sessions", "sessions_rotated")]:
        target = backup_root / subdir
        if not target.exists():
            continue
        for f in sorted(target.glob("*.jsonl")):
            if f.stat().st_mtime < cutoff:
                f.unlink()
                summary[key] += 1

    return summary


# ── Source rotation ────────────────────────────────────────────────────


def rotate_old_source_logs(
    events_dir: Path,
    sessions_dir: Path,
    rotate_after_days: int,
) -> dict:
    """Rotate (delete) source log files older than rotate_after_days.

    Applies the same age-based rotation to the original data/events/
    and data/sessions/ directories — only after they've been backed up.

    Returns a summary dict with counts of files rotated.
    """
    cutoff = time.time() - (rotate_after_days * 24 * 3600)
    summary = {"events_rotated": 0, "sessions_rotated": 0}

    for src_dir, key in [
        (events_dir, "events_rotated"),
        (sessions_dir, "sessions_rotated"),
    ]:
        if not src_dir.exists():
            continue
        for f in sorted(src_dir.glob("*.jsonl")):
            if f.stat().st_mtime < cutoff:
                f.unlink()
                summary[key] += 1

    return summary


# ── Combined daily maintenance entry point ─────────────────────────────


def run_daily_backup(
    corc_dir: Path,
    events_dir: Path,
    sessions_dir: Path,
) -> dict | None:
    """Run backup and rotation if due according to config.

    This is the entry point called by the daemon's daily maintenance cycle.

    Returns a summary dict if backup was performed, None if not due.
    """
    config = load_audit_config(corc_dir)

    if not is_backup_due(corc_dir, config["backup_interval"]):
        return None

    backup_path = config["backup_path"]
    rotate_days = config["rotate_after_days"]

    # 1. Run backup
    backup_summary = run_backup(events_dir, sessions_dir, backup_path)

    # 2. Rotate old backups
    rotation_summary = rotate_old_backups(backup_path, rotate_days)

    # 3. Rotate old source logs (only files that are beyond the retention period)
    source_rotation = rotate_old_source_logs(events_dir, sessions_dir, rotate_days)

    # 4. Record backup time
    _record_backup_time(corc_dir)

    return {
        "backup": backup_summary,
        "rotation": rotation_summary,
        "source_rotation": source_rotation,
        "config": config,
    }
