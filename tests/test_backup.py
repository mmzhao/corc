"""Tests for audit log backup — configurable backup and rotation.

Verifies:
- Config loading from .corc/config.yaml with defaults
- Backup copies event and session JSONL files to the backup path
- Rotation removes files older than the configured days
- Source log rotation removes old files from data/events/ and data/sessions/
- Daily backup scheduling (is_backup_due logic)
- Daemon integration: backup runs as part of the tick cycle
"""

import json
import os
import time
from pathlib import Path

import pytest

from corc.backup import (
    DEFAULT_BACKUP_CONFIG,
    get_last_backup_time,
    is_backup_due,
    load_audit_config,
    rotate_old_backups,
    rotate_old_source_logs,
    run_backup,
    run_daily_backup,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project structure for backup testing."""
    (tmp_path / "data" / "events").mkdir(parents=True)
    (tmp_path / "data" / "sessions").mkdir(parents=True)
    (tmp_path / ".corc").mkdir()
    return tmp_path


@pytest.fixture
def backup_dir(tmp_path):
    """A separate directory to use as the backup destination."""
    d = tmp_path / "backups" / "audit"
    d.mkdir(parents=True)
    return d


def _write_event_file(events_dir: Path, date: str, events: list[dict] | None = None):
    """Write a fake JSONL event file for a given date."""
    path = events_dir / f"{date}.jsonl"
    entries = events or [{"timestamp": f"{date}T00:00:00.000Z", "event_type": "test"}]
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return path


def _write_session_file(sessions_dir: Path, task_id: str, attempt: int = 1):
    """Write a fake session JSONL file."""
    path = sessions_dir / f"{task_id}-attempt-{attempt}.jsonl"
    entry = {
        "timestamp": "2026-03-22T00:00:00.000Z",
        "type": "output",
        "content": "test",
    }
    with open(path, "w") as f:
        f.write(json.dumps(entry) + "\n")
    return path


def _set_file_age(path: Path, days_old: int):
    """Set a file's mtime to N days in the past."""
    old_time = time.time() - (days_old * 24 * 3600)
    os.utime(path, (old_time, old_time))


# ===========================================================================
# Config loading tests
# ===========================================================================


class TestLoadAuditConfig:
    def test_defaults_when_no_config(self, tmp_project):
        """Returns defaults when config.yaml doesn't exist."""
        config = load_audit_config(tmp_project / ".corc")
        assert config["backup_interval"] == "daily"
        assert config["rotate_after_days"] == 90
        assert "corc-backups" in config["backup_path"]

    def test_loads_from_config_yaml(self, tmp_project):
        """Reads audit section from config.yaml."""
        config_path = tmp_project / ".corc" / "config.yaml"
        config_path.write_text(
            "audit:\n"
            "  backup_path: /tmp/my-backups/\n"
            "  backup_interval: weekly\n"
            "  rotate_after_days: 30\n"
        )
        config = load_audit_config(tmp_project / ".corc")
        assert config["backup_path"] == "/tmp/my-backups"
        assert config["backup_interval"] == "weekly"
        assert config["rotate_after_days"] == 30

    def test_partial_config_uses_defaults(self, tmp_project):
        """Missing keys fall back to defaults."""
        config_path = tmp_project / ".corc" / "config.yaml"
        config_path.write_text("audit:\n  rotate_after_days: 60\n")
        config = load_audit_config(tmp_project / ".corc")
        assert config["rotate_after_days"] == 60
        assert config["backup_interval"] == "daily"  # default

    def test_tilde_expansion(self, tmp_project):
        """Backup path with ~ is expanded."""
        config_path = tmp_project / ".corc" / "config.yaml"
        config_path.write_text("audit:\n  backup_path: ~/my-backups/\n")
        config = load_audit_config(tmp_project / ".corc")
        assert "~" not in config["backup_path"]
        assert config["backup_path"].endswith("my-backups")

    def test_empty_config_file(self, tmp_project):
        """Empty config file still returns defaults."""
        config_path = tmp_project / ".corc" / "config.yaml"
        config_path.write_text("")
        config = load_audit_config(tmp_project / ".corc")
        assert config["backup_interval"] == "daily"

    def test_config_without_audit_section(self, tmp_project):
        """Config with no audit section returns defaults."""
        config_path = tmp_project / ".corc" / "config.yaml"
        config_path.write_text("dispatch:\n  provider: claude-code\n")
        config = load_audit_config(tmp_project / ".corc")
        assert config["rotate_after_days"] == 90

    def test_malformed_yaml_returns_defaults(self, tmp_project):
        """Invalid YAML falls back to defaults gracefully."""
        config_path = tmp_project / ".corc" / "config.yaml"
        config_path.write_text(":\n  bad yaml {{{")
        config = load_audit_config(tmp_project / ".corc")
        assert config["backup_interval"] == "daily"


# ===========================================================================
# Backup tests
# ===========================================================================


class TestRunBackup:
    def test_copies_event_files(self, tmp_project, backup_dir):
        """Event JSONL files are copied to backup_path/events/."""
        events_dir = tmp_project / "data" / "events"
        _write_event_file(events_dir, "2026-03-20")
        _write_event_file(events_dir, "2026-03-21")

        result = run_backup(events_dir, tmp_project / "data" / "sessions", backup_dir)

        assert result["events_copied"] == 2
        assert (backup_dir / "events" / "2026-03-20.jsonl").exists()
        assert (backup_dir / "events" / "2026-03-21.jsonl").exists()

    def test_copies_session_files(self, tmp_project, backup_dir):
        """Session JSONL files are copied to backup_path/sessions/."""
        sessions_dir = tmp_project / "data" / "sessions"
        _write_session_file(sessions_dir, "task-1")
        _write_session_file(sessions_dir, "task-2", attempt=2)

        result = run_backup(tmp_project / "data" / "events", sessions_dir, backup_dir)

        assert result["sessions_copied"] == 2
        assert (backup_dir / "sessions" / "task-1-attempt-1.jsonl").exists()
        assert (backup_dir / "sessions" / "task-2-attempt-2.jsonl").exists()

    def test_backup_preserves_file_content(self, tmp_project, backup_dir):
        """Backed-up files have the same content as originals."""
        events_dir = tmp_project / "data" / "events"
        events = [{"event_type": "test_event", "data": "important"}]
        _write_event_file(events_dir, "2026-03-22", events)

        run_backup(events_dir, tmp_project / "data" / "sessions", backup_dir)

        original = (events_dir / "2026-03-22.jsonl").read_text()
        backed_up = (backup_dir / "events" / "2026-03-22.jsonl").read_text()
        assert original == backed_up

    def test_backup_creates_destination_dirs(self, tmp_path):
        """Backup creates the destination directory structure if missing."""
        events_dir = tmp_path / "data" / "events"
        events_dir.mkdir(parents=True)
        _write_event_file(events_dir, "2026-03-22")

        backup_dest = tmp_path / "new-backup-dir" / "audit"

        result = run_backup(events_dir, tmp_path / "data" / "sessions", backup_dest)
        assert result["events_copied"] == 1
        assert (backup_dest / "events" / "2026-03-22.jsonl").exists()

    def test_backup_empty_dirs(self, tmp_project, backup_dir):
        """No errors when source dirs are empty."""
        result = run_backup(
            tmp_project / "data" / "events",
            tmp_project / "data" / "sessions",
            backup_dir,
        )
        assert result["events_copied"] == 0
        assert result["sessions_copied"] == 0

    def test_backup_nonexistent_source(self, tmp_path, backup_dir):
        """No errors when source dirs don't exist."""
        result = run_backup(
            tmp_path / "nonexistent" / "events",
            tmp_path / "nonexistent" / "sessions",
            backup_dir,
        )
        assert result["events_copied"] == 0
        assert result["sessions_copied"] == 0

    def test_backup_overwrites_existing(self, tmp_project, backup_dir):
        """Re-running backup overwrites existing backup files."""
        events_dir = tmp_project / "data" / "events"
        _write_event_file(events_dir, "2026-03-22", [{"version": 1}])

        run_backup(events_dir, tmp_project / "data" / "sessions", backup_dir)

        # Update the source file
        _write_event_file(events_dir, "2026-03-22", [{"version": 2}])
        run_backup(events_dir, tmp_project / "data" / "sessions", backup_dir)

        content = json.loads(
            (backup_dir / "events" / "2026-03-22.jsonl").read_text().strip()
        )
        assert content["version"] == 2


# ===========================================================================
# Rotation tests
# ===========================================================================


class TestRotateOldBackups:
    def test_rotates_old_files(self, backup_dir):
        """Files older than rotate_after_days are removed."""
        events_dir = backup_dir / "events"
        events_dir.mkdir(parents=True)

        # Create old and new files
        old_file = _write_event_file(events_dir, "2026-01-01")
        _set_file_age(old_file, 100)

        new_file = _write_event_file(events_dir, "2026-03-22")

        result = rotate_old_backups(backup_dir, rotate_after_days=90)

        assert result["events_rotated"] == 1
        assert not old_file.exists()
        assert new_file.exists()

    def test_rotates_old_session_files(self, backup_dir):
        """Session files older than rotate_after_days are removed."""
        sessions_dir = backup_dir / "sessions"
        sessions_dir.mkdir(parents=True)

        old_file = _write_session_file(sessions_dir, "old-task")
        _set_file_age(old_file, 100)

        new_file = _write_session_file(sessions_dir, "new-task")

        result = rotate_old_backups(backup_dir, rotate_after_days=90)

        assert result["sessions_rotated"] == 1
        assert not old_file.exists()
        assert new_file.exists()

    def test_no_rotation_when_all_recent(self, backup_dir):
        """Nothing rotated when all files are within retention period."""
        events_dir = backup_dir / "events"
        events_dir.mkdir(parents=True)

        f1 = _write_event_file(events_dir, "2026-03-20")
        f2 = _write_event_file(events_dir, "2026-03-21")

        result = rotate_old_backups(backup_dir, rotate_after_days=90)

        assert result["events_rotated"] == 0
        assert f1.exists()
        assert f2.exists()

    def test_no_rotation_when_dirs_missing(self, tmp_path):
        """No errors when backup dirs don't exist yet."""
        result = rotate_old_backups(tmp_path / "nonexistent", rotate_after_days=90)
        assert result["events_rotated"] == 0
        assert result["sessions_rotated"] == 0

    def test_rotation_with_custom_days(self, backup_dir):
        """Rotation respects the configurable days threshold."""
        events_dir = backup_dir / "events"
        events_dir.mkdir(parents=True)

        file_30_days = _write_event_file(events_dir, "2026-02-20")
        _set_file_age(file_30_days, 30)

        file_10_days = _write_event_file(events_dir, "2026-03-12")
        _set_file_age(file_10_days, 10)

        # Rotate files older than 15 days
        result = rotate_old_backups(backup_dir, rotate_after_days=15)

        assert result["events_rotated"] == 1
        assert not file_30_days.exists()
        assert file_10_days.exists()


# ===========================================================================
# Source log rotation tests
# ===========================================================================


class TestRotateOldSourceLogs:
    def test_rotates_old_source_events(self, tmp_project):
        """Old source event files are removed after rotation."""
        events_dir = tmp_project / "data" / "events"
        sessions_dir = tmp_project / "data" / "sessions"

        old = _write_event_file(events_dir, "2025-12-01")
        _set_file_age(old, 120)

        recent = _write_event_file(events_dir, "2026-03-22")

        result = rotate_old_source_logs(events_dir, sessions_dir, rotate_after_days=90)

        assert result["events_rotated"] == 1
        assert not old.exists()
        assert recent.exists()

    def test_rotates_old_source_sessions(self, tmp_project):
        """Old source session files are removed after rotation."""
        events_dir = tmp_project / "data" / "events"
        sessions_dir = tmp_project / "data" / "sessions"

        old = _write_session_file(sessions_dir, "old-task")
        _set_file_age(old, 120)

        recent = _write_session_file(sessions_dir, "recent-task")

        result = rotate_old_source_logs(events_dir, sessions_dir, rotate_after_days=90)

        assert result["sessions_rotated"] == 1
        assert not old.exists()
        assert recent.exists()


# ===========================================================================
# Backup scheduling (is_backup_due) tests
# ===========================================================================


class TestIsBackupDue:
    def test_due_when_never_backed_up(self, tmp_project):
        """Backup is due when no last_backup file exists."""
        assert is_backup_due(tmp_project / ".corc") is True

    def test_not_due_within_daily_interval(self, tmp_project):
        """Backup is not due if last backup was less than 24h ago."""
        corc_dir = tmp_project / ".corc"
        # Write last backup time as 1 hour ago
        recent = time.time() - 3600
        (corc_dir / "last_backup").write_text(str(recent))

        assert is_backup_due(corc_dir, "daily") is False

    def test_due_after_daily_interval(self, tmp_project):
        """Backup is due if last backup was more than 24h ago."""
        corc_dir = tmp_project / ".corc"
        old = time.time() - (25 * 3600)  # 25 hours ago
        (corc_dir / "last_backup").write_text(str(old))

        assert is_backup_due(corc_dir, "daily") is True

    def test_weekly_interval(self, tmp_project):
        """Weekly interval requires 7 days to elapse."""
        corc_dir = tmp_project / ".corc"

        # 3 days ago — not due for weekly
        three_days = time.time() - (3 * 24 * 3600)
        (corc_dir / "last_backup").write_text(str(three_days))
        assert is_backup_due(corc_dir, "weekly") is False

        # 8 days ago — due for weekly
        eight_days = time.time() - (8 * 24 * 3600)
        (corc_dir / "last_backup").write_text(str(eight_days))
        assert is_backup_due(corc_dir, "weekly") is True

    def test_invalid_last_backup_file(self, tmp_project):
        """Backup is due if last_backup file is corrupt."""
        corc_dir = tmp_project / ".corc"
        (corc_dir / "last_backup").write_text("not-a-number")

        assert is_backup_due(corc_dir) is True


# ===========================================================================
# run_daily_backup integration tests
# ===========================================================================


class TestRunDailyBackup:
    def test_full_backup_cycle(self, tmp_project):
        """Full cycle: backup events and sessions, rotate old, record time."""
        events_dir = tmp_project / "data" / "events"
        sessions_dir = tmp_project / "data" / "sessions"
        corc_dir = tmp_project / ".corc"
        backup_dest = tmp_project / "backups" / "audit"

        # Config: backup to local dir, rotate after 30 days
        (corc_dir / "config.yaml").write_text(
            f"audit:\n"
            f"  backup_path: {backup_dest}\n"
            f"  backup_interval: daily\n"
            f"  rotate_after_days: 30\n"
        )

        # Create source files
        _write_event_file(events_dir, "2026-03-22")
        _write_session_file(sessions_dir, "task-1")

        # Create an old backup file that should be rotated
        old_backup_events = backup_dest / "events"
        old_backup_events.mkdir(parents=True)
        old_file = _write_event_file(old_backup_events, "2025-12-01")
        _set_file_age(old_file, 60)

        result = run_daily_backup(corc_dir, events_dir, sessions_dir)

        assert result is not None
        assert result["backup"]["events_copied"] == 1
        assert result["backup"]["sessions_copied"] == 1
        assert result["rotation"]["events_rotated"] == 1
        assert (backup_dest / "events" / "2026-03-22.jsonl").exists()
        assert (backup_dest / "sessions" / "task-1-attempt-1.jsonl").exists()
        assert not old_file.exists()

        # Verify last backup time was recorded
        assert get_last_backup_time(corc_dir) is not None

    def test_skips_when_not_due(self, tmp_project):
        """Returns None when backup is not due yet."""
        corc_dir = tmp_project / ".corc"

        # Mark backup as done recently
        (corc_dir / "last_backup").write_text(str(time.time()))

        result = run_daily_backup(
            corc_dir,
            tmp_project / "data" / "events",
            tmp_project / "data" / "sessions",
        )
        assert result is None

    def test_records_backup_time(self, tmp_project):
        """After backup, last_backup file is updated."""
        corc_dir = tmp_project / ".corc"

        before = time.time()
        run_daily_backup(
            corc_dir,
            tmp_project / "data" / "events",
            tmp_project / "data" / "sessions",
        )
        after = time.time()

        last = get_last_backup_time(corc_dir)
        assert last is not None
        assert before <= last <= after

    def test_source_rotation_removes_old_source_logs(self, tmp_project):
        """Old source log files are rotated alongside backups."""
        events_dir = tmp_project / "data" / "events"
        sessions_dir = tmp_project / "data" / "sessions"
        corc_dir = tmp_project / ".corc"
        backup_dest = tmp_project / "backups" / "audit"

        (corc_dir / "config.yaml").write_text(
            f"audit:\n  backup_path: {backup_dest}\n  rotate_after_days: 30\n"
        )

        # Create old source files
        old_event = _write_event_file(events_dir, "2025-11-01")
        _set_file_age(old_event, 120)
        old_session = _write_session_file(sessions_dir, "old-task")
        _set_file_age(old_session, 120)

        # Create recent source files
        recent_event = _write_event_file(events_dir, "2026-03-22")
        recent_session = _write_session_file(sessions_dir, "recent-task")

        result = run_daily_backup(corc_dir, events_dir, sessions_dir)

        assert result["source_rotation"]["events_rotated"] == 1
        assert result["source_rotation"]["sessions_rotated"] == 1
        assert not old_event.exists()
        assert not old_session.exists()
        assert recent_event.exists()
        assert recent_session.exists()


# ===========================================================================
# Daemon integration tests
# ===========================================================================


class TestDaemonBackupIntegration:
    """Verify backup runs as part of the daemon's tick cycle."""

    def test_daemon_tick_triggers_backup(self, tmp_project):
        """Daemon _check_daily_backup calls run_daily_backup."""
        from unittest.mock import MagicMock, patch

        from corc.audit import AuditLog
        from corc.daemon import Daemon
        from corc.dispatch import AgentResult, Constraints
        from corc.mutations import MutationLog
        from corc.sessions import SessionLogger
        from corc.state import WorkState

        # Set up daemon components
        ml = MutationLog(tmp_project / "data" / "mutations.jsonl")
        ws = WorkState(tmp_project / "data" / "state.db", ml)
        al = AuditLog(tmp_project / "data" / "events")
        sl = SessionLogger(tmp_project / "data" / "sessions")

        mock_dispatcher = MagicMock()
        mock_dispatcher.dispatch.return_value = AgentResult(
            output="done", exit_code=0, duration_s=0.1
        )

        daemon = Daemon(
            state=ws,
            mutation_log=ml,
            audit_log=al,
            session_logger=sl,
            dispatcher=mock_dispatcher,
            project_root=tmp_project,
            auto_reload=False,
        )

        # Write config
        (tmp_project / ".corc" / "config.yaml").write_text(
            f"audit:\n"
            f"  backup_path: {tmp_project / 'backups' / 'audit'}\n"
            f"  backup_interval: daily\n"
            f"  rotate_after_days: 90\n"
        )

        # Create some event files to back up
        _write_event_file(tmp_project / "data" / "events", "2026-03-22")

        # Run a single tick
        daemon._check_daily_backup()

        # Verify backup was created
        assert (
            tmp_project / "backups" / "audit" / "events" / "2026-03-22.jsonl"
        ).exists()

        # Verify audit log recorded the backup
        events = al.read_today()
        backup_events = [e for e in events if e["event_type"] == "backup_completed"]
        assert len(backup_events) == 1
        assert backup_events[0]["events_copied"] == 1

    def test_daemon_tick_skips_when_not_due(self, tmp_project):
        """Daemon doesn't backup when interval hasn't elapsed."""
        from unittest.mock import MagicMock

        from corc.audit import AuditLog
        from corc.daemon import Daemon
        from corc.dispatch import AgentResult
        from corc.mutations import MutationLog
        from corc.sessions import SessionLogger
        from corc.state import WorkState

        ml = MutationLog(tmp_project / "data" / "mutations.jsonl")
        ws = WorkState(tmp_project / "data" / "state.db", ml)
        al = AuditLog(tmp_project / "data" / "events")
        sl = SessionLogger(tmp_project / "data" / "sessions")

        mock_dispatcher = MagicMock()
        daemon = Daemon(
            state=ws,
            mutation_log=ml,
            audit_log=al,
            session_logger=sl,
            dispatcher=mock_dispatcher,
            project_root=tmp_project,
            auto_reload=False,
        )

        # Mark backup as recently done
        (tmp_project / ".corc" / "last_backup").write_text(str(time.time()))

        daemon._check_daily_backup()

        # No backup_completed event should be logged
        events = al.read_today()
        backup_events = [e for e in events if e["event_type"] == "backup_completed"]
        assert len(backup_events) == 0

    def test_daemon_backup_logs_error_on_failure(self, tmp_project):
        """Daemon logs backup_failed when backup raises OSError."""
        from unittest.mock import MagicMock, patch

        from corc.audit import AuditLog
        from corc.daemon import Daemon
        from corc.mutations import MutationLog
        from corc.sessions import SessionLogger
        from corc.state import WorkState

        ml = MutationLog(tmp_project / "data" / "mutations.jsonl")
        ws = WorkState(tmp_project / "data" / "state.db", ml)
        al = AuditLog(tmp_project / "data" / "events")
        sl = SessionLogger(tmp_project / "data" / "sessions")

        mock_dispatcher = MagicMock()
        daemon = Daemon(
            state=ws,
            mutation_log=ml,
            audit_log=al,
            session_logger=sl,
            dispatcher=mock_dispatcher,
            project_root=tmp_project,
            auto_reload=False,
        )

        # Simulate a backup failure
        with patch("corc.daemon.run_daily_backup", side_effect=OSError("disk full")):
            daemon._check_daily_backup()

        events = al.read_today()
        fail_events = [e for e in events if e["event_type"] == "backup_failed"]
        assert len(fail_events) == 1
        assert "disk full" in fail_events[0]["error"]
