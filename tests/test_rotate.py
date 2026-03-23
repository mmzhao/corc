"""Tests for log rotation — move old logs to date-stamped archive directories.

Verifies:
- Config loading from .corc/config.yaml with defaults
- Session logs older than configurable days moved to data/sessions/archive/YYYY-MM-DD/
- Audit/event logs same pattern to data/events/archive/YYYY-MM-DD/
- No files are ever deleted, only moved to archive
- Archive directory structure is created correctly
- Rotation scheduling (is_rotation_due logic)
- Combined rotation of sessions and events
- Daemon integration: rotation runs as part of the tick cycle
- CLI `corc logs rotate` manual rotation
"""

import json
import os
import time
from pathlib import Path

import pytest

from corc.rotate import (
    DEFAULT_ROTATION_CONFIG,
    _date_from_mtime,
    _move_old_files,
    get_last_rotation_time,
    is_rotation_due,
    load_rotation_config,
    rotate_event_logs,
    rotate_logs,
    rotate_session_logs,
    run_daily_rotation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project structure for rotation testing."""
    (tmp_path / "data" / "events").mkdir(parents=True)
    (tmp_path / "data" / "sessions").mkdir(parents=True)
    (tmp_path / ".corc").mkdir()
    return tmp_path


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


class TestLoadRotationConfig:
    def test_defaults_when_no_config(self, tmp_project):
        """Returns defaults when config.yaml doesn't exist."""
        config = load_rotation_config(tmp_project / ".corc")
        assert config["rotate_after_days"] == 7
        assert config["session_archive"] is None
        assert config["events_archive"] is None

    def test_loads_from_config_yaml(self, tmp_project):
        """Reads rotation section from config.yaml."""
        config_path = tmp_project / ".corc" / "config.yaml"
        config_path.write_text(
            "rotation:\n"
            "  rotate_after_days: 14\n"
            "  session_archive: /tmp/session-archive\n"
            "  events_archive: /tmp/events-archive\n"
        )
        config = load_rotation_config(tmp_project / ".corc")
        assert config["rotate_after_days"] == 14
        assert config["session_archive"] == "/tmp/session-archive"
        assert config["events_archive"] == "/tmp/events-archive"

    def test_partial_config_uses_defaults(self, tmp_project):
        """Missing keys fall back to defaults."""
        config_path = tmp_project / ".corc" / "config.yaml"
        config_path.write_text("rotation:\n  rotate_after_days: 30\n")
        config = load_rotation_config(tmp_project / ".corc")
        assert config["rotate_after_days"] == 30
        assert config["session_archive"] is None  # default

    def test_empty_config_file(self, tmp_project):
        """Empty config file still returns defaults."""
        config_path = tmp_project / ".corc" / "config.yaml"
        config_path.write_text("")
        config = load_rotation_config(tmp_project / ".corc")
        assert config["rotate_after_days"] == 7

    def test_config_without_rotation_section(self, tmp_project):
        """Config with no rotation section returns defaults."""
        config_path = tmp_project / ".corc" / "config.yaml"
        config_path.write_text("audit:\n  backup_interval: daily\n")
        config = load_rotation_config(tmp_project / ".corc")
        assert config["rotate_after_days"] == 7

    def test_malformed_yaml_returns_defaults(self, tmp_project):
        """Invalid YAML falls back to defaults gracefully."""
        config_path = tmp_project / ".corc" / "config.yaml"
        config_path.write_text(":\n  bad yaml {{{")
        config = load_rotation_config(tmp_project / ".corc")
        assert config["rotate_after_days"] == 7


# ===========================================================================
# Session log rotation tests
# ===========================================================================


class TestRotateSessionLogs:
    def test_moves_old_session_files_to_archive(self, tmp_project):
        """Session files older than rotate_after_days are moved to archive."""
        sessions_dir = tmp_project / "data" / "sessions"

        old_file = _write_session_file(sessions_dir, "old-task")
        _set_file_age(old_file, 10)

        new_file = _write_session_file(sessions_dir, "new-task")

        result = rotate_session_logs(sessions_dir, rotate_after_days=7)

        assert result["moved"] == 1
        # Old file no longer in source directory
        assert not old_file.exists()
        # New file untouched
        assert new_file.exists()

    def test_archive_directory_structure(self, tmp_project):
        """Files are moved to archive/YYYY-MM-DD/ based on mtime."""
        sessions_dir = tmp_project / "data" / "sessions"
        archive_dir = sessions_dir / "archive"

        old_file = _write_session_file(sessions_dir, "task-1")
        _set_file_age(old_file, 10)

        # Get the expected date string from the file's mtime
        expected_date = _date_from_mtime(old_file.stat().st_mtime)

        rotate_session_logs(sessions_dir, rotate_after_days=7)

        # File should be in archive/YYYY-MM-DD/
        archived = archive_dir / expected_date / "task-1-attempt-1.jsonl"
        assert archived.exists()

    def test_archived_file_content_preserved(self, tmp_project):
        """Archived files have the same content as originals."""
        sessions_dir = tmp_project / "data" / "sessions"

        old_file = _write_session_file(sessions_dir, "task-1")
        original_content = old_file.read_text()
        _set_file_age(old_file, 10)
        expected_date = _date_from_mtime(old_file.stat().st_mtime)

        rotate_session_logs(sessions_dir, rotate_after_days=7)

        archived = sessions_dir / "archive" / expected_date / "task-1-attempt-1.jsonl"
        assert archived.read_text() == original_content

    def test_no_rotation_when_all_recent(self, tmp_project):
        """Nothing moved when all files are within retention period."""
        sessions_dir = tmp_project / "data" / "sessions"

        f1 = _write_session_file(sessions_dir, "task-1")
        f2 = _write_session_file(sessions_dir, "task-2")

        result = rotate_session_logs(sessions_dir, rotate_after_days=7)

        assert result["moved"] == 0
        assert f1.exists()
        assert f2.exists()

    def test_custom_archive_dir(self, tmp_project):
        """Files can be moved to a custom archive directory."""
        sessions_dir = tmp_project / "data" / "sessions"
        custom_archive = tmp_project / "custom-archive" / "sessions"

        old_file = _write_session_file(sessions_dir, "task-1")
        _set_file_age(old_file, 10)
        expected_date = _date_from_mtime(old_file.stat().st_mtime)

        result = rotate_session_logs(
            sessions_dir, rotate_after_days=7, archive_dir=custom_archive
        )

        assert result["moved"] == 1
        assert not old_file.exists()
        archived = custom_archive / expected_date / "task-1-attempt-1.jsonl"
        assert archived.exists()

    def test_multiple_files_different_dates(self, tmp_project):
        """Files from different dates go to different archive subdirs."""
        sessions_dir = tmp_project / "data" / "sessions"

        f1 = _write_session_file(sessions_dir, "task-a")
        _set_file_age(f1, 20)
        date1 = _date_from_mtime(f1.stat().st_mtime)

        f2 = _write_session_file(sessions_dir, "task-b")
        _set_file_age(f2, 30)
        date2 = _date_from_mtime(f2.stat().st_mtime)

        result = rotate_session_logs(sessions_dir, rotate_after_days=7)

        assert result["moved"] == 2
        archive = sessions_dir / "archive"
        assert (archive / date1 / "task-a-attempt-1.jsonl").exists()
        assert (archive / date2 / "task-b-attempt-1.jsonl").exists()

    def test_nonexistent_source_dir(self, tmp_path):
        """No errors when source dir doesn't exist."""
        result = rotate_session_logs(
            tmp_path / "nonexistent" / "sessions", rotate_after_days=7
        )
        assert result["moved"] == 0

    def test_empty_source_dir(self, tmp_project):
        """No errors when source dir is empty."""
        sessions_dir = tmp_project / "data" / "sessions"
        result = rotate_session_logs(sessions_dir, rotate_after_days=7)
        assert result["moved"] == 0

    def test_files_never_deleted(self, tmp_project):
        """Verify files are moved, not deleted — they exist in archive."""
        sessions_dir = tmp_project / "data" / "sessions"

        old_file = _write_session_file(sessions_dir, "important-task")
        original_content = old_file.read_text()
        _set_file_age(old_file, 10)

        rotate_session_logs(sessions_dir, rotate_after_days=7)

        # File removed from source
        assert not old_file.exists()

        # File exists in archive with same content
        archive_files = list((sessions_dir / "archive").rglob("*.jsonl"))
        assert len(archive_files) == 1
        assert archive_files[0].read_text() == original_content


# ===========================================================================
# Event/audit log rotation tests
# ===========================================================================


class TestRotateEventLogs:
    def test_moves_old_event_files_to_archive(self, tmp_project):
        """Event files older than rotate_after_days are moved to archive."""
        events_dir = tmp_project / "data" / "events"

        old_file = _write_event_file(events_dir, "2026-01-01")
        _set_file_age(old_file, 10)

        new_file = _write_event_file(events_dir, "2026-03-22")

        result = rotate_event_logs(events_dir, rotate_after_days=7)

        assert result["moved"] == 1
        assert not old_file.exists()
        assert new_file.exists()

    def test_archive_directory_structure(self, tmp_project):
        """Event files are moved to archive/YYYY-MM-DD/ based on mtime."""
        events_dir = tmp_project / "data" / "events"
        archive_dir = events_dir / "archive"

        old_file = _write_event_file(events_dir, "2026-01-15")
        _set_file_age(old_file, 10)
        expected_date = _date_from_mtime(old_file.stat().st_mtime)

        rotate_event_logs(events_dir, rotate_after_days=7)

        archived = archive_dir / expected_date / "2026-01-15.jsonl"
        assert archived.exists()

    def test_archived_event_content_preserved(self, tmp_project):
        """Archived event files have the same content as originals."""
        events_dir = tmp_project / "data" / "events"

        events = [{"event_type": "important", "data": "must_not_lose"}]
        old_file = _write_event_file(events_dir, "2026-01-15", events)
        original_content = old_file.read_text()
        _set_file_age(old_file, 10)
        expected_date = _date_from_mtime(old_file.stat().st_mtime)

        rotate_event_logs(events_dir, rotate_after_days=7)

        archived = events_dir / "archive" / expected_date / "2026-01-15.jsonl"
        assert archived.read_text() == original_content

    def test_custom_archive_dir(self, tmp_project):
        """Events can be moved to a custom archive directory."""
        events_dir = tmp_project / "data" / "events"
        custom_archive = tmp_project / "custom-archive" / "events"

        old_file = _write_event_file(events_dir, "2026-01-15")
        _set_file_age(old_file, 10)
        expected_date = _date_from_mtime(old_file.stat().st_mtime)

        result = rotate_event_logs(
            events_dir, rotate_after_days=7, archive_dir=custom_archive
        )

        assert result["moved"] == 1
        archived = custom_archive / expected_date / "2026-01-15.jsonl"
        assert archived.exists()

    def test_events_never_deleted(self, tmp_project):
        """Event files are moved, not deleted — they exist in archive."""
        events_dir = tmp_project / "data" / "events"

        old_file = _write_event_file(events_dir, "2026-01-01")
        original_content = old_file.read_text()
        _set_file_age(old_file, 10)

        rotate_event_logs(events_dir, rotate_after_days=7)

        assert not old_file.exists()

        archive_files = list((events_dir / "archive").rglob("*.jsonl"))
        assert len(archive_files) == 1
        assert archive_files[0].read_text() == original_content


# ===========================================================================
# Combined rotation tests
# ===========================================================================


class TestRotateLogs:
    def test_rotates_both_sessions_and_events(self, tmp_project):
        """Combined rotation moves both session and event files."""
        events_dir = tmp_project / "data" / "events"
        sessions_dir = tmp_project / "data" / "sessions"

        old_event = _write_event_file(events_dir, "2026-01-01")
        _set_file_age(old_event, 10)

        old_session = _write_session_file(sessions_dir, "old-task")
        _set_file_age(old_session, 10)

        new_event = _write_event_file(events_dir, "2026-03-22")
        new_session = _write_session_file(sessions_dir, "new-task")

        result = rotate_logs(events_dir, sessions_dir, rotate_after_days=7)

        assert result["events"]["moved"] == 1
        assert result["sessions"]["moved"] == 1
        assert result["rotate_after_days"] == 7

        # Old files gone from source
        assert not old_event.exists()
        assert not old_session.exists()

        # New files still present
        assert new_event.exists()
        assert new_session.exists()

        # Archived files present
        event_archives = list((events_dir / "archive").rglob("*.jsonl"))
        session_archives = list((sessions_dir / "archive").rglob("*.jsonl"))
        assert len(event_archives) == 1
        assert len(session_archives) == 1

    def test_custom_rotate_days(self, tmp_project):
        """Rotation respects the configurable days threshold."""
        sessions_dir = tmp_project / "data" / "sessions"

        f_5days = _write_session_file(sessions_dir, "task-5")
        _set_file_age(f_5days, 5)

        f_15days = _write_session_file(sessions_dir, "task-15")
        _set_file_age(f_15days, 15)

        events_dir = tmp_project / "data" / "events"

        # Rotate files older than 10 days
        result = rotate_logs(events_dir, sessions_dir, rotate_after_days=10)

        assert result["sessions"]["moved"] == 1
        # 5-day-old file should still be there
        assert f_5days.exists()
        # 15-day-old file should be archived
        assert not f_15days.exists()

    def test_custom_archive_dirs(self, tmp_project):
        """Custom archive directories are respected."""
        events_dir = tmp_project / "data" / "events"
        sessions_dir = tmp_project / "data" / "sessions"
        events_archive = tmp_project / "archive" / "events"
        sessions_archive = tmp_project / "archive" / "sessions"

        old_event = _write_event_file(events_dir, "2026-01-01")
        _set_file_age(old_event, 10)

        old_session = _write_session_file(sessions_dir, "old-task")
        _set_file_age(old_session, 10)

        rotate_logs(
            events_dir,
            sessions_dir,
            rotate_after_days=7,
            events_archive=events_archive,
            sessions_archive=sessions_archive,
        )

        assert list(events_archive.rglob("*.jsonl"))
        assert list(sessions_archive.rglob("*.jsonl"))


# ===========================================================================
# Rotation scheduling (is_rotation_due) tests
# ===========================================================================


class TestIsRotationDue:
    def test_due_when_never_rotated(self, tmp_project):
        """Rotation is due when no last_rotation file exists."""
        assert is_rotation_due(tmp_project / ".corc") is True

    def test_not_due_within_daily_interval(self, tmp_project):
        """Rotation is not due if last rotation was less than 24h ago."""
        corc_dir = tmp_project / ".corc"
        recent = time.time() - 3600  # 1 hour ago
        (corc_dir / "last_rotation").write_text(str(recent))

        assert is_rotation_due(corc_dir) is False

    def test_due_after_daily_interval(self, tmp_project):
        """Rotation is due if last rotation was more than 24h ago."""
        corc_dir = tmp_project / ".corc"
        old = time.time() - (25 * 3600)  # 25 hours ago
        (corc_dir / "last_rotation").write_text(str(old))

        assert is_rotation_due(corc_dir) is True

    def test_invalid_last_rotation_file(self, tmp_project):
        """Rotation is due if last_rotation file is corrupt."""
        corc_dir = tmp_project / ".corc"
        (corc_dir / "last_rotation").write_text("not-a-number")

        assert is_rotation_due(corc_dir) is True


# ===========================================================================
# run_daily_rotation integration tests
# ===========================================================================


class TestRunDailyRotation:
    def test_full_rotation_cycle(self, tmp_project):
        """Full cycle: rotate old sessions and events, record time."""
        events_dir = tmp_project / "data" / "events"
        sessions_dir = tmp_project / "data" / "sessions"
        corc_dir = tmp_project / ".corc"

        # Config: rotate after 5 days
        (corc_dir / "config.yaml").write_text("rotation:\n  rotate_after_days: 5\n")

        # Create old and new files
        old_event = _write_event_file(events_dir, "2026-01-01")
        _set_file_age(old_event, 10)
        old_session = _write_session_file(sessions_dir, "old-task")
        _set_file_age(old_session, 10)

        new_event = _write_event_file(events_dir, "2026-03-22")
        new_session = _write_session_file(sessions_dir, "new-task")

        result = run_daily_rotation(corc_dir, events_dir, sessions_dir)

        assert result is not None
        assert result["events"]["moved"] == 1
        assert result["sessions"]["moved"] == 1

        # Old files archived, not deleted
        assert not old_event.exists()
        assert not old_session.exists()
        assert list((events_dir / "archive").rglob("*.jsonl"))
        assert list((sessions_dir / "archive").rglob("*.jsonl"))

        # New files untouched
        assert new_event.exists()
        assert new_session.exists()

        # Rotation time recorded
        assert get_last_rotation_time(corc_dir) is not None

    def test_skips_when_not_due(self, tmp_project):
        """Returns None when rotation is not due yet."""
        corc_dir = tmp_project / ".corc"

        # Mark rotation as done recently
        (corc_dir / "last_rotation").write_text(str(time.time()))

        result = run_daily_rotation(
            corc_dir,
            tmp_project / "data" / "events",
            tmp_project / "data" / "sessions",
        )
        assert result is None

    def test_records_rotation_time(self, tmp_project):
        """After rotation, last_rotation file is updated."""
        corc_dir = tmp_project / ".corc"

        before = time.time()
        run_daily_rotation(
            corc_dir,
            tmp_project / "data" / "events",
            tmp_project / "data" / "sessions",
        )
        after = time.time()

        last = get_last_rotation_time(corc_dir)
        assert last is not None
        assert before <= last <= after

    def test_uses_default_config(self, tmp_project):
        """Uses default 7-day rotation when no config exists."""
        events_dir = tmp_project / "data" / "events"
        sessions_dir = tmp_project / "data" / "sessions"
        corc_dir = tmp_project / ".corc"

        # Create a file 5 days old (should NOT be rotated with default 7)
        f5 = _write_event_file(events_dir, "2026-03-18")
        _set_file_age(f5, 5)

        # Create a file 10 days old (SHOULD be rotated with default 7)
        f10 = _write_event_file(events_dir, "2026-03-13")
        _set_file_age(f10, 10)

        result = run_daily_rotation(corc_dir, events_dir, sessions_dir)

        assert result is not None
        assert result["events"]["moved"] == 1
        assert f5.exists()  # 5-day-old stays
        assert not f10.exists()  # 10-day-old moved


# ===========================================================================
# Daemon integration tests
# ===========================================================================


class TestDaemonRotationIntegration:
    """Verify rotation runs as part of the daemon's tick cycle."""

    def test_daemon_tick_triggers_rotation(self, tmp_project):
        """Daemon _check_log_rotation calls run_daily_rotation."""
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

        # Create old session file that should be rotated
        sessions_dir = tmp_project / "data" / "sessions"
        old_session = _write_session_file(sessions_dir, "old-task")
        _set_file_age(old_session, 10)

        # Run rotation via daemon
        daemon._check_log_rotation()

        # Verify file was moved to archive
        assert not old_session.exists()
        archive_files = list((sessions_dir / "archive").rglob("*.jsonl"))
        assert len(archive_files) >= 1

        # Verify audit log recorded the rotation
        events = al.read_today()
        rotation_events = [
            e for e in events if e["event_type"] == "log_rotation_completed"
        ]
        assert len(rotation_events) == 1

    def test_daemon_tick_skips_when_not_due(self, tmp_project):
        """Daemon doesn't rotate when interval hasn't elapsed."""
        from unittest.mock import MagicMock

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

        # Mark rotation as recently done
        (tmp_project / ".corc" / "last_rotation").write_text(str(time.time()))

        daemon._check_log_rotation()

        # No rotation event should be logged
        events = al.read_today()
        rotation_events = [
            e for e in events if e["event_type"] == "log_rotation_completed"
        ]
        assert len(rotation_events) == 0

    def test_daemon_rotation_logs_error_on_failure(self, tmp_project):
        """Daemon logs rotation_failed when rotation raises OSError."""
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

        with patch(
            "corc.daemon.run_daily_rotation", side_effect=OSError("permission denied")
        ):
            daemon._check_log_rotation()

        events = al.read_today()
        fail_events = [e for e in events if e["event_type"] == "log_rotation_failed"]
        assert len(fail_events) == 1
        assert "permission denied" in fail_events[0]["error"]


# ===========================================================================
# Helper function tests
# ===========================================================================


class TestDateFromMtime:
    def test_returns_correct_date_string(self):
        """Converts mtime epoch to YYYY-MM-DD UTC string."""
        # Use a known epoch: 2020-01-01T00:00:00 UTC = 1577836800
        ts = 1577836800.0
        result = _date_from_mtime(ts)
        assert result == "2020-01-01"

    def test_handles_epoch_zero(self):
        """Handles epoch 0 (1970-01-01)."""
        result = _date_from_mtime(0.0)
        assert result == "1970-01-01"


class TestMoveOldFiles:
    def test_skips_archive_subdirectory_files(self, tmp_project):
        """Files already in archive subdirectories are not re-archived."""
        sessions_dir = tmp_project / "data" / "sessions"
        archive_dir = sessions_dir / "archive"

        # Create an archived file
        date_dir = archive_dir / "2026-01-01"
        date_dir.mkdir(parents=True)
        archived_file = date_dir / "old-task-attempt-1.jsonl"
        archived_file.write_text('{"type":"test"}\n')
        _set_file_age(archived_file, 100)

        # Create a source file that should be moved
        source_file = _write_session_file(sessions_dir, "task-new")
        _set_file_age(source_file, 10)

        result = _move_old_files(sessions_dir, archive_dir, rotate_after_days=7)

        # Only the source file should be moved, not the already-archived one
        assert result["moved"] == 1
        # The archived file should still be in its original location
        assert archived_file.exists()


# ===========================================================================
# CLI integration tests
# ===========================================================================


class TestCLILogsRotate:
    def test_cli_logs_rotate_invocable(self, tmp_project):
        """The `corc logs rotate` CLI command can be invoked."""
        from unittest.mock import patch

        from click.testing import CliRunner

        from corc.cli import cli

        runner = CliRunner()

        with patch(
            "corc.cli.get_paths",
            return_value={
                "root": tmp_project,
                "data_dir": tmp_project / "data",
                "mutations": tmp_project / "data" / "mutations.jsonl",
                "state_db": tmp_project / "data" / "state.db",
                "events_dir": tmp_project / "data" / "events",
                "sessions_dir": tmp_project / "data" / "sessions",
                "knowledge_dir": tmp_project / "knowledge",
                "knowledge_db": tmp_project / "data" / "knowledge.db",
                "corc_dir": tmp_project / ".corc",
                "ratings_dir": tmp_project / "data" / "ratings",
                "retry_outcomes": tmp_project / "data" / "retry_outcomes.jsonl",
            },
        ):
            # Create old files to rotate
            sessions_dir = tmp_project / "data" / "sessions"
            old_session = _write_session_file(sessions_dir, "old-task")
            _set_file_age(old_session, 10)

            result = runner.invoke(cli, ["logs", "rotate"])

            assert result.exit_code == 0
            assert (
                "session" in result.output.lower() or "moved" in result.output.lower()
            )

    def test_cli_logs_rotate_with_custom_days(self, tmp_project):
        """The `corc logs rotate --days` option works."""
        from unittest.mock import patch

        from click.testing import CliRunner

        from corc.cli import cli

        runner = CliRunner()

        with patch(
            "corc.cli.get_paths",
            return_value={
                "root": tmp_project,
                "data_dir": tmp_project / "data",
                "mutations": tmp_project / "data" / "mutations.jsonl",
                "state_db": tmp_project / "data" / "state.db",
                "events_dir": tmp_project / "data" / "events",
                "sessions_dir": tmp_project / "data" / "sessions",
                "knowledge_dir": tmp_project / "knowledge",
                "knowledge_db": tmp_project / "data" / "knowledge.db",
                "corc_dir": tmp_project / ".corc",
                "ratings_dir": tmp_project / "data" / "ratings",
                "retry_outcomes": tmp_project / "data" / "retry_outcomes.jsonl",
            },
        ):
            sessions_dir = tmp_project / "data" / "sessions"

            # 5-day-old file should NOT be rotated with --days=7
            f5 = _write_session_file(sessions_dir, "task-5")
            _set_file_age(f5, 5)

            # But SHOULD be rotated with --days=3
            result = runner.invoke(cli, ["logs", "rotate", "--days", "3"])

            assert result.exit_code == 0
            assert not f5.exists()
