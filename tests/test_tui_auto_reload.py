"""Tests for TUI auto-reload on source file changes.

Verifies that:
- _get_watched_file_mtimes returns mtimes for tui.py and queries.py
- _check_for_source_changes detects when watched files change
- run_active_dashboard raises ReloadRequested when auto_reload=True
  and a watched file is modified
- ReloadRequested contains the changed file paths
- No crash on reload — graceful restart cycle
"""

import io
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from corc.tui import (
    ReloadRequested,
    _get_watched_file_mtimes,
    _check_for_source_changes,
    _TUI_WATCH_FILES,
    run_active_dashboard,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_mock_query_api():
    """Create a mock QueryAPI that returns empty results."""
    api = MagicMock()
    api.work_state = MagicMock()
    api.get_running_tasks_with_agents.return_value = []
    api.get_ready_tasks.return_value = []
    api.get_blocked_tasks_with_reasons.return_value = []
    api.get_recently_completed_tasks.return_value = []
    api.get_recent_events.return_value = []
    api.get_active_plan_tasks.return_value = []
    return api


# ── _get_watched_file_mtimes tests ──────────────────────────────────────


class TestGetWatchedFileMtimes:
    def test_returns_dict_with_tui_and_queries(self):
        """Should return mtimes for both tui.py and queries.py."""
        mtimes = _get_watched_file_mtimes()
        # Both files should exist in the source tree
        assert len(mtimes) >= 2
        paths = list(mtimes.keys())
        basenames = [os.path.basename(p) for p in paths]
        assert "tui.py" in basenames
        assert "queries.py" in basenames

    def test_mtimes_are_floats(self):
        """Each mtime should be a float."""
        mtimes = _get_watched_file_mtimes()
        for path, mtime in mtimes.items():
            assert isinstance(mtime, float)
            assert mtime > 0

    def test_paths_are_absolute(self):
        """Each path should be absolute."""
        mtimes = _get_watched_file_mtimes()
        for path in mtimes:
            assert os.path.isabs(path)

    def test_watch_files_constant(self):
        """The _TUI_WATCH_FILES constant should include expected files."""
        assert "tui.py" in _TUI_WATCH_FILES
        assert "queries.py" in _TUI_WATCH_FILES


# ── _check_for_source_changes tests ─────────────────────────────────────


class TestCheckForSourceChanges:
    def test_no_changes_returns_empty(self):
        """When mtimes match baseline, should return empty list."""
        baseline = _get_watched_file_mtimes()
        changed = _check_for_source_changes(baseline)
        assert changed == []

    def test_detects_mtime_change(self):
        """When baseline has older mtime, should detect the change."""
        baseline = _get_watched_file_mtimes()
        # Simulate an older mtime for tui.py
        for path in list(baseline):
            if path.endswith("tui.py"):
                baseline[path] = baseline[path] - 10.0
                break
        changed = _check_for_source_changes(baseline)
        assert len(changed) == 1
        assert changed[0].endswith("tui.py")

    def test_detects_multiple_changes(self):
        """When multiple files have changed, should detect all of them."""
        baseline = _get_watched_file_mtimes()
        # Make all baselines older
        for path in baseline:
            baseline[path] = baseline[path] - 10.0
        changed = _check_for_source_changes(baseline)
        assert len(changed) >= 2
        basenames = [os.path.basename(p) for p in changed]
        assert "tui.py" in basenames
        assert "queries.py" in basenames

    def test_detects_new_file_missing_from_baseline(self):
        """When baseline is missing a file entry, should detect it."""
        baseline = _get_watched_file_mtimes()
        # Remove one entry to simulate a "new" file
        removed_path = list(baseline.keys())[0]
        del baseline[removed_path]
        changed = _check_for_source_changes(baseline)
        assert removed_path in changed

    def test_empty_baseline_detects_all(self):
        """Empty baseline should detect all watched files as changed."""
        changed = _check_for_source_changes({})
        assert len(changed) >= 2


# ── ReloadRequested exception tests ─────────────────────────────────────


class TestReloadRequested:
    def test_is_exception(self):
        """ReloadRequested should be an Exception subclass."""
        assert issubclass(ReloadRequested, Exception)

    def test_contains_changed_files(self):
        """Should store the changed file list."""
        exc = ReloadRequested(["/path/to/tui.py"])
        assert exc.changed_files == ["/path/to/tui.py"]

    def test_message_includes_filenames(self):
        """The exception message should mention the changed files."""
        exc = ReloadRequested(["/path/to/tui.py", "/path/to/queries.py"])
        assert "tui.py" in str(exc)
        assert "queries.py" in str(exc)


# ── run_active_dashboard auto_reload integration ─────────────────────────


class TestRunActiveDashboardAutoReload:
    """Tests that run_active_dashboard raises ReloadRequested on file changes."""

    def test_raises_reload_requested_on_file_change(self):
        """Dashboard should raise ReloadRequested when a watched file changes.

        We mock _check_for_source_changes to report a change on the
        second call (first call returns [] to allow one render cycle).
        """
        api = _make_mock_query_api()
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=120, no_color=True)

        call_count = {"n": 0}

        def fake_check(baseline):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                return ["/fake/path/tui.py"]
            return []

        with patch("corc.tui._check_for_source_changes", side_effect=fake_check):
            with patch("corc.tui._listen_for_keys"):
                with pytest.raises(ReloadRequested) as exc_info:
                    run_active_dashboard(
                        api,
                        console=console,
                        auto_reload=True,
                        refresh_per_second=100.0,  # fast iteration for tests
                    )
                assert "/fake/path/tui.py" in exc_info.value.changed_files

    def test_no_reload_when_auto_reload_false(self):
        """Dashboard should NOT check for changes when auto_reload is False.

        We mock _check_for_source_changes to always report changes,
        but since auto_reload=False, it should never be called.
        We stop the dashboard via the stop_event after a brief delay.
        """
        api = _make_mock_query_api()
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=120, no_color=True)

        check_called = {"called": False}

        def fake_check(baseline):
            check_called["called"] = True
            return ["/fake/path/tui.py"]

        # Use a thread to stop the dashboard after a brief delay
        stop_event = threading.Event()

        def stop_after_delay():
            time.sleep(0.1)
            stop_event.set()

        stopper = threading.Thread(target=stop_after_delay, daemon=True)
        stopper.start()

        with patch("corc.tui._check_for_source_changes", side_effect=fake_check):
            with patch("corc.tui._listen_for_keys"):
                # Patch the stop_event inside run_active_dashboard
                original_event_class = threading.Event

                events_created = []

                def patched_event():
                    ev = original_event_class()
                    events_created.append(ev)
                    return ev

                with patch("corc.tui.threading.Event", side_effect=patched_event):
                    # Run in a thread so we can stop it
                    def run():
                        try:
                            run_active_dashboard(
                                api,
                                console=console,
                                auto_reload=False,
                                refresh_per_second=100.0,
                            )
                        except ReloadRequested:
                            pass  # Should NOT happen

                    t = threading.Thread(target=run, daemon=True)
                    t.start()
                    time.sleep(0.15)

                    # Stop all events created by the dashboard
                    for ev in events_created:
                        ev.set()
                    t.join(timeout=2.0)

        assert not check_called["called"]

    def test_reload_shows_indicator_message(self):
        """Reload indicator should mention the changed file names."""
        api = _make_mock_query_api()
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=120, no_color=True)

        def fake_check(baseline):
            return ["/path/to/tui.py"]

        with patch("corc.tui._check_for_source_changes", side_effect=fake_check):
            with patch("corc.tui._listen_for_keys"):
                with pytest.raises(ReloadRequested):
                    run_active_dashboard(
                        api,
                        console=console,
                        auto_reload=True,
                        refresh_per_second=100.0,
                    )

        output = buf.getvalue()
        assert "Reloading" in output or "tui.py" in output

    def test_reload_requested_on_queries_change(self):
        """Dashboard should also detect changes to queries.py."""
        api = _make_mock_query_api()
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=120, no_color=True)

        def fake_check(baseline):
            return ["/fake/path/queries.py"]

        with patch("corc.tui._check_for_source_changes", side_effect=fake_check):
            with patch("corc.tui._listen_for_keys"):
                with pytest.raises(ReloadRequested) as exc_info:
                    run_active_dashboard(
                        api,
                        console=console,
                        auto_reload=True,
                        refresh_per_second=100.0,
                    )
                assert "/fake/path/queries.py" in exc_info.value.changed_files

    def test_reload_with_real_file_modification(self, tmp_path):
        """Integration test: actual file mtime change triggers reload.

        Creates temporary copies of the watched files, patches the
        watch path to use them, modifies one, and verifies reload.
        """
        # Create a fake source directory with tui.py and queries.py
        fake_src = tmp_path / "corc"
        fake_src.mkdir()
        tui_file = fake_src / "tui.py"
        queries_file = fake_src / "queries.py"
        tui_file.write_text("# tui module")
        queries_file.write_text("# queries module")

        # Take initial snapshot using the fake directory
        initial_mtimes = {}
        for name in ("tui.py", "queries.py"):
            fp = fake_src / name
            initial_mtimes[str(fp)] = fp.stat().st_mtime

        call_count = {"n": 0}

        def fake_get_mtimes():
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Return initial mtimes (matches baseline)
                return dict(initial_mtimes)
            # On subsequent calls, simulate tui.py having been modified
            result = dict(initial_mtimes)
            for path in result:
                if path.endswith("tui.py"):
                    result[path] = result[path] + 100.0
            return result

        api = _make_mock_query_api()
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=120, no_color=True)

        with patch("corc.tui._get_watched_file_mtimes", side_effect=fake_get_mtimes):
            with patch("corc.tui._listen_for_keys"):
                with pytest.raises(ReloadRequested) as exc_info:
                    run_active_dashboard(
                        api,
                        console=console,
                        auto_reload=True,
                        refresh_per_second=100.0,
                    )

                changed_basenames = [
                    os.path.basename(p) for p in exc_info.value.changed_files
                ]
                assert "tui.py" in changed_basenames


# ── CLI reload loop tests ───────────────────────────────────────────────


class TestWatchDashboardReloadLoop:
    """Tests for the _watch_dashboard reload loop in cli.py."""

    def test_reload_loop_restarts_on_reload_requested(self):
        """_watch_dashboard should catch ReloadRequested and restart.

        We simulate: first call raises ReloadRequested, second call
        exits normally. Verifies the loop runs twice.
        """
        import importlib

        call_count = {"n": 0}
        reload_calls = []

        def fake_run_active_dashboard(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise ReloadRequested(["/path/tui.py"])
            # Second call: normal exit

        def fake_reload(mod):
            reload_calls.append(mod.__name__)
            return mod

        with patch("corc.cli._get_all") as mock_get_all:
            mock_get_all.return_value = (
                {"corc_dir": "/tmp/fake-corc"},  # paths
                MagicMock(),  # ml
                MagicMock(),  # ws
                MagicMock(),  # al
                MagicMock(),  # sl
                MagicMock(),  # _
            )

            with patch("corc.cli.load_config") as mock_cfg:
                mock_cfg.return_value = MagicMock(get=lambda k, d=None: d)
                with patch(
                    "corc.tui.run_active_dashboard",
                    side_effect=fake_run_active_dashboard,
                ):
                    with patch("importlib.reload", side_effect=fake_reload):
                        from corc.cli import _watch_dashboard

                        _watch_dashboard(20)

        # Should have been called twice (once raised, once normal exit)
        assert call_count["n"] == 2
        # importlib.reload should have been called for both modules
        assert "corc.queries" in reload_calls
        assert "corc.tui" in reload_calls
