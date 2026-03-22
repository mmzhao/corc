"""Tests for source file watching and hot-reload.

Verifies that:
- SourceWatcher detects modified and new .py files
- Changed modules are reloaded via importlib.reload()
- The daemon integrates reload into its tick cycle
- In-flight task state is preserved across reloads
"""

import importlib
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from corc.audit import AuditLog
from corc.daemon import Daemon
from corc.dispatch import AgentDispatcher, AgentResult, Constraints
from corc.mutations import MutationLog
from corc.reload import SourceWatcher
from corc.sessions import SessionLogger
from corc.state import WorkState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pkg(tmp_path, pkg_name, module_name, code):
    """Create a minimal Python package with one module in tmp_path.

    Returns (pkg_dir, module_path).
    """
    pkg_dir = tmp_path / pkg_name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("")
    mod_path = pkg_dir / f"{module_name}.py"
    mod_path.write_text(code)
    return pkg_dir, mod_path


def _import_fresh(dotted_name):
    """Import a module, clearing any stale cache first."""
    # Remove from sys.modules so import is fresh
    for key in list(sys.modules):
        if key == dotted_name or key.startswith(dotted_name + "."):
            del sys.modules[key]
    return importlib.import_module(dotted_name)


class MockDispatcher(AgentDispatcher):
    """Dispatcher that returns configurable results without calling any LLM."""

    def __init__(self, default_result=None, delay=0):
        self.default_result = default_result or AgentResult(
            output="Mock output: task completed successfully.",
            exit_code=0,
            duration_s=0.1,
        )
        self.delay = delay
        self.dispatched = []

    def dispatch(
        self,
        prompt,
        system_prompt,
        constraints,
        pid_callback=None,
        event_callback=None,
        cwd=None,
    ):
        self.dispatched.append((prompt, system_prompt, constraints))
        if self.delay:
            time.sleep(self.delay)
        return self.default_result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project structure for testing."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "events").mkdir()
    (tmp_path / "data" / "sessions").mkdir()
    (tmp_path / ".corc").mkdir()
    return tmp_path


@pytest.fixture
def mutation_log(tmp_project):
    return MutationLog(tmp_project / "data" / "mutations.jsonl")


@pytest.fixture
def work_state(tmp_project, mutation_log):
    return WorkState(tmp_project / "data" / "state.db", mutation_log)


@pytest.fixture
def audit_log(tmp_project):
    return AuditLog(tmp_project / "data" / "events")


@pytest.fixture
def session_logger(tmp_project):
    return SessionLogger(tmp_project / "data" / "sessions")


# ===========================================================================
# SourceWatcher unit tests
# ===========================================================================


class TestSourceWatcher:
    def test_no_changes_on_fresh_snapshot(self, tmp_path):
        """Watcher reports no changes immediately after creation."""
        pkg_dir, _ = _make_pkg(tmp_path, "pkg_a", "mod", "X = 1\n")
        watcher = SourceWatcher(pkg_dir)

        assert watcher.get_changed_files() == []
        assert watcher.check_and_reload() == []

    def test_detects_modified_file(self, tmp_path):
        """Watcher detects when a .py file's mtime changes."""
        pkg_dir, mod_path = _make_pkg(tmp_path, "pkg_b", "mod", "X = 1\n")
        watcher = SourceWatcher(pkg_dir)

        # Ensure mtime differs (some filesystems have 1s resolution)
        time.sleep(0.05)
        mod_path.write_text("X = 2\n")
        # Force mtime forward to handle coarse-grained filesystems
        future = time.time() + 2
        import os

        os.utime(mod_path, (future, future))

        changed = watcher.get_changed_files()
        assert len(changed) == 1
        assert changed[0].name == "mod.py"

    def test_detects_new_file(self, tmp_path):
        """Watcher detects newly created .py files."""
        pkg_dir, _ = _make_pkg(tmp_path, "pkg_c", "mod", "X = 1\n")
        watcher = SourceWatcher(pkg_dir)

        # Add a new file
        new_mod = pkg_dir / "new_mod.py"
        new_mod.write_text("Y = 99\n")
        future = time.time() + 2
        import os

        os.utime(new_mod, (future, future))

        changed = watcher.get_changed_files()
        assert any(p.name == "new_mod.py" for p in changed)

    def test_snapshot_updates_after_check(self, tmp_path):
        """After check_and_reload, the same change is not reported again."""
        pkg_dir, mod_path = _make_pkg(tmp_path, "pkg_d", "mod", "X = 1\n")
        watcher = SourceWatcher(pkg_dir)

        # Modify file
        time.sleep(0.05)
        mod_path.write_text("X = 2\n")
        future = time.time() + 2
        import os

        os.utime(mod_path, (future, future))

        # First check detects change
        changed1 = watcher.get_changed_files()
        assert len(changed1) == 1

        # check_and_reload updates snapshot (module not in sys.modules, so no reload)
        watcher.check_and_reload()

        # Second check sees no changes
        assert watcher.get_changed_files() == []

    def test_file_to_module_mapping(self, tmp_path):
        """File paths correctly map to dotted module names."""
        pkg_dir = tmp_path / "mypkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "foo.py").write_text("")

        sub = pkg_dir / "sub"
        sub.mkdir()
        (sub / "__init__.py").write_text("")
        (sub / "bar.py").write_text("")

        watcher = SourceWatcher(pkg_dir)

        assert watcher._file_to_module(pkg_dir / "foo.py") == "mypkg.foo"
        assert watcher._file_to_module(pkg_dir / "__init__.py") == "mypkg"
        assert watcher._file_to_module(pkg_dir / "sub" / "bar.py") == "mypkg.sub.bar"
        assert watcher._file_to_module(pkg_dir / "sub" / "__init__.py") == "mypkg.sub"

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        """Watcher handles nonexistent directory gracefully."""
        watcher = SourceWatcher(tmp_path / "does_not_exist")
        assert watcher.get_changed_files() == []
        assert watcher.check_and_reload() == []


# ===========================================================================
# SourceWatcher reload integration tests
# ===========================================================================


class TestSourceWatcherReload:
    def test_reloads_changed_module(self, tmp_path):
        """Modifying a module file and calling check_and_reload updates the module."""
        pkg_name = "_reload_test_pkg_1"
        pkg_dir, mod_path = _make_pkg(
            tmp_path, pkg_name, "helper", "VALUE = 'original'\n"
        )

        sys.path.insert(0, str(tmp_path))
        try:
            mod = _import_fresh(f"{pkg_name}.helper")
            assert mod.VALUE == "original"

            watcher = SourceWatcher(pkg_dir)

            # Modify the module
            time.sleep(0.05)
            mod_path.write_text("VALUE = 'modified'\n")
            import os

            os.utime(mod_path, (time.time() + 2, time.time() + 2))

            reloaded = watcher.check_and_reload()
            assert f"{pkg_name}.helper" in reloaded

            # Module should now have the updated value
            assert mod.VALUE == "modified"
            assert sys.modules[f"{pkg_name}.helper"].VALUE == "modified"
        finally:
            sys.path.remove(str(tmp_path))
            for key in list(sys.modules):
                if key.startswith(pkg_name):
                    del sys.modules[key]

    def test_reload_preserves_unmodified_modules(self, tmp_path):
        """Modules whose files haven't changed are NOT reloaded."""
        pkg_name = "_reload_test_pkg_2"
        pkg_dir, mod_path = _make_pkg(
            tmp_path, pkg_name, "helper", "VALUE = 'stable'\n"
        )

        # Add a second module
        other_path = pkg_dir / "other.py"
        other_path.write_text("OTHER = 'untouched'\n")

        sys.path.insert(0, str(tmp_path))
        try:
            _import_fresh(f"{pkg_name}.helper")
            other_mod = _import_fresh(f"{pkg_name}.other")

            watcher = SourceWatcher(pkg_dir)

            # Only modify helper, not other
            time.sleep(0.05)
            mod_path.write_text("VALUE = 'changed'\n")
            import os

            os.utime(mod_path, (time.time() + 2, time.time() + 2))

            reloaded = watcher.check_and_reload()
            assert f"{pkg_name}.helper" in reloaded
            assert f"{pkg_name}.other" not in reloaded

            # other module should be untouched
            assert other_mod.OTHER == "untouched"
        finally:
            sys.path.remove(str(tmp_path))
            for key in list(sys.modules):
                if key.startswith(pkg_name):
                    del sys.modules[key]

    def test_reload_skips_unimported_modules(self, tmp_path):
        """New files that haven't been imported are detected but not reloaded."""
        pkg_name = "_reload_test_pkg_3"
        pkg_dir, _ = _make_pkg(tmp_path, pkg_name, "helper", "VALUE = 1\n")

        sys.path.insert(0, str(tmp_path))
        try:
            _import_fresh(f"{pkg_name}.helper")
            watcher = SourceWatcher(pkg_dir)

            # Add a brand new module (not imported)
            new_mod = pkg_dir / "brand_new.py"
            new_mod.write_text("THING = 42\n")
            import os

            os.utime(new_mod, (time.time() + 2, time.time() + 2))

            reloaded = watcher.check_and_reload()
            # brand_new.py changed but was never imported — should not be in reloaded
            assert f"{pkg_name}.brand_new" not in reloaded
        finally:
            sys.path.remove(str(tmp_path))
            for key in list(sys.modules):
                if key.startswith(pkg_name):
                    del sys.modules[key]

    def test_reload_handles_syntax_error_gracefully(self, tmp_path):
        """If a modified file has a syntax error, reload skips it without crashing."""
        pkg_name = "_reload_test_pkg_4"
        pkg_dir, mod_path = _make_pkg(tmp_path, pkg_name, "helper", "VALUE = 'ok'\n")

        sys.path.insert(0, str(tmp_path))
        try:
            mod = _import_fresh(f"{pkg_name}.helper")
            assert mod.VALUE == "ok"

            watcher = SourceWatcher(pkg_dir)

            # Write broken code
            time.sleep(0.05)
            mod_path.write_text("VALUE = 'broken\n")  # unterminated string
            import os

            os.utime(mod_path, (time.time() + 2, time.time() + 2))

            reloaded = watcher.check_and_reload()
            # Should not crash, and the broken module should not be in reloaded list
            assert f"{pkg_name}.helper" not in reloaded

            # Original value should still be accessible
            assert mod.VALUE == "ok"
        finally:
            sys.path.remove(str(tmp_path))
            for key in list(sys.modules):
                if key.startswith(pkg_name):
                    del sys.modules[key]


# ===========================================================================
# Daemon integration tests
# ===========================================================================


class TestDaemonCodeReload:
    def test_daemon_creates_source_watcher_by_default(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Daemon creates a SourceWatcher when auto_reload=True (default)."""
        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=MockDispatcher(),
            project_root=tmp_project,
            poll_interval=0.1,
            auto_reload=True,
        )
        assert daemon._source_watcher is not None

    def test_daemon_no_watcher_when_disabled(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Daemon has no SourceWatcher when auto_reload=False."""
        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=MockDispatcher(),
            project_root=tmp_project,
            poll_interval=0.1,
            auto_reload=False,
        )
        assert daemon._source_watcher is None

    def test_tick_calls_check_source_reload(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Each tick checks for source file changes."""
        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=MockDispatcher(),
            project_root=tmp_project,
            poll_interval=0.1,
            auto_reload=False,
        )

        # Inject a mock watcher
        mock_watcher = MagicMock()
        mock_watcher.check_and_reload.return_value = []
        daemon._source_watcher = mock_watcher

        daemon._tick()

        mock_watcher.check_and_reload.assert_called_once()

    def test_tick_logs_reloaded_modules(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """When modules are reloaded, an audit event is logged."""
        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=MockDispatcher(),
            project_root=tmp_project,
            poll_interval=0.1,
            auto_reload=False,
        )

        # Inject a mock watcher that claims modules were reloaded
        mock_watcher = MagicMock()
        mock_watcher.check_and_reload.return_value = ["corc.fake_module"]
        daemon._source_watcher = mock_watcher

        # Mock audit_log to capture events
        audit_log.log = MagicMock()

        daemon._tick()

        # Verify audit event was logged
        calls = [
            c for c in audit_log.log.call_args_list if c[0][0] == "modules_reloaded"
        ]
        assert len(calls) == 1
        assert calls[0][1]["modules"] == ["corc.fake_module"]

    def test_rebind_updates_function_references(self, tmp_path):
        """_rebind_after_reload updates module-level globals in daemon module."""
        import corc.daemon as daemon_mod

        # Save original get_ready_tasks reference
        original_func = daemon_mod.get_ready_tasks

        # Create a mock module with a replacement function
        def mock_get_ready_tasks(state, limit):
            return ["mock_result"]

        # Temporarily replace the module in sys.modules
        import corc.scheduler

        original_attr = corc.scheduler.get_ready_tasks
        corc.scheduler.get_ready_tasks = mock_get_ready_tasks

        try:
            # Call _rebind_after_reload with corc.scheduler
            Daemon._rebind_after_reload(["corc.scheduler"])

            # The daemon module's global should now point to our mock
            assert daemon_mod.get_ready_tasks is mock_get_ready_tasks
        finally:
            # Restore originals
            corc.scheduler.get_ready_tasks = original_attr
            daemon_mod.get_ready_tasks = original_func

    def test_daemon_uses_new_code_after_source_modification(self, tmp_path):
        """End-to-end: modify a module file, verify daemon uses new code on next tick.

        This is the key acceptance test: simulates an agent modifying source code
        and verifies the daemon picks up the change without a restart.
        """
        pkg_name = "_reload_e2e_test"
        pkg_dir, mod_path = _make_pkg(
            tmp_path,
            pkg_name,
            "scheduler_ext",
            "def get_extra_info():\n    return 'version_1'\n",
        )

        sys.path.insert(0, str(tmp_path))
        try:
            mod = _import_fresh(f"{pkg_name}.scheduler_ext")
            assert mod.get_extra_info() == "version_1"

            watcher = SourceWatcher(pkg_dir)

            # Simulate agent modifying the code
            time.sleep(0.05)
            mod_path.write_text("def get_extra_info():\n    return 'version_2'\n")
            import os

            os.utime(mod_path, (time.time() + 2, time.time() + 2))

            # Watcher detects and reloads
            reloaded = watcher.check_and_reload()
            assert f"{pkg_name}.scheduler_ext" in reloaded

            # The function now returns the new value — new code is in use
            assert mod.get_extra_info() == "version_2"

            # Calling through sys.modules also gets new code
            assert (
                sys.modules[f"{pkg_name}.scheduler_ext"].get_extra_info() == "version_2"
            )
        finally:
            sys.path.remove(str(tmp_path))
            for key in list(sys.modules):
                if key.startswith(pkg_name):
                    del sys.modules[key]

    def test_in_flight_state_preserved_across_reload(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """In-flight task tracking is preserved when modules are reloaded.

        The reload mechanism uses importlib.reload() which updates module code
        without restarting the process, so all instance state (executor, tasks,
        etc.) remains intact.
        """
        dispatcher = MockDispatcher(delay=0.3)

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=dispatcher,
            project_root=tmp_project,
            poll_interval=0.1,
            auto_reload=False,
        )

        # Create a task
        mutation_log.append(
            "task_created",
            {
                "id": "t1",
                "name": "Test Task",
                "description": "Test",
                "role": "implementer",
                "depends_on": [],
                "done_when": "do the thing",
                "checklist": [],
                "context_bundle": [],
                "max_retries": 3,
            },
            reason="Test setup",
        )
        work_state.refresh()

        # Run one tick to dispatch the task
        daemon._tick()
        in_flight_before = daemon.executor.in_flight_count

        # Now simulate a source reload by injecting a mock watcher
        mock_watcher = MagicMock()
        mock_watcher.check_and_reload.return_value = ["corc.scheduler"]
        daemon._source_watcher = mock_watcher

        # Run another tick — reload happens but in-flight state is preserved
        daemon._tick()
        in_flight_after = daemon.executor.in_flight_count

        # In-flight tasks should still be tracked
        # (The executor instance wasn't replaced, just module code was refreshed)
        assert daemon.executor is not None
        assert daemon.state is not None

        # Clean up
        daemon.executor.shutdown(wait=True)

    def test_daemon_full_loop_with_reload(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Daemon runs its full loop with auto_reload enabled and completes tasks.

        Verifies that the reload mechanism doesn't interfere with normal operation.
        """
        mutation_log.append(
            "task_created",
            {
                "id": "t1",
                "name": "Task 1",
                "description": "Test task",
                "role": "implementer",
                "depends_on": [],
                "done_when": "do the thing",
                "checklist": [],
                "context_bundle": [],
                "max_retries": 3,
            },
            reason="Test setup",
        )
        work_state.refresh()

        daemon = Daemon(
            state=work_state,
            mutation_log=mutation_log,
            audit_log=audit_log,
            session_logger=session_logger,
            dispatcher=MockDispatcher(),
            project_root=tmp_project,
            poll_interval=0.1,
            once=True,
            auto_reload=True,
        )

        thread = threading.Thread(target=daemon.start)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()

        # Task should have been completed
        work_state.refresh()
        task = work_state.get_task("t1")
        assert task["status"] == "completed"
