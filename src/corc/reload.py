"""Source file watcher — detects code changes and reloads modules.

The daemon caches Python modules at startup. When agents modify the codebase,
this module detects changes by comparing file modification times and uses
importlib.reload() to pick up new code without restarting the process.

This preserves all in-flight state (running tasks, executor handles, etc.)
while ensuring the daemon uses the latest code on the next dispatch cycle.
"""

import importlib
import sys
from pathlib import Path


class SourceWatcher:
    """Tracks modification times of Python source files and reloads changed modules.

    Watches a package directory (e.g., src/corc/) for .py file changes.
    When check_and_reload() finds changed files, it reloads the corresponding
    modules via importlib.reload() and returns the list of reloaded module names.

    Usage:
        watcher = SourceWatcher(Path("src/corc"))
        # In a loop:
        reloaded = watcher.check_and_reload()
        if reloaded:
            # Rebind any cached function references...
    """

    def __init__(self, watch_dir: Path):
        """Initialize watcher for .py files under watch_dir.

        Takes an initial snapshot of all .py file modification times.

        Args:
            watch_dir: Directory to watch (e.g., the corc package directory).
                       Its parent is assumed to be the package root for
                       module name resolution.
        """
        self._watch_dir = Path(watch_dir)
        self._mtimes: dict[str, float] = self._take_snapshot()

    def _take_snapshot(self) -> dict[str, float]:
        """Return current mtimes of all .py files in watch directory.

        Returns:
            Dict mapping absolute file path strings to their mtime.
        """
        mtimes: dict[str, float] = {}
        if not self._watch_dir.exists():
            return mtimes
        for py_file in self._watch_dir.rglob("*.py"):
            try:
                mtimes[str(py_file)] = py_file.stat().st_mtime
            except OSError:
                pass
        return mtimes

    def get_changed_files(self) -> list[Path]:
        """Return list of .py files that have changed since last snapshot.

        Detects both modified existing files and newly created files.
        Does NOT update the snapshot — call check_and_reload() for that.

        Returns:
            List of Path objects for changed/new .py files.
        """
        if not self._watch_dir.exists():
            return []

        changed = []
        for py_file in self._watch_dir.rglob("*.py"):
            path_str = str(py_file)
            try:
                mtime = py_file.stat().st_mtime
                if path_str not in self._mtimes or self._mtimes[path_str] != mtime:
                    changed.append(py_file)
            except OSError:
                pass
        return changed

    def check_and_reload(self) -> list[str]:
        """Check for changed .py files and reload their modules.

        Compares current file mtimes against the stored snapshot.
        For each changed file, maps it to a Python module name and
        reloads that module via importlib.reload() if it's currently loaded.

        Always updates the snapshot after checking, so a file is only
        detected as changed once (until it changes again).

        Returns:
            List of reloaded module names (e.g., ['corc.scheduler']).
            Empty list if nothing changed.
        """
        changed_files = self.get_changed_files()

        # Always update snapshot
        new_mtimes = self._take_snapshot()

        if not changed_files:
            self._mtimes = new_mtimes
            return []

        # Map changed files to module names and reload
        reloaded = []
        for py_file in changed_files:
            mod_name = self._file_to_module(py_file)
            if mod_name and mod_name in sys.modules:
                try:
                    importlib.reload(sys.modules[mod_name])
                    reloaded.append(mod_name)
                except ImportError:
                    # Module can't be reloaded (e.g., missing dependency)
                    pass
                except SyntaxError:
                    # File has syntax errors — skip, will retry next check
                    pass

        self._mtimes = new_mtimes
        return reloaded

    def _file_to_module(self, filepath: Path) -> str | None:
        """Convert a file path under watch_dir to a dotted module name.

        The watch_dir's parent is treated as the package root. For example:
          watch_dir = /path/src/corc
          filepath  = /path/src/corc/scheduler.py
          result    = "corc.scheduler"

        Args:
            filepath: Absolute path to a .py file under watch_dir.

        Returns:
            Dotted module name, or None if path can't be converted.
        """
        try:
            rel = filepath.relative_to(self._watch_dir.parent)
            parts = list(rel.with_suffix("").parts)
            if parts and parts[-1] == "__init__":
                parts = parts[:-1]
            return ".".join(parts) if parts else None
        except ValueError:
            return None
