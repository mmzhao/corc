"""Daemon — thin event loop connecting scheduler, executor, processor.

The daemon is a minimal polling loop that delegates to three independent modules.
Each module reads from and writes to the work state — they do not call each other.

Loop: scheduler → executor → processor, every poll_interval seconds.
"""

import os
import signal
import time
from pathlib import Path

from corc.audit import AuditLog
from corc.dispatch import AgentDispatcher
from corc.executor import Executor
from corc.mutations import MutationLog
from corc.pause import is_paused
from corc.processor import process_completed
from corc.scheduler import get_ready_tasks
from corc.sessions import SessionLogger
from corc.state import WorkState


class Daemon:
    """The CORC daemon — a thin event loop.

    Connects scheduler, executor, and processor through work state.
    Designed to run in a terminal tab and be left alone.
    """

    def __init__(
        self,
        state: WorkState,
        mutation_log: MutationLog,
        audit_log: AuditLog,
        session_logger: SessionLogger,
        dispatcher: AgentDispatcher,
        project_root: Path,
        parallel: int = 1,
        poll_interval: float = 5.0,
        task_id: str | None = None,
        once: bool = False,
    ):
        self.state = state
        self.mutation_log = mutation_log
        self.audit_log = audit_log
        self.session_logger = session_logger
        self.project_root = Path(project_root)
        self.parallel = parallel
        self.poll_interval = poll_interval
        self.target_task_id = task_id
        self.once = once
        self._running = False
        self._pid_file = self.project_root / ".corc" / "daemon.pid"
        self._tasks_completed = 0

        self.executor = Executor(
            dispatcher=dispatcher,
            mutation_log=mutation_log,
            state=state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=project_root,
            max_workers=parallel,
        )

    def start(self):
        """Run the daemon loop until stopped or task complete."""
        self._running = True
        self._write_pid()
        self._setup_signals()

        self.audit_log.log("daemon_started", parallel=self.parallel)

        try:
            while self._running:
                self._tick()

                # Check exit conditions for --once and --task modes
                if self._should_stop():
                    break

                # Sleep in small increments to respond to signals quickly
                self._interruptible_sleep(self.poll_interval)
        finally:
            self._cleanup()

    def _tick(self):
        """One iteration of the daemon loop: schedule → execute → process."""
        # Refresh state to pick up new tasks/mutations
        self.state.refresh()

        # Check for pause lock — skip scheduling new tasks but still process in-flight
        paused = is_paused(self.project_root / ".corc")

        if not paused:
            # 1. Scheduler: find ready tasks
            if self.target_task_id:
                ready = self._get_target_task()
            else:
                ready = get_ready_tasks(self.state, self.parallel)

            # 2. Executor: dispatch ready tasks
            for task in ready:
                self.executor.dispatch(task)

        # 3. Executor: poll for completed dispatches (always — in-flight tasks finish)
        completed = self.executor.poll_completed()

        # 4. Processor: validate and update state
        for item in completed:
            process_completed(
                task=item.task,
                result=item.result,
                attempt=item.attempt,
                mutation_log=self.mutation_log,
                state=self.state,
                audit_log=self.audit_log,
                session_logger=self.session_logger,
                project_root=self.project_root,
            )
            self._tasks_completed += 1

    def _get_target_task(self) -> list[dict]:
        """Get the specific target task if it's ready for dispatch."""
        task = self.state.get_task(self.target_task_id)
        if task and task["status"] == "pending":
            return [task]
        return []

    def _should_stop(self) -> bool:
        """Check if daemon should exit (--once or --task modes)."""
        if self.once and self._tasks_completed > 0:
            return True
        if self.target_task_id and self._tasks_completed > 0:
            # Check if the target task is done
            task = self.state.get_task(self.target_task_id)
            if task and task["status"] in ("completed", "failed"):
                return True
        return False

    def stop(self):
        """Signal the daemon to stop gracefully."""
        self._running = False

    def _write_pid(self):
        """Write PID file for corc stop to find us."""
        self._pid_file.parent.mkdir(parents=True, exist_ok=True)
        self._pid_file.write_text(str(os.getpid()))

    def _cleanup(self):
        """Shutdown executor and remove PID file."""
        self.audit_log.log("daemon_stopping")
        self.executor.shutdown(wait=True)
        if self._pid_file.exists():
            self._pid_file.unlink()
        self.audit_log.log("daemon_stopped")

    def _setup_signals(self):
        """Register signal handlers for graceful shutdown.

        Only works from the main thread; silently skips otherwise (e.g. in tests).
        """
        import threading
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, self._handle_signal)
            signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """Handle SIGTERM/SIGINT by setting running to False."""
        self._running = False

    def _interruptible_sleep(self, duration: float):
        """Sleep in small increments so we can respond to signals."""
        end = time.time() + duration
        while time.time() < end and self._running:
            time.sleep(min(0.1, end - time.time()))


def stop_daemon(project_root: Path) -> bool:
    """Stop a running daemon by sending SIGTERM to its PID.

    Returns True if a daemon was found and signaled, False otherwise.
    """
    pid_file = Path(project_root) / ".corc" / "daemon.pid"
    if not pid_file.exists():
        return False

    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        return True
    except (ValueError, ProcessLookupError, PermissionError):
        # PID file is stale or invalid — clean up
        pid_file.unlink(missing_ok=True)
        return False
