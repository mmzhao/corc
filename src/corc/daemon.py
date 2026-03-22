"""Daemon — thin event loop connecting scheduler, executor, processor.

The daemon is a minimal polling loop that delegates to three independent modules.
Each module reads from and writes to the work state — they do not call each other.

Loop: scheduler → executor → processor, every poll_interval seconds.

Retry logic: the processor marks failed tasks with attempt_count. The scheduler
automatically includes failed tasks that haven't exceeded max_retries. When
max_retries is exhausted, the processor sets status to 'escalated'.
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
from corc.reconcile import _get_agent_pid, _get_last_agent_output, reconcile_on_startup
from corc.retry import RetryPolicy
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
        retry_policy: RetryPolicy | None = None,
        pid_checker=None,
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
        self.retry_policy = retry_policy or RetryPolicy()
        self._running = False
        self._pid_file = self.project_root / ".corc" / "daemon.pid"
        self._tasks_completed = 0
        self._pid_checker = pid_checker
        self._reconcile_summary = None

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
        """Run the daemon loop until stopped or task complete.

        On startup, runs reconciliation to recover from any prior crash:
        rebuilds SQLite from mutation log, checks running tasks for PID
        liveness, processes dead agent output, and cleans stale worktrees.

        Failed tasks from reconciliation are automatically picked up by the
        scheduler if they haven't exceeded max_retries (no manual reset needed).
        """
        self._running = True
        self._write_pid()
        self._setup_signals()

        # Reconcile state from mutation log before entering the loop.
        # This handles daemon restarts after crashes: rebuilds SQLite,
        # checks if any "running" agents are actually dead, processes
        # their output or marks them failed.
        self._reconcile_summary = reconcile_on_startup(
            state=self.state,
            mutation_log=self.mutation_log,
            audit_log=self.audit_log,
            session_logger=self.session_logger,
            project_root=self.project_root,
            pid_checker=self._pid_checker,
        )

        # No need to manually retry reconciled failures — the scheduler
        # automatically includes failed tasks that haven't exceeded
        # max_retries in its ready queue.

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
        """One iteration of the daemon loop: schedule → execute → process.

        The processor handles retry vs escalation decisions. Failed tasks
        with attempt_count <= max_retries are automatically picked up by
        the scheduler on the next tick.
        """
        # Refresh state to pick up new tasks/mutations
        self.state.refresh()

        # Check for pause lock — skip scheduling new tasks but still process in-flight
        paused = is_paused(self.project_root / ".corc")

        if not paused:
            # 1. Scheduler: find ready tasks (includes retriable failed tasks)
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
        #    The processor handles failed vs escalated decisions based on
        #    attempt count and max_retries per task.
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

            # Only count terminal completions (completed or escalated)
            # Retriable failures don't count — they'll be retried.
            self.state.refresh()
            task = self.state.get_task(item.task["id"])
            if task and task["status"] in ("completed", "escalated"):
                self._tasks_completed += 1

        # 5. Reconcile externally-dispatched tasks (e.g. via 'corc dispatch')
        #    These are running tasks with no matching executor handle.
        self._reconcile_external_tasks()

    def _reconcile_external_tasks(self):
        """Reconcile running tasks not tracked by this daemon's executor.

        Tasks dispatched externally (e.g. via 'corc dispatch') are in 'running'
        state but have no matching Future in the executor. For each such task:
        - If the agent PID is still alive: leave it alone
        - If the agent is dead and has session output: process through the
          normal validation pipeline (may complete or fail)
        - If the agent is dead with no output: mark as failed
        """
        from corc.reconcile import _default_pid_checker

        pid_checker = self._pid_checker or _default_pid_checker

        running_tasks = self.state.list_tasks(status="running")
        in_flight_ids = self.executor.in_flight_task_ids

        for task in running_tasks:
            task_id = task["id"]

            # Skip tasks the executor is actively tracking
            if task_id in in_flight_ids:
                continue

            # Check if the agent process is still alive
            pid = _get_agent_pid(self.state, task_id)
            if pid and pid_checker(pid):
                # Agent still running — leave it alone
                continue

            # Agent is dead (or has no PID) — check for output
            output = _get_last_agent_output(self.session_logger, task_id)

            if output is not None:
                # Process output through normal validation pipeline
                attempt = self.session_logger.get_latest_attempt(task_id)
                process_completed(
                    task=task,
                    result=output,
                    attempt=attempt,
                    mutation_log=self.mutation_log,
                    state=self.state,
                    audit_log=self.audit_log,
                    session_logger=self.session_logger,
                    project_root=self.project_root,
                )
                self.audit_log.log(
                    "reconcile_external_processed",
                    task_id=task_id,
                )
            else:
                # No output — mark as failed so retry can kick in
                attempt = self.session_logger.get_latest_attempt(task_id) or 1
                self.mutation_log.append(
                    "task_failed",
                    {
                        "attempt": attempt,
                        "attempt_count": attempt,
                        "exit_code": -1,
                        "reconciled": True,
                    },
                    reason="Reconciliation: externally-dispatched agent finished without output",
                    task_id=task_id,
                )
                self.audit_log.log(
                    "reconcile_external_marked_failed",
                    task_id=task_id,
                )

            # Refresh state after each reconciliation to see updates
            self.state.refresh()

            # Count terminal completions from reconciliation
            updated_task = self.state.get_task(task_id)
            if updated_task and updated_task["status"] in ("completed", "escalated"):
                self._tasks_completed += 1

    def _get_target_task(self) -> list[dict]:
        """Get the specific target task if it's ready for dispatch."""
        task = self.state.get_task(self.target_task_id)
        if task and task["status"] in ("pending", "failed"):
            # For failed tasks, check retry eligibility
            if task["status"] == "failed":
                attempt_count = task.get("attempt_count", 0)
                max_retries = task.get("max_retries", 3)
                if attempt_count > max_retries:
                    return []
            return [task]
        return []

    def _should_stop(self) -> bool:
        """Check if daemon should exit (--once or --task modes)."""
        if self.once and self._tasks_completed > 0:
            return True
        if self.target_task_id and self._tasks_completed > 0:
            # Check if the target task is done (terminal state)
            task = self.state.get_task(self.target_task_id)
            if task and task["status"] in ("completed", "escalated"):
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
