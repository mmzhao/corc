"""Daemon — thin event loop connecting scheduler, executor, processor.

The daemon is a minimal polling loop that delegates to three independent modules.
Each module reads from and writes to the work state — they do not call each other.

Loop: scheduler → executor → processor, every poll_interval seconds.

Retry logic: the processor marks failed tasks with attempt_count. The scheduler
automatically includes failed tasks that haven't exceeded max_retries. When
max_retries is exhausted, the processor sets status to 'escalated'.

Hot reload: the daemon watches Python source files in the corc package directory.
When agents modify the codebase, changed modules are reloaded via importlib.reload()
before the next dispatch cycle. This preserves in-flight task state while ensuring
new code is used.
"""

import os
import signal
import sys
import time
from pathlib import Path

from corc.audit import AuditLog
from corc.backup import run_daily_backup
from corc.rotate import run_daily_rotation
from corc.chaos import (
    ChaosMonkey,
    is_chaos_enabled,
    mark_event_recovered,
    read_chaos_config,
)
from corc.dispatch import AgentDispatcher
from corc.executor import Executor
from corc.mutations import MutationLog
from corc.pause import is_paused
from corc.pr import pull_main
from corc.processor import process_completed
from corc.reconcile import _get_agent_pid, _get_last_agent_output, reconcile_on_startup
from corc.repo_policy import get_repo_policy
from corc.reload import SourceWatcher
from corc.retry import RetryPolicy
from corc.scheduler import get_ready_tasks
from corc.sessions import SessionLogger
from corc.state import WorkState

# Module-level function references that get rebound after hot reload.
# Maps module name → list of (global_name, attr_name) pairs.
_RELOAD_BINDINGS: dict[str, list[tuple[str, str]]] = {
    "corc.scheduler": [("get_ready_tasks", "get_ready_tasks")],
    "corc.processor": [("process_completed", "process_completed")],
    "corc.pause": [("is_paused", "is_paused")],
    "corc.reconcile": [
        ("_get_agent_pid", "_get_agent_pid"),
        ("_get_last_agent_output", "_get_last_agent_output"),
        ("reconcile_on_startup", "reconcile_on_startup"),
    ],
}


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
        auto_reload: bool = True,
        chaos_monkey: ChaosMonkey | None = None,
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

        # Chaos monkey — optional, auto-detected from config on disk
        corc_dir = self.project_root / ".corc"
        if chaos_monkey is not None:
            self._chaos_monkey = chaos_monkey
        elif is_chaos_enabled(corc_dir):
            self._chaos_monkey = ChaosMonkey(corc_dir)
        else:
            self._chaos_monkey = None

        # Source file watcher for hot-reload
        self._source_watcher = self._create_source_watcher() if auto_reload else None

        self.executor = Executor(
            dispatcher=dispatcher,
            mutation_log=mutation_log,
            state=state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=project_root,
            max_workers=parallel,
            defer_merge=True,
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
        # Hot-reload changed source modules before dispatching
        self._check_source_reload()

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

        # 3a. Chaos monkey: randomly kill agents / corrupt state
        self._chaos_tick()

        # 3. Executor: poll for completed dispatches (always — in-flight tasks finish)
        completed = self.executor.poll_completed()

        # 4. Processor: validate and update state, then optimistic merge
        #    The processor handles failed vs escalated decisions based on
        #    attempt count and max_retries per task.
        for item in completed:
            proc_result = process_completed(
                task=item.task,
                result=item.result,
                attempt=item.attempt,
                mutation_log=self.mutation_log,
                state=self.state,
                audit_log=self.audit_log,
                session_logger=self.session_logger,
                project_root=self.project_root,
                pr_info=item.pr_info,
            )

            # 5. Optimistic merge: merge worktree → main after validation passes
            if item.worktree_path:
                self._handle_worktree_merge(item, proc_result)
            elif proc_result.passed:
                # No worktree (fallback mode) — just count completion
                pass

            # Only count terminal completions (completed or escalated)
            # Retriable failures don't count — they'll be retried.
            self.state.refresh()
            task = self.state.get_task(item.task["id"])
            if task and task["status"] in ("completed", "escalated"):
                self._tasks_completed += 1
                # Track chaos recovery: if this task had a chaos event, mark it recovered
                if self._chaos_monkey and task["status"] == "completed":
                    corc_dir = self.project_root / ".corc"
                    mark_event_recovered(corc_dir, item.task["id"])

        # 5. Reconcile externally-dispatched tasks (e.g. via 'corc dispatch')
        #    These are running tasks with no matching executor handle.
        self._reconcile_external_tasks()

        # 6. Daily maintenance: audit log backup and rotation
        self._check_daily_backup()

        # 7. Daily maintenance: log rotation (move old logs to archive)
        self._check_log_rotation()

    def _handle_worktree_merge(self, item, proc_result):
        """Handle worktree merge after agent completion and validation.

        When a PR exists (PR workflow):
        - Auto-merge repos: if the processor already merged the PR via
          ``gh pr merge``, pull main to sync local repo, record merge
          status, and clean up the worktree.  If the processor did not
          merge, retry via ``try_merge_worktree`` with the PR info.
        - Human-only repos: PR is left open for human review.  Clean up
          the worktree (keeping the branch for the PR).  Task stays in
          ``pending_merge`` status.

        When no PR exists (fallback — e.g. no remote configured):
        - Direct git merge (optimistic merge strategy) as before.

        If validation failed: clean up worktree (processor already set
        the appropriate failed/escalated status).
        """
        task_id = item.task["id"]
        worktree_path = item.worktree_path

        if not proc_result.passed:
            # Validation failed — clean up worktree, processor already marked failed/escalated
            self.executor.cleanup_worktree(task_id, worktree_path)
            return

        # PR-based workflow: use gh pr merge instead of direct git merge
        if item.pr_info:
            self._handle_pr_based_merge(item, proc_result)
            return

        # No PR (fallback): try direct git merge (optimistic merge)
        self._handle_direct_merge(item)

    def _handle_pr_based_merge(self, item, proc_result):
        """Handle merge via PR workflow (``gh pr merge``) instead of direct git merge.

        For auto-merge repos: merge the PR (or acknowledge the processor
        already merged it), pull main to sync, and clean up the worktree.

        For human-only repos: leave the PR open, clean up the worktree
        but keep the branch so the PR remains valid on GitHub.
        """
        task_id = item.task["id"]
        worktree_path = item.worktree_path
        pr_info = item.pr_info
        policy = get_repo_policy(self.project_root)

        if policy.is_human_only:
            # Human-only: PR is open, task is pending_merge (set by processor).
            # Clean up worktree but keep the branch (PR references it).
            self.audit_log.log(
                "worktree_merge_skipped_human_only",
                task_id=task_id,
                pr_url=pr_info.url,
                pr_number=pr_info.number,
                reason="Human-only repo; PR left open for human review",
            )
            self.executor.cleanup_worktree(task_id, worktree_path, remove_branch=False)
            return

        # Auto-merge repo —
        if proc_result.pr_merged:
            # Processor already merged the PR via gh pr merge.
            # Pull main to sync local repo with the merged state.
            pull_main(self.project_root)
            self.mutation_log.append(
                "task_updated",
                {"merge_status": "merged"},
                reason=f"PR #{pr_info.number} merged via gh pr merge",
                task_id=task_id,
            )
            self.audit_log.log(
                "pr_merge_synced",
                task_id=task_id,
                pr_number=pr_info.number,
                pr_url=pr_info.url,
            )
            self.executor.cleanup_worktree(task_id, worktree_path)
            return

        # PR exists but processor didn't merge it (merge_pr failed).
        # Retry the merge via try_merge_worktree with PR info.
        merge_status = self.executor.try_merge_worktree(
            task_id, worktree_path, pr_info=pr_info
        )
        self._apply_merge_result(item, merge_status)

    def _handle_direct_merge(self, item):
        """Handle direct git merge (no PR — fallback / backward compat).

        Implements the optimistic merge strategy:
        - Success: record merge_status, clean up worktree
        - Conflict: merge main into worktree, mark task failed for retry
        - Error: clean up, leave task as completed
        """
        task_id = item.task["id"]
        worktree_path = item.worktree_path
        merge_status = self.executor.try_merge_worktree(task_id, worktree_path)
        self._apply_merge_result(item, merge_status)

    def _apply_merge_result(self, item, merge_status):
        """Record merge result, handle conflicts, and clean up worktree."""
        task_id = item.task["id"]
        worktree_path = item.worktree_path

        if merge_status in ("merged", "no_changes"):
            # Success! Record merge status and clean up
            self.mutation_log.append(
                "task_updated",
                {"merge_status": merge_status},
                reason=f"Worktree {merge_status} to main",
                task_id=task_id,
            )
            self.executor.cleanup_worktree(task_id, worktree_path)

        elif merge_status == "conflict":
            # Merge conflict — prepare worktree for retry
            self.mutation_log.append(
                "task_updated",
                {"merge_status": "conflict"},
                reason="Merge conflict detected, preparing retry with merged state",
                task_id=task_id,
            )

            # Try merging main into the worktree for retry baseline
            retry_prepared = self.executor.prepare_conflict_retry(
                task_id,
                worktree_path,
            )

            if retry_prepared:
                reason = (
                    "Merge conflict with main; retrying with merged state as baseline"
                )
            else:
                reason = "Merge conflict unresolvable; retrying from fresh state"

            # Revert task from completed to failed so scheduler retries it
            self.mutation_log.append(
                "task_failed",
                {
                    "attempt": item.attempt,
                    "attempt_count": item.attempt,
                    "merge_conflict": True,
                },
                reason=reason,
                task_id=task_id,
            )
            self.audit_log.log(
                "merge_conflict_retry",
                task_id=task_id,
                attempt=item.attempt,
                retry_prepared=retry_prepared,
            )

        else:
            # Error during merge — clean up, leave task as completed
            self.mutation_log.append(
                "task_updated",
                {"merge_status": "error"},
                reason="Merge error, task marked completed without merge",
                task_id=task_id,
            )
            self.executor.cleanup_worktree(task_id, worktree_path)

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

    # ------------------------------------------------------------------
    # Daily maintenance: audit log backup
    # ------------------------------------------------------------------

    def _check_daily_backup(self):
        """Run audit log backup and rotation if due.

        Called every tick but only performs work when the configured
        interval (daily/weekly) has elapsed since the last backup.
        Copies data/events/ and data/sessions/ to the backup path,
        then rotates files older than the configured retention period.
        """
        corc_dir = self.project_root / ".corc"
        events_dir = self.project_root / "data" / "events"
        sessions_dir = self.project_root / "data" / "sessions"

        try:
            result = run_daily_backup(corc_dir, events_dir, sessions_dir)
            if result is not None:
                self.audit_log.log(
                    "backup_completed",
                    backup_path=result["backup"]["backup_path"],
                    events_copied=result["backup"]["events_copied"],
                    sessions_copied=result["backup"]["sessions_copied"],
                    events_rotated=result["rotation"]["events_rotated"],
                    sessions_rotated=result["rotation"]["sessions_rotated"],
                    source_events_rotated=result["source_rotation"]["events_rotated"],
                    source_sessions_rotated=result["source_rotation"][
                        "sessions_rotated"
                    ],
                )
        except OSError as e:
            self.audit_log.log("backup_failed", error=str(e))

    # ------------------------------------------------------------------
    # Daily maintenance: log rotation (archive old logs)
    # ------------------------------------------------------------------

    def _check_log_rotation(self):
        """Move old session and audit logs to date-stamped archive directories.

        Called every tick but only performs work once per day.
        Files are never deleted — only moved to archive/YYYY-MM-DD/.
        """
        corc_dir = self.project_root / ".corc"
        events_dir = self.project_root / "data" / "events"
        sessions_dir = self.project_root / "data" / "sessions"

        try:
            result = run_daily_rotation(corc_dir, events_dir, sessions_dir)
            if result is not None:
                self.audit_log.log(
                    "log_rotation_completed",
                    sessions_moved=result["sessions"]["moved"],
                    events_moved=result["events"]["moved"],
                    rotate_after_days=result["rotate_after_days"],
                )
        except OSError as e:
            self.audit_log.log("log_rotation_failed", error=str(e))

    # ------------------------------------------------------------------
    # Chaos monkey integration
    # ------------------------------------------------------------------

    def _chaos_tick(self):
        """Run chaos monkey checks if enabled.

        Checks each running agent for random kill and optionally
        corrupts the state DB. The daemon's reconciliation and retry
        mechanisms handle recovery.
        """
        if not self._chaos_monkey:
            return

        # Re-read config each tick so operator can adjust rates live
        self._chaos_monkey.reload_config()
        if not self._chaos_monkey.config.enabled:
            return

        # Gather running agents with PIDs
        running_agents = self.state.list_agents(status="idle")
        # Also check agents from "running" tasks
        running_tasks = self.state.list_tasks(status="running")
        for task in running_tasks:
            agents = self.state.get_agents_for_task(task["id"])
            for agent in agents:
                if agent.get("pid"):
                    running_agents.append(agent)

        # Run chaos on agents
        for agent in running_agents:
            pid = agent.get("pid")
            task_id = agent.get("task_id")
            if pid:
                killed = self._chaos_monkey.maybe_kill_agent(pid, task_id)
                if killed:
                    self.audit_log.log(
                        "chaos_agent_killed",
                        task_id=task_id,
                        pid=pid,
                    )

        # Optionally corrupt the state DB (will be rebuilt from mutation log)
        state_db = self.state.db_path
        corrupted = self._chaos_monkey.maybe_corrupt_state(state_db)
        if corrupted:
            self.audit_log.log("chaos_state_corrupted", file=str(state_db))
            # Immediately rebuild state from mutation log to recover
            try:
                self.state.rebuild()
                self.audit_log.log("chaos_state_recovered", file=str(state_db))
            except Exception as e:
                self.audit_log.log(
                    "chaos_state_recovery_failed",
                    file=str(state_db),
                    error=str(e),
                )

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

    # ------------------------------------------------------------------
    # Hot reload
    # ------------------------------------------------------------------

    @staticmethod
    def _create_source_watcher() -> SourceWatcher | None:
        """Create a SourceWatcher for the corc package directory.

        Uses the corc package's __file__ to find the actual source directory
        that Python loads modules from (works with editable installs).

        Returns None if the package directory can't be determined.
        """
        try:
            import corc as _corc_pkg

            corc_src = Path(_corc_pkg.__file__).resolve().parent
            return SourceWatcher(corc_src)
        except (ImportError, AttributeError, TypeError):
            return None

    def _check_source_reload(self):
        """Check for source file changes and reload modules if needed.

        Called at the start of each _tick() to ensure the daemon uses the
        latest code before dispatching new tasks.
        """
        if not self._source_watcher:
            return

        reloaded = self._source_watcher.check_and_reload()
        if not reloaded:
            return

        # Re-bind module-level function references used by _tick
        self._rebind_after_reload(reloaded)
        self.audit_log.log("modules_reloaded", modules=reloaded)

    @staticmethod
    def _rebind_after_reload(reloaded_modules: list[str]):
        """Update module-level function references after hot reload.

        When modules are reloaded via importlib.reload(), existing references
        to their functions (bound at import time via 'from X import Y') become
        stale. This method re-binds the specific functions that _tick() and
        related methods use, so the daemon picks up the new code.
        """
        g = globals()
        for mod_name in reloaded_modules:
            bindings = _RELOAD_BINDINGS.get(mod_name)
            if not bindings:
                continue
            mod = sys.modules.get(mod_name)
            if not mod:
                continue
            for global_name, attr_name in bindings:
                if hasattr(mod, attr_name):
                    g[global_name] = getattr(mod, attr_name)


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
