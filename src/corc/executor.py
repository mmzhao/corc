"""Executor module — dispatches agents and polls for completion.

Dispatches agents via the dispatch abstraction layer, captures output.
Uses a thread pool for parallel dispatch. Can be called standalone: `corc dispatch TASK_ID`

Each agent runs in its own git worktree for filesystem isolation.
Worktrees are created before dispatch and cleaned up after completion.

PR workflow: before creating a worktree, the executor git-pulls main to
ensure branches start from the latest state. After agent completion, the
worktree branch is pushed and a PR is created via `gh pr create`.
"""

import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import json

from corc.audit import AuditLog
from corc.context import assemble_context, check_context_staleness
from corc.dispatch import AgentDispatcher, AgentResult, Constraints
from corc.mutations import MutationLog
from corc.retry import get_retry_context
from corc.roles import RoleLoader, constraints_from_role, get_system_prompt_for_role
from corc.sessions import SessionLogger
from corc.state import WorkState
from corc.pr import (
    create_pr,
    merge_pr,
    pull_main,
    push_branch,
    get_worktree_branch,
    PRInfo,
)
from corc.repo_policy import get_repo_policy
from corc.worktree import (
    WorktreeError,
    create_worktree,
    merge_main_into_worktree,
    merge_worktree,
    remove_worktree,
)


@dataclass
class CompletedTask:
    """A task that has finished executing, with its result."""

    task: dict
    result: AgentResult
    attempt: int
    worktree_path: Path | None = None
    agent_id: str | None = None
    merge_status: str | None = None
    pr_info: PRInfo | None = None


class Executor:
    """Manages agent dispatch and tracks in-flight tasks.

    When defer_merge=True (used by daemon for optimistic merge strategy),
    poll_completed() does NOT merge worktrees automatically. The caller
    is responsible for calling try_merge_worktree() after validation passes,
    and cleanup_worktree() when done.

    When defer_merge=False (default, backward compatible), poll_completed()
    merges and cleans up worktrees as before.
    """

    def __init__(
        self,
        dispatcher: AgentDispatcher,
        mutation_log: MutationLog,
        state: WorkState,
        audit_log: AuditLog,
        session_logger: SessionLogger,
        project_root: Path,
        max_workers: int = 1,
        role_loader: RoleLoader | None = None,
        defer_merge: bool = False,
    ):
        self.dispatcher = dispatcher
        self.mutation_log = mutation_log
        self.state = state
        self.audit_log = audit_log
        self.session_logger = session_logger
        self.project_root = project_root
        self.role_loader = role_loader or RoleLoader(project_root)
        self.defer_merge = defer_merge
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: dict[Future, tuple[dict, int, Path | None, str | None]] = {}
        # Track which futures are reattached (skip session logging for these)
        self._reattached_futures: set[Future] = set()
        # Track in-flight task IDs to prevent duplicate dispatches
        self._in_flight_tasks: set[str] = set()
        # Worktrees saved for conflict retry: task_id → worktree_path
        self._conflict_worktrees: dict[str, Path] = {}

    def is_in_flight(self, task_id: str) -> bool:
        """Check if a task already has an in-flight dispatch."""
        return task_id in self._in_flight_tasks

    def dispatch(self, task: dict):
        """Dispatch an agent for a task (non-blocking).

        Creates a git worktree for isolation, marks the task as running,
        builds prompt/context, submits to thread pool.
        """
        task_id = task["id"]
        if task_id in self._in_flight_tasks:
            self.audit_log.log(
                "dispatch_skipped_duplicate",
                task_id=task_id,
                reason="Task already has an in-flight agent",
            )
            return

        self._in_flight_tasks.add(task_id)
        attempt = self.session_logger.get_latest_attempt(task["id"]) + 1

        # Check for a saved conflict worktree (from a previous merge conflict)
        worktree_path = None
        agent_id = f"agent-{uuid.uuid4().hex[:8]}"
        conflict_worktree = self._conflict_worktrees.pop(task["id"], None)

        if conflict_worktree and conflict_worktree.exists():
            # Reuse the conflict worktree (already has main merged in)
            worktree_path = conflict_worktree
            self.audit_log.log(
                "worktree_reused",
                task_id=task["id"],
                worktree_path=str(worktree_path),
                reason="Reusing worktree after merge conflict",
            )
        else:
            # Pull latest main before creating worktree so branch starts
            # from the most recent state.
            pulled = pull_main(self.project_root)
            self.audit_log.log(
                "main_pulled",
                task_id=task["id"],
                success=pulled,
            )

            # Create git worktree for agent isolation
            try:
                worktree_path, branch_name = create_worktree(
                    self.project_root, task["id"], attempt
                )
                self.audit_log.log(
                    "worktree_created",
                    task_id=task["id"],
                    worktree_path=str(worktree_path),
                    branch_name=branch_name,
                )
            except (WorktreeError, Exception) as e:
                # If worktree creation fails, fall back to running in project root
                self.audit_log.log(
                    "worktree_creation_failed",
                    task_id=task["id"],
                    error=str(e),
                )

        # Create agent record with worktree path
        self.mutation_log.append(
            "agent_created",
            {
                "id": agent_id,
                "role": task.get("role", "implementer"),
                "task_id": task["id"],
                "worktree_path": str(worktree_path) if worktree_path else None,
            },
            reason="Agent created for dispatch",
        )

        # Mark task as running in state
        self.mutation_log.append(
            "task_started",
            {"attempt": attempt},
            reason="Dispatched by daemon",
            task_id=task["id"],
        )
        self.audit_log.log(
            "task_dispatched",
            task_id=task["id"],
            role=task.get("role", "implementer"),
            attempt=attempt,
            agent_id=agent_id,
            worktree_path=str(worktree_path) if worktree_path else None,
        )

        # Check for stale context bundle files
        stale_files = check_context_staleness(task, self.project_root)
        if stale_files:
            stale_names = [s["file"] for s in stale_files]
            self.audit_log.log(
                "context_staleness_warning",
                task_id=task["id"],
                stale_files=stale_names,
                details=stale_files,
            )

        # Build prompt and context using role config
        context = assemble_context(task, self.project_root)
        role_name = task.get("role", "implementer")

        # Load role config and derive constraints + system prompt
        try:
            role_config = self.role_loader.load(role_name)
            constraints = constraints_from_role(role_config)
            system_prompt = get_system_prompt_for_role(role_config, task, context)
        except ValueError:
            # Fallback if role YAML not found — use defaults
            constraints = Constraints()
            system_prompt = (
                f"You are a {role_name} working on task '{task['name']}'.\n\n{context}"
            )

        prompt = (
            f"Task ID: {task['id']}\n\n"
            f"Complete the following task.\n\n"
            f"Done when: {task['done_when']}\n\n"
            f"Work in the current directory. Write tests alongside implementation. "
            f"Commit your changes with a clear message referencing the task."
        )

        # Enrich prompt with previous session context for retries
        if attempt > 1:
            retry_context = get_retry_context(task["id"], attempt, self.session_logger)
            if retry_context:
                prompt = prompt + "\n" + retry_context

        # Log the dispatch
        self.session_logger.log_dispatch(
            task["id"],
            attempt,
            prompt,
            system_prompt,
            constraints.allowed_tools,
            constraints.max_budget_usd,
        )

        # Build streaming event callback for real-time logging
        event_callback = self._make_event_callback(task["id"], attempt)

        # Build PID callback to record the agent's PID in the mutation log.
        # This is critical for daemon restart recovery — the PID allows
        # reconciliation to check if the agent is still alive and re-attach.
        pid_callback = self._make_pid_callback(agent_id, task["id"])

        # Determine working directory
        cwd = str(worktree_path) if worktree_path else None

        # Submit to thread pool (non-blocking)
        future = self._pool.submit(
            self.dispatcher.dispatch,
            prompt,
            system_prompt,
            constraints,
            pid_callback=pid_callback,
            event_callback=event_callback,
            cwd=cwd,
        )
        self._futures[future] = (task, attempt, worktree_path, agent_id)

    def _make_event_callback(self, task_id: str, attempt: int):
        """Create a streaming event callback for a dispatch.

        The returned callback:
        1. Writes every event to the session log immediately (crash-safe).
        2. Writes tool_use events to the audit log with task_id.
        3. Writes assistant message events to the audit log for TUI visibility.
        """

        def callback(event):
            event_type = event.get("type", "unknown")

            # 1. Write every event to session log immediately
            self.session_logger.log_stream_event(task_id, attempt, event)

            # 2. Write tool_use events to audit log
            if event_type == "tool_use":
                tool = event.get("tool", {})
                self.audit_log.log(
                    "tool_use",
                    task_id=task_id,
                    tool_name=tool.get("name", "unknown"),
                    tool_input=json.dumps(tool.get("input", {}), separators=(",", ":")),
                )

            # 3. Write assistant messages to audit log for TUI visibility
            elif event_type == "assistant":
                message = event.get("message", {})
                content_blocks = message.get("content", [])
                text_parts = []
                for block in content_blocks:
                    if block.get("type") == "text":
                        text_parts.append(block["text"])
                if text_parts:
                    self.audit_log.log(
                        "assistant_message",
                        task_id=task_id,
                        content="\n".join(text_parts),
                    )

        return callback

    def _make_pid_callback(self, agent_id: str, task_id: str):
        """Create a callback that records the agent's PID in the mutation log.

        Called by the dispatcher once the subprocess starts.  Persists the PID
        so that ``reconcile_on_startup`` can later check process liveness and
        re-attach monitoring after a daemon restart.
        """

        def callback(pid: int):
            self.mutation_log.append(
                "agent_updated",
                {"agent_id": agent_id, "pid": pid},
                reason="Agent process started, recording PID for recovery",
                task_id=task_id,
            )

        return callback

    # ------------------------------------------------------------------
    # Re-attachment: recover monitoring after daemon restart
    # ------------------------------------------------------------------

    def reattach(
        self,
        task: dict,
        pid: int,
        attempt: int,
        worktree_path: Path | None,
        agent_id: str | None,
    ):
        """Re-attach monitoring for a running agent found during reconciliation.

        Submits a PID-watching function to the thread pool.  When the agent
        process exits, reads session logs for captured output and produces a
        ``CompletedTask`` through the normal ``poll_completed`` flow.

        This prevents duplicate dispatches: the task stays in 'running' state
        (so the scheduler ignores it) and is tracked in ``in_flight_task_ids``
        (so external reconciliation skips it).
        """
        self._in_flight_tasks.add(task["id"])
        future = self._pool.submit(self._wait_for_pid, pid, task["id"])
        self._futures[future] = (task, attempt, worktree_path, agent_id)
        self._reattached_futures.add(future)
        self.audit_log.log(
            "agent_reattached",
            task_id=task["id"],
            pid=pid,
            attempt=attempt,
            agent_id=agent_id,
        )

    def _wait_for_pid(self, pid: int, task_id: str) -> AgentResult:
        """Poll until *pid* exits, then return the result from session logs.

        Used by ``reattach`` to monitor an agent process that was already
        running when the daemon restarted.  The session log may contain
        output captured before the daemon died; if so we return it.  If no
        output was captured, we return a failure result so the retry policy
        can kick in.
        """
        from corc.reconcile import _get_last_agent_output, is_pid_alive

        while is_pid_alive(pid):
            time.sleep(1.0)

        # Process exited — check session logs for captured output
        output = _get_last_agent_output(self.session_logger, task_id)
        if output is not None:
            return output

        return AgentResult(
            output="Agent process exited without captured output (daemon restarted mid-flight)",
            exit_code=-1,
            duration_s=0,
        )

    def _create_pr_from_worktree(
        self, task: dict, worktree_path: Path
    ) -> PRInfo | None:
        """Push worktree branch and create a PR.

        After an agent completes successfully, pushes the branch to origin
        and creates a PR via `gh pr create`. This ensures no code is ever
        pushed directly to main — all changes go through PRs.

        Args:
            task: The task dict.
            worktree_path: Path to the worktree.

        Returns:
            PRInfo if PR was created successfully, None otherwise.
        """
        task_id = task["id"]
        branch_name = get_worktree_branch(self.project_root, worktree_path)
        if not branch_name:
            self.audit_log.log(
                "pr_creation_skipped",
                task_id=task_id,
                reason="Could not determine branch name for worktree",
            )
            return None

        # Push the branch to remote
        pushed, push_error = push_branch(self.project_root, branch_name)
        if not pushed:
            self.audit_log.log(
                "pr_push_failed",
                task_id=task_id,
                branch=branch_name,
                error=push_error,
            )
            return None

        # Create the PR
        pr_info, pr_error = create_pr(self.project_root, branch_name, task)
        if pr_info:
            self.audit_log.log(
                "pr_created",
                task_id=task_id,
                pr_url=pr_info.url,
                pr_number=pr_info.number,
                branch=branch_name,
            )
        else:
            self.audit_log.log(
                "pr_creation_failed",
                task_id=task_id,
                branch=branch_name,
                error=pr_error,
            )

        return pr_info

    def poll_completed(self) -> list[CompletedTask]:
        """Check for completed dispatches. Non-blocking.

        Returns list of CompletedTask for tasks that have finished since last poll.

        When defer_merge=False (default): merges worktree changes and cleans up.
        When defer_merge=True: returns CompletedTask with worktree still alive.
        Caller must call try_merge_worktree() and cleanup_worktree() explicitly.
        """
        completed = []
        done_futures = [f for f in self._futures if f.done()]

        for future in done_futures:
            task, attempt, worktree_path, agent_id = self._futures.pop(future)
            self._in_flight_tasks.discard(task["id"])
            is_reattached = future in self._reattached_futures
            if is_reattached:
                self._reattached_futures.discard(future)

            try:
                result = future.result()
            except Exception as e:
                result = AgentResult(
                    output=f"Dispatch error: {e}",
                    exit_code=-1,
                    duration_s=0,
                )

            if not is_reattached:
                # Log the output (skip for reattached — session already has it)
                self.session_logger.log_output(
                    task["id"],
                    attempt,
                    result.output,
                    result.exit_code,
                    result.duration_s,
                )
            self.audit_log.log(
                "reattached_task_complete"
                if is_reattached
                else "task_dispatch_complete",
                task_id=task["id"],
                exit_code=result.exit_code,
                duration_s=result.duration_s,
                attempt=attempt,
            )

            # Create PR from worktree branch (PR-based workflow)
            # PR creation is mandatory — if it fails, the task fails
            pr_info = None
            if worktree_path and result.exit_code == 0:
                pr_info = self._create_pr_from_worktree(task, worktree_path)
                if pr_info is None:
                    self.audit_log.log(
                        "task_failed_pr_required",
                        task_id=task["id"],
                        reason="PR creation failed — all repos require PRs",
                    )
                    # Return as failed so retry/escalation kicks in
                    completed.append(
                        CompletedTask(
                            task=task,
                            result=AgentResult(
                                output=result.output
                                + "\n\n[CORC] PR creation failed. Task cannot complete without a PR.",
                                exit_code=1,
                                duration_s=result.duration_s,
                            ),
                            worktree_path=worktree_path,
                            pr_info=None,
                            attempt=attempt,
                        )
                    )
                    continue

            if not self.defer_merge and worktree_path:
                # Legacy behavior: merge and clean up immediately
                # Respect repo merge policy: human-only repos skip the merge
                policy = get_repo_policy(self.project_root)
                if policy.is_human_only:
                    self.audit_log.log(
                        "worktree_merge_skipped",
                        task_id=task["id"],
                        worktree_path=str(worktree_path),
                        merge_policy="human-only",
                        reason="Merge policy is human-only; PR left for human review",
                    )
                    try:
                        remove_worktree(
                            self.project_root, worktree_path, remove_branch=False
                        )
                        self.audit_log.log(
                            "worktree_removed",
                            task_id=task["id"],
                            worktree_path=str(worktree_path),
                        )
                    except Exception as e:
                        self.audit_log.log(
                            "worktree_remove_error",
                            task_id=task["id"],
                            error=str(e),
                        )
                else:
                    self._merge_and_cleanup_worktree(task["id"], worktree_path)

            completed.append(
                CompletedTask(
                    task=task,
                    result=result,
                    attempt=attempt,
                    worktree_path=worktree_path,
                    agent_id=agent_id,
                    pr_info=pr_info,
                )
            )

        return completed

    def _merge_and_cleanup_worktree(self, task_id: str, worktree_path: Path):
        """Merge worktree changes back to main and remove the worktree."""
        try:
            merged = merge_worktree(self.project_root, worktree_path)
            if merged:
                self.audit_log.log(
                    "worktree_merged",
                    task_id=task_id,
                    worktree_path=str(worktree_path),
                )
            else:
                self.audit_log.log(
                    "worktree_merge_conflict",
                    task_id=task_id,
                    worktree_path=str(worktree_path),
                )
        except Exception as e:
            self.audit_log.log(
                "worktree_merge_error",
                task_id=task_id,
                error=str(e),
            )

        try:
            remove_worktree(self.project_root, worktree_path)
            self.audit_log.log(
                "worktree_removed",
                task_id=task_id,
                worktree_path=str(worktree_path),
            )
        except Exception as e:
            self.audit_log.log(
                "worktree_remove_error",
                task_id=task_id,
                error=str(e),
            )

    def try_merge_worktree(
        self, task_id: str, worktree_path: Path, pr_info: PRInfo | None = None
    ) -> str:
        """Try to merge worktree branch into main.

        When pr_info is provided, merges via ``gh pr merge`` instead of
        direct git merge.  After a successful PR merge, pulls main to
        sync the local repo.

        When no pr_info is given, falls back to the direct git merge
        (optimistic merge strategy).

        Returns merge status:
        - "merged": successfully merged to main
        - "no_changes": nothing to merge
        - "conflict": merge conflict detected / PR merge failed
        - "error": unexpected error
        """
        if pr_info and pr_info.number:
            return self._try_pr_merge(task_id, worktree_path, pr_info)

        # Direct git merge (no PR — fallback / backward compat)
        try:
            merged = merge_worktree(self.project_root, worktree_path)
            if merged:
                self.audit_log.log(
                    "worktree_merged",
                    task_id=task_id,
                    worktree_path=str(worktree_path),
                )
                return "merged"
            else:
                self.audit_log.log(
                    "worktree_merge_conflict",
                    task_id=task_id,
                    worktree_path=str(worktree_path),
                )
                return "conflict"
        except Exception as e:
            self.audit_log.log(
                "worktree_merge_error",
                task_id=task_id,
                error=str(e),
            )
            return "error"

    def _try_pr_merge(self, task_id: str, worktree_path: Path, pr_info: PRInfo) -> str:
        """Merge via ``gh pr merge`` and sync local main.

        Returns the same status strings as :meth:`try_merge_worktree`.
        """
        try:
            merged = merge_pr(self.project_root, pr_info.number)
            if merged:
                # Sync local main with the remote after PR merge
                pull_main(self.project_root)
                self.audit_log.log(
                    "worktree_merged_via_pr",
                    task_id=task_id,
                    pr_number=pr_info.number,
                    pr_url=pr_info.url,
                    worktree_path=str(worktree_path),
                )
                return "merged"
            else:
                self.audit_log.log(
                    "pr_merge_failed",
                    task_id=task_id,
                    pr_number=pr_info.number,
                    worktree_path=str(worktree_path),
                )
                return "conflict"
        except Exception as e:
            self.audit_log.log(
                "pr_merge_error",
                task_id=task_id,
                pr_number=pr_info.number,
                error=str(e),
            )
            return "error"

    def prepare_conflict_retry(self, task_id: str, worktree_path: Path) -> bool:
        """Prepare a worktree for retry after merge conflict.

        Merges main into the worktree so the next agent dispatch sees
        both its previous work and the latest main state. Saves the
        worktree for reuse by the next dispatch of this task.

        Returns True if main was successfully merged into worktree.
        """
        try:
            success = merge_main_into_worktree(self.project_root, worktree_path)
            if success:
                # Save worktree for reuse by next dispatch
                self._conflict_worktrees[task_id] = worktree_path
                self.audit_log.log(
                    "worktree_conflict_retry_prepared",
                    task_id=task_id,
                    worktree_path=str(worktree_path),
                )
                return True
            else:
                # Both merges failed — clean up and let retry create fresh worktree
                self.audit_log.log(
                    "worktree_conflict_unresolvable",
                    task_id=task_id,
                    worktree_path=str(worktree_path),
                )
                self.cleanup_worktree(task_id, worktree_path)
                return False
        except Exception as e:
            self.audit_log.log(
                "worktree_conflict_retry_error",
                task_id=task_id,
                error=str(e),
            )
            self.cleanup_worktree(task_id, worktree_path)
            return False

    def cleanup_worktree(
        self, task_id: str, worktree_path: Path, remove_branch: bool = True
    ):
        """Remove a worktree and optionally its branch.

        Args:
            task_id: Task identifier for audit logging.
            worktree_path: Path to the worktree to remove.
            remove_branch: If False, keep the branch (needed when PR
                references it, e.g. human-only repos).
        """
        try:
            remove_worktree(
                self.project_root, worktree_path, remove_branch=remove_branch
            )
            self.audit_log.log(
                "worktree_removed",
                task_id=task_id,
                worktree_path=str(worktree_path),
            )
        except Exception as e:
            self.audit_log.log(
                "worktree_remove_error",
                task_id=task_id,
                error=str(e),
            )

    def set_conflict_worktree(self, task_id: str, worktree_path: Path):
        """Register a worktree for reuse on the next dispatch of a task."""
        self._conflict_worktrees[task_id] = worktree_path

    @property
    def in_flight_count(self) -> int:
        """Number of tasks currently being executed."""
        return len(self._futures)

    @property
    def in_flight_task_ids(self) -> set[str]:
        """Set of task IDs currently tracked by the executor."""
        return {task["id"] for task, _attempt, _wt, _aid in self._futures.values()}

    def shutdown(self, wait: bool = True):
        """Shutdown the thread pool, optionally waiting for in-flight tasks."""
        self._pool.shutdown(wait=wait)
