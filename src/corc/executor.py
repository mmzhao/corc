"""Executor module — dispatches agents and polls for completion.

Dispatches agents via the dispatch abstraction layer, captures output.
Uses a thread pool for parallel dispatch. Can be called standalone: `corc dispatch TASK_ID`

Each agent runs in its own git worktree for filesystem isolation.
Worktrees are created before dispatch and cleaned up after completion.
"""

import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import json

from corc.audit import AuditLog
from corc.context import assemble_context
from corc.dispatch import AgentDispatcher, AgentResult, Constraints
from corc.mutations import MutationLog
from corc.retry import get_retry_context
from corc.roles import RoleLoader, constraints_from_role, get_system_prompt_for_role
from corc.sessions import SessionLogger
from corc.state import WorkState
from corc.worktree import WorktreeError, create_worktree, merge_worktree, remove_worktree


@dataclass
class CompletedTask:
    """A task that has finished executing, with its result."""
    task: dict
    result: AgentResult
    attempt: int
    worktree_path: Path | None = None
    agent_id: str | None = None


class Executor:
    """Manages agent dispatch and tracks in-flight tasks."""

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
    ):
        self.dispatcher = dispatcher
        self.mutation_log = mutation_log
        self.state = state
        self.audit_log = audit_log
        self.session_logger = session_logger
        self.project_root = project_root
        self.role_loader = role_loader or RoleLoader(project_root)
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: dict[Future, tuple[dict, int, Path | None, str | None]] = {}

    def dispatch(self, task: dict):
        """Dispatch an agent for a task (non-blocking).

        Creates a git worktree for isolation, marks the task as running,
        builds prompt/context, submits to thread pool.
        """
        attempt = self.session_logger.get_latest_attempt(task["id"]) + 1

        # Create git worktree for agent isolation
        worktree_path = None
        agent_id = f"agent-{uuid.uuid4().hex[:8]}"
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
            system_prompt = f"You are a {role_name} working on task '{task['name']}'.\n\n{context}"

        prompt = (
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
            task["id"], attempt, prompt, system_prompt,
            constraints.allowed_tools, constraints.max_budget_usd,
        )

        # Build streaming event callback for real-time logging
        event_callback = self._make_event_callback(task["id"], attempt)

        # Determine working directory
        cwd = str(worktree_path) if worktree_path else None

        # Submit to thread pool (non-blocking)
        future = self._pool.submit(
            self.dispatcher.dispatch, prompt, system_prompt, constraints,
            event_callback=event_callback, cwd=cwd,
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

    def poll_completed(self) -> list[CompletedTask]:
        """Check for completed dispatches. Non-blocking.

        Returns list of CompletedTask for tasks that have finished since last poll.
        Attempts to merge worktree changes and cleans up worktrees.
        """
        completed = []
        done_futures = [f for f in self._futures if f.done()]

        for future in done_futures:
            task, attempt, worktree_path, agent_id = self._futures.pop(future)
            try:
                result = future.result()
            except Exception as e:
                result = AgentResult(
                    output=f"Dispatch error: {e}",
                    exit_code=-1,
                    duration_s=0,
                )

            # Log the output
            self.session_logger.log_output(
                task["id"], attempt, result.output,
                result.exit_code, result.duration_s,
            )
            self.audit_log.log(
                "task_dispatch_complete",
                task_id=task["id"],
                exit_code=result.exit_code,
                duration_s=result.duration_s,
                attempt=attempt,
            )

            # Merge worktree changes back and clean up
            if worktree_path:
                self._merge_and_cleanup_worktree(task["id"], worktree_path)

            completed.append(CompletedTask(
                task=task, result=result, attempt=attempt,
                worktree_path=worktree_path, agent_id=agent_id,
            ))

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
