"""Executor module — dispatches agents and polls for completion.

Dispatches agents via the dispatch abstraction layer, captures output.
Uses a thread pool for parallel dispatch. Can be called standalone: `corc dispatch TASK_ID`
"""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from corc.audit import AuditLog
from corc.context import assemble_context
from corc.dispatch import AgentDispatcher, AgentResult, Constraints
from corc.mutations import MutationLog
from corc.sessions import SessionLogger
from corc.state import WorkState


@dataclass
class CompletedTask:
    """A task that has finished executing, with its result."""
    task: dict
    result: AgentResult
    attempt: int


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
    ):
        self.dispatcher = dispatcher
        self.mutation_log = mutation_log
        self.state = state
        self.audit_log = audit_log
        self.session_logger = session_logger
        self.project_root = project_root
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: dict[Future, tuple[dict, int]] = {}

    def dispatch(self, task: dict):
        """Dispatch an agent for a task (non-blocking).

        Marks the task as running, builds prompt/context, submits to thread pool.
        """
        attempt = self.session_logger.get_latest_attempt(task["id"]) + 1

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
        )

        # Build prompt and context
        context = assemble_context(task, self.project_root)
        role = task.get("role", "implementer")
        system_prompt = f"You are a {role} working on task '{task['name']}'.\n\n{context}"

        prompt = (
            f"Complete the following task.\n\n"
            f"Done when: {task['done_when']}\n\n"
            f"Work in the current directory. Write tests alongside implementation. "
            f"Commit your changes with a clear message referencing the task."
        )

        constraints = Constraints()

        # Log the dispatch
        self.session_logger.log_dispatch(
            task["id"], attempt, prompt, system_prompt,
            constraints.allowed_tools, constraints.max_budget_usd,
        )

        # Submit to thread pool (non-blocking)
        future = self._pool.submit(
            self.dispatcher.dispatch, prompt, system_prompt, constraints,
        )
        self._futures[future] = (task, attempt)

    def poll_completed(self) -> list[CompletedTask]:
        """Check for completed dispatches. Non-blocking.

        Returns list of CompletedTask for tasks that have finished since last poll.
        """
        completed = []
        done_futures = [f for f in self._futures if f.done()]

        for future in done_futures:
            task, attempt = self._futures.pop(future)
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

            completed.append(CompletedTask(task=task, result=result, attempt=attempt))

        return completed

    @property
    def in_flight_count(self) -> int:
        """Number of tasks currently being executed."""
        return len(self._futures)

    def shutdown(self, wait: bool = True):
        """Shutdown the thread pool, optionally waiting for in-flight tasks."""
        self._pool.shutdown(wait=wait)
