"""Query API — data layer for TUI and future web GUI.

Clean query API returning JSON-serializable dicts. All reads are
derived from the three data layers: WorkState, AuditLog, SessionLogger.
No mutations — this module is strictly read-only.
"""

import json
from datetime import datetime, timezone, timedelta

from corc.audit import AuditLog
from corc.sessions import SessionLogger
from corc.state import WorkState


# Statuses that represent "active" work in the plan
_ACTIVE_STATUSES = frozenset(
    {
        "pending",
        "assigned",
        "running",
        "escalated",
        "handed_off",
        "pending_merge",
        "failed",
    }
)


class QueryAPI:
    """Stateless query facade over the three data layers.

    Every method returns plain ``list[dict]`` or ``dict`` that is
    directly JSON-serializable (no SQLite Row objects, no datetime, etc.).
    """

    def __init__(
        self,
        work_state: WorkState,
        audit_log: AuditLog,
        session_logger: SessionLogger,
    ):
        self.work_state = work_state
        self.audit_log = audit_log
        self.session_logger = session_logger

    # ------------------------------------------------------------------
    # Plan-level queries
    # ------------------------------------------------------------------

    def get_active_plan_tasks(self) -> list[dict]:
        """All non-completed tasks — the currently active plan.

        Returns tasks with status in: pending, assigned, running,
        escalated, handed_off, pending_merge, failed.
        Sorted by priority ascending (lower number = higher priority).
        """
        all_tasks = self.work_state.list_tasks()
        active = [t for t in all_tasks if t.get("status") in _ACTIVE_STATUSES]
        active.sort(key=lambda t: t.get("priority", 100))
        return active

    def get_running_tasks_with_agents(self) -> list[dict]:
        """Running tasks enriched with their associated agent records.

        Each returned dict contains all task fields plus an ``agents``
        key holding a list of agent dicts for that task.

        When a task has been dispatched multiple times (e.g. after a
        daemon restart), only the most recent agent record is kept.
        Active (non-idle) agents are preferred over idle ones.
        """
        from corc.tui import _deduplicate_agents

        running = self.work_state.list_tasks(status="running")
        result = []
        for task in running:
            agents = self.work_state.get_agents_for_task(task["id"])
            result.append({**task, "agents": _deduplicate_agents(agents)})
        return result

    def get_ready_tasks(self) -> list[dict]:
        """Tasks ready for dispatch (deps satisfied, retriable failures).

        Delegates to ``WorkState.get_ready_tasks()`` which handles
        dependency checking and retry eligibility.
        """
        return self.work_state.get_ready_tasks()

    def get_blocked_tasks_with_reasons(self) -> list[dict]:
        """Pending tasks whose dependencies are not yet satisfied.

        Each returned dict contains all task fields plus:
        - ``blocked_by``: list of dependency task IDs that are incomplete
        - ``reason``: human-readable explanation of what's blocking
        """
        all_tasks = self.work_state.list_tasks()
        completed_ids = {t["id"] for t in all_tasks if t["status"] == "completed"}

        blocked = []
        for task in all_tasks:
            if task["status"] != "pending":
                continue
            deps = task.get("depends_on", [])
            if isinstance(deps, str):
                deps = json.loads(deps)
            unsatisfied = [d for d in deps if d not in completed_ids]
            if unsatisfied:
                blocked.append(
                    {
                        **task,
                        "blocked_by": unsatisfied,
                        "reason": (
                            f"Waiting on {len(unsatisfied)} incomplete "
                            f"dependenc{'y' if len(unsatisfied) == 1 else 'ies'}: "
                            f"{', '.join(unsatisfied)}"
                        ),
                    }
                )
        return blocked

    def get_recently_completed_tasks(self, hours: float = 1.0) -> list[dict]:
        """Tasks completed within the last *hours* hours.

        Uses the ``completed`` timestamp field on each task. Tasks without
        a ``completed`` timestamp or completed earlier are excluded.
        Sorted most-recently-completed first.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_str = cutoff.isoformat()
        completed = self.work_state.list_tasks(status="completed")
        recent = []
        for t in completed:
            ts = t.get("completed", "")
            if not ts:
                continue
            # Normalise: timestamps may lack timezone info
            if ts >= cutoff_str[:19]:  # compare at least YYYY-MM-DDTHH:MM:SS
                recent.append(t)
        recent.sort(key=lambda t: t.get("completed", ""), reverse=True)
        return recent

    # ------------------------------------------------------------------
    # Failure history
    # ------------------------------------------------------------------

    def get_task_failure_history(self, task_id: str) -> list[dict]:
        """Get failure reasons for each attempt of a task.

        Reads task_failed mutations and enriches with the last assistant
        message from the session log to provide actionable context.
        Deduplicates by attempt number (keeps the last failure per attempt).
        Only returns failures from the current task lifecycle (based on the
        most recent task_created mutation for this task_id).
        """
        mutations = self.work_state.mutation_log.read_all()

        # Find the seq of the most recent task_created for this task
        created_seq = 0
        for m in mutations:
            if m["type"] == "task_created" and m.get("data", {}).get("id") == task_id:
                created_seq = m.get("seq", 0)
            elif m["type"] == "task_created" and m.get("task_id") == task_id:
                created_seq = m.get("seq", 0)

        # Collect failures after the most recent creation, dedup by attempt
        by_attempt: dict[int, dict] = {}
        for m in mutations:
            if m.get("seq", 0) < created_seq:
                continue
            if m.get("task_id") == task_id and m["type"] == "task_failed":
                attempt = m["data"].get("attempt", 0) or 0
                by_attempt[attempt] = {
                    "attempt": attempt,
                    "reason": m.get("reason", "Unknown"),
                    "merge_conflict": m["data"].get("merge_conflict", False),
                    "exit_code": m["data"].get("exit_code"),
                }

        # Enrich with last assistant message from session log
        for info in by_attempt.values():
            attempt = info["attempt"]
            if not attempt:
                continue
            try:
                entries = self.session_logger.read_session(task_id, attempt)
                last_msg = self._extract_last_assistant_text(entries)
                if last_msg:
                    info["last_activity"] = last_msg
            except Exception:
                pass

        failures = sorted(by_attempt.values(), key=lambda f: f.get("attempt", 0))
        return failures

    @staticmethod
    def _extract_last_assistant_text(
        entries: list[dict], max_len: int = 120
    ) -> str | None:
        """Extract the last meaningful assistant message from session entries."""
        import json as _json

        last_text = None
        for entry in entries:
            if entry.get("type") != "stream_event":
                continue
            try:
                inner = _json.loads(entry.get("content", "{}"))
            except (ValueError, TypeError):
                continue
            if inner.get("type") != "assistant":
                continue
            for block in inner.get("message", {}).get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block["text"].strip()
                    if len(text) > 30:
                        last_text = text

        if not last_text:
            return None
        # Truncate to first line or max_len
        first_line = last_text.split("\n")[0].strip()
        if len(first_line) > max_len:
            return first_line[: max_len - 3] + "..."
        return first_line

    # ------------------------------------------------------------------
    # Event / stream queries
    # ------------------------------------------------------------------

    def get_recent_events(self, n: int = 50) -> list[dict]:
        """Most recent *n* audit log events across all log files."""
        return self.audit_log.read_recent(n)

    def get_task_stream_events(self, task_id: str) -> list[dict]:
        """Stream events from the latest session attempt for a task.

        Returns only entries of type ``stream_event`` from the most
        recent attempt's session log.  Returns ``[]`` if the task has
        no recorded session.
        """
        latest_attempt = self.session_logger.get_latest_attempt(task_id)
        if latest_attempt == 0:
            return []
        entries = self.session_logger.read_session(task_id, latest_attempt)
        return [e for e in entries if e.get("type") == "stream_event"]

    # ------------------------------------------------------------------
    # Cost queries
    # ------------------------------------------------------------------

    def get_cost_summary(self) -> dict:
        """Get cost summary from today's task_cost audit events.

        Returns a dict with:
        - ``total_cost_usd``: float — sum of all task costs today
        - ``task_costs``: list of dicts, each with task_id, cost_usd,
          input_tokens, output_tokens, cache_tokens, duration_ms
        """
        events = self.audit_log.read_today()
        cost_events = [e for e in events if e.get("event_type") == "task_cost"]

        total = sum(float(e.get("cost_usd", 0)) for e in cost_events)
        task_costs = []
        for e in cost_events:
            task_costs.append(
                {
                    "task_id": e.get("task_id", "unknown"),
                    "cost_usd": float(e.get("cost_usd", 0)),
                    "input_tokens": e.get("input_tokens", 0),
                    "output_tokens": e.get("output_tokens", 0),
                    "cache_tokens": e.get("cache_tokens", 0),
                    "duration_ms": e.get("duration_ms", 0),
                }
            )

        return {
            "total_cost_usd": total,
            "task_costs": task_costs,
        }
