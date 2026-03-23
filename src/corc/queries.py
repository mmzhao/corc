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
        """
        running = self.work_state.list_tasks(status="running")
        result = []
        for task in running:
            agents = self.work_state.get_agents_for_task(task["id"])
            result.append({**task, "agents": agents})
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
