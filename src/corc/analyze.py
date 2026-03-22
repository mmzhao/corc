"""Cost analysis, duration trends, and failure reporting from audit log data.

Reads immutable audit log events to produce cost breakdowns, duration trends,
failure summaries, and configurable cost threshold alerts.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from corc.audit import AuditLog


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CostAlertConfig:
    """Cost alert thresholds loaded from .corc/config.yaml."""

    daily_limit_usd: float = 50.0
    project_limit_usd: float = 200.0
    task_limit_usd: float = 10.0
    enabled: bool = True


def load_alert_config(corc_dir: Path) -> CostAlertConfig:
    """Load cost alert thresholds from .corc/config.yaml.

    Example config:
        alerts:
          cost:
            enabled: true
            daily_limit_usd: 50.0
            project_limit_usd: 200.0
            task_limit_usd: 10.0
    """
    config_path = Path(corc_dir) / "config.yaml"
    if not config_path.exists():
        return CostAlertConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    alerts = raw.get("alerts", {})
    cost = alerts.get("cost", {})

    return CostAlertConfig(
        daily_limit_usd=float(cost.get("daily_limit_usd", 50.0)),
        project_limit_usd=float(cost.get("project_limit_usd", 200.0)),
        task_limit_usd=float(cost.get("task_limit_usd", 10.0)),
        enabled=bool(cost.get("enabled", True)),
    )


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CostBreakdown:
    """Aggregated cost data."""

    total_usd: float = 0.0
    by_task: dict[str, float] = field(default_factory=dict)
    by_role: dict[str, float] = field(default_factory=dict)
    by_project: dict[str, float] = field(default_factory=dict)
    event_count: int = 0


@dataclass
class CostAlert:
    """A triggered cost alert."""

    alert_type: str  # "daily", "project", "task"
    current_usd: float
    threshold_usd: float
    subject: str  # date string, project name, or task_id
    message: str


@dataclass
class DurationEntry:
    """A single task duration record."""

    task_id: str
    duration_s: float
    timestamp: str
    attempt: int = 1
    exit_code: int = 0


@dataclass
class FailureEntry:
    """A single task failure record."""

    task_id: str
    timestamp: str
    exit_code: int
    attempt: int = 1
    name: str = ""


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------


def _extract_cost_events(events: list[dict]) -> list[dict]:
    """Filter events that carry cost information."""
    return [e for e in events if e.get("cost_usd") is not None]


def _extract_dispatch_events(events: list[dict]) -> list[dict]:
    """Filter dispatch-complete events (carry duration_s)."""
    return [e for e in events if e.get("event_type") == "task_dispatch_complete"]


def _extract_failure_events(events: list[dict]) -> list[dict]:
    """Filter task failure events."""
    return [e for e in events if e.get("event_type") == "task_failed"]


def aggregate_costs(events: list[dict]) -> CostBreakdown:
    """Aggregate cost_usd across events into a breakdown.

    Events are expected to have optional fields:
      - cost_usd: float
      - task_id: str
      - role: str
      - project: str
    """
    breakdown = CostBreakdown()
    cost_events = _extract_cost_events(events)
    breakdown.event_count = len(cost_events)

    for e in cost_events:
        cost = float(e["cost_usd"])
        breakdown.total_usd += cost

        task_id = e.get("task_id", "unknown")
        breakdown.by_task[task_id] = breakdown.by_task.get(task_id, 0.0) + cost

        role = e.get("role", "unknown")
        breakdown.by_role[role] = breakdown.by_role.get(role, 0.0) + cost

        project = e.get("project", "unassigned")
        breakdown.by_project[project] = breakdown.by_project.get(project, 0.0) + cost

    return breakdown


def compute_costs_today(audit_log: AuditLog) -> CostBreakdown:
    """Compute cost breakdown for today's events."""
    events = audit_log.read_today()
    return aggregate_costs(events)


def compute_costs_project(
    audit_log: AuditLog,
    project: str,
    since: str | None = None,
) -> CostBreakdown:
    """Compute cost breakdown filtered by project name.

    Since audit events may or may not carry a 'project' field,
    this filters on project= matching the provided name.
    """
    events = audit_log.read_all(since=since)
    project_events = [e for e in events if e.get("project") == project]
    return aggregate_costs(project_events)


def compute_duration_trends(
    audit_log: AuditLog,
    last_n: int = 20,
) -> list[DurationEntry]:
    """Return the last N task durations from dispatch-complete events.

    Sorted by timestamp ascending (oldest first).
    """
    events = audit_log.read_all()
    dispatch_events = _extract_dispatch_events(events)

    entries = []
    for e in dispatch_events:
        duration = e.get("duration_s")
        if duration is not None:
            entries.append(
                DurationEntry(
                    task_id=e.get("task_id", "unknown"),
                    duration_s=float(duration),
                    timestamp=e.get("timestamp", ""),
                    attempt=e.get("attempt", 1),
                    exit_code=e.get("exit_code", 0),
                )
            )

    # Return last N sorted by timestamp
    entries.sort(key=lambda x: x.timestamp)
    return entries[-last_n:]


def compute_failures(
    audit_log: AuditLog,
    since: str | None = None,
) -> list[FailureEntry]:
    """Return failure events, optionally filtered by since date."""
    events = audit_log.read_all(since=since)
    failure_events = _extract_failure_events(events)

    entries = []
    for e in failure_events:
        entries.append(
            FailureEntry(
                task_id=e.get("task_id", "unknown"),
                timestamp=e.get("timestamp", ""),
                exit_code=e.get("exit_code", 1),
                attempt=e.get("attempt", 1),
                name=e.get("name", ""),
            )
        )

    entries.sort(key=lambda x: x.timestamp)
    return entries


# ---------------------------------------------------------------------------
# Cost alerts
# ---------------------------------------------------------------------------


def check_cost_alerts(
    audit_log: AuditLog,
    config: CostAlertConfig,
) -> list[CostAlert]:
    """Check all cost thresholds and return any triggered alerts.

    Checks:
      1. Daily spend vs daily_limit_usd
      2. Per-project spend vs project_limit_usd
      3. Per-task spend vs task_limit_usd
    """
    if not config.enabled:
        return []

    alerts: list[CostAlert] = []

    # --- Daily limit ---
    today_breakdown = compute_costs_today(audit_log)
    if today_breakdown.total_usd > config.daily_limit_usd:
        today_str = time.strftime("%Y-%m-%d", time.gmtime())
        alerts.append(
            CostAlert(
                alert_type="daily",
                current_usd=today_breakdown.total_usd,
                threshold_usd=config.daily_limit_usd,
                subject=today_str,
                message=(
                    f"Daily cost ${today_breakdown.total_usd:.2f} exceeds "
                    f"threshold ${config.daily_limit_usd:.2f}"
                ),
            )
        )

    # --- Project limits (all-time) ---
    all_events = audit_log.read_all()
    all_breakdown = aggregate_costs(all_events)
    for project, cost in all_breakdown.by_project.items():
        if cost > config.project_limit_usd:
            alerts.append(
                CostAlert(
                    alert_type="project",
                    current_usd=cost,
                    threshold_usd=config.project_limit_usd,
                    subject=project,
                    message=(
                        f"Project '{project}' cost ${cost:.2f} exceeds "
                        f"threshold ${config.project_limit_usd:.2f}"
                    ),
                )
            )

    # --- Task limits ---
    for task_id, cost in all_breakdown.by_task.items():
        if cost > config.task_limit_usd:
            alerts.append(
                CostAlert(
                    alert_type="task",
                    current_usd=cost,
                    threshold_usd=config.task_limit_usd,
                    subject=task_id,
                    message=(
                        f"Task '{task_id}' cost ${cost:.2f} exceeds "
                        f"threshold ${config.task_limit_usd:.2f}"
                    ),
                )
            )

    return alerts


# ---------------------------------------------------------------------------
# Formatting helpers for CLI output
# ---------------------------------------------------------------------------


def format_cost_breakdown(
    breakdown: CostBreakdown, title: str = "Cost Breakdown"
) -> str:
    """Format a CostBreakdown into human-readable text."""
    lines = [title, "=" * len(title)]

    if breakdown.event_count == 0:
        lines.append("No cost data found.")
        return "\n".join(lines)

    lines.append(f"Total: ${breakdown.total_usd:.2f} ({breakdown.event_count} events)")
    lines.append("")

    if breakdown.by_task:
        lines.append("By Task:")
        for task_id, cost in sorted(breakdown.by_task.items(), key=lambda x: -x[1]):
            lines.append(f"  {task_id:<20} ${cost:.2f}")
        lines.append("")

    if breakdown.by_role:
        lines.append("By Role:")
        for role, cost in sorted(breakdown.by_role.items(), key=lambda x: -x[1]):
            lines.append(f"  {role:<20} ${cost:.2f}")
        lines.append("")

    if breakdown.by_project:
        lines.append("By Project:")
        for project, cost in sorted(breakdown.by_project.items(), key=lambda x: -x[1]):
            lines.append(f"  {project:<20} ${cost:.2f}")

    return "\n".join(lines)


def format_duration_trends(entries: list[DurationEntry]) -> str:
    """Format duration entries into human-readable text."""
    lines = ["Duration Trends", "==============="]

    if not entries:
        lines.append("No duration data found.")
        return "\n".join(lines)

    total = sum(e.duration_s for e in entries)
    avg = total / len(entries)
    min_d = min(e.duration_s for e in entries)
    max_d = max(e.duration_s for e in entries)

    lines.append(
        f"Tasks: {len(entries)}  Avg: {avg:.1f}s  Min: {min_d:.1f}s  Max: {max_d:.1f}s"
    )
    lines.append("")

    for e in entries:
        status = "✅" if e.exit_code == 0 else "❌"
        ts = e.timestamp[:19].replace("T", " ") if e.timestamp else ""
        lines.append(
            f"  {status} {e.task_id:<15} {e.duration_s:>8.1f}s  "
            f"attempt={e.attempt}  {ts}"
        )

    return "\n".join(lines)


def format_failures(entries: list[FailureEntry]) -> str:
    """Format failure entries into human-readable text."""
    lines = ["Failure Report", "=============="]

    if not entries:
        lines.append("No failures found.")
        return "\n".join(lines)

    lines.append(f"Total failures: {len(entries)}")
    lines.append("")

    # Group by task_id
    by_task: dict[str, list[FailureEntry]] = {}
    for e in entries:
        by_task.setdefault(e.task_id, []).append(e)

    for task_id, failures in sorted(by_task.items()):
        lines.append(f"  {task_id}: {len(failures)} failure(s)")
        for f in failures:
            ts = f.timestamp[:19].replace("T", " ") if f.timestamp else ""
            lines.append(f"    attempt={f.attempt}  exit_code={f.exit_code}  {ts}")

    return "\n".join(lines)


def format_alerts(alerts: list[CostAlert]) -> str:
    """Format cost alerts into human-readable text."""
    if not alerts:
        return "No cost alerts."

    lines = ["⚠️  Cost Alerts", "==============="]
    for a in alerts:
        lines.append(f"  🔴 [{a.alert_type}] {a.message}")

    return "\n".join(lines)
