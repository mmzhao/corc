"""TUI v2 — Active-plan-focused dashboard.

Shows only what matters *right now*:
  - Running tasks (prominent, with elapsed time + agent info)
  - Ready tasks (marked as dispatchable)
  - Blocked tasks (with dependency info)
  - Recently completed tasks (dimmed, last hour)
  - Event stream (color-coded)

Historical completed tasks from old phases are hidden.
Data comes from queries.py (QueryAPI data layer).

Press 'q' to quit, or Ctrl+C.
"""

import sys
import threading
import time
from datetime import datetime, timezone

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live


# ── Event type -> Rich style mapping ─────────────────────────────────────

EVENT_STYLES = {
    "task_created": "cyan",
    "task_dispatched": "yellow",
    "task_dispatch_complete": "yellow",
    "task_completed": "green bold",
    "task_failed": "red bold",
    "task_started": "yellow",
    "escalation": "red bold",
    "pause": "red",
    "resume": "green",
    # Streaming event types -- real-time agent visibility
    "tool_use": "magenta",
    "assistant_message": "blue",
}


# ── Elapsed time helper ──────────────────────────────────────────────────


def _elapsed_since(iso_ts: str) -> str:
    """Human-readable elapsed time since an ISO timestamp.

    Returns e.g. '2m 30s', '1h 5m', '< 1s'.
    Gracefully returns '' on parse failure.
    """
    if not iso_ts:
        return ""
    try:
        # Strip trailing Z, handle with/without timezone
        clean = iso_ts.replace("Z", "+00:00")
        if "+" not in clean and clean.count("-") <= 2:
            # Naive timestamp — assume UTC
            clean += "+00:00"
        start = datetime.fromisoformat(clean)
        now = datetime.now(timezone.utc)
        delta = now - start
        total_s = int(delta.total_seconds())
        if total_s < 0:
            return ""
        if total_s < 1:
            return "< 1s"
        if total_s < 60:
            return f"{total_s}s"
        minutes = total_s // 60
        seconds = total_s % 60
        if minutes < 60:
            return f"{minutes}m {seconds}s"
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours}h {mins}m"
    except (ValueError, TypeError, OverflowError):
        return ""


# ── Panel builders (testable, pure rendering) ───────────────────────────


def build_active_plan_panel(
    running_tasks: list[dict],
    ready_tasks: list[dict],
    blocked_tasks: list[dict],
    recently_completed: list[dict],
    other_active: list[dict] | None = None,
) -> Panel:
    """Build the active plan panel showing only current/relevant tasks.

    Sections (in display order):
      1. RUNNING  — prominent, yellow, with elapsed time + agent info
      2. READY    — dispatchable, cyan
      3. BLOCKED  — with dependency info, dim
      4. Other active (escalated, failed, pending_merge, etc.) — red/yellow
      5. RECENTLY COMPLETED — dimmed, last hour
    """
    lines: list[Text] = []

    # Helper to get task display name
    def _name(t: dict) -> str:
        return t.get("name") or t.get("id", "?")[:12]

    def _tid(t: dict) -> str:
        return t.get("id", "")[:8]

    # ── Running tasks (prominent) ──────────────────────────────────
    if running_tasks:
        header = Text()
        header.append("  RUNNING", style="bold yellow")
        header.append(f"  ({len(running_tasks)})", style="yellow")
        lines.append(header)

        for t in running_tasks:
            line = Text()
            line.append("  🔄 ", style="yellow")
            line.append(f"{_name(t):<30}", style="bold yellow")
            line.append(f"  {_tid(t)}", style="dim")

            # Elapsed time
            started = t.get("started") or t.get("updated", "")
            elapsed = _elapsed_since(started)
            if elapsed:
                line.append(f"  ⏱ {elapsed}", style="yellow")

            # Agent info
            agents = t.get("agents", [])
            if agents:
                for ag in agents:
                    role = ag.get("role", "agent")
                    pid = ag.get("pid", "")
                    ag_status = ag.get("status", "")
                    line.append(f"  [{role}", style="dim yellow")
                    if pid:
                        line.append(f" pid={pid}", style="dim")
                    if ag_status:
                        line.append(f" {ag_status}", style="dim")
                    line.append("]", style="dim yellow")

            lines.append(line)
        lines.append(Text(""))

    # ── Ready tasks (dispatchable) ─────────────────────────────────
    if ready_tasks:
        header = Text()
        header.append("  READY", style="bold cyan")
        header.append(f"  ({len(ready_tasks)})", style="cyan")
        header.append("  — dispatchable", style="dim cyan")
        lines.append(header)

        for t in ready_tasks:
            line = Text()
            line.append("  ▶ ", style="cyan")
            line.append(f"{_name(t):<30}", style="cyan")
            line.append(f"  {_tid(t)}", style="dim")
            pri = t.get("priority", 100)
            if pri != 100:
                line.append(f"  pri={pri}", style="dim")
            lines.append(line)
        lines.append(Text(""))

    # ── Blocked tasks ──────────────────────────────────────────────
    if blocked_tasks:
        header = Text()
        header.append("  BLOCKED", style="bold bright_black")
        header.append(f"  ({len(blocked_tasks)})", style="bright_black")
        lines.append(header)

        for t in blocked_tasks:
            line = Text()
            line.append("  ◻ ", style="bright_black")
            line.append(f"{_name(t):<30}", style="bright_black")
            line.append(f"  {_tid(t)}", style="dim")

            # Show what blocks this task
            reason = t.get("reason", "")
            blocked_by = t.get("blocked_by", [])
            if reason:
                line.append(f"  {reason}", style="dim red")
            elif blocked_by:
                line.append(
                    f"  waiting on: {', '.join(b[:8] for b in blocked_by)}",
                    style="dim red",
                )
            lines.append(line)
        lines.append(Text(""))

    # ── Other active (escalated, failed, pending_merge, etc.) ──────
    if other_active:
        header = Text()
        header.append("  OTHER", style="bold red")
        header.append(f"  ({len(other_active)})", style="red")
        lines.append(header)

        for t in other_active:
            status = t.get("status", "unknown")
            line = Text()
            icon = {
                "failed": "❌",
                "escalated": "🚨",
                "pending_merge": "🔀",
                "assigned": "📋",
                "handed_off": "↗",
            }.get(status, "?")
            style = {
                "failed": "red",
                "escalated": "red bold",
                "pending_merge": "magenta",
                "assigned": "yellow",
                "handed_off": "yellow",
            }.get(status, "white")
            line.append(f"  {icon} ", style=style)
            line.append(f"{_name(t):<30}", style=style)
            line.append(f"  {_tid(t)}", style="dim")
            line.append(f"  [{status}]", style="dim")
            lines.append(line)
        lines.append(Text(""))

    # ── Recently completed (dimmed) ────────────────────────────────
    if recently_completed:
        header = Text()
        header.append("  RECENTLY COMPLETED", style="dim green")
        header.append(f"  ({len(recently_completed)})", style="dim")
        lines.append(header)

        for t in recently_completed:
            line = Text()
            line.append("  ✅ ", style="dim green")
            line.append(f"{_name(t):<30}", style="dim")
            line.append(f"  {_tid(t)}", style="dim")
            completed_ts = t.get("completed", "")
            elapsed = _elapsed_since(completed_ts)
            if elapsed:
                line.append(f"  {elapsed} ago", style="dim")
            lines.append(line)
        lines.append(Text(""))

    # ── Empty state ────────────────────────────────────────────────
    if not lines:
        content = Text(
            "No active tasks. Create tasks with: corc task create", style="dim"
        )
    else:
        # Summary line at top
        total = (
            len(running_tasks)
            + len(ready_tasks)
            + len(blocked_tasks)
            + len(other_active or [])
        )
        summary = Text()
        summary.append(f"  {total} active", style="bold")
        if running_tasks:
            summary.append(f"  {len(running_tasks)} running", style="yellow")
        if ready_tasks:
            summary.append(f"  {len(ready_tasks)} ready", style="cyan")
        if blocked_tasks:
            summary.append(f"  {len(blocked_tasks)} blocked", style="bright_black")
        if recently_completed:
            summary.append(
                f"  {len(recently_completed)} done recently", style="dim green"
            )
        lines.insert(0, summary)
        lines.insert(1, Text(""))
        content = Text("\n").join(lines)

    return Panel(
        content,
        title="[bold] Active Plan [/bold]",
        border_style="blue",
    )


def build_event_panel(events: list[dict], max_events: int = 20) -> Panel:
    """Build the color-coded event stream panel.

    Each event line shows: timestamp  event_type  task_id  extras
    Colour is determined by event type.
    """
    display_events = events[-max_events:] if len(events) > max_events else events

    if not display_events:
        content = Text("No events yet. Waiting...", style="dim")
    else:
        lines: list[Text] = []
        for e in display_events:
            ts = e.get("timestamp", "")[:19].replace("T", " ")
            etype = e.get("event_type", "unknown")
            tid = e.get("task_id", "")[:8] if e.get("task_id") else ""
            style = EVENT_STYLES.get(etype, "white")

            extra_parts: list[str] = []
            if e.get("duration_s"):
                extra_parts.append(f"({e['duration_s']:.1f}s)")
            if e.get("exit_code") is not None and e["exit_code"] != 0:
                extra_parts.append(f"exit={e['exit_code']}")
            if e.get("name"):
                extra_parts.append(e["name"])
            # Streaming event extras: tool name and assistant content
            if e.get("tool_name"):
                extra_parts.append(f"-> {e['tool_name']}")
            if e.get("tool_input"):
                extra_parts.append(e["tool_input"])
            extra = " ".join(extra_parts)

            line = Text()
            line.append(f"{ts}  ", style="dim")
            line.append(f"{etype:<25}", style=style)
            line.append(f"  {tid}", style="bold")
            if extra:
                line.append(f"  {extra}", style="dim")
            lines.append(line)

            # Show full assistant message content on subsequent lines
            # (verbose output -- no truncation)
            if e.get("content") and etype == "assistant_message":
                for content_line in e["content"].split("\n"):
                    detail = Text()
                    detail.append("                           ", style="dim")
                    detail.append(content_line, style="blue dim")
                    lines.append(detail)

        content = Text("\n").join(lines)

    return Panel(
        content,
        title="[bold] Events [/bold]",
        subtitle="q to quit | Ctrl+C",
        border_style="green",
    )


# ── Legacy compatibility: build_dag_panel ────────────────────────────────


def build_dag_panel(tasks: list[dict]) -> Panel:
    """Build the DAG status panel from current task list.

    Legacy compatibility wrapper. Reuses render_ascii_dag() with ANSI
    output, converts to Rich Text via Text.from_ansi().
    """
    from corc.dag import render_ascii_dag

    if not tasks:
        content = Text(
            "No tasks found. Create tasks with: corc task create", style="dim"
        )
    else:
        ascii_output = render_ascii_dag(tasks, use_color=True)
        content = Text.from_ansi(ascii_output)

    return Panel(
        content,
        title="[bold] DAG [/bold]",
        border_style="blue",
    )


# ── Dashboard builders ───────────────────────────────────────────────────


def build_dashboard(
    tasks: list[dict],
    events: list[dict],
    max_events: int = 20,
) -> Layout:
    """Build the legacy two-panel dashboard layout (DAG + events).

    Kept for backward compatibility. New code should use
    build_active_dashboard() with QueryAPI data.
    """
    layout = Layout()
    layout.split_column(
        Layout(name="dag", ratio=1),
        Layout(name="events", ratio=1),
    )
    layout["dag"].update(build_dag_panel(tasks))
    layout["events"].update(build_event_panel(events, max_events))
    return layout


def build_active_dashboard(
    running_tasks: list[dict],
    ready_tasks: list[dict],
    blocked_tasks: list[dict],
    recently_completed: list[dict],
    other_active: list[dict],
    events: list[dict],
    max_events: int = 20,
) -> Layout:
    """Build the active-plan-focused dashboard layout.

    Returns a Rich Layout with:
      - top: Active plan panel (ratio 2 — more prominent)
      - bottom: event stream panel (ratio 1)
    """
    layout = Layout()
    layout.split_column(
        Layout(name="active_plan", ratio=2),
        Layout(name="events", ratio=1),
    )
    layout["active_plan"].update(
        build_active_plan_panel(
            running_tasks,
            ready_tasks,
            blocked_tasks,
            recently_completed,
            other_active,
        )
    )
    layout["events"].update(build_event_panel(events, max_events))
    return layout


# ── Live dashboard runner ────────────────────────────────────────────────


def _listen_for_quit(stop_event: threading.Event) -> None:
    """Background thread: listen for 'q' keypress to signal quit.

    Uses cbreak mode on Unix terminals. Silently falls back to doing nothing
    if stdin is not a real terminal (pipes, CI, tests).
    """
    try:
        import tty
        import termios
        import select

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not stop_event.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                if ready:
                    ch = sys.stdin.read(1)
                    if ch.lower() == "q":
                        stop_event.set()
                        break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except (ImportError, OSError, ValueError, AttributeError):
        # Not a real terminal -- just wait for the stop event
        while not stop_event.is_set():
            stop_event.wait(0.5)


def run_dashboard(
    get_tasks,
    get_events,
    max_events: int = 20,
    refresh_per_second: float = 2.0,
    console: Console | None = None,
) -> None:
    """Run the legacy live two-panel dashboard until 'q' or Ctrl+C.

    Args:
        get_tasks: Callable returning list[dict] of current tasks.
        get_events: Callable returning list[dict] of recent events.
        max_events: Maximum events to show in the bottom panel.
        refresh_per_second: How often to redraw.
        console: Optional Rich Console (for testing).
    """
    console = console or Console()
    stop_event = threading.Event()

    key_thread = threading.Thread(
        target=_listen_for_quit, args=(stop_event,), daemon=True
    )
    key_thread.start()

    interval = 1.0 / refresh_per_second

    try:
        with Live(
            console=console, refresh_per_second=refresh_per_second, screen=False
        ) as live:
            while not stop_event.is_set():
                tasks = get_tasks()
                events = get_events()
                dashboard = build_dashboard(tasks, events, max_events)
                live.update(dashboard)
                stop_event.wait(interval)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()


def run_active_dashboard(
    query_api,
    max_events: int = 20,
    refresh_per_second: float = 2.0,
    console: Console | None = None,
) -> None:
    """Run the active-plan-focused dashboard until 'q' or Ctrl+C.

    Uses QueryAPI to fetch only relevant data:
      - Running tasks with agent info
      - Ready (dispatchable) tasks
      - Blocked tasks with reasons
      - Recently completed (last hour)
      - Recent events

    Args:
        query_api: A QueryAPI instance (from corc.queries).
        max_events: Maximum events to show in the bottom panel.
        refresh_per_second: How often to redraw.
        console: Optional Rich Console (for testing).
    """
    console = console or Console()
    stop_event = threading.Event()

    key_thread = threading.Thread(
        target=_listen_for_quit, args=(stop_event,), daemon=True
    )
    key_thread.start()

    interval = 1.0 / refresh_per_second

    # IDs we've seen as running/ready/blocked/completed to filter "other active"
    _CATEGORIZED_STATUSES = frozenset({"running", "completed"})

    try:
        with Live(
            console=console, refresh_per_second=refresh_per_second, screen=False
        ) as live:
            while not stop_event.is_set():
                # Refresh underlying state
                query_api.work_state.refresh()

                # Fetch categorized data from QueryAPI
                running = query_api.get_running_tasks_with_agents()
                ready = query_api.get_ready_tasks()
                blocked = query_api.get_blocked_tasks_with_reasons()
                recent_done = query_api.get_recently_completed_tasks(hours=1.0)
                events = query_api.get_recent_events(max_events)

                # "Other active" = active tasks not in running/ready/blocked
                categorized_ids = (
                    {t["id"] for t in running}
                    | {t["id"] for t in ready}
                    | {t["id"] for t in blocked}
                )
                all_active = query_api.get_active_plan_tasks()
                other = [t for t in all_active if t["id"] not in categorized_ids]

                dashboard = build_active_dashboard(
                    running,
                    ready,
                    blocked,
                    recent_done,
                    other,
                    events,
                    max_events,
                )
                live.update(dashboard)
                stop_event.wait(interval)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
