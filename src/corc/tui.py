"""TUI v2 — Active-plan-focused dashboard.

Layout: left two-thirds split vertically into DAG status (top) and
agent detail with streaming (bottom); right one-third is a tall
event stream column.

Shows only what matters *right now*:
  - DAG status / Active Plan (top-left): running, ready, blocked tasks
  - Agent detail (bottom-left): live tool calls, reasoning, checklist
  - Event stream (right): color-coded event log

Historical completed tasks from old phases are hidden.
Data comes from queries.py (QueryAPI data layer).

Press 'q' to quit, or Ctrl+C.  ↑/↓ to scroll streaming detail.
"""

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live


# ── Auto-reload support ──────────────────────────────────────────────────

# Files the TUI watches for changes (relative to this module's directory).
_TUI_WATCH_FILES = ("tui.py", "queries.py")


class ReloadRequested(Exception):
    """Raised when watched source files change and the TUI should restart."""

    def __init__(self, changed_files: list[str]):
        self.changed_files = changed_files
        super().__init__(f"Source files changed: {', '.join(changed_files)}")


def _get_watched_file_mtimes() -> dict[str, float]:
    """Snapshot modification times for the TUI source files we watch.

    Returns a dict mapping absolute file path → mtime for each file in
    ``_TUI_WATCH_FILES`` that exists on disk.
    """
    src_dir = Path(__file__).resolve().parent
    mtimes: dict[str, float] = {}
    for name in _TUI_WATCH_FILES:
        filepath = src_dir / name
        try:
            mtimes[str(filepath)] = filepath.stat().st_mtime
        except OSError:
            pass
    return mtimes


def _check_for_source_changes(
    baseline: dict[str, float],
) -> list[str]:
    """Compare current mtimes against *baseline* and return changed file paths.

    Returns an empty list if nothing changed.
    """
    current = _get_watched_file_mtimes()
    changed: list[str] = []
    for path, mtime in current.items():
        if path not in baseline or baseline[path] != mtime:
            changed.append(path)
    return changed


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


def _deduplicate_agents(agents: list[dict]) -> list[dict]:
    """Filter a list of agent records to show at most one per task.

    When a task has been dispatched multiple times (e.g. after a daemon
    restart), there may be multiple agent records for the same task_id.
    This function keeps only the best agent per task:

      1. Prefer an active (non-idle) agent if one exists.
      2. Otherwise, prefer the most recently started agent.

    If agents don't have a ``task_id`` (shouldn't happen in practice),
    they are all kept.

    Returns a new list — input is not modified.
    """
    if len(agents) <= 1:
        return list(agents)

    # Group by task_id
    by_task: dict[str, list[dict]] = {}
    no_task: list[dict] = []
    for ag in agents:
        tid = ag.get("task_id")
        if tid:
            by_task.setdefault(tid, []).append(ag)
        else:
            no_task.append(ag)

    result: list[dict] = list(no_task)
    for _tid, group in by_task.items():
        if len(group) == 1:
            result.append(group[0])
            continue

        # Prefer active (non-idle) agent
        active = [a for a in group if a.get("status") not in (None, "idle")]
        if active:
            # Among active agents, pick the most recently started
            active.sort(key=lambda a: a.get("started") or "", reverse=True)
            result.append(active[0])
        else:
            # All idle — pick the most recently started
            group.sort(key=lambda a: a.get("started") or "", reverse=True)
            result.append(group[0])

    return result


def _deduplicate_task_agents(task: dict) -> dict:
    """Return a copy of *task* with its ``agents`` list deduplicated.

    Convenience wrapper around :func:`_deduplicate_agents` that operates
    on a task dict containing an ``agents`` key.  Returns the task
    unchanged if there is no ``agents`` key.
    """
    agents = task.get("agents")
    if not agents or len(agents) <= 1:
        return task
    return {**task, "agents": _deduplicate_agents(agents)}


def _format_attempt_count(task: dict) -> str | None:
    """Format the attempt count for display, e.g. 'attempt 2/3'.

    Returns ``None`` for first-attempt tasks (attempt_count == 0) so
    that no retry indicator clutters the display for normal runs.
    Only returns a string when the task is on a retry (attempt_count >= 1).

    Uses ``attempt_count`` and ``max_retries`` from the task dict.
    ``attempt_count`` represents completed attempts (0 = first try).
    Display shows ``attempt_count + 1`` as the current attempt number
    and ``max_retries + 1`` as the total allowed attempts.
    """
    attempt_count = task.get("attempt_count", 0)
    if not isinstance(attempt_count, int) or attempt_count < 1:
        return None
    max_retries = task.get("max_retries", 3)
    if not isinstance(max_retries, int):
        max_retries = 3
    # Current attempt = attempt_count + 1 (attempt_count is completed attempts)
    # Total attempts = max_retries + 1 (max_retries is number of retries after first)
    current = attempt_count + 1
    total = max_retries + 1
    return f"attempt {current}/{total}"


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


# ── Stream event helpers (testable, pure) ────────────────────────────────


def _parse_stream_content(entry: dict) -> dict | None:
    """Parse the JSON content field of a stream_event entry.

    Session log stream_event entries store the original event as a
    JSON-encoded string in the ``content`` field.  Returns the parsed
    dict or ``None`` on failure.
    """
    content = entry.get("content", "")
    if not content:
        return None
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None


def _format_tool_call(event: dict) -> str:
    """Format a tool_use event as a concise one-liner.

    Extracts the most useful input parameter for display:
      - file_path for Read/Write/Edit
      - command for Bash
      - pattern for Grep/Glob
      - Falls back to first key=value

    Returns e.g. ``'Read /src/main.py'`` or ``'Bash ls -la'``.
    """
    tool = event.get("tool", {})
    name = tool.get("name", "?")
    tool_input = tool.get("input", {})

    if "file_path" in tool_input:
        return f"{name} {tool_input['file_path']}"
    if "command" in tool_input:
        cmd = tool_input["command"]
        if len(cmd) > 60:
            cmd = cmd[:57] + "..."
        return f"{name} {cmd}"
    if "pattern" in tool_input:
        return f"{name} {tool_input['pattern']}"
    # Fallback: show first input key=value
    if tool_input:
        key = next(iter(tool_input))
        val = str(tool_input[key])
        if len(val) > 40:
            val = val[:37] + "..."
        return f"{name} {key}={val}"
    return name


def _truncate_reasoning(text: str, max_len: int = 120) -> str:
    """Truncate assistant reasoning text for single-line display.

    Collapses newlines into spaces, strips whitespace, and truncates
    with ``...`` if longer than *max_len*.
    """
    clean = text.replace("\n", " ").strip()
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


def _format_checklist_progress(checklist) -> str | None:
    """Format a task checklist as a progress string.

    *checklist* may be a JSON string or a list of dicts, each with an
    optional ``done`` boolean.  Returns e.g. ``'3/5 items done'`` or
    ``None`` if there is no valid checklist.
    """
    if not checklist:
        return None
    if isinstance(checklist, str):
        try:
            checklist = json.loads(checklist)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(checklist, list) or not checklist:
        return None

    total = len(checklist)
    done = sum(1 for item in checklist if isinstance(item, dict) and item.get("done"))
    return f"{done}/{total} items done"


def _format_attempt_count(task: dict) -> str | None:
    """Format the retry attempt count for display.

    Returns e.g. ``'attempt 2/4'`` when the task is on its second attempt
    (out of 4 total possible = 1 initial + 3 retries).

    Returns ``None`` for first-attempt tasks (attempt_count == 0) since
    no retry indicator is needed.

    Args:
        task: Task dict with optional ``attempt_count`` and ``max_retries``.
    """
    attempt_count = task.get("attempt_count", 0)
    if not attempt_count or attempt_count <= 0:
        return None
    max_retries = task.get("max_retries", 3)
    current = attempt_count + 1  # attempt_count=1 means 2nd attempt
    total = max_retries + 1  # max_retries=3 means 4 total possible attempts
    return f"attempt {current}/{total}"


def build_streaming_detail_panel(
    running_tasks: list[dict],
    stream_events_by_task: dict[str, list[dict]],
    scroll_offset: int = 0,
    max_lines: int = 30,
) -> Panel:
    """Build the streaming detail panel showing live agent activity.

    For each running task, displays:
      - Task header (name + id)
      - Checklist progress (if checklist present)
      - Tool calls with file paths (🔧)
      - Assistant reasoning truncated (💭)
      - Result summaries (✅)

    Supports scrolling via *scroll_offset* (lines from the bottom).

    Args:
        running_tasks: Currently running task dicts (may include checklist).
        stream_events_by_task: Maps task_id → list of stream_event entries
            from ``QueryAPI.get_task_stream_events()``.
        scroll_offset: Lines to scroll up from the bottom (0 = newest).
        max_lines: Maximum number of lines to display.
    """
    lines: list[Text] = []

    if not running_tasks:
        content = Text("No running agents. Waiting for dispatch...", style="dim")
        return Panel(
            content,
            title="[bold] Streaming Detail [/bold]",
            subtitle="↑/↓ scroll",
            border_style="magenta",
        )

    def _name(t: dict) -> str:
        return t.get("name") or t.get("id", "?")[:12]

    def _tid(t: dict) -> str:
        return t.get("id", "")[:8]

    for task in running_tasks:
        tid = task.get("id", "?")

        # ── Task header ────────────────────────────────────────────
        header = Text()
        header.append(f"  🔄 {_name(task)}", style="bold yellow")
        header.append(f"  ({_tid(task)})", style="dim")
        attempt_str = _format_attempt_count(task)
        if attempt_str:
            header.append(f"  🔁 {attempt_str}", style="yellow")
        lines.append(header)

        # ── Checklist progress ─────────────────────────────────────
        checklist = task.get("checklist")
        progress = _format_checklist_progress(checklist)
        if progress:
            prog_line = Text()
            prog_line.append(f"    ☑ {progress}", style="cyan")
            lines.append(prog_line)

        # ── Stream events for this task ────────────────────────────
        events = stream_events_by_task.get(tid, [])
        if not events:
            empty_line = Text()
            empty_line.append("    Waiting for stream events...", style="dim")
            lines.append(empty_line)
        else:
            # Show recent events (last 20 per task to keep panel readable)
            display_events = events[-20:]
            for entry in display_events:
                stream_type = entry.get("stream_type", "unknown")
                parsed = _parse_stream_content(entry)
                if parsed is None:
                    continue

                if stream_type == "tool_use":
                    desc = _format_tool_call(parsed)
                    line = Text()
                    line.append("    🔧 ", style="magenta")
                    line.append(desc, style="magenta")
                    lines.append(line)

                elif stream_type == "assistant":
                    # Extract text content from assistant message
                    message = parsed.get("message", {})
                    content_blocks = message.get("content", [])
                    texts = [
                        b.get("text", "")
                        for b in content_blocks
                        if b.get("type") == "text"
                    ]
                    full_text = " ".join(texts)
                    if full_text:
                        truncated = _truncate_reasoning(full_text)
                        line = Text()
                        line.append("    💭 ", style="blue")
                        line.append(truncated, style="blue dim")
                        lines.append(line)

                elif stream_type == "result":
                    result_text = parsed.get("result", "")
                    if result_text:
                        truncated = _truncate_reasoning(result_text, max_len=80)
                        line = Text()
                        line.append("    ✅ ", style="green")
                        line.append(truncated, style="green")
                        lines.append(line)

        lines.append(Text(""))  # Spacer between tasks

    # ── Apply scroll offset ────────────────────────────────────────
    total_lines = len(lines)
    if scroll_offset > 0 and total_lines > max_lines:
        end = max(0, total_lines - scroll_offset)
        start = max(0, end - max_lines)
        lines = lines[start:end]
    elif total_lines > max_lines:
        # Default: show the most recent lines (bottom)
        lines = lines[-max_lines:]

    if lines:
        content = Text("\n").join(lines)
    else:
        content = Text("No stream data yet.", style="dim")

    # Scroll indicator in subtitle
    subtitle = "↑/↓ scroll"
    if scroll_offset > 0:
        subtitle += f" | offset: {scroll_offset}"

    return Panel(
        content,
        title="[bold] Streaming Detail [/bold]",
        subtitle=subtitle,
        border_style="magenta",
    )


# ── Panel builders (testable, pure rendering) ───────────────────────────


def build_active_plan_panel(
    running_tasks: list[dict],
    ready_tasks: list[dict],
    blocked_tasks: list[dict],
    recently_completed: list[dict],
    other_active: list[dict] | None = None,
    scroll_offset: int = 0,
    max_lines: int = 40,
    focused_panel: str | None = None,
) -> Panel:
    """Build the active plan panel showing only current/relevant tasks.

    Sections (in display order):
      1. RUNNING  — prominent, yellow, with elapsed time + agent info
      2. READY    — dispatchable, cyan
      3. BLOCKED  — with dependency info, dim
      4. Other active (escalated, failed, pending_merge, etc.) — red/yellow
      5. RECENTLY COMPLETED — dimmed, last hour

    Supports scrolling via *scroll_offset* (lines from the top).
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

            # Attempt count (only shown on retries)
            attempt_str = _format_attempt_count(t)
            if attempt_str:
                line.append(f"  🔁 {attempt_str}", style="yellow")

            # Agent info — deduplicate to show only latest per task
            agents = _deduplicate_agents(t.get("agents", []))
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

            # Attempt count for failed/escalated tasks
            if status in ("failed", "escalated"):
                attempt_str = _format_attempt_count(t)
                if attempt_str:
                    line.append(f"  🔁 {attempt_str}", style=style)
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
    stream_events_by_task: dict[str, list[dict]] | None = None,
    scroll_offset: int = 0,
) -> Layout:
    """Build the active-plan-focused dashboard layout.

    Returns a Rich Layout with a left/right split:
      - Left two-thirds: split vertically 50/50 into
        - top: DAG status / Active Plan panel
        - bottom: Streaming detail panel (agent activity)
      - Right one-third: full-height event stream column

    When *stream_events_by_task* is ``None``, the left side shows only
    the active plan panel (no streaming detail).
    """
    layout = Layout()

    if stream_events_by_task is not None:
        # Three-panel layout: left (DAG + streaming) | right (events)
        layout.split_row(
            Layout(name="left", ratio=2),
            Layout(name="events", ratio=1),
        )
        layout["left"].split_column(
            Layout(name="active_plan", ratio=1),
            Layout(name="streaming", ratio=1),
        )
        layout["streaming"].update(
            build_streaming_detail_panel(
                running_tasks,
                stream_events_by_task,
                scroll_offset=scroll_offset,
            )
        )
    else:
        # Two-panel layout: left (active plan) | right (events)
        layout.split_row(
            Layout(name="left", ratio=2),
            Layout(name="events", ratio=1),
        )
        layout["left"].split_column(
            Layout(name="active_plan", ratio=1),
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


def _listen_for_keys(
    stop_event: threading.Event,
    scroll_state: dict,
) -> None:
    """Background thread: listen for 'q' and arrow keys.

    Handles:
      - ``q``: quit (sets *stop_event*)
      - ``↑`` (ESC [ A): scroll streaming panel up
      - ``↓`` (ESC [ B): scroll streaming panel down (towards newest)

    *scroll_state* is a mutable dict with key ``offset`` (int).
    Uses cbreak mode on Unix; falls back to quit-only on non-terminals.
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
                    if ch == "\x1b":
                        # Possible escape sequence (arrow key)
                        ready2, _, _ = select.select([sys.stdin], [], [], 0.05)
                        if ready2:
                            ch2 = sys.stdin.read(1)
                            if ch2 == "[":
                                ready3, _, _ = select.select([sys.stdin], [], [], 0.05)
                                if ready3:
                                    ch3 = sys.stdin.read(1)
                                    if ch3 == "A":  # Up arrow
                                        scroll_state["offset"] = (
                                            scroll_state.get("offset", 0) + 3
                                        )
                                    elif ch3 == "B":  # Down arrow
                                        scroll_state["offset"] = max(
                                            0,
                                            scroll_state.get("offset", 0) - 3,
                                        )
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
    auto_reload: bool = False,
) -> None:
    """Run the active-plan-focused dashboard until 'q' or Ctrl+C.

    Uses QueryAPI to fetch only relevant data:
      - Running tasks with agent info
      - Ready (dispatchable) tasks
      - Blocked tasks with reasons
      - Recently completed (last hour)
      - Streaming detail from session log stream events
      - Recent events

    The streaming detail panel auto-updates each refresh cycle by
    calling ``query_api.get_task_stream_events()`` for every running
    task.  Arrow keys scroll the streaming panel up/down.

    When *auto_reload* is ``True``, the dashboard monitors its own
    source files (``tui.py`` and ``queries.py``) for changes.  If a
    change is detected, the function raises :class:`ReloadRequested`
    so the caller can reload modules and restart the TUI.

    Args:
        query_api: A QueryAPI instance (from corc.queries).
        max_events: Maximum events to show in the bottom panel.
        refresh_per_second: How often to redraw.
        console: Optional Rich Console (for testing).
        auto_reload: Watch source files and raise ReloadRequested on change.

    Raises:
        ReloadRequested: When auto_reload is True and a watched source file
            is modified on disk.
    """
    console = console or Console()
    stop_event = threading.Event()
    scroll_state: dict = {"offset": 0}

    key_thread = threading.Thread(
        target=_listen_for_keys, args=(stop_event, scroll_state), daemon=True
    )
    key_thread.start()

    interval = 1.0 / refresh_per_second

    # Snapshot source file mtimes for auto-reload detection
    baseline_mtimes: dict[str, float] = {}
    if auto_reload:
        baseline_mtimes = _get_watched_file_mtimes()

    try:
        with Live(
            console=console, refresh_per_second=refresh_per_second, screen=False
        ) as live:
            while not stop_event.is_set():
                # ── Auto-reload check ──────────────────────────────
                if auto_reload:
                    changed = _check_for_source_changes(baseline_mtimes)
                    if changed:
                        # Show brief reload indicator before restarting
                        names = [os.path.basename(p) for p in changed]
                        reload_msg = Text()
                        reload_msg.append(
                            f"  ⟳ Reloading TUI — changed: {', '.join(names)}",
                            style="bold yellow",
                        )
                        live.update(
                            Panel(
                                reload_msg,
                                title="[bold] Reloading [/bold]",
                                border_style="yellow",
                            )
                        )
                        live.refresh()
                        stop_event.set()
                        raise ReloadRequested(changed)

                # Refresh underlying state
                query_api.work_state.refresh()

                # Fetch categorized data from QueryAPI
                running = query_api.get_running_tasks_with_agents()
                ready = query_api.get_ready_tasks()
                blocked = query_api.get_blocked_tasks_with_reasons()
                recent_done = query_api.get_recently_completed_tasks(hours=1.0)
                events = query_api.get_recent_events(max_events)

                # Fetch stream events for each running task
                stream_events_by_task: dict[str, list[dict]] = {}
                for task in running:
                    tid = task["id"]
                    stream_events_by_task[tid] = query_api.get_task_stream_events(tid)

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
                    stream_events_by_task=stream_events_by_task,
                    scroll_offset=scroll_state.get("offset", 0),
                )
                live.update(dashboard)
                stop_event.wait(interval)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
