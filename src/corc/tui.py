"""TUI v1 — Two-panel dashboard with DAG view and event stream.

Uses Rich's Layout + Live to display:
  - Top panel: DAG with live status updates (reuses dag.py rendering)
  - Bottom panel: Color-coded event stream

Press 'q' to quit, or Ctrl+C.
"""

import sys
import threading
import time

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.live import Live

from corc.dag import render_ascii_dag, compute_progress


# ── Event type → Rich style mapping ─────────────────────────────────────

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
}


# ── Panel builders (testable, pure rendering) ───────────────────────────

def build_dag_panel(tasks: list[dict]) -> Panel:
    """Build the DAG status panel from current task list.

    Reuses render_ascii_dag() with ANSI output, converts to Rich Text via
    Text.from_ansi() so the existing colour logic is preserved.
    """
    if not tasks:
        content = Text("No tasks found. Create tasks with: corc task create", style="dim")
    else:
        ascii_output = render_ascii_dag(tasks, use_color=True)
        content = Text.from_ansi(ascii_output)

    return Panel(
        content,
        title="[bold] DAG [/bold]",
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
            extra = " ".join(extra_parts)

            line = Text()
            line.append(f"{ts}  ", style="dim")
            line.append(f"{etype:<25}", style=style)
            line.append(f"  {tid}", style="bold")
            if extra:
                line.append(f"  {extra}", style="dim")
            lines.append(line)

        content = Text("\n").join(lines)

    return Panel(
        content,
        title="[bold] Events [/bold]",
        subtitle="q to quit | Ctrl+C",
        border_style="green",
    )


def build_dashboard(
    tasks: list[dict],
    events: list[dict],
    max_events: int = 20,
) -> Layout:
    """Build the full two-panel dashboard layout.

    Returns a Rich Layout with:
      - top: DAG status panel (ratio 1)
      - bottom: event stream panel (ratio 1)

    The layout resizes automatically with the terminal.
    """
    layout = Layout()
    layout.split_column(
        Layout(name="dag", ratio=1),
        Layout(name="events", ratio=1),
    )
    layout["dag"].update(build_dag_panel(tasks))
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
        # Not a real terminal — just wait for the stop event
        while not stop_event.is_set():
            stop_event.wait(0.5)


def run_dashboard(
    get_tasks,
    get_events,
    max_events: int = 20,
    refresh_per_second: float = 2.0,
    console: Console | None = None,
) -> None:
    """Run the live two-panel dashboard until 'q' or Ctrl+C.

    Args:
        get_tasks: Callable returning list[dict] of current tasks.
        get_events: Callable returning list[dict] of recent events.
        max_events: Maximum events to show in the bottom panel.
        refresh_per_second: How often to redraw.
        console: Optional Rich Console (for testing).
    """
    console = console or Console()
    stop_event = threading.Event()

    # Start keyboard listener in a daemon thread
    key_thread = threading.Thread(target=_listen_for_quit, args=(stop_event,), daemon=True)
    key_thread.start()

    interval = 1.0 / refresh_per_second

    try:
        with Live(console=console, refresh_per_second=refresh_per_second, screen=False) as live:
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
