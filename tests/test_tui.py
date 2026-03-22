"""Tests for TUI v1 — two-panel dashboard with DAG view and event stream."""

import io
import time
import threading

import pytest
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from corc.tui import (
    build_dag_panel,
    build_event_panel,
    build_dashboard,
    EVENT_STYLES,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_task(tid, name, depends_on=None, status="pending"):
    """Build a minimal task dict for unit tests."""
    return {
        "id": tid,
        "name": name,
        "depends_on": depends_on or [],
        "status": status,
        "done_when": "tests pass",
    }


def _render_to_str(renderable, width=120) -> str:
    """Render a Rich renderable to a plain string for assertions."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=width, color_system="truecolor")
    console.print(renderable)
    return buf.getvalue()


def _render_to_plain(renderable, width=120) -> str:
    """Render a Rich renderable to plain text (no ANSI escapes)."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=width, no_color=True)
    console.print(renderable)
    return buf.getvalue()


# ── DAG Panel ────────────────────────────────────────────────────────────

class TestBuildDagPanel:
    def test_empty_tasks(self):
        panel = build_dag_panel([])
        assert isinstance(panel, Panel)
        text = _render_to_plain(panel)
        assert "No tasks found" in text

    def test_single_task(self):
        tasks = [_make_task("a1", "parser")]
        panel = build_dag_panel(tasks)
        assert isinstance(panel, Panel)
        text = _render_to_plain(panel)
        assert "parser" in text
        assert "DAG" in text

    def test_multiple_tasks_with_statuses(self):
        tasks = [
            _make_task("a1", "parser", status="completed"),
            _make_task("a2", "schema", status="completed"),
            _make_task("a3", "fts5", depends_on=["a1"], status="running"),
            _make_task("a4", "semantic", depends_on=["a2"], status="pending"),
        ]
        panel = build_dag_panel(tasks)
        text = _render_to_plain(panel)
        assert "parser" in text
        assert "schema" in text
        assert "fts5" in text
        assert "semantic" in text

    def test_dag_panel_shows_status_icons(self):
        tasks = [
            _make_task("a1", "parser", status="completed"),
            _make_task("a2", "runner", status="running"),
            _make_task("a3", "failed-task", status="failed"),
        ]
        panel = build_dag_panel(tasks)
        text = _render_to_plain(panel)
        # Status icons from dag.py
        assert "✅" in text  # completed
        assert "🔄" in text  # running
        assert "❌" in text  # failed

    def test_dag_panel_shows_progress(self):
        tasks = [
            _make_task("a1", "done", status="completed"),
            _make_task("a2", "also-done", status="completed"),
            _make_task("a3", "pending", status="pending"),
        ]
        panel = build_dag_panel(tasks)
        text = _render_to_plain(panel)
        # Progress bar from render_ascii_dag
        assert "Progress:" in text
        assert "2/3" in text

    def test_dag_panel_title(self):
        panel = build_dag_panel([])
        assert isinstance(panel, Panel)
        text = _render_to_plain(panel)
        assert "DAG" in text

    def test_dag_panel_has_blue_border(self):
        panel = build_dag_panel([])
        assert panel.border_style == "blue"


# ── Event Panel ──────────────────────────────────────────────────────────

class TestBuildEventPanel:
    def test_empty_events(self):
        panel = build_event_panel([])
        assert isinstance(panel, Panel)
        text = _render_to_plain(panel)
        assert "No events yet" in text

    def test_single_event(self):
        events = [{
            "timestamp": "2025-01-15T10:30:00.000Z",
            "event_type": "task_created",
            "task_id": "abc12345",
            "name": "parser",
        }]
        panel = build_event_panel(events)
        text = _render_to_plain(panel)
        assert "2025-01-15 10:30:00" in text
        assert "task_created" in text
        assert "abc12345" in text

    def test_multiple_event_types(self):
        events = [
            {"timestamp": "2025-01-15T10:00:00.000Z", "event_type": "task_created", "task_id": "a1"},
            {"timestamp": "2025-01-15T10:01:00.000Z", "event_type": "task_dispatched", "task_id": "a1"},
            {"timestamp": "2025-01-15T10:05:00.000Z", "event_type": "task_completed", "task_id": "a1"},
            {"timestamp": "2025-01-15T10:06:00.000Z", "event_type": "task_failed", "task_id": "a2"},
        ]
        panel = build_event_panel(events)
        text = _render_to_plain(panel)
        assert "task_created" in text
        assert "task_dispatched" in text
        assert "task_completed" in text
        assert "task_failed" in text

    def test_event_extras_duration(self):
        events = [{
            "timestamp": "2025-01-15T10:05:00.000Z",
            "event_type": "task_dispatch_complete",
            "task_id": "a1",
            "duration_s": 42.5,
        }]
        panel = build_event_panel(events)
        text = _render_to_plain(panel)
        assert "42.5s" in text

    def test_event_extras_exit_code(self):
        events = [{
            "timestamp": "2025-01-15T10:05:00.000Z",
            "event_type": "task_failed",
            "task_id": "a1",
            "exit_code": 1,
        }]
        panel = build_event_panel(events)
        text = _render_to_plain(panel)
        assert "exit=1" in text

    def test_event_extras_zero_exit_code_not_shown(self):
        """Exit code 0 is success — should not be shown as extra info."""
        events = [{
            "timestamp": "2025-01-15T10:05:00.000Z",
            "event_type": "task_completed",
            "task_id": "a1",
            "exit_code": 0,
        }]
        panel = build_event_panel(events)
        text = _render_to_plain(panel)
        assert "exit=" not in text

    def test_max_events_truncation(self):
        events = [
            {"timestamp": f"2025-01-15T10:{i:02d}:00.000Z", "event_type": "task_created", "task_id": f"t{i}"}
            for i in range(50)
        ]
        panel = build_event_panel(events, max_events=5)
        text = _render_to_plain(panel)
        # Should only show last 5 events (t45-t49)
        assert "t49" in text
        assert "t45" in text
        # Earlier events should not appear
        assert "t0 " not in text or text.count("task_created") <= 5

    def test_event_panel_title(self):
        panel = build_event_panel([])
        text = _render_to_plain(panel)
        assert "Events" in text

    def test_event_panel_subtitle_shows_quit_hint(self):
        panel = build_event_panel([])
        text = _render_to_plain(panel)
        assert "q to quit" in text

    def test_event_panel_has_green_border(self):
        panel = build_event_panel([])
        assert panel.border_style == "green"


# ── Color-coding verification ────────────────────────────────────────────

class TestEventColorCoding:
    """Verify that event types produce styled (colored) output."""

    def _has_style(self, event_type: str) -> bool:
        """Check that the event type is styled (not plain white)."""
        events = [{
            "timestamp": "2025-01-15T10:00:00.000Z",
            "event_type": event_type,
            "task_id": "a1",
        }]
        panel = build_event_panel(events)
        # Render with color support enabled
        colored = _render_to_str(panel)
        # Render without color
        plain = _render_to_plain(panel)
        # If colored output differs from plain, styling was applied
        return colored != plain

    def test_task_created_is_styled(self):
        assert self._has_style("task_created")

    def test_task_completed_is_styled(self):
        assert self._has_style("task_completed")

    def test_task_failed_is_styled(self):
        assert self._has_style("task_failed")

    def test_task_dispatched_is_styled(self):
        assert self._has_style("task_dispatched")

    def test_pause_is_styled(self):
        assert self._has_style("pause")

    def test_resume_is_styled(self):
        assert self._has_style("resume")

    def test_all_known_event_types_have_styles(self):
        """Every key in EVENT_STYLES should produce styled output."""
        for etype in EVENT_STYLES:
            assert self._has_style(etype), f"{etype} should be styled"


# ── Dashboard Layout ─────────────────────────────────────────────────────

class TestBuildDashboard:
    def test_returns_layout(self):
        layout = build_dashboard([], [])
        assert isinstance(layout, Layout)

    def test_has_two_panels(self):
        layout = build_dashboard([], [])
        # Layout should have children named "dag" and "events"
        child_names = [c.name for c in layout.children]
        assert "dag" in child_names
        assert "events" in child_names

    def test_exactly_two_children(self):
        layout = build_dashboard([], [])
        assert len(layout.children) == 2

    def test_dag_panel_is_first(self):
        layout = build_dashboard([], [])
        assert layout.children[0].name == "dag"

    def test_events_panel_is_second(self):
        layout = build_dashboard([], [])
        assert layout.children[1].name == "events"

    def test_renders_with_tasks_and_events(self):
        tasks = [
            _make_task("a1", "parser", status="completed"),
            _make_task("a2", "fts5", depends_on=["a1"], status="running"),
        ]
        events = [
            {"timestamp": "2025-01-15T10:00:00.000Z", "event_type": "task_completed", "task_id": "a1"},
            {"timestamp": "2025-01-15T10:01:00.000Z", "event_type": "task_dispatched", "task_id": "a2"},
        ]
        layout = build_dashboard(tasks, events)
        text = _render_to_plain(layout, width=120)
        # Both panels should contain their content
        assert "parser" in text
        assert "fts5" in text
        assert "task_completed" in text
        assert "task_dispatched" in text

    def test_renders_full_dashboard_with_color(self):
        """Full integration: ensure the dashboard renders without errors when color is on."""
        tasks = [
            _make_task("a1", "parser", status="completed"),
            _make_task("a2", "schema", status="completed"),
            _make_task("a3", "fts5", depends_on=["a1"], status="running"),
            _make_task("a4", "semantic", depends_on=["a2"], status="pending"),
            _make_task("a5", "hybrid", depends_on=["a3", "a4"], status="pending"),
        ]
        events = [
            {"timestamp": "2025-01-15T10:00:00.000Z", "event_type": "task_created", "task_id": "a1", "name": "parser"},
            {"timestamp": "2025-01-15T10:01:00.000Z", "event_type": "task_completed", "task_id": "a1"},
            {"timestamp": "2025-01-15T10:02:00.000Z", "event_type": "task_created", "task_id": "a2", "name": "schema"},
            {"timestamp": "2025-01-15T10:03:00.000Z", "event_type": "task_dispatched", "task_id": "a3"},
            {"timestamp": "2025-01-15T10:04:00.000Z", "event_type": "task_failed", "task_id": "a5", "exit_code": 1},
        ]
        layout = build_dashboard(tasks, events, max_events=10)
        # Should render without exceptions with color enabled
        colored = _render_to_str(layout, width=120)
        assert len(colored) > 0

    def test_dashboard_both_panels_have_titles(self):
        layout = build_dashboard([], [])
        text = _render_to_plain(layout)
        assert "DAG" in text
        assert "Events" in text


# ── Terminal resize (layout ratio) ───────────────────────────────────────

class TestPanelResize:
    """Verify that the layout works at different terminal widths."""

    def test_narrow_terminal(self):
        tasks = [_make_task("a1", "parser", status="completed")]
        events = [{"timestamp": "2025-01-15T10:00:00.000Z", "event_type": "task_created", "task_id": "a1"}]
        layout = build_dashboard(tasks, events)
        # Should not crash at narrow width
        text = _render_to_plain(layout, width=60)
        assert "DAG" in text
        assert "Events" in text

    def test_wide_terminal(self):
        tasks = [_make_task("a1", "parser", status="completed")]
        events = [{"timestamp": "2025-01-15T10:00:00.000Z", "event_type": "task_created", "task_id": "a1"}]
        layout = build_dashboard(tasks, events)
        # Should not crash at wide width
        text = _render_to_plain(layout, width=200)
        assert "DAG" in text
        assert "Events" in text

    def test_both_panels_ratio_equal(self):
        """Both panels should have equal ratio for balanced sizing."""
        layout = build_dashboard([], [])
        assert layout.children[0].ratio == 1
        assert layout.children[1].ratio == 1


# ── DAG live updates (simulated) ────────────────────────────────────────

class TestDagLiveUpdates:
    """Verify that re-rendering the DAG panel with changed task status
    reflects the updates (simulating what happens in the live loop).
    """

    def test_status_change_reflected(self):
        """When a task goes from running to completed, the panel output changes."""
        tasks_v1 = [
            _make_task("a1", "parser", status="running"),
            _make_task("a2", "fts5", depends_on=["a1"], status="pending"),
        ]
        tasks_v2 = [
            _make_task("a1", "parser", status="completed"),
            _make_task("a2", "fts5", depends_on=["a1"], status="running"),
        ]

        panel_v1 = build_dag_panel(tasks_v1)
        panel_v2 = build_dag_panel(tasks_v2)

        text_v1 = _render_to_plain(panel_v1)
        text_v2 = _render_to_plain(panel_v2)

        # Output should differ as task statuses changed
        assert text_v1 != text_v2
        # v2 should show completed icon for parser
        assert "✅" in text_v2

    def test_new_events_appear(self):
        """When new events are added, the event panel shows them."""
        events_v1 = [
            {"timestamp": "2025-01-15T10:00:00.000Z", "event_type": "task_created", "task_id": "a1"},
        ]
        events_v2 = events_v1 + [
            {"timestamp": "2025-01-15T10:05:00.000Z", "event_type": "task_completed", "task_id": "a1"},
        ]

        panel_v1 = build_event_panel(events_v1)
        panel_v2 = build_event_panel(events_v2)

        text_v1 = _render_to_plain(panel_v1)
        text_v2 = _render_to_plain(panel_v2)

        assert "task_completed" not in text_v1
        assert "task_completed" in text_v2


# ── EVENT_STYLES coverage ────────────────────────────────────────────────

class TestEventStyles:
    def test_all_common_event_types_covered(self):
        """All commonly emitted event types should have a style."""
        expected = {
            "task_created", "task_dispatched", "task_dispatch_complete",
            "task_completed", "task_failed", "pause", "resume",
        }
        for etype in expected:
            assert etype in EVENT_STYLES, f"Missing style for {etype}"

    def test_unknown_event_type_still_renders(self):
        """Events with unknown types should render without errors."""
        events = [{
            "timestamp": "2025-01-15T10:00:00.000Z",
            "event_type": "some_unknown_event",
            "task_id": "a1",
        }]
        panel = build_event_panel(events)
        text = _render_to_plain(panel)
        assert "some_unknown_event" in text

    def test_event_without_task_id(self):
        """Events without task_id should render without errors."""
        events = [{
            "timestamp": "2025-01-15T10:00:00.000Z",
            "event_type": "pause",
        }]
        panel = build_event_panel(events)
        text = _render_to_plain(panel)
        assert "pause" in text
