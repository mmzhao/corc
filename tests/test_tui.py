"""Tests for TUI v2 — active-plan-focused dashboard.

Tests cover:
  - Active plan panel: running, ready, blocked, recently completed, other
  - Filtering: only active/relevant tasks shown, old completed hidden
  - Running tasks: elapsed time, agent info
  - Ready tasks: marked as dispatchable
  - Blocked tasks: dependency info shown
  - Recently completed: dimmed rendering
  - Event panel: unchanged from v1
  - Dashboard layout: active_plan + events
  - Legacy compatibility: build_dag_panel, build_dashboard still work
  - QueryAPI integration: get_recently_completed_tasks
"""

import io
import time
import threading
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

from corc.tui import (
    build_active_plan_panel,
    build_event_panel,
    build_dag_panel,
    build_dashboard,
    build_active_dashboard,
    EVENT_STYLES,
    _elapsed_since,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_task(tid, name, depends_on=None, status="pending", **kwargs):
    """Build a minimal task dict for unit tests."""
    t = {
        "id": tid,
        "name": name,
        "depends_on": depends_on or [],
        "status": status,
        "done_when": "tests pass",
    }
    t.update(kwargs)
    return t


def _render_to_str(renderable, width=120) -> str:
    """Render a Rich renderable to a string with terminal colors."""
    buf = io.StringIO()
    console = Console(
        file=buf, force_terminal=True, width=width, color_system="truecolor"
    )
    console.print(renderable)
    return buf.getvalue()


def _render_to_plain(renderable, width=120) -> str:
    """Render a Rich renderable to plain text (no ANSI escapes)."""
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=width, no_color=True)
    console.print(renderable)
    return buf.getvalue()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ago_iso(minutes: int) -> str:
    """Return ISO timestamp N minutes ago."""
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


# ── Elapsed time helper ─────────────────────────────────────────────────


class TestElapsedSince:
    def test_empty_string(self):
        assert _elapsed_since("") == ""

    def test_none(self):
        assert _elapsed_since(None) == ""

    def test_recent_timestamp(self):
        # A timestamp from just a few seconds ago
        ts = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()
        result = _elapsed_since(ts)
        assert "s" in result
        assert "m" not in result or result == "< 1s"

    def test_minutes_ago(self):
        ts = (datetime.now(timezone.utc) - timedelta(minutes=5, seconds=30)).isoformat()
        result = _elapsed_since(ts)
        assert "5m" in result

    def test_hours_ago(self):
        ts = (datetime.now(timezone.utc) - timedelta(hours=2, minutes=15)).isoformat()
        result = _elapsed_since(ts)
        assert "2h" in result
        assert "15m" in result

    def test_invalid_timestamp(self):
        assert _elapsed_since("not-a-timestamp") == ""

    def test_z_suffix(self):
        ts = (datetime.now(timezone.utc) - timedelta(minutes=3)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        result = _elapsed_since(ts)
        assert "3m" in result


# ── Active Plan Panel ────────────────────────────────────────────────────


class TestBuildActivePlanPanel:
    """Test the main active plan panel builder."""

    def test_empty_state(self):
        """No tasks at all shows empty message."""
        panel = build_active_plan_panel([], [], [], [])
        assert isinstance(panel, Panel)
        text = _render_to_plain(panel)
        assert "No active tasks" in text

    def test_panel_title(self):
        panel = build_active_plan_panel([], [], [], [])
        text = _render_to_plain(panel)
        assert "Active Plan" in text

    def test_panel_border_style(self):
        panel = build_active_plan_panel([], [], [], [])
        assert panel.border_style == "blue"

    # ── Running tasks ──────────────────────────────────────────────

    def test_running_tasks_shown(self):
        running = [_make_task("r1", "build-parser", status="running")]
        panel = build_active_plan_panel(running, [], [], [])
        text = _render_to_plain(panel)
        assert "RUNNING" in text
        assert "build-parser" in text
        assert "r1" in text

    def test_running_task_count(self):
        running = [
            _make_task("r1", "task-a", status="running"),
            _make_task("r2", "task-b", status="running"),
        ]
        panel = build_active_plan_panel(running, [], [], [])
        text = _render_to_plain(panel)
        assert "(2)" in text
        assert "2 running" in text

    def test_running_task_with_elapsed_time(self):
        """Running tasks with a started timestamp show elapsed time."""
        started = _ago_iso(5)  # 5 minutes ago
        running = [_make_task("r1", "slow-task", status="running", started=started)]
        panel = build_active_plan_panel(running, [], [], [])
        text = _render_to_plain(panel)
        # Should show elapsed time indicator
        assert "5m" in text

    def test_running_task_with_agent_info(self):
        """Running tasks with agents list show agent details."""
        running = [
            _make_task(
                "r1",
                "agent-task",
                status="running",
                agents=[{"role": "implementer", "pid": 12345, "status": "running"}],
            )
        ]
        panel = build_active_plan_panel(running, [], [], [])
        text = _render_to_plain(panel)
        assert "implementer" in text
        assert "12345" in text

    def test_running_tasks_prominent_style(self):
        """Running tasks should be styled differently (yellow/bold) from dim text."""
        running = [_make_task("r1", "important-task", status="running")]
        panel = build_active_plan_panel(running, [], [], [])
        colored = _render_to_str(panel)
        plain = _render_to_plain(panel)
        # Colored output should differ (has ANSI codes)
        assert colored != plain

    # ── Ready tasks ────────────────────────────────────────────────

    def test_ready_tasks_shown(self):
        ready = [_make_task("rd1", "next-task", status="pending")]
        panel = build_active_plan_panel([], ready, [], [])
        text = _render_to_plain(panel)
        assert "READY" in text
        assert "next-task" in text

    def test_ready_tasks_marked_dispatchable(self):
        """Ready tasks section should indicate they are dispatchable."""
        ready = [_make_task("rd1", "next-task", status="pending")]
        panel = build_active_plan_panel([], ready, [], [])
        text = _render_to_plain(panel)
        assert "dispatchable" in text

    def test_ready_tasks_show_priority(self):
        """Non-default priority shown on ready tasks."""
        ready = [_make_task("rd1", "high-pri", status="pending", priority=10)]
        panel = build_active_plan_panel([], ready, [], [])
        text = _render_to_plain(panel)
        assert "pri=10" in text

    def test_ready_tasks_default_priority_hidden(self):
        """Default priority (100) should not be shown."""
        ready = [_make_task("rd1", "normal-pri", status="pending", priority=100)]
        panel = build_active_plan_panel([], ready, [], [])
        text = _render_to_plain(panel)
        assert "pri=" not in text

    def test_ready_task_arrow_icon(self):
        """Ready tasks use arrow icon to indicate dispatchability."""
        ready = [_make_task("rd1", "next-task", status="pending")]
        panel = build_active_plan_panel([], ready, [], [])
        text = _render_to_plain(panel)
        assert "▶" in text

    # ── Blocked tasks ──────────────────────────────────────────────

    def test_blocked_tasks_shown(self):
        blocked = [
            _make_task(
                "b1",
                "waiting-task",
                status="pending",
                blocked_by=["dep1"],
                reason="Waiting on 1 incomplete dependency: dep1",
            )
        ]
        panel = build_active_plan_panel([], [], blocked, [])
        text = _render_to_plain(panel)
        assert "BLOCKED" in text
        assert "waiting-task" in text

    def test_blocked_tasks_show_reason(self):
        """Blocked tasks show the reason from queries.py."""
        blocked = [
            _make_task(
                "b1",
                "stuck-task",
                status="pending",
                blocked_by=["abc12345"],
                reason="Waiting on 1 incomplete dependency: abc12345",
            )
        ]
        panel = build_active_plan_panel([], [], blocked, [])
        text = _render_to_plain(panel)
        assert "Waiting on 1 incomplete dependency" in text

    def test_blocked_tasks_show_blocked_by_ids(self):
        """If no reason but blocked_by present, show the IDs."""
        blocked = [
            _make_task(
                "b1", "stuck-task", status="pending", blocked_by=["dep1", "dep2"]
            )
        ]
        panel = build_active_plan_panel([], [], blocked, [])
        text = _render_to_plain(panel)
        assert "waiting on:" in text
        assert "dep1" in text
        assert "dep2" in text

    # ── Recently completed ─────────────────────────────────────────

    def test_recently_completed_shown(self):
        completed = [
            _make_task("c1", "done-task", status="completed", completed=_ago_iso(30))
        ]
        panel = build_active_plan_panel([], [], [], completed)
        text = _render_to_plain(panel)
        assert "RECENTLY COMPLETED" in text
        assert "done-task" in text

    def test_recently_completed_shows_elapsed(self):
        """Recently completed tasks show how long ago they finished."""
        completed = [
            _make_task("c1", "done-task", status="completed", completed=_ago_iso(10))
        ]
        panel = build_active_plan_panel([], [], [], completed)
        text = _render_to_plain(panel)
        assert "ago" in text

    def test_recently_completed_dimmed(self):
        """Recently completed tasks should be dimmed (styled differently)."""
        completed = [
            _make_task("c1", "done-task", status="completed", completed=_ago_iso(30))
        ]
        panel = build_active_plan_panel([], [], [], completed)
        colored = _render_to_str(panel)
        # The dimmed styling should produce ANSI codes
        assert "\033[" in colored  # Contains ANSI escape

    # ── Other active ───────────────────────────────────────────────

    def test_other_active_shown(self):
        other = [_make_task("o1", "failing-task", status="failed")]
        panel = build_active_plan_panel([], [], [], [], other_active=other)
        text = _render_to_plain(panel)
        assert "OTHER" in text
        assert "failing-task" in text
        assert "failed" in text

    def test_escalated_shown_with_icon(self):
        other = [_make_task("o1", "escalated-task", status="escalated")]
        panel = build_active_plan_panel([], [], [], [], other_active=other)
        text = _render_to_plain(panel)
        assert "escalated-task" in text

    # ── Summary line ───────────────────────────────────────────────

    def test_summary_counts(self):
        """Summary line shows total active and per-category counts."""
        running = [_make_task("r1", "running-task", status="running")]
        ready = [_make_task("rd1", "ready-task")]
        blocked = [_make_task("b1", "blocked-task", blocked_by=["x"])]
        completed = [
            _make_task("c1", "done-task", status="completed", completed=_ago_iso(5))
        ]
        panel = build_active_plan_panel(running, ready, blocked, completed)
        text = _render_to_plain(panel)
        assert "3 active" in text  # running + ready + blocked
        assert "1 running" in text
        assert "1 ready" in text
        assert "1 blocked" in text
        assert "1 done recently" in text

    # ── Filtering: old completed hidden ────────────────────────────

    def test_old_completed_not_shown(self):
        """Historical completed tasks are not passed to the panel at all.

        This tests the contract: only recently completed tasks are shown.
        The filtering happens in QueryAPI, but we verify the panel
        doesn't show tasks that aren't provided.
        """
        # Only provide running tasks, no completed
        running = [_make_task("r1", "current-work", status="running")]
        panel = build_active_plan_panel(running, [], [], [])
        text = _render_to_plain(panel)
        assert "RECENTLY COMPLETED" not in text
        assert "old-phase-task" not in text

    # ── Mixed scenario ─────────────────────────────────────────────

    def test_all_sections_present(self):
        """When all categories have tasks, all sections appear."""
        running = [
            _make_task("r1", "running-task", status="running", started=_ago_iso(2))
        ]
        ready = [_make_task("rd1", "ready-task")]
        blocked = [
            _make_task(
                "b1",
                "blocked-task",
                blocked_by=["r1"],
                reason="Waiting on 1 incomplete dependency: r1",
            )
        ]
        completed = [
            _make_task("c1", "done-task", status="completed", completed=_ago_iso(15))
        ]
        other = [_make_task("o1", "failed-task", status="failed")]

        panel = build_active_plan_panel(running, ready, blocked, completed, other)
        text = _render_to_plain(panel)

        assert "RUNNING" in text
        assert "READY" in text
        assert "BLOCKED" in text
        assert "RECENTLY COMPLETED" in text
        assert "OTHER" in text
        assert "running-task" in text
        assert "ready-task" in text
        assert "blocked-task" in text
        assert "done-task" in text
        assert "failed-task" in text


# ── Active Dashboard Layout ──────────────────────────────────────────────


class TestBuildActiveDashboard:
    def test_returns_layout(self):
        layout = build_active_dashboard([], [], [], [], [], [])
        assert isinstance(layout, Layout)

    def test_has_two_panels(self):
        layout = build_active_dashboard([], [], [], [], [], [])
        child_names = [c.name for c in layout.children]
        assert "active_plan" in child_names
        assert "events" in child_names

    def test_active_plan_is_first(self):
        layout = build_active_dashboard([], [], [], [], [], [])
        assert layout.children[0].name == "active_plan"

    def test_events_is_second(self):
        layout = build_active_dashboard([], [], [], [], [], [])
        assert layout.children[1].name == "events"

    def test_active_plan_has_higher_ratio(self):
        """Active plan panel should have higher ratio (more prominent)."""
        layout = build_active_dashboard([], [], [], [], [], [])
        assert layout.children[0].ratio == 2
        assert layout.children[1].ratio == 1

    def test_renders_with_data(self):
        running = [_make_task("r1", "parser", status="running")]
        events = [
            {
                "timestamp": "2025-01-15T10:00:00.000Z",
                "event_type": "task_dispatched",
                "task_id": "r1",
            }
        ]
        layout = build_active_dashboard(running, [], [], [], [], events)
        text = _render_to_plain(layout, width=120)
        assert "parser" in text
        assert "task_dispatched" in text
        assert "Active Plan" in text
        assert "Events" in text

    def test_renders_without_crash_at_narrow_width(self):
        running = [_make_task("r1", "parser", status="running")]
        layout = build_active_dashboard(running, [], [], [], [], [])
        text = _render_to_plain(layout, width=60)
        assert "Active Plan" in text


# ── Event Panel (unchanged from v1) ──────────────────────────────────────


class TestBuildEventPanel:
    def test_empty_events(self):
        panel = build_event_panel([])
        assert isinstance(panel, Panel)
        text = _render_to_plain(panel)
        assert "No events yet" in text

    def test_single_event(self):
        events = [
            {
                "timestamp": "2025-01-15T10:30:00.000Z",
                "event_type": "task_created",
                "task_id": "abc12345",
                "name": "parser",
            }
        ]
        panel = build_event_panel(events)
        text = _render_to_plain(panel)
        assert "2025-01-15 10:30:00" in text
        assert "task_created" in text
        assert "abc12345" in text

    def test_multiple_event_types(self):
        events = [
            {
                "timestamp": "2025-01-15T10:00:00.000Z",
                "event_type": "task_created",
                "task_id": "a1",
            },
            {
                "timestamp": "2025-01-15T10:01:00.000Z",
                "event_type": "task_dispatched",
                "task_id": "a1",
            },
            {
                "timestamp": "2025-01-15T10:05:00.000Z",
                "event_type": "task_completed",
                "task_id": "a1",
            },
            {
                "timestamp": "2025-01-15T10:06:00.000Z",
                "event_type": "task_failed",
                "task_id": "a2",
            },
        ]
        panel = build_event_panel(events)
        text = _render_to_plain(panel)
        assert "task_created" in text
        assert "task_dispatched" in text
        assert "task_completed" in text
        assert "task_failed" in text

    def test_event_extras_duration(self):
        events = [
            {
                "timestamp": "2025-01-15T10:05:00.000Z",
                "event_type": "task_dispatch_complete",
                "task_id": "a1",
                "duration_s": 42.5,
            }
        ]
        panel = build_event_panel(events)
        text = _render_to_plain(panel)
        assert "42.5s" in text

    def test_event_extras_exit_code(self):
        events = [
            {
                "timestamp": "2025-01-15T10:05:00.000Z",
                "event_type": "task_failed",
                "task_id": "a1",
                "exit_code": 1,
            }
        ]
        panel = build_event_panel(events)
        text = _render_to_plain(panel)
        assert "exit=1" in text

    def test_event_extras_zero_exit_code_not_shown(self):
        events = [
            {
                "timestamp": "2025-01-15T10:05:00.000Z",
                "event_type": "task_completed",
                "task_id": "a1",
                "exit_code": 0,
            }
        ]
        panel = build_event_panel(events)
        text = _render_to_plain(panel)
        assert "exit=" not in text

    def test_max_events_truncation(self):
        events = [
            {
                "timestamp": f"2025-01-15T10:{i:02d}:00.000Z",
                "event_type": "task_created",
                "task_id": f"t{i}",
            }
            for i in range(50)
        ]
        panel = build_event_panel(events, max_events=5)
        text = _render_to_plain(panel)
        assert "t49" in text
        assert "t45" in text
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
        events = [
            {
                "timestamp": "2025-01-15T10:00:00.000Z",
                "event_type": event_type,
                "task_id": "a1",
            }
        ]
        panel = build_event_panel(events)
        colored = _render_to_str(panel)
        plain = _render_to_plain(panel)
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
        for etype in EVENT_STYLES:
            assert self._has_style(etype), f"{etype} should be styled"


# ── Legacy DAG Panel ─────────────────────────────────────────────────────


class TestBuildDagPanel:
    """Legacy build_dag_panel still works for backward compat."""

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

    def test_dag_panel_has_blue_border(self):
        panel = build_dag_panel([])
        assert panel.border_style == "blue"


# ── Legacy Dashboard ─────────────────────────────────────────────────────


class TestBuildDashboard:
    """Legacy build_dashboard still works."""

    def test_returns_layout(self):
        layout = build_dashboard([], [])
        assert isinstance(layout, Layout)

    def test_has_two_panels(self):
        layout = build_dashboard([], [])
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
            {
                "timestamp": "2025-01-15T10:00:00.000Z",
                "event_type": "task_completed",
                "task_id": "a1",
            },
            {
                "timestamp": "2025-01-15T10:01:00.000Z",
                "event_type": "task_dispatched",
                "task_id": "a2",
            },
        ]
        layout = build_dashboard(tasks, events)
        text = _render_to_plain(layout, width=120)
        assert "parser" in text
        assert "fts5" in text
        assert "task_completed" in text
        assert "task_dispatched" in text


# ── EVENT_STYLES coverage ────────────────────────────────────────────────


class TestEventStyles:
    def test_all_common_event_types_covered(self):
        expected = {
            "task_created",
            "task_dispatched",
            "task_dispatch_complete",
            "task_completed",
            "task_failed",
            "pause",
            "resume",
        }
        for etype in expected:
            assert etype in EVENT_STYLES, f"Missing style for {etype}"

    def test_unknown_event_type_still_renders(self):
        events = [
            {
                "timestamp": "2025-01-15T10:00:00.000Z",
                "event_type": "some_unknown_event",
                "task_id": "a1",
            }
        ]
        panel = build_event_panel(events)
        text = _render_to_plain(panel)
        assert "some_unknown_event" in text

    def test_event_without_task_id(self):
        events = [
            {
                "timestamp": "2025-01-15T10:00:00.000Z",
                "event_type": "pause",
            }
        ]
        panel = build_event_panel(events)
        text = _render_to_plain(panel)
        assert "pause" in text


# ── Filtering contract tests ─────────────────────────────────────────────


class TestFilteringContract:
    """Verify that the TUI panels correctly filter/display based on
    the data provided by QueryAPI methods.

    These tests simulate what run_active_dashboard does: call QueryAPI
    methods and pass results to panel builders.
    """

    def test_only_active_tasks_shown_not_historical(self):
        """Old completed tasks are excluded by the QueryAPI contract.

        Only recently completed tasks (provided separately) appear.
        """
        # Simulate: 3 completed from months ago (not provided)
        # 1 running task (provided as running)
        # 1 recently completed (provided as recently_completed)
        running = [_make_task("r1", "current-work", status="running")]
        recent = [
            _make_task("c1", "just-done", status="completed", completed=_ago_iso(20))
        ]

        panel = build_active_plan_panel(running, [], [], recent)
        text = _render_to_plain(panel)

        assert "current-work" in text
        assert "just-done" in text
        # Old tasks not provided, so not shown
        assert "old-phase1" not in text
        assert "old-phase2" not in text

    def test_running_more_prominent_than_blocked(self):
        """Running section appears before blocked section in output."""
        running = [_make_task("r1", "RUNNING_MARKER", status="running")]
        blocked = [
            _make_task(
                "b1",
                "BLOCKED_MARKER",
                status="pending",
                blocked_by=["x"],
                reason="blocked",
            )
        ]

        panel = build_active_plan_panel(running, [], blocked, [])
        text = _render_to_plain(panel)

        running_pos = text.index("RUNNING_MARKER")
        blocked_pos = text.index("BLOCKED_MARKER")
        assert running_pos < blocked_pos

    def test_ready_before_blocked(self):
        """Ready section appears before blocked section."""
        ready = [_make_task("rd1", "READY_MARKER")]
        blocked = [
            _make_task("b1", "BLOCKED_MARKER", blocked_by=["x"], reason="blocked")
        ]

        panel = build_active_plan_panel([], ready, blocked, [])
        text = _render_to_plain(panel)

        ready_pos = text.index("READY_MARKER")
        blocked_pos = text.index("BLOCKED_MARKER")
        assert ready_pos < blocked_pos

    def test_recently_completed_last(self):
        """Recently completed section appears after all active sections."""
        running = [_make_task("r1", "RUNNING_MARKER", status="running")]
        completed = [
            _make_task(
                "c1", "COMPLETED_MARKER", status="completed", completed=_ago_iso(10)
            )
        ]

        panel = build_active_plan_panel(running, [], [], completed)
        text = _render_to_plain(panel)

        running_pos = text.index("RUNNING_MARKER")
        completed_pos = text.index("COMPLETED_MARKER")
        assert running_pos < completed_pos


# ── QueryAPI.get_recently_completed_tasks ────────────────────────────────


class TestQueryAPIRecentlyCompleted:
    """Test the new get_recently_completed_tasks method.

    Uses mock WorkState to avoid database setup.
    """

    def _make_query_api(self, tasks):
        """Create a QueryAPI with mocked dependencies returning given tasks."""
        from corc.queries import QueryAPI

        ws = MagicMock()
        ws.list_tasks.return_value = tasks
        al = MagicMock()
        sl = MagicMock()
        return QueryAPI(ws, al, sl)

    def test_returns_recently_completed(self):
        """Tasks completed within the last hour are returned."""
        tasks = [
            _make_task("c1", "recent", status="completed", completed=_ago_iso(30)),
        ]
        api = self._make_query_api(tasks)
        result = api.get_recently_completed_tasks(hours=1.0)
        assert len(result) == 1
        assert result[0]["id"] == "c1"

    def test_excludes_old_completed(self):
        """Tasks completed more than 1 hour ago are excluded."""
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        tasks = [
            _make_task("c1", "old", status="completed", completed=old_ts),
        ]
        api = self._make_query_api(tasks)
        result = api.get_recently_completed_tasks(hours=1.0)
        assert len(result) == 0

    def test_mixed_recent_and_old(self):
        """Only recently completed tasks are returned, old ones excluded."""
        recent_ts = _ago_iso(20)
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        tasks = [
            _make_task("c1", "recent", status="completed", completed=recent_ts),
            _make_task("c2", "old", status="completed", completed=old_ts),
        ]
        api = self._make_query_api(tasks)
        result = api.get_recently_completed_tasks(hours=1.0)
        assert len(result) == 1
        assert result[0]["id"] == "c1"

    def test_sorted_most_recent_first(self):
        """Results are sorted most-recently-completed first."""
        ts_10m = _ago_iso(10)
        ts_30m = _ago_iso(30)
        tasks = [
            _make_task("c1", "older", status="completed", completed=ts_30m),
            _make_task("c2", "newer", status="completed", completed=ts_10m),
        ]
        api = self._make_query_api(tasks)
        result = api.get_recently_completed_tasks(hours=1.0)
        assert len(result) == 2
        assert result[0]["id"] == "c2"  # More recent first
        assert result[1]["id"] == "c1"

    def test_no_completed_timestamp(self):
        """Tasks without a completed timestamp are excluded."""
        tasks = [
            _make_task("c1", "no-ts", status="completed"),
        ]
        api = self._make_query_api(tasks)
        result = api.get_recently_completed_tasks(hours=1.0)
        assert len(result) == 0

    def test_custom_hours_window(self):
        """Respects the hours parameter."""
        ts_90m = (datetime.now(timezone.utc) - timedelta(minutes=90)).isoformat()
        tasks = [
            _make_task("c1", "recent-ish", status="completed", completed=ts_90m),
        ]
        api = self._make_query_api(tasks)

        # 1 hour window: should exclude
        result_1h = api.get_recently_completed_tasks(hours=1.0)
        assert len(result_1h) == 0

        # 2 hour window: should include
        result_2h = api.get_recently_completed_tasks(hours=2.0)
        assert len(result_2h) == 1

    def test_only_completed_status(self):
        """Only tasks with status=completed are considered.

        Tests that the method correctly calls list_tasks(status='completed').
        """
        api = self._make_query_api([])
        api.get_recently_completed_tasks()
        api.work_state.list_tasks.assert_called_once_with(status="completed")


# ── QueryAPI integration with panel builders ─────────────────────────────


class TestQueryAPIPanelIntegration:
    """Test that QueryAPI methods produce data compatible with panel builders."""

    def _make_query_api(self, all_tasks, agents_map=None):
        """Create a QueryAPI with mocked dependencies."""
        from corc.queries import QueryAPI

        ws = MagicMock()
        ws.list_tasks.side_effect = lambda status=None: (
            [t for t in all_tasks if t["status"] == status] if status else all_tasks
        )
        ws.get_ready_tasks.return_value = [
            t
            for t in all_tasks
            if t["status"] == "pending"
            and all(
                d in {x["id"] for x in all_tasks if x["status"] == "completed"}
                for d in (t.get("depends_on") or [])
            )
        ]
        ws.get_agents_for_task.side_effect = lambda tid: (agents_map or {}).get(tid, [])

        al = MagicMock()
        al.read_recent.return_value = []
        sl = MagicMock()

        return QueryAPI(ws, al, sl)

    def test_full_pipeline(self):
        """Simulate what run_active_dashboard does: query + render."""
        tasks = [
            _make_task(
                "c1",
                "old-done",
                status="completed",
                completed=(datetime.now(timezone.utc) - timedelta(hours=5)).isoformat(),
            ),
            _make_task("c2", "recent-done", status="completed", completed=_ago_iso(20)),
            _make_task("r1", "in-progress", status="running", started=_ago_iso(5)),
            _make_task("rd1", "next-up", status="pending"),
            _make_task("b1", "waiting", status="pending", depends_on=["r1"]),
        ]
        agents = {"r1": [{"role": "implementer", "pid": 999, "status": "running"}]}
        api = self._make_query_api(tasks, agents)

        running = api.get_running_tasks_with_agents()
        ready = api.get_ready_tasks()
        blocked = api.get_blocked_tasks_with_reasons()
        recent_done = api.get_recently_completed_tasks(hours=1.0)
        events = api.get_recent_events(20)

        # Build the panel
        panel = build_active_plan_panel(running, ready, blocked, recent_done, [])
        text = _render_to_plain(panel)

        # Running task shown with agent info
        assert "in-progress" in text
        assert "implementer" in text

        # Ready task shown
        assert "next-up" in text

        # Blocked task shown
        assert "waiting" in text

        # Recently completed shown
        assert "recent-done" in text

        # Old completed NOT shown
        assert "old-done" not in text

    def test_full_dashboard_renders_without_error(self):
        """Full active dashboard renders without exceptions."""
        tasks = [
            _make_task("r1", "active-work", status="running", started=_ago_iso(3)),
            _make_task("rd1", "ready-work", status="pending"),
        ]
        api = self._make_query_api(tasks)

        running = api.get_running_tasks_with_agents()
        ready = api.get_ready_tasks()
        blocked = api.get_blocked_tasks_with_reasons()
        recent = api.get_recently_completed_tasks()
        events = api.get_recent_events(20)

        categorized_ids = (
            {t["id"] for t in running}
            | {t["id"] for t in ready}
            | {t["id"] for t in blocked}
        )
        all_active = api.get_active_plan_tasks()
        other = [t for t in all_active if t["id"] not in categorized_ids]

        layout = build_active_dashboard(running, ready, blocked, recent, other, events)
        text = _render_to_plain(layout, width=120)
        assert "Active Plan" in text
        assert "Events" in text
        assert len(text) > 0


# ── Terminal resize ──────────────────────────────────────────────────────


class TestActivePanelResize:
    def test_narrow_terminal(self):
        running = [_make_task("r1", "task", status="running")]
        layout = build_active_dashboard(running, [], [], [], [], [])
        text = _render_to_plain(layout, width=60)
        assert "Active Plan" in text

    def test_wide_terminal(self):
        running = [_make_task("r1", "task", status="running")]
        layout = build_active_dashboard(running, [], [], [], [], [])
        text = _render_to_plain(layout, width=200)
        assert "Active Plan" in text
