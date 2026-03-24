"""Tests for TUI v2 — active-plan-focused dashboard.

Tests cover:
  - Active plan panel: running, ready, blocked, recently completed, other
  - Filtering: only active/relevant tasks shown, old completed hidden
  - Running tasks: elapsed time, agent info
  - Ready tasks: marked as dispatchable
  - Blocked tasks: dependency info shown
  - Recently completed: dimmed rendering
  - Streaming detail panel: tool calls, reasoning, checklist, scrolling
  - Event panel: unchanged from v1
  - Dashboard layout: left 2/3 (DAG top + streaming bottom) | right 1/3 events
  - Layout resize: panels resize proportionally with terminal width
  - Legacy compatibility: build_dag_panel, build_dashboard still work
  - QueryAPI integration: get_recently_completed_tasks
"""

import io
import json
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
    build_streaming_detail_panel,
    build_dag_panel,
    build_dashboard,
    build_active_dashboard,
    EVENT_STYLES,
    _elapsed_since,
    _parse_stream_content,
    _format_tool_call,
    _truncate_reasoning,
    _format_checklist_progress,
    _format_attempt_count,
    _deduplicate_agents,
    _deduplicate_task_agents,
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

    def test_left_right_split(self):
        """Root layout splits into left (2/3) and events (1/3) columns."""
        layout = build_active_dashboard([], [], [], [], [], [])
        root_names = [c.name for c in layout.children]
        assert "left" in root_names
        assert "events" in root_names
        assert len(layout.children) == 2

    def test_left_two_thirds_events_one_third(self):
        """Left side has ratio 2, events has ratio 1 (2/3 vs 1/3)."""
        layout = build_active_dashboard([], [], [], [], [], [])
        assert layout.children[0].name == "left"
        assert layout.children[0].ratio == 2
        assert layout.children[1].name == "events"
        assert layout.children[1].ratio == 1

    def test_active_plan_nested_in_left(self):
        """Active plan panel is inside the left container."""
        layout = build_active_dashboard([], [], [], [], [], [])
        left = layout["left"]
        left_child_names = [c.name for c in left.children]
        assert "active_plan" in left_child_names

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
    """Panels render without crashing at various terminal widths."""

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

    def test_narrow_terminal_with_streaming(self):
        """Three-panel layout renders at narrow width."""
        running = [_make_task("r1", "task", status="running")]
        layout = build_active_dashboard(
            running, [], [], [], [], [], stream_events_by_task={}
        )
        text = _render_to_plain(layout, width=60)
        assert "Active Plan" in text

    def test_wide_terminal_with_streaming(self):
        """Three-panel layout renders at wide width."""
        running = [_make_task("r1", "task", status="running")]
        layout = build_active_dashboard(
            running, [], [], [], [], [], stream_events_by_task={}
        )
        text = _render_to_plain(layout, width=200)
        assert "Active Plan" in text
        assert "Streaming Detail" in text
        assert "Events" in text

    def test_medium_terminal_all_panels_visible(self):
        """At medium width all three panel titles are present."""
        running = [_make_task("r1", "task", status="running")]
        events = [
            {
                "timestamp": "2026-03-23T10:00:00.000Z",
                "event_type": "task_dispatched",
                "task_id": "r1",
            }
        ]
        layout = build_active_dashboard(
            running, [], [], [], [], events, stream_events_by_task={}
        )
        text = _render_to_plain(layout, width=120)
        assert "Active Plan" in text
        assert "Streaming Detail" in text
        assert "Events" in text


# ── Stream event helpers ─────────────────────────────────────────────────


def _make_stream_entry(stream_type: str, event_data: dict) -> dict:
    """Build a stream_event entry as returned by QueryAPI.get_task_stream_events()."""
    return {
        "timestamp": "2026-03-23T10:00:00.000Z",
        "type": "stream_event",
        "content": json.dumps(event_data, separators=(",", ":")),
        "stream_type": stream_type,
    }


# ── _parse_stream_content ────────────────────────────────────────────────


class TestParseStreamContent:
    def test_valid_json_content(self):
        entry = _make_stream_entry(
            "tool_use", {"type": "tool_use", "tool": {"name": "Read"}}
        )
        parsed = _parse_stream_content(entry)
        assert parsed is not None
        assert parsed["type"] == "tool_use"

    def test_empty_content(self):
        assert _parse_stream_content({"content": ""}) is None

    def test_missing_content(self):
        assert _parse_stream_content({}) is None

    def test_invalid_json(self):
        assert _parse_stream_content({"content": "not json"}) is None

    def test_none_content(self):
        assert _parse_stream_content({"content": None}) is None


# ── _format_tool_call ────────────────────────────────────────────────────


class TestFormatToolCall:
    def test_read_file_path(self):
        event = {"tool": {"name": "Read", "input": {"file_path": "/src/main.py"}}}
        assert _format_tool_call(event) == "Read /src/main.py"

    def test_write_file_path(self):
        event = {
            "tool": {
                "name": "Write",
                "input": {"file_path": "/src/new.py", "content": "code"},
            }
        }
        assert _format_tool_call(event) == "Write /src/new.py"

    def test_bash_command(self):
        event = {"tool": {"name": "Bash", "input": {"command": "ls -la"}}}
        assert _format_tool_call(event) == "Bash ls -la"

    def test_bash_long_command_truncated(self):
        long_cmd = "x" * 100
        event = {"tool": {"name": "Bash", "input": {"command": long_cmd}}}
        result = _format_tool_call(event)
        assert len(result) < 70
        assert result.endswith("...")

    def test_grep_pattern(self):
        event = {"tool": {"name": "Grep", "input": {"pattern": "def main"}}}
        assert _format_tool_call(event) == "Grep def main"

    def test_glob_pattern(self):
        event = {"tool": {"name": "Glob", "input": {"pattern": "**/*.py"}}}
        assert _format_tool_call(event) == "Glob **/*.py"

    def test_fallback_first_key(self):
        event = {"tool": {"name": "Custom", "input": {"query": "find something"}}}
        result = _format_tool_call(event)
        assert "Custom" in result
        assert "query=" in result

    def test_no_input(self):
        event = {"tool": {"name": "NoArgs", "input": {}}}
        assert _format_tool_call(event) == "NoArgs"

    def test_missing_tool(self):
        event = {}
        result = _format_tool_call(event)
        assert result == "?"

    def test_fallback_long_value_truncated(self):
        event = {"tool": {"name": "Custom", "input": {"data": "a" * 100}}}
        result = _format_tool_call(event)
        assert "..." in result
        assert len(result) < 60


# ── _truncate_reasoning ──────────────────────────────────────────────────


class TestTruncateReasoning:
    def test_short_text_unchanged(self):
        assert _truncate_reasoning("Short text.") == "Short text."

    def test_long_text_truncated(self):
        long_text = "a" * 200
        result = _truncate_reasoning(long_text, max_len=50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_newlines_collapsed(self):
        text = "Line one.\nLine two.\nLine three."
        result = _truncate_reasoning(text)
        assert "\n" not in result
        assert "Line one. Line two. Line three." == result

    def test_whitespace_stripped(self):
        assert _truncate_reasoning("  hello  ") == "hello"

    def test_exact_max_len(self):
        text = "x" * 120
        result = _truncate_reasoning(text, max_len=120)
        assert result == text  # Exactly at limit, no truncation

    def test_one_over_max_len(self):
        text = "x" * 121
        result = _truncate_reasoning(text, max_len=120)
        assert len(result) == 120
        assert result.endswith("...")


# ── _format_checklist_progress ───────────────────────────────────────────


class TestFormatChecklistProgress:
    def test_no_checklist(self):
        assert _format_checklist_progress(None) is None
        assert _format_checklist_progress("") is None
        assert _format_checklist_progress([]) is None

    def test_list_of_dicts(self):
        checklist = [
            {"item": "Write tests", "done": True},
            {"item": "Implement feature", "done": True},
            {"item": "Review PR", "done": False},
        ]
        result = _format_checklist_progress(checklist)
        assert result == "2/3 items done"

    def test_all_done(self):
        checklist = [{"done": True}, {"done": True}]
        assert _format_checklist_progress(checklist) == "2/2 items done"

    def test_none_done(self):
        checklist = [{"done": False}, {"done": False}]
        assert _format_checklist_progress(checklist) == "0/2 items done"

    def test_json_string_input(self):
        checklist_str = json.dumps([{"done": True}, {"done": False}])
        result = _format_checklist_progress(checklist_str)
        assert result == "1/2 items done"

    def test_invalid_json_string(self):
        assert _format_checklist_progress("not json") is None

    def test_items_without_done_key(self):
        """Items without 'done' key count as not done."""
        checklist = [{"item": "task"}]
        assert _format_checklist_progress(checklist) == "0/1 items done"

    def test_non_list_after_parse(self):
        """If JSON parses to non-list, return None."""
        assert _format_checklist_progress('{"not": "a list"}') is None


# ── _format_attempt_count ──────────────────────────────────────────────


class TestFormatAttemptCount:
    """Tests for _format_attempt_count — retry attempt indicator."""

    def test_first_attempt_returns_none(self):
        """First attempt (attempt_count=0) shows no indicator."""
        task = {"attempt_count": 0, "max_retries": 3}
        assert _format_attempt_count(task) is None

    def test_missing_attempt_count_returns_none(self):
        """Tasks without attempt_count field show no indicator."""
        task = {"max_retries": 3}
        assert _format_attempt_count(task) is None

    def test_second_attempt(self):
        """attempt_count=1 means 2nd attempt out of 4 total (max_retries=3)."""
        task = {"attempt_count": 1, "max_retries": 3}
        assert _format_attempt_count(task) == "attempt 2/4"

    def test_third_attempt(self):
        """attempt_count=2, max_retries=3 → attempt 3/4."""
        task = {"attempt_count": 2, "max_retries": 3}
        assert _format_attempt_count(task) == "attempt 3/4"

    def test_final_attempt(self):
        """attempt_count=3, max_retries=3 → attempt 4/4."""
        task = {"attempt_count": 3, "max_retries": 3}
        assert _format_attempt_count(task) == "attempt 4/4"

    def test_custom_max_retries(self):
        """Different max_retries values produce correct totals."""
        task = {"attempt_count": 1, "max_retries": 5}
        assert _format_attempt_count(task) == "attempt 2/6"

    def test_default_max_retries(self):
        """When max_retries is missing, defaults to 3."""
        task = {"attempt_count": 2}
        assert _format_attempt_count(task) == "attempt 3/4"

    def test_zero_max_retries(self):
        """max_retries=0 means only 1 total attempt possible."""
        task = {"attempt_count": 1, "max_retries": 0}
        assert _format_attempt_count(task) == "attempt 2/1"

    def test_negative_attempt_count_returns_none(self):
        """Negative attempt_count is treated as no retries."""
        task = {"attempt_count": -1, "max_retries": 3}
        assert _format_attempt_count(task) is None


# ── Attempt count in panels ──────────────────────────────────────────────


class TestAttemptCountInPanels:
    """Test attempt count display in DAG status and streaming detail panels."""

    # ── Active plan panel (DAG status) ────────────────────────────────

    def test_running_task_retry_shows_attempt(self):
        """Running task on retry shows attempt count in DAG panel."""
        running = [
            _make_task(
                "r1",
                "retrying-task",
                status="running",
                attempt_count=2,
                max_retries=3,
            )
        ]
        panel = build_active_plan_panel(running, [], [], [])
        text = _render_to_plain(panel)
        assert "attempt 3/4" in text
        assert "retrying-task" in text

    def test_running_task_first_attempt_no_indicator(self):
        """Running task on first attempt shows no retry indicator."""
        running = [
            _make_task(
                "r1",
                "fresh-task",
                status="running",
                attempt_count=0,
                max_retries=3,
            )
        ]
        panel = build_active_plan_panel(running, [], [], [])
        text = _render_to_plain(panel)
        assert "fresh-task" in text
        assert "attempt" not in text

    def test_running_task_no_attempt_count_no_indicator(self):
        """Running task without attempt_count field shows no retry indicator."""
        running = [_make_task("r1", "normal-task", status="running")]
        panel = build_active_plan_panel(running, [], [], [])
        text = _render_to_plain(panel)
        assert "normal-task" in text
        assert "attempt" not in text

    def test_failed_task_shows_attempt_in_other_section(self):
        """Failed task in OTHER section shows attempt count."""
        other = [
            _make_task(
                "o1",
                "failed-retry",
                status="failed",
                attempt_count=1,
                max_retries=3,
            )
        ]
        panel = build_active_plan_panel([], [], [], [], other_active=other)
        text = _render_to_plain(panel)
        assert "attempt 2/4" in text
        assert "failed-retry" in text

    def test_failed_task_first_attempt_no_indicator(self):
        """Failed task on first attempt shows no retry indicator."""
        other = [
            _make_task(
                "o1",
                "first-fail",
                status="failed",
                attempt_count=0,
                max_retries=3,
            )
        ]
        panel = build_active_plan_panel([], [], [], [], other_active=other)
        text = _render_to_plain(panel)
        assert "first-fail" in text
        assert "attempt" not in text

    def test_escalated_task_shows_attempt(self):
        """Escalated task in OTHER section shows attempt count."""
        other = [
            _make_task(
                "o1",
                "escalated-retry",
                status="escalated",
                attempt_count=3,
                max_retries=3,
            )
        ]
        panel = build_active_plan_panel([], [], [], [], other_active=other)
        text = _render_to_plain(panel)
        assert "attempt 4/4" in text

    # ── Streaming detail panel (agent detail) ─────────────────────────

    def test_streaming_panel_retry_shows_attempt(self):
        """Streaming detail panel shows attempt count for retry tasks."""
        running = [
            _make_task(
                "t1",
                "streaming-retry",
                status="running",
                attempt_count=1,
                max_retries=3,
            )
        ]
        panel = build_streaming_detail_panel(running, {})
        text = _render_to_plain(panel)
        assert "attempt 2/4" in text
        assert "streaming-retry" in text

    def test_streaming_panel_first_attempt_no_indicator(self):
        """Streaming detail panel shows no retry indicator on first attempt."""
        running = [
            _make_task(
                "t1",
                "first-run",
                status="running",
                attempt_count=0,
                max_retries=3,
            )
        ]
        panel = build_streaming_detail_panel(running, {})
        text = _render_to_plain(panel)
        assert "first-run" in text
        assert "attempt" not in text

    def test_streaming_panel_no_attempt_count_no_indicator(self):
        """Streaming panel with no attempt_count shows no indicator."""
        running = [_make_task("t1", "plain-task", status="running")]
        panel = build_streaming_detail_panel(running, {})
        text = _render_to_plain(panel)
        assert "plain-task" in text
        assert "attempt" not in text

    def test_attempt_updates_on_retry(self):
        """Different attempt_count values produce different display strings."""
        # Simulate task at attempt 1
        running_a1 = [
            _make_task(
                "t1",
                "evolving-task",
                status="running",
                attempt_count=1,
                max_retries=3,
            )
        ]
        panel_a1 = build_active_plan_panel(running_a1, [], [], [])
        text_a1 = _render_to_plain(panel_a1)
        assert "attempt 2/4" in text_a1

        # Simulate same task at attempt 2
        running_a2 = [
            _make_task(
                "t1",
                "evolving-task",
                status="running",
                attempt_count=2,
                max_retries=3,
            )
        ]
        panel_a2 = build_active_plan_panel(running_a2, [], [], [])
        text_a2 = _render_to_plain(panel_a2)
        assert "attempt 3/4" in text_a2

    def test_both_panels_show_attempt_for_same_task(self):
        """Both DAG panel and streaming panel show attempt count consistently."""
        task_data = _make_task(
            "t1",
            "dual-panel-task",
            status="running",
            attempt_count=2,
            max_retries=3,
        )
        dag_panel = build_active_plan_panel([task_data], [], [], [])
        streaming_panel = build_streaming_detail_panel([task_data], {})

        dag_text = _render_to_plain(dag_panel)
        streaming_text = _render_to_plain(streaming_panel)

        assert "attempt 3/4" in dag_text
        assert "attempt 3/4" in streaming_text


# ── build_streaming_detail_panel ─────────────────────────────────────────


class TestBuildStreamingDetailPanel:
    """Test the streaming detail panel builder."""

    def test_empty_state_no_running_tasks(self):
        """No running tasks shows empty message."""
        panel = build_streaming_detail_panel([], {})
        assert isinstance(panel, Panel)
        text = _render_to_plain(panel)
        assert "No running agents" in text

    def test_panel_title(self):
        panel = build_streaming_detail_panel([], {})
        text = _render_to_plain(panel)
        assert "Streaming Detail" in text

    def test_panel_border_style(self):
        panel = build_streaming_detail_panel([], {})
        assert panel.border_style == "magenta"

    def test_scroll_hint_in_subtitle(self):
        panel = build_streaming_detail_panel([], {})
        text = _render_to_plain(panel)
        assert "scroll" in text

    def test_running_task_header(self):
        """Running task name and ID are displayed."""
        running = [_make_task("task-abc123", "build-parser", status="running")]
        panel = build_streaming_detail_panel(running, {})
        text = _render_to_plain(panel)
        assert "build-parser" in text
        assert "task-abc" in text

    def test_waiting_for_stream_events(self):
        """When no stream events exist yet, shows waiting message."""
        running = [_make_task("t1", "new-task", status="running")]
        panel = build_streaming_detail_panel(running, {})
        text = _render_to_plain(panel)
        assert "Waiting for stream events" in text

    def test_tool_use_displayed(self):
        """Tool use events show tool name and file path."""
        running = [_make_task("t1", "coding-task", status="running")]
        events = {
            "t1": [
                _make_stream_entry(
                    "tool_use",
                    {
                        "type": "tool_use",
                        "tool": {
                            "name": "Read",
                            "input": {"file_path": "/src/main.py"},
                        },
                    },
                ),
            ]
        }
        panel = build_streaming_detail_panel(running, events)
        text = _render_to_plain(panel)
        assert "Read" in text
        assert "/src/main.py" in text

    def test_assistant_reasoning_displayed(self):
        """Assistant messages show truncated reasoning."""
        running = [_make_task("t1", "reasoning-task", status="running")]
        events = {
            "t1": [
                _make_stream_entry(
                    "assistant",
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Let me analyze the code structure.",
                                }
                            ],
                        },
                    },
                ),
            ]
        }
        panel = build_streaming_detail_panel(running, events)
        text = _render_to_plain(panel)
        assert "Let me analyze the code structure." in text

    def test_assistant_reasoning_truncated(self):
        """Long assistant messages are truncated."""
        running = [_make_task("t1", "verbose-task", status="running")]
        long_text = "x" * 200
        events = {
            "t1": [
                _make_stream_entry(
                    "assistant",
                    {
                        "type": "assistant",
                        "message": {
                            "content": [{"type": "text", "text": long_text}],
                        },
                    },
                ),
            ]
        }
        panel = build_streaming_detail_panel(running, events)
        text = _render_to_plain(panel)
        assert "..." in text
        # Should not contain the full 200-char string
        assert long_text not in text

    def test_result_event_displayed(self):
        """Result events show completion summary."""
        running = [_make_task("t1", "done-task", status="running")]
        events = {
            "t1": [
                _make_stream_entry(
                    "result",
                    {
                        "type": "result",
                        "result": "Task completed successfully.",
                    },
                ),
            ]
        }
        panel = build_streaming_detail_panel(running, events)
        text = _render_to_plain(panel)
        assert "Task completed successfully." in text

    def test_checklist_progress_displayed(self):
        """Tasks with checklist show progress."""
        checklist = [
            {"item": "Write tests", "done": True},
            {"item": "Implement feature", "done": False},
            {"item": "Review PR", "done": False},
        ]
        running = [
            _make_task("t1", "checklist-task", status="running", checklist=checklist)
        ]
        panel = build_streaming_detail_panel(running, {})
        text = _render_to_plain(panel)
        assert "1/3 items done" in text

    def test_multiple_running_tasks(self):
        """Multiple running tasks each get their own section."""
        running = [
            _make_task("t1", "task-alpha", status="running"),
            _make_task("t2", "task-beta", status="running"),
        ]
        events = {
            "t1": [
                _make_stream_entry(
                    "tool_use",
                    {
                        "type": "tool_use",
                        "tool": {"name": "Read", "input": {"file_path": "/alpha.py"}},
                    },
                ),
            ],
            "t2": [
                _make_stream_entry(
                    "tool_use",
                    {
                        "type": "tool_use",
                        "tool": {"name": "Write", "input": {"file_path": "/beta.py"}},
                    },
                ),
            ],
        }
        panel = build_streaming_detail_panel(running, events)
        text = _render_to_plain(panel)
        assert "task-alpha" in text
        assert "task-beta" in text
        assert "/alpha.py" in text
        assert "/beta.py" in text

    def test_mixed_event_types(self):
        """Mix of tool_use, assistant, and result events render together."""
        running = [_make_task("t1", "mixed-task", status="running")]
        events = {
            "t1": [
                _make_stream_entry(
                    "assistant",
                    {
                        "type": "assistant",
                        "message": {
                            "content": [{"type": "text", "text": "Analyzing..."}]
                        },
                    },
                ),
                _make_stream_entry(
                    "tool_use",
                    {
                        "type": "tool_use",
                        "tool": {"name": "Bash", "input": {"command": "ls"}},
                    },
                ),
                _make_stream_entry(
                    "result",
                    {
                        "type": "result",
                        "result": "Done.",
                    },
                ),
            ]
        }
        panel = build_streaming_detail_panel(running, events)
        text = _render_to_plain(panel)
        assert "Analyzing" in text
        assert "Bash" in text
        assert "Done." in text

    def test_system_events_skipped(self):
        """System events don't produce visible output."""
        running = [_make_task("t1", "sys-task", status="running")]
        events = {
            "t1": [
                _make_stream_entry("system", {"type": "system", "subtype": "init"}),
            ]
        }
        panel = build_streaming_detail_panel(running, events)
        text = _render_to_plain(panel)
        # System events produce no display lines; task header still there
        assert "sys-task" in text
        assert "init" not in text

    def test_tool_result_events_skipped(self):
        """tool_result events don't produce visible output (too verbose)."""
        running = [_make_task("t1", "result-task", status="running")]
        events = {
            "t1": [
                _make_stream_entry(
                    "tool_result",
                    {
                        "type": "tool_result",
                        "tool": {"content": "long output..."},
                    },
                ),
            ]
        }
        panel = build_streaming_detail_panel(running, events)
        text = _render_to_plain(panel)
        assert "result-task" in text
        assert "long output" not in text

    def test_scroll_offset_applied(self):
        """Scroll offset shifts the visible window upward."""
        running = [_make_task("t1", "scrolling-task", status="running")]
        # Create many events to exceed max_lines
        many_events = []
        for i in range(40):
            many_events.append(
                _make_stream_entry(
                    "tool_use",
                    {
                        "type": "tool_use",
                        "tool": {
                            "name": "Read",
                            "input": {"file_path": f"/file{i}.py"},
                        },
                    },
                )
            )
        events = {"t1": many_events}

        # Without scroll: shows newest (last events)
        panel_default = build_streaming_detail_panel(
            running, events, scroll_offset=0, max_lines=10
        )
        text_default = _render_to_plain(panel_default)
        assert "/file39.py" in text_default

        # With scroll offset: shows older events
        panel_scrolled = build_streaming_detail_panel(
            running, events, scroll_offset=20, max_lines=10
        )
        text_scrolled = _render_to_plain(panel_scrolled)
        # Should show events further back
        assert "/file39.py" not in text_scrolled
        assert "offset: 20" in text_scrolled

    def test_scroll_offset_zero_shows_newest(self):
        """Default scroll_offset=0 shows the most recent events."""
        running = [_make_task("t1", "recent-task", status="running")]
        events = {
            "t1": [
                _make_stream_entry(
                    "tool_use",
                    {
                        "type": "tool_use",
                        "tool": {"name": "Read", "input": {"file_path": "/old.py"}},
                    },
                ),
                _make_stream_entry(
                    "tool_use",
                    {
                        "type": "tool_use",
                        "tool": {"name": "Read", "input": {"file_path": "/newest.py"}},
                    },
                ),
            ]
        }
        panel = build_streaming_detail_panel(running, events, scroll_offset=0)
        text = _render_to_plain(panel)
        assert "/newest.py" in text

    def test_max_lines_limit(self):
        """Panel respects max_lines limit."""
        running = [_make_task("t1", "busy-task", status="running")]
        many_events = []
        for i in range(50):
            many_events.append(
                _make_stream_entry(
                    "tool_use",
                    {
                        "type": "tool_use",
                        "tool": {
                            "name": "Read",
                            "input": {"file_path": f"/file{i}.py"},
                        },
                    },
                )
            )
        events = {"t1": many_events}
        panel = build_streaming_detail_panel(running, events, max_lines=5)
        text = _render_to_plain(panel)
        # Should not contain all 50 files - limited by max_lines
        file_count = sum(1 for i in range(50) if f"/file{i}.py" in text)
        assert file_count <= 5

    def test_assistant_no_text_blocks_skipped(self):
        """Assistant message without text blocks produces no output."""
        running = [_make_task("t1", "tool-only-task", status="running")]
        events = {
            "t1": [
                _make_stream_entry(
                    "assistant",
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {"type": "tool_use", "name": "Read", "input": {}}
                            ],
                        },
                    },
                ),
            ]
        }
        panel = build_streaming_detail_panel(running, events)
        text = _render_to_plain(panel)
        assert "tool-only-task" in text
        # No reasoning line should appear

    def test_result_no_text_skipped(self):
        """Result event with empty result text produces no output."""
        running = [_make_task("t1", "empty-result-task", status="running")]
        events = {
            "t1": [
                _make_stream_entry(
                    "result",
                    {
                        "type": "result",
                        "result": "",
                    },
                ),
            ]
        }
        panel = build_streaming_detail_panel(running, events)
        text = _render_to_plain(panel)
        assert "empty-result-task" in text

    def test_invalid_content_skipped_gracefully(self):
        """Entries with unparseable content are silently skipped."""
        running = [_make_task("t1", "bad-data-task", status="running")]
        events = {
            "t1": [
                {
                    "timestamp": "...",
                    "type": "stream_event",
                    "content": "not-json",
                    "stream_type": "unknown",
                },
                _make_stream_entry(
                    "tool_use",
                    {
                        "type": "tool_use",
                        "tool": {"name": "Read", "input": {"file_path": "/good.py"}},
                    },
                ),
            ]
        }
        panel = build_streaming_detail_panel(running, events)
        text = _render_to_plain(panel)
        assert "/good.py" in text
        assert "bad-data-task" in text

    def test_renders_at_narrow_width(self):
        """Panel renders without crash at narrow terminal width."""
        running = [_make_task("t1", "narrow-task", status="running")]
        events = {
            "t1": [
                _make_stream_entry(
                    "tool_use",
                    {
                        "type": "tool_use",
                        "tool": {
                            "name": "Read",
                            "input": {"file_path": "/src/main.py"},
                        },
                    },
                ),
            ]
        }
        panel = build_streaming_detail_panel(running, events)
        text = _render_to_plain(panel, width=40)
        assert "Streaming Detail" in text

    def test_renders_at_wide_width(self):
        """Panel renders without crash at wide terminal width."""
        running = [_make_task("t1", "wide-task", status="running")]
        panel = build_streaming_detail_panel(running, {})
        text = _render_to_plain(panel, width=200)
        assert "Streaming Detail" in text


# ── Active Dashboard with Streaming ──────────────────────────────────────


class TestBuildActiveDashboardWithStreaming:
    """Test the three-panel dashboard layout when streaming data is provided.

    New layout: left two-thirds split vertically (active_plan top,
    streaming bottom) | right one-third event stream.
    """

    def test_three_panel_layout_with_streaming(self):
        """When stream_events_by_task is provided, left side has 2 panels."""
        layout = build_active_dashboard(
            [], [], [], [], [], [], stream_events_by_task={}
        )
        # Root has left + events
        root_names = [c.name for c in layout.children]
        assert "left" in root_names
        assert "events" in root_names
        assert len(layout.children) == 2
        # Left container has active_plan + streaming
        left = layout["left"]
        left_names = [c.name for c in left.children]
        assert "active_plan" in left_names
        assert "streaming" in left_names
        assert len(left.children) == 2

    def test_two_panel_layout_without_streaming(self):
        """When stream_events_by_task is None (default), left has only active_plan."""
        layout = build_active_dashboard([], [], [], [], [], [])
        root_names = [c.name for c in layout.children]
        assert "left" in root_names
        assert "events" in root_names
        assert len(layout.children) == 2
        # Left only has active_plan, no streaming
        left = layout["left"]
        left_names = [c.name for c in left.children]
        assert "active_plan" in left_names
        assert "streaming" not in left_names

    def test_streaming_panel_ratios(self):
        """Left/right split: left ratio=2, events ratio=1; left splits 50/50."""
        layout = build_active_dashboard(
            [], [], [], [], [], [], stream_events_by_task={}
        )
        # Root split: left 2/3, events 1/3
        assert layout.children[0].name == "left"
        assert layout.children[0].ratio == 2
        assert layout.children[1].name == "events"
        assert layout.children[1].ratio == 1
        # Left vertical split: 50/50
        left = layout["left"]
        assert left.children[0].name == "active_plan"
        assert left.children[0].ratio == 1
        assert left.children[1].name == "streaming"
        assert left.children[1].ratio == 1

    def test_streaming_renders_with_data(self):
        """Full dashboard with streaming data renders correctly."""
        running = [_make_task("r1", "live-task", status="running")]
        stream_events = {
            "r1": [
                _make_stream_entry(
                    "tool_use",
                    {
                        "type": "tool_use",
                        "tool": {"name": "Read", "input": {"file_path": "/src/app.py"}},
                    },
                ),
            ]
        }
        events = [
            {
                "timestamp": "2026-03-23T10:00:00.000Z",
                "event_type": "task_dispatched",
                "task_id": "r1",
            }
        ]

        layout = build_active_dashboard(
            running,
            [],
            [],
            [],
            [],
            events,
            stream_events_by_task=stream_events,
        )
        text = _render_to_plain(layout, width=120)

        assert "Active Plan" in text
        assert "Streaming Detail" in text
        assert "Events" in text
        assert "/src/app.py" in text
        assert "task_dispatched" in text

    def test_scroll_offset_passed_through(self):
        """Scroll offset is passed to the streaming detail panel."""
        running = [_make_task("t1", "scroll-task", status="running")]
        many_events = [
            _make_stream_entry(
                "tool_use",
                {
                    "type": "tool_use",
                    "tool": {"name": "Read", "input": {"file_path": f"/f{i}.py"}},
                },
            )
            for i in range(50)
        ]
        layout = build_active_dashboard(
            running,
            [],
            [],
            [],
            [],
            [],
            stream_events_by_task={"t1": many_events},
            scroll_offset=10,
        )
        text = _render_to_plain(layout, width=120)
        assert "offset: 10" in text

    def test_backward_compat_no_streaming_args(self):
        """Calling build_active_dashboard without new args works exactly as before."""
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
        # No streaming panel
        assert "Streaming Detail" not in text


# ── Layout structure verification ────────────────────────────────────────


class TestLayoutStructure:
    """Verify the new left/right layout structure.

    New layout:
      root (split_row)
      ├── left (ratio=2, split_column)
      │   ├── active_plan (ratio=1) — DAG status (top-left)
      │   └── streaming (ratio=1)   — agent detail (bottom-left)
      └── events (ratio=1)          — event stream (right, full-height)
    """

    def test_root_is_horizontal_split(self):
        """Root layout uses split_row (horizontal) not split_column."""
        layout = build_active_dashboard(
            [], [], [], [], [], [], stream_events_by_task={}
        )
        # Rich Layout stores the split direction; children of a row split
        # are arranged horizontally. We verify by checking child count and names.
        assert len(layout.children) == 2
        assert layout.children[0].name == "left"
        assert layout.children[1].name == "events"

    def test_left_container_is_vertical_split(self):
        """Left container splits vertically into active_plan and streaming."""
        layout = build_active_dashboard(
            [], [], [], [], [], [], stream_events_by_task={}
        )
        left = layout["left"]
        assert len(left.children) == 2
        assert left.children[0].name == "active_plan"
        assert left.children[1].name == "streaming"

    def test_left_takes_two_thirds_width(self):
        """Left side has ratio=2, events has ratio=1 (2:1 = two-thirds)."""
        layout = build_active_dashboard(
            [], [], [], [], [], [], stream_events_by_task={}
        )
        assert layout.children[0].ratio == 2  # left
        assert layout.children[1].ratio == 1  # events

    def test_left_panels_split_50_50(self):
        """Active plan and streaming each have ratio=1 (50/50 vertical)."""
        layout = build_active_dashboard(
            [], [], [], [], [], [], stream_events_by_task={}
        )
        left = layout["left"]
        assert left.children[0].ratio == 1  # active_plan
        assert left.children[1].ratio == 1  # streaming

    def test_events_is_full_height_right_column(self):
        """Events panel is a direct child of root (not nested in left)."""
        layout = build_active_dashboard(
            [], [], [], [], [], [], stream_events_by_task={}
        )
        # events is at root level, not inside left
        root_names = [c.name for c in layout.children]
        assert "events" in root_names
        left_names = [c.name for c in layout["left"].children]
        assert "events" not in left_names

    def test_all_three_panels_accessible_by_name(self):
        """All three panels can be accessed via layout['name']."""
        layout = build_active_dashboard(
            [], [], [], [], [], [], stream_events_by_task={}
        )
        # These should not raise KeyError
        assert layout["active_plan"] is not None
        assert layout["streaming"] is not None
        assert layout["events"] is not None

    def test_without_streaming_still_horizontal(self):
        """Without streaming, layout is still left/right split."""
        layout = build_active_dashboard([], [], [], [], [], [])
        assert len(layout.children) == 2
        assert layout.children[0].name == "left"
        assert layout.children[1].name == "events"
        # Left only has active_plan
        left = layout["left"]
        assert len(left.children) == 1
        assert left.children[0].name == "active_plan"


# ── QueryAPI + Streaming Panel integration ───────────────────────────────


class TestQueryAPIStreamingIntegration:
    """Test that QueryAPI.get_task_stream_events() data renders correctly
    in the streaming detail panel."""

    def test_full_pipeline_with_session_data(self):
        """Simulate: session logger → QueryAPI → streaming panel."""
        from corc.sessions import SessionLogger

        # Create a session logger with stream events
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            sl = SessionLogger(tmpdir)
            sl.log_stream_event(
                "t1",
                1,
                {
                    "type": "assistant",
                    "message": {
                        "content": [{"type": "text", "text": "Analyzing codebase."}],
                    },
                },
            )
            sl.log_stream_event(
                "t1",
                1,
                {
                    "type": "tool_use",
                    "tool": {"name": "Read", "input": {"file_path": "/src/main.py"}},
                },
            )
            sl.log_stream_event(
                "t1",
                1,
                {
                    "type": "tool_use",
                    "tool": {
                        "name": "Write",
                        "input": {"file_path": "/src/feature.py"},
                    },
                },
            )

            # Read back via session logger (same path as QueryAPI.get_task_stream_events)
            entries = sl.read_session("t1", 1)
            stream_entries = [e for e in entries if e.get("type") == "stream_event"]

            running = [_make_task("t1", "feature-impl", status="running")]
            panel = build_streaming_detail_panel(running, {"t1": stream_entries})
            text = _render_to_plain(panel)

            assert "feature-impl" in text
            assert "Analyzing codebase." in text
            assert "/src/main.py" in text
            assert "/src/feature.py" in text

    def test_real_format_stream_events(self):
        """Stream events matching real claude CLI output render correctly."""
        # Real-format events with extra fields (session_id, uuid, etc.)
        running = [_make_task("t1", "real-task", status="running")]
        events = {
            "t1": [
                _make_stream_entry(
                    "assistant",
                    {
                        "type": "assistant",
                        "message": {
                            "model": "claude-sonnet-4-20250514",
                            "id": "msg_01abc",
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "I'll read the file first."}
                            ],
                            "stop_reason": None,
                            "usage": {"input_tokens": 100, "output_tokens": 10},
                        },
                        "parent_tool_use_id": None,
                        "session_id": "866b421f-2c52-418d-93a9-4c6481efc10f",
                        "uuid": "b570c34f-eef4-4662-9287-887250a4dafe",
                    },
                ),
                _make_stream_entry(
                    "tool_use",
                    {
                        "type": "tool_use",
                        "tool": {
                            "type": "tool_use",
                            "id": "toolu_01abc",
                            "name": "Read",
                            "input": {"file_path": "/src/main.py"},
                        },
                        "session_id": "866b421f-2c52-418d-93a9-4c6481efc10f",
                        "uuid": "c1d2e3f4-5678-9abc-def0-123456789012",
                    },
                ),
            ]
        }
        panel = build_streaming_detail_panel(running, events)
        text = _render_to_plain(panel)

        assert "I'll read the file first." in text
        assert "Read" in text
        assert "/src/main.py" in text

    def test_empty_stream_events_shows_waiting(self):
        """When get_task_stream_events returns [], shows waiting message."""
        running = [_make_task("t1", "new-task", status="running")]
        events = {"t1": []}
        panel = build_streaming_detail_panel(running, events)
        text = _render_to_plain(panel)
        assert "Waiting for stream events" in text

    def test_checklist_from_task_displayed(self):
        """Task checklist field is rendered as progress in streaming panel."""
        checklist = json.dumps(
            [
                {"item": "Write unit tests", "done": True},
                {"item": "Implement function", "done": True},
                {"item": "Update docs", "done": False},
                {"item": "Run CI", "done": False},
            ]
        )
        running = [
            _make_task("t1", "checklist-task", status="running", checklist=checklist)
        ]
        events = {
            "t1": [
                _make_stream_entry(
                    "tool_use",
                    {
                        "type": "tool_use",
                        "tool": {"name": "Read", "input": {"file_path": "/test.py"}},
                    },
                ),
            ]
        }
        panel = build_streaming_detail_panel(running, events)
        text = _render_to_plain(panel)
        assert "2/4 items done" in text
        assert "/test.py" in text


# ── Agent deduplication ──────────────────────────────────────────────────


class TestDeduplicateAgents:
    """Tests for _deduplicate_agents — filtering to latest agent per task."""

    def test_empty_list(self):
        assert _deduplicate_agents([]) == []

    def test_single_agent_unchanged(self):
        agents = [
            {
                "id": "agent-1",
                "task_id": "t1",
                "role": "implementer",
                "status": "running",
                "started": "2026-01-01T00:00:00Z",
            }
        ]
        result = _deduplicate_agents(agents)
        assert len(result) == 1
        assert result[0]["id"] == "agent-1"

    def test_multiple_agents_same_task_keeps_latest(self):
        """When multiple idle agents exist for the same task, keep the most recent."""
        agents = [
            {
                "id": "agent-old",
                "task_id": "t1",
                "role": "implementer",
                "status": "idle",
                "started": "2026-01-01T00:00:00Z",
            },
            {
                "id": "agent-new",
                "task_id": "t1",
                "role": "implementer",
                "status": "idle",
                "started": "2026-01-02T00:00:00Z",
            },
        ]
        result = _deduplicate_agents(agents)
        assert len(result) == 1
        assert result[0]["id"] == "agent-new"

    def test_prefers_active_over_idle(self):
        """An active (non-idle) agent is preferred over a newer idle one."""
        agents = [
            {
                "id": "agent-idle",
                "task_id": "t1",
                "role": "implementer",
                "status": "idle",
                "started": "2026-01-02T00:00:00Z",
            },
            {
                "id": "agent-running",
                "task_id": "t1",
                "role": "implementer",
                "status": "running",
                "started": "2026-01-01T00:00:00Z",
            },
        ]
        result = _deduplicate_agents(agents)
        assert len(result) == 1
        assert result[0]["id"] == "agent-running"

    def test_multiple_active_keeps_latest(self):
        """Among multiple active agents for same task, keep the most recently started."""
        agents = [
            {
                "id": "agent-old-run",
                "task_id": "t1",
                "role": "implementer",
                "status": "running",
                "started": "2026-01-01T00:00:00Z",
            },
            {
                "id": "agent-new-run",
                "task_id": "t1",
                "role": "implementer",
                "status": "running",
                "started": "2026-01-02T00:00:00Z",
            },
        ]
        result = _deduplicate_agents(agents)
        assert len(result) == 1
        assert result[0]["id"] == "agent-new-run"

    def test_different_tasks_not_deduplicated(self):
        """Agents for different tasks are all kept."""
        agents = [
            {
                "id": "agent-1",
                "task_id": "t1",
                "role": "implementer",
                "status": "idle",
                "started": "2026-01-01T00:00:00Z",
            },
            {
                "id": "agent-2",
                "task_id": "t2",
                "role": "implementer",
                "status": "idle",
                "started": "2026-01-01T00:00:00Z",
            },
        ]
        result = _deduplicate_agents(agents)
        assert len(result) == 2

    def test_three_agents_same_task_one_active(self):
        """Three agents for same task: 2 idle + 1 running → keep running."""
        agents = [
            {
                "id": "agent-1",
                "task_id": "t1",
                "role": "implementer",
                "status": "idle",
                "started": "2026-01-01T00:00:00Z",
            },
            {
                "id": "agent-2",
                "task_id": "t1",
                "role": "implementer",
                "status": "running",
                "started": "2026-01-02T00:00:00Z",
            },
            {
                "id": "agent-3",
                "task_id": "t1",
                "role": "implementer",
                "status": "idle",
                "started": "2026-01-03T00:00:00Z",
            },
        ]
        result = _deduplicate_agents(agents)
        assert len(result) == 1
        assert result[0]["id"] == "agent-2"

    def test_agents_without_task_id_kept(self):
        """Agents without a task_id are all kept (edge case)."""
        agents = [
            {"id": "agent-1", "role": "implementer", "status": "idle"},
            {"id": "agent-2", "role": "reviewer", "status": "idle"},
        ]
        result = _deduplicate_agents(agents)
        assert len(result) == 2

    def test_agents_without_started_field(self):
        """Agents without a started timestamp are handled gracefully."""
        agents = [
            {"id": "agent-1", "task_id": "t1", "role": "implementer", "status": "idle"},
            {
                "id": "agent-2",
                "task_id": "t1",
                "role": "implementer",
                "status": "idle",
                "started": "2026-01-01T00:00:00Z",
            },
        ]
        result = _deduplicate_agents(agents)
        assert len(result) == 1
        # Agent with a timestamp should be preferred (sorts higher)
        assert result[0]["id"] == "agent-2"

    def test_does_not_mutate_input(self):
        """Input list is not modified."""
        agents = [
            {
                "id": "agent-1",
                "task_id": "t1",
                "role": "implementer",
                "status": "idle",
                "started": "2026-01-01T00:00:00Z",
            },
            {
                "id": "agent-2",
                "task_id": "t1",
                "role": "implementer",
                "status": "idle",
                "started": "2026-01-02T00:00:00Z",
            },
        ]
        original_len = len(agents)
        _deduplicate_agents(agents)
        assert len(agents) == original_len


class TestDeduplicateTaskAgents:
    """Tests for _deduplicate_task_agents — task-level wrapper."""

    def test_task_without_agents_key(self):
        task = {"id": "t1", "name": "test", "status": "running"}
        result = _deduplicate_task_agents(task)
        assert result == task

    def test_task_with_single_agent(self):
        task = {
            "id": "t1",
            "name": "test",
            "status": "running",
            "agents": [
                {
                    "id": "agent-1",
                    "task_id": "t1",
                    "role": "implementer",
                    "status": "running",
                }
            ],
        }
        result = _deduplicate_task_agents(task)
        assert len(result["agents"]) == 1

    def test_task_with_duplicate_agents(self):
        task = {
            "id": "t1",
            "name": "test",
            "status": "running",
            "agents": [
                {
                    "id": "agent-old",
                    "task_id": "t1",
                    "role": "implementer",
                    "status": "idle",
                    "started": "2026-01-01T00:00:00Z",
                },
                {
                    "id": "agent-new",
                    "task_id": "t1",
                    "role": "implementer",
                    "status": "running",
                    "started": "2026-01-02T00:00:00Z",
                },
            ],
        }
        result = _deduplicate_task_agents(task)
        assert len(result["agents"]) == 1
        assert result["agents"][0]["id"] == "agent-new"
        # Original task should not be modified
        assert len(task["agents"]) == 2


class TestActivePlanPanelDuplicateAgents:
    """Test that the active plan panel only shows the latest agent per running task.

    This is the key integration test verifying that duplicate [role idle] tags
    no longer appear when a task has been dispatched multiple times.
    """

    def test_single_agent_tag_with_multiple_agent_records(self):
        """When a task has multiple agent records, only one [role ...] tag is shown."""
        running = [
            _make_task(
                "t1",
                "my-task",
                status="running",
                agents=[
                    {
                        "id": "agent-old",
                        "task_id": "t1",
                        "role": "implementer",
                        "status": "idle",
                        "started": "2026-01-01T00:00:00Z",
                    },
                    {
                        "id": "agent-new",
                        "task_id": "t1",
                        "role": "implementer",
                        "status": "running",
                        "pid": 9999,
                        "started": "2026-01-02T00:00:00Z",
                    },
                ],
            )
        ]
        panel = build_active_plan_panel(running, [], [], [])
        text = _render_to_plain(panel)
        # Should show agent info exactly once, not twice
        assert text.count("[implementer") == 1
        # Should show the active agent's info
        assert "9999" in text
        assert "running" in text

    def test_no_duplicate_idle_tags(self):
        """Duplicate [role idle] tags should not appear when all agents are idle."""
        running = [
            _make_task(
                "t1",
                "idle-task",
                status="running",
                agents=[
                    {
                        "id": "agent-1",
                        "task_id": "t1",
                        "role": "implementer",
                        "status": "idle",
                        "started": "2026-01-01T00:00:00Z",
                    },
                    {
                        "id": "agent-2",
                        "task_id": "t1",
                        "role": "implementer",
                        "status": "idle",
                        "started": "2026-01-02T00:00:00Z",
                    },
                    {
                        "id": "agent-3",
                        "task_id": "t1",
                        "role": "implementer",
                        "status": "idle",
                        "started": "2026-01-03T00:00:00Z",
                    },
                ],
            )
        ]
        panel = build_active_plan_panel(running, [], [], [])
        text = _render_to_plain(panel)
        # Only one agent tag should appear
        assert text.count("[implementer") == 1

    def test_multiple_running_tasks_each_deduplicated(self):
        """Each running task gets its own deduplication."""
        running = [
            _make_task(
                "t1",
                "task-one",
                status="running",
                agents=[
                    {
                        "id": "a1",
                        "task_id": "t1",
                        "role": "implementer",
                        "status": "idle",
                        "started": "2026-01-01T00:00:00Z",
                    },
                    {
                        "id": "a2",
                        "task_id": "t1",
                        "role": "implementer",
                        "status": "running",
                        "pid": 111,
                        "started": "2026-01-02T00:00:00Z",
                    },
                ],
            ),
            _make_task(
                "t2",
                "task-two",
                status="running",
                agents=[
                    {
                        "id": "a3",
                        "task_id": "t2",
                        "role": "reviewer",
                        "status": "idle",
                        "started": "2026-01-01T00:00:00Z",
                    },
                    {
                        "id": "a4",
                        "task_id": "t2",
                        "role": "reviewer",
                        "status": "idle",
                        "started": "2026-01-03T00:00:00Z",
                    },
                ],
            ),
        ]
        panel = build_active_plan_panel(running, [], [], [])
        text = _render_to_plain(panel)
        # Each task should have exactly one agent tag
        assert text.count("[implementer") == 1
        assert text.count("[reviewer") == 1


class TestQueryAPIDuplicateAgents:
    """Test that QueryAPI.get_running_tasks_with_agents deduplicates agents."""

    def _make_query_api(self, all_tasks, agents_map=None):
        """Create a QueryAPI with mocked dependencies."""
        from corc.queries import QueryAPI

        ws = MagicMock()
        ws.list_tasks.side_effect = lambda status=None: (
            [t for t in all_tasks if t["status"] == status] if status else all_tasks
        )
        ws.get_agents_for_task.side_effect = lambda tid: (agents_map or {}).get(tid, [])

        al = MagicMock()
        al.read_recent.return_value = []
        sl = MagicMock()

        return QueryAPI(ws, al, sl)

    def test_deduplicates_agents_in_query(self):
        """get_running_tasks_with_agents should return at most one agent per task."""
        tasks = [
            _make_task("t1", "my-task", status="running"),
        ]
        agents = {
            "t1": [
                {
                    "id": "agent-old",
                    "task_id": "t1",
                    "role": "implementer",
                    "status": "idle",
                    "started": "2026-01-01T00:00:00Z",
                },
                {
                    "id": "agent-new",
                    "task_id": "t1",
                    "role": "implementer",
                    "status": "running",
                    "pid": 5555,
                    "started": "2026-01-02T00:00:00Z",
                },
            ]
        }
        api = self._make_query_api(tasks, agents)
        running = api.get_running_tasks_with_agents()
        assert len(running) == 1
        assert len(running[0]["agents"]) == 1
        assert running[0]["agents"][0]["id"] == "agent-new"

    def test_no_agents_returns_empty_list(self):
        """Tasks with no agents still work fine."""
        tasks = [_make_task("t1", "no-agents", status="running")]
        api = self._make_query_api(tasks)
        running = api.get_running_tasks_with_agents()
        assert len(running) == 1
        assert running[0]["agents"] == []

    def test_single_agent_unchanged(self):
        """Tasks with a single agent are returned as-is."""
        tasks = [_make_task("t1", "single-agent", status="running")]
        agents = {
            "t1": [
                {
                    "id": "agent-1",
                    "task_id": "t1",
                    "role": "implementer",
                    "status": "running",
                }
            ]
        }
        api = self._make_query_api(tasks, agents)
        running = api.get_running_tasks_with_agents()
        assert len(running[0]["agents"]) == 1
