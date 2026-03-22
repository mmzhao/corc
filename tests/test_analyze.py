"""Tests for cost analysis, duration trends, failure reporting, and alerts."""

import json
import time
from pathlib import Path

import pytest
import yaml

from corc.audit import AuditLog
from corc.analyze import (
    CostAlertConfig,
    CostBreakdown,
    CostAlert,
    DurationEntry,
    FailureEntry,
    aggregate_costs,
    compute_costs_today,
    compute_costs_project,
    compute_duration_trends,
    compute_failures,
    check_cost_alerts,
    load_alert_config,
    format_cost_breakdown,
    format_duration_trends,
    format_failures,
    format_alerts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def audit_dir(tmp_path):
    """Create a temporary audit log directory."""
    d = tmp_path / "events"
    d.mkdir()
    return d


@pytest.fixture
def audit_log(audit_dir):
    """Create an AuditLog pointing at a temp directory."""
    return AuditLog(audit_dir)


@pytest.fixture
def corc_dir(tmp_path):
    """Create a temporary .corc directory."""
    d = tmp_path / ".corc"
    d.mkdir()
    return d


def _write_events(audit_dir, date_str, events):
    """Write events directly to a specific date file."""
    path = audit_dir / f"{date_str}.jsonl"
    with open(path, "a") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


# ---------------------------------------------------------------------------
# Cost aggregation tests
# ---------------------------------------------------------------------------


class TestAggregateCosts:
    def test_empty_events(self):
        breakdown = aggregate_costs([])
        assert breakdown.total_usd == 0.0
        assert breakdown.event_count == 0
        assert breakdown.by_task == {}
        assert breakdown.by_role == {}
        assert breakdown.by_project == {}

    def test_single_cost_event(self):
        events = [
            {
                "event_type": "task_dispatch_complete",
                "cost_usd": 1.50,
                "task_id": "t1",
                "role": "implementer",
                "project": "corc",
            }
        ]
        breakdown = aggregate_costs(events)
        assert breakdown.total_usd == 1.50
        assert breakdown.event_count == 1
        assert breakdown.by_task == {"t1": 1.50}
        assert breakdown.by_role == {"implementer": 1.50}
        assert breakdown.by_project == {"corc": 1.50}

    def test_multiple_cost_events(self):
        events = [
            {
                "cost_usd": 1.00,
                "task_id": "t1",
                "role": "implementer",
                "project": "corc",
            },
            {"cost_usd": 2.00, "task_id": "t2", "role": "reviewer", "project": "corc"},
            {
                "cost_usd": 0.50,
                "task_id": "t1",
                "role": "implementer",
                "project": "other",
            },
        ]
        breakdown = aggregate_costs(events)
        assert breakdown.total_usd == pytest.approx(3.50)
        assert breakdown.event_count == 3
        assert breakdown.by_task["t1"] == pytest.approx(1.50)
        assert breakdown.by_task["t2"] == pytest.approx(2.00)
        assert breakdown.by_role["implementer"] == pytest.approx(1.50)
        assert breakdown.by_role["reviewer"] == pytest.approx(2.00)
        assert breakdown.by_project["corc"] == pytest.approx(3.00)
        assert breakdown.by_project["other"] == pytest.approx(0.50)

    def test_events_without_cost_ignored(self):
        events = [
            {"event_type": "task_created", "task_id": "t1"},
            {"event_type": "task_dispatch_complete", "cost_usd": 1.00, "task_id": "t1"},
            {"event_type": "tool_use", "task_id": "t1"},
        ]
        breakdown = aggregate_costs(events)
        assert breakdown.total_usd == pytest.approx(1.00)
        assert breakdown.event_count == 1

    def test_missing_optional_fields_default(self):
        events = [{"cost_usd": 2.00}]
        breakdown = aggregate_costs(events)
        assert breakdown.total_usd == pytest.approx(2.00)
        assert breakdown.by_task == {"unknown": 2.00}
        assert breakdown.by_role == {"unknown": 2.00}
        assert breakdown.by_project == {"unassigned": 2.00}

    def test_zero_cost_events_counted(self):
        events = [{"cost_usd": 0.0, "task_id": "t1", "role": "scout", "project": "p1"}]
        breakdown = aggregate_costs(events)
        assert breakdown.event_count == 1
        assert breakdown.total_usd == 0.0


class TestComputeCostsToday:
    def test_today_costs(self, audit_log):
        today = time.strftime("%Y-%m-%d", time.gmtime())
        audit_log.log(
            "task_dispatch_complete",
            task_id="t1",
            cost_usd=1.25,
            role="implementer",
            project="corc",
        )
        audit_log.log(
            "task_dispatch_complete",
            task_id="t2",
            cost_usd=0.75,
            role="reviewer",
            project="corc",
        )

        breakdown = compute_costs_today(audit_log)
        assert breakdown.total_usd == pytest.approx(2.00)
        assert breakdown.event_count == 2

    def test_today_no_costs(self, audit_log):
        # Log events without cost_usd
        audit_log.log("task_created", task_id="t1", name="test")
        breakdown = compute_costs_today(audit_log)
        assert breakdown.total_usd == 0.0
        assert breakdown.event_count == 0


class TestComputeCostsProject:
    def test_filter_by_project(self, audit_log, audit_dir):
        today = time.strftime("%Y-%m-%d", time.gmtime())
        _write_events(
            audit_dir,
            today,
            [
                {
                    "timestamp": "2026-03-22T10:00:00.000Z",
                    "cost_usd": 1.00,
                    "task_id": "t1",
                    "project": "corc",
                },
                {
                    "timestamp": "2026-03-22T10:01:00.000Z",
                    "cost_usd": 2.00,
                    "task_id": "t2",
                    "project": "other",
                },
                {
                    "timestamp": "2026-03-22T10:02:00.000Z",
                    "cost_usd": 0.50,
                    "task_id": "t3",
                    "project": "corc",
                },
            ],
        )

        breakdown = compute_costs_project(audit_log, "corc")
        assert breakdown.total_usd == pytest.approx(1.50)
        assert breakdown.event_count == 2

    def test_project_with_since(self, audit_dir):
        _write_events(
            audit_dir,
            "2026-03-20",
            [
                {
                    "timestamp": "2026-03-20T10:00:00.000Z",
                    "cost_usd": 5.00,
                    "task_id": "t1",
                    "project": "corc",
                },
            ],
        )
        _write_events(
            audit_dir,
            "2026-03-22",
            [
                {
                    "timestamp": "2026-03-22T10:00:00.000Z",
                    "cost_usd": 1.00,
                    "task_id": "t2",
                    "project": "corc",
                },
            ],
        )

        al = AuditLog(audit_dir)
        breakdown = compute_costs_project(al, "corc", since="2026-03-21")
        assert breakdown.total_usd == pytest.approx(1.00)
        assert breakdown.event_count == 1

    def test_nonexistent_project(self, audit_log):
        audit_log.log(
            "task_dispatch_complete", task_id="t1", cost_usd=1.00, project="corc"
        )
        breakdown = compute_costs_project(audit_log, "nonexistent")
        assert breakdown.total_usd == 0.0
        assert breakdown.event_count == 0


# ---------------------------------------------------------------------------
# Duration trend tests
# ---------------------------------------------------------------------------


class TestComputeDurationTrends:
    def test_basic_duration_trends(self, audit_dir):
        _write_events(
            audit_dir,
            "2026-03-22",
            [
                {
                    "timestamp": "2026-03-22T10:00:00.000Z",
                    "event_type": "task_dispatch_complete",
                    "task_id": "t1",
                    "duration_s": 120.5,
                    "attempt": 1,
                    "exit_code": 0,
                },
                {
                    "timestamp": "2026-03-22T11:00:00.000Z",
                    "event_type": "task_dispatch_complete",
                    "task_id": "t2",
                    "duration_s": 45.3,
                    "attempt": 1,
                    "exit_code": 0,
                },
                {
                    "timestamp": "2026-03-22T12:00:00.000Z",
                    "event_type": "task_dispatch_complete",
                    "task_id": "t3",
                    "duration_s": 200.0,
                    "attempt": 2,
                    "exit_code": 1,
                },
            ],
        )

        al = AuditLog(audit_dir)
        entries = compute_duration_trends(al, last_n=10)
        assert len(entries) == 3
        assert entries[0].task_id == "t1"
        assert entries[0].duration_s == pytest.approx(120.5)
        assert entries[2].task_id == "t3"
        assert entries[2].exit_code == 1

    def test_duration_last_n_limit(self, audit_dir):
        events = []
        for i in range(10):
            events.append(
                {
                    "timestamp": f"2026-03-22T{10 + i:02d}:00:00.000Z",
                    "event_type": "task_dispatch_complete",
                    "task_id": f"t{i}",
                    "duration_s": float(i * 10),
                    "attempt": 1,
                    "exit_code": 0,
                }
            )
        _write_events(audit_dir, "2026-03-22", events)

        al = AuditLog(audit_dir)
        entries = compute_duration_trends(al, last_n=3)
        assert len(entries) == 3
        # Should be the last 3 by timestamp
        assert entries[0].task_id == "t7"
        assert entries[1].task_id == "t8"
        assert entries[2].task_id == "t9"

    def test_duration_empty(self, audit_log):
        entries = compute_duration_trends(audit_log, last_n=10)
        assert entries == []

    def test_duration_ignores_non_dispatch_events(self, audit_dir):
        _write_events(
            audit_dir,
            "2026-03-22",
            [
                {
                    "timestamp": "2026-03-22T10:00:00.000Z",
                    "event_type": "task_created",
                    "task_id": "t1",
                },
                {
                    "timestamp": "2026-03-22T10:05:00.000Z",
                    "event_type": "task_dispatch_complete",
                    "task_id": "t1",
                    "duration_s": 60.0,
                    "attempt": 1,
                    "exit_code": 0,
                },
                {
                    "timestamp": "2026-03-22T10:10:00.000Z",
                    "event_type": "task_failed",
                    "task_id": "t1",
                },
            ],
        )

        al = AuditLog(audit_dir)
        entries = compute_duration_trends(al, last_n=10)
        assert len(entries) == 1
        assert entries[0].task_id == "t1"
        assert entries[0].duration_s == 60.0

    def test_duration_across_multiple_days(self, audit_dir):
        _write_events(
            audit_dir,
            "2026-03-20",
            [
                {
                    "timestamp": "2026-03-20T10:00:00.000Z",
                    "event_type": "task_dispatch_complete",
                    "task_id": "t1",
                    "duration_s": 100.0,
                    "attempt": 1,
                    "exit_code": 0,
                },
            ],
        )
        _write_events(
            audit_dir,
            "2026-03-22",
            [
                {
                    "timestamp": "2026-03-22T10:00:00.000Z",
                    "event_type": "task_dispatch_complete",
                    "task_id": "t2",
                    "duration_s": 200.0,
                    "attempt": 1,
                    "exit_code": 0,
                },
            ],
        )

        al = AuditLog(audit_dir)
        entries = compute_duration_trends(al, last_n=10)
        assert len(entries) == 2
        assert entries[0].task_id == "t1"
        assert entries[1].task_id == "t2"


# ---------------------------------------------------------------------------
# Failure report tests
# ---------------------------------------------------------------------------


class TestComputeFailures:
    def test_basic_failures(self, audit_dir):
        _write_events(
            audit_dir,
            "2026-03-22",
            [
                {
                    "timestamp": "2026-03-22T10:00:00.000Z",
                    "event_type": "task_failed",
                    "task_id": "t1",
                    "exit_code": 1,
                    "attempt": 1,
                },
                {
                    "timestamp": "2026-03-22T11:00:00.000Z",
                    "event_type": "task_failed",
                    "task_id": "t1",
                    "exit_code": 1,
                    "attempt": 2,
                },
                {
                    "timestamp": "2026-03-22T12:00:00.000Z",
                    "event_type": "task_dispatch_complete",
                    "task_id": "t2",
                    "exit_code": 0,
                },
            ],
        )

        al = AuditLog(audit_dir)
        entries = compute_failures(al)
        assert len(entries) == 2
        assert all(e.task_id == "t1" for e in entries)

    def test_failures_since(self, audit_dir):
        _write_events(
            audit_dir,
            "2026-03-20",
            [
                {
                    "timestamp": "2026-03-20T10:00:00.000Z",
                    "event_type": "task_failed",
                    "task_id": "t1",
                    "exit_code": 1,
                    "attempt": 1,
                },
            ],
        )
        _write_events(
            audit_dir,
            "2026-03-22",
            [
                {
                    "timestamp": "2026-03-22T10:00:00.000Z",
                    "event_type": "task_failed",
                    "task_id": "t2",
                    "exit_code": 1,
                    "attempt": 1,
                },
            ],
        )

        al = AuditLog(audit_dir)
        entries = compute_failures(al, since="2026-03-21")
        assert len(entries) == 1
        assert entries[0].task_id == "t2"

    def test_no_failures(self, audit_log):
        audit_log.log("task_dispatch_complete", task_id="t1", exit_code=0)
        entries = compute_failures(audit_log)
        assert entries == []


# ---------------------------------------------------------------------------
# Cost alert tests
# ---------------------------------------------------------------------------


class TestCostAlertConfig:
    def test_default_config(self):
        config = CostAlertConfig()
        assert config.daily_limit_usd == 50.0
        assert config.project_limit_usd == 200.0
        assert config.task_limit_usd == 10.0
        assert config.enabled is True

    def test_load_from_yaml(self, corc_dir):
        config_data = {
            "alerts": {
                "cost": {
                    "enabled": True,
                    "daily_limit_usd": 25.0,
                    "project_limit_usd": 100.0,
                    "task_limit_usd": 5.0,
                }
            }
        }
        config_path = corc_dir / "config.yaml"
        config_path.write_text(yaml.dump(config_data))

        config = load_alert_config(corc_dir)
        assert config.daily_limit_usd == 25.0
        assert config.project_limit_usd == 100.0
        assert config.task_limit_usd == 5.0
        assert config.enabled is True

    def test_load_missing_config_uses_defaults(self, corc_dir):
        config = load_alert_config(corc_dir)
        assert config.daily_limit_usd == 50.0
        assert config.project_limit_usd == 200.0
        assert config.task_limit_usd == 10.0

    def test_load_partial_config(self, corc_dir):
        config_data = {
            "alerts": {
                "cost": {
                    "daily_limit_usd": 15.0,
                    # project_limit_usd and task_limit_usd missing -> use defaults
                }
            }
        }
        config_path = corc_dir / "config.yaml"
        config_path.write_text(yaml.dump(config_data))

        config = load_alert_config(corc_dir)
        assert config.daily_limit_usd == 15.0
        assert config.project_limit_usd == 200.0  # default
        assert config.task_limit_usd == 10.0  # default

    def test_load_empty_yaml(self, corc_dir):
        config_path = corc_dir / "config.yaml"
        config_path.write_text("")

        config = load_alert_config(corc_dir)
        assert config.daily_limit_usd == 50.0  # all defaults

    def test_disabled_config(self, corc_dir):
        config_data = {"alerts": {"cost": {"enabled": False}}}
        config_path = corc_dir / "config.yaml"
        config_path.write_text(yaml.dump(config_data))

        config = load_alert_config(corc_dir)
        assert config.enabled is False


class TestCheckCostAlerts:
    def test_no_alerts_when_disabled(self, audit_log):
        config = CostAlertConfig(enabled=False)
        audit_log.log("step_completed", cost_usd=9999.0)
        alerts = check_cost_alerts(audit_log, config)
        assert alerts == []

    def test_daily_limit_alert(self, audit_log):
        config = CostAlertConfig(daily_limit_usd=1.00)
        # Log events that exceed daily limit
        audit_log.log(
            "step_completed",
            task_id="t1",
            cost_usd=0.60,
            role="implementer",
            project="corc",
        )
        audit_log.log(
            "step_completed",
            task_id="t2",
            cost_usd=0.60,
            role="implementer",
            project="corc",
        )

        alerts = check_cost_alerts(audit_log, config)
        daily_alerts = [a for a in alerts if a.alert_type == "daily"]
        assert len(daily_alerts) == 1
        assert daily_alerts[0].current_usd == pytest.approx(1.20)
        assert daily_alerts[0].threshold_usd == 1.00

    def test_no_daily_alert_under_threshold(self, audit_log):
        config = CostAlertConfig(daily_limit_usd=10.00)
        audit_log.log("step_completed", task_id="t1", cost_usd=0.50)
        alerts = check_cost_alerts(audit_log, config)
        daily_alerts = [a for a in alerts if a.alert_type == "daily"]
        assert daily_alerts == []

    def test_project_limit_alert(self, audit_log):
        config = CostAlertConfig(
            project_limit_usd=2.00, daily_limit_usd=999.0, task_limit_usd=999.0
        )
        audit_log.log("step_completed", task_id="t1", cost_usd=1.50, project="corc")
        audit_log.log("step_completed", task_id="t2", cost_usd=1.00, project="corc")
        audit_log.log("step_completed", task_id="t3", cost_usd=0.10, project="other")

        alerts = check_cost_alerts(audit_log, config)
        proj_alerts = [a for a in alerts if a.alert_type == "project"]
        assert len(proj_alerts) == 1
        assert proj_alerts[0].subject == "corc"
        assert proj_alerts[0].current_usd == pytest.approx(2.50)

    def test_task_limit_alert(self, audit_log):
        config = CostAlertConfig(
            task_limit_usd=1.00, daily_limit_usd=999.0, project_limit_usd=999.0
        )
        # Same task, two cost events
        audit_log.log("step_completed", task_id="t1", cost_usd=0.60, project="p")
        audit_log.log("step_completed", task_id="t1", cost_usd=0.60, project="p")
        audit_log.log("step_completed", task_id="t2", cost_usd=0.50, project="p")

        alerts = check_cost_alerts(audit_log, config)
        task_alerts = [a for a in alerts if a.alert_type == "task"]
        assert len(task_alerts) == 1
        assert task_alerts[0].subject == "t1"
        assert task_alerts[0].current_usd == pytest.approx(1.20)

    def test_multiple_alerts_at_once(self, audit_log):
        config = CostAlertConfig(
            daily_limit_usd=0.50, project_limit_usd=0.50, task_limit_usd=0.50
        )
        audit_log.log(
            "step_completed",
            task_id="t1",
            cost_usd=0.60,
            project="corc",
            role="implementer",
        )

        alerts = check_cost_alerts(audit_log, config)
        alert_types = {a.alert_type for a in alerts}
        assert "daily" in alert_types
        assert "project" in alert_types
        assert "task" in alert_types

    def test_no_alerts_when_no_cost_data(self, audit_log):
        config = CostAlertConfig(daily_limit_usd=0.01)
        audit_log.log("task_created", task_id="t1", name="test")
        alerts = check_cost_alerts(audit_log, config)
        assert alerts == []


# ---------------------------------------------------------------------------
# Formatting tests
# ---------------------------------------------------------------------------


class TestFormatCostBreakdown:
    def test_format_empty(self):
        breakdown = CostBreakdown()
        result = format_cost_breakdown(breakdown)
        assert "No cost data found." in result

    def test_format_with_data(self):
        breakdown = CostBreakdown(
            total_usd=3.50,
            event_count=3,
            by_task={"t1": 2.00, "t2": 1.50},
            by_role={"implementer": 3.50},
            by_project={"corc": 3.50},
        )
        result = format_cost_breakdown(breakdown, title="Test Costs")
        assert "Test Costs" in result
        assert "$3.50" in result
        assert "3 events" in result
        assert "t1" in result
        assert "t2" in result
        assert "implementer" in result
        assert "corc" in result

    def test_format_custom_title(self):
        breakdown = CostBreakdown(total_usd=1.0, event_count=1, by_task={"t1": 1.0})
        result = format_cost_breakdown(breakdown, title="Today's Costs")
        assert "Today's Costs" in result


class TestFormatDurationTrends:
    def test_format_empty(self):
        result = format_duration_trends([])
        assert "No duration data found." in result

    def test_format_with_entries(self):
        entries = [
            DurationEntry(
                task_id="t1",
                duration_s=120.0,
                timestamp="2026-03-22T10:00:00.000Z",
                attempt=1,
                exit_code=0,
            ),
            DurationEntry(
                task_id="t2",
                duration_s=60.0,
                timestamp="2026-03-22T11:00:00.000Z",
                attempt=1,
                exit_code=1,
            ),
        ]
        result = format_duration_trends(entries)
        assert "Duration Trends" in result
        assert "Avg: 90.0s" in result
        assert "Min: 60.0s" in result
        assert "Max: 120.0s" in result
        assert "t1" in result
        assert "t2" in result


class TestFormatFailures:
    def test_format_empty(self):
        result = format_failures([])
        assert "No failures found." in result

    def test_format_with_entries(self):
        entries = [
            FailureEntry(
                task_id="t1",
                timestamp="2026-03-22T10:00:00.000Z",
                exit_code=1,
                attempt=1,
            ),
            FailureEntry(
                task_id="t1",
                timestamp="2026-03-22T11:00:00.000Z",
                exit_code=1,
                attempt=2,
            ),
        ]
        result = format_failures(entries)
        assert "Total failures: 2" in result
        assert "t1: 2 failure(s)" in result


class TestFormatAlerts:
    def test_no_alerts(self):
        result = format_alerts([])
        assert "No cost alerts." in result

    def test_with_alerts(self):
        alerts = [
            CostAlert(
                alert_type="daily",
                current_usd=55.0,
                threshold_usd=50.0,
                subject="2026-03-22",
                message="Daily cost $55.00 exceeds threshold $50.00",
            )
        ]
        result = format_alerts(alerts)
        assert "Cost Alerts" in result
        assert "[daily]" in result
        assert "$55.00" in result


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestAnalyzeCLI:
    """Test the analyze CLI commands via Click test runner."""

    def test_analyze_costs_today(self, tmp_path):
        from click.testing import CliRunner
        from corc.cli import cli

        # Set up minimal project structure
        events_dir = tmp_path / "data" / "events"
        events_dir.mkdir(parents=True)
        (tmp_path / "data").mkdir(exist_ok=True)
        (tmp_path / ".corc").mkdir(exist_ok=True)
        (tmp_path / "knowledge").mkdir(exist_ok=True)

        # Write a mutations file
        mutations_path = tmp_path / "data" / "mutations.jsonl"
        mutations_path.touch()

        # Write cost events for today
        al = AuditLog(events_dir)
        al.log(
            "step_completed",
            task_id="t1",
            cost_usd=1.25,
            role="implementer",
            project="corc",
        )
        al.log(
            "step_completed",
            task_id="t2",
            cost_usd=0.75,
            role="reviewer",
            project="corc",
        )

        runner = CliRunner()
        import os

        with runner.isolated_filesystem(temp_dir=tmp_path.parent):
            os.chdir(str(tmp_path))
            # Create .git to make it detectable as project root
            (tmp_path / ".git").mkdir(exist_ok=True)

            result = runner.invoke(cli, ["analyze", "costs", "--today"])
            assert result.exit_code == 0
            assert "Today's Costs" in result.output
            assert "$2.00" in result.output

    def test_analyze_costs_project(self, tmp_path):
        from click.testing import CliRunner
        from corc.cli import cli

        events_dir = tmp_path / "data" / "events"
        events_dir.mkdir(parents=True)
        (tmp_path / "data").mkdir(exist_ok=True)
        (tmp_path / ".corc").mkdir(exist_ok=True)
        (tmp_path / ".git").mkdir(exist_ok=True)
        (tmp_path / "knowledge").mkdir(exist_ok=True)
        (tmp_path / "data" / "mutations.jsonl").touch()

        al = AuditLog(events_dir)
        al.log("step_completed", task_id="t1", cost_usd=3.00, project="myapp")
        al.log("step_completed", task_id="t2", cost_usd=1.00, project="other")

        runner = CliRunner()
        import os

        with runner.isolated_filesystem(temp_dir=tmp_path.parent):
            os.chdir(str(tmp_path))
            result = runner.invoke(cli, ["analyze", "costs", "--project", "myapp"])
            assert result.exit_code == 0
            assert "myapp" in result.output
            assert "$3.00" in result.output

    def test_analyze_duration(self, tmp_path):
        from click.testing import CliRunner
        from corc.cli import cli

        events_dir = tmp_path / "data" / "events"
        events_dir.mkdir(parents=True)
        (tmp_path / "data").mkdir(exist_ok=True)
        (tmp_path / ".corc").mkdir(exist_ok=True)
        (tmp_path / ".git").mkdir(exist_ok=True)
        (tmp_path / "knowledge").mkdir(exist_ok=True)
        (tmp_path / "data" / "mutations.jsonl").touch()

        al = AuditLog(events_dir)
        al.log(
            "task_dispatch_complete",
            task_id="t1",
            duration_s=120.5,
            attempt=1,
            exit_code=0,
        )
        al.log(
            "task_dispatch_complete",
            task_id="t2",
            duration_s=60.0,
            attempt=1,
            exit_code=0,
        )

        runner = CliRunner()
        import os

        with runner.isolated_filesystem(temp_dir=tmp_path.parent):
            os.chdir(str(tmp_path))
            result = runner.invoke(cli, ["analyze", "duration", "--last", "10"])
            assert result.exit_code == 0
            assert "Duration Trends" in result.output
            assert "t1" in result.output
            assert "t2" in result.output

    def test_analyze_failures(self, tmp_path):
        from click.testing import CliRunner
        from corc.cli import cli

        events_dir = tmp_path / "data" / "events"
        events_dir.mkdir(parents=True)
        (tmp_path / "data").mkdir(exist_ok=True)
        (tmp_path / ".corc").mkdir(exist_ok=True)
        (tmp_path / ".git").mkdir(exist_ok=True)
        (tmp_path / "knowledge").mkdir(exist_ok=True)
        (tmp_path / "data" / "mutations.jsonl").touch()

        al = AuditLog(events_dir)
        al.log("task_failed", task_id="t1", exit_code=1, attempt=1)
        al.log("task_failed", task_id="t1", exit_code=1, attempt=2)

        runner = CliRunner()
        import os

        with runner.isolated_filesystem(temp_dir=tmp_path.parent):
            os.chdir(str(tmp_path))
            result = runner.invoke(cli, ["analyze", "failures"])
            assert result.exit_code == 0
            assert "Failure Report" in result.output
            assert "t1" in result.output

    def test_analyze_costs_alerts(self, tmp_path):
        from click.testing import CliRunner
        from corc.cli import cli

        events_dir = tmp_path / "data" / "events"
        events_dir.mkdir(parents=True)
        (tmp_path / "data").mkdir(exist_ok=True)
        corc_dir = tmp_path / ".corc"
        corc_dir.mkdir(exist_ok=True)
        (tmp_path / ".git").mkdir(exist_ok=True)
        (tmp_path / "knowledge").mkdir(exist_ok=True)
        (tmp_path / "data" / "mutations.jsonl").touch()

        # Set a low threshold
        config_data = {"alerts": {"cost": {"daily_limit_usd": 0.50}}}
        (corc_dir / "config.yaml").write_text(yaml.dump(config_data))

        al = AuditLog(events_dir)
        al.log("step_completed", task_id="t1", cost_usd=0.80, project="corc")

        runner = CliRunner()
        import os

        with runner.isolated_filesystem(temp_dir=tmp_path.parent):
            os.chdir(str(tmp_path))
            result = runner.invoke(cli, ["analyze", "costs", "--alerts"])
            assert result.exit_code == 0
            assert "daily" in result.output.lower() or "Cost Alerts" in result.output
