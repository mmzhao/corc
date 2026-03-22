"""Tests for DAG visualization — layout, rendering, and CLI integration."""

import json

import pytest
from click.testing import CliRunner

from corc.cli import cli
from corc.dag import (
    assign_levels,
    assign_rows,
    build_adjacency,
    compute_effective_status,
    compute_progress,
    render_ascii_dag,
    render_mermaid,
)
from corc.mutations import MutationLog
from corc.state import WorkState


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def state(tmp_path):
    ml = MutationLog(tmp_path / "mutations.jsonl")
    ws = WorkState(tmp_path / "state.db", ml)
    return ml, ws


def _make_task(tid, name, depends_on=None, status="pending"):
    """Build a minimal task dict for unit tests (no mutation log needed)."""
    return {
        "id": tid,
        "name": name,
        "depends_on": depends_on or [],
        "status": status,
        "done_when": "tests pass",
    }


# ── compute_effective_status ──────────────────────────────────────────────

class TestEffectiveStatus:
    def test_completed_stays_completed(self):
        tasks = [_make_task("a", "A", status="completed")]
        assert compute_effective_status(tasks) == {"a": "completed"}

    def test_running_stays_running(self):
        tasks = [_make_task("a", "A", status="running")]
        assert compute_effective_status(tasks) == {"a": "running"}

    def test_failed_stays_failed(self):
        tasks = [_make_task("a", "A", status="failed")]
        assert compute_effective_status(tasks) == {"a": "failed"}

    def test_pending_no_deps_is_ready(self):
        tasks = [_make_task("a", "A")]
        assert compute_effective_status(tasks)["a"] == "ready"

    def test_pending_deps_met_is_ready(self):
        tasks = [
            _make_task("a", "A", status="completed"),
            _make_task("b", "B", depends_on=["a"]),
        ]
        assert compute_effective_status(tasks)["b"] == "ready"

    def test_pending_deps_unmet_is_blocked(self):
        tasks = [
            _make_task("a", "A"),
            _make_task("b", "B", depends_on=["a"]),
        ]
        assert compute_effective_status(tasks)["b"] == "blocked"

    def test_deps_outside_task_set_are_ignored(self):
        # Dependency on a task not in our list → treated as met
        tasks = [_make_task("b", "B", depends_on=["unknown"])]
        assert compute_effective_status(tasks)["b"] == "ready"


# ── assign_levels ─────────────────────────────────────────────────────────

class TestAssignLevels:
    def test_single_task(self):
        tasks = [_make_task("a", "A")]
        assert assign_levels(tasks) == {"a": 0}

    def test_linear_chain(self):
        tasks = [
            _make_task("a", "A"),
            _make_task("b", "B", depends_on=["a"]),
            _make_task("c", "C", depends_on=["b"]),
        ]
        levels = assign_levels(tasks)
        assert levels == {"a": 0, "b": 1, "c": 2}

    def test_parallel_branches(self):
        tasks = [
            _make_task("a", "A"),
            _make_task("b", "B"),
            _make_task("c", "C", depends_on=["a", "b"]),
        ]
        levels = assign_levels(tasks)
        assert levels["a"] == 0
        assert levels["b"] == 0
        assert levels["c"] == 1

    def test_diamond(self):
        tasks = [
            _make_task("a", "A"),
            _make_task("b", "B", depends_on=["a"]),
            _make_task("c", "C", depends_on=["a"]),
            _make_task("d", "D", depends_on=["b", "c"]),
        ]
        levels = assign_levels(tasks)
        assert levels == {"a": 0, "b": 1, "c": 1, "d": 2}

    def test_deep_chain(self):
        tasks = [
            _make_task("a", "A"),
            _make_task("b", "B", depends_on=["a"]),
            _make_task("c", "C", depends_on=["b"]),
            _make_task("d", "D", depends_on=["c"]),
            _make_task("e", "E", depends_on=["d"]),
        ]
        levels = assign_levels(tasks)
        assert levels == {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}


# ── build_adjacency ──────────────────────────────────────────────────────

class TestBuildAdjacency:
    def test_forward_and_reverse(self):
        tasks = [
            _make_task("a", "A"),
            _make_task("b", "B", depends_on=["a"]),
        ]
        fwd, rev = build_adjacency(tasks)
        assert fwd == {"a": ["b"]}
        assert rev == {"b": ["a"]}

    def test_diamond_adjacency(self):
        tasks = [
            _make_task("a", "A"),
            _make_task("b", "B", depends_on=["a"]),
            _make_task("c", "C", depends_on=["a"]),
            _make_task("d", "D", depends_on=["b", "c"]),
        ]
        fwd, rev = build_adjacency(tasks)
        assert set(fwd["a"]) == {"b", "c"}
        assert set(rev["d"]) == {"b", "c"}


# ── assign_rows ───────────────────────────────────────────────────────────

class TestAssignRows:
    def test_linear_chain_same_row(self):
        tasks = [
            _make_task("a", "A"),
            _make_task("b", "B", depends_on=["a"]),
            _make_task("c", "C", depends_on=["b"]),
        ]
        levels = assign_levels(tasks)
        fwd, rev = build_adjacency(tasks)
        rows = assign_rows(tasks, levels, fwd, rev)
        # Linear chain → all on row 0
        assert rows["a"] == rows["b"] == rows["c"] == 0

    def test_parallel_branches_different_rows(self):
        tasks = [
            _make_task("a", "A"),
            _make_task("b", "B"),
            _make_task("c", "C", depends_on=["a", "b"]),
        ]
        levels = assign_levels(tasks)
        fwd, rev = build_adjacency(tasks)
        rows = assign_rows(tasks, levels, fwd, rev)
        # a and b should be on different rows
        assert rows["a"] != rows["b"]
        # c gets its own row (near average of a and b)
        assert rows["c"] in (rows["a"], rows["b"], 0, 1)

    def test_diamond_row_assignment(self):
        tasks = [
            _make_task("a", "A"),
            _make_task("b", "B", depends_on=["a"]),
            _make_task("c", "C", depends_on=["a"]),
            _make_task("d", "D", depends_on=["b", "c"]),
        ]
        levels = assign_levels(tasks)
        fwd, rev = build_adjacency(tasks)
        rows = assign_rows(tasks, levels, fwd, rev)
        # b and c at level 1 should be on different rows
        assert rows["b"] != rows["c"]
        # All rows should be contiguous starting from 0
        assert set(rows.values()) <= {0, 1}


# ── compute_progress ──────────────────────────────────────────────────────

class TestComputeProgress:
    def test_empty_tasks(self):
        p = compute_progress([])
        assert p["total"] == 0
        assert p["percent"] == 0

    def test_all_completed(self):
        tasks = [
            _make_task("a", "A", status="completed"),
            _make_task("b", "B", status="completed"),
        ]
        p = compute_progress(tasks)
        assert p["total"] == 2
        assert p["completed"] == 2
        assert p["percent"] == 100

    def test_mixed_statuses(self):
        tasks = [
            _make_task("a", "A", status="completed"),
            _make_task("b", "B", depends_on=["a"], status="running"),
            _make_task("c", "C", depends_on=["b"]),  # pending → blocked
            _make_task("d", "D"),                     # pending → ready
        ]
        p = compute_progress(tasks)
        assert p["total"] == 4
        assert p["completed"] == 1
        assert p["running"] == 1
        assert p["blocked"] == 1
        assert p["ready"] == 1
        assert p["percent"] == 25

    def test_failed_counted(self):
        tasks = [
            _make_task("a", "A", status="failed"),
            _make_task("b", "B", status="completed"),
        ]
        p = compute_progress(tasks)
        assert p["failed"] == 1
        assert p["completed"] == 1

    def test_cost_absent_when_no_data(self):
        tasks = [_make_task("a", "A", status="completed")]
        p = compute_progress(tasks)
        assert "cost_usd" not in p

    def test_cost_aggregated(self):
        tasks = [
            {**_make_task("a", "A", status="completed"), "cost_usd": 1.50},
            {**_make_task("b", "B", status="completed"), "cost_usd": 2.75},
            _make_task("c", "C"),  # no cost field
        ]
        p = compute_progress(tasks)
        assert p["cost_usd"] == pytest.approx(4.25)

    def test_cost_zero_still_reported(self):
        tasks = [{**_make_task("a", "A", status="completed"), "cost_usd": 0.0}]
        p = compute_progress(tasks)
        assert "cost_usd" in p
        assert p["cost_usd"] == 0.0


# ── render_ascii_dag ──────────────────────────────────────────────────────

class TestRenderAsciiDag:
    def test_empty_tasks(self):
        assert "No tasks" in render_ascii_dag([])

    def test_single_task(self):
        tasks = [_make_task("a", "Alpha")]
        out = render_ascii_dag(tasks, use_color=False)
        assert "Alpha" in out
        assert "⬚" in out  # ready
        assert "Progress: 0/1" in out

    def test_linear_chain_has_arrows(self):
        tasks = [
            _make_task("a", "First", status="completed"),
            _make_task("b", "Second", depends_on=["a"], status="running"),
            _make_task("c", "Third", depends_on=["b"]),
        ]
        out = render_ascii_dag(tasks, use_color=False)
        assert "First" in out
        assert "Second" in out
        assert "Third" in out
        assert "✅" in out
        assert "🔄" in out
        # Should have arrow characters connecting tasks
        assert "►" in out
        assert "─" in out
        assert "Progress: 1/3" in out

    def test_parallel_branches(self):
        tasks = [
            _make_task("a", "BranchA", status="completed"),
            _make_task("b", "BranchB", status="completed"),
            _make_task("c", "Merge", depends_on=["a", "b"]),
        ]
        out = render_ascii_dag(tasks, use_color=False)
        assert "BranchA" in out
        assert "BranchB" in out
        assert "Merge" in out
        assert "►" in out
        assert "Progress: 2/3" in out

    def test_diamond_dependency(self):
        tasks = [
            _make_task("a", "Root", status="completed"),
            _make_task("b", "Left", depends_on=["a"], status="completed"),
            _make_task("c", "Right", depends_on=["a"], status="running"),
            _make_task("d", "Sink", depends_on=["b", "c"]),
        ]
        out = render_ascii_dag(tasks, use_color=False)
        assert "Root" in out
        assert "Left" in out
        assert "Right" in out
        assert "Sink" in out
        # Arrow from Root to Left/Right, and from both to Sink
        assert out.count("►") >= 3  # at least 3 arrow heads (root→left, root→right, something→sink)
        assert "Progress: 2/4" in out

    def test_legend_present(self):
        tasks = [_make_task("a", "Task")]
        out = render_ascii_dag(tasks, use_color=False)
        assert "complete" in out
        assert "running" in out
        assert "ready" in out
        assert "blocked" in out
        assert "failed" in out

    def test_progress_bar_present(self):
        tasks = [
            _make_task("a", "Done", status="completed"),
            _make_task("b", "Todo"),
        ]
        out = render_ascii_dag(tasks, use_color=False)
        assert "█" in out or "░" in out
        assert "50%" in out

    def test_color_output_contains_ansi(self):
        tasks = [_make_task("a", "Task", status="completed")]
        out = render_ascii_dag(tasks, use_color=True)
        assert "\033[" in out  # ANSI escape present

    def test_no_color_output_clean(self):
        tasks = [_make_task("a", "Task", status="completed")]
        out = render_ascii_dag(tasks, use_color=False)
        assert "\033[" not in out

    def test_wide_dag_renders(self):
        """Many levels deep — should not crash."""
        tasks = []
        for i in range(10):
            deps = [f"t{i-1}"] if i > 0 else []
            tasks.append(_make_task(f"t{i}", f"Task{i}", depends_on=deps, status="completed"))
        out = render_ascii_dag(tasks, use_color=False)
        assert "Task0" in out
        assert "Task9" in out
        assert "100%" in out

    def test_disconnected_components(self):
        """Two independent chains rendered without errors."""
        tasks = [
            _make_task("a1", "ChainA1", status="completed"),
            _make_task("a2", "ChainA2", depends_on=["a1"]),
            _make_task("b1", "ChainB1", status="completed"),
            _make_task("b2", "ChainB2", depends_on=["b1"]),
        ]
        out = render_ascii_dag(tasks, use_color=False)
        assert "ChainA1" in out
        assert "ChainA2" in out
        assert "ChainB1" in out
        assert "ChainB2" in out

    def test_all_statuses_shown(self):
        """Every status type renders its icon."""
        tasks = [
            _make_task("a", "Done", status="completed"),
            _make_task("b", "Active", status="running"),
            _make_task("c", "Broke", status="failed"),
            _make_task("d", "Waiting"),  # pending, no deps → ready
            _make_task("e", "Stuck", depends_on=["b"]),  # pending, dep running → blocked
        ]
        out = render_ascii_dag(tasks, use_color=False)
        assert "✅" in out
        assert "🔄" in out
        assert "❌" in out
        assert "⬚" in out
        assert "◻" in out

    def test_cost_shown_when_available(self):
        """Cost summary appears when tasks carry cost_usd."""
        tasks = [
            {**_make_task("a", "Done", status="completed"), "cost_usd": 2.50},
            {**_make_task("b", "Todo", depends_on=["a"]), "cost_usd": 1.00},
        ]
        out = render_ascii_dag(tasks, use_color=False)
        assert "Cost: $3.50" in out

    def test_cost_hidden_when_absent(self):
        """No cost line when tasks have no cost data."""
        tasks = [_make_task("a", "Task")]
        out = render_ascii_dag(tasks, use_color=False)
        assert "Cost:" not in out


# ── render_mermaid ────────────────────────────────────────────────────────

class TestRenderMermaid:
    def test_empty_tasks(self):
        out = render_mermaid([])
        assert "graph LR" in out
        assert "No tasks" in out

    def test_single_task(self):
        tasks = [_make_task("a", "Alpha")]
        out = render_mermaid(tasks)
        assert "graph LR" in out
        assert "Alpha" in out

    def test_linear_chain_edges(self):
        tasks = [
            _make_task("a", "First"),
            _make_task("b", "Second", depends_on=["a"]),
            _make_task("c", "Third", depends_on=["b"]),
        ]
        out = render_mermaid(tasks)
        assert "a --> b" in out
        assert "b --> c" in out

    def test_diamond_edges(self):
        tasks = [
            _make_task("a", "Root"),
            _make_task("b", "Left", depends_on=["a"]),
            _make_task("c", "Right", depends_on=["a"]),
            _make_task("d", "Sink", depends_on=["b", "c"]),
        ]
        out = render_mermaid(tasks)
        assert "a --> b" in out
        assert "a --> c" in out
        assert "b --> d" in out
        assert "c --> d" in out

    def test_status_icons_in_labels(self):
        tasks = [
            _make_task("a", "Done", status="completed"),
            _make_task("b", "Active", status="running"),
        ]
        out = render_mermaid(tasks)
        assert "✅" in out
        assert "🔄" in out

    def test_style_directives_present(self):
        tasks = [
            _make_task("a", "Done", status="completed"),
            _make_task("b", "Active", status="running"),
        ]
        out = render_mermaid(tasks)
        assert "style a" in out
        assert "fill:#22c55e" in out  # green for completed

    def test_progress_comment(self):
        tasks = [
            _make_task("a", "Done", status="completed"),
            _make_task("b", "Todo"),
        ]
        out = render_mermaid(tasks)
        assert "%% Progress:" in out
        assert "1/2" in out

    def test_hyphenated_ids_sanitised(self):
        tasks = [
            _make_task("my-task", "My Task"),
            _make_task("other-task", "Other", depends_on=["my-task"]),
        ]
        out = render_mermaid(tasks)
        # Hyphens should be replaced with underscores for Mermaid
        assert "my_task" in out
        assert "other_task" in out
        assert "my_task --> other_task" in out

    def test_mermaid_cost_in_comment(self):
        tasks = [
            {**_make_task("a", "Done", status="completed"), "cost_usd": 4.23},
        ]
        out = render_mermaid(tasks)
        assert "Cost: $4.23" in out

    def test_mermaid_no_cost_when_absent(self):
        tasks = [_make_task("a", "Alpha")]
        out = render_mermaid(tasks)
        assert "Cost:" not in out

    def test_parallel_branches_mermaid(self):
        tasks = [
            _make_task("a", "BranchA", status="completed"),
            _make_task("b", "BranchB", status="completed"),
            _make_task("c", "Merge", depends_on=["a", "b"]),
        ]
        out = render_mermaid(tasks)
        assert "a --> c" in out
        assert "b --> c" in out
        assert "2/3" in out


# ── CLI integration ───────────────────────────────────────────────────────

class TestDagCLI:
    @pytest.fixture
    def cli_state(self, tmp_path, monkeypatch):
        """Set up a temp project with tasks for CLI testing."""
        # Create minimal project structure
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "events").mkdir()
        (tmp_path / "data" / "sessions").mkdir()
        (tmp_path / "knowledge").mkdir()
        (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")

        monkeypatch.setattr("corc.config.get_project_root", lambda: tmp_path)

        ml = MutationLog(tmp_path / "data" / "mutations.jsonl")
        # Create a small DAG: A → B → C
        ml.append("task_created", {
            "id": "t1", "name": "setup", "done_when": "done",
            "depends_on": [],
        }, reason="test")
        ml.append("task_created", {
            "id": "t2", "name": "build", "done_when": "done",
            "depends_on": ["t1"],
        }, reason="test")
        ml.append("task_created", {
            "id": "t3", "name": "test", "done_when": "done",
            "depends_on": ["t2"],
        }, reason="test")
        # Complete first task
        ml.append("task_completed", {}, reason="test", task_id="t1")
        # Start second task
        ml.append("task_started", {"attempt": 1}, reason="test", task_id="t2")

        return tmp_path

    def test_dag_command_runs(self, cli_state):
        runner = CliRunner()
        result = runner.invoke(cli, ["dag", "--no-color"])
        assert result.exit_code == 0
        assert "setup" in result.output
        assert "build" in result.output
        assert "test" in result.output
        assert "Progress:" in result.output

    def test_dag_mermaid_flag(self, cli_state):
        runner = CliRunner()
        result = runner.invoke(cli, ["dag", "--mermaid"])
        assert result.exit_code == 0
        assert "graph LR" in result.output
        assert "t1 --> t2" in result.output
        assert "t2 --> t3" in result.output

    def test_dag_no_tasks(self, tmp_path, monkeypatch):
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "events").mkdir()
        (tmp_path / "data" / "sessions").mkdir()
        (tmp_path / "knowledge").mkdir()
        (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
        monkeypatch.setattr("corc.config.get_project_root", lambda: tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["dag"])
        assert result.exit_code == 0
        assert "No tasks" in result.output

    def test_dag_diamond_via_cli(self, tmp_path, monkeypatch):
        """Full diamond DAG through the CLI."""
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "events").mkdir()
        (tmp_path / "data" / "sessions").mkdir()
        (tmp_path / "knowledge").mkdir()
        (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
        monkeypatch.setattr("corc.config.get_project_root", lambda: tmp_path)

        ml = MutationLog(tmp_path / "data" / "mutations.jsonl")
        ml.append("task_created", {
            "id": "root", "name": "root-task", "done_when": "d",
            "depends_on": [],
        }, reason="test")
        ml.append("task_created", {
            "id": "left", "name": "left-branch", "done_when": "d",
            "depends_on": ["root"],
        }, reason="test")
        ml.append("task_created", {
            "id": "right", "name": "right-branch", "done_when": "d",
            "depends_on": ["root"],
        }, reason="test")
        ml.append("task_created", {
            "id": "sink", "name": "final-merge", "done_when": "d",
            "depends_on": ["left", "right"],
        }, reason="test")
        ml.append("task_completed", {}, reason="test", task_id="root")

        runner = CliRunner()
        result = runner.invoke(cli, ["dag", "--no-color"])
        assert result.exit_code == 0
        assert "root-task" in result.output
        assert "left-branch" in result.output
        assert "right-branch" in result.output
        assert "final-merge" in result.output
        assert "1/4" in result.output  # 1 completed out of 4
