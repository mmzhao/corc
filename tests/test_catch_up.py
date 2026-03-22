"""Tests for catch-up summary injection in context assembly."""

from pathlib import Path

from corc.context import assemble_context, generate_catch_up_summary


# ---------------------------------------------------------------------------
# Helpers to build test fixtures
# ---------------------------------------------------------------------------


def _make_plan_tasks():
    """Create a multi-task plan with dependencies.

    Plan structure:
        task-a (completed) ──┐
                              ├──> task-c (pending, depends on a + b)
        task-b (completed) ──┘
        task-d (running, no deps)
        task-e (pending, depends on c)
    """
    return [
        {
            "id": "task-a",
            "name": "implement-fts5-search",
            "status": "completed",
            "depends_on": [],
        },
        {
            "id": "task-b",
            "name": "implement-mutation-log",
            "status": "completed",
            "depends_on": [],
        },
        {
            "id": "task-c",
            "name": "implement-hybrid-search",
            "status": "pending",
            "depends_on": ["task-a", "task-b"],
        },
        {
            "id": "task-d",
            "name": "implement-dag-viz",
            "status": "running",
            "depends_on": [],
        },
        {
            "id": "task-e",
            "name": "implement-context-assembly",
            "status": "pending",
            "depends_on": ["task-c"],
        },
    ]


def _make_mutations():
    """Create mutation history for the plan tasks."""
    return [
        {
            "seq": 1,
            "ts": "2026-03-20T10:00:00Z",
            "type": "task_created",
            "data": {"id": "task-a", "name": "implement-fts5-search", "done_when": "FTS5 works"},
            "reason": "Plan created",
            "task_id": "task-a",
        },
        {
            "seq": 2,
            "ts": "2026-03-20T10:00:01Z",
            "type": "task_created",
            "data": {"id": "task-b", "name": "implement-mutation-log", "done_when": "Mutations work"},
            "reason": "Plan created",
            "task_id": "task-b",
        },
        {
            "seq": 3,
            "ts": "2026-03-20T10:00:02Z",
            "type": "task_created",
            "data": {"id": "task-c", "name": "implement-hybrid-search", "done_when": "Hybrid search works"},
            "reason": "Plan created",
            "task_id": "task-c",
        },
        {
            "seq": 4,
            "ts": "2026-03-20T12:00:00Z",
            "type": "task_started",
            "data": {},
            "reason": "Agent dispatched",
            "task_id": "task-a",
        },
        {
            "seq": 5,
            "ts": "2026-03-20T14:00:00Z",
            "type": "task_completed",
            "data": {
                "pr_url": "https://github.com/org/repo/pull/12",
                "findings": [
                    "FTS5 tokenizer needs 'porter' for stemming",
                    "unicode61 tokenizer handles CJK better",
                ],
            },
            "reason": "Agent completed task",
            "task_id": "task-a",
        },
        {
            "seq": 6,
            "ts": "2026-03-21T09:00:00Z",
            "type": "task_started",
            "data": {},
            "reason": "Agent dispatched",
            "task_id": "task-b",
        },
        {
            "seq": 7,
            "ts": "2026-03-21T11:00:00Z",
            "type": "task_completed",
            "data": {
                "findings": [{"content": "flock-based append is sufficient for single-machine"}],
            },
            "reason": "Agent completed task",
            "task_id": "task-b",
        },
        {
            "seq": 8,
            "ts": "2026-03-21T12:00:00Z",
            "type": "task_started",
            "data": {},
            "reason": "Agent dispatched",
            "task_id": "task-d",
        },
    ]


def _make_target_task():
    """The task being dispatched (task-c), which depends on a and b."""
    return {
        "id": "task-c",
        "name": "implement-hybrid-search",
        "description": "Combine FTS5 and vector search",
        "done_when": "Hybrid search returns ranked results",
        "depends_on": ["task-a", "task-b"],
        "context_bundle": [],
    }


# ---------------------------------------------------------------------------
# generate_catch_up_summary unit tests
# ---------------------------------------------------------------------------


class TestGenerateCatchUpSummary:
    def test_completed_deps_shown(self):
        """Completed dependency tasks appear in catch-up summary."""
        task = _make_target_task()
        mutations = _make_mutations()
        plan_tasks = _make_plan_tasks()

        summary = generate_catch_up_summary(task, mutations, plan_tasks)
        assert summary is not None
        assert "CATCH-UP SUMMARY" in summary
        assert '"implement-fts5-search" was completed' in summary
        assert '"implement-mutation-log" was completed' in summary

    def test_pr_url_included(self):
        """PR URL is shown for completed deps that have one."""
        task = _make_target_task()
        mutations = _make_mutations()
        plan_tasks = _make_plan_tasks()

        summary = generate_catch_up_summary(task, mutations, plan_tasks)
        assert "PR https://github.com/org/repo/pull/12" in summary

    def test_string_findings_from_deps(self):
        """String findings from completed deps are reported."""
        task = _make_target_task()
        mutations = _make_mutations()
        plan_tasks = _make_plan_tasks()

        summary = generate_catch_up_summary(task, mutations, plan_tasks)
        assert "FTS5 tokenizer needs 'porter' for stemming" in summary
        assert "unicode61 tokenizer handles CJK better" in summary

    def test_dict_findings_from_deps(self):
        """Dict findings (with 'content' key) from completed deps are reported."""
        task = _make_target_task()
        mutations = _make_mutations()
        plan_tasks = _make_plan_tasks()

        summary = generate_catch_up_summary(task, mutations, plan_tasks)
        assert "flock-based append is sufficient for single-machine" in summary

    def test_overall_progress(self):
        """Progress line shows completed and remaining counts."""
        task = _make_target_task()
        mutations = _make_mutations()
        plan_tasks = _make_plan_tasks()

        summary = generate_catch_up_summary(task, mutations, plan_tasks)
        assert "2 tasks completed, 3 remaining" in summary

    def test_no_deps_no_plan_returns_none(self):
        """No summary when task has no deps and no plan tasks."""
        task = {"name": "solo", "done_when": "done", "depends_on": []}
        summary = generate_catch_up_summary(task, [], [])
        assert summary is None

    def test_dep_not_yet_completed(self):
        """Pending/running deps show their current status."""
        task = {
            "name": "blocked-task",
            "done_when": "done",
            "depends_on": ["task-d"],
        }
        mutations = _make_mutations()
        plan_tasks = _make_plan_tasks()

        summary = generate_catch_up_summary(task, mutations, plan_tasks)
        assert summary is not None
        assert '"implement-dag-viz" is running' in summary

    def test_deps_as_json_string(self):
        """depends_on stored as a JSON string (from SQLite) is handled."""
        task = {
            "name": "test",
            "done_when": "done",
            "depends_on": '["task-a"]',
        }
        mutations = _make_mutations()
        plan_tasks = _make_plan_tasks()

        summary = generate_catch_up_summary(task, mutations, plan_tasks)
        assert '"implement-fts5-search" was completed' in summary

    def test_unknown_dep_id_uses_id_as_name(self):
        """Dep not in plan_tasks uses the task_id as fallback name."""
        task = {
            "name": "test",
            "done_when": "done",
            "depends_on": ["unknown-id"],
        }
        summary = generate_catch_up_summary(task, [], [{"id": "x", "status": "pending"}])
        assert summary is not None
        assert '"unknown-id"' in summary

    def test_summary_format_markers(self):
        """Summary has correct start/end markers."""
        task = _make_target_task()
        mutations = _make_mutations()
        plan_tasks = _make_plan_tasks()

        summary = generate_catch_up_summary(task, mutations, plan_tasks)
        assert summary.startswith("=== CATCH-UP SUMMARY ===")
        assert summary.endswith("=== END CATCH-UP ===")
        assert "Since your last context:" in summary

    def test_no_deps_but_has_plan_tasks(self):
        """Task with no deps but plan_tasks still shows progress."""
        task = {
            "name": "independent",
            "done_when": "done",
            "depends_on": [],
        }
        plan_tasks = _make_plan_tasks()
        summary = generate_catch_up_summary(task, [], plan_tasks)
        assert summary is not None
        assert "2 tasks completed, 3 remaining" in summary

    def test_finding_with_description_key(self):
        """Dict findings with 'description' key (no 'content') are handled."""
        task = {
            "name": "test",
            "done_when": "done",
            "depends_on": ["task-x"],
        }
        mutations = [
            {
                "seq": 1,
                "ts": "2026-03-20T10:00:00Z",
                "type": "task_completed",
                "data": {
                    "findings": [{"description": "Important discovery here"}],
                },
                "reason": "done",
                "task_id": "task-x",
            }
        ]
        plan_tasks = [{"id": "task-x", "name": "research-task", "status": "completed"}]

        summary = generate_catch_up_summary(task, mutations, plan_tasks)
        assert "Important discovery here" in summary


# ---------------------------------------------------------------------------
# assemble_context integration tests for catch-up injection
# ---------------------------------------------------------------------------


class TestCatchUpInAssembledContext:
    def test_catch_up_injected_before_context_bundle(self, tmp_path):
        """Catch-up summary appears between task definition and context bundle."""
        (tmp_path / "doc.md").write_text("# Reference\nSome reference content.")
        task = _make_target_task()
        task["context_bundle"] = ["doc.md"]
        mutations = _make_mutations()
        plan_tasks = _make_plan_tasks()

        ctx = assemble_context(
            task, tmp_path, mutations=mutations, plan_tasks=plan_tasks
        )

        # All sections present
        assert "=== TASK DEFINITION ===" in ctx
        assert "=== CATCH-UP SUMMARY ===" in ctx
        assert "=== CONTEXT: doc.md ===" in ctx

        # Verify ordering: task def < catch-up < context bundle
        task_end = ctx.index("=== END TASK DEFINITION ===")
        catchup_start = ctx.index("=== CATCH-UP SUMMARY ===")
        bundle_start = ctx.index("=== CONTEXT: doc.md ===")
        assert task_end < catchup_start < bundle_start

    def test_catch_up_before_blacklist(self, tmp_path):
        """Catch-up summary appears before the blacklist."""
        corc_dir = tmp_path / ".corc"
        corc_dir.mkdir()
        (corc_dir / "blacklist.md").write_text("- No eval()")

        task = _make_target_task()
        mutations = _make_mutations()
        plan_tasks = _make_plan_tasks()

        ctx = assemble_context(
            task, tmp_path, mutations=mutations, plan_tasks=plan_tasks
        )

        catchup_end = ctx.index("=== END CATCH-UP ===")
        blacklist_start = ctx.index("=== AGENT BLACKLIST ===")
        assert catchup_end < blacklist_start

    def test_no_catch_up_without_mutations(self, tmp_path):
        """No catch-up section when mutations/plan_tasks not provided."""
        task = _make_target_task()
        ctx = assemble_context(task, tmp_path)
        assert "CATCH-UP SUMMARY" not in ctx

    def test_no_catch_up_when_nothing_to_report(self, tmp_path):
        """No catch-up section when task has no deps and no plan tasks."""
        task = {
            "name": "solo",
            "done_when": "done",
            "depends_on": [],
            "context_bundle": [],
        }
        ctx = assemble_context(task, tmp_path, mutations=[], plan_tasks=[])
        assert "CATCH-UP SUMMARY" not in ctx

    def test_catch_up_contains_completed_deps(self, tmp_path):
        """Assembled context includes completed dependency info."""
        task = _make_target_task()
        mutations = _make_mutations()
        plan_tasks = _make_plan_tasks()

        ctx = assemble_context(
            task, tmp_path, mutations=mutations, plan_tasks=plan_tasks
        )

        assert '"implement-fts5-search" was completed' in ctx
        assert '"implement-mutation-log" was completed' in ctx

    def test_catch_up_contains_findings(self, tmp_path):
        """Assembled context includes findings from completed deps."""
        task = _make_target_task()
        mutations = _make_mutations()
        plan_tasks = _make_plan_tasks()

        ctx = assemble_context(
            task, tmp_path, mutations=mutations, plan_tasks=plan_tasks
        )

        assert "FTS5 tokenizer needs 'porter' for stemming" in ctx
        assert "flock-based append is sufficient for single-machine" in ctx

    def test_catch_up_contains_progress(self, tmp_path):
        """Assembled context includes overall progress line."""
        task = _make_target_task()
        mutations = _make_mutations()
        plan_tasks = _make_plan_tasks()

        ctx = assemble_context(
            task, tmp_path, mutations=mutations, plan_tasks=plan_tasks
        )

        assert "2 tasks completed, 3 remaining" in ctx

    def test_backward_compatible_without_kwargs(self, tmp_path):
        """Existing calls without mutations/plan_tasks still work."""
        (tmp_path / "doc.md").write_text("# Doc\nContent.")
        task = {
            "name": "test",
            "done_when": "done",
            "context_bundle": ["doc.md"],
        }
        ctx = assemble_context(task, tmp_path)
        assert "TASK DEFINITION" in ctx
        assert "Content." in ctx
        assert "CATCH-UP" not in ctx

    def test_full_context_ordering(self, tmp_path):
        """Full integration: task def → catch-up → context → blacklist."""
        corc_dir = tmp_path / ".corc"
        corc_dir.mkdir()
        (corc_dir / "blacklist.md").write_text("- No eval()")
        (tmp_path / "ref.md").write_text("# Ref\nReference doc.")

        task = _make_target_task()
        task["context_bundle"] = ["ref.md"]
        mutations = _make_mutations()
        plan_tasks = _make_plan_tasks()

        ctx = assemble_context(
            task, tmp_path, mutations=mutations, plan_tasks=plan_tasks
        )

        # Verify all sections present and in order
        positions = [
            ctx.index("=== TASK DEFINITION ==="),
            ctx.index("=== END TASK DEFINITION ==="),
            ctx.index("=== CATCH-UP SUMMARY ==="),
            ctx.index("=== END CATCH-UP ==="),
            ctx.index("=== CONTEXT: ref.md ==="),
            ctx.index("=== END CONTEXT ==="),
            ctx.index("=== AGENT BLACKLIST ==="),
            ctx.index("=== END AGENT BLACKLIST ==="),
        ]
        assert positions == sorted(positions), "Sections are not in expected order"

    def test_multi_task_plan_some_completed(self, tmp_path):
        """End-to-end: multi-task plan where some tasks are completed.

        Simulates dispatching task-c after task-a and task-b have completed.
        Verifies the catch-up gives task-c's agent full situational awareness.
        """
        task = _make_target_task()
        mutations = _make_mutations()
        plan_tasks = _make_plan_tasks()

        ctx = assemble_context(
            task, tmp_path, mutations=mutations, plan_tasks=plan_tasks
        )

        # Agent should know:
        # 1. Its deps are done
        assert '"implement-fts5-search" was completed' in ctx
        assert '"implement-mutation-log" was completed' in ctx

        # 2. Key findings from deps
        assert "FTS5 tokenizer needs 'porter' for stemming" in ctx
        assert "flock-based append is sufficient for single-machine" in ctx

        # 3. PR reference
        assert "PR https://github.com/org/repo/pull/12" in ctx

        # 4. Overall plan progress
        assert "2 tasks completed, 3 remaining" in ctx

        # 5. Task definition is still present
        assert "implement-hybrid-search" in ctx
        assert "Hybrid search returns ranked results" in ctx
