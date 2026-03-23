"""Tests for interactive planning sessions (corc plan)."""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from corc.mutations import MutationLog
from corc.state import WorkState
from corc.knowledge import KnowledgeStore
from corc.plan import (
    PLANNER_ROLE,
    build_system_prompt,
    save_session_metadata,
    mark_session_complete,
    load_latest_draft,
    get_drafts_dir,
    launch_interactive_claude,
    _get_knowledge_summary,
    _get_work_state_summary,
    _get_repo_context,
    _check_prompt_size,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def corc_env(tmp_path):
    """Set up a minimal CORC environment for testing."""
    # Create directories
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    corc_dir = tmp_path / ".corc"
    corc_dir.mkdir()
    (tmp_path / "src" / "corc").mkdir(parents=True)
    (tmp_path / "src" / "corc" / "__init__.py").write_text("")
    (tmp_path / "src" / "corc" / "cli.py").write_text("# cli")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_plan.py").write_text("# test")

    ml = MutationLog(data_dir / "mutations.jsonl")
    ws = WorkState(data_dir / "state.db", ml)
    ks = KnowledgeStore(knowledge_dir, data_dir / "knowledge.db")

    paths = {
        "root": tmp_path,
        "mutations": data_dir / "mutations.jsonl",
        "state_db": data_dir / "state.db",
        "events_dir": data_dir / "events",
        "sessions_dir": data_dir / "sessions",
        "knowledge_dir": knowledge_dir,
        "knowledge_db": data_dir / "knowledge.db",
        "corc_dir": corc_dir,
    }

    return paths, ml, ws, ks


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    """Tests for build_system_prompt and its context helpers."""

    def test_includes_planner_role(self, corc_env):
        paths, ml, ws, ks = corc_env
        prompt = build_system_prompt(paths, ws, ks)
        assert "CORC Planning Agent" in prompt
        assert "corc task create" in prompt

    def test_includes_spec_template(self, corc_env):
        paths, ml, ws, ks = corc_env
        prompt = build_system_prompt(paths, ws, ks)
        assert "## Problem" in prompt
        assert "## Requirements" in prompt
        assert "## Testing Strategy" in prompt

    def test_includes_quick_task_detection(self, corc_env):
        paths, ml, ws, ks = corc_env
        prompt = build_system_prompt(paths, ws, ks)
        assert "Quick task" in prompt
        assert "Standard task" in prompt
        assert "Epic" in prompt

    def test_includes_knowledge_summary_empty(self, corc_env):
        paths, ml, ws, ks = corc_env
        prompt = build_system_prompt(paths, ws, ks)
        assert "Knowledge Store" in prompt
        assert "empty" in prompt

    def test_includes_knowledge_summary_with_docs(self, corc_env):
        paths, ml, ws, ks = corc_env
        # Add a document to the knowledge store
        doc_content = (
            "---\nid: doc1\ntype: decision\n---\n# Test Decision\nWe decided X."
        )
        doc_path = paths["knowledge_dir"] / "test-decision.md"
        doc_path.write_text(doc_content)
        ks.add(file_path=doc_path, doc_type="decision")

        prompt = build_system_prompt(paths, ws, ks)
        assert "Test Decision" in prompt
        assert "doc1" in prompt

    def test_includes_work_state_summary_empty(self, corc_env):
        paths, ml, ws, ks = corc_env
        prompt = build_system_prompt(paths, ws, ks)
        assert "Work State" in prompt
        assert "No tasks" in prompt

    def test_includes_work_state_summary_with_tasks(self, corc_env):
        paths, ml, ws, ks = corc_env
        ml.append(
            "task_created",
            {
                "id": "t1",
                "name": "build-feature",
                "done_when": "tests pass and file exists",
                "role": "implementer",
            },
            reason="test",
        )
        ml.append(
            "task_created",
            {
                "id": "t2",
                "name": "review-feature",
                "done_when": "review posted",
                "role": "reviewer",
                "depends_on": ["t1"],
            },
            reason="test",
        )
        ws.refresh()

        prompt = build_system_prompt(paths, ws, ks)
        assert "build-feature" in prompt
        assert "review-feature" in prompt
        assert "t1" in prompt
        assert "t2" in prompt

    def test_includes_repo_context(self, corc_env):
        paths, ml, ws, ks = corc_env
        prompt = build_system_prompt(paths, ws, ks)
        assert "Repository Context" in prompt

    def test_includes_source_files_in_repo_context(self, corc_env):
        paths, ml, ws, ks = corc_env
        prompt = build_system_prompt(paths, ws, ks)
        # Our fixture creates src/corc/cli.py and tests/test_plan.py
        assert "Source files" in prompt
        assert "cli.py" in prompt

    def test_includes_draft_auto_save_dir(self, corc_env):
        paths, ml, ws, ks = corc_env
        prompt = build_system_prompt(paths, ws, ks)
        assert "Draft Auto-Save" in prompt
        assert "drafts" in prompt

    def test_seed_content_included(self, corc_env):
        paths, ml, ws, ks = corc_env
        seed = "# My Idea\n\nBuild a widget that does X."
        prompt = build_system_prompt(paths, ws, ks, seed_content=seed)
        assert "Seed Document" in prompt
        assert "Build a widget that does X" in prompt

    def test_seed_content_absent_when_none(self, corc_env):
        paths, ml, ws, ks = corc_env
        prompt = build_system_prompt(paths, ws, ks, seed_content=None)
        assert "Seed Document" not in prompt

    def test_draft_content_included_on_resume(self, corc_env):
        paths, ml, ws, ks = corc_env
        draft = "# Draft Spec\n\n## Problem\nThing is broken."
        prompt = build_system_prompt(
            paths,
            ws,
            ks,
            draft_content=draft,
            resume_meta={"timestamp": "2026-01-01T00:00:00Z"},
        )
        assert "Previous Draft" in prompt
        assert "Thing is broken" in prompt
        assert "Resuming" in prompt
        assert "2026-01-01" in prompt

    def test_draft_absent_when_not_resuming(self, corc_env):
        paths, ml, ws, ks = corc_env
        prompt = build_system_prompt(paths, ws, ks)
        assert "Previous Draft" not in prompt


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------


class TestKnowledgeSummary:
    def test_empty_store(self, corc_env):
        _, _, _, ks = corc_env
        summary = _get_knowledge_summary(ks)
        assert "empty" in summary

    def test_groups_by_type(self, corc_env):
        paths, _, _, ks = corc_env
        for i in range(3):
            content = f"---\nid: d{i}\ntype: decision\n---\n# Decision {i}\nContent."
            p = paths["knowledge_dir"] / f"d{i}.md"
            p.write_text(content)
            ks.add(file_path=p, doc_type="decision")

        content = "---\nid: r1\ntype: research\n---\n# Research 1\nContent."
        p = paths["knowledge_dir"] / "r1.md"
        p.write_text(content)
        ks.add(file_path=p, doc_type="research")

        summary = _get_knowledge_summary(ks)
        assert "decision (3)" in summary
        assert "research (1)" in summary


class TestWorkStateSummary:
    def test_empty_state(self, corc_env):
        _, _, ws, _ = corc_env
        summary = _get_work_state_summary(ws)
        assert "No tasks" in summary

    def test_shows_tasks_by_status(self, corc_env):
        _, ml, ws, _ = corc_env
        ml.append(
            "task_created",
            {
                "id": "t1",
                "name": "task-a",
                "done_when": "file exists",
            },
            reason="test",
        )
        ml.append(
            "task_created",
            {
                "id": "t2",
                "name": "task-b",
                "done_when": "tests pass",
            },
            reason="test",
        )
        ml.append("task_completed", {}, reason="test", task_id="t1")
        ws.refresh()

        summary = _get_work_state_summary(ws)
        assert "task-a" in summary
        assert "task-b" in summary
        assert "completed" in summary
        assert "pending" in summary

    def test_shows_ready_tasks(self, corc_env):
        _, ml, ws, _ = corc_env
        ml.append(
            "task_created",
            {
                "id": "t1",
                "name": "ready-task",
                "done_when": "done",
            },
            reason="test",
        )
        ws.refresh()

        summary = _get_work_state_summary(ws)
        assert "Ready to dispatch" in summary
        assert "ready-task" in summary


class TestRepoContext:
    def test_lists_source_files(self, corc_env):
        paths, _, _, _ = corc_env
        ctx = _get_repo_context(paths["root"])
        assert "Source files" in ctx
        assert "cli.py" in ctx

    def test_lists_test_files(self, corc_env):
        paths, _, _, _ = corc_env
        ctx = _get_repo_context(paths["root"])
        assert "Test files" in ctx
        assert "test_plan.py" in ctx

    def test_no_crash_on_missing_dirs(self, tmp_path):
        """If src/ and tests/ don't exist, still returns something."""
        ctx = _get_repo_context(tmp_path)
        # Should not crash; may return empty or git info only
        assert isinstance(ctx, str)


# ---------------------------------------------------------------------------
# Draft / session management
# ---------------------------------------------------------------------------


class TestDraftManagement:
    def test_get_drafts_dir_creates_directory(self, tmp_path):
        corc_dir = tmp_path / ".corc"
        corc_dir.mkdir()
        drafts = get_drafts_dir(corc_dir)
        assert drafts.exists()
        assert drafts == corc_dir / "drafts"

    def test_save_session_metadata(self, tmp_path):
        corc_dir = tmp_path / ".corc"
        corc_dir.mkdir()
        meta_path = save_session_metadata(
            corc_dir, "20260101-120000", seed_file="idea.md"
        )
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["session_id"] == "20260101-120000"
        assert meta["seed_file"] == "idea.md"
        assert meta["status"] == "active"
        assert "timestamp" in meta

    def test_save_session_metadata_stores_claude_session_id(self, tmp_path):
        """Session metadata includes claude_session_id for precise resume."""
        corc_dir = tmp_path / ".corc"
        corc_dir.mkdir()
        test_uuid = "12345678-1234-1234-1234-123456789abc"
        meta_path = save_session_metadata(
            corc_dir, "sess1", claude_session_id=test_uuid
        )
        meta = json.loads(meta_path.read_text())
        assert meta["claude_session_id"] == test_uuid

    def test_save_session_metadata_claude_session_id_defaults_none(self, tmp_path):
        """Without claude_session_id, the field is null."""
        corc_dir = tmp_path / ".corc"
        corc_dir.mkdir()
        meta_path = save_session_metadata(corc_dir, "sess1")
        meta = json.loads(meta_path.read_text())
        assert meta["claude_session_id"] is None

    def test_mark_session_complete(self, tmp_path):
        corc_dir = tmp_path / ".corc"
        corc_dir.mkdir()
        save_session_metadata(corc_dir, "sess1")
        mark_session_complete(corc_dir, "sess1")

        meta_path = corc_dir / "drafts" / "session-sess1.json"
        meta = json.loads(meta_path.read_text())
        assert meta["status"] == "complete"
        assert "completed" in meta

    def test_load_latest_draft_no_sessions(self, tmp_path):
        corc_dir = tmp_path / ".corc"
        corc_dir.mkdir()
        meta, content = load_latest_draft(corc_dir)
        assert meta is None
        assert content is None

    def test_load_latest_draft_with_session(self, tmp_path):
        corc_dir = tmp_path / ".corc"
        corc_dir.mkdir()
        save_session_metadata(corc_dir, "sess1", seed_file="idea.md")

        meta, content = load_latest_draft(corc_dir)
        assert meta is not None
        assert meta["session_id"] == "sess1"
        assert content is None  # No draft file yet

    def test_load_latest_draft_with_draft_file(self, tmp_path):
        corc_dir = tmp_path / ".corc"
        corc_dir.mkdir()
        save_session_metadata(corc_dir, "sess1")

        # Create a draft spec file
        drafts = get_drafts_dir(corc_dir)
        draft_path = drafts / "plan-my-feature.md"
        draft_path.write_text("# My Feature\n\n## Problem\nThings are slow.")

        meta, content = load_latest_draft(corc_dir)
        assert meta is not None
        assert content is not None
        assert "Things are slow" in content

    def test_load_latest_draft_picks_newest(self, tmp_path):
        corc_dir = tmp_path / ".corc"
        corc_dir.mkdir()
        save_session_metadata(corc_dir, "sess1")
        time.sleep(0.05)
        save_session_metadata(corc_dir, "sess2")

        meta, _ = load_latest_draft(corc_dir)
        assert meta["session_id"] == "sess2"

    def test_load_latest_draft_skips_completed_sessions(self, tmp_path):
        """Resume should skip completed sessions and find active ones."""
        corc_dir = tmp_path / ".corc"
        corc_dir.mkdir()

        # Create an active session, then a completed one
        save_session_metadata(corc_dir, "active-sess")
        time.sleep(0.05)
        save_session_metadata(corc_dir, "completed-sess")
        mark_session_complete(corc_dir, "completed-sess")

        meta, _ = load_latest_draft(corc_dir)
        assert meta is not None
        assert meta["session_id"] == "active-sess"

    def test_load_latest_draft_returns_none_when_all_completed(self, tmp_path):
        """If all sessions are completed, resume returns None."""
        corc_dir = tmp_path / ".corc"
        corc_dir.mkdir()

        save_session_metadata(corc_dir, "sess1")
        mark_session_complete(corc_dir, "sess1")
        time.sleep(0.05)
        save_session_metadata(corc_dir, "sess2")
        mark_session_complete(corc_dir, "sess2")

        meta, content = load_latest_draft(corc_dir)
        assert meta is None
        assert content is None

    def test_load_latest_draft_skips_multiple_completed(self, tmp_path):
        """Active session is found even behind multiple completed ones."""
        corc_dir = tmp_path / ".corc"
        corc_dir.mkdir()

        save_session_metadata(corc_dir, "active-sess")
        time.sleep(0.05)
        save_session_metadata(corc_dir, "done1")
        mark_session_complete(corc_dir, "done1")
        time.sleep(0.05)
        save_session_metadata(corc_dir, "done2")
        mark_session_complete(corc_dir, "done2")

        meta, _ = load_latest_draft(corc_dir)
        assert meta is not None
        assert meta["session_id"] == "active-sess"

    def test_load_latest_draft_returns_claude_session_id(self, tmp_path):
        """Session metadata includes claude_session_id for --resume."""
        corc_dir = tmp_path / ".corc"
        corc_dir.mkdir()
        test_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        save_session_metadata(corc_dir, "sess1", claude_session_id=test_uuid)

        meta, _ = load_latest_draft(corc_dir)
        assert meta is not None
        assert meta["claude_session_id"] == test_uuid


# ---------------------------------------------------------------------------
# Interactive session launch
# ---------------------------------------------------------------------------


class TestLaunchInteractiveClaude:
    @patch("corc.plan.subprocess.run")
    def test_launches_claude_with_system_prompt(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        exit_code = launch_interactive_claude("test prompt")

        assert exit_code == 0
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "claude"
        assert "--system-prompt" in cmd
        assert "test prompt" in cmd

    @patch("corc.plan.subprocess.run")
    def test_includes_dangerously_skip_permissions(self, mock_run):
        """Interactive sessions always pass --dangerously-skip-permissions."""
        mock_run.return_value = MagicMock(returncode=0)
        launch_interactive_claude("prompt")

        cmd = mock_run.call_args[0][0]
        assert "--dangerously-skip-permissions" in cmd

    @patch("corc.plan.subprocess.run")
    def test_passes_continue_flag(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        launch_interactive_claude("prompt", continue_session=True)

        cmd = mock_run.call_args[0][0]
        assert "--continue" in cmd

    @patch("corc.plan.subprocess.run")
    def test_no_continue_by_default(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        launch_interactive_claude("prompt")

        cmd = mock_run.call_args[0][0]
        assert "--continue" not in cmd

    @patch("corc.plan.subprocess.run")
    def test_returns_exit_code(self, mock_run):
        mock_run.return_value = MagicMock(returncode=42)
        assert launch_interactive_claude("prompt") == 42

    @patch("corc.plan.subprocess.run")
    def test_passes_session_id_for_new_session(self, mock_run):
        """New sessions get --session-id for precise resume later."""
        mock_run.return_value = MagicMock(returncode=0)
        test_uuid = "12345678-1234-1234-1234-123456789abc"
        launch_interactive_claude("prompt", claude_session_id=test_uuid)

        cmd = mock_run.call_args[0][0]
        assert "--session-id" in cmd
        idx = cmd.index("--session-id")
        assert cmd[idx + 1] == test_uuid

    @patch("corc.plan.subprocess.run")
    def test_passes_resume_with_session_id(self, mock_run):
        """Resume uses --resume <id> instead of --continue."""
        mock_run.return_value = MagicMock(returncode=0)
        test_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        launch_interactive_claude("prompt", resume_claude_session_id=test_uuid)

        cmd = mock_run.call_args[0][0]
        assert "--resume" in cmd
        idx = cmd.index("--resume")
        assert cmd[idx + 1] == test_uuid
        # Should NOT also pass --continue
        assert "--continue" not in cmd
        # Should NOT pass --session-id (resuming, not creating)
        assert "--session-id" not in cmd

    @patch("corc.plan.subprocess.run")
    def test_resume_session_id_takes_precedence_over_continue(self, mock_run):
        """resume_claude_session_id takes precedence over continue_session."""
        mock_run.return_value = MagicMock(returncode=0)
        test_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        launch_interactive_claude(
            "prompt",
            continue_session=True,
            resume_claude_session_id=test_uuid,
        )

        cmd = mock_run.call_args[0][0]
        assert "--resume" in cmd
        assert "--continue" not in cmd

    @patch("corc.plan.subprocess.run")
    def test_resume_does_not_pass_new_session_id(self, mock_run):
        """When resuming, don't pass --session-id (it's for new sessions only)."""
        mock_run.return_value = MagicMock(returncode=0)
        launch_interactive_claude(
            "prompt",
            claude_session_id="new-id",
            resume_claude_session_id="old-id",
        )

        cmd = mock_run.call_args[0][0]
        assert "--session-id" not in cmd
        assert "--resume" in cmd

    def test_raises_file_not_found_for_missing_claude(self):
        """Missing claude binary raises a clear FileNotFoundError."""
        with patch("corc.plan.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError, match="command not found"):
                launch_interactive_claude("prompt")


class TestPromptSizeCheck:
    """Tests for _check_prompt_size warning."""

    def test_no_warning_for_small_prompt(self):
        """Small prompts should not trigger a warning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_prompt_size("small prompt")
            assert len(w) == 0

    def test_warns_for_large_prompt(self):
        """Prompts exceeding 800KB should trigger a warning."""
        import warnings

        large_prompt = "x" * 900_000
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_prompt_size(large_prompt)
            assert len(w) == 1
            assert "argument length" in str(w[0].message)


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestPlanCLI:
    """Tests for the corc plan CLI command."""

    @patch("corc.plan.subprocess.run")
    @patch("corc.cli._get_all")
    def test_plan_basic(self, mock_get_all, mock_run, corc_env):
        """corc plan launches claude with system prompt containing context."""
        paths, ml, ws, ks = corc_env
        al = MagicMock()
        sl = MagicMock()
        mock_get_all.return_value = (paths, ml, ws, al, sl, ks)
        mock_run.return_value = MagicMock(returncode=0)

        from click.testing import CliRunner
        from corc.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["plan"], catch_exceptions=False)

        # Find the claude launch call (not git log)
        claude_calls = [c for c in mock_run.call_args_list if c[0][0][0] == "claude"]
        assert len(claude_calls) == 1
        cmd = claude_calls[0][0][0]
        assert "--system-prompt" in cmd

        # System prompt should contain context sections
        system_prompt = cmd[cmd.index("--system-prompt") + 1]
        assert "CORC Planning Agent" in system_prompt
        assert "Knowledge Store" in system_prompt
        assert "Work State" in system_prompt

    @patch("corc.plan.subprocess.run")
    @patch("corc.cli._get_all")
    def test_plan_passes_session_id(self, mock_get_all, mock_run, corc_env):
        """corc plan passes --session-id for new sessions."""
        paths, ml, ws, ks = corc_env
        al = MagicMock()
        sl = MagicMock()
        mock_get_all.return_value = (paths, ml, ws, al, sl, ks)
        mock_run.return_value = MagicMock(returncode=0)

        from click.testing import CliRunner
        from corc.cli import cli

        runner = CliRunner()
        runner.invoke(cli, ["plan"], catch_exceptions=False)

        claude_calls = [c for c in mock_run.call_args_list if c[0][0][0] == "claude"]
        assert len(claude_calls) == 1
        cmd = claude_calls[0][0][0]
        assert "--session-id" in cmd

    @patch("corc.plan.subprocess.run")
    @patch("corc.cli._get_all")
    def test_plan_stores_claude_session_id_in_metadata(
        self, mock_get_all, mock_run, corc_env
    ):
        """Session metadata includes the Claude session UUID."""
        paths, ml, ws, ks = corc_env
        al = MagicMock()
        sl = MagicMock()
        mock_get_all.return_value = (paths, ml, ws, al, sl, ks)
        mock_run.return_value = MagicMock(returncode=0)

        from click.testing import CliRunner
        from corc.cli import cli

        runner = CliRunner()
        runner.invoke(cli, ["plan"], catch_exceptions=False)

        drafts_dir = paths["corc_dir"] / "drafts"
        sessions = list(drafts_dir.glob("session-*.json"))
        assert len(sessions) >= 1
        meta = json.loads(sessions[0].read_text())
        assert meta["claude_session_id"] is not None
        # Should look like a UUID
        assert len(meta["claude_session_id"]) == 36

    @patch("corc.plan.subprocess.run")
    @patch("corc.cli._get_all")
    def test_plan_with_seed_file(self, mock_get_all, mock_run, corc_env, tmp_path):
        """corc plan FILE pre-loads file content into context."""
        paths, ml, ws, ks = corc_env
        al = MagicMock()
        sl = MagicMock()
        mock_get_all.return_value = (paths, ml, ws, al, sl, ks)
        mock_run.return_value = MagicMock(returncode=0)

        # Create a seed file
        seed_file = tmp_path / "my-idea.md"
        seed_file.write_text("# Widget Idea\n\nBuild a widget that automates X.")

        from click.testing import CliRunner
        from corc.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["plan", str(seed_file)], catch_exceptions=False)

        claude_calls = [c for c in mock_run.call_args_list if c[0][0][0] == "claude"]
        cmd = claude_calls[0][0][0]
        system_prompt = cmd[cmd.index("--system-prompt") + 1]
        assert "Seed Document" in system_prompt
        assert "Build a widget that automates X" in system_prompt

    @patch("corc.plan.subprocess.run")
    @patch("corc.cli._get_all")
    def test_plan_resume_with_claude_session_id(self, mock_get_all, mock_run, corc_env):
        """corc plan --resume uses --resume <id> when claude_session_id exists."""
        paths, ml, ws, ks = corc_env
        al = MagicMock()
        sl = MagicMock()
        mock_get_all.return_value = (paths, ml, ws, al, sl, ks)
        mock_run.return_value = MagicMock(returncode=0)

        # Create a previous session with a Claude session UUID
        test_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        save_session_metadata(
            paths["corc_dir"], "prev-sess", claude_session_id=test_uuid
        )
        drafts = get_drafts_dir(paths["corc_dir"])
        (drafts / "plan-my-feature.md").write_text("# Draft\n\nWork in progress.")

        from click.testing import CliRunner
        from corc.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["plan", "--resume"], catch_exceptions=False)

        assert "Resuming session" in result.output

        claude_calls = [c for c in mock_run.call_args_list if c[0][0][0] == "claude"]
        cmd = claude_calls[0][0][0]
        assert "--resume" in cmd
        idx = cmd.index("--resume")
        assert cmd[idx + 1] == test_uuid
        assert "--continue" not in cmd

    @patch("corc.plan.subprocess.run")
    @patch("corc.cli._get_all")
    def test_plan_resume_fallback_to_continue(self, mock_get_all, mock_run, corc_env):
        """corc plan --resume falls back to --continue when no claude_session_id."""
        paths, ml, ws, ks = corc_env
        al = MagicMock()
        sl = MagicMock()
        mock_get_all.return_value = (paths, ml, ws, al, sl, ks)
        mock_run.return_value = MagicMock(returncode=0)

        # Create a previous session WITHOUT claude_session_id (legacy)
        save_session_metadata(paths["corc_dir"], "prev-sess")
        drafts = get_drafts_dir(paths["corc_dir"])
        (drafts / "plan-my-feature.md").write_text("# Draft\n\nWork in progress.")

        from click.testing import CliRunner
        from corc.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["plan", "--resume"], catch_exceptions=False)

        assert "Resuming session" in result.output
        # Should warn about fallback
        assert "falling back" in result.output.lower() or "Warning" in result.output

        claude_calls = [c for c in mock_run.call_args_list if c[0][0][0] == "claude"]
        cmd = claude_calls[0][0][0]
        assert "--continue" in cmd

    @patch("corc.plan.subprocess.run")
    @patch("corc.cli._get_all")
    def test_plan_resume_no_previous_session(self, mock_get_all, mock_run, corc_env):
        """corc plan --resume with no previous session starts fresh."""
        paths, ml, ws, ks = corc_env
        al = MagicMock()
        sl = MagicMock()
        mock_get_all.return_value = (paths, ml, ws, al, sl, ks)
        mock_run.return_value = MagicMock(returncode=0)

        from click.testing import CliRunner
        from corc.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["plan", "--resume"], catch_exceptions=False)

        assert "No previous session" in result.output

        # Should still launch (fresh session)
        claude_calls = [c for c in mock_run.call_args_list if c[0][0][0] == "claude"]
        cmd = claude_calls[0][0][0]
        assert "--continue" not in cmd
        assert "--resume" not in cmd or cmd[cmd.index("--resume") + 1] != ""

    @patch("corc.plan.subprocess.run")
    @patch("corc.cli._get_all")
    def test_plan_resume_skips_completed_sessions(
        self, mock_get_all, mock_run, corc_env
    ):
        """corc plan --resume skips completed sessions."""
        paths, ml, ws, ks = corc_env
        al = MagicMock()
        sl = MagicMock()
        mock_get_all.return_value = (paths, ml, ws, al, sl, ks)
        mock_run.return_value = MagicMock(returncode=0)

        # Create a session and mark it complete
        save_session_metadata(paths["corc_dir"], "done-sess")
        mark_session_complete(paths["corc_dir"], "done-sess")

        from click.testing import CliRunner
        from corc.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["plan", "--resume"], catch_exceptions=False)

        # Should start fresh since only completed session exists
        assert "No previous session" in result.output

    @patch("corc.plan.subprocess.run")
    @patch("corc.cli._get_all")
    def test_plan_saves_session_metadata(self, mock_get_all, mock_run, corc_env):
        """corc plan creates a session metadata file in .corc/drafts/."""
        paths, ml, ws, ks = corc_env
        al = MagicMock()
        sl = MagicMock()
        mock_get_all.return_value = (paths, ml, ws, al, sl, ks)
        mock_run.return_value = MagicMock(returncode=0)

        from click.testing import CliRunner
        from corc.cli import cli

        runner = CliRunner()
        runner.invoke(cli, ["plan"], catch_exceptions=False)

        drafts_dir = paths["corc_dir"] / "drafts"
        sessions = list(drafts_dir.glob("session-*.json"))
        assert len(sessions) >= 1

    @patch("corc.plan.subprocess.run")
    @patch("corc.cli._get_all")
    def test_plan_context_includes_tasks(self, mock_get_all, mock_run, corc_env):
        """System prompt includes work state tasks when they exist."""
        paths, ml, ws, ks = corc_env
        al = MagicMock()
        sl = MagicMock()

        # Create some tasks
        ml.append(
            "task_created",
            {
                "id": "abc1",
                "name": "setup-database",
                "done_when": "schema.sql file exists",
                "role": "implementer",
            },
            reason="test",
        )
        ws.refresh()

        mock_get_all.return_value = (paths, ml, ws, al, sl, ks)
        mock_run.return_value = MagicMock(returncode=0)

        from click.testing import CliRunner
        from corc.cli import cli

        runner = CliRunner()
        runner.invoke(cli, ["plan"], catch_exceptions=False)

        claude_calls = [c for c in mock_run.call_args_list if c[0][0][0] == "claude"]
        cmd = claude_calls[0][0][0]
        system_prompt = cmd[cmd.index("--system-prompt") + 1]
        assert "setup-database" in system_prompt
        assert "abc1" in system_prompt

    @patch("corc.plan.subprocess.run")
    @patch("corc.cli._get_all")
    def test_plan_handles_missing_claude(self, mock_get_all, mock_run, corc_env):
        """corc plan shows a helpful error if claude is not installed."""
        paths, ml, ws, ks = corc_env
        al = MagicMock()
        sl = MagicMock()
        mock_get_all.return_value = (paths, ml, ws, al, sl, ks)
        mock_run.side_effect = FileNotFoundError

        from click.testing import CliRunner
        from corc.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["plan"])

        assert result.exit_code != 0
        assert "command not found" in result.output.lower() or result.exit_code == 1


# ---------------------------------------------------------------------------
# Task creation from planning session (integration)
# ---------------------------------------------------------------------------


class TestTaskCreationFromPlan:
    """Verify that `corc task create` works as expected,
    since the planning session uses it to create tasks."""

    def test_task_create_cli(self, corc_env):
        """corc task create works standalone (used by planner in session)."""
        paths, ml, ws, ks = corc_env

        with patch("corc.cli._get_all") as mock_get_all:
            al = MagicMock()
            sl = MagicMock()
            mock_get_all.return_value = (paths, ml, ws, al, sl, ks)

            from click.testing import CliRunner
            from corc.cli import cli

            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "task",
                    "create",
                    "implement-widget",
                    "--done-when",
                    "widget.py file exists and pytest passes",
                    "--role",
                    "implementer",
                    "--depends-on",
                    "",
                    "--checklist",
                    "write widget,write tests,update docs",
                ],
                catch_exceptions=False,
            )

            assert "Created task" in result.output
            assert result.exit_code == 0

    def test_task_create_with_context_bundle(self, corc_env):
        """Task creation supports context bundle for file injection."""
        paths, ml, ws, ks = corc_env

        with patch("corc.cli._get_all") as mock_get_all:
            al = MagicMock()
            sl = MagicMock()
            mock_get_all.return_value = (paths, ml, ws, al, sl, ks)

            from click.testing import CliRunner
            from corc.cli import cli

            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "task",
                    "create",
                    "review-code",
                    "--done-when",
                    "review posted on PR",
                    "--role",
                    "reviewer",
                    "--context",
                    "src/corc/plan.py,SPEC.md",
                ],
                catch_exceptions=False,
            )

            assert "Created task" in result.output
            assert result.exit_code == 0

    def test_created_task_appears_in_state(self, corc_env):
        """Tasks created via CLI are visible in work state."""
        paths, ml, ws, ks = corc_env

        with patch("corc.cli._get_all") as mock_get_all:
            al = MagicMock()
            sl = MagicMock()
            mock_get_all.return_value = (paths, ml, ws, al, sl, ks)

            from click.testing import CliRunner
            from corc.cli import cli

            runner = CliRunner()
            runner.invoke(
                cli,
                [
                    "task",
                    "create",
                    "my-task",
                    "--done-when",
                    "tests pass",
                ],
                catch_exceptions=False,
            )

            ws.refresh()
            tasks = ws.list_tasks()
            assert len(tasks) == 1
            assert tasks[0]["name"] == "my-task"
            assert tasks[0]["done_when"] == "tests pass"
            assert tasks[0]["status"] == "pending"


# ---------------------------------------------------------------------------
# Crash recovery (resume)
# ---------------------------------------------------------------------------


class TestCrashRecovery:
    """Test that corc plan --resume recovers from interrupted sessions."""

    def test_resume_finds_latest_session(self, tmp_path):
        """Resume picks the most recent active session metadata."""
        corc_dir = tmp_path / ".corc"
        corc_dir.mkdir()

        save_session_metadata(corc_dir, "sess1")
        time.sleep(0.05)
        save_session_metadata(corc_dir, "sess2")
        time.sleep(0.05)
        save_session_metadata(corc_dir, "sess3")

        meta, _ = load_latest_draft(corc_dir)
        assert meta["session_id"] == "sess3"

    def test_resume_includes_draft_content(self, tmp_path):
        """Resume includes the latest draft spec content."""
        corc_dir = tmp_path / ".corc"
        corc_dir.mkdir()

        save_session_metadata(corc_dir, "sess1")
        drafts = get_drafts_dir(corc_dir)
        (drafts / "plan-feature-x.md").write_text("# Feature X\n\nDraft content here.")

        meta, content = load_latest_draft(corc_dir)
        assert content is not None
        assert "Draft content here" in content

    def test_resume_with_no_drafts_still_works(self, tmp_path):
        """Resume with session metadata but no draft files is valid."""
        corc_dir = tmp_path / ".corc"
        corc_dir.mkdir()

        save_session_metadata(corc_dir, "sess1")

        meta, content = load_latest_draft(corc_dir)
        assert meta is not None
        assert content is None  # No draft file, just session metadata

    @patch("corc.plan.subprocess.run")
    def test_resume_builds_context_with_draft(self, mock_run, corc_env):
        """Full resume flow: metadata + draft -> system prompt."""
        paths, ml, ws, ks = corc_env
        mock_run.return_value = MagicMock(returncode=0)

        # Create previous session
        save_session_metadata(paths["corc_dir"], "crashed-sess")
        drafts = get_drafts_dir(paths["corc_dir"])
        (drafts / "plan-recovery-test.md").write_text(
            "# Recovery\n\n## Problem\nSession crashed mid-planning."
        )

        # Load and build prompt
        resume_meta, draft_content = load_latest_draft(paths["corc_dir"])
        prompt = build_system_prompt(
            paths,
            ws,
            ks,
            draft_content=draft_content,
            resume_meta=resume_meta,
        )

        assert "Previous Draft" in prompt
        assert "Session crashed mid-planning" in prompt
        assert "crashed-sess" in resume_meta["session_id"]

    def test_resume_skips_completed_finds_crashed(self, tmp_path):
        """Resume skips completed sessions and finds the crashed one."""
        corc_dir = tmp_path / ".corc"
        corc_dir.mkdir()

        # First session completed normally
        save_session_metadata(corc_dir, "good-sess")
        mark_session_complete(corc_dir, "good-sess")

        time.sleep(0.05)

        # Second session crashed (still active)
        crashed_uuid = "11111111-2222-3333-4444-555555555555"
        save_session_metadata(corc_dir, "crashed-sess", claude_session_id=crashed_uuid)

        time.sleep(0.05)

        # Third session also completed
        save_session_metadata(corc_dir, "also-good")
        mark_session_complete(corc_dir, "also-good")

        meta, _ = load_latest_draft(corc_dir)
        assert meta is not None
        assert meta["session_id"] == "crashed-sess"
        assert meta["claude_session_id"] == crashed_uuid


# ---------------------------------------------------------------------------
# End-to-end flow verification
# ---------------------------------------------------------------------------


class TestEndToEndPlanFlow:
    """Verify the complete corc plan flow works end-to-end."""

    @patch("corc.plan.subprocess.run")
    @patch("corc.cli._get_all")
    def test_new_session_then_resume(self, mock_get_all, mock_run, corc_env):
        """Full flow: start session -> crash -> resume -> complete."""
        paths, ml, ws, ks = corc_env
        al = MagicMock()
        sl = MagicMock()
        mock_get_all.return_value = (paths, ml, ws, al, sl, ks)

        from click.testing import CliRunner
        from corc.cli import cli

        runner = CliRunner()

        # Step 1: Start a session that "crashes" (non-zero exit)
        mock_run.return_value = MagicMock(returncode=1)
        result1 = runner.invoke(cli, ["plan"])
        assert result1.exit_code == 1

        # Session should be saved but NOT marked complete
        drafts_dir = paths["corc_dir"] / "drafts"
        sessions = list(drafts_dir.glob("session-*.json"))
        assert len(sessions) == 1
        meta = json.loads(sessions[0].read_text())
        assert meta["status"] == "active"
        saved_claude_id = meta["claude_session_id"]
        assert saved_claude_id is not None

        # Step 2: Resume the crashed session
        mock_run.return_value = MagicMock(returncode=0)
        result2 = runner.invoke(cli, ["plan", "--resume"])
        assert "Resuming session" in result2.output

        # Should have used --resume with the saved UUID
        claude_calls = [c for c in mock_run.call_args_list if c[0][0][0] == "claude"]
        resume_call = claude_calls[-1]
        cmd = resume_call[0][0]
        assert "--resume" in cmd
        idx = cmd.index("--resume")
        assert cmd[idx + 1] == saved_claude_id

    @patch("corc.plan.subprocess.run")
    @patch("corc.cli._get_all")
    def test_completed_session_not_resumed(self, mock_get_all, mock_run, corc_env):
        """After a successful session, --resume starts fresh."""
        paths, ml, ws, ks = corc_env
        al = MagicMock()
        sl = MagicMock()
        mock_get_all.return_value = (paths, ml, ws, al, sl, ks)

        from click.testing import CliRunner
        from corc.cli import cli

        runner = CliRunner()

        # Step 1: Start and complete a session successfully
        mock_run.return_value = MagicMock(returncode=0)
        runner.invoke(cli, ["plan"])

        # Step 2: Try to resume — should start fresh
        mock_run.return_value = MagicMock(returncode=0)
        result = runner.invoke(cli, ["plan", "--resume"])
        assert "No previous session" in result.output
