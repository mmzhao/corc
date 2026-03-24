"""Tests for streaming dispatch — real-time event parsing and logging.

Tests that ClaudeCodeDispatcher:
  - Adds --output-format stream-json to the command
  - Reads stdout line-by-line as JSON events
  - Calls event_callback for each parsed event
  - Extracts final output from result event
  - Handles timeout correctly
  - Handles malformed JSON gracefully

Tests that the streaming integration:
  - Writes each event to session log immediately
  - Writes tool_use events to audit log with task_id
  - Writes assistant_message events to audit log
  - TUI styles include streaming event types
"""

import io
import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from corc.audit import AuditLog
from corc.dispatch import (
    AgentDispatcher,
    AgentResult,
    ClaudeCodeDispatcher,
    Constraints,
)
from corc.executor import Executor
from corc.mutations import MutationLog
from corc.sessions import SessionLogger
from corc.state import WorkState
from corc.tui import EVENT_STYLES, build_event_panel


# ---------------------------------------------------------------------------
# Mock subprocess for streaming tests
# ---------------------------------------------------------------------------


class MockPopen:
    """Mock subprocess.Popen that simulates streaming JSON output."""

    def __init__(self, events, exit_code=0, stderr_text=""):
        self.pid = 12345
        self.returncode = exit_code
        lines = "\n".join(json.dumps(e) for e in events) + "\n"
        self.stdout = io.StringIO(lines)
        self.stderr = io.StringIO(stderr_text)

    def wait(self):
        pass

    def kill(self):
        pass


def _make_mock_popen(events, exit_code=0, stderr_text=""):
    """Create a factory that returns a MockPopen with the given events."""

    def factory(cmd, **kwargs):
        return MockPopen(events, exit_code, stderr_text)

    return factory


# ---------------------------------------------------------------------------
# Sample streaming events
# ---------------------------------------------------------------------------


SAMPLE_EVENTS = [
    {
        "type": "system",
        "subtype": "init",
        "session_id": "test-session-1",
        "tools": ["Read", "Write"],
        "model": "claude-sonnet-4-20250514",
    },
    {
        "type": "assistant",
        "message": {
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Let me analyze the codebase."}],
            "stop_reason": "tool_use",
        },
    },
    {
        "type": "tool_use",
        "tool": {
            "type": "tool_use",
            "id": "toolu_01",
            "name": "Read",
            "input": {"file_path": "/src/main.py"},
        },
    },
    {
        "type": "tool_result",
        "tool": {
            "type": "tool_result",
            "tool_use_id": "toolu_01",
            "content": "def main():\n    print('hello')\n",
        },
    },
    {
        "type": "assistant",
        "message": {
            "id": "msg_02",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Now I'll implement the feature."}],
            "stop_reason": "tool_use",
        },
    },
    {
        "type": "tool_use",
        "tool": {
            "type": "tool_use",
            "id": "toolu_02",
            "name": "Write",
            "input": {"file_path": "/src/feature.py", "content": "# new feature\n"},
        },
    },
    {
        "type": "tool_result",
        "tool": {
            "type": "tool_result",
            "tool_use_id": "toolu_02",
            "content": "File written successfully.",
        },
    },
    {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "Task completed. Created feature.py with the new feature.",
        "duration_ms": 5000,
        "num_turns": 3,
        "total_cost_usd": 0.05,
    },
]


# Real events captured from `claude -p --output-format stream-json --verbose`
# Used to verify the parser handles the actual CLI output format (which
# includes extra fields like session_id, uuid, parent_tool_use_id, etc.).
REAL_FORMAT_EVENTS = [
    {
        "type": "system",
        "subtype": "init",
        "cwd": "/tmp/test",
        "session_id": "866b421f-2c52-418d-93a9-4c6481efc10f",
        "tools": ["Bash", "Read", "Write", "Edit", "Grep", "Glob"],
        "model": "claude-sonnet-4-20250514",
        "permissionMode": "default",
        "uuid": "26a8ca40-917d-48ed-87b4-66866d845b05",
    },
    {
        "type": "assistant",
        "message": {
            "model": "claude-sonnet-4-20250514",
            "id": "msg_01CKpMXE8xMyRhjjw9NU9zB6",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "I'll read the file first."}],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 10},
        },
        "parent_tool_use_id": None,
        "session_id": "866b421f-2c52-418d-93a9-4c6481efc10f",
        "uuid": "b570c34f-eef4-4662-9287-887250a4dafe",
    },
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
    {
        "type": "tool_result",
        "tool": {
            "type": "tool_result",
            "tool_use_id": "toolu_01abc",
            "content": "def main():\n    pass\n",
        },
        "session_id": "866b421f-2c52-418d-93a9-4c6481efc10f",
        "uuid": "d4e5f6a7-8901-2bcd-ef34-567890123456",
    },
    {
        "type": "assistant",
        "message": {
            "model": "claude-sonnet-4-20250514",
            "id": "msg_02XYZ",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Done implementing."}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 200, "output_tokens": 20},
        },
        "parent_tool_use_id": None,
        "session_id": "866b421f-2c52-418d-93a9-4c6481efc10f",
        "uuid": "e5f6a7b8-0123-4cde-f567-890123456789",
    },
    {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "duration_ms": 5000,
        "duration_api_ms": 4800,
        "num_turns": 2,
        "result": "Implementation complete.",
        "stop_reason": "end_turn",
        "session_id": "866b421f-2c52-418d-93a9-4c6481efc10f",
        "total_cost_usd": 0.05,
        "uuid": "f6a7b8c9-1234-5def-a678-901234567890",
    },
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project structure for testing."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "events").mkdir()
    (tmp_path / "data" / "sessions").mkdir()
    (tmp_path / ".corc").mkdir()
    return tmp_path


@pytest.fixture
def session_logger(tmp_project):
    return SessionLogger(tmp_project / "data" / "sessions")


@pytest.fixture
def audit_log(tmp_project):
    return AuditLog(tmp_project / "data" / "events")


@pytest.fixture
def mutation_log(tmp_project):
    return MutationLog(tmp_project / "data" / "mutations.jsonl")


@pytest.fixture
def work_state(tmp_project, mutation_log):
    return WorkState(tmp_project / "data" / "state.db", mutation_log)


def _create_task(mutation_log, task_id, name, done_when="tests_pass", depends_on=None):
    mutation_log.append(
        "task_created",
        {
            "id": task_id,
            "name": name,
            "description": f"Test task: {name}",
            "role": "implementer",
            "depends_on": depends_on or [],
            "done_when": done_when,
            "checklist": [],
            "context_bundle": [],
        },
        reason="Test setup",
    )


# ===========================================================================
# ClaudeCodeDispatcher streaming tests
# ===========================================================================


class TestStreamingDispatch:
    """Test ClaudeCodeDispatcher with --output-format stream-json."""

    def test_command_includes_stream_json(self, monkeypatch):
        """Verify the command includes --output-format stream-json --verbose."""
        captured_cmd = {}

        def mock_popen(cmd, **kwargs):
            captured_cmd["cmd"] = cmd
            return MockPopen(
                [
                    {
                        "type": "result",
                        "subtype": "success",
                        "result": "done",
                        "is_error": False,
                    }
                ],
                exit_code=0,
            )

        monkeypatch.setattr("corc.dispatch.subprocess.Popen", mock_popen)
        dispatcher = ClaudeCodeDispatcher()
        dispatcher.dispatch("test prompt", "system prompt", Constraints())

        cmd = captured_cmd["cmd"]
        assert "--output-format" in cmd
        assert "stream-json" in cmd
        assert "--verbose" in cmd, (
            "--verbose is required for --output-format stream-json in claude -p mode"
        )
        assert "claude" == cmd[0]
        assert "-p" == cmd[1]
        assert "test prompt" == cmd[2]

    def test_command_includes_dangerously_skip_permissions(self, monkeypatch):
        """Dispatch always passes --dangerously-skip-permissions to claude -p."""
        captured_cmd = {}

        def mock_popen(cmd, **kwargs):
            captured_cmd["cmd"] = cmd
            return MockPopen(
                [
                    {
                        "type": "result",
                        "subtype": "success",
                        "result": "done",
                        "is_error": False,
                    }
                ],
                exit_code=0,
            )

        monkeypatch.setattr("corc.dispatch.subprocess.Popen", mock_popen)
        dispatcher = ClaudeCodeDispatcher()
        dispatcher.dispatch("test prompt", "system prompt", Constraints())

        cmd = captured_cmd["cmd"]
        assert "--dangerously-skip-permissions" in cmd

    def test_event_callback_called_for_each_event(self, monkeypatch):
        """Each parsed JSON event triggers the event_callback."""
        monkeypatch.setattr(
            "corc.dispatch.subprocess.Popen",
            _make_mock_popen(SAMPLE_EVENTS),
        )

        received = []
        dispatcher = ClaudeCodeDispatcher()
        dispatcher.dispatch(
            "test",
            "system",
            Constraints(),
            event_callback=lambda e: received.append(e),
        )

        assert len(received) == len(SAMPLE_EVENTS)
        assert received[0]["type"] == "system"
        assert received[1]["type"] == "assistant"
        assert received[2]["type"] == "tool_use"
        assert received[3]["type"] == "tool_result"
        assert received[-1]["type"] == "result"

    def test_result_text_extracted(self, monkeypatch):
        """AgentResult.output is the 'result' field from the result event."""
        monkeypatch.setattr(
            "corc.dispatch.subprocess.Popen",
            _make_mock_popen(SAMPLE_EVENTS),
        )

        dispatcher = ClaudeCodeDispatcher()
        result = dispatcher.dispatch("test", "system", Constraints())

        assert (
            result.output == "Task completed. Created feature.py with the new feature."
        )
        assert result.exit_code == 0

    def test_stderr_appended(self, monkeypatch):
        """Stderr content is appended to the output."""
        monkeypatch.setattr(
            "corc.dispatch.subprocess.Popen",
            _make_mock_popen(
                [{"type": "result", "result": "done", "is_error": False}],
                stderr_text="Warning: something\n",
            ),
        )

        dispatcher = ClaudeCodeDispatcher()
        result = dispatcher.dispatch("test", "system", Constraints())

        assert "done" in result.output
        assert "STDERR" in result.output
        assert "Warning: something" in result.output

    def test_malformed_json_skipped(self, monkeypatch):
        """Non-JSON lines are silently skipped."""

        def mock_popen(cmd, **kwargs):
            lines = 'not json\n{"type":"result","result":"ok","is_error":false}\n'
            proc = MockPopen([], exit_code=0)
            proc.stdout = io.StringIO(lines)
            return proc

        monkeypatch.setattr("corc.dispatch.subprocess.Popen", mock_popen)

        received = []
        dispatcher = ClaudeCodeDispatcher()
        result = dispatcher.dispatch(
            "test",
            "system",
            Constraints(),
            event_callback=lambda e: received.append(e),
        )

        # Only the valid JSON line should be received
        assert len(received) == 1
        assert received[0]["type"] == "result"
        assert result.output == "ok"

    def test_empty_lines_skipped(self, monkeypatch):
        """Blank lines in stdout are skipped."""

        def mock_popen(cmd, **kwargs):
            lines = '\n\n{"type":"result","result":"ok","is_error":false}\n\n'
            proc = MockPopen([], exit_code=0)
            proc.stdout = io.StringIO(lines)
            return proc

        monkeypatch.setattr("corc.dispatch.subprocess.Popen", mock_popen)

        received = []
        dispatcher = ClaudeCodeDispatcher()
        result = dispatcher.dispatch(
            "test",
            "system",
            Constraints(),
            event_callback=lambda e: received.append(e),
        )

        assert len(received) == 1
        assert result.output == "ok"

    def test_no_result_event_empty_output(self, monkeypatch):
        """If no result event is received, output is empty."""
        events = [
            {"type": "system", "subtype": "init"},
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "hello"}]},
            },
        ]
        monkeypatch.setattr(
            "corc.dispatch.subprocess.Popen",
            _make_mock_popen(events),
        )

        dispatcher = ClaudeCodeDispatcher()
        result = dispatcher.dispatch("test", "system", Constraints())

        assert result.output == ""

    def test_pid_callback_called(self, monkeypatch):
        """pid_callback receives the mock process PID."""
        monkeypatch.setattr(
            "corc.dispatch.subprocess.Popen",
            _make_mock_popen(
                [{"type": "result", "result": "done", "is_error": False}],
            ),
        )

        pids = []
        dispatcher = ClaudeCodeDispatcher()
        dispatcher.dispatch(
            "test",
            "system",
            Constraints(),
            pid_callback=lambda pid: pids.append(pid),
        )

        assert len(pids) == 1
        assert pids[0] == 12345

    def test_no_callback_still_works(self, monkeypatch):
        """Dispatch works fine without an event_callback."""
        monkeypatch.setattr(
            "corc.dispatch.subprocess.Popen",
            _make_mock_popen(SAMPLE_EVENTS),
        )

        dispatcher = ClaudeCodeDispatcher()
        result = dispatcher.dispatch("test", "system", Constraints())

        assert result.exit_code == 0
        assert "Task completed" in result.output

    def test_nonzero_exit_code(self, monkeypatch):
        """Non-zero exit code is propagated."""
        monkeypatch.setattr(
            "corc.dispatch.subprocess.Popen",
            _make_mock_popen(
                [
                    {
                        "type": "result",
                        "result": "",
                        "is_error": True,
                        "subtype": "error",
                    }
                ],
                exit_code=1,
            ),
        )

        dispatcher = ClaudeCodeDispatcher()
        result = dispatcher.dispatch("test", "system", Constraints())

        assert result.exit_code == 1

    def test_duration_measured(self, monkeypatch):
        """Duration is measured from start to finish."""
        monkeypatch.setattr(
            "corc.dispatch.subprocess.Popen",
            _make_mock_popen(
                [{"type": "result", "result": "ok", "is_error": False}],
            ),
        )

        dispatcher = ClaudeCodeDispatcher()
        result = dispatcher.dispatch("test", "system", Constraints())

        assert result.duration_s >= 0
        assert result.duration_s < 5  # Should be nearly instant with mock

    def test_constraints_in_command(self, monkeypatch):
        """Constraints are translated to CLI flags."""
        captured_cmd = {}

        def mock_popen(cmd, **kwargs):
            captured_cmd["cmd"] = cmd
            return MockPopen(
                [{"type": "result", "result": "ok", "is_error": False}],
            )

        monkeypatch.setattr("corc.dispatch.subprocess.Popen", mock_popen)

        constraints = Constraints(
            allowed_tools=["Read", "Write"],
        )
        dispatcher = ClaudeCodeDispatcher()
        dispatcher.dispatch("prompt", "system", constraints)

        cmd = captured_cmd["cmd"]
        assert "--allowedTools" in cmd
        idx = cmd.index("--allowedTools")
        assert cmd[idx + 1] == "Read,Write"
        assert "--max-turns" not in cmd

    def test_real_format_events_parsed(self, monkeypatch):
        """Events matching the real claude CLI stream-json format are parsed correctly.

        The real output includes extra fields (session_id, uuid, parent_tool_use_id)
        that the simplified SAMPLE_EVENTS don't have. Verify these are handled.
        """
        monkeypatch.setattr(
            "corc.dispatch.subprocess.Popen",
            _make_mock_popen(REAL_FORMAT_EVENTS),
        )

        received = []
        dispatcher = ClaudeCodeDispatcher()
        result = dispatcher.dispatch(
            "test",
            "system",
            Constraints(),
            event_callback=lambda e: received.append(e),
        )

        assert len(received) == len(REAL_FORMAT_EVENTS)
        types = [e["type"] for e in received]
        assert types == [
            "system",
            "assistant",
            "tool_use",
            "tool_result",
            "assistant",
            "result",
        ]
        assert result.output == "Implementation complete."

    def test_real_format_assistant_has_extra_fields(self, monkeypatch):
        """Real assistant events include model, usage, parent_tool_use_id, session_id."""
        monkeypatch.setattr(
            "corc.dispatch.subprocess.Popen",
            _make_mock_popen(REAL_FORMAT_EVENTS),
        )

        received = []
        dispatcher = ClaudeCodeDispatcher()
        dispatcher.dispatch(
            "test",
            "system",
            Constraints(),
            event_callback=lambda e: received.append(e),
        )

        # First assistant event
        assistant = received[1]
        assert assistant["type"] == "assistant"
        assert assistant["message"]["model"] == "claude-sonnet-4-20250514"
        assert "session_id" in assistant
        assert "uuid" in assistant

    def test_non_json_error_line_logged_and_skipped(self, monkeypatch):
        """When claude outputs a non-JSON error (e.g. missing --verbose), it is skipped."""

        def mock_popen(cmd, **kwargs):
            # Simulate what happens without --verbose: error message on stdout
            lines = "Error: When using --print, --output-format=stream-json requires --verbose\n"
            proc = MockPopen([], exit_code=1)
            proc.stdout = io.StringIO(lines)
            return proc

        monkeypatch.setattr("corc.dispatch.subprocess.Popen", mock_popen)

        received = []
        dispatcher = ClaudeCodeDispatcher()
        result = dispatcher.dispatch(
            "test",
            "system",
            Constraints(),
            event_callback=lambda e: received.append(e),
        )

        # No events should be received (error line is not valid JSON)
        assert len(received) == 0
        assert result.output == ""


# ===========================================================================
# Session log streaming tests
# ===========================================================================


class TestSessionLogStreaming:
    """Test that streaming events are written to session log incrementally."""

    def test_log_stream_event(self, session_logger):
        """log_stream_event writes one JSONL entry per call."""
        event = {"type": "tool_use", "tool": {"name": "Read", "input": {}}}
        session_logger.log_stream_event("task-1", 1, event)

        entries = session_logger.read_session("task-1", 1)
        assert len(entries) == 1
        assert entries[0]["type"] == "stream_event"
        assert entries[0]["stream_type"] == "tool_use"
        # The content field holds the JSON-encoded event
        parsed = json.loads(entries[0]["content"])
        assert parsed["type"] == "tool_use"

    def test_multiple_stream_events(self, session_logger):
        """Multiple stream events are appended incrementally."""
        events = [
            {"type": "system", "subtype": "init"},
            {"type": "assistant", "message": {"content": []}},
            {"type": "tool_use", "tool": {"name": "Read", "input": {}}},
            {"type": "result", "result": "done"},
        ]
        for event in events:
            session_logger.log_stream_event("task-1", 1, event)

        entries = session_logger.read_session("task-1", 1)
        assert len(entries) == 4
        types = [e["stream_type"] for e in entries]
        assert types == ["system", "assistant", "tool_use", "result"]

    def test_stream_events_mixed_with_dispatch_output(self, session_logger):
        """Stream events coexist with dispatch and output entries."""
        session_logger.log_dispatch("task-1", 1, "prompt", "system", ["Read"], 3.0)
        session_logger.log_stream_event(
            "task-1", 1, {"type": "assistant", "message": {}}
        )
        session_logger.log_stream_event(
            "task-1", 1, {"type": "result", "result": "done"}
        )
        session_logger.log_output("task-1", 1, "done", 0, 5.0)

        entries = session_logger.read_session("task-1", 1)
        types = [e["type"] for e in entries]
        assert types == ["dispatch", "stream_event", "stream_event", "output"]

    def test_crash_safety_incremental_writes(self, session_logger):
        """Each event is independently readable — crash between writes is safe."""
        session_logger.log_stream_event("task-1", 1, {"type": "system"})

        # Read after first write
        entries = session_logger.read_session("task-1", 1)
        assert len(entries) == 1

        session_logger.log_stream_event(
            "task-1", 1, {"type": "assistant", "message": {}}
        )

        # Read after second write — both entries are there
        entries = session_logger.read_session("task-1", 1)
        assert len(entries) == 2


# ===========================================================================
# Audit log streaming tests
# ===========================================================================


class TestAuditLogStreaming:
    """Test that tool_use and assistant events are written to audit log."""

    def test_tool_use_in_audit_log(
        self, audit_log, session_logger, mutation_log, work_state, tmp_project
    ):
        """tool_use events are logged to the audit log with task_id."""
        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        # Create the executor's event callback and call it directly
        executor = Executor(
            dispatcher=MagicMock(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        callback = executor._make_event_callback("t1", 1)
        callback(
            {
                "type": "tool_use",
                "tool": {"name": "Read", "input": {"file_path": "/test.py"}},
            }
        )

        events = audit_log.read_today()
        tool_events = [e for e in events if e["event_type"] == "tool_use"]
        assert len(tool_events) == 1
        assert tool_events[0]["task_id"] == "t1"
        assert tool_events[0]["tool_name"] == "Read"
        assert "/test.py" in tool_events[0]["tool_input"]

    def test_assistant_message_in_audit_log(
        self, audit_log, session_logger, mutation_log, work_state, tmp_project
    ):
        """assistant events with text content are logged to audit log."""
        executor = Executor(
            dispatcher=MagicMock(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        callback = executor._make_event_callback("t1", 1)
        callback(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "I will analyze the code."},
                    ],
                },
            }
        )

        events = audit_log.read_today()
        msg_events = [e for e in events if e["event_type"] == "assistant_message"]
        assert len(msg_events) == 1
        assert msg_events[0]["task_id"] == "t1"
        assert msg_events[0]["content"] == "I will analyze the code."

    def test_assistant_message_no_text_blocks(
        self, audit_log, session_logger, mutation_log, work_state, tmp_project
    ):
        """assistant events without text blocks do not create audit entries."""
        executor = Executor(
            dispatcher=MagicMock(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        callback = executor._make_event_callback("t1", 1)
        callback(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "Read", "input": {}},
                    ],
                },
            }
        )

        events = audit_log.read_today()
        msg_events = [e for e in events if e["event_type"] == "assistant_message"]
        assert len(msg_events) == 0

    def test_all_events_written_to_session_log(
        self, audit_log, session_logger, mutation_log, work_state, tmp_project
    ):
        """Every event (regardless of type) is written to the session log."""
        executor = Executor(
            dispatcher=MagicMock(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        callback = executor._make_event_callback("t1", 1)
        for event in SAMPLE_EVENTS:
            callback(event)

        entries = session_logger.read_session("t1", 1)
        assert len(entries) == len(SAMPLE_EVENTS)
        # All entries are stream_event type
        assert all(e["type"] == "stream_event" for e in entries)

    def test_system_events_only_in_session_log(
        self, audit_log, session_logger, mutation_log, work_state, tmp_project
    ):
        """system and result events go to session log but NOT audit log."""
        executor = Executor(
            dispatcher=MagicMock(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        callback = executor._make_event_callback("t1", 1)
        callback({"type": "system", "subtype": "init"})
        callback({"type": "result", "result": "done"})

        # Session log has both
        entries = session_logger.read_session("t1", 1)
        assert len(entries) == 2

        # Audit log has neither (system/result are not interesting for audit)
        events = audit_log.read_today()
        assert len(events) == 0


# ===========================================================================
# Cost extraction tests
# ===========================================================================


class TestCostExtraction:
    """Test that result events with cost data produce task_cost audit events."""

    def test_result_event_with_cost_produces_task_cost(
        self, audit_log, session_logger, mutation_log, work_state, tmp_project
    ):
        """Result event with total_cost_usd creates a task_cost audit event."""
        executor = Executor(
            dispatcher=MagicMock(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        callback = executor._make_event_callback("t1", 1, "implementer")
        callback(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "Task done.",
                "total_cost_usd": 0.05,
                "duration_ms": 5000,
                "num_turns": 3,
            }
        )

        events = audit_log.read_today()
        cost_events = [e for e in events if e["event_type"] == "task_cost"]
        assert len(cost_events) == 1
        ce = cost_events[0]
        assert ce["task_id"] == "t1"
        assert ce["cost_usd"] == 0.05
        assert ce["duration_ms"] == 5000
        assert ce["num_turns"] == 3
        assert ce["role"] == "implementer"
        assert ce["attempt"] == 1

    def test_result_event_without_cost_no_task_cost(
        self, audit_log, session_logger, mutation_log, work_state, tmp_project
    ):
        """Result event without total_cost_usd does NOT create task_cost event."""
        executor = Executor(
            dispatcher=MagicMock(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        callback = executor._make_event_callback("t1", 1)
        callback({"type": "result", "result": "done"})

        events = audit_log.read_today()
        cost_events = [e for e in events if e["event_type"] == "task_cost"]
        assert len(cost_events) == 0

    def test_token_accumulation_from_assistant_messages(
        self, audit_log, session_logger, mutation_log, work_state, tmp_project
    ):
        """Token counts are accumulated from assistant message usage fields."""
        executor = Executor(
            dispatcher=MagicMock(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        callback = executor._make_event_callback("t1", 1, "implementer")

        # Two assistant messages with usage
        callback(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "First message."}],
                    "usage": {"input_tokens": 100, "output_tokens": 10},
                },
            }
        )
        callback(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Second message."}],
                    "usage": {"input_tokens": 200, "output_tokens": 20},
                },
            }
        )

        # Result event triggers cost logging with accumulated tokens
        callback(
            {
                "type": "result",
                "total_cost_usd": 0.03,
                "duration_ms": 3000,
                "num_turns": 2,
            }
        )

        events = audit_log.read_today()
        cost_events = [e for e in events if e["event_type"] == "task_cost"]
        assert len(cost_events) == 1
        ce = cost_events[0]
        assert ce["input_tokens"] == 300  # 100 + 200
        assert ce["output_tokens"] == 30  # 10 + 20
        assert ce["cost_usd"] == 0.03

    def test_cache_token_accumulation(
        self, audit_log, session_logger, mutation_log, work_state, tmp_project
    ):
        """Cache tokens (creation + read) are accumulated and reported."""
        executor = Executor(
            dispatcher=MagicMock(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        callback = executor._make_event_callback("t1", 1, "implementer")

        callback(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Msg."}],
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 10,
                        "cache_creation_input_tokens": 50,
                        "cache_read_input_tokens": 30,
                    },
                },
            }
        )

        callback(
            {
                "type": "result",
                "total_cost_usd": 0.02,
                "duration_ms": 2000,
                "num_turns": 1,
            }
        )

        events = audit_log.read_today()
        cost_events = [e for e in events if e["event_type"] == "task_cost"]
        assert len(cost_events) == 1
        ce = cost_events[0]
        assert ce["cache_tokens"] == 80  # 50 + 30
        assert ce["input_tokens"] == 100
        assert ce["output_tokens"] == 10

    def test_full_sample_events_produce_task_cost(
        self, audit_log, session_logger, mutation_log, work_state, tmp_project
    ):
        """Full SAMPLE_EVENTS sequence (which has total_cost_usd) produces task_cost."""
        executor = Executor(
            dispatcher=MagicMock(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        callback = executor._make_event_callback("t1", 1, "implementer")
        for event in SAMPLE_EVENTS:
            callback(event)

        events = audit_log.read_today()
        cost_events = [e for e in events if e["event_type"] == "task_cost"]
        assert len(cost_events) == 1
        ce = cost_events[0]
        assert ce["cost_usd"] == 0.05
        assert ce["duration_ms"] == 5000
        assert ce["num_turns"] == 3

    def test_real_format_events_produce_task_cost_with_tokens(
        self, audit_log, session_logger, mutation_log, work_state, tmp_project
    ):
        """REAL_FORMAT_EVENTS (with usage in assistant messages) produce correct token counts."""
        executor = Executor(
            dispatcher=MagicMock(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        callback = executor._make_event_callback("t1", 1, "implementer")
        for event in REAL_FORMAT_EVENTS:
            callback(event)

        events = audit_log.read_today()
        cost_events = [e for e in events if e["event_type"] == "task_cost"]
        assert len(cost_events) == 1
        ce = cost_events[0]
        assert ce["cost_usd"] == 0.05
        assert ce["input_tokens"] == 300  # 100 + 200
        assert ce["output_tokens"] == 30  # 10 + 20
        assert ce["cache_tokens"] == 0  # No cache tokens in real format events
        assert ce["num_turns"] == 2
        assert ce["duration_ms"] == 5000

    def test_analyze_aggregation_with_task_cost_events(
        self, audit_log, session_logger, mutation_log, work_state, tmp_project
    ):
        """analyze.py aggregate_costs picks up task_cost events via cost_usd field."""
        from corc.analyze import aggregate_costs

        executor = Executor(
            dispatcher=MagicMock(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        # Generate two task_cost events
        cb1 = executor._make_event_callback("t1", 1, "implementer")
        cb1(
            {
                "type": "result",
                "total_cost_usd": 0.05,
                "duration_ms": 5000,
                "num_turns": 3,
            }
        )

        cb2 = executor._make_event_callback("t2", 1, "reviewer")
        cb2(
            {
                "type": "result",
                "total_cost_usd": 0.10,
                "duration_ms": 8000,
                "num_turns": 5,
            }
        )

        events = audit_log.read_today()
        breakdown = aggregate_costs(events)

        assert breakdown.total_usd == pytest.approx(0.15)
        assert breakdown.event_count == 2
        assert "t1" in breakdown.by_task
        assert "t2" in breakdown.by_task
        assert breakdown.by_task["t1"] == pytest.approx(0.05)
        assert breakdown.by_task["t2"] == pytest.approx(0.10)
        assert "implementer" in breakdown.by_role
        assert "reviewer" in breakdown.by_role


# ===========================================================================
# Executor integration tests
# ===========================================================================


class MockStreamingDispatcher(AgentDispatcher):
    """A mock dispatcher that emits stream events via event_callback.

    Unlike the ClaudeCodeDispatcher tests which mock subprocess.Popen,
    this tests the Executor→Dispatcher→callback wiring at the interface
    level: the dispatcher receives event_callback and invokes it directly.
    """

    def __init__(self, events: list[dict], result: AgentResult | None = None):
        self.events = events
        self._result = result or AgentResult(
            output="Mock completed.",
            exit_code=0,
            duration_s=0.1,
        )
        self.received_event_callback = None

    def dispatch(
        self,
        prompt: str,
        system_prompt: str,
        constraints: Constraints,
        pid_callback=None,
        event_callback=None,
        cwd=None,
    ) -> AgentResult:
        self.received_event_callback = event_callback
        if event_callback:
            for event in self.events:
                event_callback(event)
        return self._result


class TestExecutorStreamingIntegration:
    """Test that the executor wires up event callbacks correctly."""

    def test_executor_creates_event_callback(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Executor creates an event callback for each dispatch."""
        executor = Executor(
            dispatcher=MagicMock(),
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        callback = executor._make_event_callback("task-1", 1)
        assert callable(callback)

    def test_streaming_dispatch_via_executor(
        self,
        monkeypatch,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        tmp_project,
    ):
        """Full integration: executor dispatches with streaming, logs correctly."""
        monkeypatch.setattr(
            "corc.dispatch.subprocess.Popen",
            _make_mock_popen(SAMPLE_EVENTS),
        )

        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        dispatcher = ClaudeCodeDispatcher()
        executor = Executor(
            dispatcher=dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        executor.dispatch(task)
        time.sleep(0.5)
        completed = executor.poll_completed()

        assert len(completed) == 1
        assert (
            completed[0].result.output
            == "Task completed. Created feature.py with the new feature."
        )

        # Session log should have dispatch + stream events + output
        session = session_logger.read_session("t1", 1)
        types = [e["type"] for e in session]
        assert "dispatch" in types
        assert "stream_event" in types
        assert "output" in types

        # Count stream events
        stream_events = [e for e in session if e["type"] == "stream_event"]
        assert len(stream_events) == len(SAMPLE_EVENTS)

        # Audit log should have tool_use events
        all_audit = audit_log.read_today()
        tool_events = [e for e in all_audit if e["event_type"] == "tool_use"]
        assert len(tool_events) == 2  # Read and Write
        assert tool_events[0]["tool_name"] == "Read"
        assert tool_events[1]["tool_name"] == "Write"

        # Audit log should have assistant_message events
        msg_events = [e for e in all_audit if e["event_type"] == "assistant_message"]
        assert len(msg_events) == 2  # Two assistant messages
        assert "analyze the codebase" in msg_events[0]["content"]

        executor.shutdown()

    def test_mock_dispatcher_stream_events_captured(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Mock dispatcher emits stream events; executor captures them in session and audit logs.

        This test verifies the full wiring path: Executor builds event_callback,
        passes it to dispatcher.dispatch(), dispatcher invokes callback for each
        event, and the callback writes to both session logger and audit log.
        """
        mock_dispatcher = MockStreamingDispatcher(
            events=SAMPLE_EVENTS,
            result=AgentResult(
                output="Task completed. Created feature.py with the new feature.",
                exit_code=0,
                duration_s=5.0,
            ),
        )

        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = Executor(
            dispatcher=mock_dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        executor.dispatch(task)
        time.sleep(0.5)
        completed = executor.poll_completed()

        assert len(completed) == 1
        assert completed[0].result.exit_code == 0

        # -- Verify event_callback was passed to the dispatcher --
        assert mock_dispatcher.received_event_callback is not None

        # -- Verify session log contains stream_event entries --
        session = session_logger.read_session("t1", 1)
        stream_entries = [e for e in session if e["type"] == "stream_event"]
        assert len(stream_entries) == len(SAMPLE_EVENTS)

        # Verify stream_type metadata on each entry
        stream_types = [e["stream_type"] for e in stream_entries]
        assert "assistant" in stream_types
        assert "tool_use" in stream_types
        assert "tool_result" in stream_types
        assert "system" in stream_types
        assert "result" in stream_types

        # -- Verify audit log has real-time tool_use entries --
        all_audit = audit_log.read_today()
        tool_events = [e for e in all_audit if e["event_type"] == "tool_use"]
        assert len(tool_events) == 2  # Read and Write from SAMPLE_EVENTS
        assert tool_events[0]["task_id"] == "t1"
        assert tool_events[0]["tool_name"] == "Read"
        assert tool_events[1]["tool_name"] == "Write"

        # -- Verify audit log has assistant_message entries --
        msg_events = [e for e in all_audit if e["event_type"] == "assistant_message"]
        assert len(msg_events) == 2
        assert "analyze the codebase" in msg_events[0]["content"]
        assert "implement the feature" in msg_events[1]["content"]

        executor.shutdown()

    def test_mock_dispatcher_no_events_still_works(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Dispatcher that emits zero stream events still completes normally."""
        mock_dispatcher = MockStreamingDispatcher(
            events=[],
            result=AgentResult(output="Done.", exit_code=0, duration_s=0.1),
        )

        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = Executor(
            dispatcher=mock_dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        executor.dispatch(task)
        time.sleep(0.5)
        completed = executor.poll_completed()

        assert len(completed) == 1
        assert completed[0].result.output == "Done."

        # Session log has dispatch + output but no stream events
        session = session_logger.read_session("t1", 1)
        stream_entries = [e for e in session if e["type"] == "stream_event"]
        assert len(stream_entries) == 0

        executor.shutdown()

    def test_tool_result_events_in_session_log(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """tool_result events are captured in session log for full conversation replay."""
        tool_result_events = [
            {"type": "tool_use", "tool": {"name": "Bash", "input": {"command": "ls"}}},
            {
                "type": "tool_result",
                "tool": {
                    "tool_use_id": "toolu_99",
                    "content": "file1.py\nfile2.py\n",
                },
            },
            {"type": "result", "result": "Listed files.", "is_error": False},
        ]

        mock_dispatcher = MockStreamingDispatcher(
            events=tool_result_events,
            result=AgentResult(output="Listed files.", exit_code=0, duration_s=0.2),
        )

        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = Executor(
            dispatcher=mock_dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        executor.dispatch(task)
        time.sleep(0.5)
        executor.poll_completed()

        session = session_logger.read_session("t1", 1)
        stream_entries = [e for e in session if e["type"] == "stream_event"]
        stream_types = [e["stream_type"] for e in stream_entries]

        assert "tool_use" in stream_types
        assert "tool_result" in stream_types
        assert "result" in stream_types

        # Verify tool_result content is preserved in session log
        tool_result_entry = [
            e for e in stream_entries if e["stream_type"] == "tool_result"
        ][0]
        parsed = json.loads(tool_result_entry["content"])
        assert "file1.py" in parsed["tool"]["content"]

        executor.shutdown()

    def test_real_format_events_in_session_log(
        self, mutation_log, work_state, audit_log, session_logger, tmp_project
    ):
        """Real-format events (with session_id, uuid, etc.) are captured in session log.

        Verifies the full pipeline: REAL_FORMAT_EVENTS flow through
        MockStreamingDispatcher → event_callback → session_logger, producing
        stream_event entries with correct stream_type values.
        """
        mock_dispatcher = MockStreamingDispatcher(
            events=REAL_FORMAT_EVENTS,
            result=AgentResult(
                output="Implementation complete.",
                exit_code=0,
                duration_s=5.0,
            ),
        )

        _create_task(mutation_log, "t1", "Task 1")
        work_state.refresh()
        task = work_state.get_task("t1")

        executor = Executor(
            dispatcher=mock_dispatcher,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
        )

        executor.dispatch(task)
        time.sleep(0.5)
        completed = executor.poll_completed()

        assert len(completed) == 1

        # Session log should contain stream_event entries
        session = session_logger.read_session("t1", 1)
        stream_entries = [e for e in session if e["type"] == "stream_event"]
        assert len(stream_entries) == len(REAL_FORMAT_EVENTS)

        # Verify stream_type values from real format
        stream_types = [e["stream_type"] for e in stream_entries]
        assert "assistant" in stream_types
        assert "tool_use" in stream_types
        assert "tool_result" in stream_types
        assert "system" in stream_types
        assert "result" in stream_types

        # Verify content preserves extra fields (session_id, uuid)
        system_entry = [e for e in stream_entries if e["stream_type"] == "system"][0]
        parsed = json.loads(system_entry["content"])
        assert "session_id" in parsed
        assert "uuid" in parsed

        # Audit log should have tool_use and assistant_message entries
        all_audit = audit_log.read_today()
        tool_events = [e for e in all_audit if e["event_type"] == "tool_use"]
        assert len(tool_events) == 1  # One Read tool use
        assert tool_events[0]["tool_name"] == "Read"

        msg_events = [e for e in all_audit if e["event_type"] == "assistant_message"]
        assert len(msg_events) == 2  # Two assistant messages
        assert "read the file first" in msg_events[0]["content"]
        assert "Done implementing" in msg_events[1]["content"]

        executor.shutdown()


# ===========================================================================
# TUI streaming event display tests
# ===========================================================================


class TestTUIStreamingEvents:
    """Test that the TUI renders streaming event types correctly."""

    def test_event_styles_include_streaming_types(self):
        """EVENT_STYLES includes tool_use and assistant_message."""
        assert "tool_use" in EVENT_STYLES
        assert "assistant_message" in EVENT_STYLES

    def test_tool_use_event_rendered(self):
        """tool_use events are rendered with tool name in the event panel."""
        events = [
            {
                "timestamp": "2026-03-22T10:00:00.000Z",
                "event_type": "tool_use",
                "task_id": "task-abc1",
                "tool_name": "Read",
                "tool_input": '{"file_path":"/src/main.py"}',
            }
        ]

        panel = build_event_panel(events)
        # Render to plain text for assertions
        import io
        from rich.console import Console

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=120, no_color=True)
        console.print(panel)
        text = buf.getvalue()

        assert "tool_use" in text
        assert "Read" in text
        assert "task-abc" in text

    def test_assistant_message_event_filtered_from_events_panel(self):
        """assistant_message events are filtered out of the events panel.

        These noisy events are shown in the streaming detail panel instead.
        """
        events = [
            {
                "timestamp": "2026-03-22T10:00:00.000Z",
                "event_type": "assistant_message",
                "task_id": "task-abc1",
                "content": "Let me analyze the code structure.",
            }
        ]

        panel = build_event_panel(events)
        import io
        from rich.console import Console

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=120, no_color=True)
        console.print(panel)
        text = buf.getvalue()

        assert "assistant_message" not in text
        assert "Let me analyze the code structure." not in text

    def test_mixed_events_rendered(self):
        """Mixed traditional and streaming events render together.

        assistant_message events are filtered out of the events panel;
        structural events (dispatched, tool_use, completed) still appear.
        """
        events = [
            {
                "timestamp": "2026-03-22T10:00:00.000Z",
                "event_type": "task_dispatched",
                "task_id": "task-abc1",
            },
            {
                "timestamp": "2026-03-22T10:00:01.000Z",
                "event_type": "assistant_message",
                "task_id": "task-abc1",
                "content": "Analyzing...",
            },
            {
                "timestamp": "2026-03-22T10:00:02.000Z",
                "event_type": "tool_use",
                "task_id": "task-abc1",
                "tool_name": "Bash",
                "tool_input": '{"command":"ls"}',
            },
            {
                "timestamp": "2026-03-22T10:00:10.000Z",
                "event_type": "task_completed",
                "task_id": "task-abc1",
                "duration_s": 10.0,
            },
        ]

        panel = build_event_panel(events)
        import io
        from rich.console import Console

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=120, no_color=True)
        console.print(panel)
        text = buf.getvalue()

        assert "task_dispatched" in text
        assert "assistant_message" not in text  # filtered out
        assert "tool_use" in text
        assert "task_completed" in text
        assert "Bash" in text

    def test_multiline_assistant_content_filtered(self):
        """assistant_message events (even with multiline content) are filtered
        out of the events panel — they appear in the streaming panel instead."""
        events = [
            {
                "timestamp": "2026-03-22T10:00:00.000Z",
                "event_type": "assistant_message",
                "task_id": "task-abc1",
                "content": "Line 1 of reasoning.\nLine 2 of reasoning.\nLine 3.",
            }
        ]

        panel = build_event_panel(events)
        import io
        from rich.console import Console

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=120, no_color=True)
        console.print(panel)
        text = buf.getvalue()

        assert "Line 1 of reasoning." not in text
        assert "assistant_message" not in text
