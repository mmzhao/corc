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
from corc.dispatch import AgentDispatcher, AgentResult, ClaudeCodeDispatcher, Constraints
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
    {"type": "system", "subtype": "init", "session_id": "test-session-1",
     "tools": ["Read", "Write"], "model": "claude-sonnet-4-20250514"},
    {"type": "assistant", "message": {
        "id": "msg_01", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": "Let me analyze the codebase."}],
        "stop_reason": "tool_use",
    }},
    {"type": "tool_use", "tool": {
        "type": "tool_use", "id": "toolu_01", "name": "Read",
        "input": {"file_path": "/src/main.py"},
    }},
    {"type": "tool_result", "tool": {
        "type": "tool_result", "tool_use_id": "toolu_01",
        "content": "def main():\n    print('hello')\n",
    }},
    {"type": "assistant", "message": {
        "id": "msg_02", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": "Now I'll implement the feature."}],
        "stop_reason": "tool_use",
    }},
    {"type": "tool_use", "tool": {
        "type": "tool_use", "id": "toolu_02", "name": "Write",
        "input": {"file_path": "/src/feature.py", "content": "# new feature\n"},
    }},
    {"type": "tool_result", "tool": {
        "type": "tool_result", "tool_use_id": "toolu_02",
        "content": "File written successfully.",
    }},
    {"type": "result", "subtype": "success", "is_error": False,
     "result": "Task completed. Created feature.py with the new feature.",
     "duration_ms": 5000, "num_turns": 3, "total_cost_usd": 0.05},
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
    mutation_log.append("task_created", {
        "id": task_id,
        "name": name,
        "description": f"Test task: {name}",
        "role": "implementer",
        "depends_on": depends_on or [],
        "done_when": done_when,
        "checklist": [],
        "context_bundle": [],
    }, reason="Test setup")


# ===========================================================================
# ClaudeCodeDispatcher streaming tests
# ===========================================================================


class TestStreamingDispatch:
    """Test ClaudeCodeDispatcher with --output-format stream-json."""

    def test_command_includes_stream_json(self, monkeypatch):
        """Verify the command includes --output-format stream-json."""
        captured_cmd = {}

        def mock_popen(cmd, **kwargs):
            captured_cmd["cmd"] = cmd
            return MockPopen(
                [{"type": "result", "subtype": "success", "result": "done", "is_error": False}],
                exit_code=0,
            )

        monkeypatch.setattr("corc.dispatch.subprocess.Popen", mock_popen)
        dispatcher = ClaudeCodeDispatcher()
        dispatcher.dispatch("test prompt", "system prompt", Constraints())

        cmd = captured_cmd["cmd"]
        assert "--output-format" in cmd
        assert "stream-json" in cmd
        assert "claude" == cmd[0]
        assert "-p" == cmd[1]
        assert "test prompt" == cmd[2]

    def test_event_callback_called_for_each_event(self, monkeypatch):
        """Each parsed JSON event triggers the event_callback."""
        monkeypatch.setattr(
            "corc.dispatch.subprocess.Popen",
            _make_mock_popen(SAMPLE_EVENTS),
        )

        received = []
        dispatcher = ClaudeCodeDispatcher()
        dispatcher.dispatch(
            "test", "system", Constraints(),
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

        assert result.output == "Task completed. Created feature.py with the new feature."
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
            "test", "system", Constraints(),
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
            "test", "system", Constraints(),
            event_callback=lambda e: received.append(e),
        )

        assert len(received) == 1
        assert result.output == "ok"

    def test_no_result_event_empty_output(self, monkeypatch):
        """If no result event is received, output is empty."""
        events = [
            {"type": "system", "subtype": "init"},
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "hello"}]}},
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
            "test", "system", Constraints(),
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
                [{"type": "result", "result": "", "is_error": True, "subtype": "error"}],
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
            max_turns=10,
        )
        dispatcher = ClaudeCodeDispatcher()
        dispatcher.dispatch("prompt", "system", constraints)

        cmd = captured_cmd["cmd"]
        assert "--allowedTools" in cmd
        idx = cmd.index("--allowedTools")
        assert cmd[idx + 1] == "Read,Write"
        assert "--max-turns" in cmd
        idx = cmd.index("--max-turns")
        assert cmd[idx + 1] == "10"


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
        session_logger.log_stream_event("task-1", 1, {"type": "assistant", "message": {}})
        session_logger.log_stream_event("task-1", 1, {"type": "result", "result": "done"})
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

        session_logger.log_stream_event("task-1", 1, {"type": "assistant", "message": {}})

        # Read after second write — both entries are there
        entries = session_logger.read_session("task-1", 1)
        assert len(entries) == 2


# ===========================================================================
# Audit log streaming tests
# ===========================================================================


class TestAuditLogStreaming:
    """Test that tool_use and assistant events are written to audit log."""

    def test_tool_use_in_audit_log(self, audit_log, session_logger, mutation_log,
                                    work_state, tmp_project):
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
        callback({"type": "tool_use", "tool": {"name": "Read", "input": {"file_path": "/test.py"}}})

        events = audit_log.read_today()
        tool_events = [e for e in events if e["event_type"] == "tool_use"]
        assert len(tool_events) == 1
        assert tool_events[0]["task_id"] == "t1"
        assert tool_events[0]["tool_name"] == "Read"
        assert "/test.py" in tool_events[0]["tool_input"]

    def test_assistant_message_in_audit_log(self, audit_log, session_logger,
                                             mutation_log, work_state, tmp_project):
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
        callback({
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "I will analyze the code."},
                ],
            },
        })

        events = audit_log.read_today()
        msg_events = [e for e in events if e["event_type"] == "assistant_message"]
        assert len(msg_events) == 1
        assert msg_events[0]["task_id"] == "t1"
        assert msg_events[0]["content"] == "I will analyze the code."

    def test_assistant_message_no_text_blocks(self, audit_log, session_logger,
                                               mutation_log, work_state, tmp_project):
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
        callback({
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "name": "Read", "input": {}},
                ],
            },
        })

        events = audit_log.read_today()
        msg_events = [e for e in events if e["event_type"] == "assistant_message"]
        assert len(msg_events) == 0

    def test_all_events_written_to_session_log(self, audit_log, session_logger,
                                                 mutation_log, work_state, tmp_project):
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

    def test_system_events_only_in_session_log(self, audit_log, session_logger,
                                                 mutation_log, work_state, tmp_project):
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
# Executor integration tests
# ===========================================================================


class TestExecutorStreamingIntegration:
    """Test that the executor wires up event callbacks correctly."""

    def test_executor_creates_event_callback(self, mutation_log, work_state,
                                               audit_log, session_logger, tmp_project):
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

    def test_streaming_dispatch_via_executor(self, monkeypatch, mutation_log,
                                              work_state, audit_log,
                                              session_logger, tmp_project):
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
        assert completed[0].result.output == "Task completed. Created feature.py with the new feature."

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
        events = [{
            "timestamp": "2026-03-22T10:00:00.000Z",
            "event_type": "tool_use",
            "task_id": "task-abc1",
            "tool_name": "Read",
            "tool_input": '{"file_path":"/src/main.py"}',
        }]

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

    def test_assistant_message_event_rendered(self):
        """assistant_message events show content in the panel."""
        events = [{
            "timestamp": "2026-03-22T10:00:00.000Z",
            "event_type": "assistant_message",
            "task_id": "task-abc1",
            "content": "Let me analyze the code structure.",
        }]

        panel = build_event_panel(events)
        import io
        from rich.console import Console
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=120, no_color=True)
        console.print(panel)
        text = buf.getvalue()

        assert "assistant_message" in text
        assert "Let me analyze the code structure." in text

    def test_mixed_events_rendered(self):
        """Mixed traditional and streaming events render together."""
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
        assert "assistant_message" in text
        assert "tool_use" in text
        assert "task_completed" in text
        assert "Bash" in text

    def test_multiline_assistant_content(self):
        """assistant_message with multiline content is shown fully."""
        events = [{
            "timestamp": "2026-03-22T10:00:00.000Z",
            "event_type": "assistant_message",
            "task_id": "task-abc1",
            "content": "Line 1 of reasoning.\nLine 2 of reasoning.\nLine 3.",
        }]

        panel = build_event_panel(events)
        import io
        from rich.console import Console
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=120, no_color=True)
        console.print(panel)
        text = buf.getvalue()

        assert "Line 1 of reasoning." in text
        assert "Line 2 of reasoning." in text
        assert "Line 3." in text
