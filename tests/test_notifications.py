"""Tests for the notification system: channels, config loading, and manager routing.

Covers:
- NotificationChannel interface contract
- TerminalNotification output and color formatting
- SlackWebhookNotification payload construction and HTTP posting
- DiscordWebhookNotification payload construction and HTTP posting
- Configuration loading from .corc/config.yaml
- Default configuration when no config file exists
- NotificationManager routing events to configured channels
- Custom channel registration
- Convenience helpers for escalation, failure, cost_threshold, pause
- Error handling (webhook failures, missing channels)
"""

import io
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from unittest.mock import patch

import pytest
import yaml

from corc.notifications import (
    DiscordWebhookNotification,
    NotificationChannel,
    NotificationConfig,
    NotificationManager,
    SlackWebhookNotification,
    TerminalNotification,
    load_notification_config,
    notify_cost_threshold,
    notify_escalation,
    notify_pause,
    notify_task_failure,
    SEVERITY_CRITICAL,
    SEVERITY_ERROR,
    SEVERITY_INFO,
    SEVERITY_WARNING,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def corc_dir(tmp_path):
    """Create a minimal .corc directory."""
    d = tmp_path / ".corc"
    d.mkdir()
    return d


@pytest.fixture
def string_stream():
    """A StringIO stream for capturing terminal output."""
    return io.StringIO()


class RecordingChannel(NotificationChannel):
    """A test channel that records all calls."""

    def __init__(self, success: bool = True):
        self.calls: list[dict] = []
        self._success = success

    def send(self, event_type: str, title: str, body: str, severity: str) -> bool:
        self.calls.append(
            {
                "event_type": event_type,
                "title": title,
                "body": body,
                "severity": severity,
            }
        )
        return self._success


class FailingChannel(NotificationChannel):
    """A channel that always raises an exception."""

    def send(self, event_type: str, title: str, body: str, severity: str) -> bool:
        raise ConnectionError("connection failed")


# ---------------------------------------------------------------------------
# NotificationChannel interface
# ---------------------------------------------------------------------------


class TestNotificationChannelInterface:
    def test_interface_is_abstract(self):
        """NotificationChannel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            NotificationChannel()

    def test_subclass_must_implement_send(self):
        """Subclass that doesn't implement send() raises TypeError."""

        class Incomplete(NotificationChannel):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_subclass_with_send_works(self):
        """Subclass implementing send() can be instantiated."""
        recorder = RecordingChannel()
        result = recorder.send("test", "title", "body", SEVERITY_INFO)
        assert result is True
        assert len(recorder.calls) == 1


# ---------------------------------------------------------------------------
# TerminalNotification
# ---------------------------------------------------------------------------


class TestTerminalNotification:
    def test_send_writes_to_stream(self, string_stream):
        """TerminalNotification writes notification to the provided stream."""
        channel = TerminalNotification(stream=string_stream, use_color=False)
        result = channel.send(
            "escalation", "Task escalated", "Details here", SEVERITY_ERROR
        )

        assert result is True
        output = string_stream.getvalue()
        assert "[CORC ERROR]" in output
        assert "Task escalated" in output
        assert "Details here" in output

    def test_send_with_color(self, string_stream):
        """TerminalNotification uses ANSI codes when color is enabled."""
        channel = TerminalNotification(stream=string_stream, use_color=True)
        channel.send("test", "Title", "Body", SEVERITY_WARNING)

        output = string_stream.getvalue()
        assert "\033[33m" in output  # yellow
        assert "\033[0m" in output  # reset

    def test_send_without_color(self, string_stream):
        """TerminalNotification omits ANSI codes when color is disabled."""
        channel = TerminalNotification(stream=string_stream, use_color=False)
        channel.send("test", "Title", "Body", SEVERITY_WARNING)

        output = string_stream.getvalue()
        assert "\033[" not in output

    def test_severity_levels(self, string_stream):
        """Each severity level produces the correct tag."""
        channel = TerminalNotification(stream=string_stream, use_color=False)

        for severity, expected_tag in [
            (SEVERITY_INFO, "[CORC INFO]"),
            (SEVERITY_WARNING, "[CORC WARNING]"),
            (SEVERITY_ERROR, "[CORC ERROR]"),
            (SEVERITY_CRITICAL, "[CORC CRITICAL]"),
        ]:
            string_stream.truncate(0)
            string_stream.seek(0)
            channel.send("test", "Title", "", severity)
            assert expected_tag in string_stream.getvalue()

    def test_empty_body(self, string_stream):
        """TerminalNotification handles empty body gracefully."""
        channel = TerminalNotification(stream=string_stream, use_color=False)
        result = channel.send("test", "Title only", "", SEVERITY_INFO)

        assert result is True
        output = string_stream.getvalue()
        assert "Title only" in output

    def test_multiline_body(self, string_stream):
        """Multi-line body is indented correctly."""
        channel = TerminalNotification(stream=string_stream, use_color=False)
        channel.send("test", "Title", "line 1\nline 2\nline 3", SEVERITY_INFO)

        output = string_stream.getvalue()
        lines = output.strip().splitlines()
        assert len(lines) == 4  # title + 3 body lines
        assert lines[1].startswith("  line 1")
        assert lines[2].startswith("  line 2")
        assert lines[3].startswith("  line 3")

    def test_returns_false_on_write_error(self):
        """TerminalNotification returns False if write fails."""

        class BrokenStream:
            def write(self, _):
                raise OSError("pipe broken")

            def flush(self):
                pass

            def isatty(self):
                return False

        channel = TerminalNotification(stream=BrokenStream(), use_color=False)
        result = channel.send("test", "Title", "Body", SEVERITY_INFO)
        assert result is False


# ---------------------------------------------------------------------------
# SlackWebhookNotification
# ---------------------------------------------------------------------------


class _WebhookHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that records POST requests."""

    received_payloads = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _WebhookHandler.received_payloads.append(json.loads(body))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format, *args):
        pass  # Suppress request logging


@pytest.fixture
def webhook_server():
    """Start a local HTTP server to receive webhook POSTs."""
    _WebhookHandler.received_payloads = []
    server = HTTPServer(("127.0.0.1", 0), _WebhookHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/webhook"
    server.shutdown()


class TestSlackWebhookNotification:
    def test_send_posts_to_webhook(self, webhook_server):
        """SlackWebhookNotification sends JSON payload to the webhook URL."""
        channel = SlackWebhookNotification(webhook_url=webhook_server)
        result = channel.send("escalation", "Task escalated", "Details", SEVERITY_ERROR)

        assert result is True
        assert len(_WebhookHandler.received_payloads) == 1

        payload = _WebhookHandler.received_payloads[0]
        assert "Task escalated" in payload["text"]
        assert payload["blocks"][0]["type"] == "header"
        assert "escalation" in payload["blocks"][1]["text"]["text"]

    def test_payload_includes_severity_emoji(self, webhook_server):
        """Slack payload includes severity-appropriate emoji."""
        channel = SlackWebhookNotification(webhook_url=webhook_server)
        channel.send("test", "Title", "Body", SEVERITY_CRITICAL)

        payload = _WebhookHandler.received_payloads[0]
        assert ":rotating_light:" in payload["text"]

    def test_returns_false_on_bad_url(self):
        """SlackWebhookNotification returns False for unreachable URL."""
        channel = SlackWebhookNotification(
            webhook_url="http://127.0.0.1:1/nonexistent",
            timeout=0.5,
        )
        result = channel.send("test", "Title", "Body", SEVERITY_INFO)
        assert result is False

    def test_payload_structure(self, webhook_server):
        """Verify the complete Slack Block Kit payload structure."""
        channel = SlackWebhookNotification(webhook_url=webhook_server)
        channel.send("task_failure", "Task failed: foo", "Details", SEVERITY_ERROR)

        payload = _WebhookHandler.received_payloads[0]
        # Header block
        assert payload["blocks"][0]["text"]["type"] == "plain_text"
        assert payload["blocks"][0]["text"]["text"] == "Task failed: foo"
        # Section block
        section = payload["blocks"][1]
        assert section["type"] == "section"
        assert section["text"]["type"] == "mrkdwn"
        assert "ERROR" in section["text"]["text"]
        assert "`task_failure`" in section["text"]["text"]


# ---------------------------------------------------------------------------
# DiscordWebhookNotification
# ---------------------------------------------------------------------------


class TestDiscordWebhookNotification:
    def test_send_posts_to_webhook(self, webhook_server):
        """DiscordWebhookNotification sends embed payload to the webhook URL."""
        channel = DiscordWebhookNotification(webhook_url=webhook_server)
        result = channel.send("escalation", "Escalated!", "Details", SEVERITY_CRITICAL)

        assert result is True
        assert len(_WebhookHandler.received_payloads) == 1

        payload = _WebhookHandler.received_payloads[0]
        embed = payload["embeds"][0]
        assert embed["title"] == "Escalated!"
        assert embed["description"] == "Details"
        assert embed["color"] == 0x992D22  # dark red for critical

    def test_severity_colors(self, webhook_server):
        """Each severity maps to a different Discord embed color."""
        channel = DiscordWebhookNotification(webhook_url=webhook_server)

        expected_colors = {
            SEVERITY_INFO: 0x3498DB,
            SEVERITY_WARNING: 0xF39C12,
            SEVERITY_ERROR: 0xE74C3C,
            SEVERITY_CRITICAL: 0x992D22,
        }

        for severity, expected_color in expected_colors.items():
            _WebhookHandler.received_payloads.clear()
            channel.send("test", "Title", "Body", severity)
            payload = _WebhookHandler.received_payloads[0]
            assert payload["embeds"][0]["color"] == expected_color, (
                f"Wrong color for severity={severity}"
            )

    def test_embed_fields(self, webhook_server):
        """Discord embed includes event_type and severity as fields."""
        channel = DiscordWebhookNotification(webhook_url=webhook_server)
        channel.send("task_failure", "Failed!", "Error info", SEVERITY_ERROR)

        embed = _WebhookHandler.received_payloads[0]["embeds"][0]
        fields = {f["name"]: f["value"] for f in embed["fields"]}
        assert fields["Event"] == "task_failure"
        assert fields["Severity"] == "ERROR"

    def test_returns_false_on_bad_url(self):
        """DiscordWebhookNotification returns False for unreachable URL."""
        channel = DiscordWebhookNotification(
            webhook_url="http://127.0.0.1:1/nonexistent",
            timeout=0.5,
        )
        result = channel.send("test", "Title", "Body", SEVERITY_INFO)
        assert result is False


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


class TestNotificationConfig:
    def test_defaults_when_no_config_file(self, corc_dir):
        """Returns defaults when .corc/config.yaml doesn't exist."""
        config = load_notification_config(corc_dir)

        assert config.channels["terminal"]["enabled"] is True
        assert config.channels["slack"]["enabled"] is False
        assert config.channels["discord"]["enabled"] is False
        assert "escalation" in config.triggers
        assert "terminal" in config.triggers["escalation"]

    def test_load_from_yaml(self, corc_dir):
        """Loads notification config from YAML file."""
        config_data = {
            "notifications": {
                "channels": {
                    "terminal": {"enabled": True},
                    "slack": {
                        "enabled": True,
                        "webhook_url": "https://hooks.slack.com/test",
                    },
                },
                "triggers": {
                    "escalation": ["terminal", "slack"],
                    "task_failure": ["terminal"],
                },
            }
        }
        (corc_dir / "config.yaml").write_text(yaml.dump(config_data))

        config = load_notification_config(corc_dir)

        assert config.channels["slack"]["enabled"] is True
        assert config.channels["slack"]["webhook_url"] == "https://hooks.slack.com/test"
        assert config.triggers["escalation"] == ["terminal", "slack"]

    def test_partial_config_merges_with_defaults(self, corc_dir):
        """Partial config merges with defaults — missing keys use defaults."""
        config_data = {
            "notifications": {
                "channels": {
                    "slack": {
                        "enabled": True,
                        "webhook_url": "https://hooks.slack.com/test",
                    },
                },
            }
        }
        (corc_dir / "config.yaml").write_text(yaml.dump(config_data))

        config = load_notification_config(corc_dir)

        # Terminal should still be enabled from defaults
        assert config.channels["terminal"]["enabled"] is True
        # Slack should be overridden
        assert config.channels["slack"]["enabled"] is True
        # Default triggers should be present
        assert "escalation" in config.triggers

    def test_empty_trigger_list(self, corc_dir):
        """Trigger with empty list means no notifications for that event."""
        config_data = {
            "notifications": {
                "triggers": {
                    "escalation": [],
                },
            }
        }
        (corc_dir / "config.yaml").write_text(yaml.dump(config_data))

        config = load_notification_config(corc_dir)
        assert config.triggers["escalation"] == []

    def test_malformed_yaml_returns_defaults(self, corc_dir):
        """Malformed YAML file returns defaults gracefully."""
        (corc_dir / "config.yaml").write_text("not: valid: yaml: {{{}}")

        config = load_notification_config(corc_dir)
        # Should fall back to defaults
        assert config.channels["terminal"]["enabled"] is True

    def test_discord_config(self, corc_dir):
        """Discord webhook configuration loads correctly."""
        config_data = {
            "notifications": {
                "channels": {
                    "discord": {
                        "enabled": True,
                        "webhook_url": "https://discord.com/api/webhooks/test",
                    },
                },
                "triggers": {
                    "escalation": ["terminal", "discord"],
                },
            }
        }
        (corc_dir / "config.yaml").write_text(yaml.dump(config_data))

        config = load_notification_config(corc_dir)
        assert config.channels["discord"]["enabled"] is True
        assert (
            config.channels["discord"]["webhook_url"]
            == "https://discord.com/api/webhooks/test"
        )

    def test_config_with_other_sections(self, corc_dir):
        """Config file with other sections only reads notifications."""
        config_data = {
            "alerts": {
                "cost": {"enabled": True},
            },
            "notifications": {
                "channels": {
                    "terminal": {"enabled": False},
                },
            },
        }
        (corc_dir / "config.yaml").write_text(yaml.dump(config_data))

        config = load_notification_config(corc_dir)
        assert config.channels["terminal"]["enabled"] is False


# ---------------------------------------------------------------------------
# NotificationManager
# ---------------------------------------------------------------------------


class TestNotificationManager:
    def test_terminal_only_default(self):
        """Default config creates only a terminal channel."""
        manager = NotificationManager()
        assert "terminal" in manager.active_channels
        assert "slack" not in manager.active_channels

    def test_notify_routes_to_configured_channels(self):
        """notify() sends to all channels listed in the trigger config."""
        config = NotificationConfig(
            channels={"test_ch": {"enabled": True}},
            triggers={"escalation": ["test_ch"]},
        )
        manager = NotificationManager(config=config)

        recorder = RecordingChannel()
        manager.register_channel("test_ch", recorder)

        results = manager.notify(
            "escalation", "Escalated!", "Details", SEVERITY_CRITICAL
        )

        assert results["test_ch"] is True
        assert len(recorder.calls) == 1
        assert recorder.calls[0]["title"] == "Escalated!"
        assert recorder.calls[0]["severity"] == SEVERITY_CRITICAL

    def test_notify_skips_unconfigured_events(self):
        """Events not in triggers config produce no notifications."""
        config = NotificationConfig(
            channels={"terminal": {"enabled": True}},
            triggers={"escalation": ["terminal"]},
        )
        manager = NotificationManager(config=config)

        recorder = RecordingChannel()
        manager.register_channel("terminal", recorder)

        results = manager.notify("unknown_event", "Title", "Body", SEVERITY_INFO)
        assert results == {}
        assert len(recorder.calls) == 0

    def test_notify_missing_channel_returns_false(self):
        """If a trigger references a non-existent channel, result is False."""
        config = NotificationConfig(
            channels={},
            triggers={"escalation": ["nonexistent"]},
        )
        manager = NotificationManager(config=config)

        results = manager.notify("escalation", "Title", "Body", SEVERITY_ERROR)
        assert results["nonexistent"] is False

    def test_notify_exception_in_channel_returns_false(self):
        """If a channel raises an exception, result is False (no crash)."""
        config = NotificationConfig(
            channels={"broken": {"enabled": True}},
            triggers={"escalation": ["broken"]},
        )
        manager = NotificationManager(config=config)
        manager.register_channel("broken", FailingChannel())

        results = manager.notify("escalation", "Title", "Body", SEVERITY_ERROR)
        assert results["broken"] is False

    def test_register_custom_channel(self):
        """Custom channels can be registered and receive notifications."""
        manager = NotificationManager()
        recorder = RecordingChannel()
        manager.register_channel("custom", recorder)

        # Add custom channel to triggers
        manager.config.triggers["escalation"] = ["custom"]

        results = manager.notify("escalation", "Custom test", "Body", SEVERITY_INFO)
        assert results["custom"] is True
        assert recorder.calls[0]["title"] == "Custom test"

    def test_multiple_channels_per_trigger(self):
        """Multiple channels receive the same notification for one event."""
        config = NotificationConfig(
            channels={
                "ch1": {"enabled": True},
                "ch2": {"enabled": True},
            },
            triggers={"escalation": ["ch1", "ch2"]},
        )
        manager = NotificationManager(config=config)

        r1 = RecordingChannel()
        r2 = RecordingChannel()
        manager.register_channel("ch1", r1)
        manager.register_channel("ch2", r2)

        results = manager.notify("escalation", "Title", "Body", SEVERITY_ERROR)
        assert results == {"ch1": True, "ch2": True}
        assert len(r1.calls) == 1
        assert len(r2.calls) == 1

    def test_load_from_corc_dir(self, corc_dir):
        """Manager loads config from corc_dir."""
        config_data = {
            "notifications": {
                "channels": {
                    "terminal": {"enabled": True},
                },
                "triggers": {
                    "escalation": ["terminal"],
                },
            }
        }
        (corc_dir / "config.yaml").write_text(yaml.dump(config_data))

        manager = NotificationManager(corc_dir=corc_dir)
        assert "terminal" in manager.active_channels

    def test_slack_channel_created_with_webhook_url(self, corc_dir):
        """Slack channel is created when enabled with webhook_url."""
        config_data = {
            "notifications": {
                "channels": {
                    "slack": {
                        "enabled": True,
                        "webhook_url": "https://hooks.slack.com/services/test",
                    },
                },
            }
        }
        (corc_dir / "config.yaml").write_text(yaml.dump(config_data))

        manager = NotificationManager(corc_dir=corc_dir)
        assert "slack" in manager.active_channels

    def test_slack_channel_not_created_without_url(self, corc_dir):
        """Slack channel is NOT created when enabled but no webhook_url."""
        config_data = {
            "notifications": {
                "channels": {
                    "slack": {
                        "enabled": True,
                        "webhook_url": None,
                    },
                },
            }
        }
        (corc_dir / "config.yaml").write_text(yaml.dump(config_data))

        manager = NotificationManager(corc_dir=corc_dir)
        assert "slack" not in manager.active_channels

    def test_discord_channel_created_with_webhook_url(self, corc_dir):
        """Discord channel is created when enabled with webhook_url."""
        config_data = {
            "notifications": {
                "channels": {
                    "discord": {
                        "enabled": True,
                        "webhook_url": "https://discord.com/api/webhooks/test",
                    },
                },
            }
        }
        (corc_dir / "config.yaml").write_text(yaml.dump(config_data))

        manager = NotificationManager(corc_dir=corc_dir)
        assert "discord" in manager.active_channels


# ---------------------------------------------------------------------------
# Webhook integration tests (Slack + Discord against local server)
# ---------------------------------------------------------------------------


class TestWebhookIntegration:
    def test_slack_to_local_server(self, webhook_server):
        """Full round-trip: SlackWebhookNotification -> local HTTP server."""
        config = NotificationConfig(
            channels={"slack": {"enabled": True, "webhook_url": webhook_server}},
            triggers={"escalation": ["slack"]},
        )
        manager = NotificationManager(config=config)

        # Manager should have created the slack channel from config
        assert "slack" in manager.active_channels

        results = manager.notify(
            "escalation", "Task escalated", "Retries exhausted", SEVERITY_CRITICAL
        )
        assert results["slack"] is True
        assert len(_WebhookHandler.received_payloads) == 1

    def test_discord_to_local_server(self, webhook_server):
        """Full round-trip: DiscordWebhookNotification -> local HTTP server."""
        config = NotificationConfig(
            channels={"discord": {"enabled": True, "webhook_url": webhook_server}},
            triggers={"task_failure": ["discord"]},
        )
        manager = NotificationManager(config=config)

        assert "discord" in manager.active_channels

        results = manager.notify(
            "task_failure", "Task failed", "Exit code 1", SEVERITY_ERROR
        )
        assert results["discord"] is True
        assert len(_WebhookHandler.received_payloads) == 1

        embed = _WebhookHandler.received_payloads[0]["embeds"][0]
        assert embed["title"] == "Task failed"


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


class TestConvenienceHelpers:
    def test_notify_escalation(self):
        """notify_escalation sends correct event_type and severity."""
        config = NotificationConfig(
            channels={"test": {"enabled": True}},
            triggers={"escalation": ["test"]},
        )
        manager = NotificationManager(config=config)
        recorder = RecordingChannel()
        manager.register_channel("test", recorder)

        task = {"id": "t1", "name": "Build feature"}
        escalation = {"attempts": 4, "error": "Agent exited with code 1"}

        results = notify_escalation(manager, task, escalation)

        assert results["test"] is True
        call = recorder.calls[0]
        assert call["event_type"] == "escalation"
        assert call["severity"] == SEVERITY_CRITICAL
        assert "Build feature" in call["title"]
        assert "t1" in call["title"]
        assert "4" in call["body"]

    def test_notify_task_failure(self):
        """notify_task_failure sends correct event_type and severity."""
        config = NotificationConfig(
            channels={"test": {"enabled": True}},
            triggers={"task_failure": ["test"]},
        )
        manager = NotificationManager(config=config)
        recorder = RecordingChannel()
        manager.register_channel("test", recorder)

        task = {"id": "t2", "name": "Fix bug"}
        results = notify_task_failure(manager, task, attempt=2, error="Timeout")

        assert results["test"] is True
        call = recorder.calls[0]
        assert call["event_type"] == "task_failure"
        assert call["severity"] == SEVERITY_ERROR
        assert "Fix bug" in call["title"]
        assert "Attempt 2" in call["body"]

    def test_notify_cost_threshold(self):
        """notify_cost_threshold sends correct event and amounts."""
        config = NotificationConfig(
            channels={"test": {"enabled": True}},
            triggers={"cost_threshold": ["test"]},
        )
        manager = NotificationManager(config=config)
        recorder = RecordingChannel()
        manager.register_channel("test", recorder)

        results = notify_cost_threshold(manager, "daily", current=55.0, limit=50.0)

        assert results["test"] is True
        call = recorder.calls[0]
        assert call["event_type"] == "cost_threshold"
        assert call["severity"] == SEVERITY_WARNING
        assert "$55.00" in call["body"]
        assert "$50.00" in call["body"]

    def test_notify_pause(self):
        """notify_pause sends correct event and details."""
        config = NotificationConfig(
            channels={"test": {"enabled": True}},
            triggers={"pause": ["test"]},
        )
        manager = NotificationManager(config=config)
        recorder = RecordingChannel()
        manager.register_channel("test", recorder)

        results = notify_pause(manager, reason="deploy", source="operator")

        assert results["test"] is True
        call = recorder.calls[0]
        assert call["event_type"] == "pause"
        assert call["severity"] == SEVERITY_WARNING
        assert "deploy" in call["body"]
        assert "operator" in call["body"]

    def test_escalation_with_no_channels_configured(self):
        """Escalation with empty trigger list returns empty dict."""
        config = NotificationConfig(
            channels={},
            triggers={"escalation": []},
        )
        manager = NotificationManager(config=config)

        results = notify_escalation(
            manager,
            {"id": "t1", "name": "Task"},
            {"attempts": 4, "error": "failed"},
        )
        assert results == {}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_channel_returning_false(self):
        """Channel returning False is reflected in results."""
        config = NotificationConfig(
            channels={"failing": {"enabled": True}},
            triggers={"escalation": ["failing"]},
        )
        manager = NotificationManager(config=config)
        manager.register_channel("failing", RecordingChannel(success=False))

        results = manager.notify("escalation", "Title", "Body", SEVERITY_ERROR)
        assert results["failing"] is False

    def test_mixed_success_and_failure(self):
        """Multiple channels with mixed results are all reported."""
        config = NotificationConfig(
            channels={
                "good": {"enabled": True},
                "bad": {"enabled": True},
            },
            triggers={"escalation": ["good", "bad"]},
        )
        manager = NotificationManager(config=config)
        manager.register_channel("good", RecordingChannel(success=True))
        manager.register_channel("bad", RecordingChannel(success=False))

        results = manager.notify("escalation", "Title", "Body", SEVERITY_ERROR)
        assert results["good"] is True
        assert results["bad"] is False

    def test_null_trigger_list_in_yaml(self, corc_dir):
        """null trigger list in YAML is treated as empty list."""
        config_data = {
            "notifications": {
                "triggers": {
                    "escalation": None,
                },
            }
        }
        (corc_dir / "config.yaml").write_text(yaml.dump(config_data))

        config = load_notification_config(corc_dir)
        assert config.triggers["escalation"] == []


# ---------------------------------------------------------------------------
# Processor integration tests — notifications fire on escalation + failure
# ---------------------------------------------------------------------------


class TestProcessorNotificationIntegration:
    """Verify that process_completed fires notifications on failure/escalation."""

    @pytest.fixture
    def tmp_project(self, tmp_path):
        """Create a minimal project structure for processor tests."""
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "events").mkdir()
        (tmp_path / "data" / "sessions").mkdir()
        (tmp_path / ".corc").mkdir()
        return tmp_path

    @pytest.fixture
    def mutation_log(self, tmp_project):
        from corc.mutations import MutationLog

        return MutationLog(tmp_project / "data" / "mutations.jsonl")

    @pytest.fixture
    def work_state(self, tmp_project, mutation_log):
        from corc.state import WorkState

        return WorkState(tmp_project / "data" / "state.db", mutation_log)

    @pytest.fixture
    def audit_log(self, tmp_project):
        from corc.audit import AuditLog

        return AuditLog(tmp_project / "data" / "events")

    @pytest.fixture
    def session_logger(self, tmp_project):
        from corc.sessions import SessionLogger

        return SessionLogger(tmp_project / "data" / "sessions")

    @pytest.fixture
    def notification_recorder(self):
        """Create a notification manager with a recording channel."""
        config = NotificationConfig(
            channels={"test": {"enabled": True}},
            triggers={
                "escalation": ["test"],
                "task_failure": ["test"],
            },
        )
        manager = NotificationManager(config=config)
        recorder = RecordingChannel()
        manager.register_channel("test", recorder)
        return manager, recorder

    def _create_task(
        self,
        mutation_log,
        task_id="t1",
        name="Test task",
        done_when="do something",
        max_retries=3,
    ):
        mutation_log.append(
            "task_created",
            {
                "id": task_id,
                "name": name,
                "description": "Test",
                "role": "implementer",
                "depends_on": [],
                "done_when": done_when,
                "checklist": [],
                "context_bundle": [],
                "max_retries": max_retries,
            },
            reason="Test setup",
        )

    def test_escalation_sends_notification(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        notification_recorder,
        tmp_project,
    ):
        """Escalation (retries exhausted) triggers escalation notification."""
        from corc.dispatch import AgentResult
        from corc.processor import process_completed

        manager, recorder = notification_recorder
        self._create_task(mutation_log, max_retries=0)
        work_state.refresh()
        task = work_state.get_task("t1")

        result = AgentResult(output="error output", exit_code=1, duration_s=1.0)

        process_completed(
            task=task,
            result=result,
            attempt=1,  # > max_retries=0, triggers escalation
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
            notification_manager=manager,
        )

        # Should have fired escalation notification
        assert len(recorder.calls) == 1
        call = recorder.calls[0]
        assert call["event_type"] == "escalation"
        assert call["severity"] == SEVERITY_CRITICAL
        assert "Test task" in call["title"]

    def test_task_failure_sends_notification(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        notification_recorder,
        tmp_project,
    ):
        """Task failure (retriable) triggers task_failure notification."""
        from corc.dispatch import AgentResult
        from corc.processor import process_completed

        manager, recorder = notification_recorder
        self._create_task(mutation_log, max_retries=3)
        work_state.refresh()
        task = work_state.get_task("t1")

        result = AgentResult(output="crash output", exit_code=1, duration_s=1.0)

        process_completed(
            task=task,
            result=result,
            attempt=1,  # <= max_retries=3, retriable failure
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
            notification_manager=manager,
        )

        # Should have fired task_failure notification
        assert len(recorder.calls) == 1
        call = recorder.calls[0]
        assert call["event_type"] == "task_failure"
        assert call["severity"] == SEVERITY_ERROR
        assert "Test task" in call["title"]

    def test_no_notification_without_manager(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        tmp_project,
    ):
        """Without notification_manager, process_completed still works (no crash)."""
        from corc.dispatch import AgentResult
        from corc.processor import process_completed

        self._create_task(mutation_log, max_retries=0)
        work_state.refresh()
        task = work_state.get_task("t1")

        result = AgentResult(output="error", exit_code=1, duration_s=1.0)

        # Should not crash even without notification_manager
        proc_result = process_completed(
            task=task,
            result=result,
            attempt=1,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
            # no notification_manager — backward compatible
        )

        assert proc_result.passed is False

    def test_successful_task_no_notification(
        self,
        mutation_log,
        work_state,
        audit_log,
        session_logger,
        notification_recorder,
        tmp_project,
    ):
        """Successful task completion does NOT trigger failure/escalation notifications."""
        from corc.dispatch import AgentResult
        from corc.processor import process_completed

        manager, recorder = notification_recorder
        self._create_task(mutation_log)
        work_state.refresh()
        task = work_state.get_task("t1")

        result = AgentResult(output="success", exit_code=0, duration_s=1.0)

        process_completed(
            task=task,
            result=result,
            attempt=1,
            mutation_log=mutation_log,
            state=work_state,
            audit_log=audit_log,
            session_logger=session_logger,
            project_root=tmp_project,
            notification_manager=manager,
        )

        # No failure or escalation notification should have been sent
        assert len(recorder.calls) == 0
