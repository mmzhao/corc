"""Modular notification system — terminal first, pluggable for external channels.

Notification channels are configured in .corc/config.yaml:

    notifications:
      channels:
        terminal:
          enabled: true
        slack:
          enabled: false
          webhook_url: null
        discord:
          enabled: false
          webhook_url: null

      triggers:
        escalation: [terminal]
        task_failure: [terminal]
        cost_threshold: [terminal]
        pause: [terminal]

Each channel implements the NotificationChannel interface. The NotificationManager
loads configuration and routes events to the appropriate channels.
"""

import copy
import json
import sys
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from corc.config import load_config


# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_ERROR = "error"
SEVERITY_CRITICAL = "critical"


# ---------------------------------------------------------------------------
# NotificationChannel interface
# ---------------------------------------------------------------------------


class NotificationChannel(ABC):
    """Base interface for notification backends.

    Implement this interface to add new notification channels (e.g. email,
    Telegram, PagerDuty). Each channel must implement send() which returns
    True on success, False on failure.
    """

    @abstractmethod
    def send(self, event_type: str, title: str, body: str, severity: str) -> bool:
        """Send a notification.

        Args:
            event_type: The event that triggered this notification
                        (e.g. "escalation", "task_failure").
            title: Short summary of the notification.
            body: Detailed notification body.
            severity: One of "info", "warning", "error", "critical".

        Returns:
            True if the notification was sent successfully, False otherwise.
        """
        ...


# ---------------------------------------------------------------------------
# Terminal notification
# ---------------------------------------------------------------------------


class TerminalNotification(NotificationChannel):
    """Prints notifications to stderr with severity-based formatting.

    Uses ANSI color codes when writing to a real terminal, plain text otherwise.
    """

    # ANSI color codes for severity levels
    _COLORS = {
        SEVERITY_INFO: "\033[36m",  # cyan
        SEVERITY_WARNING: "\033[33m",  # yellow
        SEVERITY_ERROR: "\033[31m",  # red
        SEVERITY_CRITICAL: "\033[1;31m",  # bold red
    }
    _RESET = "\033[0m"

    def __init__(self, stream=None, use_color: bool | None = None):
        self._stream = stream or sys.stderr
        if use_color is not None:
            self._use_color = use_color
        else:
            self._use_color = hasattr(self._stream, "isatty") and self._stream.isatty()

    def send(self, event_type: str, title: str, body: str, severity: str) -> bool:
        """Print notification to terminal (stderr)."""
        tag = severity.upper()
        label = f"[CORC {tag}]"

        if self._use_color:
            color = self._COLORS.get(severity, "")
            label = f"{color}{label}{self._RESET}"

        lines = [f"{label} {title}"]
        if body:
            # Indent body lines for readability
            for line in body.strip().splitlines():
                lines.append(f"  {line}")

        output = "\n".join(lines) + "\n"
        try:
            self._stream.write(output)
            self._stream.flush()
            return True
        except OSError:
            return False


# ---------------------------------------------------------------------------
# Slack webhook notification
# ---------------------------------------------------------------------------


class SlackWebhookNotification(NotificationChannel):
    """Sends notifications via Slack incoming webhook.

    Requires a webhook_url from https://api.slack.com/messaging/webhooks.
    Uses the Slack Block Kit format for rich notifications.
    """

    _SEVERITY_EMOJI = {
        SEVERITY_INFO: ":information_source:",
        SEVERITY_WARNING: ":warning:",
        SEVERITY_ERROR: ":x:",
        SEVERITY_CRITICAL: ":rotating_light:",
    }

    def __init__(self, webhook_url: str, timeout: float = 10.0):
        self.webhook_url = webhook_url
        self.timeout = timeout

    def send(self, event_type: str, title: str, body: str, severity: str) -> bool:
        """Send notification to Slack via incoming webhook."""
        emoji = self._SEVERITY_EMOJI.get(severity, ":bell:")
        payload = {
            "text": f"{emoji} *{title}*\n{body}",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{title}",
                        "emoji": True,
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"{emoji} *{severity.upper()}* | `{event_type}`\n\n{body}",
                    },
                },
            ],
        }

        return self._post(payload)

    def _post(self, payload: dict) -> bool:
        """POST JSON payload to the webhook URL."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
            return False


# ---------------------------------------------------------------------------
# Discord webhook notification
# ---------------------------------------------------------------------------


class DiscordWebhookNotification(NotificationChannel):
    """Sends notifications via Discord webhook.

    Requires a webhook_url from Discord channel settings > Integrations > Webhooks.
    Uses Discord embed format for rich notifications.
    """

    _SEVERITY_COLOR = {
        SEVERITY_INFO: 0x3498DB,  # blue
        SEVERITY_WARNING: 0xF39C12,  # yellow/orange
        SEVERITY_ERROR: 0xE74C3C,  # red
        SEVERITY_CRITICAL: 0x992D22,  # dark red
    }

    def __init__(self, webhook_url: str, timeout: float = 10.0):
        self.webhook_url = webhook_url
        self.timeout = timeout

    def send(self, event_type: str, title: str, body: str, severity: str) -> bool:
        """Send notification to Discord via webhook."""
        color = self._SEVERITY_COLOR.get(severity, 0x95A5A6)
        payload = {
            "embeds": [
                {
                    "title": title,
                    "description": body,
                    "color": color,
                    "fields": [
                        {"name": "Event", "value": event_type, "inline": True},
                        {"name": "Severity", "value": severity.upper(), "inline": True},
                    ],
                    "footer": {"text": "CORC Notification System"},
                }
            ]
        }

        return self._post(payload)

    def _post(self, payload: dict) -> bool:
        """POST JSON payload to the Discord webhook URL."""
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
            return False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class NotificationConfig:
    """Parsed notification configuration from .corc/config.yaml."""

    channels: dict[str, dict] = field(default_factory=dict)
    triggers: dict[str, list[str]] = field(default_factory=dict)


# Default configuration matching the spec
_DEFAULT_CONFIG = {
    "channels": {
        "terminal": {"enabled": True},
        "slack": {"enabled": False, "webhook_url": None},
        "discord": {"enabled": False, "webhook_url": None},
    },
    "triggers": {
        "escalation": ["terminal"],
        "task_failure": ["terminal"],
        "cost_threshold": ["terminal"],
        "pause": ["terminal"],
    },
}


def load_notification_config(corc_dir: Path) -> NotificationConfig:
    """Load notification configuration via centralized config.

    Returns defaults if the config file doesn't exist or the notifications
    section is missing.
    """
    root = Path(corc_dir).parent
    cfg = load_config(root)
    channels = cfg.get("notifications.channels") or copy.deepcopy(
        _DEFAULT_CONFIG["channels"]
    )
    triggers = cfg.get("notifications.triggers") or copy.deepcopy(
        _DEFAULT_CONFIG["triggers"]
    )
    # Normalize null trigger lists to empty lists (YAML null → [])
    for event, channel_list in triggers.items():
        if channel_list is None:
            triggers[event] = []
    return NotificationConfig(channels=channels, triggers=triggers)


# ---------------------------------------------------------------------------
# NotificationManager — routes events to channels
# ---------------------------------------------------------------------------


class NotificationManager:
    """Routes notification events to configured channels.

    Loads configuration from .corc/config.yaml and instantiates the
    appropriate NotificationChannel objects. On each notify() call,
    looks up which channels are configured for the event type and
    sends to each one.
    """

    def __init__(
        self, corc_dir: Path | None = None, config: NotificationConfig | None = None
    ):
        """Initialize the notification manager.

        Args:
            corc_dir: Path to .corc directory. Used to load config if config is None.
            config: Pre-loaded config (useful for testing). Overrides corc_dir loading.
        """
        if config is not None:
            self._config = config
        elif corc_dir is not None:
            self._config = load_notification_config(corc_dir)
        else:
            self._config = NotificationConfig(
                channels=copy.deepcopy(_DEFAULT_CONFIG["channels"]),
                triggers=copy.deepcopy(_DEFAULT_CONFIG["triggers"]),
            )

        self._channels: dict[str, NotificationChannel] = {}
        self._init_channels()

    def _init_channels(self):
        """Instantiate enabled notification channels from config."""
        for name, conf in self._config.channels.items():
            if not conf.get("enabled", False):
                continue

            channel = self._create_channel(name, conf)
            if channel is not None:
                self._channels[name] = channel

    def _create_channel(self, name: str, conf: dict) -> NotificationChannel | None:
        """Create a channel instance by name.

        Override or extend this to add custom channel types.
        """
        if name == "terminal":
            return TerminalNotification()
        elif name == "slack":
            url = conf.get("webhook_url")
            if url:
                return SlackWebhookNotification(webhook_url=url)
            return None
        elif name == "discord":
            url = conf.get("webhook_url")
            if url:
                return DiscordWebhookNotification(webhook_url=url)
            return None
        return None

    def register_channel(self, name: str, channel: NotificationChannel):
        """Register a custom notification channel.

        Use this to add channels not built into the config system
        (e.g. for plugins or testing).
        """
        self._channels[name] = channel

    def notify(
        self, event_type: str, title: str, body: str, severity: str = SEVERITY_INFO
    ) -> dict[str, bool]:
        """Send a notification to all channels configured for this event type.

        Args:
            event_type: The event that triggered this notification.
            title: Short summary.
            body: Detailed body.
            severity: Severity level.

        Returns:
            Dict mapping channel name to success/failure boolean.
        """
        target_channels = self._config.triggers.get(event_type, [])
        results: dict[str, bool] = {}

        for channel_name in target_channels:
            channel = self._channels.get(channel_name)
            if channel is None:
                results[channel_name] = False
                continue
            try:
                results[channel_name] = channel.send(event_type, title, body, severity)
            except Exception:
                results[channel_name] = False

        return results

    @property
    def active_channels(self) -> list[str]:
        """Return names of currently active (enabled + initialized) channels."""
        return list(self._channels.keys())

    @property
    def config(self) -> NotificationConfig:
        """Return the current notification configuration."""
        return self._config


# ---------------------------------------------------------------------------
# Convenience helpers for common notification events
# ---------------------------------------------------------------------------


def notify_escalation(
    manager: NotificationManager, task: dict, escalation: dict
) -> dict[str, bool]:
    """Send escalation notification."""
    task_name = task.get("name", "unknown")
    task_id = task.get("id", "unknown")
    attempts = escalation.get("attempts", "?")
    error = escalation.get("error", "no details")

    title = f"Task escalated: {task_name} ({task_id})"
    body = (
        f"Retries exhausted after {attempts} attempts.\n"
        f"Error: {error}\n"
        f"Run: corc escalations"
    )
    return manager.notify("escalation", title, body, severity=SEVERITY_CRITICAL)


def notify_task_failure(
    manager: NotificationManager, task: dict, attempt: int, error: str
) -> dict[str, bool]:
    """Send task failure notification."""
    task_name = task.get("name", "unknown")
    task_id = task.get("id", "unknown")

    title = f"Task failed: {task_name} ({task_id})"
    body = f"Attempt {attempt} failed.\nError: {error}"
    return manager.notify("task_failure", title, body, severity=SEVERITY_ERROR)


def notify_cost_threshold(
    manager: NotificationManager, alert_type: str, current: float, limit: float
) -> dict[str, bool]:
    """Send cost threshold notification."""
    title = f"Cost threshold exceeded: {alert_type}"
    body = f"Current: ${current:.2f} | Limit: ${limit:.2f}"
    return manager.notify("cost_threshold", title, body, severity=SEVERITY_WARNING)


def notify_pause(
    manager: NotificationManager, reason: str, source: str
) -> dict[str, bool]:
    """Send pause notification."""
    title = "System paused"
    body = f"Reason: {reason}\nSource: {source}"
    return manager.notify("pause", title, body, severity=SEVERITY_WARNING)


def notify_pr_awaiting_human_merge(
    manager: NotificationManager, task: dict, pr_info
) -> dict[str, bool]:
    """Send notification that a PR is awaiting human merge.

    Used for human-only repos where the agent creates the PR but
    only a human can merge it.
    """
    task_name = task.get("name", "unknown")
    task_id = task.get("id", "unknown")
    pr_url = pr_info.url if pr_info else "N/A"
    pr_number = pr_info.number if pr_info else "?"

    title = f"PR #{pr_number} awaiting human merge: {task_name}"
    body = (
        f"Task: {task_name} ({task_id})\n"
        f"PR: {pr_url}\n"
        f"Validation passed. Merge policy is human-only.\n"
        f"Please review and merge the PR manually."
    )
    return manager.notify("pr_awaiting_merge", title, body, severity=SEVERITY_WARNING)
