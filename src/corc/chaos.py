"""Chaos Monkey — built-in resilience testing.

When enabled, randomly kills agent processes mid-task and corrupts
intermediate state files. The daemon should recover cleanly from every
chaos event. Configuration is stored in `.corc/chaos.json`.

Usage:
    corc chaos enable [--kill-rate 0.1] [--corrupt-rate 0.05]
    corc chaos disable
    corc chaos status
"""

import json
import os
import random
import signal
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Configuration (file-based, like pause.py)
# ---------------------------------------------------------------------------

_CONFIG_FILE = "chaos.json"
_EVENTS_FILE = "chaos_events.jsonl"


@dataclass
class ChaosConfig:
    """Chaos monkey settings."""

    enabled: bool = False
    kill_rate: float = 0.1  # Probability of killing an agent per tick
    corrupt_rate: float = 0.05  # Probability of corrupting a state file per tick
    seed: int | None = None  # Optional RNG seed for reproducibility

    def validate(self) -> list[str]:
        """Return list of validation errors (empty if valid)."""
        errors = []
        if not 0.0 <= self.kill_rate <= 1.0:
            errors.append(f"kill_rate must be 0.0–1.0, got {self.kill_rate}")
        if not 0.0 <= self.corrupt_rate <= 1.0:
            errors.append(f"corrupt_rate must be 0.0–1.0, got {self.corrupt_rate}")
        return errors


def _config_path(corc_dir: Path) -> Path:
    return Path(corc_dir) / _CONFIG_FILE


def _events_path(corc_dir: Path) -> Path:
    return Path(corc_dir) / _EVENTS_FILE


def write_chaos_config(corc_dir: Path, config: ChaosConfig) -> ChaosConfig:
    """Write chaos config to disk. Returns the config written."""
    corc_dir = Path(corc_dir)
    corc_dir.mkdir(parents=True, exist_ok=True)
    _config_path(corc_dir).write_text(json.dumps(asdict(config), indent=2) + "\n")
    return config


def read_chaos_config(corc_dir: Path) -> ChaosConfig:
    """Read chaos config from disk. Returns defaults if missing/corrupt."""
    path = _config_path(Path(corc_dir))
    if not path.exists():
        return ChaosConfig()
    try:
        data = json.loads(path.read_text())
        return ChaosConfig(
            enabled=data.get("enabled", False),
            kill_rate=data.get("kill_rate", 0.1),
            corrupt_rate=data.get("corrupt_rate", 0.05),
            seed=data.get("seed"),
        )
    except (json.JSONDecodeError, OSError):
        return ChaosConfig()


def remove_chaos_config(corc_dir: Path) -> bool:
    """Remove chaos config file. Returns True if removed."""
    path = _config_path(Path(corc_dir))
    if path.exists():
        path.unlink()
        return True
    return False


def is_chaos_enabled(corc_dir: Path) -> bool:
    """Check if chaos mode is active."""
    return read_chaos_config(corc_dir).enabled


# ---------------------------------------------------------------------------
# Event tracking
# ---------------------------------------------------------------------------


@dataclass
class ChaosEvent:
    """A single chaos event."""

    timestamp: str
    event_type: str  # "agent_killed", "state_corrupted"
    task_id: str | None = None
    details: dict = field(default_factory=dict)
    recovered: bool | None = None  # Set later by daemon


def _append_event(corc_dir: Path, event: ChaosEvent) -> None:
    """Append a chaos event to the events log."""
    path = _events_path(Path(corc_dir))
    line = json.dumps(asdict(event), separators=(",", ":")) + "\n"
    with open(path, "a") as f:
        f.write(line)


def read_chaos_events(corc_dir: Path) -> list[dict]:
    """Read all chaos events from the log."""
    path = _events_path(Path(corc_dir))
    if not path.exists():
        return []
    events = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def mark_event_recovered(corc_dir: Path, task_id: str) -> None:
    """Mark the most recent unrecovered event for a task as recovered.

    Rewrites the events file to update the recovered flag.
    """
    events = read_chaos_events(corc_dir)
    updated = False
    for event in reversed(events):
        if event.get("task_id") == task_id and event.get("recovered") is None:
            event["recovered"] = True
            updated = True
            break

    if updated:
        path = _events_path(Path(corc_dir))
        with open(path, "w") as f:
            for event in events:
                f.write(json.dumps(event, separators=(",", ":")) + "\n")


def get_chaos_stats(corc_dir: Path) -> dict:
    """Compute recovery statistics from chaos events."""
    events = read_chaos_events(corc_dir)
    total = len(events)
    recovered = sum(1 for e in events if e.get("recovered") is True)
    failed = sum(1 for e in events if e.get("recovered") is False)
    pending = sum(1 for e in events if e.get("recovered") is None)

    kills = sum(1 for e in events if e.get("event_type") == "agent_killed")
    corruptions = sum(1 for e in events if e.get("event_type") == "state_corrupted")

    recovery_rate = (recovered / total * 100) if total > 0 else 0.0

    return {
        "total_events": total,
        "recovered": recovered,
        "failed": failed,
        "pending": pending,
        "kills": kills,
        "corruptions": corruptions,
        "recovery_rate": recovery_rate,
    }


# ---------------------------------------------------------------------------
# ChaosMonkey — the engine
# ---------------------------------------------------------------------------


class ChaosMonkey:
    """Randomly injects failures into agent processes and state files.

    Designed to be called from the daemon's _tick() loop. Each method
    rolls dice against the configured rate and, if triggered, performs
    the chaos action and logs the event.
    """

    def __init__(
        self,
        corc_dir: Path,
        config: ChaosConfig | None = None,
        rng: random.Random | None = None,
        kill_fn: Callable[[int, int], None] | None = None,
    ):
        self.corc_dir = Path(corc_dir)
        self.config = config or read_chaos_config(self.corc_dir)
        self._rng = rng or random.Random(self.config.seed)
        # Allow injecting a custom kill function for testing
        self._kill_fn = kill_fn or os.kill

    def reload_config(self) -> None:
        """Re-read config from disk (hot reload)."""
        self.config = read_chaos_config(self.corc_dir)

    # ------------------------------------------------------------------
    # Agent killing
    # ------------------------------------------------------------------

    def maybe_kill_agent(self, pid: int, task_id: str | None = None) -> bool:
        """With probability kill_rate, send SIGKILL to the given PID.

        Returns True if the agent was killed.
        """
        if not self.config.enabled:
            return False
        if self._rng.random() >= self.config.kill_rate:
            return False

        try:
            self._kill_fn(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            # Process already dead or not killable — still counts as event
            pass

        event = ChaosEvent(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            event_type="agent_killed",
            task_id=task_id,
            details={"pid": pid},
        )
        _append_event(self.corc_dir, event)
        return True

    # ------------------------------------------------------------------
    # State corruption
    # ------------------------------------------------------------------

    def maybe_corrupt_state(
        self,
        state_path: Path,
        task_id: str | None = None,
    ) -> bool:
        """With probability corrupt_rate, inject garbage into a state file.

        Corrupts by truncating the file to a random fraction of its size
        and appending garbage bytes. The mutation log replay mechanism
        should recover from this.

        Returns True if corruption was injected.
        """
        if not self.config.enabled:
            return False
        if self._rng.random() >= self.config.corrupt_rate:
            return False

        state_path = Path(state_path)
        if not state_path.exists():
            return False

        original_size = state_path.stat().st_size
        if original_size == 0:
            return False

        # Truncate to a random fraction and append garbage
        truncate_at = self._rng.randint(0, max(1, original_size - 1))
        garbage = bytes(self._rng.getrandbits(8) for _ in range(32))

        with open(state_path, "r+b") as f:
            f.seek(truncate_at)
            f.truncate()
            f.write(garbage)

        event = ChaosEvent(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            event_type="state_corrupted",
            task_id=task_id,
            details={
                "file": str(state_path),
                "original_size": original_size,
                "truncated_at": truncate_at,
            },
        )
        _append_event(self.corc_dir, event)
        return True

    # ------------------------------------------------------------------
    # Tick integration — call from daemon._tick()
    # ------------------------------------------------------------------

    def tick(
        self,
        running_agents: list[dict],
        state_path: Path | None = None,
    ) -> list[ChaosEvent]:
        """Run chaos checks for one daemon tick.

        Args:
            running_agents: List of agent dicts with 'pid' and 'task_id'.
            state_path: Path to the SQLite state DB (for corruption).

        Returns:
            List of chaos events that fired this tick.
        """
        if not self.config.enabled:
            return []

        events_fired: list[ChaosEvent] = []

        # Try killing each running agent
        for agent in running_agents:
            pid = agent.get("pid")
            task_id = agent.get("task_id")
            if pid and self.maybe_kill_agent(pid, task_id):
                events_fired.append(
                    ChaosEvent(
                        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        event_type="agent_killed",
                        task_id=task_id,
                        details={"pid": pid},
                    )
                )

        # Try corrupting the state DB
        if state_path and self.maybe_corrupt_state(state_path):
            events_fired.append(
                ChaosEvent(
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    event_type="state_corrupted",
                    details={"file": str(state_path)},
                )
            )

        return events_fired
