"""Adaptive retry policy based on accumulated success/failure data.

Tracks first-attempt success rate by task type and role. Automatically
adjusts retry counts:
  - >90% first-attempt success rate => reduce retries (default 2 -> 1)
  - <50% first-attempt success rate => increase retries (default 2 -> 3)

Outcomes stored in JSONL (append-only, matching project conventions).
Adaptations logged to the audit log for operator review.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from corc.audit import AuditLog


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_RETRIES = 2
REDUCED_RETRIES = 1
INCREASED_RETRIES = 3

# Minimum number of first-attempt outcomes before adapting
MIN_SAMPLES = 5

# Thresholds
HIGH_SUCCESS_THRESHOLD = 0.90  # >90% => reduce
LOW_SUCCESS_THRESHOLD = 0.50  # <50% => increase


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TaskOutcome:
    """A single task attempt outcome."""

    task_type: str
    role: str
    attempt: int  # 1 = first attempt
    success: bool
    task_id: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type,
            "role": self.role,
            "attempt": self.attempt,
            "success": self.success,
            "task_id": self.task_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TaskOutcome:
        return cls(
            task_type=d.get("task_type", "general"),
            role=d.get("role", "unknown"),
            attempt=d.get("attempt", 1),
            success=d.get("success", False),
            task_id=d.get("task_id", ""),
            timestamp=d.get("timestamp", ""),
        )


@dataclass
class RetryStats:
    """Aggregated retry statistics for a (task_type, role) pair."""

    task_type: str
    role: str
    total_first_attempts: int = 0
    first_attempt_successes: int = 0
    total_attempts: int = 0
    total_successes: int = 0

    @property
    def first_attempt_success_rate(self) -> float:
        if self.total_first_attempts == 0:
            return 0.0
        return self.first_attempt_successes / self.total_first_attempts

    @property
    def adapted_retries(self) -> int:
        """Compute adapted retry count based on first-attempt success rate."""
        if self.total_first_attempts < MIN_SAMPLES:
            return DEFAULT_RETRIES
        rate = self.first_attempt_success_rate
        if rate > HIGH_SUCCESS_THRESHOLD:
            return REDUCED_RETRIES
        if rate < LOW_SUCCESS_THRESHOLD:
            return INCREASED_RETRIES
        return DEFAULT_RETRIES

    @property
    def flagged(self) -> bool:
        """Whether this task type is flagged for investigation."""
        if self.total_first_attempts < MIN_SAMPLES:
            return False
        return self.first_attempt_success_rate < LOW_SUCCESS_THRESHOLD


@dataclass
class AdaptiveRetryConfig:
    """Tunable knobs for adaptive retry behavior."""

    default_retries: int = DEFAULT_RETRIES
    reduced_retries: int = REDUCED_RETRIES
    increased_retries: int = INCREASED_RETRIES
    min_samples: int = MIN_SAMPLES
    high_success_threshold: float = HIGH_SUCCESS_THRESHOLD
    low_success_threshold: float = LOW_SUCCESS_THRESHOLD


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class AdaptiveRetryTracker:
    """Tracks task outcomes and computes adaptive retry counts.

    Outcomes are persisted to a JSONL file (append-only).
    """

    def __init__(
        self,
        data_path: Path,
        config: AdaptiveRetryConfig | None = None,
    ):
        self.data_path = Path(data_path)
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config or AdaptiveRetryConfig()

    def record_outcome(self, outcome: TaskOutcome) -> None:
        """Append a task outcome to the JSONL store."""
        if not outcome.timestamp:
            outcome.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        line = json.dumps(outcome.to_dict(), separators=(",", ":")) + "\n"
        with open(self.data_path, "a") as f:
            f.write(line)

    def read_outcomes(self) -> list[TaskOutcome]:
        """Read all stored outcomes."""
        if not self.data_path.exists():
            return []
        outcomes: list[TaskOutcome] = []
        with open(self.data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    outcomes.append(TaskOutcome.from_dict(json.loads(line)))
        return outcomes

    def compute_stats(self) -> dict[tuple[str, str], RetryStats]:
        """Compute per-(task_type, role) statistics from all outcomes."""
        outcomes = self.read_outcomes()
        return self._compute_stats_from(outcomes)

    def _compute_stats_from(
        self, outcomes: list[TaskOutcome]
    ) -> dict[tuple[str, str], RetryStats]:
        stats: dict[tuple[str, str], RetryStats] = {}
        for o in outcomes:
            key = (o.task_type, o.role)
            if key not in stats:
                stats[key] = RetryStats(task_type=o.task_type, role=o.role)
            s = stats[key]
            s.total_attempts += 1
            if o.success:
                s.total_successes += 1
            if o.attempt == 1:
                s.total_first_attempts += 1
                if o.success:
                    s.first_attempt_successes += 1
        return stats

    def get_adaptive_max_retries(
        self,
        task_type: str,
        role: str,
        audit_log: AuditLog | None = None,
    ) -> int:
        """Return the adaptive retry count for a given task_type + role.

        If audit_log is provided and the adaptation differs from default,
        logs a retry_adaptation event.
        """
        stats_map = self.compute_stats()
        key = (task_type, role)
        stats = stats_map.get(key)

        if stats is None or stats.total_first_attempts < self.config.min_samples:
            return self.config.default_retries

        rate = stats.first_attempt_success_rate
        if rate > self.config.high_success_threshold:
            adapted = self.config.reduced_retries
            reason = (
                f"High success rate ({rate:.1%}) — reduced retries "
                f"{self.config.default_retries} -> {adapted}"
            )
        elif rate < self.config.low_success_threshold:
            adapted = self.config.increased_retries
            reason = (
                f"Low success rate ({rate:.1%}) — increased retries "
                f"{self.config.default_retries} -> {adapted}"
            )
        else:
            return self.config.default_retries

        if audit_log is not None:
            audit_log.log(
                "retry_adaptation",
                task_type=task_type,
                role=role,
                first_attempt_success_rate=round(rate, 4),
                sample_count=stats.total_first_attempts,
                default_retries=self.config.default_retries,
                adapted_retries=adapted,
                reason=reason,
            )

        return adapted

    def get_flagged_task_types(self) -> list[RetryStats]:
        """Return stats for task types flagged for investigation (<50% success)."""
        stats_map = self.compute_stats()
        return [s for s in stats_map.values() if s.flagged]


# ---------------------------------------------------------------------------
# Analysis helpers (used by corc analyze retries)
# ---------------------------------------------------------------------------


def compute_retry_statistics(tracker: AdaptiveRetryTracker) -> dict:
    """Compute full retry statistics for display.

    Returns a dict with:
      - stats: list of RetryStats (sorted by success rate ascending)
      - flagged: list of RetryStats flagged for investigation
      - total_outcomes: int
      - total_first_attempts: int
    """
    stats_map = tracker.compute_stats()
    all_stats = sorted(
        stats_map.values(),
        key=lambda s: s.first_attempt_success_rate,
    )
    flagged = [s for s in all_stats if s.flagged]
    total_outcomes = sum(s.total_attempts for s in all_stats)
    total_first = sum(s.total_first_attempts for s in all_stats)

    return {
        "stats": all_stats,
        "flagged": flagged,
        "total_outcomes": total_outcomes,
        "total_first_attempts": total_first,
    }


def format_retry_statistics(report: dict) -> str:
    """Format retry statistics into human-readable text."""
    lines = ["Adaptive Retry Statistics", "=" * 25]

    all_stats: list[RetryStats] = report["stats"]
    flagged: list[RetryStats] = report["flagged"]
    total_outcomes: int = report["total_outcomes"]
    total_first: int = report["total_first_attempts"]

    if not all_stats:
        lines.append("No retry outcome data found.")
        return "\n".join(lines)

    lines.append(f"Total outcomes: {total_outcomes}  First attempts: {total_first}")
    lines.append("")

    # Table header
    lines.append(
        f"  {'Task Type':<20} {'Role':<15} {'1st Att':>7} "
        f"{'Succ':>5} {'Rate':>7} {'Retries':>8} {'Status':>10}"
    )
    lines.append("  " + "-" * 78)

    for s in all_stats:
        rate = s.first_attempt_success_rate
        rate_str = f"{rate:.1%}" if s.total_first_attempts > 0 else "N/A"
        adapted = s.adapted_retries
        status = ""
        if s.flagged:
            status = "FLAGGED"
        elif s.total_first_attempts >= MIN_SAMPLES and rate > HIGH_SUCCESS_THRESHOLD:
            status = "reduced"
        elif s.total_first_attempts >= MIN_SAMPLES:
            status = "default"
        else:
            status = f"<{MIN_SAMPLES} samples"

        lines.append(
            f"  {s.task_type:<20} {s.role:<15} {s.total_first_attempts:>7} "
            f"{s.first_attempt_successes:>5} {rate_str:>7} "
            f"{adapted:>8} {status:>10}"
        )

    if flagged:
        lines.append("")
        lines.append("Flagged for investigation (<50% first-attempt success):")
        for s in flagged:
            lines.append(
                f"  - {s.task_type} / {s.role}: "
                f"{s.first_attempt_success_rate:.1%} "
                f"({s.total_first_attempts} first attempts)"
            )

    return "\n".join(lines)
