"""Rating engine — scores completed workflow runs across 7 dimensions.

Uses a separate claude -p evaluator invocation with the spec as rubric.
Ratings stored as JSONL in data/ratings/.

Philosophy: critical, not generous. A "10" means flawless execution — rare.
"""

import json
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Scoring dimensions (from SPEC.md §10)
# ---------------------------------------------------------------------------

DIMENSIONS = {
    "correctness": {
        "weight": 0.25,
        "description": "Did the output meet the 'done when' criteria?",
        "source": "Automated (tests) + Claude evaluation",
    },
    "completeness": {
        "weight": 0.15,
        "description": "Were all requirements addressed?",
        "source": "Automated (checklist)",
    },
    "code-quality": {
        "weight": 0.15,
        "description": "Measurable code health metrics (lint, complexity, test coverage)",
        "source": "Automated (lint, complexity, test coverage)",
    },
    "efficiency": {
        "weight": 0.15,
        "description": "Resource usage relative to task complexity (tokens, cost, wall clock)",
        "source": "Automated (tokens, cost, wall clock time)",
    },
    "determinism": {
        "weight": 0.10,
        "description": "Did the agent follow the prescribed workflow? Any deviations?",
        "source": "Automated (event log analysis)",
    },
    "resilience": {
        "weight": 0.10,
        "description": "Did recovery mechanisms work when failures occurred?",
        "source": "Automated (chaos monkey results)",
    },
    "human-intervention": {
        "weight": 0.10,
        "description": "How many times did the operator need to step in?",
        "source": "Automated (escalation count)",
    },
}

DIMENSION_NAMES = list(DIMENSIONS.keys())


def weighted_score(scores: dict[str, int]) -> float:
    """Compute weighted overall score from dimension scores.

    Missing dimensions are excluded from the weighted average.
    Returns 0.0 if no valid scores are present.
    """
    total_weight = 0.0
    total = 0.0
    for dim, info in DIMENSIONS.items():
        if dim in scores:
            total += scores[dim] * info["weight"]
            total_weight += info["weight"]
    if total_weight == 0.0:
        return 0.0
    # Normalize to account for missing dimensions
    return total / total_weight


def flagged_dimensions(scores: dict[str, int], threshold: int = 7) -> list[str]:
    """Return dimension names that scored below the threshold."""
    return [dim for dim, score in scores.items() if score < threshold]


# ---------------------------------------------------------------------------
# Rating data model
# ---------------------------------------------------------------------------


@dataclass
class Rating:
    """A single task rating across all dimensions."""

    task_id: str
    task_name: str
    scores: dict[str, int]  # dimension -> score (1-10)
    overall: float  # weighted score
    flags: list[str]  # dimensions below threshold
    method: str  # "evaluator" or "heuristic"
    timestamp: str
    metadata: dict = field(default_factory=dict)  # extra info (cost, attempt, etc.)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Rating":
        return cls(
            task_id=d["task_id"],
            task_name=d["task_name"],
            scores=d["scores"],
            overall=d["overall"],
            flags=d["flags"],
            method=d["method"],
            timestamp=d["timestamp"],
            metadata=d.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Rating store — JSONL persistence
# ---------------------------------------------------------------------------


class RatingStore:
    """Append-only JSONL storage for ratings in data/ratings/."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _ratings_path(self) -> Path:
        return self.base_dir / "ratings.jsonl"

    def save(self, rating: Rating) -> None:
        """Append a rating to the JSONL file."""
        line = json.dumps(rating.to_dict(), separators=(",", ":")) + "\n"
        with open(self._ratings_path(), "a") as f:
            f.write(line)

    def read_all(self) -> list[Rating]:
        """Read all ratings from the JSONL file."""
        path = self._ratings_path()
        if not path.exists():
            return []
        ratings = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ratings.append(Rating.from_dict(json.loads(line)))
        return ratings

    def get_for_task(self, task_id: str) -> Rating | None:
        """Get the most recent rating for a task."""
        ratings = [r for r in self.read_all() if r.task_id == task_id]
        return ratings[-1] if ratings else None

    def is_rated(self, task_id: str) -> bool:
        """Check if a task has been rated."""
        return self.get_for_task(task_id) is not None

    def get_trend(self, last_n: int = 30) -> list[Rating]:
        """Return the last N ratings sorted by timestamp."""
        ratings = self.read_all()
        ratings.sort(key=lambda r: r.timestamp)
        return ratings[-last_n:]

    def get_by_dimension(self, dimension: str) -> list[dict]:
        """Return all ratings drill-down for a specific dimension.

        Returns list of dicts with task_id, task_name, score, timestamp.
        """
        if dimension not in DIMENSIONS:
            raise ValueError(
                f"Unknown dimension: {dimension}. Valid: {DIMENSION_NAMES}"
            )
        results = []
        for r in self.read_all():
            if dimension in r.scores:
                results.append(
                    {
                        "task_id": r.task_id,
                        "task_name": r.task_name,
                        "score": r.scores[dimension],
                        "overall": r.overall,
                        "timestamp": r.timestamp,
                    }
                )
        results.sort(key=lambda x: x["timestamp"])
        return results


# ---------------------------------------------------------------------------
# Heuristic scoring (no claude -p needed)
# ---------------------------------------------------------------------------


def heuristic_scores(
    task: dict,
    events: list[dict],
    session_entries: list[dict],
) -> dict[str, int]:
    """Compute scores from available data without an LLM evaluator.

    Uses task status, checklist progress, attempt count, event history,
    and session data to produce reasonable (but less nuanced) scores.
    """
    scores: dict[str, int] = {}

    # -- Correctness: based on task completion status
    status = task.get("status", "unknown")
    if status == "completed":
        scores["correctness"] = 8
    elif status == "failed":
        scores["correctness"] = 3
    else:
        scores["correctness"] = 5

    # -- Completeness: based on checklist progress
    checklist = task.get("checklist", [])
    if isinstance(checklist, str):
        try:
            checklist = json.loads(checklist)
        except (json.JSONDecodeError, TypeError):
            checklist = []
    if checklist:
        done_count = sum(
            1 for item in checklist if (isinstance(item, dict) and item.get("done"))
        )
        ratio = done_count / len(checklist) if checklist else 0
        scores["completeness"] = max(1, min(10, round(ratio * 10)))
    else:
        # No checklist — assume complete if task completed
        scores["completeness"] = 8 if status == "completed" else 5

    # -- Code quality: default mid-range (can't measure lint/coverage without tools)
    scores["code-quality"] = 7 if status == "completed" else 5

    # -- Efficiency: based on attempt count and event count
    attempt_count = task.get("attempt_count", 1)
    if attempt_count <= 1:
        scores["efficiency"] = 9
    elif attempt_count == 2:
        scores["efficiency"] = 7
    elif attempt_count == 3:
        scores["efficiency"] = 5
    else:
        scores["efficiency"] = 3

    # -- Determinism: based on deviations and tool call patterns
    deviations = task.get("micro_deviations", [])
    if isinstance(deviations, str):
        try:
            deviations = json.loads(deviations)
        except (json.JSONDecodeError, TypeError):
            deviations = []
    deviation_count = len(deviations)
    if deviation_count == 0:
        scores["determinism"] = 9
    elif deviation_count <= 2:
        scores["determinism"] = 7
    else:
        scores["determinism"] = 5

    # -- Resilience: based on failure/retry events
    failure_events = [e for e in events if e.get("event_type") == "task_failed"]
    retry_events = [e for e in events if e.get("event_type") == "step_retried"]
    if not failure_events and not retry_events:
        scores["resilience"] = 9
    elif len(failure_events) <= 1:
        scores["resilience"] = 7
    else:
        scores["resilience"] = 5

    # -- Human intervention: based on escalation events
    escalation_events = [
        e for e in events if e.get("event_type") in ("escalation", "task_escalated")
    ]
    if not escalation_events:
        scores["human-intervention"] = 10
    elif len(escalation_events) == 1:
        scores["human-intervention"] = 6
    else:
        scores["human-intervention"] = 3

    return scores


# ---------------------------------------------------------------------------
# Evaluator prompt building (for claude -p)
# ---------------------------------------------------------------------------


EVALUATOR_SYSTEM_PROMPT = """You are a critical evaluator scoring completed AI agent task runs.

You score across 7 dimensions, each 1-10 (integer). Be CRITICAL, not generous.
A "10" means flawless — rare. Most good work scores 7-8.

Dimensions:
1. correctness (weight 0.25): Did the output meet the "done when" criteria?
2. completeness (weight 0.15): Were all requirements addressed?
3. code-quality (weight 0.15): Code health — clean, well-structured, tested?
4. efficiency (weight 0.15): Resource usage relative to task complexity
5. determinism (weight 0.10): Did the agent follow the prescribed workflow?
6. resilience (weight 0.10): Did recovery mechanisms work on failures?
7. human-intervention (weight 0.10): How many times did the operator step in?

Respond with ONLY a JSON object, no other text:
{
  "correctness": <1-10>,
  "completeness": <1-10>,
  "code-quality": <1-10>,
  "efficiency": <1-10>,
  "determinism": <1-10>,
  "resilience": <1-10>,
  "human-intervention": <1-10>
}
"""


def build_evaluator_prompt(
    task: dict,
    events: list[dict],
    session_summary: str,
    spec_excerpt: str = "",
) -> str:
    """Build a prompt for the claude -p evaluator with task info and evidence."""
    parts = []

    parts.append("# Task Under Evaluation\n")
    parts.append(f"**Name**: {task.get('name', 'unknown')}")
    parts.append(f"**ID**: {task.get('id', 'unknown')}")
    parts.append(f"**Status**: {task.get('status', 'unknown')}")
    parts.append(f"**Done When**: {task.get('done_when', 'N/A')}")
    parts.append(f"**Role**: {task.get('role', 'unknown')}")
    parts.append(f"**Attempts**: {task.get('attempt_count', 1)}")

    # Checklist
    checklist = task.get("checklist", [])
    if isinstance(checklist, str):
        try:
            checklist = json.loads(checklist)
        except (json.JSONDecodeError, TypeError):
            checklist = []
    if checklist:
        parts.append("\n## Checklist")
        for item in checklist:
            if isinstance(item, dict):
                status = "✅" if item.get("done") else "☐"
                parts.append(f"  {status} {item.get('item', str(item))}")
            else:
                parts.append(f"  ☐ {item}")

    # Findings / deviations
    findings = task.get("findings", [])
    if isinstance(findings, str):
        try:
            findings = json.loads(findings)
        except (json.JSONDecodeError, TypeError):
            findings = []
    if findings:
        parts.append(f"\n## Findings\n{json.dumps(findings, indent=2)}")

    deviations = task.get("micro_deviations", [])
    if isinstance(deviations, str):
        try:
            deviations = json.loads(deviations)
        except (json.JSONDecodeError, TypeError):
            deviations = []
    if deviations:
        parts.append(f"\n## Micro-Deviations\n{json.dumps(deviations, indent=2)}")

    # PR URL
    if task.get("pr_url"):
        parts.append(f"\n**PR**: {task['pr_url']}")

    # Event summary
    if events:
        parts.append(f"\n## Event History ({len(events)} events)")
        failure_count = sum(1 for e in events if e.get("event_type") == "task_failed")
        escalation_count = sum(
            1 for e in events if e.get("event_type") in ("escalation", "task_escalated")
        )
        parts.append(f"- Failures: {failure_count}")
        parts.append(f"- Escalations: {escalation_count}")
        # Show last few events
        for e in events[-10:]:
            ts = e.get("timestamp", "")[:19]
            parts.append(f"  [{ts}] {e.get('event_type', '?')}")

    # Session summary
    if session_summary:
        parts.append(f"\n## Session Summary\n{session_summary[:5000]}")

    # Spec excerpt
    if spec_excerpt:
        parts.append(f"\n## Spec Rubric\n{spec_excerpt[:5000]}")

    parts.append(
        "\n---\nScore this task run across all 7 dimensions (1-10 each). Respond with ONLY a JSON object."
    )

    return "\n".join(parts)


def parse_evaluator_response(response: str) -> dict[str, int] | None:
    """Parse the evaluator's JSON response into dimension scores.

    Handles wrapped responses (```json ... ```) and validates ranges.
    Returns None if parsing fails.
    """
    text = response.strip()

    # Strip markdown code fences if present
    if "```" in text:
        # Extract content between code fences
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break

    # Find the JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None

    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None

    # Validate
    scores = {}
    for dim in DIMENSION_NAMES:
        if dim in data:
            try:
                val = int(data[dim])
                scores[dim] = max(1, min(10, val))
            except (ValueError, TypeError):
                pass

    # Must have at least some valid scores
    return scores if scores else None


# ---------------------------------------------------------------------------
# Rating engine
# ---------------------------------------------------------------------------


class RatingEngine:
    """Orchestrates scoring of completed task runs.

    Gathers task data, audit events, and session logs, then invokes either
    a claude -p evaluator or falls back to heuristic scoring.
    """

    def __init__(
        self,
        store: RatingStore,
        work_state,  # WorkState
        audit_log,  # AuditLog
        session_logger,  # SessionLogger
        spec_path: Path | None = None,
    ):
        self.store = store
        self.work_state = work_state
        self.audit_log = audit_log
        self.session_logger = session_logger
        self.spec_path = spec_path

    def _load_spec_excerpt(self) -> str:
        """Load relevant spec sections for the evaluator rubric."""
        if self.spec_path and self.spec_path.exists():
            text = self.spec_path.read_text()
            # Extract rating section if present
            marker = "## 10. Rating"
            idx = text.find(marker)
            if idx >= 0:
                end_marker = "\n## "
                end_idx = text.find(end_marker, idx + len(marker))
                if end_idx >= 0:
                    return text[idx:end_idx]
                return text[idx : idx + 5000]
            # Fallback: return first 3000 chars
            return text[:3000]
        return ""

    def _get_session_summary(self, task_id: str) -> str:
        """Build a summary of the agent's session for the evaluator."""
        latest_attempt = self.session_logger.get_latest_attempt(task_id)
        if latest_attempt == 0:
            return ""

        entries = self.session_logger.read_session(task_id, latest_attempt)
        if not entries:
            return ""

        # Summarize: count tool calls, extract final output
        tool_calls = [e for e in entries if e.get("type") == "stream_event"]
        output_entries = [e for e in entries if e.get("type") == "output"]

        lines = [f"Session attempt {latest_attempt}: {len(entries)} entries"]
        lines.append(f"  Tool call events: {len(tool_calls)}")

        if output_entries:
            last_output = output_entries[-1].get("content", "")
            lines.append(f"  Final output: {last_output[:1000]}")

        # Include dispatch info if available
        dispatch_entries = [e for e in entries if e.get("type") == "dispatch"]
        if dispatch_entries:
            d = dispatch_entries[0]
            lines.append(f"  Budget: ${d.get('budget_usd', '?')}")
            lines.append(f"  Tools: {d.get('tools', '?')}")

        return "\n".join(lines)

    def rate_task(self, task_id: str, use_claude: bool = True) -> Rating:
        """Score a completed task across all 7 dimensions.

        Args:
            task_id: The task to rate
            use_claude: If True, use claude -p evaluator; else use heuristics

        Returns:
            Rating object with scores

        Raises:
            ValueError: If task not found or not completed
        """
        task = self.work_state.get_task(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        if task["status"] != "completed":
            raise ValueError(
                f"Task {task_id} is '{task['status']}', not 'completed'. "
                "Only completed tasks can be rated."
            )

        events = self.audit_log.read_for_task(task_id)
        session_summary = self._get_session_summary(task_id)

        if use_claude:
            scores = self._evaluate_with_claude(task, events, session_summary)
            method = "evaluator"
        else:
            scores = None
            method = "heuristic"

        # Fallback to heuristics if evaluator fails
        if scores is None:
            session_entries = []
            latest = self.session_logger.get_latest_attempt(task_id)
            if latest > 0:
                session_entries = self.session_logger.read_session(task_id, latest)
            scores = heuristic_scores(task, events, session_entries)
            method = "heuristic"

        overall = weighted_score(scores)
        flags = flagged_dimensions(scores)

        rating = Rating(
            task_id=task_id,
            task_name=task.get("name", "unknown"),
            scores=scores,
            overall=round(overall, 2),
            flags=flags,
            method=method,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            metadata={
                "attempt_count": task.get("attempt_count", 1),
                "role": task.get("role", "unknown"),
            },
        )

        self.store.save(rating)
        return rating

    def _evaluate_with_claude(
        self,
        task: dict,
        events: list[dict],
        session_summary: str,
    ) -> dict[str, int] | None:
        """Run a claude -p evaluator invocation to score the task.

        Returns parsed scores or None if the evaluation fails.
        """
        spec_excerpt = self._load_spec_excerpt()
        prompt = build_evaluator_prompt(task, events, session_summary, spec_excerpt)

        try:
            result = subprocess.run(
                [
                    "claude",
                    "-p",
                    prompt,
                    "--system-prompt",
                    EVALUATOR_SYSTEM_PROMPT,
                    "--max-turns",
                    "1",
                    "--output-format",
                    "text",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                return None
            return parse_evaluator_response(result.stdout)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return None

    def rate_auto(self) -> list[Rating]:
        """Score all unscored completed tasks.

        Returns list of new ratings created.
        """
        all_tasks = self.work_state.list_tasks(status="completed")
        new_ratings = []

        for task in all_tasks:
            task_id = task["id"]
            if not self.store.is_rated(task_id):
                try:
                    rating = self.rate_task(task_id, use_claude=True)
                    new_ratings.append(rating)
                except ValueError:
                    # Task not in a ratable state, skip
                    pass

        return new_ratings


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def format_rating(rating: Rating) -> str:
    """Format a single rating for display."""
    lines = [
        f"Rating: {rating.task_name} ({rating.task_id})",
        f"Method: {rating.method}",
        f"Overall: {rating.overall:.1f}/10.0",
        f"Time: {rating.timestamp}",
        "",
        "Scores:",
    ]

    for dim in DIMENSION_NAMES:
        score = rating.scores.get(dim, 0)
        weight = DIMENSIONS[dim]["weight"]
        bar = "█" * score + "░" * (10 - score)
        flag = " ⚠" if dim in rating.flags else ""
        lines.append(f"  {dim:<20} {bar} {score:>2}/10 (w={weight}){flag}")

    if rating.flags:
        lines.append(f"\n⚠  Flagged: {', '.join(rating.flags)}")

    return "\n".join(lines)


def format_trend(ratings: list[Rating]) -> str:
    """Format a time-series view of ratings."""
    lines = ["Rating Trend", "============"]

    if not ratings:
        lines.append("No ratings found.")
        return "\n".join(lines)

    total_overall = sum(r.overall for r in ratings)
    avg_overall = total_overall / len(ratings)

    # Compute per-dimension averages
    dim_totals: dict[str, list[int]] = {d: [] for d in DIMENSION_NAMES}
    for r in ratings:
        for dim in DIMENSION_NAMES:
            if dim in r.scores:
                dim_totals[dim].append(r.scores[dim])

    lines.append(f"Ratings: {len(ratings)}  Avg overall: {avg_overall:.1f}/10.0")
    lines.append("")
    lines.append("Dimension averages:")
    for dim in DIMENSION_NAMES:
        vals = dim_totals[dim]
        if vals:
            avg = sum(vals) / len(vals)
            lines.append(f"  {dim:<20} {avg:.1f}/10.0 ({len(vals)} ratings)")
        else:
            lines.append(f"  {dim:<20} no data")

    lines.append("")
    lines.append("Recent ratings:")
    for r in ratings[-10:]:
        ts = r.timestamp[:19]
        lines.append(
            f"  [{ts}] {r.task_name:<30} overall={r.overall:.1f}  method={r.method}"
        )

    return "\n".join(lines)


def format_dimension_drilldown(dimension: str, entries: list[dict]) -> str:
    """Format drill-down for a specific dimension."""
    lines = [f"Dimension: {dimension}", "=" * (len(dimension) + 12)]

    if not entries:
        lines.append("No data for this dimension.")
        return "\n".join(lines)

    scores = [e["score"] for e in entries]
    avg = sum(scores) / len(scores)
    min_s = min(scores)
    max_s = max(scores)

    lines.append(f"Ratings: {len(entries)}  Avg: {avg:.1f}  Min: {min_s}  Max: {max_s}")
    lines.append(f"Weight: {DIMENSIONS[dimension]['weight']}")
    lines.append(f"Description: {DIMENSIONS[dimension]['description']}")
    lines.append("")

    for e in entries:
        ts = e["timestamp"][:19]
        bar = "█" * e["score"] + "░" * (10 - e["score"])
        lines.append(f"  [{ts}] {e['task_name']:<30} {bar} {e['score']:>2}/10")

    return "\n".join(lines)
