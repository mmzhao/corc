"""Planning self-improvement — feedback loop from execution outcomes to planning.

Records PlanningOutcome after each `corc rate`, then assembles a Planning
Feedback section for the planner system prompt. The feedback is deterministically
computed from:

- Retrospective highlights (what went well / poorly)
- Pattern analysis (role/task-type correlations)
- Done-when calibration (empirical success rates by specificity)
- Context bundle effectiveness
- Recent failures
- Operator-curated planning lessons (.corc/planning_lessons.md)

The feedback section is capped at ~3000 chars so it fits comfortably in the
planner's system prompt without crowding out other context.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from corc.rating import Rating, RatingStore, DIMENSION_NAMES


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

MAX_FEEDBACK_CHARS = 3000


@dataclass
class PlanningOutcome:
    """Captures planning inputs vs execution outcomes for a single task.

    Recorded after `corc rate` scores a task so we can learn what planning
    decisions led to good or bad outcomes.
    """

    task_id: str
    task_name: str
    timestamp: str

    # Planning inputs
    role: str = ""
    task_type: str = "implementation"
    done_when: str = ""
    done_when_length: int = 0
    checklist_size: int = 0
    dependency_count: int = 0
    context_bundle_size: int = 0
    context_bundle_files: list[str] = field(default_factory=list)

    # Execution outcomes
    overall_score: float = 0.0
    dimension_scores: dict[str, int] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    attempt_count: int = 1
    status: str = "completed"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlanningOutcome":
        return cls(
            task_id=d["task_id"],
            task_name=d["task_name"],
            timestamp=d["timestamp"],
            role=d.get("role", ""),
            task_type=d.get("task_type", "implementation"),
            done_when=d.get("done_when", ""),
            done_when_length=d.get("done_when_length", 0),
            checklist_size=d.get("checklist_size", 0),
            dependency_count=d.get("dependency_count", 0),
            context_bundle_size=d.get("context_bundle_size", 0),
            context_bundle_files=d.get("context_bundle_files", []),
            overall_score=d.get("overall_score", 0.0),
            dimension_scores=d.get("dimension_scores", {}),
            flags=d.get("flags", []),
            attempt_count=d.get("attempt_count", 1),
            status=d.get("status", "completed"),
        )


# ---------------------------------------------------------------------------
# Feedback store — append-only JSONL
# ---------------------------------------------------------------------------


class PlanningFeedbackStore:
    """Append-only JSONL storage for planning outcomes."""

    def __init__(self, feedback_path: Path):
        """Initialize with the path to the JSONL file (not directory)."""
        self.feedback_path = Path(feedback_path)
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, outcome: PlanningOutcome) -> None:
        """Append a planning outcome to the JSONL file."""
        line = json.dumps(outcome.to_dict(), separators=(",", ":")) + "\n"
        with open(self.feedback_path, "a") as f:
            f.write(line)

    def read_all(self) -> list[PlanningOutcome]:
        """Read all planning outcomes from the JSONL file."""
        if not self.feedback_path.exists():
            return []
        outcomes: list[PlanningOutcome] = []
        with open(self.feedback_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        outcomes.append(PlanningOutcome.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        continue
        return outcomes

    def read_recent(self, n: int = 20) -> list[PlanningOutcome]:
        """Return the last N outcomes sorted by timestamp."""
        outcomes = self.read_all()
        outcomes.sort(key=lambda o: o.timestamp)
        return outcomes[-n:]


# ---------------------------------------------------------------------------
# Record planning outcome (called after `corc rate`)
# ---------------------------------------------------------------------------


def record_planning_outcome(
    task: dict,
    rating: Rating,
    feedback_store: PlanningFeedbackStore,
) -> PlanningOutcome:
    """Create and persist a PlanningOutcome from a task + its rating.

    Called after `corc rate` scores a task so we capture how planning
    inputs correlated with execution outcomes.
    """
    checklist = task.get("checklist", [])
    if isinstance(checklist, str):
        try:
            checklist = json.loads(checklist)
        except (json.JSONDecodeError, TypeError, ValueError):
            checklist = []

    deps = task.get("depends_on", [])
    if isinstance(deps, str):
        try:
            deps = json.loads(deps)
        except (json.JSONDecodeError, TypeError, ValueError):
            deps = []

    bundle = task.get("context_bundle", [])
    if isinstance(bundle, str):
        try:
            bundle = json.loads(bundle)
        except (json.JSONDecodeError, TypeError, ValueError):
            bundle = []

    done_when = task.get("done_when", "")

    outcome = PlanningOutcome(
        task_id=rating.task_id,
        task_name=rating.task_name,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        role=task.get("role", "unknown"),
        task_type=task.get("task_type", "implementation"),
        done_when=done_when,
        done_when_length=len(done_when),
        checklist_size=len(checklist) if isinstance(checklist, list) else 0,
        dependency_count=len(deps) if isinstance(deps, list) else 0,
        context_bundle_size=len(bundle) if isinstance(bundle, list) else 0,
        context_bundle_files=bundle if isinstance(bundle, list) else [],
        overall_score=rating.overall,
        dimension_scores=rating.scores,
        flags=rating.flags,
        attempt_count=task.get("attempt_count", 1),
        status=task.get("status", "completed"),
    )

    feedback_store.save(outcome)
    return outcome


# ---------------------------------------------------------------------------
# Feedback section builder — deterministic digest for planner prompt
# ---------------------------------------------------------------------------


def _done_when_calibration(outcomes: list[PlanningOutcome]) -> list[str]:
    """Compute empirical success rates by done-when specificity.

    Buckets: vague (<30 chars), moderate (30-100), specific (100+).
    """
    buckets: dict[str, list[float]] = {
        "vague (<30 chars)": [],
        "moderate (30-100 chars)": [],
        "specific (100+ chars)": [],
    }
    for o in outcomes:
        length = o.done_when_length
        if length < 30:
            key = "vague (<30 chars)"
        elif length < 100:
            key = "moderate (30-100 chars)"
        else:
            key = "specific (100+ chars)"
        buckets[key].append(o.overall_score)

    lines: list[str] = []
    for bucket, scores in buckets.items():
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        success = sum(1 for s in scores if s >= 7.0)
        rate = success / len(scores)
        lines.append(
            f"  {bucket}: avg {avg:.1f}/10, {rate:.0%} success (n={len(scores)})"
        )
    return lines


def _context_bundle_effectiveness(outcomes: list[PlanningOutcome]) -> list[str]:
    """Analyze context bundle size vs outcome quality."""
    buckets: dict[str, list[float]] = {
        "none (0)": [],
        "small (1-3)": [],
        "medium (4-7)": [],
        "large (8+)": [],
    }
    for o in outcomes:
        size = o.context_bundle_size
        if size == 0:
            key = "none (0)"
        elif size <= 3:
            key = "small (1-3)"
        elif size <= 7:
            key = "medium (4-7)"
        else:
            key = "large (8+)"
        buckets[key].append(o.overall_score)

    lines: list[str] = []
    for bucket, scores in buckets.items():
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        lines.append(f"  {bucket} files: avg {avg:.1f}/10 (n={len(scores)})")
    return lines


def _recent_failures(outcomes: list[PlanningOutcome], max_items: int = 5) -> list[str]:
    """Extract recent failures (score < 6.0) with their planning inputs."""
    failures = [o for o in outcomes if o.overall_score < 6.0]
    # Sort by timestamp descending
    failures.sort(key=lambda o: o.timestamp, reverse=True)
    lines: list[str] = []
    for o in failures[:max_items]:
        flags_str = ", ".join(o.flags[:3]) if o.flags else "none"
        lines.append(
            f"  {o.task_name} ({o.overall_score:.1f}/10): "
            f"role={o.role}, type={o.task_type}, flags=[{flags_str}]"
        )
    return lines


def _pattern_highlights(outcomes: list[PlanningOutcome]) -> list[str]:
    """Extract key pattern highlights from outcomes."""
    if not outcomes:
        return []

    lines: list[str] = []

    # Best/worst roles
    role_scores: dict[str, list[float]] = {}
    for o in outcomes:
        role_scores.setdefault(o.role, []).append(o.overall_score)

    for role, scores in sorted(role_scores.items()):
        if len(scores) >= 3:
            avg = sum(scores) / len(scores)
            if avg >= 8.0:
                lines.append(
                    f"  {role}: strong performer (avg {avg:.1f}/10, n={len(scores)})"
                )
            elif avg < 6.0:
                lines.append(
                    f"  {role}: underperforming (avg {avg:.1f}/10, n={len(scores)})"
                )

    # Best/worst task types
    type_scores: dict[str, list[float]] = {}
    for o in outcomes:
        type_scores.setdefault(o.task_type, []).append(o.overall_score)

    for tt, scores in sorted(type_scores.items()):
        if len(scores) >= 3:
            avg = sum(scores) / len(scores)
            if avg >= 8.0:
                lines.append(
                    f"  {tt} tasks: strong (avg {avg:.1f}/10, n={len(scores)})"
                )
            elif avg < 6.0:
                lines.append(f"  {tt} tasks: weak (avg {avg:.1f}/10, n={len(scores)})")

    return lines


def _retro_highlights(outcomes: list[PlanningOutcome]) -> list[str]:
    """Extract retrospective-style highlights from recent outcomes."""
    if not outcomes:
        return []

    recent = sorted(outcomes, key=lambda o: o.timestamp)[-20:]
    lines: list[str] = []

    # Overall trend
    if len(recent) >= 4:
        mid = len(recent) // 2
        first_avg = sum(o.overall_score for o in recent[:mid]) / mid
        second_avg = sum(o.overall_score for o in recent[mid:]) / (len(recent) - mid)
        diff = second_avg - first_avg
        if diff > 0.5:
            lines.append(
                f"  Quality trend: IMPROVING ({first_avg:.1f} -> {second_avg:.1f})"
            )
        elif diff < -0.5:
            lines.append(
                f"  Quality trend: DECLINING ({first_avg:.1f} -> {second_avg:.1f})"
            )
        else:
            lines.append(
                f"  Quality trend: stable (~{(first_avg + second_avg) / 2:.1f})"
            )

    # Overall stats
    all_scores = [o.overall_score for o in recent]
    avg_all = sum(all_scores) / len(all_scores)
    success_count = sum(1 for s in all_scores if s >= 7.0)
    lines.append(
        f"  Recent avg: {avg_all:.1f}/10, "
        f"success rate: {success_count}/{len(recent)} ({success_count / len(recent):.0%})"
    )

    # Multi-attempt rate
    multi = sum(1 for o in recent if o.attempt_count > 1)
    if multi > 0:
        lines.append(
            f"  Multi-attempt tasks: {multi}/{len(recent)} ({multi / len(recent):.0%})"
        )

    return lines


def load_planning_lessons(corc_dir: Path) -> str:
    """Load operator-curated planning lessons from .corc/planning_lessons.md.

    Returns the content if the file exists and is non-empty, otherwise "".
    """
    lessons_path = corc_dir / "planning_lessons.md"
    if lessons_path.exists():
        content = lessons_path.read_text().strip()
        return content
    return ""


def build_planning_feedback_section(
    feedback_store: PlanningFeedbackStore,
    corc_dir: Path,
    max_chars: int = MAX_FEEDBACK_CHARS,
) -> str:
    """Build the Planning Feedback section for the planner system prompt.

    Deterministically computed from recorded outcomes and planning lessons.
    Capped at ~3000 chars with truncation.
    """
    outcomes = feedback_store.read_all()

    parts: list[str] = []
    parts.append("## Planning Feedback (Self-Improvement)")
    parts.append(
        "This section is auto-generated from execution outcomes. "
        "Use it to improve future planning decisions.\n"
    )

    # 1. Retrospective highlights
    if outcomes:
        retro = _retro_highlights(outcomes)
        if retro:
            parts.append("### Retrospective Highlights")
            parts.extend(retro)
            parts.append("")

    # 2. Pattern analysis
    if outcomes:
        patterns = _pattern_highlights(outcomes)
        if patterns:
            parts.append("### Pattern Analysis")
            parts.extend(patterns)
            parts.append("")

    # 3. Done-when calibration
    if outcomes:
        calibration = _done_when_calibration(outcomes)
        if calibration:
            parts.append("### Done-When Calibration")
            parts.extend(calibration)
            parts.append("")

    # 4. Context bundle effectiveness
    if outcomes:
        ctx_eff = _context_bundle_effectiveness(outcomes)
        if ctx_eff:
            parts.append("### Context Bundle Effectiveness")
            parts.extend(ctx_eff)
            parts.append("")

    # 5. Recent failures
    if outcomes:
        failures = _recent_failures(outcomes)
        if failures:
            parts.append("### Recent Failures (score < 6.0)")
            parts.extend(failures)
            parts.append("")

    # 6. Planning lessons (human-curated)
    lessons = load_planning_lessons(corc_dir)
    if lessons:
        parts.append("### Operator Planning Lessons")
        parts.append(lessons)
        parts.append("")

    # If no outcomes and no lessons, still provide a stub
    if not outcomes and not lessons:
        parts.append(
            "No planning outcomes recorded yet. Run `corc rate` after completing tasks to build the feedback loop."
        )
        return "\n".join(parts)

    # Assemble and truncate
    section = "\n".join(parts)
    if len(section) > max_chars:
        # Truncate from the bottom, preserving header + retro + patterns
        section = section[: max_chars - 50] + "\n\n... (truncated to ~3000 chars)"

    return section
