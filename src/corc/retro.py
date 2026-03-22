"""Project retrospective — structured analysis of a complete project's execution.

Aggregates task data, ratings, costs, durations, and findings across all tasks
in a project to produce a structured retrospective document. The result is
saved to the knowledge store as a task-outcome document.
"""

import time
from dataclasses import dataclass, field

from corc.analyze import (
    aggregate_costs,
    CostBreakdown,
    DurationEntry,
)
from corc.audit import AuditLog
from corc.knowledge import KnowledgeStore
from corc.rating import (
    DIMENSION_NAMES,
    DIMENSIONS,
    Rating,
    RatingStore,
    weighted_score,
)
from corc.state import WorkState


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Retrospective:
    """A structured project retrospective."""

    project_name: str
    generated_at: str

    # Task summary
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    escalated_tasks: int = 0

    # Cost and duration
    total_cost_usd: float = 0.0
    cost_estimate_usd: float | None = None
    total_duration_s: float = 0.0
    avg_duration_s: float = 0.0

    # Quality
    avg_overall_rating: float = 0.0
    dimension_averages: dict[str, float] = field(default_factory=dict)
    best_dimensions: list[str] = field(default_factory=list)
    worst_dimensions: list[str] = field(default_factory=list)
    quality_trend: str = ""  # "improving", "declining", "stable", "insufficient_data"

    # Findings
    what_went_well: list[str] = field(default_factory=list)
    what_didnt_go_well: list[str] = field(default_factory=list)
    top_findings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # Per-task ratings for detail
    task_ratings: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def _collect_project_tasks(work_state: WorkState, project_name: str) -> list[dict]:
    """Collect all tasks that belong to the given project.

    Tasks are matched by name prefix convention: tasks whose name starts with
    the project name (case-insensitive), or tasks whose description mentions
    the project name. If no tasks match by name/description, returns all tasks
    (assumes single-project usage).
    """
    all_tasks = work_state.list_tasks()
    project_lower = project_name.lower()

    # Try matching by name prefix or description
    matched = []
    for t in all_tasks:
        name = (t.get("name") or "").lower()
        desc = (t.get("description") or "").lower()
        if project_lower in name or project_lower in desc:
            matched.append(t)

    # If no matches by name/description, return all tasks
    if not matched:
        matched = all_tasks

    return matched


def _collect_project_ratings(
    rating_store: RatingStore, task_ids: set[str]
) -> list[Rating]:
    """Collect ratings for the given task IDs."""
    all_ratings = rating_store.read_all()
    return [r for r in all_ratings if r.task_id in task_ids]


def _collect_project_events(audit_log: AuditLog, task_ids: set[str]) -> list[dict]:
    """Collect audit events for the given task IDs."""
    all_events = audit_log.read_all()
    # Include events that match any project task, plus untagged cost events
    return [e for e in all_events if e.get("task_id") in task_ids]


def _compute_quality_trend(ratings: list[Rating]) -> str:
    """Determine quality trend from chronological ratings.

    Compares the average of the first half to the second half.
    """
    if len(ratings) < 4:
        return "insufficient_data"

    sorted_ratings = sorted(ratings, key=lambda r: r.timestamp)
    mid = len(sorted_ratings) // 2
    first_half = sorted_ratings[:mid]
    second_half = sorted_ratings[mid:]

    avg_first = sum(r.overall for r in first_half) / len(first_half)
    avg_second = sum(r.overall for r in second_half) / len(second_half)

    diff = avg_second - avg_first
    if diff > 0.5:
        return "improving"
    elif diff < -0.5:
        return "declining"
    return "stable"


def _extract_findings(tasks: list[dict]) -> list[str]:
    """Extract and deduplicate findings from all tasks."""
    findings: list[str] = []
    for t in tasks:
        task_findings = t.get("findings", [])
        if isinstance(task_findings, str):
            import json

            try:
                task_findings = json.loads(task_findings)
            except (json.JSONDecodeError, TypeError):
                task_findings = []
        for f in task_findings:
            if isinstance(f, dict):
                content = f.get("content", f.get("finding", str(f)))
            else:
                content = str(f)
            if content and content not in findings:
                findings.append(content)
    return findings


def _identify_what_went_well(
    tasks: list[dict],
    ratings: list[Rating],
    dimension_avgs: dict[str, float],
) -> list[str]:
    """Identify positive aspects of the project."""
    items: list[str] = []

    completed = [t for t in tasks if t.get("status") == "completed"]
    total = len(tasks)

    if total > 0:
        completion_rate = len(completed) / total
        if completion_rate >= 0.9:
            items.append(
                f"High completion rate: {len(completed)}/{total} tasks completed "
                f"({completion_rate:.0%})"
            )

    # First-attempt successes
    first_attempt = [t for t in completed if t.get("attempt_count", 1) <= 1]
    if first_attempt and completed:
        rate = len(first_attempt) / len(completed)
        if rate >= 0.7:
            items.append(
                f"Strong first-attempt success: {len(first_attempt)}/{len(completed)} "
                f"tasks completed on first try ({rate:.0%})"
            )

    # High-scoring dimensions
    for dim, avg in sorted(dimension_avgs.items(), key=lambda x: -x[1]):
        if avg >= 8.0:
            items.append(f"Excellent {dim}: averaged {avg:.1f}/10")

    # High-rated tasks
    high_rated = [r for r in ratings if r.overall >= 8.5]
    if high_rated:
        names = [r.task_name for r in high_rated[:3]]
        items.append(f"Top-performing tasks: {', '.join(names)}")

    if not items:
        items.append("No standout positive patterns identified")

    return items


def _identify_what_didnt_go_well(
    tasks: list[dict],
    ratings: list[Rating],
    dimension_avgs: dict[str, float],
) -> list[str]:
    """Identify areas that need improvement."""
    items: list[str] = []

    failed = [t for t in tasks if t.get("status") == "failed"]
    escalated = [t for t in tasks if t.get("status") == "escalated"]

    if failed:
        items.append(
            f"{len(failed)} task(s) failed: "
            + ", ".join(t.get("name", t["id"]) for t in failed[:5])
        )

    if escalated:
        items.append(
            f"{len(escalated)} task(s) required escalation: "
            + ", ".join(t.get("name", t["id"]) for t in escalated[:5])
        )

    # Multi-attempt tasks
    multi_attempt = [
        t
        for t in tasks
        if t.get("attempt_count", 0) > 1 and t.get("status") == "completed"
    ]
    if multi_attempt:
        items.append(f"{len(multi_attempt)} task(s) required multiple attempts")

    # Low-scoring dimensions
    for dim, avg in sorted(dimension_avgs.items(), key=lambda x: x[1]):
        if avg < 6.0:
            items.append(f"Low {dim}: averaged {avg:.1f}/10")

    # Low-rated tasks
    low_rated = [r for r in ratings if r.overall < 5.0]
    if low_rated:
        names = [r.task_name for r in low_rated[:3]]
        items.append(f"Lowest-performing tasks: {', '.join(names)}")

    if not items:
        items.append("No major issues identified")

    return items


def _generate_recommendations(
    retro: Retrospective,
    tasks: list[dict],
    ratings: list[Rating],
) -> list[str]:
    """Generate actionable recommendations for the next project."""
    recs: list[str] = []

    # Recommendation based on completion rate
    if retro.total_tasks > 0:
        completion_rate = retro.completed_tasks / retro.total_tasks
        if completion_rate < 0.8:
            recs.append(
                "Improve task decomposition: completion rate was "
                f"{completion_rate:.0%}. Consider smaller, more focused tasks."
            )

    # Recommendation based on quality trend
    if retro.quality_trend == "declining":
        recs.append(
            "Quality declined over the project. Consider adding review gates "
            "or scout phases for later tasks."
        )
    elif retro.quality_trend == "improving":
        recs.append(
            "Quality improved over time. Current practices are working — "
            "continue iterating on prompts and processes."
        )

    # Recommendation based on worst dimensions
    for dim in retro.worst_dimensions[:2]:
        avg = retro.dimension_averages.get(dim, 0)
        if avg < 7.0:
            desc = DIMENSIONS.get(dim, {}).get("description", dim)
            recs.append(f"Focus on improving {dim} (avg {avg:.1f}/10): {desc}")

    # Recommendation based on escalation rate
    if retro.escalated_tasks > 0 and retro.total_tasks > 0:
        esc_rate = retro.escalated_tasks / retro.total_tasks
        if esc_rate > 0.2:
            recs.append(
                f"High escalation rate ({esc_rate:.0%}). Review task "
                "complexity and consider adding more context to prompts."
            )

    # Recommendation based on multi-attempt tasks
    multi_attempt = [t for t in tasks if t.get("attempt_count", 0) > 2]
    if multi_attempt:
        recs.append(
            f"{len(multi_attempt)} task(s) needed 3+ attempts. "
            "Consider adding scout phases for complex tasks."
        )

    # Cost recommendation
    if retro.cost_estimate_usd and retro.total_cost_usd > retro.cost_estimate_usd * 1.2:
        overage = retro.total_cost_usd - retro.cost_estimate_usd
        recs.append(
            f"Project went ${overage:.2f} over budget. "
            "Review cost estimation methodology."
        )

    if not recs:
        recs.append(
            "Project executed well overall. Continue current practices "
            "and look for incremental improvements."
        )

    return recs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_retrospective(
    project_name: str,
    work_state: WorkState,
    audit_log: AuditLog,
    rating_store: RatingStore,
    cost_estimate_usd: float | None = None,
) -> Retrospective:
    """Generate a structured project retrospective.

    Args:
        project_name: Name of the project to analyze.
        work_state: WorkState instance for task data.
        audit_log: AuditLog instance for cost/event data.
        rating_store: RatingStore instance for quality ratings.
        cost_estimate_usd: Optional cost estimate for comparison.

    Returns:
        Retrospective dataclass with all analysis results.
    """
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Collect project data
    tasks = _collect_project_tasks(work_state, project_name)
    task_ids = {t["id"] for t in tasks}
    ratings = _collect_project_ratings(rating_store, task_ids)
    events = _collect_project_events(audit_log, task_ids)

    # Task counts
    completed = [t for t in tasks if t.get("status") == "completed"]
    failed = [t for t in tasks if t.get("status") == "failed"]
    escalated = [t for t in tasks if t.get("status") == "escalated"]

    # Cost analysis
    cost_breakdown = aggregate_costs(events)

    # Duration analysis
    dispatch_events = [
        e
        for e in events
        if e.get("event_type") == "task_dispatch_complete" and e.get("duration_s")
    ]
    durations = [float(e["duration_s"]) for e in dispatch_events]
    total_duration = sum(durations)
    avg_duration = total_duration / len(durations) if durations else 0.0

    # Rating analysis
    dimension_avgs: dict[str, float] = {}
    if ratings:
        for dim in DIMENSION_NAMES:
            scores = [r.scores[dim] for r in ratings if dim in r.scores]
            if scores:
                dimension_avgs[dim] = sum(scores) / len(scores)

    avg_overall = sum(r.overall for r in ratings) / len(ratings) if ratings else 0.0

    # Best and worst dimensions
    sorted_dims = sorted(dimension_avgs.items(), key=lambda x: x[1])
    worst_dims = [d[0] for d in sorted_dims[:3]] if sorted_dims else []
    best_dims = [d[0] for d in sorted_dims[-3:]] if sorted_dims else []
    # Reverse best_dims so highest is first
    best_dims = list(reversed(best_dims))

    # Quality trend
    quality_trend = _compute_quality_trend(ratings)

    # Findings
    top_findings = _extract_findings(tasks)

    # Build per-task rating detail
    task_rating_details = []
    for r in sorted(ratings, key=lambda x: x.overall, reverse=True):
        task_rating_details.append(
            {
                "task_id": r.task_id,
                "task_name": r.task_name,
                "overall": r.overall,
                "scores": r.scores,
                "method": r.method,
                "flags": r.flags,
            }
        )

    # Build the retrospective
    retro = Retrospective(
        project_name=project_name,
        generated_at=now,
        total_tasks=len(tasks),
        completed_tasks=len(completed),
        failed_tasks=len(failed),
        escalated_tasks=len(escalated),
        total_cost_usd=cost_breakdown.total_usd,
        cost_estimate_usd=cost_estimate_usd,
        total_duration_s=total_duration,
        avg_duration_s=avg_duration,
        avg_overall_rating=round(avg_overall, 2),
        dimension_averages={k: round(v, 2) for k, v in dimension_avgs.items()},
        best_dimensions=best_dims,
        worst_dimensions=worst_dims,
        quality_trend=quality_trend,
        top_findings=top_findings[:10],  # Cap at 10
        task_ratings=task_rating_details,
    )

    # Generate what went well / didn't
    retro.what_went_well = _identify_what_went_well(tasks, ratings, dimension_avgs)
    retro.what_didnt_go_well = _identify_what_didnt_go_well(
        tasks, ratings, dimension_avgs
    )
    retro.recommendations = _generate_recommendations(retro, tasks, ratings)

    return retro


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_retrospective(retro: Retrospective) -> str:
    """Format a Retrospective into human-readable CLI output."""
    lines: list[str] = []
    title = f"Project Retrospective: {retro.project_name}"
    lines.append(title)
    lines.append("=" * len(title))
    lines.append(f"Generated: {retro.generated_at}")
    lines.append("")

    # --- Task Summary ---
    lines.append("Task Summary")
    lines.append("------------")
    lines.append(f"  Total:     {retro.total_tasks}")
    lines.append(f"  Completed: {retro.completed_tasks}")
    lines.append(f"  Failed:    {retro.failed_tasks}")
    lines.append(f"  Escalated: {retro.escalated_tasks}")
    if retro.total_tasks > 0:
        rate = retro.completed_tasks / retro.total_tasks
        lines.append(f"  Completion rate: {rate:.0%}")
    lines.append("")

    # --- Cost vs Estimate ---
    lines.append("Cost & Duration")
    lines.append("---------------")
    lines.append(f"  Total cost:     ${retro.total_cost_usd:.2f}")
    if retro.cost_estimate_usd is not None:
        lines.append(f"  Cost estimate:  ${retro.cost_estimate_usd:.2f}")
        diff = retro.total_cost_usd - retro.cost_estimate_usd
        pct = (diff / retro.cost_estimate_usd * 100) if retro.cost_estimate_usd else 0
        direction = "over" if diff > 0 else "under"
        lines.append(
            f"  Variance:       ${abs(diff):.2f} {direction} ({abs(pct):.0f}%)"
        )
    lines.append(f"  Total duration: {retro.total_duration_s:.1f}s")
    lines.append(f"  Avg duration:   {retro.avg_duration_s:.1f}s")
    lines.append("")

    # --- Quality ---
    lines.append("Quality")
    lines.append("-------")
    lines.append(f"  Overall rating: {retro.avg_overall_rating:.1f}/10.0")
    lines.append(f"  Quality trend:  {retro.quality_trend}")
    if retro.dimension_averages:
        lines.append("")
        lines.append("  Dimension averages:")
        for dim in DIMENSION_NAMES:
            avg = retro.dimension_averages.get(dim)
            if avg is not None:
                bar = "█" * round(avg) + "░" * (10 - round(avg))
                lines.append(f"    {dim:<20} {bar} {avg:.1f}/10")

    if retro.best_dimensions:
        lines.append(f"\n  Best:  {', '.join(retro.best_dimensions)}")
    if retro.worst_dimensions:
        lines.append(f"  Worst: {', '.join(retro.worst_dimensions)}")
    lines.append("")

    # --- What Went Well ---
    lines.append("What Went Well")
    lines.append("--------------")
    for item in retro.what_went_well:
        lines.append(f"  + {item}")
    lines.append("")

    # --- What Didn't Go Well ---
    lines.append("What Didn't Go Well")
    lines.append("-------------------")
    for item in retro.what_didnt_go_well:
        lines.append(f"  - {item}")
    lines.append("")

    # --- Top Findings ---
    if retro.top_findings:
        lines.append("Top Findings")
        lines.append("------------")
        for i, finding in enumerate(retro.top_findings, 1):
            lines.append(f"  {i}. {finding}")
        lines.append("")

    # --- Recommendations ---
    lines.append("Recommendations")
    lines.append("---------------")
    for i, rec in enumerate(retro.recommendations, 1):
        lines.append(f"  {i}. {rec}")

    return "\n".join(lines)


def retrospective_to_markdown(retro: Retrospective) -> str:
    """Convert a Retrospective to a markdown document with YAML frontmatter.

    Suitable for saving to the knowledge store as a task-outcome document.
    """
    lines: list[str] = []

    # YAML frontmatter
    lines.append("---")
    lines.append(f"type: task-outcome")
    lines.append(f"project: {retro.project_name}")
    lines.append(f'title: "Retrospective: {retro.project_name}"')
    lines.append(f"created: {retro.generated_at}")
    lines.append(f"updated: {retro.generated_at}")
    lines.append(f"status: active")
    lines.append(f"source: system")
    lines.append(f"tags:")
    lines.append(f"  - retrospective")
    lines.append(f"  - project-review")
    lines.append("---")
    lines.append("")

    # Title
    lines.append(f"# Retrospective: {retro.project_name}")
    lines.append("")
    lines.append(f"Generated: {retro.generated_at}")
    lines.append("")

    # Task Summary
    lines.append("## Task Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total tasks | {retro.total_tasks} |")
    lines.append(f"| Completed | {retro.completed_tasks} |")
    lines.append(f"| Failed | {retro.failed_tasks} |")
    lines.append(f"| Escalated | {retro.escalated_tasks} |")
    if retro.total_tasks > 0:
        rate = retro.completed_tasks / retro.total_tasks
        lines.append(f"| Completion rate | {rate:.0%} |")
    lines.append("")

    # Cost vs Estimate
    lines.append("## Cost & Duration")
    lines.append("")
    lines.append(f"- **Total cost**: ${retro.total_cost_usd:.2f}")
    if retro.cost_estimate_usd is not None:
        diff = retro.total_cost_usd - retro.cost_estimate_usd
        pct = (diff / retro.cost_estimate_usd * 100) if retro.cost_estimate_usd else 0
        direction = "over" if diff > 0 else "under"
        lines.append(f"- **Cost estimate**: ${retro.cost_estimate_usd:.2f}")
        lines.append(f"- **Variance**: ${abs(diff):.2f} {direction} ({abs(pct):.0f}%)")
    lines.append(f"- **Total duration**: {retro.total_duration_s:.1f}s")
    lines.append(f"- **Avg task duration**: {retro.avg_duration_s:.1f}s")
    lines.append("")

    # Quality
    lines.append("## Quality")
    lines.append("")
    lines.append(f"- **Overall rating**: {retro.avg_overall_rating:.1f}/10.0")
    lines.append(f"- **Quality trend**: {retro.quality_trend}")
    if retro.best_dimensions:
        lines.append(f"- **Best dimensions**: {', '.join(retro.best_dimensions)}")
    if retro.worst_dimensions:
        lines.append(f"- **Weakest dimensions**: {', '.join(retro.worst_dimensions)}")
    lines.append("")

    if retro.dimension_averages:
        lines.append("### Dimension Scores")
        lines.append("")
        lines.append("| Dimension | Average |")
        lines.append("|-----------|---------|")
        for dim in DIMENSION_NAMES:
            avg = retro.dimension_averages.get(dim)
            if avg is not None:
                lines.append(f"| {dim} | {avg:.1f}/10 |")
        lines.append("")

    # What Went Well
    lines.append("## What Went Well")
    lines.append("")
    for item in retro.what_went_well:
        lines.append(f"- {item}")
    lines.append("")

    # What Didn't Go Well
    lines.append("## What Didn't Go Well")
    lines.append("")
    for item in retro.what_didnt_go_well:
        lines.append(f"- {item}")
    lines.append("")

    # Top Findings
    if retro.top_findings:
        lines.append("## Top Findings")
        lines.append("")
        for i, finding in enumerate(retro.top_findings, 1):
            lines.append(f"{i}. {finding}")
        lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    for i, rec in enumerate(retro.recommendations, 1):
        lines.append(f"{i}. {rec}")
    lines.append("")

    # Task Ratings Detail
    if retro.task_ratings:
        lines.append("## Task Ratings Detail")
        lines.append("")
        lines.append("| Task | Overall | Flags |")
        lines.append("|------|---------|-------|")
        for tr in retro.task_ratings:
            flags = ", ".join(tr.get("flags", [])) or "none"
            lines.append(f"| {tr['task_name']} | {tr['overall']:.1f}/10 | {flags} |")
        lines.append("")

    return "\n".join(lines)


def save_retrospective(
    retro: Retrospective,
    knowledge_store: KnowledgeStore,
) -> str:
    """Save a retrospective to the knowledge store as a task-outcome document.

    Returns the document ID in the knowledge store.
    """
    markdown = retrospective_to_markdown(retro)
    doc_id = knowledge_store.add(
        content=markdown,
        doc_type="task-outcome",
        project=retro.project_name,
        tags=["retrospective", "project-review"],
    )
    return doc_id
