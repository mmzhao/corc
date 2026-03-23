"""Pattern analysis — automated detection from rating data.

Identifies what's working and what isn't across roles, task types,
context bundles, and prompt versions. Produces actionable recommendations.

Consumed by:
  corc analyze patterns   — correlations between roles/task-types/context-bundles and scores
  corc analyze prompts    — scores by prompt version per role
  corc analyze planning   — spec structure vs outcome analysis
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from corc.config import DEFAULTS
from corc.rating import Rating, RatingStore, DIMENSIONS, DIMENSION_NAMES


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GroupStats:
    """Statistics for a grouping key (role, task-type, context-bundle, etc.)."""

    key: str
    count: int = 0
    avg_overall: float = 0.0
    dimension_avgs: dict[str, float] = field(default_factory=dict)
    low_dimensions: list[str] = field(default_factory=list)  # dims with avg < threshold


@dataclass
class Correlation:
    """A detected correlation between a grouping factor and score."""

    group_type: str  # "role", "task_type", "context_bundle"
    group_key: str  # e.g. "implementer", "sql-migration"
    dimension: str  # e.g. "correctness"
    avg_score: float
    count: int
    direction: str  # "low" or "high"


@dataclass
class Recommendation:
    """An actionable recommendation from pattern analysis."""

    category: str  # "prompt", "workflow", "trust", "planning"
    severity: str  # "critical", "warning", "info"
    message: str
    evidence: str  # supporting data summary


@dataclass
class PatternReport:
    """Full pattern analysis report."""

    correlations: list[Correlation] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    by_role: list[GroupStats] = field(default_factory=list)
    by_task_type: list[GroupStats] = field(default_factory=list)
    by_context_bundle: list[GroupStats] = field(default_factory=list)
    trust_suggestions: list[Recommendation] = field(default_factory=list)
    total_ratings: int = 0


@dataclass
class PromptVersionStats:
    """Scores aggregated by prompt version for a role."""

    version: str
    count: int = 0
    avg_overall: float = 0.0
    dimension_avgs: dict[str, float] = field(default_factory=dict)


@dataclass
class PromptReport:
    """Prompt version analysis report for a specific role."""

    role: str
    versions: list[PromptVersionStats] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    total_ratings: int = 0


@dataclass
class PlanningStats:
    """Statistics for a planning factor (checklist size, dependency depth, etc.)."""

    factor: str
    bucket: str  # e.g. "small (1-3)", "medium (4-7)", "large (8+)"
    count: int = 0
    avg_overall: float = 0.0
    success_rate: float = 0.0  # fraction with overall >= 7.0


@dataclass
class PlanningReport:
    """Planning pattern analysis report."""

    by_checklist_size: list[PlanningStats] = field(default_factory=list)
    by_dependency_count: list[PlanningStats] = field(default_factory=list)
    by_context_size: list[PlanningStats] = field(default_factory=list)
    by_done_when_specificity: list[PlanningStats] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    total_ratings: int = 0


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

_PATTERN_DEFAULTS = DEFAULTS["patterns"]

LOW_SCORE_THRESHOLD = _PATTERN_DEFAULTS["low_score_threshold"]
HIGH_SCORE_THRESHOLD = _PATTERN_DEFAULTS["high_score_threshold"]
FLAG_THRESHOLD = _PATTERN_DEFAULTS["flag_threshold"]
MIN_SAMPLE_SIZE = _PATTERN_DEFAULTS["min_sample_size"]
TRUST_MIN_SAMPLE = _PATTERN_DEFAULTS["trust_min_sample"]


def _compute_dimension_avgs(ratings: list[Rating]) -> dict[str, float]:
    """Compute average score per dimension across ratings."""
    dim_totals: dict[str, list[float]] = {d: [] for d in DIMENSION_NAMES}
    for r in ratings:
        for dim in DIMENSION_NAMES:
            if dim in r.scores:
                dim_totals[dim].append(float(r.scores[dim]))

    avgs = {}
    for dim, vals in dim_totals.items():
        if vals:
            avgs[dim] = round(sum(vals) / len(vals), 1)
    return avgs


def _compute_avg_overall(ratings: list[Rating]) -> float:
    """Compute average overall score across ratings."""
    if not ratings:
        return 0.0
    return round(sum(r.overall for r in ratings) / len(ratings), 1)


def _group_ratings(ratings: list[Rating], key_fn) -> dict[str, list[Rating]]:
    """Group ratings by a key function that extracts the grouping value."""
    groups: dict[str, list[Rating]] = {}
    for r in ratings:
        key = key_fn(r)
        if key:
            groups.setdefault(key, []).append(r)
    return groups


def _compute_group_stats(
    groups: dict[str, list[Rating]],
) -> list[GroupStats]:
    """Compute statistics for each group."""
    stats = []
    for key, ratings in sorted(groups.items()):
        if len(ratings) < MIN_SAMPLE_SIZE:
            continue
        dim_avgs = _compute_dimension_avgs(ratings)
        avg_overall = _compute_avg_overall(ratings)
        low_dims = [d for d, avg in dim_avgs.items() if avg < FLAG_THRESHOLD]
        stats.append(
            GroupStats(
                key=key,
                count=len(ratings),
                avg_overall=avg_overall,
                dimension_avgs=dim_avgs,
                low_dimensions=low_dims,
            )
        )
    return stats


# ---------------------------------------------------------------------------
# Pattern analysis
# ---------------------------------------------------------------------------


def analyze_patterns(ratings: list[Rating]) -> PatternReport:
    """Identify correlations between roles/task-types/context-bundles and scores.

    Produces a report with correlations and actionable recommendations.
    """
    report = PatternReport(total_ratings=len(ratings))

    if not ratings:
        return report

    # --- Group by role ---
    role_groups = _group_ratings(ratings, lambda r: r.metadata.get("role", "unknown"))
    report.by_role = _compute_group_stats(role_groups)

    # --- Group by task type (from metadata.task_type or inferred from name) ---
    task_type_groups = _group_ratings(
        ratings, lambda r: r.metadata.get("task_type", "")
    )
    report.by_task_type = _compute_group_stats(task_type_groups)

    # --- Group by context bundle size category ---
    def _context_bundle_key(r: Rating) -> str:
        bundle = r.metadata.get("context_bundle", [])
        if isinstance(bundle, list):
            size = len(bundle)
        elif isinstance(bundle, str):
            size = len(bundle.split(",")) if bundle else 0
        else:
            return ""
        if size == 0:
            return ""
        if size <= 3:
            return "small (1-3 files)"
        if size <= 7:
            return "medium (4-7 files)"
        return "large (8+ files)"

    bundle_groups = _group_ratings(ratings, _context_bundle_key)
    report.by_context_bundle = _compute_group_stats(bundle_groups)

    # --- Find correlations ---
    all_groups = [
        ("role", report.by_role),
        ("task_type", report.by_task_type),
        ("context_bundle", report.by_context_bundle),
    ]

    for group_type, stats_list in all_groups:
        for stat in stats_list:
            for dim, avg in stat.dimension_avgs.items():
                if avg < LOW_SCORE_THRESHOLD:
                    report.correlations.append(
                        Correlation(
                            group_type=group_type,
                            group_key=stat.key,
                            dimension=dim,
                            avg_score=avg,
                            count=stat.count,
                            direction="low",
                        )
                    )
                elif avg >= HIGH_SCORE_THRESHOLD:
                    report.correlations.append(
                        Correlation(
                            group_type=group_type,
                            group_key=stat.key,
                            dimension=dim,
                            avg_score=avg,
                            count=stat.count,
                            direction="high",
                        )
                    )

    # --- Trust level suggestions ---
    report.trust_suggestions = _compute_trust_suggestions(role_groups)

    # --- Generate recommendations ---
    report.recommendations = _generate_pattern_recommendations(report)

    return report


def _compute_trust_suggestions(
    role_groups: dict[str, list[Rating]],
) -> list[Recommendation]:
    """Suggest trust level changes based on consistent score patterns.

    - Consistently >9 on a dimension across 20+ tasks → suggest raising trust
    - Scores drop below 5 → suggest lowering trust
    """
    suggestions: list[Recommendation] = []

    for role, ratings in role_groups.items():
        if len(ratings) < TRUST_MIN_SAMPLE:
            continue

        dim_avgs = _compute_dimension_avgs(ratings)
        for dim, avg in dim_avgs.items():
            if avg >= HIGH_SCORE_THRESHOLD:
                suggestions.append(
                    Recommendation(
                        category="trust",
                        severity="info",
                        message=(
                            f"Consider raising trust level for '{role}' on "
                            f"'{dim}': avg {avg:.1f}/10 across {len(ratings)} tasks."
                        ),
                        evidence=f"role={role} dim={dim} avg={avg:.1f} n={len(ratings)}",
                    )
                )
            elif avg < LOW_SCORE_THRESHOLD:
                suggestions.append(
                    Recommendation(
                        category="trust",
                        severity="warning",
                        message=(
                            f"Consider lowering trust level for '{role}' on "
                            f"'{dim}': avg {avg:.1f}/10 across {len(ratings)} tasks."
                        ),
                        evidence=f"role={role} dim={dim} avg={avg:.1f} n={len(ratings)}",
                    )
                )

    return suggestions


def _generate_pattern_recommendations(report: PatternReport) -> list[Recommendation]:
    """Generate actionable recommendations from pattern data."""
    recs: list[Recommendation] = []

    # Low-scoring roles
    for stat in report.by_role:
        if stat.avg_overall < LOW_SCORE_THRESHOLD:
            recs.append(
                Recommendation(
                    category="prompt",
                    severity="critical",
                    message=(
                        f"Role '{stat.key}' scores {stat.avg_overall:.1f} avg overall "
                        f"across {stat.count} tasks. Consider adjusting its system prompt "
                        f"or adding a scout phase for these task types."
                    ),
                    evidence=f"role={stat.key} avg={stat.avg_overall:.1f} n={stat.count}",
                )
            )
        elif stat.low_dimensions:
            dim_list = ", ".join(stat.low_dimensions)
            recs.append(
                Recommendation(
                    category="prompt",
                    severity="warning",
                    message=(
                        f"Role '{stat.key}' has low scores in: {dim_list}. "
                        f"Avg overall: {stat.avg_overall:.1f}/10 ({stat.count} tasks). "
                        f"Consider targeted prompt improvements for these dimensions."
                    ),
                    evidence=f"role={stat.key} low_dims=[{dim_list}] n={stat.count}",
                )
            )

    # Low-scoring task types
    for stat in report.by_task_type:
        if stat.avg_overall < LOW_SCORE_THRESHOLD:
            recs.append(
                Recommendation(
                    category="workflow",
                    severity="critical",
                    message=(
                        f"Task type '{stat.key}' scores {stat.avg_overall:.1f} avg "
                        f"across {stat.count} tasks. Consider decomposing differently "
                        f"or assigning to a different role."
                    ),
                    evidence=f"task_type={stat.key} avg={stat.avg_overall:.1f} n={stat.count}",
                )
            )

    # Dimension-level correlations
    for corr in report.correlations:
        if corr.direction == "low":
            recs.append(
                Recommendation(
                    category="workflow",
                    severity="warning",
                    message=(
                        f"{corr.group_type} '{corr.group_key}' scores {corr.avg_score:.1f} "
                        f"avg on '{corr.dimension}' ({corr.count} tasks). "
                        f"Investigate root cause and consider adjustments."
                    ),
                    evidence=(
                        f"{corr.group_type}={corr.group_key} dim={corr.dimension} "
                        f"avg={corr.avg_score:.1f} n={corr.count}"
                    ),
                )
            )

    # Trust suggestions are included
    recs.extend(report.trust_suggestions)

    return recs


# ---------------------------------------------------------------------------
# Prompt version analysis
# ---------------------------------------------------------------------------


def analyze_prompts(ratings: list[Rating], role: str) -> PromptReport:
    """Analyze quality scores by prompt version for a specific role.

    Ratings must have metadata.prompt_version set for meaningful results.
    Falls back to metadata.role matching to filter, then groups by prompt_version.
    """
    report = PromptReport(role=role)

    # Filter to ratings for this role
    role_ratings = [r for r in ratings if r.metadata.get("role") == role]
    report.total_ratings = len(role_ratings)

    if not role_ratings:
        return report

    # Group by prompt version
    version_groups = _group_ratings(
        role_ratings,
        lambda r: r.metadata.get("prompt_version", "unknown"),
    )

    for version, vr in sorted(version_groups.items()):
        dim_avgs = _compute_dimension_avgs(vr)
        avg_overall = _compute_avg_overall(vr)
        report.versions.append(
            PromptVersionStats(
                version=version,
                count=len(vr),
                avg_overall=avg_overall,
                dimension_avgs=dim_avgs,
            )
        )

    # Generate recommendations
    report.recommendations = _generate_prompt_recommendations(report)

    return report


def _generate_prompt_recommendations(report: PromptReport) -> list[Recommendation]:
    """Generate recommendations based on prompt version analysis."""
    recs: list[Recommendation] = []

    if len(report.versions) < 2:
        return recs

    # Sort by avg_overall descending
    sorted_versions = sorted(report.versions, key=lambda v: v.avg_overall, reverse=True)
    best = sorted_versions[0]
    worst = sorted_versions[-1]

    if best.avg_overall - worst.avg_overall > 1.0:
        recs.append(
            Recommendation(
                category="prompt",
                severity="warning",
                message=(
                    f"Prompt version '{best.version}' for role '{report.role}' "
                    f"scores {best.avg_overall:.1f} avg vs '{worst.version}' at "
                    f"{worst.avg_overall:.1f}. Consider rolling back to "
                    f"'{best.version}' or investigating the regression."
                ),
                evidence=(
                    f"best={best.version}({best.avg_overall:.1f}, n={best.count}) "
                    f"worst={worst.version}({worst.avg_overall:.1f}, n={worst.count})"
                ),
            )
        )

    # Flag versions with declining scores
    for v in sorted_versions:
        if v.avg_overall < LOW_SCORE_THRESHOLD and v.count >= MIN_SAMPLE_SIZE:
            recs.append(
                Recommendation(
                    category="prompt",
                    severity="critical",
                    message=(
                        f"Prompt version '{v.version}' for role '{report.role}' "
                        f"scores {v.avg_overall:.1f} avg across {v.count} tasks. "
                        f"This version should be retired or significantly revised."
                    ),
                    evidence=f"version={v.version} avg={v.avg_overall:.1f} n={v.count}",
                )
            )

    return recs


# ---------------------------------------------------------------------------
# Planning pattern analysis
# ---------------------------------------------------------------------------


def _bucket_checklist_size(size: int) -> str:
    """Bucket checklist size into categories."""
    if size == 0:
        return "none (0)"
    if size <= 3:
        return "small (1-3)"
    if size <= 7:
        return "medium (4-7)"
    return "large (8+)"


def _bucket_dependency_count(count: int) -> str:
    """Bucket dependency count into categories."""
    if count == 0:
        return "independent (0)"
    if count <= 2:
        return "few (1-2)"
    return "many (3+)"


def _bucket_context_size(size: int) -> str:
    """Bucket context bundle size into categories."""
    if size == 0:
        return "none (0)"
    if size <= 3:
        return "small (1-3)"
    if size <= 7:
        return "medium (4-7)"
    return "large (8+)"


def _bucket_done_when_specificity(text: str) -> str:
    """Bucket done_when text by specificity (rough heuristic based on length)."""
    if not text:
        return "absent"
    length = len(text)
    if length < 30:
        return "vague (<30 chars)"
    if length < 100:
        return "moderate (30-100 chars)"
    return "specific (100+ chars)"


def analyze_planning(ratings: list[Rating]) -> PlanningReport:
    """Analyze which spec structures produce better outcomes.

    Looks at:
    - Checklist size vs outcome
    - Dependency count vs outcome
    - Context bundle size vs outcome
    - Done-when specificity vs outcome
    """
    report = PlanningReport(total_ratings=len(ratings))

    if not ratings:
        return report

    # Group by checklist size
    checklist_groups: dict[str, list[Rating]] = {}
    dep_groups: dict[str, list[Rating]] = {}
    context_groups: dict[str, list[Rating]] = {}
    done_when_groups: dict[str, list[Rating]] = {}

    for r in ratings:
        # Checklist size
        checklist = r.metadata.get("checklist", [])
        if isinstance(checklist, str):
            try:
                checklist = json.loads(checklist)
            except (json.JSONDecodeError, TypeError, ValueError):
                checklist = []
        if isinstance(checklist, list):
            cl_bucket = _bucket_checklist_size(len(checklist))
            checklist_groups.setdefault(cl_bucket, []).append(r)

        # Dependency count
        deps = r.metadata.get("depends_on", [])
        if isinstance(deps, str):
            try:
                deps = json.loads(deps)
            except (json.JSONDecodeError, TypeError, ValueError):
                deps = []
        if isinstance(deps, list):
            dep_bucket = _bucket_dependency_count(len(deps))
            dep_groups.setdefault(dep_bucket, []).append(r)

        # Context bundle size
        bundle = r.metadata.get("context_bundle", [])
        if isinstance(bundle, str):
            try:
                bundle = json.loads(bundle)
            except (json.JSONDecodeError, TypeError, ValueError):
                bundle = []
        if isinstance(bundle, list):
            ctx_bucket = _bucket_context_size(len(bundle))
            context_groups.setdefault(ctx_bucket, []).append(r)

        # Done-when specificity
        done_when = r.metadata.get("done_when", "")
        if isinstance(done_when, str):
            dw_bucket = _bucket_done_when_specificity(done_when)
            done_when_groups.setdefault(dw_bucket, []).append(r)

    # Compute stats for each factor
    report.by_checklist_size = _planning_stats("checklist_size", checklist_groups)
    report.by_dependency_count = _planning_stats("dependency_count", dep_groups)
    report.by_context_size = _planning_stats("context_size", context_groups)
    report.by_done_when_specificity = _planning_stats(
        "done_when_specificity", done_when_groups
    )

    # Generate recommendations
    report.recommendations = _generate_planning_recommendations(report)

    return report


def _planning_stats(
    factor: str, groups: dict[str, list[Rating]]
) -> list[PlanningStats]:
    """Compute PlanningStats for each bucket in a factor."""
    stats = []
    for bucket, ratings in sorted(groups.items()):
        avg = _compute_avg_overall(ratings)
        success_count = sum(1 for r in ratings if r.overall >= 7.0)
        success_rate = round(success_count / len(ratings), 2) if ratings else 0.0
        stats.append(
            PlanningStats(
                factor=factor,
                bucket=bucket,
                count=len(ratings),
                avg_overall=avg,
                success_rate=success_rate,
            )
        )
    return stats


def _generate_planning_recommendations(report: PlanningReport) -> list[Recommendation]:
    """Generate recommendations based on planning pattern analysis."""
    recs: list[Recommendation] = []

    # Analyze checklist size impact
    _add_factor_recs(recs, "checklist_size", "Checklist size", report.by_checklist_size)
    _add_factor_recs(
        recs, "dependency_count", "Dependency count", report.by_dependency_count
    )
    _add_factor_recs(
        recs, "context_size", "Context bundle size", report.by_context_size
    )
    _add_factor_recs(
        recs,
        "done_when_specificity",
        "Done-when specificity",
        report.by_done_when_specificity,
    )

    return recs


def _add_factor_recs(
    recs: list[Recommendation],
    factor: str,
    label: str,
    stats_list: list[PlanningStats],
) -> None:
    """Add recommendations for a planning factor based on comparative stats."""
    qualified = [s for s in stats_list if s.count >= MIN_SAMPLE_SIZE]

    # Comparative recommendation requires 2+ qualified buckets
    if len(qualified) >= 2:
        best = max(qualified, key=lambda s: s.avg_overall)
        worst = min(qualified, key=lambda s: s.avg_overall)

        if best.avg_overall - worst.avg_overall > 1.0:
            recs.append(
                Recommendation(
                    category="planning",
                    severity="info",
                    message=(
                        f"{label}: '{best.bucket}' produces better outcomes "
                        f"(avg {best.avg_overall:.1f}, {best.success_rate:.0%} success) "
                        f"vs '{worst.bucket}' "
                        f"(avg {worst.avg_overall:.1f}, {worst.success_rate:.0%} success). "
                        f"Consider adjusting future task specs accordingly."
                    ),
                    evidence=(
                        f"factor={factor} "
                        f"best={best.bucket}(avg={best.avg_overall:.1f}, n={best.count}) "
                        f"worst={worst.bucket}(avg={worst.avg_overall:.1f}, n={worst.count})"
                    ),
                )
            )

    # Flag any bucket with very low success rate
    for s in qualified:
        if s.success_rate < 0.3 and s.count >= MIN_SAMPLE_SIZE:
            recs.append(
                Recommendation(
                    category="planning",
                    severity="warning",
                    message=(
                        f"{label} '{s.bucket}' has only {s.success_rate:.0%} success rate "
                        f"across {s.count} tasks (avg {s.avg_overall:.1f}). "
                        f"This pattern consistently produces poor outcomes."
                    ),
                    evidence=(
                        f"factor={factor} bucket={s.bucket} "
                        f"success_rate={s.success_rate:.2f} n={s.count}"
                    ),
                )
            )


# ---------------------------------------------------------------------------
# Formatting helpers for CLI output
# ---------------------------------------------------------------------------


def format_pattern_report(report: PatternReport) -> str:
    """Format a PatternReport into human-readable text."""
    lines = ["Pattern Analysis", "================"]

    if report.total_ratings == 0:
        lines.append("No rating data found. Rate completed tasks with `corc rate`.")
        return "\n".join(lines)

    lines.append(f"Analyzed {report.total_ratings} ratings.\n")

    # --- By role ---
    if report.by_role:
        lines.append("Scores by Role:")
        lines.append("-" * 40)
        for stat in sorted(report.by_role, key=lambda s: s.avg_overall):
            flag = " ⚠" if stat.avg_overall < FLAG_THRESHOLD else ""
            lines.append(
                f"  {stat.key:<20} avg={stat.avg_overall:.1f}/10  n={stat.count}{flag}"
            )
            if stat.low_dimensions:
                lines.append(f"    low: {', '.join(stat.low_dimensions)}")
        lines.append("")

    # --- By task type ---
    if report.by_task_type:
        lines.append("Scores by Task Type:")
        lines.append("-" * 40)
        for stat in sorted(report.by_task_type, key=lambda s: s.avg_overall):
            flag = " ⚠" if stat.avg_overall < FLAG_THRESHOLD else ""
            lines.append(
                f"  {stat.key:<20} avg={stat.avg_overall:.1f}/10  n={stat.count}{flag}"
            )
        lines.append("")

    # --- By context bundle ---
    if report.by_context_bundle:
        lines.append("Scores by Context Bundle Size:")
        lines.append("-" * 40)
        for stat in sorted(report.by_context_bundle, key=lambda s: s.avg_overall):
            lines.append(
                f"  {stat.key:<25} avg={stat.avg_overall:.1f}/10  n={stat.count}"
            )
        lines.append("")

    # --- Correlations ---
    low_corr = [c for c in report.correlations if c.direction == "low"]
    high_corr = [c for c in report.correlations if c.direction == "high"]

    if low_corr:
        lines.append("Low-scoring correlations:")
        lines.append("-" * 40)
        for c in sorted(low_corr, key=lambda x: x.avg_score):
            lines.append(
                f"  🔴 {c.group_type} '{c.group_key}' → '{c.dimension}': "
                f"{c.avg_score:.1f}/10 (n={c.count})"
            )
        lines.append("")

    if high_corr:
        lines.append("High-scoring correlations:")
        lines.append("-" * 40)
        for c in sorted(high_corr, key=lambda x: -x.avg_score):
            lines.append(
                f"  🟢 {c.group_type} '{c.group_key}' → '{c.dimension}': "
                f"{c.avg_score:.1f}/10 (n={c.count})"
            )
        lines.append("")

    # --- Recommendations ---
    if report.recommendations:
        lines.append("Recommendations:")
        lines.append("=" * 40)
        for rec in report.recommendations:
            icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(
                rec.severity, "•"
            )
            lines.append(f"  {icon} [{rec.category}] {rec.message}")
        lines.append("")

    return "\n".join(lines)


def format_prompt_report(report: PromptReport) -> str:
    """Format a PromptReport into human-readable text."""
    lines = [
        f"Prompt Version Analysis: {report.role}",
        "=" * (26 + len(report.role)),
    ]

    if report.total_ratings == 0:
        lines.append(f"No ratings found for role '{report.role}'.")
        return "\n".join(lines)

    lines.append(f"Total ratings: {report.total_ratings}\n")

    if report.versions:
        lines.append("Version Performance:")
        lines.append("-" * 50)
        for v in sorted(report.versions, key=lambda x: x.avg_overall, reverse=True):
            lines.append(f"  {v.version:<20} avg={v.avg_overall:.1f}/10  n={v.count}")
            if v.dimension_avgs:
                dim_strs = [f"{d}={s:.1f}" for d, s in sorted(v.dimension_avgs.items())]
                lines.append(f"    {', '.join(dim_strs)}")
        lines.append("")

    if report.recommendations:
        lines.append("Recommendations:")
        lines.append("-" * 50)
        for rec in report.recommendations:
            icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(
                rec.severity, "•"
            )
            lines.append(f"  {icon} {rec.message}")
        lines.append("")

    return "\n".join(lines)


def format_planning_report(report: PlanningReport) -> str:
    """Format a PlanningReport into human-readable text."""
    lines = ["Planning Pattern Analysis", "========================"]

    if report.total_ratings == 0:
        lines.append("No rating data found.")
        return "\n".join(lines)

    lines.append(f"Analyzed {report.total_ratings} ratings.\n")

    sections = [
        ("Checklist Size vs Outcome:", report.by_checklist_size),
        ("Dependency Count vs Outcome:", report.by_dependency_count),
        ("Context Bundle Size vs Outcome:", report.by_context_size),
        ("Done-When Specificity vs Outcome:", report.by_done_when_specificity),
    ]

    for title, stats_list in sections:
        if stats_list:
            lines.append(title)
            lines.append("-" * 50)
            for s in stats_list:
                success_pct = f"{s.success_rate:.0%}"
                lines.append(
                    f"  {s.bucket:<25} avg={s.avg_overall:.1f}/10  "
                    f"success={success_pct:<5} n={s.count}"
                )
            lines.append("")

    if report.recommendations:
        lines.append("Recommendations:")
        lines.append("=" * 50)
        for rec in report.recommendations:
            icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(
                rec.severity, "•"
            )
            lines.append(f"  {icon} [{rec.category}] {rec.message}")
        lines.append("")

    return "\n".join(lines)
