"""Deterministic context assembly.

Reads a task definition, resolves context bundle file paths to contents,
assembles into a single context document for injection.

Same task + same files on disk = same context output.
"""

import json
from pathlib import Path
from typing import Any


def _load_blacklist(project_root: Path) -> str | None:
    """Load the agent blacklist from .corc/blacklist.md.

    Returns the file contents if the blacklist exists, None otherwise.
    Handles missing files and read errors gracefully.
    """
    blacklist_path = project_root / ".corc" / "blacklist.md"
    try:
        if blacklist_path.exists():
            return blacklist_path.read_text()
    except OSError:
        pass
    return None


def assemble_context(
    task: dict,
    project_root: Path,
    *,
    mutations: list[dict] | None = None,
    plan_tasks: list[dict] | None = None,
) -> str:
    """Assemble the full context for a task dispatch.

    Returns the concatenated content of all files in the context bundle,
    prefixed with the task definition, optionally including a catch-up summary,
    and suffixed with the agent blacklist.
    The blacklist is always appended (when present) regardless of context bundle.

    If mutations and plan_tasks are provided, a catch-up summary is generated
    from recent mutations relevant to the task's dependencies and injected
    between the task definition and the context bundle.
    """
    parts = []

    parts.append("=== TASK DEFINITION ===")
    parts.append(f"Task: {task['name']}")
    if task.get("description"):
        parts.append(f"Description: {task['description']}")
    parts.append(f"Done when: {task['done_when']}")

    checklist = task.get("checklist", [])
    if checklist:
        parts.append("\nChecklist:")
        for item in checklist:
            if isinstance(item, dict):
                status = "✅" if item.get("done") else "☐"
                parts.append(f"  {status} {item.get('item', item)}")
            else:
                parts.append(f"  ☐ {item}")

    parts.append("=== END TASK DEFINITION ===\n")

    # Inject catch-up summary before context bundle
    if mutations is not None and plan_tasks is not None:
        summary = generate_catch_up_summary(task, mutations, plan_tasks)
        if summary:
            parts.append(summary)
            parts.append("")  # blank line separator

    bundle = task.get("context_bundle", [])
    for ref in bundle:
        ref_str = str(ref)
        section = None
        if "#" in ref_str:
            ref_str, section = ref_str.rsplit("#", 1)

        file_path = project_root / ref_str
        if not file_path.exists():
            parts.append(f"=== CONTEXT: {ref} ===")
            parts.append(f"[WARNING: File not found: {file_path}]")
            parts.append(f"=== END CONTEXT ===\n")
            continue

        content = file_path.read_text()

        if section:
            content = _extract_section(content, section)

        parts.append(f"=== CONTEXT: {ref} ===")
        parts.append(content.strip())
        parts.append(f"=== END CONTEXT ===\n")

    # Always inject the blacklist at the end of assembled context
    blacklist = _load_blacklist(project_root)
    if blacklist:
        parts.append("=== AGENT BLACKLIST ===")
        parts.append(blacklist.strip())
        parts.append("=== END AGENT BLACKLIST ===\n")

    return "\n".join(parts)


def _get_task_status_from_mutations(task_id: str, mutations: list[dict]) -> str:
    """Derive the latest status for a task from mutation types.

    Scans mutations in order and returns the last known status.
    Returns 'pending' if no status-changing mutation is found.
    """
    status_map = {
        "task_created": "pending",
        "task_assigned": "assigned",
        "task_started": "running",
        "task_completed": "completed",
        "task_failed": "failed",
        "task_escalated": "escalated",
        "task_handed_off": "handed_off",
        "task_paused": "paused",
        "task_pending_merge": "pending_merge",
    }
    latest = "pending"
    for m in mutations:
        if m.get("task_id") == task_id and m["type"] in status_map:
            latest = status_map[m["type"]]
    return latest


def generate_catch_up_summary(
    task: dict, mutations: list[dict], plan_tasks: list[dict]
) -> str | None:
    """Generate a catch-up summary from recent mutations.

    Deterministically summarizes what happened to this task's dependencies
    and the overall plan progress, so agents know the current state.

    Returns a formatted catch-up block, or None if there is nothing to report.
    """
    dep_ids = task.get("depends_on", [])
    if isinstance(dep_ids, str):
        dep_ids = json.loads(dep_ids)

    if not dep_ids and not plan_tasks:
        return None

    # Build a task name/status lookup from plan_tasks
    task_info: dict[str, dict] = {}
    for t in plan_tasks:
        tid = t.get("id", "")
        task_info[tid] = {
            "name": t.get("name", tid),
            "status": t.get("status", "pending"),
        }

    # Collect completed deps and their findings from mutations
    completed_deps: dict[str, dict] = {}  # dep_id -> mutation data
    dep_findings: dict[str, list] = {}  # dep_id -> list of findings

    for mutation in mutations:
        m_task_id = mutation.get("task_id")
        if m_task_id not in dep_ids:
            continue

        if mutation["type"] == "task_completed":
            completed_deps[m_task_id] = mutation.get("data", {})
            findings = mutation.get("data", {}).get("findings", [])
            if findings:
                dep_findings[m_task_id] = findings

    lines = []
    lines.append("=== CATCH-UP SUMMARY ===")
    lines.append("Since your last context:")
    has_content = False

    # Report dependency statuses
    for dep_id in dep_ids:
        info = task_info.get(dep_id, {"name": dep_id, "status": "unknown"})
        name = info["name"]

        if dep_id in completed_deps:
            pr_url = completed_deps[dep_id].get("pr_url")
            if pr_url:
                lines.append(f'- Task "{name}" was completed (PR {pr_url})')
            else:
                lines.append(f'- Task "{name}" was completed')
            has_content = True
        else:
            status = _get_task_status_from_mutations(dep_id, mutations)
            lines.append(f'- Task "{name}" is {status}')
            has_content = True

    # Report findings from completed deps
    for dep_id, findings in dep_findings.items():
        name = task_info.get(dep_id, {"name": dep_id})["name"]
        for finding in findings:
            if isinstance(finding, str):
                lines.append(f'- Finding from "{name}": {finding}')
            elif isinstance(finding, dict):
                content = finding.get(
                    "content", finding.get("description", str(finding))
                )
                lines.append(f'- Finding from "{name}": {content}')
            has_content = True

    # Overall progress from plan_tasks
    if plan_tasks:
        completed_count = sum(1 for t in plan_tasks if t.get("status") == "completed")
        total = len(plan_tasks)
        remaining = total - completed_count
        lines.append(f"- {completed_count} tasks completed, {remaining} remaining")
        has_content = True

    lines.append("=== END CATCH-UP ===")

    return "\n".join(lines) if has_content else None


def _extract_section(content: str, section_slug: str) -> str:
    """Extract a markdown section by heading slug.

    Matches headings where the slug (lowercased, hyphenated) matches.
    Returns everything from that heading to the next heading of same or higher level.
    """
    lines = content.split("\n")
    section_slug = section_slug.lower().replace(" ", "-")

    capturing = False
    capture_level = 0
    captured = []

    for line in lines:
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            heading_text = line.lstrip("#").strip()
            slug = (
                heading_text.lower()
                .replace(" ", "-")
                .replace(":", "")
                .replace("(", "")
                .replace(")", "")
            )

            if not capturing and section_slug in slug:
                capturing = True
                capture_level = level
                captured.append(line)
                continue

            if capturing and level <= capture_level:
                break

        if capturing:
            captured.append(line)

    if captured:
        return "\n".join(captured)
    return content  # Fallback: return full content if section not found


def validate_context_bundle_paths(
    context_bundle: list[str], project_root: Path
) -> list[dict[str, str]]:
    """Validate that all context bundle paths exist on disk.

    Returns a list of dicts describing each missing path, e.g.:

        [{"ref": "src/foo.py", "file_path": "src/foo.py", "reason": "file not found"}]

    For section references like ``spec.md#design``, validates that the
    underlying file (``spec.md``) exists. Returns an empty list if all
    paths are valid.
    """
    missing: list[dict[str, str]] = []
    for ref in context_bundle:
        ref_str = str(ref)
        file_part = ref_str
        if "#" in ref_str:
            file_part = ref_str.rsplit("#", 1)[0]

        full_path = project_root / file_part
        try:
            if not full_path.exists():
                missing.append(
                    {
                        "ref": ref_str,
                        "file_path": file_part,
                        "reason": "file not found",
                    }
                )
        except OSError:
            missing.append(
                {
                    "ref": ref_str,
                    "file_path": file_part,
                    "reason": "file not accessible",
                }
            )
    return missing


def record_context_mtimes(
    context_bundle: list[str], project_root: Path
) -> dict[str, float]:
    """Record modification times for all files in a context bundle.

    Called at task creation time to snapshot file mtimes. Returns a dict
    mapping each file path (without section fragment) to its mtime as a float.
    Files that don't exist are silently skipped.
    """
    mtimes: dict[str, float] = {}
    for ref in context_bundle:
        ref_str = str(ref)
        # Strip section fragment (e.g. "spec.md#design" → "spec.md")
        if "#" in ref_str:
            ref_str = ref_str.rsplit("#", 1)[0]

        file_path = project_root / ref_str
        try:
            if file_path.exists():
                mtimes[ref_str] = file_path.stat().st_mtime
        except OSError:
            pass
    return mtimes


def check_context_staleness(task: dict, project_root: Path) -> list[dict[str, Any]]:
    """Check if context bundle files have changed since task creation.

    Compares current file mtimes against those recorded in
    task["context_bundle_mtimes"]. Returns a list of dicts describing
    each stale file, e.g.:

        [{"file": "src/foo.py", "recorded_mtime": 1234.0, "current_mtime": 1235.0}]

    Returns an empty list if no staleness is detected or if no mtimes
    were recorded.
    """
    recorded = task.get("context_bundle_mtimes")
    if not recorded or not isinstance(recorded, dict):
        return []

    stale: list[dict[str, Any]] = []
    for ref_str, recorded_mtime in recorded.items():
        file_path = project_root / ref_str
        try:
            if file_path.exists():
                current_mtime = file_path.stat().st_mtime
                if current_mtime != recorded_mtime:
                    stale.append(
                        {
                            "file": ref_str,
                            "recorded_mtime": recorded_mtime,
                            "current_mtime": current_mtime,
                        }
                    )
        except OSError:
            pass
    return stale
