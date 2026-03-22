"""Interactive planning sessions.

`corc plan` opens an interactive Claude Code session for collaborative
spec development and task decomposition. The session receives full CORC
context (knowledge store, work state, repo structure) and can create
tasks via `corc task create`.
"""

import json
import os
import subprocess
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Planner role system prompt
# ---------------------------------------------------------------------------

PLANNER_ROLE = """\
You are a CORC Planning Agent — a collaborative partner for spec development and task decomposition.

## Your Role
You help the operator develop specifications and decompose them into executable tasks. \
You have full visibility into the CORC knowledge store, work state, and repository context.

## Available Commands
Create tasks by running shell commands:
```bash
corc task create "task-name" --done-when "testable criteria" --role implementer \\
  --depends-on "id1,id2" --checklist "item1,item2" --context "file1,file2"
```

Search the knowledge store:
```bash
corc knowledge search "query"
```

## Draft Specs
Save draft specifications as the session progresses. Write them to the drafts \
directory indicated below so the session can be resumed if interrupted.

## Planning Process

### Quick Task Detection
First, determine the appropriate level of formality:
- **Quick task**: A single code change with clear "done when." No decomposition needed. \
Write directly as one task. Example: "Fix the typo in README.md" or "Add a --verbose flag to corc search."
- **Standard task**: A feature needing a brief spec and 2-5 tasks. Scout phase optional. \
Example: "Add semantic search to the knowledge store."
- **Epic**: A large feature needing full spec, detailed decomposition, 5+ tasks, scout phases, \
adversarial review. Example: "Build the workflow engine."

Propose the level and get operator confirmation before proceeding.

### Three-Stage Process (for standard/epic)

**Stage 1 — Spec Development**: Discuss the problem, research options, write a spec. \
The spec follows the module spec template below.

**Stage 2 — Task Decomposition**: Break the spec into a DAG of concrete tasks. Each task gets:
- A name and description
- A role (scout, implementer, reviewer, adversarial-reviewer)
- depends_on — which tasks must complete first
- done_when — machine-testable completion criteria (no subjective words)
- checklist — sub-steps for structured progress tracking
- context_bundle — specific files to inject as context

**Stage 3 — Review and Commit**: Review the full plan (spec + task DAG). \
When operator approves, create all tasks via `corc task create`.

### Spec Template

```markdown
# [Feature Name]

## Problem
What problem does this solve? Why now?

## Requirements
- [ ] Requirement 1
- [ ] Requirement 2

## Non-Requirements
What is explicitly out of scope.

## Design
How the feature works. Include key decisions and trade-offs.

## Testing Strategy
How to verify the feature works correctly.

## Rationale
Key planning decisions: why decomposed this way, why certain dependencies exist, \
what alternatives were considered.
```
"""


# ---------------------------------------------------------------------------
# Context assembly helpers
# ---------------------------------------------------------------------------

def _get_knowledge_summary(ks) -> str:
    """Summarize the knowledge store for the planning session."""
    docs = ks.list_docs()
    if not docs:
        return "  (empty — no documents in knowledge store)"

    lines = []
    by_type: dict[str, list] = {}
    for doc in docs:
        by_type.setdefault(doc["type"], []).append(doc)

    for doc_type, group in sorted(by_type.items()):
        lines.append(f"  {doc_type} ({len(group)}):")
        for doc in group[:10]:
            lines.append(f"    - {doc['title']} [{doc['id']}]")
        if len(group) > 10:
            lines.append(f"    ... and {len(group) - 10} more")

    return "\n".join(lines)


def _get_work_state_summary(ws) -> str:
    """Summarize the current work state for the planning session."""
    tasks = ws.list_tasks()
    if not tasks:
        return "  No tasks in work state."

    lines = []
    by_status: dict[str, list] = {}
    for t in tasks:
        by_status.setdefault(t["status"], []).append(t)

    for status in ["completed", "running", "pending", "failed", "blocked"]:
        group = by_status.get(status, [])
        if not group:
            continue
        icon = {
            "completed": "done", "running": "running",
            "pending": "pending", "failed": "failed", "blocked": "blocked",
        }.get(status, "?")
        lines.append(f"  [{icon}] {status} ({len(group)}):")
        for t in group:
            deps = t.get("depends_on", [])
            dep_str = f" (depends: {', '.join(deps)})" if deps else ""
            lines.append(f"    - [{t['id']}] {t['name']}: {t['done_when']}{dep_str}")

    ready = ws.get_ready_tasks()
    if ready:
        lines.append(f"\n  Ready to dispatch: {', '.join(t['name'] for t in ready)}")

    return "\n".join(lines)


def _get_repo_context(project_root: Path) -> str:
    """Get repository context: file structure and recent commits."""
    lines = []

    # Recent git commits
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-20"],
            capture_output=True, text=True, cwd=str(project_root),
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines.append("  Recent commits:")
            for line in result.stdout.strip().split("\n"):
                lines.append(f"    {line}")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Source file structure
    src_dir = project_root / "src"
    if src_dir.exists():
        lines.append("\n  Source files:")
        for py_file in sorted(src_dir.rglob("*.py")):
            try:
                rel = py_file.relative_to(project_root)
                lines.append(f"    {rel}")
            except ValueError:
                pass

    # Test structure
    test_dir = project_root / "tests"
    if test_dir.exists():
        lines.append("\n  Test files:")
        for py_file in sorted(test_dir.rglob("*.py")):
            try:
                rel = py_file.relative_to(project_root)
                lines.append(f"    {rel}")
            except ValueError:
                pass

    return "\n".join(lines) if lines else "  (no repo context available)"


# ---------------------------------------------------------------------------
# Draft / session management
# ---------------------------------------------------------------------------

def get_drafts_dir(corc_dir: Path) -> Path:
    """Return the drafts directory path, creating it if needed."""
    drafts_dir = corc_dir / "drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)
    return drafts_dir


def save_session_metadata(corc_dir: Path, session_id: str,
                          seed_file: str | None = None) -> Path:
    """Save session metadata for crash recovery."""
    drafts_dir = get_drafts_dir(corc_dir)
    meta = {
        "session_id": session_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed_file": seed_file,
        "status": "active",
    }
    meta_path = drafts_dir / f"session-{session_id}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path


def mark_session_complete(corc_dir: Path, session_id: str) -> None:
    """Mark a session as complete in its metadata file."""
    drafts_dir = get_drafts_dir(corc_dir)
    meta_path = drafts_dir / f"session-{session_id}.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            meta["status"] = "complete"
            meta["completed"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            meta_path.write_text(json.dumps(meta, indent=2))
        except (json.JSONDecodeError, IOError):
            pass


def load_latest_draft(corc_dir: Path) -> tuple[dict | None, str | None]:
    """Load the most recent session metadata and any draft content.

    Returns (metadata_dict, draft_content) or (None, None) if nothing found.
    """
    drafts_dir = get_drafts_dir(corc_dir)

    # Find latest session metadata
    sessions = sorted(
        drafts_dir.glob("session-*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not sessions:
        return None, None

    latest_meta_path = sessions[-1]
    try:
        meta = json.loads(latest_meta_path.read_text())
    except (json.JSONDecodeError, IOError):
        return None, None

    # Find latest draft spec file
    draft_files = sorted(
        drafts_dir.glob("plan-*.md"),
        key=lambda p: p.stat().st_mtime,
    )
    draft_content = None
    if draft_files:
        try:
            draft_content = draft_files[-1].read_text()
        except IOError:
            pass

    return meta, draft_content


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def build_system_prompt(
    paths: dict,
    ws,
    ks,
    seed_content: str | None = None,
    draft_content: str | None = None,
    resume_meta: dict | None = None,
) -> str:
    """Build the complete system prompt for the planning session.

    Combines:
    - Planner role instructions and spec template
    - Knowledge store summary
    - Work state summary
    - Repository context
    - (optional) Seed document content
    - (optional) Previous draft for resume
    """
    parts = [PLANNER_ROLE]

    # Knowledge store summary
    parts.append("\n## Current Knowledge Store")
    parts.append(_get_knowledge_summary(ks))
    parts.append("")

    # Work state summary
    parts.append("## Current Work State")
    parts.append(_get_work_state_summary(ws))
    parts.append("")

    # Repo context
    parts.append("## Repository Context")
    parts.append(_get_repo_context(paths["root"]))
    parts.append("")

    # Draft directory info
    drafts_dir = get_drafts_dir(paths["corc_dir"])
    parts.append("## Draft Auto-Save")
    parts.append(f"Save draft specs to: {drafts_dir}/")
    parts.append("Use filename pattern: plan-<descriptive-name>.md")
    parts.append("")

    # Seed document
    if seed_content:
        parts.append("## Seed Document (Pre-loaded)")
        parts.append("The operator provided the following document to seed this planning session:")
        parts.append("```")
        parts.append(seed_content)
        parts.append("```")
        parts.append("")

    # Resume context
    if draft_content:
        parts.append("## Previous Draft (Resuming)")
        parts.append("This is a resumed planning session. Here is the last saved draft:")
        parts.append("```")
        parts.append(draft_content)
        parts.append("```")
        parts.append("Continue from where we left off.")
        parts.append("")

    if resume_meta:
        parts.append(f"Previous session started: {resume_meta.get('timestamp', 'unknown')}")
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Session launcher
# ---------------------------------------------------------------------------

def launch_interactive_claude(system_prompt: str,
                              continue_session: bool = False) -> int:
    """Launch an interactive claude session with the given system prompt.

    The child process inherits stdin/stdout/stderr so the user gets a
    fully interactive terminal session.

    Args:
        system_prompt: The system prompt to inject.
        continue_session: If True, pass --continue to resume last conversation.

    Returns:
        The exit code of the claude process.
    """
    cmd = ["claude", "--system-prompt", system_prompt]

    if continue_session:
        cmd.append("--continue")

    result = subprocess.run(cmd)
    return result.returncode
