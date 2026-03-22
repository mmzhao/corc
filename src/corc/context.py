"""Deterministic context assembly.

Reads a task definition, resolves context bundle file paths to contents,
assembles into a single context document for injection.

Same task + same files on disk = same context output.
"""

from pathlib import Path


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


def assemble_context(task: dict, project_root: Path) -> str:
    """Assemble the full context for a task dispatch.

    Returns the concatenated content of all files in the context bundle,
    prefixed with the task definition and suffixed with the agent blacklist.
    The blacklist is always appended (when present) regardless of context bundle.
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
            slug = heading_text.lower().replace(" ", "-").replace(":", "").replace("(", "").replace(")", "")

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
