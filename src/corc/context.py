"""Deterministic context assembly.

Reads a task definition, resolves context bundle file paths to contents,
assembles into a single context document for injection.

Same task + same files on disk = same context output.
"""

from pathlib import Path


def assemble_context(task: dict, project_root: Path) -> str:
    """Assemble the full context for a task dispatch.

    Returns the concatenated content of all files in the context bundle,
    prefixed with the task definition.
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
    if not bundle:
        return "\n".join(parts)

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
