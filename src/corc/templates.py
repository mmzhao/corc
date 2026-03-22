"""Document templates for the knowledge store.

Loads markdown templates with YAML frontmatter from knowledge/_templates/
and supports variable substitution for generating new documents.
"""

from pathlib import Path

VALID_TYPES = ("decision", "task-outcome", "architecture", "repo-context", "research")

# Templates directory is at knowledge/_templates/ relative to the project root,
# but we also bundle them alongside this module for reliable access.
_TEMPLATES_DIR = Path(__file__).parent.parent.parent / "knowledge" / "_templates"


def get_templates_dir(project_root: Path | None = None) -> Path:
    """Return the templates directory path.

    Uses project_root/knowledge/_templates/ if given, otherwise
    falls back to the path relative to this module.
    """
    if project_root:
        return project_root / "knowledge" / "_templates"
    return _TEMPLATES_DIR


def list_types() -> list[str]:
    """Return all valid template type names."""
    return list(VALID_TYPES)


def get_template(doc_type: str, project_root: Path | None = None) -> str:
    """Load and return the raw template content for a document type.

    Args:
        doc_type: One of the valid document types.
        project_root: Optional project root for locating templates.

    Returns:
        The template content as a string.

    Raises:
        ValueError: If the document type is not valid.
        FileNotFoundError: If the template file is missing.
    """
    if doc_type not in VALID_TYPES:
        raise ValueError(
            f"Unknown template type: {doc_type!r}. "
            f"Valid types: {', '.join(VALID_TYPES)}"
        )

    templates_dir = get_templates_dir(project_root)
    template_path = templates_dir / f"{doc_type}.md"

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    return template_path.read_text()


def render_template(
    doc_type: str,
    *,
    title: str = "Untitled",
    project: str = "",
    doc_id: str | None = None,
    project_root: Path | None = None,
) -> str:
    """Load a template and substitute placeholder variables.

    Replaces ${id}, ${title}, ${project}, ${created}, ${updated} placeholders.

    Args:
        doc_type: Document type to render.
        title: Document title.
        project: Project name.
        doc_id: Document ID (generated UUID if not provided).
        project_root: Optional project root for locating templates.

    Returns:
        Rendered template content.
    """
    import uuid
    from datetime import datetime, timezone

    content = get_template(doc_type, project_root=project_root)

    if doc_id is None:
        doc_id = str(uuid.uuid4())

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    content = content.replace("${id}", doc_id)
    content = content.replace("${title}", title)
    content = content.replace("${project}", project)
    content = content.replace("${created}", now)
    content = content.replace("${updated}", now)

    return content
