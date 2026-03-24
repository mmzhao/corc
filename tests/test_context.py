"""Tests for context assembly."""

from pathlib import Path

from corc.context import assemble_context, _extract_section, _load_blacklist


def test_basic_assembly(tmp_path):
    (tmp_path / "doc.md").write_text("# My Doc\nSome content here.")
    task = {
        "name": "test task",
        "done_when": "tests pass",
        "context_bundle": ["doc.md"],
    }
    ctx = assemble_context(task, tmp_path)
    assert "<definition>" in ctx
    assert "test task" in ctx
    assert "Some content here" in ctx


def test_missing_file(tmp_path):
    task = {
        "name": "test",
        "done_when": "done",
        "context_bundle": ["nonexistent.md"],
    }
    ctx = assemble_context(task, tmp_path)
    assert "WARNING: File not found" in ctx


def test_section_extraction():
    content = """# Top

## Module 1: Knowledge Store

Store stuff here.

### Subsection

Details.

## Module 2: Workflow

Other stuff.
"""
    section = _extract_section(content, "module-1-knowledge-store")
    assert "Store stuff here" in section
    assert "Details" in section
    assert "Other stuff" not in section


def test_section_reference(tmp_path):
    (tmp_path / "spec.md").write_text("""# Spec

## Module 1: Search

Search implementation details.

## Module 2: Other

Other stuff.
""")
    task = {
        "name": "test",
        "done_when": "done",
        "context_bundle": ["spec.md#module-1-search"],
    }
    ctx = assemble_context(task, tmp_path)
    assert "Search implementation" in ctx
    assert "Other stuff" not in ctx


def test_checklist_in_context(tmp_path):
    task = {
        "name": "test",
        "done_when": "done",
        "checklist": [
            {"item": "Step 1", "done": True},
            {"item": "Step 2", "done": False},
        ],
        "context_bundle": [],
    }
    ctx = assemble_context(task, tmp_path)
    assert "✅ Step 1" in ctx
    assert "☐ Step 2" in ctx


def test_empty_bundle(tmp_path):
    task = {
        "name": "test",
        "done_when": "done",
        "context_bundle": [],
    }
    ctx = assemble_context(task, tmp_path)
    assert "<definition>" in ctx
    assert "test" in ctx


# ---------------------------------------------------------------------------
# Blacklist injection tests
# ---------------------------------------------------------------------------


def _make_blacklist(tmp_path, content=None):
    """Helper: create a .corc/blacklist.md file in the tmp project root."""
    corc_dir = tmp_path / ".corc"
    corc_dir.mkdir(exist_ok=True)
    blacklist_content = (
        (
            "# Agent Blacklist\n\n"
            "- Never use eval(). (Reason: security)\n"
            "- Never merge directly to main. (Reason: review gate)\n"
        )
        if content is None
        else content
    )
    (corc_dir / "blacklist.md").write_text(blacklist_content)
    return blacklist_content


def test_blacklist_injected_into_context(tmp_path):
    """Blacklist content appears in assembled context when file exists."""
    blacklist_content = _make_blacklist(tmp_path)
    task = {
        "name": "test",
        "done_when": "done",
        "context_bundle": [],
    }
    ctx = assemble_context(task, tmp_path)
    assert "<blacklist>" in ctx
    assert "</blacklist>" in ctx
    assert "Never use eval()" in ctx
    assert "Never merge directly to main" in ctx


def test_blacklist_injected_with_context_bundle(tmp_path):
    """Blacklist is appended even when a context bundle is present."""
    _make_blacklist(tmp_path)
    (tmp_path / "doc.md").write_text("# Doc\nSome content.")
    task = {
        "name": "test",
        "done_when": "done",
        "context_bundle": ["doc.md"],
    }
    ctx = assemble_context(task, tmp_path)
    # Both context bundle content and blacklist should be present
    assert "Some content" in ctx
    assert "<blacklist>" in ctx
    assert "Never use eval()" in ctx


def test_blacklist_missing_file_graceful(tmp_path):
    """No error and no blacklist section when .corc/blacklist.md is missing."""
    task = {
        "name": "test",
        "done_when": "done",
        "context_bundle": [],
    }
    ctx = assemble_context(task, tmp_path)
    assert "<blacklist>" not in ctx
    # Context should still assemble normally
    assert "<definition>" in ctx


def test_blacklist_missing_corc_dir_graceful(tmp_path):
    """No error when .corc directory doesn't exist at all."""
    task = {
        "name": "test",
        "done_when": "done",
        "context_bundle": [],
    }
    # tmp_path has no .corc directory
    ctx = assemble_context(task, tmp_path)
    assert "<blacklist>" not in ctx
    assert "<definition>" in ctx


def test_blacklist_empty_file(tmp_path):
    """Empty blacklist file is treated as no blacklist (no section added)."""
    _make_blacklist(tmp_path, content="")
    task = {
        "name": "test",
        "done_when": "done",
        "context_bundle": [],
    }
    ctx = assemble_context(task, tmp_path)
    # Empty content after strip() is falsy, so no blacklist section
    assert "<blacklist>" not in ctx


def test_load_blacklist_returns_content(tmp_path):
    """_load_blacklist returns file contents when file exists."""
    content = _make_blacklist(tmp_path)
    result = _load_blacklist(tmp_path)
    assert result == content


def test_load_blacklist_returns_none_when_missing(tmp_path):
    """_load_blacklist returns None when file does not exist."""
    result = _load_blacklist(tmp_path)
    assert result is None


def test_blacklist_appears_after_context_bundle(tmp_path):
    """Blacklist section comes after context bundle sections."""
    _make_blacklist(tmp_path)
    (tmp_path / "doc.md").write_text("# Doc\nContent.")
    task = {
        "name": "test",
        "done_when": "done",
        "context_bundle": ["doc.md"],
    }
    ctx = assemble_context(task, tmp_path)
    bundle_end = ctx.index("</file>")
    blacklist_start = ctx.index("<blacklist>")
    assert blacklist_start > bundle_end
