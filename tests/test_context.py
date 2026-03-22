"""Tests for context assembly."""

from pathlib import Path

from corc.context import assemble_context, _extract_section


def test_basic_assembly(tmp_path):
    (tmp_path / "doc.md").write_text("# My Doc\nSome content here.")
    task = {
        "name": "test task",
        "done_when": "tests pass",
        "context_bundle": ["doc.md"],
    }
    ctx = assemble_context(task, tmp_path)
    assert "TASK DEFINITION" in ctx
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
    assert "TASK DEFINITION" in ctx
    assert "test" in ctx
