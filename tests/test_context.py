"""Tests for context assembly."""

import warnings
from pathlib import Path

import pytest

from corc.context import (
    ContextResult,
    assemble_context,
    _extract_python_symbols,
    _extract_section,
    _load_blacklist,
    _normalize_slug,
)


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


# ---------------------------------------------------------------------------
# Slug normalization tests
# ---------------------------------------------------------------------------


def test_normalize_slug_basic():
    """Basic slug normalization: lowercase, hyphens, strip specials."""
    assert _normalize_slug("Module 1: Knowledge Store") == "module-1-knowledge-store"


def test_normalize_slug_ampersand():
    """Ampersand is stripped from slugs."""
    assert _normalize_slug("Roles & Permissions") == "roles-permissions"


def test_normalize_slug_dots():
    """Dots are stripped from slugs."""
    assert _normalize_slug("10. Context Assembly") == "10-context-assembly"


def test_normalize_slug_parens_colon():
    """Parentheses and colons are stripped."""
    assert _normalize_slug("Setup (advanced): Config") == "setup-advanced-config"


def test_normalize_slug_multiple_specials():
    """Multiple special characters are all stripped."""
    assert _normalize_slug("A & B: C (D) 1.2") == "a-b-c-d-12"


# ---------------------------------------------------------------------------
# _extract_section with special characters
# ---------------------------------------------------------------------------


def test_extract_section_with_ampersand():
    """Slug with & in heading matches correctly."""
    content = """# Top

## Roles & Permissions

Permission details here.

## Other Section

Other stuff.
"""
    section = _extract_section(content, "roles-permissions")
    assert "Permission details here" in section
    assert "Other stuff" not in section


def test_extract_section_with_numbered_prefix():
    """Slug with '10.' prefix in heading matches correctly."""
    content = """# Spec

## 10. Context Assembly

Context assembly details.

## 11. Dispatch

Dispatch details.
"""
    section = _extract_section(content, "10-context-assembly")
    assert "Context assembly details" in section
    assert "Dispatch details" not in section


def test_extract_section_with_dots_in_heading():
    """Heading with dots normalizes correctly."""
    content = """# Spec

## v2.0 Release Notes

Release notes here.

## Changelog

Changes.
"""
    section = _extract_section(content, "v20-release-notes")
    assert "Release notes here" in section
    assert "Changes" not in section


def test_extract_section_unmatched_emits_warning():
    """Unmatched section slug emits warnings.warn and returns full content."""
    content = """# Top

## Section A

Content A.
"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _extract_section(content, "nonexistent-section")
        assert len(w) == 1
        assert "nonexistent-section" in str(w[0].message)
        assert "not found" in str(w[0].message)
    # Fallback: returns full content
    assert result == content


def test_extract_section_existing_slug_no_warning():
    """Matched section slug does NOT emit a warning."""
    content = """# Top

## Section A

Content A.

## Section B

Content B.
"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _extract_section(content, "section-a")
        assert len(w) == 0
    assert "Content A" in result
    assert "Content B" not in result


# ---------------------------------------------------------------------------
# Context size metadata tests
# ---------------------------------------------------------------------------


def test_assemble_context_returns_context_result(tmp_path):
    """assemble_context returns a ContextResult with size_info."""
    task = {
        "name": "test",
        "done_when": "done",
        "context_bundle": [],
    }
    ctx = assemble_context(task, tmp_path)
    assert isinstance(ctx, ContextResult)
    assert isinstance(ctx.size_info, dict)
    assert "total_chars" in ctx.size_info
    assert "estimated_tokens" in ctx.size_info


def test_context_size_info_values(tmp_path):
    """size_info total_chars matches actual string length."""
    (tmp_path / "doc.md").write_text("Hello world content.")
    task = {
        "name": "test",
        "done_when": "done",
        "context_bundle": ["doc.md"],
    }
    ctx = assemble_context(task, tmp_path)
    assert ctx.size_info["total_chars"] == len(ctx)
    assert ctx.size_info["estimated_tokens"] == len(ctx) // 4


def test_context_result_behaves_as_string(tmp_path):
    """ContextResult works as a normal string for all operations."""
    task = {
        "name": "test",
        "done_when": "done",
        "context_bundle": [],
    }
    ctx = assemble_context(task, tmp_path)
    # String operations work
    assert "<definition>" in ctx
    assert ctx.startswith("<definition>")
    assert isinstance(ctx, str)


def test_context_size_grows_with_content(tmp_path):
    """Larger context bundles produce larger size_info values."""
    task_empty = {
        "name": "test",
        "done_when": "done",
        "context_bundle": [],
    }
    ctx_empty = assemble_context(task_empty, tmp_path)

    (tmp_path / "big.md").write_text("x" * 10000)
    task_big = {
        "name": "test",
        "done_when": "done",
        "context_bundle": ["big.md"],
    }
    ctx_big = assemble_context(task_big, tmp_path)

    assert ctx_big.size_info["total_chars"] > ctx_empty.size_info["total_chars"]
    assert (
        ctx_big.size_info["estimated_tokens"] > ctx_empty.size_info["estimated_tokens"]
    )


# ---------------------------------------------------------------------------
# Python symbol extraction tests (:: syntax)
# ---------------------------------------------------------------------------

SAMPLE_PY_SOURCE = '''\
"""Module docstring."""

import os
import sys
from pathlib import Path


def alpha():
    """First function."""
    return 1


def beta(x, y):
    """Second function."""
    return x + y


class Gamma:
    """A class."""

    def method(self):
        pass


def delta():
    return "last"
'''


def test_extract_single_function():
    """Extract a single top-level function."""
    result = _extract_python_symbols(SAMPLE_PY_SOURCE, ["alpha"])
    assert "def alpha():" in result
    assert "return 1" in result
    # Should not contain other symbols
    assert "def beta" not in result
    assert "class Gamma" not in result


def test_extract_multiple_symbols():
    """Extract multiple comma-separated symbols."""
    result = _extract_python_symbols(SAMPLE_PY_SOURCE, ["alpha", "delta"])
    assert "def alpha():" in result
    assert "def delta():" in result
    assert "def beta" not in result


def test_extract_class():
    """Extract a top-level class."""
    result = _extract_python_symbols(SAMPLE_PY_SOURCE, ["Gamma"])
    assert "class Gamma:" in result
    assert "def method(self):" in result
    assert "def alpha" not in result


def test_import_block_included():
    """The import block (with module docstring) is always prepended."""
    result = _extract_python_symbols(SAMPLE_PY_SOURCE, ["delta"])
    assert '"""Module docstring."""' in result
    assert "import os" in result
    assert "import sys" in result
    assert "from pathlib import Path" in result


def test_missing_symbol_warns_and_falls_back():
    """Missing symbol emits warning and returns full file content."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _extract_python_symbols(SAMPLE_PY_SOURCE, ["nonexistent"])
        assert len(w) == 1
        assert "nonexistent" in str(w[0].message)
        assert "not found" in str(w[0].message)
    # Full content returned as fallback
    assert "def alpha" in result
    assert "def beta" in result
    assert "class Gamma" in result


def test_partial_missing_symbol_falls_back():
    """If one of several requested symbols is missing, fall back to full file."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = _extract_python_symbols(SAMPLE_PY_SOURCE, ["alpha", "nonexistent"])
        assert len(w) == 1
    # Full file returned
    assert "def beta" in result
    assert "class Gamma" in result


def test_extract_with_decorators():
    """Decorators are included in the extraction."""
    source = """\
import os


@some_decorator
def decorated():
    pass


@first
@second
def multi_decorated():
    pass
"""
    result = _extract_python_symbols(source, ["decorated"])
    assert "@some_decorator" in result
    assert "def decorated():" in result
    assert "def multi_decorated" not in result

    result2 = _extract_python_symbols(source, ["multi_decorated"])
    assert "@first" in result2
    assert "@second" in result2
    assert "def multi_decorated():" in result2


def test_no_import_block():
    """Files without imports still work — just the symbols are returned."""
    source = """\
def only_func():
    return 42
"""
    result = _extract_python_symbols(source, ["only_func"])
    assert "def only_func():" in result
    assert "return 42" in result


# ---------------------------------------------------------------------------
# assemble_context integration tests for :: syntax
# ---------------------------------------------------------------------------


def test_assemble_context_symbol_extraction(tmp_path):
    """assemble_context with :: extracts only requested symbols."""
    (tmp_path / "mod.py").write_text(SAMPLE_PY_SOURCE)
    task = {
        "name": "test",
        "done_when": "done",
        "context_bundle": ["mod.py::alpha"],
    }
    ctx = assemble_context(task, tmp_path)
    assert "def alpha():" in ctx
    assert "def beta" not in ctx
    assert "import os" in ctx


def test_assemble_context_multiple_symbols(tmp_path):
    """assemble_context with :: and comma-separated symbols."""
    (tmp_path / "mod.py").write_text(SAMPLE_PY_SOURCE)
    task = {
        "name": "test",
        "done_when": "done",
        "context_bundle": ["mod.py::alpha,Gamma"],
    }
    ctx = assemble_context(task, tmp_path)
    assert "def alpha():" in ctx
    assert "class Gamma:" in ctx
    assert "def beta" not in ctx


def test_assemble_context_missing_symbol_fallback(tmp_path):
    """assemble_context falls back to full file when symbol is missing."""
    (tmp_path / "mod.py").write_text(SAMPLE_PY_SOURCE)
    task = {
        "name": "test",
        "done_when": "done",
        "context_bundle": ["mod.py::nonexistent"],
    }
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        ctx = assemble_context(task, tmp_path)
    # Full file included as fallback
    assert "def alpha" in ctx
    assert "def beta" in ctx
    assert "class Gamma" in ctx


def test_assemble_context_non_python_raises(tmp_path):
    """:: syntax on non-Python files raises ValueError."""
    (tmp_path / "readme.md").write_text("# Hello")
    task = {
        "name": "test",
        "done_when": "done",
        "context_bundle": ["readme.md::something"],
    }
    with pytest.raises(ValueError, match="only supported for Python"):
        assemble_context(task, tmp_path)


def test_assemble_context_missing_py_file_with_symbols(tmp_path):
    """:: syntax on a missing .py file produces a file-not-found warning."""
    task = {
        "name": "test",
        "done_when": "done",
        "context_bundle": ["missing.py::foo"],
    }
    ctx = assemble_context(task, tmp_path)
    assert "WARNING: File not found" in ctx
