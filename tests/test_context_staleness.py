"""Tests for context staleness detection.

Verifies that:
- File mtimes are recorded at task creation time
- Stale files are detected when mtimes change
- Staleness warnings appear in audit log at dispatch time
"""

import time
from pathlib import Path

from corc.context import (
    check_context_staleness,
    record_context_mtimes,
)
from corc.audit import AuditLog


# ---------------------------------------------------------------------------
# record_context_mtimes tests
# ---------------------------------------------------------------------------


def test_record_mtimes_captures_existing_files(tmp_path):
    """record_context_mtimes returns mtime for each existing file."""
    (tmp_path / "a.py").write_text("hello")
    (tmp_path / "b.py").write_text("world")

    mtimes = record_context_mtimes(["a.py", "b.py"], tmp_path)
    assert "a.py" in mtimes
    assert "b.py" in mtimes
    assert isinstance(mtimes["a.py"], float)
    assert isinstance(mtimes["b.py"], float)


def test_record_mtimes_skips_missing_files(tmp_path):
    """Missing files are silently skipped in mtime recording."""
    (tmp_path / "exists.py").write_text("hi")

    mtimes = record_context_mtimes(["exists.py", "missing.py"], tmp_path)
    assert "exists.py" in mtimes
    assert "missing.py" not in mtimes


def test_record_mtimes_strips_section_fragments(tmp_path):
    """Section fragments (#section) are stripped before recording mtime."""
    (tmp_path / "spec.md").write_text("# Spec\n\ncontent")

    mtimes = record_context_mtimes(["spec.md#design"], tmp_path)
    assert "spec.md" in mtimes
    assert "spec.md#design" not in mtimes


def test_record_mtimes_empty_bundle(tmp_path):
    """Empty context bundle returns empty dict."""
    mtimes = record_context_mtimes([], tmp_path)
    assert mtimes == {}


# ---------------------------------------------------------------------------
# check_context_staleness tests
# ---------------------------------------------------------------------------


def test_no_staleness_when_files_unchanged(tmp_path):
    """No staleness detected when files haven't changed."""
    (tmp_path / "a.py").write_text("hello")
    mtime = (tmp_path / "a.py").stat().st_mtime

    task = {
        "name": "test",
        "done_when": "done",
        "context_bundle": ["a.py"],
        "context_bundle_mtimes": {"a.py": mtime},
    }

    stale = check_context_staleness(task, tmp_path)
    assert stale == []


def test_staleness_detected_when_file_modified(tmp_path):
    """Staleness detected when a file's mtime changes after task creation."""
    f = tmp_path / "a.py"
    f.write_text("original")
    original_mtime = f.stat().st_mtime

    task = {
        "name": "test",
        "done_when": "done",
        "context_bundle": ["a.py"],
        "context_bundle_mtimes": {"a.py": original_mtime},
    }

    # Modify the file — ensure mtime changes
    time.sleep(0.05)
    f.write_text("modified content")

    stale = check_context_staleness(task, tmp_path)
    assert len(stale) == 1
    assert stale[0]["file"] == "a.py"
    assert stale[0]["recorded_mtime"] == original_mtime
    assert stale[0]["current_mtime"] != original_mtime


def test_staleness_only_reports_changed_files(tmp_path):
    """Only changed files appear in staleness report."""
    (tmp_path / "a.py").write_text("stable")
    (tmp_path / "b.py").write_text("will change")

    mtime_a = (tmp_path / "a.py").stat().st_mtime
    mtime_b = (tmp_path / "b.py").stat().st_mtime

    task = {
        "name": "test",
        "done_when": "done",
        "context_bundle": ["a.py", "b.py"],
        "context_bundle_mtimes": {"a.py": mtime_a, "b.py": mtime_b},
    }

    # Only modify b.py
    time.sleep(0.05)
    (tmp_path / "b.py").write_text("changed")

    stale = check_context_staleness(task, tmp_path)
    assert len(stale) == 1
    assert stale[0]["file"] == "b.py"


def test_no_staleness_when_no_mtimes_recorded(tmp_path):
    """No staleness when context_bundle_mtimes is missing (old tasks)."""
    task = {
        "name": "test",
        "done_when": "done",
        "context_bundle": ["a.py"],
    }

    stale = check_context_staleness(task, tmp_path)
    assert stale == []


def test_no_staleness_when_mtimes_empty(tmp_path):
    """No staleness when context_bundle_mtimes is empty dict."""
    task = {
        "name": "test",
        "done_when": "done",
        "context_bundle": ["a.py"],
        "context_bundle_mtimes": {},
    }

    stale = check_context_staleness(task, tmp_path)
    assert stale == []


# ---------------------------------------------------------------------------
# Integration: staleness warning in audit log via executor dispatch
# ---------------------------------------------------------------------------


def test_staleness_warning_logged_in_audit(tmp_path):
    """End-to-end: modify file after task creation, verify audit warning.

    This test simulates the dispatch flow without spinning up a full
    Executor (which requires many dependencies). Instead it directly
    calls the context staleness check and audit log to verify the
    contract.
    """
    # 1. Create a file and record its mtime (simulating task creation)
    src = tmp_path / "src" / "module.py"
    src.parent.mkdir(parents=True)
    src.write_text("def foo(): pass")

    mtimes = record_context_mtimes(["src/module.py"], tmp_path)
    assert "src/module.py" in mtimes

    # 2. Build a task dict with recorded mtimes
    task = {
        "id": "t-stale-1",
        "name": "implement-feature",
        "done_when": "tests pass",
        "context_bundle": ["src/module.py"],
        "context_bundle_mtimes": mtimes,
    }

    # 3. Modify the file (simulating time passing between plan and dispatch)
    time.sleep(0.05)
    src.write_text("def foo(): return 42  # changed")

    # 4. Check staleness (this is what executor.dispatch does)
    stale_files = check_context_staleness(task, tmp_path)
    assert len(stale_files) == 1
    assert stale_files[0]["file"] == "src/module.py"

    # 5. Log warning to audit log (simulating executor behavior)
    audit_dir = tmp_path / "audit"
    audit = AuditLog(audit_dir)

    stale_names = [s["file"] for s in stale_files]
    audit.log(
        "context_staleness_warning",
        task_id=task["id"],
        stale_files=stale_names,
        details=stale_files,
    )

    # 6. Verify warning appears in audit log
    events = audit.read_today()
    staleness_events = [
        e for e in events if e["event_type"] == "context_staleness_warning"
    ]
    assert len(staleness_events) == 1
    evt = staleness_events[0]
    assert evt["task_id"] == "t-stale-1"
    assert "src/module.py" in evt["stale_files"]
    assert len(evt["details"]) == 1
    assert evt["details"][0]["file"] == "src/module.py"


def test_no_warning_when_files_fresh(tmp_path):
    """No audit warning when files haven't changed since task creation."""
    src = tmp_path / "module.py"
    src.write_text("content")

    mtimes = record_context_mtimes(["module.py"], tmp_path)
    task = {
        "id": "t-fresh-1",
        "name": "fresh-task",
        "done_when": "done",
        "context_bundle": ["module.py"],
        "context_bundle_mtimes": mtimes,
    }

    # Don't modify the file — check staleness
    stale_files = check_context_staleness(task, tmp_path)
    assert stale_files == []


def test_record_and_check_roundtrip(tmp_path):
    """Full roundtrip: record → modify → check detects change."""
    (tmp_path / "config.yaml").write_text("key: value")
    (tmp_path / "readme.md").write_text("# Readme")

    # Record at "creation time"
    mtimes = record_context_mtimes(
        ["config.yaml", "readme.md"], tmp_path
    )

    task = {
        "name": "test",
        "done_when": "done",
        "context_bundle": ["config.yaml", "readme.md"],
        "context_bundle_mtimes": mtimes,
    }

    # Immediately check — should be fresh
    assert check_context_staleness(task, tmp_path) == []

    # Modify one file
    time.sleep(0.05)
    (tmp_path / "config.yaml").write_text("key: new_value")

    # Now should detect staleness
    stale = check_context_staleness(task, tmp_path)
    assert len(stale) == 1
    assert stale[0]["file"] == "config.yaml"
