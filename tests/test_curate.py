"""Tests for knowledge curation — tiered write access quality gate.

Tests the CurationEngine and the CLI curate command:
- Approve findings -> written to knowledge store with source=orchestrator
- Reject findings -> logged to mutation log with reason
- Rejection count tracking by finding type
- Blacklist suggestions after 3+ rejections of same type
- CLI integration
"""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from corc.audit import AuditLog
from corc.cli import cli
from corc.config import get_paths
from corc.curate import CurationEngine, Finding, CurationResult, BLACKLIST_THRESHOLD
from corc.knowledge import KnowledgeStore
from corc.mutations import MutationLog
from corc.state import WorkState


@pytest.fixture
def tmp_project(tmp_path, monkeypatch):
    """Create a minimal project structure for testing."""
    (tmp_path / ".git").mkdir()  # So get_project_root finds it
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "events").mkdir(parents=True)
    (tmp_path / "data" / "sessions").mkdir(parents=True)
    (tmp_path / "knowledge").mkdir()
    (tmp_path / ".corc").mkdir()

    monkeypatch.setattr("corc.config.get_project_root", lambda: tmp_path)

    paths = get_paths(tmp_path)
    ml = MutationLog(paths["mutations"])
    ws = WorkState(paths["state_db"], ml)
    al = AuditLog(paths["events_dir"])
    ks = KnowledgeStore(paths["knowledge_dir"], paths["knowledge_db"])

    return paths, ml, ws, al, ks


def _create_completed_task_with_findings(ml, ws, task_id="t1", findings=None):
    """Helper to create a completed task with findings in the mutation log."""
    if findings is None:
        findings = [
            "API endpoint /health returns 200",
            "Database connection pool needs tuning",
            {"type": "architecture", "content": "Consider splitting service into microservices"},
        ]

    ml.append("task_created", {
        "id": task_id,
        "name": f"Test task {task_id}",
        "done_when": "tests pass",
        "role": "implementer",
    }, reason="Test setup")

    ml.append("task_completed", {
        "findings": findings,
        "pr_url": "https://github.com/test/pr/1",
    }, reason="Test setup", task_id=task_id)

    ws.refresh()


# --- Finding class tests ---

class TestFinding:
    def test_from_raw_string(self):
        f = Finding.from_raw(0, "Some finding text", task_id="t1")
        assert f.index == 0
        assert f.content == "Some finding text"
        assert f.finding_type == "general"
        assert f.source_task_id == "t1"

    def test_from_raw_dict_with_type(self):
        f = Finding.from_raw(1, {"type": "architecture", "content": "Split services"}, task_id="t1")
        assert f.index == 1
        assert f.content == "Split services"
        assert f.finding_type == "architecture"

    def test_from_raw_dict_without_type(self):
        f = Finding.from_raw(2, {"content": "Just content"})
        assert f.finding_type == "general"
        assert f.content == "Just content"


# --- CurationEngine tests ---

class TestCurationEngine:
    def test_get_findings_from_completed_task(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        _create_completed_task_with_findings(ml, ws)
        engine = CurationEngine(ws, ml, al, ks)

        findings = engine.get_findings("t1")
        assert len(findings) == 3
        assert findings[0].content == "API endpoint /health returns 200"
        assert findings[0].finding_type == "general"
        assert findings[2].finding_type == "architecture"
        assert findings[2].content == "Consider splitting service into microservices"

    def test_get_findings_task_not_found(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        engine = CurationEngine(ws, ml, al, ks)

        with pytest.raises(ValueError, match="not found"):
            engine.get_findings("nonexistent")

    def test_get_findings_task_not_completed(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        ml.append("task_created", {
            "id": "t2",
            "name": "Incomplete task",
            "done_when": "tests pass",
        }, reason="Test")
        ws.refresh()

        engine = CurationEngine(ws, ml, al, ks)
        with pytest.raises(ValueError, match="not 'completed'"):
            engine.get_findings("t2")

    def test_get_findings_empty(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        _create_completed_task_with_findings(ml, ws, task_id="t3", findings=[])

        engine = CurationEngine(ws, ml, al, ks)
        findings = engine.get_findings("t3")
        assert findings == []

    def test_approve_finding_writes_to_knowledge_store(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        _create_completed_task_with_findings(ml, ws)
        engine = CurationEngine(ws, ml, al, ks)

        findings = engine.get_findings("t1")
        doc_id = engine.approve_finding("t1", findings[0], doc_type="note", project="test-proj")

        # Verify written to knowledge store
        doc = ks.get(doc_id)
        assert doc is not None
        assert doc["source"] == "orchestrator"
        assert "API endpoint /health returns 200" in doc["content"]

        # Verify mutation log entry
        mutations = ml.read_all()
        approved = [m for m in mutations if m["type"] == "finding_approved"]
        assert len(approved) == 1
        assert approved[0]["data"]["finding_content"] == "API endpoint /health returns 200"
        assert approved[0]["data"]["doc_id"] == doc_id
        assert approved[0]["task_id"] == "t1"

    def test_approve_finding_with_structured_type(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        _create_completed_task_with_findings(ml, ws)
        engine = CurationEngine(ws, ml, al, ks)

        findings = engine.get_findings("t1")
        # findings[2] is the architecture finding
        doc_id = engine.approve_finding("t1", findings[2])

        doc = ks.get(doc_id)
        assert doc is not None
        assert doc["source"] == "orchestrator"
        assert "Consider splitting service into microservices" in doc["content"]

    def test_reject_finding_logged_to_mutation_log(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        _create_completed_task_with_findings(ml, ws)
        engine = CurationEngine(ws, ml, al, ks)

        findings = engine.get_findings("t1")
        engine.reject_finding("t1", findings[1], reason="Not actionable")

        # Verify mutation log entry
        mutations = ml.read_all()
        rejected = [m for m in mutations if m["type"] == "finding_rejected"]
        assert len(rejected) == 1
        assert rejected[0]["data"]["finding_content"] == "Database connection pool needs tuning"
        assert rejected[0]["data"]["rejection_reason"] == "Not actionable"
        assert rejected[0]["data"]["finding_type"] == "general"
        assert rejected[0]["task_id"] == "t1"

    def test_reject_finding_stored_in_state(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        _create_completed_task_with_findings(ml, ws)
        engine = CurationEngine(ws, ml, al, ks)

        findings = engine.get_findings("t1")
        engine.reject_finding("t1", findings[0], reason="Trivial")

        # Refresh state and check rejection tracking
        ws.refresh()
        counts = ws.get_rejection_counts()
        assert counts.get("general", 0) == 1

    def test_rejection_counts_by_type(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        engine = CurationEngine(ws, ml, al, ks)

        # Create multiple tasks with findings and reject them
        for i in range(5):
            task_id = f"task-{i}"
            _create_completed_task_with_findings(ml, ws, task_id=task_id, findings=[
                {"type": "architecture", "content": f"Architecture finding {i}"},
                {"type": "performance", "content": f"Performance finding {i}"},
                "General finding",
            ])
            findings = engine.get_findings(task_id)
            # Reject architecture findings
            engine.reject_finding(task_id, findings[0], reason="Not relevant")
            # Reject some performance findings
            if i < 3:
                engine.reject_finding(task_id, findings[1], reason="Premature optimization")

        counts = engine.get_rejection_counts()
        assert counts["architecture"] == 5
        assert counts["performance"] == 3

    def test_blacklist_suggestions_after_threshold(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        engine = CurationEngine(ws, ml, al, ks)

        # Create tasks and reject findings to reach threshold
        for i in range(BLACKLIST_THRESHOLD):
            task_id = f"task-bl-{i}"
            _create_completed_task_with_findings(ml, ws, task_id=task_id, findings=[
                {"type": "style", "content": f"Style finding {i}"},
            ])
            findings = engine.get_findings(task_id)
            engine.reject_finding(task_id, findings[0], reason="Style nit")

        suggestions = engine.get_blacklist_suggestions()
        assert len(suggestions) == 1
        assert suggestions[0]["finding_type"] == "style"
        assert suggestions[0]["rejection_count"] == BLACKLIST_THRESHOLD
        assert "blacklist" in suggestions[0]["suggestion"].lower()

    def test_no_blacklist_suggestions_below_threshold(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        engine = CurationEngine(ws, ml, al, ks)

        # Create fewer rejections than threshold
        for i in range(BLACKLIST_THRESHOLD - 1):
            task_id = f"task-nbl-{i}"
            _create_completed_task_with_findings(ml, ws, task_id=task_id, findings=[
                {"type": "docs", "content": f"Docs finding {i}"},
            ])
            findings = engine.get_findings(task_id)
            engine.reject_finding(task_id, findings[0], reason="Not needed")

        suggestions = engine.get_blacklist_suggestions()
        assert len(suggestions) == 0

    def test_blacklist_suggestions_include_reasons(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        engine = CurationEngine(ws, ml, al, ks)

        reasons = ["Too vague", "Already known", "Not actionable"]
        for i in range(BLACKLIST_THRESHOLD):
            task_id = f"task-reasons-{i}"
            _create_completed_task_with_findings(ml, ws, task_id=task_id, findings=[
                {"type": "refactor", "content": f"Refactor suggestion {i}"},
            ])
            findings = engine.get_findings(task_id)
            engine.reject_finding(task_id, findings[0], reason=reasons[i % len(reasons)])

        suggestions = engine.get_blacklist_suggestions()
        assert len(suggestions) == 1
        assert len(suggestions[0]["recent_reasons"]) > 0

    def test_mixed_approve_reject_flow(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        _create_completed_task_with_findings(ml, ws)
        engine = CurationEngine(ws, ml, al, ks)

        findings = engine.get_findings("t1")

        # Approve first finding
        doc_id = engine.approve_finding("t1", findings[0])
        assert doc_id is not None

        # Reject second finding
        engine.reject_finding("t1", findings[1], reason="Not useful")

        # Approve third finding
        doc_id2 = engine.approve_finding("t1", findings[2])
        assert doc_id2 is not None

        # Verify knowledge store has 2 docs
        docs = ks.list_docs()
        assert len(docs) == 2

        # Verify mutation log has correct entries
        mutations = ml.read_all()
        approved = [m for m in mutations if m["type"] == "finding_approved"]
        rejected = [m for m in mutations if m["type"] == "finding_rejected"]
        assert len(approved) == 2
        assert len(rejected) == 1

    def test_approve_finding_tags_include_curated(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        _create_completed_task_with_findings(ml, ws)
        engine = CurationEngine(ws, ml, al, ks)

        findings = engine.get_findings("t1")
        doc_id = engine.approve_finding("t1", findings[0], tags=["test-tag"])

        doc = ks.get(doc_id)
        assert doc is not None
        # The tags should include curated and test-tag
        content = doc["content"]
        assert "curated" in content
        assert "source: orchestrator" in content


# --- CLI tests ---

class TestCurateCLI:
    def test_curate_lists_findings_non_interactive(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        _create_completed_task_with_findings(ml, ws)

        runner = CliRunner()
        result = runner.invoke(cli, ["curate", "t1", "--non-interactive"])
        assert result.exit_code == 0
        assert "API endpoint /health returns 200" in result.output
        assert "Database connection pool needs tuning" in result.output
        assert "architecture" in result.output

    def test_curate_task_not_found(self, tmp_project):
        runner = CliRunner()
        result = runner.invoke(cli, ["curate", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_curate_task_not_completed(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        ml.append("task_created", {
            "id": "pending1",
            "name": "Pending task",
            "done_when": "tests pass",
        }, reason="Test")
        ws.refresh()

        runner = CliRunner()
        result = runner.invoke(cli, ["curate", "pending1"])
        assert result.exit_code != 0
        assert "not 'completed'" in result.output

    def test_curate_no_findings(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        _create_completed_task_with_findings(ml, ws, task_id="empty1", findings=[])

        runner = CliRunner()
        result = runner.invoke(cli, ["curate", "empty1"])
        assert result.exit_code == 0
        assert "No findings" in result.output

    def test_curate_approve_all(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        _create_completed_task_with_findings(ml, ws)

        runner = CliRunner()
        result = runner.invoke(cli, ["curate", "t1", "--approve-all"])
        assert result.exit_code == 0
        assert "Approved" in result.output
        assert "3" in result.output  # 3 approved

        # Verify all findings written to knowledge store
        docs = ks.list_docs()
        assert len(docs) == 3

    def test_curate_reject_all(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        _create_completed_task_with_findings(ml, ws)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "curate", "t1", "--reject-all",
            "--reject-reason", "Not useful",
        ])
        assert result.exit_code == 0
        assert "Rejected" in result.output

        # Verify all findings logged as rejected
        mutations = ml.read_all()
        rejected = [m for m in mutations if m["type"] == "finding_rejected"]
        assert len(rejected) == 3
        for r in rejected:
            assert r["data"]["rejection_reason"] == "Not useful"

    def test_curate_interactive_approve(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        _create_completed_task_with_findings(ml, ws, findings=["Single finding"])

        runner = CliRunner()
        # Simulate interactive input: approve
        result = runner.invoke(cli, ["curate", "t1"], input="a\n")
        assert result.exit_code == 0
        assert "Approved" in result.output

        docs = ks.list_docs()
        assert len(docs) == 1

    def test_curate_interactive_reject(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        _create_completed_task_with_findings(ml, ws, findings=["Bad finding"])

        runner = CliRunner()
        # Simulate interactive input: reject + reason
        result = runner.invoke(cli, ["curate", "t1"], input="r\nLow quality\n")
        assert result.exit_code == 0
        assert "Rejected" in result.output

        mutations = ml.read_all()
        rejected = [m for m in mutations if m["type"] == "finding_rejected"]
        assert len(rejected) == 1
        assert rejected[0]["data"]["rejection_reason"] == "Low quality"

    def test_curate_interactive_skip(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        _create_completed_task_with_findings(ml, ws, findings=["Maybe later"])

        runner = CliRunner()
        result = runner.invoke(cli, ["curate", "t1"], input="s\n")
        assert result.exit_code == 0
        assert "Skipped" in result.output

        # Nothing written to knowledge store
        docs = ks.list_docs()
        assert len(docs) == 0

        # Nothing in rejected mutations either
        mutations = ml.read_all()
        rejected = [m for m in mutations if m["type"] == "finding_rejected"]
        assert len(rejected) == 0

    def test_curate_shows_blacklist_suggestion(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project

        # Create enough rejections to trigger suggestion
        for i in range(BLACKLIST_THRESHOLD):
            task_id = f"bl-task-{i}"
            _create_completed_task_with_findings(ml, ws, task_id=task_id, findings=[
                {"type": "style", "content": f"Style nit {i}"},
            ])
            ws.refresh()
            # Reject via engine directly to build up count
            engine = CurationEngine(ws, ml, al, ks)
            findings = engine.get_findings(task_id)
            engine.reject_finding(task_id, findings[0], reason="Style nit")

        # Now curate another task that also has a style finding
        _create_completed_task_with_findings(ml, ws, task_id="final-task", findings=[
            {"type": "style", "content": "Another style nit"},
        ])
        ws.refresh()

        runner = CliRunner()
        result = runner.invoke(cli, [
            "curate", "final-task", "--reject-all",
            "--reject-reason", "Style nit",
        ])
        assert result.exit_code == 0
        assert "Blacklist suggestions" in result.output
        assert "style" in result.output


# --- State integration tests ---

class TestStateRejectionTracking:
    def test_finding_rejection_stored_in_state_table(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        _create_completed_task_with_findings(ml, ws)

        engine = CurationEngine(ws, ml, al, ks)
        findings = engine.get_findings("t1")
        engine.reject_finding("t1", findings[0], reason="Test rejection")

        ws.refresh()
        counts = ws.get_rejection_counts()
        assert counts.get("general") == 1

    def test_rejection_counts_grouped_by_type(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        _create_completed_task_with_findings(ml, ws, findings=[
            {"type": "architecture", "content": "Arch finding"},
            {"type": "performance", "content": "Perf finding"},
            {"type": "architecture", "content": "Another arch finding"},
        ])

        engine = CurationEngine(ws, ml, al, ks)
        findings = engine.get_findings("t1")
        engine.reject_finding("t1", findings[0], reason="R1")
        engine.reject_finding("t1", findings[1], reason="R2")
        engine.reject_finding("t1", findings[2], reason="R3")

        ws.refresh()
        counts = ws.get_rejection_counts()
        assert counts["architecture"] == 2
        assert counts["performance"] == 1

    def test_state_rebuild_preserves_rejections(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        _create_completed_task_with_findings(ml, ws)

        engine = CurationEngine(ws, ml, al, ks)
        findings = engine.get_findings("t1")
        engine.reject_finding("t1", findings[0], reason="Before rebuild")

        # Rebuild state from scratch
        ws.rebuild()
        counts = ws.get_rejection_counts()
        assert counts.get("general") == 1


# --- Audit log integration ---

class TestAuditIntegration:
    def test_approve_creates_audit_event(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        _create_completed_task_with_findings(ml, ws)

        engine = CurationEngine(ws, ml, al, ks)
        findings = engine.get_findings("t1")
        engine.approve_finding("t1", findings[0])

        events = al.read_for_task("t1")
        approve_events = [e for e in events if e.get("event_type") == "finding_approved"]
        assert len(approve_events) == 1
        assert approve_events[0]["finding_type"] == "general"

    def test_reject_creates_audit_event(self, tmp_project):
        paths, ml, ws, al, ks = tmp_project
        _create_completed_task_with_findings(ml, ws)

        engine = CurationEngine(ws, ml, al, ks)
        findings = engine.get_findings("t1")
        engine.reject_finding("t1", findings[1], reason="Bad data")

        events = al.read_for_task("t1")
        reject_events = [e for e in events if e.get("event_type") == "finding_rejected"]
        assert len(reject_events) == 1
        assert reject_events[0]["reason"] == "Bad data"
