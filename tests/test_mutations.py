"""Tests for the mutation log."""

import json
import multiprocessing
import tempfile
from pathlib import Path

import pytest

from corc.mutations import MutationLog, _validate_mutation


def test_append_and_read(tmp_path):
    ml = MutationLog(tmp_path / "mutations.jsonl")
    entry = ml.append("task_created", {"id": "abc", "name": "test"}, reason="test")

    assert entry["seq"] == 1
    assert entry["type"] == "task_created"
    assert entry["data"]["id"] == "abc"

    entries = ml.read_all()
    assert len(entries) == 1
    assert entries[0]["seq"] == 1


def test_sequential_seq(tmp_path):
    ml = MutationLog(tmp_path / "mutations.jsonl")
    ml.append("task_created", {"id": "a", "name": "first"}, reason="test")
    ml.append("task_created", {"id": "b", "name": "second"}, reason="test")
    ml.append("task_started", {}, reason="test", task_id="a")

    entries = ml.read_all()
    assert [e["seq"] for e in entries] == [1, 2, 3]


def test_read_since(tmp_path):
    ml = MutationLog(tmp_path / "mutations.jsonl")
    ml.append("task_created", {"id": "a", "name": "first"}, reason="test")
    ml.append("task_created", {"id": "b", "name": "second"}, reason="test")
    ml.append("task_started", {}, reason="test", task_id="a")

    since = ml.read_since(1)
    assert len(since) == 2
    assert since[0]["seq"] == 2


def test_validate_rejects_bad_type():
    with pytest.raises(ValueError, match="Unknown mutation type"):
        _validate_mutation(
            {"seq": 1, "ts": "now", "type": "bad_type", "data": {}, "reason": "test"}
        )


def test_validate_rejects_missing_fields():
    with pytest.raises(ValueError, match="missing required fields"):
        _validate_mutation({"seq": 1, "type": "task_created"})


def test_empty_log(tmp_path):
    ml = MutationLog(tmp_path / "mutations.jsonl")
    assert ml.read_all() == []
    assert ml.read_since(0) == []


def test_task_id_included(tmp_path):
    ml = MutationLog(tmp_path / "mutations.jsonl")
    entry = ml.append("task_started", {}, reason="test", task_id="abc123")
    assert entry["task_id"] == "abc123"


def _writer(path: str, n_appends: int) -> None:
    """Worker function for concurrent writer test (must be top-level for pickling)."""
    ml = MutationLog(Path(path))
    for _ in range(n_appends):
        ml.append("task_created", {"writer": "concurrent"}, reason="concurrent test")


def test_concurrent_writers(tmp_path):
    """10 concurrent process writers × 10 appends each → 100 unique sequential seqs."""
    log_path = tmp_path / "mutations.jsonl"
    n_writers = 10
    n_appends = 10

    processes = []
    for _ in range(n_writers):
        p = multiprocessing.Process(target=_writer, args=(str(log_path), n_appends))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    # Verify all processes exited cleanly
    for p in processes:
        assert p.exitcode == 0, f"Writer process exited with code {p.exitcode}"

    ml = MutationLog(log_path)
    entries = ml.read_all()

    assert len(entries) == n_writers * n_appends, (
        f"Expected {n_writers * n_appends} entries, got {len(entries)}"
    )

    seqs = [e["seq"] for e in entries]
    # All seq numbers must be unique
    assert len(set(seqs)) == len(seqs), f"Duplicate seq numbers found: {seqs}"
    # Must be strictly sequential 1..100
    assert sorted(seqs) == list(range(1, n_writers * n_appends + 1))
