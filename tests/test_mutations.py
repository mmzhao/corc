"""Tests for the mutation log."""

import json
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
        _validate_mutation({"seq": 1, "ts": "now", "type": "bad_type", "data": {}, "reason": "test"})


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
