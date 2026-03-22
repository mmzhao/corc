"""Mutation log — source of truth for work state.

Append-only JSONL with flock write safety and schema validation.
SQLite work state is derived from this log and can be rebuilt at any time.
"""

import fcntl
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

MUTATION_TYPES = {
    "task_created",
    "task_updated",
    "task_assigned",
    "task_started",
    "task_completed",
    "task_failed",
    "task_escalated",
    "task_handed_off",
    "task_paused",
    "agent_created",
    "agent_updated",
    "pause",
    "resume",
    "escalation_created",
    "escalation_resolved",
    "finding_approved",
    "finding_rejected",
}

REQUIRED_FIELDS = {"seq", "ts", "type", "data", "reason"}


def _validate_mutation(entry: dict) -> None:
    missing = REQUIRED_FIELDS - set(entry.keys())
    if missing:
        raise ValueError(f"Mutation missing required fields: {missing}")
    if entry["type"] not in MUTATION_TYPES:
        raise ValueError(f"Unknown mutation type: {entry['type']}. Valid: {MUTATION_TYPES}")


class MutationLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _next_seq(self) -> int:
        if not self.path.exists():
            return 1
        last_seq = 0
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    last_seq = entry.get("seq", last_seq)
        return last_seq + 1

    def append(self, mutation_type: str, data: dict, reason: str, task_id: str | None = None) -> dict:
        entry = {
            "seq": self._next_seq(),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "type": mutation_type,
            "data": data,
            "reason": reason,
        }
        if task_id is not None:
            entry["task_id"] = task_id

        _validate_mutation(entry)

        line = json.dumps(entry, separators=(",", ":")) + "\n"

        fd = os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_APPEND)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, line.encode())
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

        return entry

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        entries = []
        with open(self.path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def read_since(self, seq: int) -> list[dict]:
        return [e for e in self.read_all() if e["seq"] > seq]
