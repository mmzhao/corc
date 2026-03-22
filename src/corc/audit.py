"""Audit log — append-only event log.

Every tool call, task dispatch, completion, failure, cost.
Never modified, only appended. Immutable record.
"""

import json
import os
import time
from pathlib import Path


class AuditLog:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _today_path(self) -> Path:
        return self.base_dir / f"{time.strftime('%Y-%m-%d')}.jsonl"

    def log(self, event_type: str, task_id: str | None = None, **kwargs) -> dict:
        event = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "event_type": event_type,
        }
        if task_id:
            event["task_id"] = task_id
        event.update(kwargs)

        line = json.dumps(event, separators=(",", ":")) + "\n"
        path = self._today_path()
        with open(path, "a") as f:
            f.write(line)

        return event

    def read_today(self) -> list[dict]:
        return self._read_file(self._today_path())

    def read_all(self, since: str | None = None) -> list[dict]:
        events = []
        for path in sorted(self.base_dir.glob("*.jsonl")):
            events.extend(self._read_file(path))
        if since:
            events = [e for e in events if e.get("timestamp", "") >= since]
        return events

    def read_for_task(self, task_id: str) -> list[dict]:
        return [e for e in self.read_all() if e.get("task_id") == task_id]

    def _read_file(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        events = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events
