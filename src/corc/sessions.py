"""Session logging — full agent conversation capture.

Stores the complete output of each agent invocation for debugging,
retry context, and rating.
"""

import json
import time
from pathlib import Path


class SessionLogger:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def session_path(self, task_id: str, attempt: int = 1) -> Path:
        return self.base_dir / f"{task_id}-attempt-{attempt}.jsonl"

    def log_entry(self, task_id: str, attempt: int, entry_type: str, content: str, **kwargs):
        path = self.session_path(task_id, attempt)
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "type": entry_type,
            "content": content,
        }
        entry.update(kwargs)
        with open(path, "a") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")

    def log_dispatch(self, task_id: str, attempt: int, prompt: str, system_prompt: str, tools: list[str], budget: float):
        self.log_entry(task_id, attempt, "dispatch", prompt,
                       system_prompt=system_prompt, tools=tools, budget_usd=budget)

    def log_output(self, task_id: str, attempt: int, output: str, exit_code: int, duration_s: float):
        self.log_entry(task_id, attempt, "output", output,
                       exit_code=exit_code, duration_s=duration_s)

    def log_stream_event(self, task_id: str, attempt: int, event: dict):
        """Log a single streaming event to the session log.

        Each event is written immediately (one JSONL line per call) for
        crash safety — if the process dies mid-stream, all previously
        written events are preserved.
        """
        self.log_entry(
            task_id, attempt, "stream_event",
            json.dumps(event, separators=(",", ":")),
            stream_type=event.get("type", "unknown"),
        )

    def log_validation(self, task_id: str, attempt: int, passed: bool, details: str):
        self.log_entry(task_id, attempt, "validation", details, passed=passed)

    def read_session(self, task_id: str, attempt: int = 1) -> list[dict]:
        path = self.session_path(task_id, attempt)
        if not path.exists():
            return []
        entries = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def get_latest_attempt(self, task_id: str) -> int:
        attempts = sorted(self.base_dir.glob(f"{task_id}-attempt-*.jsonl"))
        if not attempts:
            return 0
        return int(attempts[-1].stem.split("-attempt-")[1])
