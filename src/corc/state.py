"""Work state — SQLite materialized view of the mutation log.

Rebuilt from mutations.jsonl on boot. All reads go through SQLite for speed.
All writes go through the mutation log for durability.
"""

import json
import sqlite3
from pathlib import Path

from corc.mutations import MutationLog

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'pending',
    role TEXT,
    agent_id TEXT,
    depends_on TEXT DEFAULT '[]',
    done_when TEXT NOT NULL,
    checklist TEXT DEFAULT '[]',
    context_bundle TEXT DEFAULT '[]',
    pr_url TEXT,
    proof_of_work TEXT,
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    completed TEXT,
    findings TEXT DEFAULT '[]',
    micro_deviations TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    task_id TEXT,
    status TEXT DEFAULT 'idle',
    worktree_path TEXT,
    pid INTEGER,
    started TEXT,
    last_activity TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
"""


class WorkState:
    def __init__(self, db_path: Path, mutation_log: MutationLog):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.mutation_log = mutation_log
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self._replay_mutations()

    def _replay_mutations(self):
        row = self.conn.execute("SELECT value FROM meta WHERE key='last_seq'").fetchone()
        last_seq = int(row["value"]) if row else 0
        entries = self.mutation_log.read_since(last_seq)
        for entry in entries:
            self._apply_mutation(entry)
        if entries:
            self.conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('last_seq', ?)",
                (str(entries[-1]["seq"]),),
            )
            self.conn.commit()

    def _apply_mutation(self, entry: dict):
        t = entry["type"]
        data = entry["data"]
        task_id = entry.get("task_id")

        if t == "task_created":
            self.conn.execute(
                """INSERT OR REPLACE INTO tasks(id, name, description, status, role, depends_on,
                   done_when, checklist, context_bundle, created, updated)
                   VALUES(?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data["id"],
                    data["name"],
                    data.get("description", ""),
                    data.get("role", "implementer"),
                    json.dumps(data.get("depends_on", [])),
                    data["done_when"],
                    json.dumps(data.get("checklist", [])),
                    json.dumps(data.get("context_bundle", [])),
                    entry["ts"],
                    entry["ts"],
                ),
            )
        elif t == "task_assigned":
            self.conn.execute(
                "UPDATE tasks SET status='assigned', agent_id=?, updated=? WHERE id=?",
                (data["agent_id"], entry["ts"], task_id),
            )
        elif t == "task_started":
            self.conn.execute(
                "UPDATE tasks SET status='running', updated=? WHERE id=?",
                (entry["ts"], task_id),
            )
        elif t == "task_completed":
            self.conn.execute(
                """UPDATE tasks SET status='completed', completed=?, updated=?,
                   pr_url=?, proof_of_work=?, findings=? WHERE id=?""",
                (
                    entry["ts"],
                    entry["ts"],
                    data.get("pr_url"),
                    json.dumps(data.get("proof_of_work")) if data.get("proof_of_work") else None,
                    json.dumps(data.get("findings", [])),
                    task_id,
                ),
            )
        elif t == "task_failed":
            self.conn.execute(
                "UPDATE tasks SET status='failed', updated=?, findings=? WHERE id=?",
                (entry["ts"], json.dumps(data.get("findings", [])), task_id),
            )
        elif t == "task_updated":
            updates = []
            params = []
            for field in ("status", "checklist", "pr_url", "proof_of_work", "findings", "micro_deviations"):
                if field in data:
                    val = data[field]
                    if isinstance(val, (list, dict)):
                        val = json.dumps(val)
                    updates.append(f"{field}=?")
                    params.append(val)
            if updates:
                updates.append("updated=?")
                params.append(entry["ts"])
                params.append(task_id)
                self.conn.execute(
                    f"UPDATE tasks SET {', '.join(updates)} WHERE id=?", params
                )
        elif t == "task_handed_off":
            self.conn.execute(
                "UPDATE tasks SET status='handed_off', updated=? WHERE id=?",
                (entry["ts"], task_id),
            )
        elif t == "agent_created":
            self.conn.execute(
                """INSERT OR REPLACE INTO agents(id, role, task_id, status, worktree_path, pid, started)
                   VALUES(?, ?, ?, 'idle', ?, ?, ?)""",
                (data["id"], data["role"], data.get("task_id"), data.get("worktree_path"), data.get("pid"), entry["ts"]),
            )
        elif t == "agent_updated":
            updates = []
            params = []
            for field in ("status", "task_id", "worktree_path", "pid", "last_activity"):
                if field in data:
                    updates.append(f"{field}=?")
                    params.append(data[field])
            if updates:
                params.append(data.get("agent_id") or entry.get("task_id"))
                self.conn.execute(
                    f"UPDATE agents SET {', '.join(updates)} WHERE id=?", params
                )
        elif t in ("pause", "resume"):
            pass  # Pause state tracked via .corc/pause.lock, not SQLite

        self.conn.commit()

    def get_task(self, task_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def list_tasks(self, status: str | None = None) -> list[dict]:
        if status:
            rows = self.conn.execute("SELECT * FROM tasks WHERE status=?", (status,)).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM tasks ORDER BY created").fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_ready_tasks(self) -> list[dict]:
        """Tasks whose dependencies are all completed."""
        all_tasks = self.list_tasks()
        completed_ids = {t["id"] for t in all_tasks if t["status"] == "completed"}
        ready = []
        for task in all_tasks:
            if task["status"] != "pending":
                continue
            deps = json.loads(task["depends_on"]) if isinstance(task["depends_on"], str) else task["depends_on"]
            if all(dep in completed_ids for dep in deps):
                ready.append(task)
        return ready

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        for field in ("depends_on", "checklist", "context_bundle", "findings", "micro_deviations"):
            if d.get(field) and isinstance(d[field], str):
                try:
                    d[field] = json.loads(d[field])
                except json.JSONDecodeError:
                    pass
        if d.get("proof_of_work") and isinstance(d["proof_of_work"], str):
            try:
                d["proof_of_work"] = json.loads(d["proof_of_work"])
            except json.JSONDecodeError:
                pass
        return d

    def refresh(self):
        self._replay_mutations()
