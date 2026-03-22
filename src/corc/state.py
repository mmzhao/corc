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
    context_bundle_mtimes TEXT DEFAULT '{}',
    pr_url TEXT,
    proof_of_work TEXT,
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    completed TEXT,
    findings TEXT DEFAULT '[]',
    micro_deviations TEXT DEFAULT '[]',
    attempt_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    merge_status TEXT
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

CREATE TABLE IF NOT EXISTS escalations (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    task_name TEXT,
    error TEXT,
    attempts INTEGER,
    session_log_path TEXT,
    suggested_actions TEXT DEFAULT '[]',
    done_when TEXT,
    status TEXT DEFAULT 'pending',
    resolution TEXT,
    created TEXT NOT NULL,
    resolved TEXT
);

CREATE TABLE IF NOT EXISTS finding_rejections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    finding_index INTEGER NOT NULL,
    finding_type TEXT NOT NULL DEFAULT 'general',
    finding_content TEXT NOT NULL,
    rejection_reason TEXT NOT NULL,
    ts TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_finding_rejections_type ON finding_rejections(finding_type);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents(status);
CREATE INDEX IF NOT EXISTS idx_escalations_status ON escalations(status);
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
                   done_when, checklist, context_bundle, context_bundle_mtimes,
                   created, updated, max_retries)
                   VALUES(?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data["id"],
                    data["name"],
                    data.get("description", ""),
                    data.get("role", "implementer"),
                    json.dumps(data.get("depends_on", [])),
                    data["done_when"],
                    json.dumps(data.get("checklist", [])),
                    json.dumps(data.get("context_bundle", [])),
                    json.dumps(data.get("context_bundle_mtimes", {})),
                    entry["ts"],
                    entry["ts"],
                    data.get("max_retries", 3),
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
                "UPDATE tasks SET status='failed', updated=?, findings=?, attempt_count=? WHERE id=?",
                (entry["ts"], json.dumps(data.get("findings", [])),
                 data.get("attempt_count", data.get("attempt", 0)), task_id),
            )
        elif t == "task_escalated":
            self.conn.execute(
                "UPDATE tasks SET status='escalated', updated=?, attempt_count=? WHERE id=?",
                (entry["ts"], data.get("attempt_count", data.get("attempt", 0)), task_id),
            )
        elif t == "task_updated":
            updates = []
            params = []
            for field in ("status", "checklist", "pr_url", "proof_of_work", "findings", "micro_deviations", "attempt_count", "max_retries", "merge_status"):
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
        elif t == "task_pending_merge":
            self.conn.execute(
                """UPDATE tasks SET status='pending_merge', updated=?,
                   proof_of_work=?, findings=? WHERE id=?""",
                (
                    entry["ts"],
                    json.dumps(data.get("proof_of_work")) if data.get("proof_of_work") else None,
                    json.dumps(data.get("findings", [])),
                    task_id,
                ),
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
        elif t == "escalation_created":
            self.conn.execute(
                """INSERT OR REPLACE INTO escalations(id, task_id, task_name, error,
                   attempts, session_log_path, suggested_actions, done_when, status, created)
                   VALUES(?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
                (
                    data["escalation_id"],
                    data["task_id"],
                    data.get("task_name", ""),
                    data.get("error", ""),
                    data.get("attempts", 0),
                    data.get("session_log_path", ""),
                    json.dumps(data.get("suggested_actions", [])),
                    data.get("done_when", ""),
                    entry["ts"],
                ),
            )
        elif t == "escalation_resolved":
            self.conn.execute(
                "UPDATE escalations SET status='resolved', resolution=?, resolved=? WHERE id=?",
                (data.get("resolution", ""), entry["ts"], data["escalation_id"]),
            )
        elif t == "finding_approved":
            pass  # Knowledge store writes handled by CurationEngine
        elif t == "finding_rejected":
            self.conn.execute(
                """INSERT INTO finding_rejections(task_id, finding_index, finding_type,
                   finding_content, rejection_reason, ts)
                   VALUES(?, ?, ?, ?, ?, ?)""",
                (
                    task_id,
                    data.get("finding_index", 0),
                    data.get("finding_type", "general"),
                    data.get("finding_content", ""),
                    data.get("rejection_reason", ""),
                    entry["ts"],
                ),
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
        """Tasks whose dependencies are all completed, including retriable failed tasks.

        Returns pending tasks with satisfied dependencies, plus failed tasks
        that haven't exceeded their max_retries limit.
        """
        all_tasks = self.list_tasks()
        completed_ids = {t["id"] for t in all_tasks if t["status"] == "completed"}
        ready = []
        for task in all_tasks:
            if task["status"] == "pending":
                deps = json.loads(task["depends_on"]) if isinstance(task["depends_on"], str) else task["depends_on"]
                if all(dep in completed_ids for dep in deps):
                    ready.append(task)
            elif task["status"] == "failed":
                # Include failed tasks that haven't exceeded max_retries
                attempt_count = task.get("attempt_count", 0)
                max_retries = task.get("max_retries", 3)
                if attempt_count <= max_retries:
                    ready.append(task)
        return ready

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        for field in ("depends_on", "checklist", "context_bundle", "context_bundle_mtimes", "findings", "micro_deviations"):
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

    def get_agents_for_task(self, task_id: str) -> list[dict]:
        """Get all agent records associated with a task."""
        rows = self.conn.execute(
            "SELECT * FROM agents WHERE task_id=?", (task_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def list_agents(self, status: str | None = None) -> list[dict]:
        """List all agents, optionally filtered by status."""
        if status:
            rows = self.conn.execute(
                "SELECT * FROM agents WHERE status=?", (status,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM agents").fetchall()
        return [dict(r) for r in rows]

    def list_escalations(self, status: str | None = None) -> list[dict]:
        """List escalations, optionally filtered by status."""
        if status:
            rows = self.conn.execute(
                "SELECT * FROM escalations WHERE status=? ORDER BY created", (status,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM escalations ORDER BY created"
            ).fetchall()
        return [self._escalation_to_dict(r) for r in rows]

    def get_escalation(self, escalation_id: str) -> dict | None:
        """Get an escalation by ID."""
        row = self.conn.execute(
            "SELECT * FROM escalations WHERE id=?", (escalation_id,)
        ).fetchone()
        if row is None:
            return None
        return self._escalation_to_dict(row)

    def _escalation_to_dict(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        if d.get("suggested_actions") and isinstance(d["suggested_actions"], str):
            try:
                d["suggested_actions"] = json.loads(d["suggested_actions"])
            except json.JSONDecodeError:
                pass
        return d

    def rebuild(self):
        """Full rebuild: clear all data and replay entire mutation log.

        Used on daemon startup to ensure SQLite matches the mutation log
        exactly. This is the recovery mechanism after crashes.
        """
        self.conn.execute("DELETE FROM tasks")
        self.conn.execute("DELETE FROM agents")
        self.conn.execute("DELETE FROM escalations")
        self.conn.execute("DELETE FROM finding_rejections")
        self.conn.execute("DELETE FROM meta")
        self.conn.commit()

        entries = self.mutation_log.read_all()
        for entry in entries:
            self._apply_mutation(entry)
        if entries:
            self.conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('last_seq', ?)",
                (str(entries[-1]["seq"]),),
            )
        self.conn.commit()

    def get_rejection_counts(self) -> dict[str, int]:
        """Get finding rejection counts grouped by finding_type."""
        rows = self.conn.execute(
            "SELECT finding_type, COUNT(*) as cnt FROM finding_rejections GROUP BY finding_type"
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def refresh(self):
        self._replay_mutations()
