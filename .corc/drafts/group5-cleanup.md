# Group 5: Cleanup & Process

Priority: 50

## 5A. Session log daily rotation (move to archive, never delete)

**Type:** implementation
**Priority:** 50
**Depends on:** none
**Done when:** Session logs older than configurable days (default 7) are moved to `data/sessions/archive/YYYY-MM-DD/`; audit logs older than configurable days moved to `data/events/archive/YYYY-MM-DD/`; `corc logs rotate` triggers manual rotation; daemon runs rotation once daily; no files are ever deleted, only moved; tests verify rotation moves files and originals are gone from active directory but present in archive
**Checklist:**
- Config: log_rotation_days in .corc/config.yaml (default 7)
- Move (not delete) old session logs to archive subdirectory
- Move (not delete) old audit logs to archive subdirectory
- `corc logs rotate` CLI for manual rotation
- Daemon runs rotation once daily (track last rotation timestamp)
- Archive directory structure: data/sessions/archive/YYYY-MM-DD/
- Tests verify files moved to archive, not deleted

**Context bundle:** src/corc/sessions.py, src/corc/audit.py, src/corc/daemon.py

---

## 5B. Planning process design for native corc plan UI

**Type:** investigation
**Priority:** 50
**Depends on:** 3A (query API — needed to understand what data the planning UI needs)
**Done when:** Design document produced describing: how the planning conversation should flow in a browser GUI; what data the planning UI needs from the query API; how draft tasks are created, reviewed, and approved in the UI; how the spec document and task DAG are linked; wireframes or text descriptions of key screens; documented as a knowledge store architecture document
**Checklist:**
- Describe the planning flow: spec → decomposition → review → approve
- What query API endpoints does the planning UI need?
- How does the operator review and modify tasks before approving?
- How is the spec document linked to its task DAG?
- How does the operator set per-task priority, type, context bundle?
- Text wireframes or descriptions of key planning screens
- Save as knowledge/architecture/planning-ui-design.md
