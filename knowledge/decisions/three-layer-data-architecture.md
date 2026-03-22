---
id: decision-three-layer-data
type: decision
project: corc
repos: [corc]
tags: [architecture, data, sqlite, knowledge-store, work-state, audit-log]
created: 2026-03-21T00:00:00Z
updated: 2026-03-21T00:00:00Z
source: human
status: active
---

# Three-Layer Data Architecture

## Context

CORC manages three distinct categories of information that have different lifecycles, write patterns, and durability requirements:

1. **What we know** — persistent knowledge that compounds over time (decisions, conventions, research)
2. **What's happening now** — operational state that changes rapidly (tasks, agents, progress)
3. **What happened** — immutable history for audit, analysis, and debugging

Mixing these in a single store creates problems: knowledge gets polluted with ephemeral state, operational state gets stale if tied to curated documents, and audit trails become hard to query when mixed with live data. Each category needs its own source of truth, write access model, and retention policy.

## Decision

**Three distinct data layers, each owning its domain. No duplication across layers.**

### Layer 1: Knowledge Store (What We Know)

- **Source of truth**: Markdown files in `knowledge/` with YAML frontmatter
- **Index**: SQLite at `data/knowledge.db` (derived, rebuildable via `corc reindex --full`)
- **Search**: FTS5 keyword + sentence-transformers semantic + hybrid (0.4/0.6 weighted)
- **Write access**: Only the orchestrator or operator curates. Agents report findings as structured output.
- **Sync**: Git tracks all changes to markdown files
- **Obsidian-compatible**: YAML frontmatter, [[wikilinks]], #tags

### Layer 2: Work State (What's Happening Now)

- **Source of truth**: Mutation log — append-only JSONL at `data/mutations.jsonl` (Git-tracked)
- **Materialized view**: SQLite at `data/state.db` (derived, rebuildable by replaying mutation log)
- **Write access**: Orchestrator writes mutations; agents update their own task status via CLI
- **Write safety**: All writes go through `flock` to prevent concurrent corruption
- **Schema enforcement**: Every mutation validated against JSON schema before write
- **Sync**: Git tracks the mutation log; SQLite is local-only and rebuilt on boot

### Layer 3: Audit Log (What Happened)

- **Source of truth**: JSONL files at `data/events/YYYY-MM-DD.jsonl`
- **Never modified**: Only appended. Immutable record.
- **Used for**: Observability, rating system, post-hoc analysis, cost tracking
- **Not Git-tracked**: Too verbose for git. Backed up via configurable strategy.

### How The Layers Interact

```
Agent completes task
  → Reports findings + PR URL via structured output
  → Orchestrator writes "task_completed" to mutation log (Layer 2)
  → Orchestrator writes "task_completed" event to audit log (Layer 3)
  → Orchestrator reviews findings, curates worthy ones into knowledge store (Layer 1)
  → Orchestrator updates DAG, dispatches next ready tasks
```

### Key Design Principles

- **Mutation log as source of truth for work state**: enables crash recovery (replay from log), audit trail (every state change recorded), resume on any machine (Git push/pull + local SQLite rebuild), and debuggability (grep the log).
- **SQLite is always derived**: both knowledge.db and state.db can be rebuilt from their respective sources (markdown files and mutation log). No data is locked in SQLite.
- **Portable formats**: markdown files, JSONL — standard formats that any tool can read. This is the graceful deprecation path.

## Consequences

### What Becomes Easier
- Each layer can evolve independently (change knowledge schema without touching work state)
- Crash recovery is straightforward (replay mutation log, reindex markdown files)
- Clear write access model prevents knowledge pollution from ephemeral agents
- Any layer can be replaced when Claude Code gains native capabilities
- Debugging is simple: grep JSONL logs, read markdown files, query SQLite views

### What Becomes Harder
- Three data stores to maintain instead of one
- Interactions between layers must be carefully orchestrated
- Consistency across layers requires the orchestrator to be correct

## Alternatives Considered

- **Single SQLite database**: Simpler initially but creates lock-in. SQLite as source of truth makes portability harder and crash recovery more complex. Rejected because we want markdown files and JSONL to be human-readable and Git-trackable.
- **All-in-one JSONL**: Simpler append model but search performance would be poor. No FTS5, no semantic search. Knowledge retrieval would be O(n) scanning.
- **PostgreSQL/external database**: Violates the zero-ops principle. Adds deployment complexity, credentials management, and a network dependency. Overkill for a single-user orchestration tool.
- **Two layers (knowledge + everything else)**: Conflates operational state with audit history. Makes it hard to rotate/archive verbose event logs without affecting live state.
