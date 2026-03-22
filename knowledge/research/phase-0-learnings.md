---
id: research-phase-0-learnings
type: research
project: corc
repos: [corc]
tags: [phase-0, learnings, bootstrap, retrospective, dispatch-loop]
created: 2026-03-21T00:00:00Z
updated: 2026-03-21T00:00:00Z
source: human
status: active
---

# Phase 0 Learnings: Foundation Bootstrap

## Question

What did we learn from Phase 0 (building the CORC foundation in a single Claude Code session) that should inform future phases?

## Summary

Phase 0 successfully built the complete dispatch loop and infrastructure in a single Claude Code session. Key learnings: the single-session bootstrap strategy works well for establishing shared context; the three-layer data architecture (knowledge, work state, audit) proved sound; FTS5 keyword search is a reliable baseline before adding semantic search; and the mutation log as source of truth enables clean crash recovery. The system is now capable of managing its own development starting Phase 1A.

## Details

### What Was Built

Phase 0 delivered the full dispatch loop and supporting infrastructure:

1. **Project structure**: Python package with Click CLI (`corc` command), 12 core modules
2. **Data foundations**:
   - Mutation log with flock write safety and schema validation
   - Work state SQLite rebuilt from mutation log on boot
   - Audit log with daily JSONL rotation
   - Session logging for full agent conversation capture
3. **Knowledge store (minimal)**: FTS5 keyword search, document add/get/search/stats/reindex
4. **Core loop**: task create → context assembly → dispatch → validation → completion
5. **Dispatch abstraction**: `AgentDispatcher` interface with `ClaudeCodeDispatcher` implementation
6. **Validation rules**: `file_exists`, `file_not_empty`, `tests_pass`, `contains_pattern`
7. **Operator tools**: `corc status`, `corc watch` (Rich live TUI), `corc self-test`

### Key Architectural Decisions Validated

- **Single-session bootstrap works**: Building the entire foundation in one Claude Code session ensures full context coherence. No handoff information loss. The session has complete awareness of every design decision and implementation detail.
- **Mutation log as source of truth**: The pattern of JSONL source of truth + SQLite materialized view works cleanly. Write safety via flock prevents concurrent corruption. Schema validation catches malformed writes before they enter the log. Replay rebuilds the view correctly.
- **FTS5 as reliable baseline**: FTS5 with porter stemming provides solid keyword search out of the box. The decision to make semantic search optional (graceful fallback) means the system works immediately, with embeddings as an enhancement rather than a requirement.
- **Content-hash change detection**: Comparing content hashes before reindexing avoids unnecessary work. Files that are touched (mtime changes) but not modified (same hash) are correctly skipped.
- **Click for CLI**: Click provides a clean, composable CLI with good help text generation. The group/command pattern maps well to CORC's module structure (task, knowledge, template, etc.).

### What Worked Well

- **Heading-based chunking**: Splitting markdown by headings produces semantically coherent chunks. Headings are natural boundaries — better than fixed token windows that split mid-thought.
- **Template system**: Document templates with variable substitution (${id}, ${title}, ${project}) ensure consistent frontmatter across all document types. New document types are easy to add.
- **Deterministic context assembly**: `corc context-for-task` resolves file paths at dispatch time and produces the same output given the same task + files. This is the foundation for reproducible agent behavior.
- **Rich for TUI**: The Rich library provides good live display capabilities without the complexity of a full TUI framework like Textual. Sufficient for Phase 0's single-panel event stream.

### What Needs Improvement (Phase 1A+)

- **Knowledge store needs YAML frontmatter on all docs**: Documents without frontmatter get minimal metadata (auto-generated IDs, "note" type). All seed documents should have proper frontmatter for accurate type classification and searchability.
- **Hybrid search needed**: FTS5 alone doesn't capture semantic similarity. A query for "how does the system handle crashes" won't find documents about "resilience" or "recovery" unless those exact keywords appear. Semantic search (Phase 1A) will bridge this gap.
- **Index freshness at scale**: The pre-query mtime check works but may become slow with hundreds of documents. Phase 1A should benchmark and consider background reindexing.
- **No daemon yet**: All dispatch is manual (`corc dispatch TASK_ID`). The daemon (Phase 1B) will automate the ready-task → dispatch → validate → complete loop.
- **No DAG resolution yet**: Tasks can declare `depends_on` but there's no topological sort or ready-task identification. Phase 1B adds the scheduler module.

### Patterns to Carry Forward

1. **Build the thinnest slice first**: Phase 0 built the minimal loop that works end-to-end. Each subsequent phase adds one capability while keeping the loop running.
2. **Test with real work immediately**: The first task dispatched through the loop is a real task, not a synthetic benchmark. This catches usability issues early.
3. **Source of truth in portable formats**: Markdown for knowledge, JSONL for state and events. SQLite is always derived. Nothing is locked in.
4. **Graceful degradation**: Optional features (semantic search, embeddings) fail gracefully. The system always works at its baseline capability.
5. **Session logging for debugging**: Full agent conversation logs are essential for understanding failures and improving prompts.

### Research Findings Applied

Phase 0 incorporated findings from several research areas:

- **sqlite-memory, OpenClaw/clawbot, ClawMem**: Informed the knowledge store design — markdown → chunking → FTS5 + embeddings → SQLite. Content-hash change detection and transactional safety from these projects.
- **Beads**: Dependency-aware task DAG model, hash-based IDs, `--ready` semantics adopted for the scheduler design.
- **Adjutant system**: Tiered knowledge write access (agents report, orchestrator curates), mutation log as source of truth, curated context bundles.
- **Claude Code hooks research**: 25 hook events identified. Hooks are deterministic guarantees vs. CLAUDE.md suggestions. All hooks fire in `claude -p` mode except PermissionRequest.

## Sources

- SPEC.md v0.5 — CORC Orchestration System Specification
- PLAN.md — CORC Implementation Plan
- Yusuke Endoh benchmark (March 2026) — "Which Programming Language Is Best for Claude Code?"
- sqlite-memory (sqliteai) — SQLite extension with hybrid semantic search
- OpenClaw/clawbot — Local-first RAG using SQLite
- Beads (Steve Yegge) — Structured task/issue tracker for AI coding agents
- Adjutant system — Multi-agent coordination reference architecture

## Recommendations

1. **Seed the knowledge store now**: Before Phase 1A adds hybrid search, populate the store with the spec, architecture decisions, and these learnings. This provides a real corpus for testing search improvements.
2. **Add frontmatter to all existing docs**: The spec file in `knowledge/architecture/` needs proper YAML frontmatter for accurate indexing.
3. **Prioritize hybrid search**: The gap between keyword and semantic search is significant for natural-language queries. This is the highest-value Phase 1A feature.
4. **Keep the mutation log compact**: As work progresses, the log will grow. Plan for compaction (snapshot + truncate) before it becomes a performance issue.
5. **Test context assembly quality**: The assembled context bundles need real-world validation. Do agents get what they need? Do they fall back to RAG search often? Track this.
