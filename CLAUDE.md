# CORC — Claude Orchestration System

## Project Overview

A lightweight, modular orchestration layer for Claude Code. Python-based CLI tool that provides persistent knowledge management, deterministic DAG-based workflows, multi-agent coordination, cost visibility, and operator control over quarter-length engineering projects. See `SPEC.md` for the full specification (v0.4) and `PLAN.md` for the phased implementation plan.

**CORC is the first project managed by itself** — the system builds itself, providing immediate feedback.

## Current Status

**Phase 1: Foundation** — not yet started.

## Architecture

- **Language**: Python
- **Three data layers**:
  - Layer 1: Knowledge Store — markdown files (source of truth) + SQLite (derived index)
  - Layer 2: Work State — mutation log JSONL (source of truth) + SQLite (materialized view)
  - Layer 3: Audit Log — append-only JSONL event log
- **Search**: FTS5 keyword + sentence-transformers semantic + hybrid
- **CLI**: Click-based CLI (`corc` command)
- **Orchestrator**: Deterministic Python process (NOT an LLM) that manages all control flow
- **Agents**: Independent `claude -p` invocations in git worktrees, each with constrained tools/budget/schema
- **Determinism**: DAGs define control flow, hooks enforce pre/post conditions, "done when" criteria validate outputs
- **Knowledge curation**: Agents report findings; only orchestrator/operator curates the knowledge store

## Key Design Decisions

- **Orchestrator is deterministic code, not an LLM** — Python makes all control flow decisions
- **Three data layers** — knowledge (what we know), work state (what's happening), audit log (what happened)
- **Mutation log as source of truth for work state** — SQLite derived, replayable, Git-tracked
- **Tiered knowledge write access** — agents report findings, orchestrator curates
- **Context budget: 100k tokens** (10% of 1M window) per invocation
- **Two-tier context**: curated bundles (deterministic, primary) + RAG search (supplementary, agent-initiated)
- **Compaction strategy** — at 90% context, save progress (PR/commit) and hand off to new agent
- **Scout → implement → review pipeline** — research before building, adversarial review after
- **No Agent Teams initially** — orchestrator dispatches independent agents via `claude -p`
- **Modular notifications** — terminal first, pluggable for Slack/Discord/Telegram/email

## Development Guidelines

- Keep it simple — don't over-engineer
- Every module must have a CLI interface
- Modules communicate via files and shell commands, not internal APIs
- Markdown files are source of truth for knowledge; mutation log for work state
- SQLite indexes are always derived and rebuildable
- Use Obsidian-compatible conventions (YAML frontmatter, [[wikilinks]], #tags)
- Graceful degradation: system must work without optional features (vector search)
- All database operations use transactions for atomicity
- Content-hash change detection to avoid re-indexing unchanged files
- Every `claude -p` invocation must specify: allowed tools, max budget, output schema (when applicable), system prompt
- Agents never write to the knowledge store — they report findings as structured output
- Every task has machine-testable "done when" criteria (no subjective criteria)
- Agents produce proof-of-work artifacts alongside PRs
- Trust level dial: guardrails configurable from strict (current) to autonomous (future)
- Build the thinnest end-to-end slice first, test on real work, then expand (see PLAN.md)
