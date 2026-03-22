# CORC — Implementation Plan

## Guiding Principle

**Build the thinnest end-to-end slice first.** Each phase produces a working system tested on real work. The TUI is the primary way the operator understands the system — every other phase includes a TUI milestone.

The loop: **plan tasks → inject context → dispatch agent → validate output → store results → learn from it.**

## Bootstrap Strategy

**Phase 0 is built in a single Claude Code session (pre-CORC).** One session writes the entire dispatch loop and first layer of infrastructure. This gives the session full context of everything it builds — no handoffs, no information loss, no dispatching through an untested system.

**Phase 1A onward is built by CORC managing itself.** The knowledge store is the first real test. By that point the dispatch loop is solid, there's a TUI to see what's happening, session logs to debug failures, and schema validation to prevent silent corruption.

---

## Phase 0: Foundation (Single Claude Code Session)

### Goal
Build the complete dispatch loop and enough infrastructure that CORC can manage its own development starting in Phase 1A. This is all built in one Claude Code session — not managed by CORC.

### Build

**Data foundations:**
- [ ] Python project structure: `pyproject.toml`, `src/corc/`, `tests/`
- [ ] `corc` CLI skeleton (Click)
- [ ] Mutation log: append-only JSONL with flock write safety and schema validation on every write
- [ ] Work state SQLite: tasks, agents tables. Rebuilt from mutation log on boot.
- [ ] Audit log: append events to `data/events/YYYY-MM-DD.jsonl`
- [ ] Session logging: full agent conversation to `data/sessions/TASK_ID-ATTEMPT.jsonl`
- [ ] Minimal knowledge store: `corc add`, `corc get`, `corc search` (FTS5 only)

**Core loop:**
- [ ] `corc task create` — write tasks to work state
- [ ] `corc context-for-task TASK_ID` — deterministic context assembly from bundle paths
- [ ] Dispatch abstraction layer: `AgentDispatcher` interface + `ClaudeCodeDispatcher`
- [ ] `corc dispatch TASK_ID` — dispatch one agent via abstraction layer
- [ ] "Done when" validation rules: `file_exists`, `file_not_empty`, `tests_pass`, `contains_pattern`

**Operator tools:**
- [ ] `corc status` — text snapshot
- [ ] **TUI v0: `corc watch`** — single-panel `rich` live display, color-coded events
- [ ] `corc self-test` — orchestrator test suite

### Test
Dispatch one real task through the loop: "Implement something small." Create the task manually, dispatch it, watch in the TUI, validate the output, review the code. This verifies the loop works before we rely on it.

### Success Criteria
- The dispatch loop completes end-to-end: task created → context assembled → agent dispatched → output validated → task marked complete
- `corc watch` shows events in real time
- Session log captures the full agent conversation
- Mutation log schema validation catches malformed writes
- `corc self-test` passes

### What This Enables
After Phase 0, CORC can:
- Store and track tasks with dependencies
- Assemble deterministic context from file paths
- Dispatch `claude -p` agents with constrained tools and budget
- Validate outputs against "done when" criteria
- Log everything (mutations, audit events, full sessions)
- Show live progress in a TUI

This is enough infrastructure to manage its own development.

---

## Phase 1A: Knowledge Store (First CORC-Managed Phase)

### Goal
Full knowledge store with hybrid search. **This is the first work managed by CORC** — tasks are created in work state, dispatched through the loop, validated by "done when" criteria.

### Build
- [ ] Frontmatter parser (all document types)
- [ ] Content chunking (heading-based, ~500 tokens)
- [ ] SQLite index schema + migration
- [ ] Indexing pipeline: scan → hash → parse → FTS5 + embed
- [ ] Index freshness: pre-query mtime/hash check, write-through on mutation
- [ ] Semantic search (sentence-transformers, cosine similarity)
- [ ] Hybrid search (0.4 keyword + 0.6 semantic, configurable)
- [ ] `corc reindex --full`, `corc stats`, `corc template TYPE`
- [ ] Document templates
- [ ] Seed knowledge store: this spec, architecture decisions, Phase 0 learnings

### How It's Managed
Tasks are created manually via `corc task create` (no `corc plan` yet — that's Phase 1B). Each task has a context bundle pointing to the relevant SPEC.md section + any existing code. Tasks are dispatched via `corc dispatch` or manually queued for when we have the daemon. The operator watches via `corc watch` and reviews output manually.

This is CORC managing work, but without the daemon or planning session yet. Dispatch is manual.

### Test
- Unit tests for every search mode
- Integration: index 20 docs, verify relevance
- **Real test**: assemble context bundles for Phase 1B tasks using the knowledge store. Does hybrid search find better docs than FTS5-only?

### Gate
Rate the Phase 1A experience. Did CORC-managed dispatch produce good code? What was painful? What did the session logs reveal about agent behavior? Adjust the plan.

---

## Phase 1B: Daemon + Workflow Engine + Planning

### Goal
The daemon runs as a background loop. Tasks are planned interactively. DAG-based execution with validation and retry. Second TUI: DAG view.

### Build

**Daemon (three modules connected through work state):**
- [ ] Scheduler: DAG resolution, topological sort, ready-task identification
- [ ] Executor: dispatch via abstraction layer, poll completion, retry logic
- [ ] Processor: "done when" validation, state updates, finding collection
- [ ] Daemon loop: Scheduler → Executor → Processor, poll every 5s
- [ ] `corc start [--parallel N]` / `corc stop`
- [ ] Daemon reconciliation on startup: rebuild SQLite from mutation log, check running tasks (PID liveness + worktree state), reconcile or retry
- [ ] Pause switch: `corc pause` / `corc resume`

**Workflow engine:**
- [ ] YAML task definition parser + validator
- [ ] "Done when" validation rules: `file_exists`, `file_not_empty`, `tests_pass`, `contains_pattern`, `json_schema`
- [ ] "Done when" quality linter: reject subjective criteria at plan time
- [ ] Retry policies: same → enriched (include session log from failed attempt) → escalate
- [ ] Escalation records: structured context, session log path, suggested actions
- [ ] `corc escalations`, `corc escalation show ESC_ID`, `corc escalation resolve ESC_ID`
- [ ] Structured sub-progress checklists
- [ ] Proof-of-work artifact generation

**Planning:**
- [ ] `corc plan [FILE]` — interactive Claude session with system context injected:
  - Knowledge store (searchable via `corc search`)
  - Current work state (all tasks, statuses, dependencies)
  - Repo context (code structure, conventions)
  - Curation blacklist
  - Spec template for consistent structure
- [ ] Three-stage flow: spec development → task decomposition → review & commit
- [ ] Planning reasoning captured in spec's "Rationale" section
- [ ] Spec saved to knowledge store + tasks written to work state on approval
- [ ] Crash-safe: spec and tasks written to disk/state as session progresses
- [ ] `corc plan --resume` from last saved state
- [ ] Quick task path: Claude determines if task needs full decomposition or is a single task

**Operator tools:**
- [ ] **TUI v1: `corc watch`** — two panels: DAG status (top) + event stream (bottom)
- [ ] **`corc dag`** — static DAG visualization with color-coded status + progress bar
- [ ] `corc log --last N` — human-readable formatted events
- [ ] `corc tasks [--status S] [--ready]`

### Test
- Unit tests: YAML parsing, DAG resolution, scheduler/executor/processor independently
- Property tests: DAG order respects dependencies, context assembly is deterministic
- Daemon restart test: kill daemon, restart, verify reconciliation
- Escalation test: exhaust retries, verify structured escalation record
- **Real test**: Plan Phase 1A as a workflow with `corc plan`. Run with `corc start`. Watch in `corc watch`. Does the DAG view help? Does escalation UX work when tasks fail?

### Gate
Rate Phase 1 workflows. Is the TUI useful? Is `corc plan` producing good task decompositions? Calibrate "done when" strictness.

---

## Phase 2A: Parallel Execution

### Goal
Multiple agents in parallel with git worktrees. Third TUI: full dashboard with live agent checklist progress.

### Build

**Parallel dispatch:**
- [ ] Git worktree creation/cleanup per agent
- [ ] `corc start --parallel N`
- [ ] Optimistic merge on task completion
- [ ] Merge conflict detection + retry with merged state
- [ ] `corc start --task TASK_ID` (run one specific task)
- [ ] `corc start --once` (one ready task, then stop)

**Context refinement:**
- [ ] Dynamic bundle resolution at dispatch time (not plan time)
- [ ] Staleness check on context bundles (warn if referenced files changed since plan)
- [ ] Catch-up injection: state summary from recent mutations
- [ ] Full session log available as retry context
- [ ] Structured handoff docs: design decisions, alternatives considered, known issues, what to do next
- [ ] Code formatting hook: `PostToolUse[Write|Edit]` → ruff/black

**Operator tools:**
- [ ] **TUI v2: `corc watch` full dashboard** — three panels:
  - DAG with live status updates (top)
  - Color-coded event stream (middle)
  - Agent detail with live checklist progress (bottom)
- [ ] `corc costs --today`, `corc costs --project`

### Test
- Integration: two non-conflicting tasks in parallel
- Integration: deliberate merge conflict, verify retry with merged state
- Staleness: modify file after plan creation, verify warning
- Handoff: force context overflow, verify structured handoff + fresh agent pickup
- **Real test**: Run Phase 2B tasks with 2-3 parallel agents. Watch checklist items tick off live.

### Gate
Parallel vs. sequential: time savings, cost, quality. Is the full TUI dashboard usable? Is it the primary way you understand the system?

---

## Phase 2B: Roles + Curation + Merge Policies

### Goal
Agent specialization. Tiered knowledge write access. Blacklist. Merge policies.

### Build

**Roles:**
- [ ] Role YAML parser and composition (`extends`)
- [ ] Role constraint enforcement: system prompt, allowed tools, cost limits
- [ ] Scout: read-only research → structured brief
- [ ] Implementer: code gen → PR + proof-of-work
- [ ] Adversarial reviewer: context-reset (only diff + spec + done_when, no implementation history)
- [ ] Scout → implement → review pipeline pattern
- [ ] Authorized micro-deviations (≤5 lines, obviously correct, documented)

**Knowledge curation:**
- [ ] Tiered write access: agents produce findings, `corc curate RUN_ID` reviews and persists
- [ ] Curation rejection tracking in mutation log
- [ ] Blacklist (`.corc/blacklist.md`): injected into every agent's context
- [ ] Auto-add to blacklist after 3+ rejected findings of same type (operator confirms)

**Repo merge policies:**
- [ ] Per-repo config in `.corc/repos.yaml`: auto vs. human-only
- [ ] Protected branches, block auto-merge, block direct push
- [ ] `PreToolUse` hooks enforce merge policies

### Test
- Unit: role parsing, composition, constraint enforcement, merge policy hooks
- **Real test**: Scout → implement → review pipeline on a real task. Does the adversarial reviewer (with no implementation context) catch real issues? Does the blacklist prevent recurring mistakes?

### Gate
Rate each role independently. Is scout worth its cost? Is context-reset review finding things normal review misses?

---

## Phase 3A: Rating + Analysis

### Goal
Systematic quality scoring. Feedback loop from ratings to system behavior.

### Build

**Rating engine:**
- [ ] 7-dimension scoring (correctness, completeness, code quality, efficiency, determinism, resilience, human intervention)
- [ ] Auto-scoring via `claude -p` evaluator using spec as rubric
- [ ] Rating JSONL storage + trend tracking
- [ ] `corc rate --auto`, `corc ratings --trend`, `corc ratings --dimension`

**Analysis:**
- [ ] `corc analyze costs [--today] [--project]`
- [ ] `corc analyze failures`, `corc analyze duration`
- [ ] `corc analyze patterns` — correlate roles/task-types with scores, produce recommendations
- [ ] `corc analyze retries` — first-attempt success rates, adaptive retry settings
- [ ] `corc analyze prompts --role NAME` — scores by prompt version
- [ ] `corc analyze planning` — which spec structures produce better outcomes

**Feedback loop:**
- [ ] Adaptive retry: reduce retries for >90% success task types, increase for <50%
- [ ] Auto-adjustment suggestions: add scouts, rollback prompts, expand bundles (operator confirms)
- [ ] Trust level recommendations: suggest loosening/tightening guardrails based on scores
- [ ] Prompt version tracking: record which version per run, correlate with quality

**Observability:**
- [ ] Audit log backup (configurable path, interval, rotation)
- [ ] Cost alerts (configurable thresholds)
- [ ] `corc retro PROJECT_NAME` — project-level retrospective → saved to knowledge store

### Test
- Backfill-rate all runs from Phases 0-2
- **Real test**: Do ratings match your subjective assessment? Does pattern analysis produce useful recommendations? After adjusting a prompt based on data, do scores improve?

### Gate
Calibration checkpoint. Are scores meaningful? Is the feedback loop actionable?

---

## Phase 3B: Resilience + Notifications

### Goal
System handles failures gracefully. External notifications work.

### Build
- [ ] Chaos monkey: kill agents, corrupt state files, corrupt mutation log, SQLite failures
- [ ] Chaos monkey tests orchestrator modules (scheduler, executor, processor) not just agents
- [ ] Notification system: terminal + one external channel (Slack, Discord, or Telegram)
- [ ] `corc self-test` expanded: orchestrator resilience tests
- [ ] `corc dag --mermaid` for Obsidian/GitHub

### Test
- Kill agents at every lifecycle point, verify clean resume
- Corrupt orchestrator state, verify recovery via mutation log replay
- Notification test: escalation reaches external channel
- **Real test**: 10% kill rate during real work

### Gate
Resilience scorecard: >95% clean resume rate.

---

## Phase 4: Full Integration

### Goal
Run a real project end-to-end, fully managed by CORC.

### Build
- [ ] Repo manager: registry, context generation, `corc repo` CLI
- [ ] Trust level dial (strict → standard → relaxed → autonomous)
- [ ] Trust level recommendations from accumulated rating data
- [ ] "Replace when native" markers on all modules
- [ ] `corc deprecation-check` — scan for native Claude features that overlap
- [ ] `corc knowledge check` — contradiction detection prototype
- [ ] End-to-end documentation (for humans and AI)
- [ ] Performance optimization

### Test
- **The real test**: Pick a substantial project. `corc plan` to decompose. `corc start --parallel 3` to execute. Only intervene on escalation. Use `corc watch` as primary interface.

### Success Criteria
- Spec → DAG → execution works end-to-end
- Multiple agents in parallel without conflicts
- Failed agents resume cleanly with full context
- TUI dashboard is the primary operator interface
- Rating feedback loop produces actionable improvements
- Blacklist prevents recurring mistakes
- Knowledge store accumulates curated, high-quality knowledge
- \>70% first-try task success rate
- Cost predictable within 2x

---

## Phase Sequence

| Phase | Built By | Key Deliverable | TUI Milestone | Real Test |
|---|---|---|---|---|
| **0** | Single Claude Code session | Full dispatch loop + infrastructure | v0: event stream | One task through the loop |
| **1A** | CORC (manual dispatch) | Hybrid search | — | Context bundles for 1B |
| **1B** | CORC (manual dispatch) | Daemon + planning + DAG | v1: + DAG view | Phase 1A as a workflow |
| **2A** | CORC (daemon) | Parallel worktrees | v2: full dashboard | 2-3 agents, live checklist |
| **2B** | CORC (daemon) | Roles + curation + blacklist | — | Scout → implement → review |
| **3A** | CORC (daemon) | Rating + feedback loop | — | Backfill + calibrate |
| **3B** | CORC (daemon) | Chaos monkey + notifications | + mermaid export | 10% kill rate |
| **4** | CORC (daemon) | Quarter-length project | — | Full e2e |

Sequential. No time estimates. Phase 0 is pre-CORC. Phase 1A is the bootstrap — first work managed by CORC. Phase 1B introduces the daemon. Phase 2A+ is the full system.

---

## Risk Checkpoints

At each gate:
1. **Is CORC faster than raw Claude Code?** If not, something is wrong.
2. **Is the TUI the primary interface?** If you're still reading logs, fix the TUI.
3. **Is determinism worth its overhead?** Planning time vs. automation savings.
4. **Are agents producing good code?** If not, fix context assembly first.
5. **Is the rating system honest?** If everything scores 8+, recalibrate.
6. **Should we stop building a module?** Deprecate when native features replace it.
7. **Is the orchestrator reliable?** `corc self-test` at every gate.

---

## TUI Evolution

| Phase | What You See |
|---|---|
| **0** | Single panel: color-coded live events |
| **1B** | Two panels: DAG status + events. `corc dag` for static view. |
| **2A** | Three panels: DAG + events + agent detail with live checklist. This is the primary interface from here on. |
| **3B** | + Mermaid export for Obsidian/GitHub |
