---
id: corc-spec-v05
type: architecture
project: corc
repos: [corc]
tags: [spec, architecture, orchestration, phase-0]
created: 2026-03-21T00:00:00Z
updated: 2026-03-21T00:00:00Z
source: human
status: active
---

# CORC — Orchestration System Specification

## Status: Draft
## Version: 0.5
## Last Updated: 2026-03-21

## 1. Problem Statement

Current AI coding tools (Claude Code, etc.) are powerful but lack:
- Persistent memory across sessions
- Deterministic, reproducible workflows
- Multi-repo awareness
- Cost and productivity visibility
- Quality gates and validation pipelines
- Structured context management to avoid compaction loss
- Coordination of quarter-length engineering projects end-to-end
- Operator visibility into what agents are doing and why
- Separation of knowledge (what we know) from work state (what's in progress)

This system fills those gaps as a lightweight, modular layer that works *with* Claude Code rather than replacing it. It's designed to last approximately one year and be progressively simplified as models gain native capabilities.

**Test project:** CORC itself is the first quarter-length project managed by this system. The system will be used to build itself, providing immediate feedback on what works and what doesn't.

### What Claude Code Is Good At (and Where It Needs Help)

Claude Code excels at two things:
1. **Being a generic algorithm that handles fuzzy tasks** — interpreting requirements, making judgment calls, reasoning about code
2. **Writing code** up to a certain level of complexity

The system should maximize use of these strengths while enforcing determinism everywhere else. Claude Code should never decide *what to do next* or *whether something passed*. It should only do the fuzzy, generative work within tightly constrained boundaries.

---

## 2. Core Philosophy

### Treat the LLM as an Algorithm, Not a Person
- Define clear inputs and expected outputs for every invocation
- Validate outputs programmatically, not hopefully
- Retry with enriched context on failure, escalate to human on repeated failure
- Never let the model make decisions above its pay grade — provide requirements upfront
- Minimize guesswork: if context is ambiguous, the system should fail fast, not improvise

### Determinism by Default, Flexibility by Design
- All control flow is deterministic — defined in DAGs, workflows, and hooks
- The model operates only within explicitly scoped steps
- Every point of agent flexibility is intentional, bounded, and low-risk
- See §3 (Determinism Architecture) for the full mapping

### File Over App
- Markdown files are the source of truth for all knowledge
- SQLite indexes are derived and rebuildable from source files
- Mutation logs (JSONL) are the source of truth for operational state; SQLite is a materialized view
- Git tracks all changes
- Any component can be replaced without data loss
- The system is Obsidian-compatible but does not depend on Obsidian

### Modularity Over Integration
- Each module has a CLI interface
- Modules communicate via files and shell commands, not internal APIs
- Any module can be removed or replaced independently
- Communication channels (notifications, escalation) are pluggable — terminal first, Slack/Discord/Telegram/email later
- Don't build what Claude Code will likely do natively within 6 months

### Resilience as a Feature
- Any agent can fail at any time without corrupting system state
- Failed agents resume from the last completed step with full context
- The system includes a chaos monkey mode to verify resilience
- State is persisted after every step, not just at workflow completion

### Knowledge Curation as a Quality Gate
- Low-level agents (implementers, scouts) **do not write to the knowledge store directly**
- They report structured findings as part of their task output
- Only the orchestrator or operator curates findings into the knowledge store
- This prevents knowledge pollution from ephemeral agents making permanent writes
- The knowledge store compounds quality over time because every entry is curated

### Operator Visibility and Control
- The human operator can see what every agent is doing in real time
- The human can pause the entire system with a single command
- Post-hoc analysis lets the operator understand what happened and why
- A critical rating system tracks system performance over time

### Designed to Be Replaced
- Every module is marked "replace when native" with a specific Claude Code feature that would supersede it
- Clean interfaces between modules so any can be swapped out independently
- A configurable "trust level" dial loosens guardrails as models improve (skip validation, expand tool access, reduce retries)
- The system should get simpler over time, not more complex
- If a Claude Code update makes a module unnecessary, deprecate it immediately

---

## 3. Determinism Architecture

### The Three Layers

| Layer | Mechanism | Enforced By | Can Agent Override? |
|---|---|---|---|
| **Structural** | Workflow DAGs define all control flow | Workflow engine (deterministic Python code) | No |
| **Enforcement** | Hooks guarantee pre/post conditions | Claude Code hooks | No |
| **Verification** | Every step has testable "done when" criteria | Validation rules + automated scoring | No |

### What Is Deterministic (Agent Cannot Influence)

| Component | How It's Enforced |
|---|---|
| **Step sequencing** | DAG defines execution order; engine resolves dependencies |
| **Tool availability** | `--allowedTools` per `claude -p` invocation restricts what the agent can use |
| **Output schema** | `--json-schema` on `claude -p` enforces structured output format |
| **File protection** | `PreToolUse` hooks block writes outside expected paths |
| **Cost limits** | `--max-budget-usd` and `--max-turns` per invocation |
| **Validation gates** | Post-step "done when" criteria (file exists, schema matches, tests pass) |
| **Quality gates** | `Stop` hooks with agent-based validation prevent premature completion |
| **Context injection** | Curated context bundles assembled deterministically per task |
| **Context preservation** | `SessionStart[compact]` hooks re-inject critical state after compaction |
| **Observability logging** | Async `PreToolUse` hooks log every tool call — no opt-out |
| **Pause switch** | Shared pause lock checked before every new step |
| **Knowledge write access** | Only orchestrator/operator writes to knowledge store; agents report findings |

### Where Agent Flexibility Is Allowed (and Why It's Safe)

| Flexible Area | Why It's Allowed | Why It's Low Risk |
|---|---|---|
| **Code generation** | This is what LLMs are good at — fuzzy, generative work | Output is validated by tests, linting, and adversarial review |
| **Search query formulation** | The model knows what it needs better than a static query | Results are ranked and limited; bad queries just return poor results, which trigger retry |
| **Error interpretation** | Novel errors require judgment | The model's interpretation feeds into a structured retry policy — the system decides what to do with it |
| **Document content** | Writing prose/analysis requires creativity | Structure is enforced by templates; content is reviewed by validation workflows |
| **Exploration within a step** | Investigating code, reading files, searching | Read-only operations; bounded by step timeout and tool restrictions |

### Hook Enforcement Map

```
SessionStart[startup]     → inject curated context bundle for current task
SessionStart[compact]     → re-inject critical context after compaction
UserPromptSubmit          → inject step-specific constraints and catch-up summary
PreToolUse[Write|Edit]    → block writes outside step's allowed paths
PreToolUse[Bash]          → block dangerous commands (rm -rf, force push, etc.)
PostToolUse[Write|Edit]   → trigger incremental reindex if file in knowledge/
PostToolUse[Write|Edit]   → run code formatter (ruff/black for Python) to enforce consistent style
PostToolUse[*]            → async: log tool call to audit log JSONL
Stop                      → agent-based validation: verify "done when" criteria
SubagentStart             → inject agent role context and constraints
SubagentStop              → extract and persist key outcomes
TaskCompleted             → validate task completion against "done when" before accepting
Notification              → forward to operator via configured channel
```

---

## 4. Research Findings

### Language Choice
A March 2026 benchmark (Yusuke Endoh, "Which Programming Language Is Best for Claude Code?") tested Claude Code Opus 4.6 implementing a simplified Git in 13 languages, 20 trials each:

- **Top 3**: Ruby (73s, $0.36), Python (75s, $0.38), JavaScript (81s, $0.39) — fast, cheap, low variance
- **Mid tier**: Go (102s, $0.50), Java (115s, $0.50), Rust (114s, $0.54) — 1.4-2x slower, much higher variance
- **Bottom tier**: TypeScript (133s, $0.62), C (156s, $0.74), Haskell (174s, $0.74)
- **Key insight**: Dynamic languages consistently outperform static ones for AI code generation. Type checking adds 1.6-3.2x overhead.
- **Decision**: Python is the primary language. Best AI ecosystem, fast generation, excellent SQLite tooling.

### Knowledge Store Architecture
The industry is converging on: markdown files → chunking → embeddings + FTS5 → SQLite storage.

Key projects studied:
- **sqlite-memory** (sqliteai): SQLite extension with hybrid semantic search, markdown-aware chunking, local embedding via llama.cpp. Sync between agents using CRDT.
- **OpenClaw/clawbot**: Local-first RAG using SQLite. Hybrid search combining vector similarity with FTS5 BM25 ranking. Graceful fallback if vector extensions fail.
- **ClawMem**: Portable vector memory DB. Single binary, single SQLite file. Segment-centroid routing for search. <10ms search at 10K memories.
- **Bitloops**: Local-first memory scoped to individual repositories. SQLite for metadata, HNSW for vector indexes. ~1-2MB per 1000 commits.

Design principles from the research:
- Zero-ops: no Postgres, no Docker, no credentials
- Local-first: knowledge base is a folder of markdown files on disk
- Resilience: system must work even if advanced features (vector search) fail
- Content-hash change detection: only re-index modified files
- Transactional safety: every sync operation in a SAVEPOINT transaction

### Beads & Task DAGs
**Beads** (Steve Yegge, ~18.7k GitHub stars) is a structured task/issue tracker for AI coding agents. Key insights adopted:
- **Dependency-aware task DAG** — tasks have explicit `blocks`, `depends_on`, `parent` relationships
- **`--ready` semantics** — "what can I work on right now?" finds tasks with no open blockers
- **Hash-based IDs** — prevent collisions when multiple agents create tasks in parallel
- **LLM-powered memory compaction** — `bd compact` summarizes old closed issues to save context

Beads tracks *what to do next*; our knowledge store tracks *what we know*. They're complementary. Our workflow engine adopts Beads' DAG model for task scheduling.

### Adjutant System (Reference Architecture)
A multi-agent coordination system with useful patterns we've adopted:
- **Tiered knowledge write access** — only the strategic agent (Adjutant) curates the knowledge store; operatives report findings
- **Mutation log as source of truth** — JSONL log of all state changes, SQLite as materialized view, replayable for resume
- **Context-reset adversarial review** — critics are deliberately context-stripped to prevent familiarity-driven agreeableness
- **Curated context bundles** — tasks specify exactly which documents to inject, curated at planning time
- **Catch-up injection** — on each turn, inject a status summary of what happened since last interaction
- **Scout as pipeline stage** — research before implementation as a formalized workflow pattern
- **PR-based operative output** — `gh pr create` as the standard completion artifact
- **"Done when" criteria** — every task has explicit, testable completion conditions

### Orchestrator Landscape
Key patterns from SWE-agent, OpenHands, Aider, Plandex, CrewAI, LangGraph, and others:
- **Layered autonomy** — advisory (prompts) → gated (permissions) → deterministic (hooks/validation). Systems moving toward configurable spectrums, not binary choices.
- **Context window as the critical resource** — every successful system treats context management as the central engineering challenge.
- **File-native configuration** — agent definitions, memory, workflows as plain files in the repo.
- **Explore → plan → implement** — separate research from execution to avoid solving the wrong problem.
- **Anti-pattern: treating agents as conversational partners** — the "LLM as algorithm" framing consistently outperforms "LLM as colleague" for deterministic tasks.

### Claude Code Hooks
25 hook events. Hooks are deterministic guarantees — unlike CLAUDE.md instructions, hooks *always* execute. Key capabilities:
- **4 hook types**: command (shell), HTTP (endpoints), prompt (single-turn LLM yes/no), agent (multi-turn subagent with full tool access, 60s timeout)
- **All hooks fire in `claude -p` mode** except `PermissionRequest` — use `PreToolUse` instead for automated runs
- **Async hooks** — fire-and-forget for logging/notifications that don't need to block
- **Matcher patterns** — regex filtering by tool name, session reason, agent type, etc.

### Claude Code Channels (Future Reference)
Push events into running sessions. Supports Telegram, Discord, webhooks, custom localhost. Useful for external notification integration. Requires v2.1.80+, research preview. May be used in later phases for external notification channels.

### Obsidian as Viewer
- Obsidian stores everything as plain markdown files locally
- Files with YAML frontmatter, [[wikilinks]], and #tags work natively in Obsidian
- The graph view provides useful visualization of knowledge connections
- Key insight: "Don't organize the vault purely for human browsing — optimize it for AI consumption"
- Decision: Use Obsidian-compatible conventions but don't depend on Obsidian as a component

---

## 5. Data Architecture

Three distinct layers, each owning its domain. No duplication across layers.

### Layer 1: Knowledge Store (What We Know)

Persistent, curated knowledge. Decisions, conventions, research findings, architecture docs, task outcomes.

- **Source of truth**: Markdown files in `knowledge/` with YAML frontmatter
- **Index**: SQLite (derived, rebuildable from files via `corc reindex --full`)
- **Write access**: Only the orchestrator or operator curates. Agents report findings as structured output; the orchestrator decides what to persist.
- **Search**: FTS5 keyword + sentence-transformers semantic + hybrid
- **Sync**: Git tracks all changes

### Layer 2: Work State (What's Happening Now)

Operational state. Tasks, agent assignments, progress, dependencies, "done when" criteria.

- **Source of truth**: Mutation log — append-only JSONL at `data/mutations.jsonl` (Git-tracked)
- **Materialized view**: SQLite at `data/state.db` (derived, rebuildable by replaying mutation log)
- **Write access**: Orchestrator process writes mutations; agents update their own task status via CLI
- **Sync**: Git tracks the mutation log; SQLite is local-only and rebuilt on boot

**Why mutation log as source of truth:**
- Crash recovery: replay from log to rebuild SQLite
- Audit trail: every state change is recorded with timestamp and reason
- Resume on any machine: Git push/pull moves the log; SQLite rebuilds locally
- Debuggability: you can grep the log to understand why the system is in any state

**Write safety:** All writes to the mutation log go through a file lock (`flock`) to prevent concurrent write corruption from the daemon and `corc plan` sessions. The lock is held only for the duration of the append — microseconds.

**Schema enforcement:** Every mutation entry is validated against a JSON schema before being written. Malformed entries are rejected with an error rather than silently corrupting the log. The schema is versioned — old entries remain valid under older schema versions, and a migration path exists for schema changes.

**Mutation log format:**
```json
{"seq": 1, "ts": "2026-03-21T10:00:00Z", "type": "task_created", "task_id": "abc123", "data": {"name": "implement-knowledge-store", "depends_on": [], "done_when": "All CLI commands work, tests pass"}, "reason": "Phase 1 kickoff"}
{"seq": 2, "ts": "2026-03-21T10:00:01Z", "type": "task_assigned", "task_id": "abc123", "data": {"agent_id": "agent-1", "role": "implementer"}, "reason": "Task ready, no blockers"}
{"seq": 3, "ts": "2026-03-21T10:30:00Z", "type": "task_completed", "task_id": "abc123", "data": {"status": "passed", "pr_url": "https://github.com/...", "findings": ["SQLite WAL mode required for concurrent reads"]}, "reason": "All done_when criteria met"}
```

**Work State SQLite Schema:**
```sql
CREATE TABLE tasks (
    id TEXT PRIMARY KEY,              -- Hash-based UUID
    name TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'pending',    -- pending | ready | assigned | running | completed | failed | blocked | handed_off
    role TEXT,                        -- Which agent role should handle this
    agent_id TEXT,                    -- Currently assigned agent (null if unassigned)
    depends_on TEXT,                  -- JSON array of task IDs
    done_when TEXT NOT NULL,          -- Testable completion criteria (machine-verifiable, not subjective)
    checklist TEXT,                   -- JSON array of sub-steps: [{"item": "...", "done": false}]
    context_bundle TEXT,              -- JSON array of file paths to inject as context
    pr_url TEXT,                      -- PR created on completion
    proof_of_work TEXT,               -- Structured "what I did and how to verify" artifact
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    completed TEXT,                   -- Timestamp of completion
    findings TEXT,                    -- JSON array of agent findings (not yet curated)
    micro_deviations TEXT             -- JSON array of small out-of-scope fixes agent made (documented)
);

CREATE TABLE task_dependencies (
    task_id TEXT REFERENCES tasks(id),
    depends_on_id TEXT REFERENCES tasks(id),
    PRIMARY KEY (task_id, depends_on_id)
);

CREATE TABLE agents (
    id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    task_id TEXT REFERENCES tasks(id),
    status TEXT DEFAULT 'idle',       -- idle | running | blocked | completed | failed
    worktree_path TEXT,               -- Git worktree for this agent
    pid INTEGER,                      -- OS process ID
    started TEXT,
    last_activity TEXT
);

CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_agent TEXT,
    to_agent TEXT,                    -- null = broadcast
    content TEXT NOT NULL,
    created TEXT NOT NULL,
    read INTEGER DEFAULT 0
);

-- Indexes
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_role ON tasks(role);
CREATE INDEX idx_agents_status ON agents(status);
CREATE INDEX idx_agents_task_id ON agents(task_id);
```

### Layer 3: Audit Log (What Happened)

Append-only event log. Every tool call, step execution, cost, validation result.

- **Source of truth**: JSONL files at `data/events/YYYY-MM-DD.jsonl`
- **Never modified**: Only appended. Immutable record.
- **Used for**: Observability, rating system, post-hoc analysis, cost tracking
- **Not Git-tracked**: Too verbose for git. Backed up via configurable strategy (see below).

**Event format:**
```json
{
  "timestamp": "2026-03-21T10:00:04.123Z",
  "event_type": "step_completed",
  "workflow": "code-review",
  "run_id": "uuid",
  "step": "gather-context",
  "task_id": "abc123",
  "status": "passed",
  "duration_s": 3.2,
  "tokens_in": 1200,
  "tokens_out": 450,
  "cost_usd": 0.08,
  "agent_role": "reviewer",
  "agent_id": "agent-1"
}
```

Event types: `task_created`, `task_assigned`, `task_started`, `task_completed`, `task_failed`, `step_started`, `step_completed`, `step_failed`, `step_retried`, `workflow_started`, `workflow_completed`, `workflow_failed`, `tool_call`, `escalation`, `pause`, `resume`, `rating`, `finding_reported`, `finding_rejected`, `knowledge_curated`, `context_injected`, `blacklist_updated`

**Audit log backup strategy:**
```yaml
# .corc/config.yaml
audit:
  backup_path: ~/corc-backups/audit/      # Configurable backup location
  backup_interval: daily                   # daily | weekly
  rotate_after_days: 90                    # Rotate log files older than this
```

Backup is a simple rsync/copy of `data/events/` and `data/sessions/` to the backup path. Runs as part of the daemon's daily cycle.

### How The Layers Interact

```
Agent completes task
  → Reports findings + PR URL via structured output
  → Orchestrator writes "task_completed" to mutation log (Layer 2)
  → Orchestrator writes "task_completed" event to audit log (Layer 3)
  → Orchestrator reviews findings, curates worthy ones into knowledge store (Layer 1)
  → Orchestrator updates DAG, dispatches next ready tasks
```

---

## 6. Context Management

### Context Budget

Starting allocation: **100,000 tokens** (10% of 1M context window). This is the maximum context injected into any single agent invocation.

Budget breakdown per invocation:
- **System prompt + role instructions**: ~5,000 tokens
- **Task definition + "done when"**: ~2,000 tokens
- **Curated context bundle**: ~60,000 tokens (the bulk)
- **Catch-up summary** (recent state changes): ~3,000 tokens
- **Headroom for conversation/tool output**: ~30,000 tokens

These are starting values — we'll calibrate during the CORC build.

### Two-Tier Context Assembly

Context assembly uses two tiers: a deterministic primary tier and a supplementary search tier.

**Tier 1: Curated context bundles (deterministic, primary)**

Each task specifies exactly which documents to inject, curated at planning time:

```yaml
tasks:
  - name: implement-fts5-search
    done_when: "FTS5 queries return ranked results; integration test passes"
    context_bundle:
      - knowledge/architecture/data-model.md
      - knowledge/decisions/sqlite-choice.md
      - knowledge/research/search-architectures.md
      - SPEC.md#module-1-knowledge-store    # Section reference
```

The orchestrator resolves these references to file content **at dispatch time** (not plan time — files may have changed). This is deterministic: same task definition + same files on disk = same context injected.

**Fallback**: If a referenced file doesn't exist, log a warning and continue with available context. If the total bundle exceeds the context budget, truncate oldest/lowest-priority documents.

**Tier 2: RAG search (supplementary, agent-initiated)**

Agents can search the knowledge store mid-task for additional context. This is a tool available to them, not part of the deterministic context assembly:

```
corc search "FTS5 tokenizer configuration"  # Agent calls this during work
```

RAG search is useful when:
- The agent encounters something unexpected that the curated bundle doesn't cover
- The planner needs to discover relevant docs when assembling context bundles
- A scout is researching a topic and needs to find related knowledge

RAG search is **not** used for primary context injection. The curated bundle is the backbone. RAG is a supplementary tool.

**The planner's role in context assembly:**
When the planner creates a task DAG, part of its job is assembling good context bundles for each task. The quality of context bundles is part of what gets rated. If the planner misses a critical doc, the implementer may still find it via RAG search, but this is a fallback — not the design intent.

### Compaction Strategy

With 1M token context windows and well-scoped tasks, compaction should be rare. But when it happens:

**Prevention (primary strategy):**
- Scope tasks small enough that they complete well within the context window
- Target: each task should complete in <500k tokens of conversation
- If a task is likely to exceed this, decompose it into smaller subtasks in the DAG

**Graceful handoff at 90% (when prevention fails):**
1. Agent detects it's at ~90% context usage (or a `PreCompact` hook fires)
2. Agent saves progress:
   - Commits current changes to the worktree branch
   - Creates a PR (even if partial/draft) documenting what's done
   - Reports findings and remaining work as structured output
3. Orchestrator captures the handoff:
   - Records completed substeps in work state
   - Marks the task as "handed-off" with a summary of remaining work
4. Orchestrator spawns a fresh agent for the remaining work:
   - New context includes: curated bundle + handoff summary + PR link
   - The new agent picks up from where the old one left off

**Context re-injection after compaction (if task continues in same session):**
- `SessionStart[compact]` hook fires
- Injects: task definition, "done when" criteria, curated context bundle (abbreviated), and a summary of work completed so far in this session
- This is deterministic — the hook runs the same script every time

### Deterministic Context Injection

Every new agent session gets context assembled by a deterministic script, not by the LLM:

```
corc context-for-task TASK_ID
```

This command:
1. Reads the task definition from work state (Layer 2)
2. Resolves the curated context bundle to file contents (Layer 1)
3. Generates a catch-up summary from recent mutations relevant to this task (Layer 2)
4. Reads the agent role's system prompt
5. Assembles everything into a single context document
6. Outputs it for injection via `--system-prompt` or `SessionStart` hook

**Same task + same state = same context injected.** This is the key determinism guarantee for context.

### Catch-Up Injection

On every new session or after compaction, the agent gets a brief summary of what happened since it last had context:

```
=== CATCH-UP SUMMARY ===
Since your last context:
- Task "implement-fts5-search" was completed by agent-2 (PR #12)
- Task "implement-hybrid-search" is now ready (was blocked on FTS5)
- Agent-3 reported finding: "FTS5 tokenizer needs 'porter' for stemming"
- 2 tasks completed today, 3 remaining in Phase 1
- Total spend today: $1.23
=== END CATCH-UP ===
```

This is generated deterministically from the mutation log and audit log.

---

## 7. System Architecture Overview

### Module Map

```
┌──────────────────────────────────────────────────────────────┐
│                        OPERATOR (Human)                       │
│  corc plan (interactive)  │  corc watch/status  │  corc pause │
└──────┬────────────────────┴──────────┬──────────┴────────────┘
       │ writes tasks                  │ monitors
       ▼                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    DAEMON (Thin Event Loop)                    │
│  Polls for ready tasks. Delegates to modules below.           │
│  Idles when no work. Picks up new tasks automatically.        │
└──────┬──────────────┬──────────────┬─────────────────────────┘
       │              │              │
       ▼              ▼              ▼
┌────────────┐ ┌────────────┐ ┌────────────┐
│ Scheduler  │ │ Executor   │ │ Processor  │
│ DAG reso-  │ │ Dispatches │ │ Validates  │
│ lution,    │ │ agents via │ │ output,    │
│ ready task │ │ dispatch   │ │ curates    │
│ selection  │ │ abstraction│ │ findings,  │
│            │ │ layer      │ │ updates    │
│            │ │            │ │ state      │
└─────┬──────┘ └─────┬──────┘ └─────┬──────┘
      │               │              │
      ▼               ▼              ▼
┌──────────────────────────────────────────────────────────────┐
│                    WORK STATE (Layer 2)                        │
│       Mutation log + SQLite — the shared bus                  │
└──────────────────────────────────────────────────────────────┘
      │               │              │
      ▼               ▼              ▼
┌─────────┐    ┌──────────┐    ┌──────────┐
│Knowledge│    │ Agent    │    │ Audit    │
│ Store   │    │ Roles    │    │ Log      │
│(Layer 1)│    │          │    │(Layer 3) │
└─────────┘    └──────────┘    └──────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│              AGENTS (via dispatch abstraction layer)           │
│  Claude Code, Gemini CLI, Codex, or any future LLM CLI        │
│  Each invocation: constrained tools, budget, schema           │
│  Hooks enforce pre/post conditions                            │
└──────────────────────────────────────────────────────────────┘
```

### Key Design Decision: The Daemon Is a Thin Loop, Not a Monolith

The daemon is a minimal event loop that delegates to three independent modules. Each module reads from and writes to the work state — they do not call each other.

**Scheduler** — resolves the DAG, determines which tasks are ready, manages parallelism limits. Reads work state, writes nothing (pure query). Can be called standalone: `corc tasks --ready`.

**Executor** — dispatches agents via the dispatch abstraction layer, captures output, handles retries. Reads work state + agent roles, writes agent records. Can be called standalone: `corc dispatch TASK_ID`.

**Processor** — validates output against "done when," curates findings into the knowledge store, merges PRs, manages repo merge policies. Reads work state + agent output, writes task completions. Can be called standalone: `corc process TASK_ID`.

The daemon loop is:
```
while running:
    ready = scheduler.get_ready_tasks()
    for task in ready (up to --parallel limit):
        executor.dispatch(task)
    for completed in executor.poll_completed():
        processor.process(completed)
    sleep(5s)
```

Each module is a separate Python module with its own CLI entry point. The daemon connects them but they don't depend on each other. Any module can be replaced independently.

### Dispatch Abstraction Layer

Agent dispatch is abstracted behind an interface so the underlying LLM CLI can be swapped:

```python
class AgentDispatcher:
    def dispatch(self, task: Task, context: str, constraints: Constraints) -> AgentHandle
    def poll(self, handle: AgentHandle) -> Optional[AgentResult]
    def kill(self, handle: AgentHandle) -> None

class ClaudeCodeDispatcher(AgentDispatcher):
    # Translates constraints to: claude -p --allowedTools --max-budget-usd --json-schema etc.

class GeminiDispatcher(AgentDispatcher):
    # Translates constraints to: gemini CLI flags

class CodexDispatcher(AgentDispatcher):
    # Translates constraints to: codex CLI flags
```

`Constraints` captures our intent (allowed tools, budget, output format, system prompt) without referencing any specific CLI's flags. The dispatcher translates intent to flags. This isolates all CLI coupling to one file per provider.

The active dispatcher is configured in `.corc/config.yaml`:
```yaml
dispatch:
  provider: claude-code   # claude-code | gemini | codex
```

### Multimodal Support

The dispatch abstraction supports non-text inputs and outputs:

```python
@dataclass
class TaskInput:
    text: str                           # Always present
    files: list[str]                    # File paths (code, docs)
    images: list[str] = field(default_factory=list)  # Screenshots, diagrams
    urls: list[str] = field(default_factory=list)     # Web pages to reference
```

Context bundles can reference images and URLs alongside text files. The dispatcher handles encoding for the target LLM.

### Orchestrator Self-Testing

The orchestrator is still a single point of brittleness — if the scheduler, executor, or processor has a bug, agents get wrong inputs. Mitigations:
- Every module has its own test suite (unit + integration + property-based)
- The orchestrator fails loudly on any ambiguity — unresolvable dependencies, missing bundle files, unevaluable "done when" → halt and notify operator
- `corc self-test` runs all module test suites
- Chaos monkey tests the orchestrator itself — corrupting mutation log, simulating SQLite failures

### Data Flow

**Planning (interactive):**
1. Operator runs `corc plan` — opens an interactive Claude session
2. Claude has access to the knowledge store, work state, and repo context
3. Operator and Claude collaborate on the spec, task decomposition, "done when" criteria, and context bundles
4. Tasks are written directly to the work state (mutation log + SQLite)

**Execution (daemon):**
5. Operator runs `corc start` — the daemon begins its loop
6. Daemon checks for ready tasks (no unblocked dependencies)
7. For each ready task: assembles context bundle, dispatches `claude -p` in a git worktree
8. Hooks enforce all pre/post conditions deterministically
9. On completion: output validated against "done when", findings reported, work state updated, audit logged
10. On failure: retry policy executed, then escalation to operator
11. Orchestrator curates agent findings into knowledge store
12. Daemon checks for newly unblocked tasks and dispatches them
13. When no tasks are ready, daemon idles and polls for new work
14. New tasks added via `corc plan` or `corc task add` are picked up automatically — no restart needed

### Communication Model

| Communication Path | Mechanism | Phase |
|---|---|---|
| Operator → Work State | `corc plan` (interactive Claude session writes tasks) | Phase 1 |
| Operator ↔ System | `corc status`, `corc watch`, `corc pause/resume` | Phase 1 |
| Operator ← System (notifications) | Terminal prompt → later: Slack/Discord/Telegram/email | Phase 1 (terminal), Phase 4 (external) |
| Daemon → Agent | `claude -p` with curated context, constrained tools | Phase 1 |
| Agent → Daemon | Structured output (findings, PR URL, proof-of-work) | Phase 1 |
| Daemon ← Work State | Polls for ready tasks every 5s | Phase 1 |
| Agent ↔ Agent | Not supported initially. Agents are independent workers. | Phase 3 (if needed) |
| External Events → System | Future: Claude Code Channels (webhook/Telegram/Discord) | Phase 4+ |

---

## 8. Module Specifications

### Spec Template

Each module follows this structure:
```
## Problem & Motivation (why this exists)
## Requirements (testable statements)
## Non-Requirements (what it explicitly won't do)
## Design (schemas, interfaces, data flow)
## CLI Interface (exact commands and flags)
## Integration with Hooks (what's enforced deterministically)
## Error Handling (failure modes and recovery)
## Testing Strategy
## Open Questions
```

---

### Module 1: Knowledge Store

#### Problem & Motivation
Agents lose all context between sessions. Relevant decisions, conventions, and outcomes from previous work are invisible to new sessions. Without structured retrieval, agents either get no context or get flooded with irrelevant context.

#### Requirements
- Store markdown documents with YAML frontmatter as the source of truth
- Maintain a derived SQLite index that is always rebuildable from files
- Support keyword search (FTS5), semantic search (sentence-transformers), and hybrid search
- Index must be correct at query time — stale indexes must be detected and refreshed
- All write operations must be transactional
- Must work without semantic search if PyTorch/sentence-transformers are unavailable
- Write access restricted: only the orchestrator or operator writes; agents report findings

#### Non-Requirements
- Will not handle binary files (images, PDFs)
- Will not implement real-time collaboration or CRDT sync
- Will not manage knowledge decay or auto-archival (future consideration)

#### Future: Contradiction Detection

The knowledge store can accumulate contradictory documents over time (e.g., two decisions that conflict). High-level plan for addressing this:

1. **On curation**: when the orchestrator curates a new finding into the knowledge store, run a lightweight check — search for existing documents with overlapping topics and flag potential conflicts for operator review. The `supersedes` field in frontmatter handles explicit supersession (new decision replaces old one).

2. **Periodic consistency scan** (`corc knowledge check`): a `claude -p` invocation that reads all active documents of the same type/project and identifies contradictions, outdated information, and redundancies. Produces a report for operator review.

3. **Automatic supersession**: when a new decision document explicitly references an older one, the older one is automatically marked `status: superseded`.

#### Document Schema (Markdown with YAML Frontmatter)

```yaml
---
id: uuid-v4
type: decision | task-outcome | architecture | repo-context | research | meeting | note
project: string (optional)
repos: [list of repo names] (optional)
tags: [list of strings]
created: ISO-8601 datetime
updated: ISO-8601 datetime
source: human | orchestrator      # Note: never "agent" — agents report findings, orchestrator curates
status: active | superseded | archived
supersedes: uuid (optional, for decision chains)
---

# Document Title

Document content in markdown. Can use [[wikilinks]] to reference other documents.
Supports standard markdown formatting.
```

#### File Organization

```
knowledge/
├── decisions/          # Architectural and design decisions (ADRs)
├── tasks/              # Task outcomes and learnings (curated from agent findings)
├── architecture/       # System design documents
├── repos/              # Per-repo context and state
│   ├── work-app/
│   └── orchestrator/
├── research/           # Research findings and analysis
└── _templates/         # Document templates
```

#### SQLite Index Schema

```sql
-- Core document metadata (mirrors frontmatter)
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    file_path TEXT UNIQUE NOT NULL,
    type TEXT NOT NULL,
    project TEXT,
    title TEXT NOT NULL,
    status TEXT DEFAULT 'active',
    source TEXT DEFAULT 'human',
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    supersedes TEXT
);

CREATE TABLE document_tags (
    document_id TEXT REFERENCES documents(id),
    tag TEXT NOT NULL,
    PRIMARY KEY (document_id, tag)
);

CREATE TABLE document_repos (
    document_id TEXT REFERENCES documents(id),
    repo TEXT NOT NULL,
    PRIMARY KEY (document_id, repo)
);

CREATE VIRTUAL TABLE documents_fts USING fts5(
    title, content,
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TABLE chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT REFERENCES documents(id),
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    embedding BLOB
);

CREATE INDEX idx_documents_type ON documents(type);
CREATE INDEX idx_documents_project ON documents(project);
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_created ON documents(created);
CREATE INDEX idx_document_tags_tag ON document_tags(tag);
CREATE INDEX idx_document_repos_repo ON document_repos(repo);
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
```

#### Search Implementation

**Keyword search**: FTS5 with BM25 ranking.

**Semantic search**: sentence-transformers (`all-MiniLM-L6-v2`, 384-dim) → cosine similarity. Fallback to keyword-only if unavailable.

**Hybrid search** (default): 0.4 keyword + 0.6 semantic, normalized to [0, 1], deduplicated by document.

#### Index Freshness Strategy

1. **Pre-query integrity check**: Compare file mtimes against indexed records. Re-index stale docs before returning results.
2. **Write-through on mutation**: `corc add`/`corc update` immediately re-index. `PostToolUse[Write|Edit]` hook triggers incremental re-index for files in `knowledge/`.
3. **Manual full rebuild**: `corc reindex --full`.

#### CLI Interface

```
corc add [--type TYPE] [--project PROJECT] [--tags TAG1,TAG2] [--file PATH | STDIN]
corc search QUERY [--semantic] [--hybrid] [--type TYPE] [--project PROJECT] [--limit N]
corc list [--type TYPE] [--project PROJECT] [--tags TAG] [--since DATE] [--status STATUS]
corc get ID_OR_PATH
corc update ID_OR_PATH [--status STATUS] [--tags TAG1,TAG2]
corc reindex [--full]
corc stats [--project PROJECT]
corc template TYPE
```

#### Testing Strategy

- Unit tests: frontmatter parsing, content chunking, FTS5 queries, semantic search, hybrid ranking
- Integration tests: full index-search cycle with sample documents
- Property tests: search results always include exact-match documents
- Benchmark: search latency at 100, 500, 1000 documents

---

### Module 2: Workflow Engine

#### Problem & Motivation
Without a deterministic workflow engine, agents decide what to do next — leading to inconsistent behavior, skipped steps, and no auditability. The engine ensures every operation follows a defined DAG with validation gates.

#### Requirements
- Parse workflow/task definitions from YAML files
- Execute tasks as a DAG — support parallel branches where dependencies allow
- Every task has a "done when" criterion — explicit, testable completion conditions
- Persist state (via mutation log) after every step for resume-on-failure
- Validate outputs after every step using "done when" criteria
- Support retry policies (same, enriched, different approach, escalate, skip)
- Every `claude -p` invocation specifies: allowed tools, max budget, output schema, system prompt
- Support a global pause switch that halts all new step execution
- Support chaos monkey mode for resilience testing
- Log every step to the audit log

#### Non-Requirements
- Will not implement a visual DAG editor

#### Task Definition

Tasks can be defined inline in a workflow YAML or decomposed from a spec:

```yaml
name: implement-knowledge-store
description: Build the knowledge store module
version: 1

tasks:
  - name: implement-markdown-parser
    role: implementer
    done_when: "Frontmatter parsing works for all document types; chunking produces ~500 token chunks; unit tests pass"
    context_bundle:
      - SPEC.md#module-1-knowledge-store
      - knowledge/decisions/sqlite-choice.md
    claude:
      allowed_tools: [Read, Edit, Write, Bash, Grep, Glob]
      max_budget_usd: 2.00
      max_turns: 50
    on_failure:
      retry: 2
      retry_with_context: "Previous attempt failed. Error: ${error}. Ensure all document types parse correctly."
      escalate: human

  - name: implement-sqlite-schema
    role: implementer
    depends_on: []
    done_when: "All tables created; migration runs cleanly; schema matches spec"
    context_bundle:
      - SPEC.md#module-1-knowledge-store
    claude:
      allowed_tools: [Read, Edit, Write, Bash, Grep, Glob]
      max_budget_usd: 1.00
      max_turns: 30

  - name: implement-fts5-search
    role: implementer
    depends_on: [implement-markdown-parser, implement-sqlite-schema]
    done_when: "FTS5 queries return BM25-ranked results; integration test with 10 sample docs passes"
    context_bundle:
      - SPEC.md#module-1-knowledge-store
      - knowledge/research/search-architectures.md
    claude:
      allowed_tools: [Read, Edit, Write, Bash, Grep, Glob]
      max_budget_usd: 2.00
      max_turns: 50

  - name: review-knowledge-store
    role: adversarial-reviewer
    depends_on: [implement-fts5-search]
    done_when: "Review posted as PR comment via gh pr review; no critical issues found, or critical issues are addressed"
    context_bundle:
      - SPEC.md#module-1-knowledge-store
    claude:
      allowed_tools: [Read, Grep, Glob, Bash(gh pr*|git diff*|git log*|python -m pytest*)]
      max_budget_usd: 1.00
      max_turns: 20
```

#### Pipeline Pattern: Scout → Implement → Review

For complex tasks, the workflow can formalize the research-before-implementation pattern:

```yaml
tasks:
  - name: scout-embedding-approach
    role: scout
    done_when: "Research brief produced comparing sentence-transformers vs fastembed vs skip-embeddings approaches"
    context_bundle:
      - knowledge/research/search-architectures.md
    claude:
      allowed_tools: [Read, Grep, Glob, WebSearch, WebFetch]
      max_budget_usd: 1.00
      max_turns: 20

  - name: implement-semantic-search
    role: implementer
    depends_on: [scout-embedding-approach]
    done_when: "Semantic search returns cosine-similarity ranked results; fallback to keyword-only works"
    # Scout's output automatically included in context

  - name: review-semantic-search
    role: adversarial-reviewer
    depends_on: [implement-semantic-search]
    done_when: "Adversarial review complete; critic given only the diff, spec, and done_when — no implementation history"
    # Context deliberately stripped — only diff + spec + done_when
```

**Adversarial reviewer context reset**: The reviewer role gets minimal context — only the diff, coding standards, and "done when" criteria. No familiarity with how the code evolved. This prevents agreeableness from accumulated context.

#### DAG Execution

- Engine resolves the dependency graph at startup
- Tasks with no unresolved dependencies are "ready" and can execute in parallel
- Each task runs as a `claude -p` invocation in its own git worktree
- State is written to the mutation log after each task completes
- On completion: PR created, proof-of-work artifact produced, findings reported, next ready tasks dispatched

#### "Done When" Quality Linting

"Done when" criteria must be machine-testable, not subjective. The orchestrator validates criteria at plan time:

- **Accepted**: "All unit tests pass", "FTS5 queries return BM25-ranked results", "Output matches JSON schema X"
- **Rejected**: "Implementation works correctly", "Code is clean", "Good performance"

The linter checks for: measurable assertions, references to specific tests/schemas/files, absence of subjective adjectives. This is enforced at plan creation — bad "done when" criteria are flagged before any agent starts work.

#### Proof-of-Work Artifacts

Every task produces a structured artifact alongside the PR:

```json
{
  "task_id": "abc123",
  "what_i_did": "Implemented FTS5 search with BM25 ranking across document titles and content",
  "files_changed": ["src/corc/search.py", "tests/test_search.py"],
  "how_to_verify": [
    "Run: python -m pytest tests/test_search.py -v",
    "Run: corc search 'test query' --limit 5 (should return ranked results)",
    "Check: search latency < 100ms for 100 documents"
  ],
  "checklist_completed": ["FTS5 table created", "BM25 ranking works", "Integration test passes"],
  "checklist_remaining": [],
  "findings": ["FTS5 tokenizer needs 'porter' flag for English stemming"],
  "micro_deviations": ["Fixed a typo in existing CLI help text (1 line, unrelated to task)"]
}
```

The adversarial reviewer uses this artifact (plus the diff) as input — it tells them exactly what to verify.

#### Authorized Micro-Deviations

Agents are scope-disciplined, but rigidly refusing all out-of-scope work wastes resources. Agents may make **micro-deviations** — small, well-defined fixes outside task scope — if they meet ALL of these criteria:

1. The fix is ≤5 lines of code
2. The fix is obviously correct (typo, missing import, broken link)
3. The fix is documented in the proof-of-work artifact
4. The fix does not change behavior or interfaces

If the fix doesn't meet these criteria, the agent reports it as a finding and stops. The orchestrator decides whether to create a new task for it.

#### Structured Sub-Progress

Complex tasks include a checklist that agents mark off as they work:

```yaml
tasks:
  - name: implement-hybrid-search
    done_when: "Hybrid search returns weighted results; all checklist items complete"
    checklist:
      - "FTS5 keyword search works independently"
      - "Semantic search works independently"
      - "Score normalization to [0,1]"
      - "Configurable weights (default 0.4/0.6)"
      - "Deduplication by document"
      - "Integration test with 10+ documents"
```

On failure or handoff, the next agent sees exactly which items are done and which remain. This prevents duplicate work and makes the handoff summary structured rather than free-form.

#### Pause Switch

```
corc pause "reason for pausing"   # Write pause lock
corc resume                        # Remove pause lock
corc status                        # Shows pause state
```

- `corc pause` writes to `.corc/pause.lock` with reason, timestamp, and source
- Before dispatching any new task, the orchestrator checks for this file
- In-flight tasks complete (cannot safely interrupt mid-execution)
- Operator is notified via configured notification channel
- Any agent can trigger via `corc pause` when it detects something that would break the plan

#### Chaos Monkey Mode

```
corc chaos enable [--kill-rate 0.1] [--corrupt-rate 0.05]
corc chaos disable
corc chaos status
```

When enabled:
- Randomly kills agent processes mid-task
- Randomly corrupts intermediate state files
- Simulates context window overflow
- Drops inter-agent messages (when applicable)

Verifies:
- System resumes from last completed step
- Knowledge store remains consistent
- Notifications fire correctly on failure
- Rating system detects degradation

#### Planning (Interactive Sessions)

`corc plan` opens an interactive Claude Code session for collaborative spec development and task decomposition. The output is both a spec document and executable work — they're one artifact, not two.

```bash
corc plan                    # Start a planning session
corc plan my-idea.md         # Start with a seed document pre-loaded
```

**Context injected into the planning session:**
- System prompt: planner role instructions + CORC spec template
- Knowledge store: all decisions, research, conventions, prior task outcomes (`corc search` available as a tool)
- Work state: current tasks, what's in progress, what's done, what's blocked
- Repo context: code structure, recent commits, conventions from registered repos
- Rating data: what's been working well, what roles/task-types have been problematic

This means the planner has full visibility into everything the system knows. It can search for prior art, understand what's already been built, and avoid duplicating work.

**Three-stage planning process:**

**Stage 1 — Spec development:** You and Claude discuss the problem, research options, and write a spec. Claude can search the knowledge store and the web. The spec follows the module spec template: problem, requirements, non-requirements, design, testing strategy.

**Stage 2 — Task decomposition:** Claude breaks the spec into a DAG of concrete tasks. Each task gets:
- A name and description
- A role (scout, implementer, reviewer, adversarial-reviewer)
- `depends_on` — which tasks must complete first
- `done_when` — machine-testable completion criteria (linted for subjectivity)
- `checklist` — sub-steps for structured progress tracking
- `context_bundle` — specific files to inject as context
- `on_failure` — retry policy

**Stage 3 — Review and commit:** You review the full plan (spec + task DAG together). Adjust anything — reorder tasks, tighten "done when" criteria, change roles, modify context bundles. When you approve:
1. The spec is saved to the knowledge store (`knowledge/` as a markdown file)
2. All tasks are written to the work state (mutation log + SQLite)
3. The daemon picks up ready tasks automatically

The spec and the task DAG are stored together — the spec frontmatter references the task IDs, and each task references the spec. They stay linked.

**Planning reasoning is preserved:** The spec document includes a "Rationale" section capturing key planning decisions: why the work was decomposed this way, why certain dependencies exist, why specific context bundles were chosen, what alternatives were considered. This is part of the spec, not a separate artifact. It uses tokens liberally — the knowledge store can handle it, and future planning sessions benefit from understanding past reasoning.

**Quick task path:** Not every task needs a full spec and multi-task decomposition. During `corc plan`, Claude determines the appropriate level of formality:

- **Quick task**: A single code change with a clear "done when." No decomposition, no scout phase. Written directly as one task. Example: "Fix the typo in README.md" or "Add a --verbose flag to corc search."
- **Standard task**: A feature that needs a brief spec and 2-5 tasks. Scout phase optional. Example: "Add semantic search to the knowledge store."
- **Epic**: A large feature that needs a full spec, detailed decomposition, 5+ tasks, scout phases, adversarial review. Example: "Build the workflow engine."

Claude proposes the level during the planning session. The operator confirms or overrides.

#### Daemon (Automatic Execution)

`corc start` runs the orchestrator as a long-running daemon:

```bash
corc start                   # Start daemon. Processes all ready work. Idles when empty.
corc start --parallel 3      # Up to 3 concurrent agents (default: 1)
corc start --task TASK_ID    # Run one specific task, then stop
corc start --once            # Process one ready task, then stop
corc stop                    # Graceful shutdown (in-flight tasks finish)
```

**Daemon loop:**
1. Query work state for tasks with status "ready" (all dependencies met)
2. If none ready: idle, poll every 5 seconds
3. If ready: dispatch up to `--parallel` agents concurrently
4. On task completion: validate, merge, update state, check for newly unblocked tasks
5. On task failure: execute retry policy
6. Repeat

The daemon picks up new tasks added by `corc plan` or `corc task add` without restarting. It's designed to run in a terminal tab and be left alone.

#### CLI — Essential Commands

These are the commands used day-to-day:

```
corc plan [FILE]                               # Interactive planning session
corc start [--parallel N]                      # Start daemon
corc stop                                      # Stop daemon
corc status                                    # Snapshot: DAG progress, agents, cost
corc dag                                       # Visual DAG with status
corc watch                                     # Live TUI dashboard
corc pause "reason"                            # Halt dispatching
corc resume                                    # Continue
corc escalations                               # Pending escalations
corc escalation resolve ESC_ID                 # Resolve escalation
```

#### CLI — Analysis & Management

```
corc tasks [--status STATUS] [--ready]         # List tasks
corc task-status TASK_ID                       # Detailed task info
corc log --last N                              # Human-readable recent events
corc log --task TASK_ID                        # All events for one task
corc analyze costs [--today] [--project P]     # Cost analysis
corc analyze patterns                          # Quality patterns + recommendations
corc analyze retries                           # Retry statistics
corc analyze planning                          # Planning pattern analysis
corc rate --auto                               # Score unscored runs
corc ratings --trend                           # Quality over time
corc curate RUN_ID                             # Review + persist agent findings
corc retro PROJECT_NAME                        # Project-level retrospective
```

#### CLI — Advanced / Debugging

```
corc start --task TASK_ID                      # Run one specific task
corc start --once                              # Process one ready task, stop
corc dispatch TASK_ID                          # Manually dispatch (bypass daemon)
corc process TASK_ID                           # Manually process output
corc dag --mermaid                             # DAG as Mermaid markdown
corc chaos enable|disable|status               # Chaos monkey
corc self-test                                 # Orchestrator health check
corc plan --resume                             # Resume crashed planning session
corc context-for-task TASK_ID                  # Show assembled context (debugging)
```

#### Error Handling

Tiered recovery:
1. **Retry same** (up to N times)
2. **Retry enriched** — add error message and relevant context
3. **Retry different** — decompose into smaller subtasks
4. **Escalate to human** — pause, present state, notify operator
5. **Log and skip** — for non-critical tasks

**Escalation mechanism** (modular):
- Phase 1: Terminal prompt (blocking)
- Later: Slack/Discord/Telegram/email via pluggable notification channels

#### Testing Strategy

- Unit tests: YAML parsing, DAG resolution, variable substitution, "done when" validation
- Integration tests: full workflow with mock `claude -p`
- Chaos monkey tests: verify resume-after-kill for every step type
- Property tests: DAG execution order always respects dependencies

---

### Module 3: Repo Manager

#### Problem & Motivation
Multi-repo projects need a central registry of repo metadata and the ability to generate context summaries for the knowledge store.

#### Requirements
- Maintain a YAML registry of repos with metadata
- Generate context summaries for the knowledge store
- Report status (recent commits, open PRs, current branch, uncommitted changes)

#### Design

**Repo Registry** (`.corc/repos.yaml`):
```yaml
repos:
  work-app:
    path: /Users/me/work-app
    remote: git@github.com:org/work-app.git
    primary_branch: main
    language: python
    project: work-platform
  corc:
    path: /Users/me/corc
    remote: git@github.com:me/corc.git
    primary_branch: main
    language: python
    project: corc
```

#### CLI Interface

```
corc repo add NAME --path PATH [--remote URL] [--branch BRANCH] [--language LANG] [--project PROJECT]
corc repo remove NAME
corc repo list
corc repo status NAME
corc repo context NAME        # Generate/update knowledge store context doc
corc repo context --all
corc repo diff NAME
```

---

### Module 4: Agent Roles

#### Problem & Motivation
Different tasks require different system prompts, tool access, knowledge context, and behavioral constraints. Roles codify these differences so agents are specialized for their work.

#### Requirements
- Define roles as YAML config files
- Each role specifies: system prompt, knowledge queries, allowed tools, cost limits, knowledge write access level
- Roles are composable — a role can extend another
- Roles are version-controlled alongside the codebase

#### Design

**Role Definition** (`.corc/roles/implementer.yaml`):
```yaml
name: implementer
description: Code generation and implementation
extends: null

system_prompt: |
  You are an implementer scoped to a specific task. Focus on:
  - Writing clean, correct code that meets the "done when" criteria
  - Writing tests alongside implementation
  - Committing changes with clear messages referencing the task

  SCOPE DISCIPLINE: You are scoped to this task only. If you discover work
  outside this boundary, report it in your findings and stop. Do not expand scope.

knowledge_write_access: findings_only  # Cannot write to knowledge store; reports findings

allowed_tools:
  - Read
  - Edit
  - Write
  - Bash
  - Grep
  - Glob

cost_limits:
  max_budget_per_invocation_usd: 3.00
  max_turns_per_invocation: 50
```

**Built-in roles:**

| Role | Purpose | Tools | Knowledge Write |
|---|---|---|---|
| `scout` | Research before implementation. Read-only. Produces structured research brief. | Read, Grep, Glob, WebSearch, WebFetch | findings_only |
| `implementer` | Code generation. Works in a worktree, creates PRs. | Read, Edit, Write, Bash, Grep, Glob | findings_only |
| `reviewer` | Code review with conventions awareness. | Read, Grep, Glob, Bash(git*, gh pr*) | findings_only |
| `adversarial-reviewer` | Context-stripped adversarial review. Only sees diff + spec + done_when. | Read, Grep, Glob, Bash(git diff*, gh pr*, python -m pytest*) | findings_only |
| `planner` | Spec decomposition and task DAG creation. Structured output. | Read, Grep, Glob | findings_only |

#### CLI Interface

```
corc role list
corc role show NAME
corc role validate NAME
```

---

## 9. Operator Visibility & Control

### Design Principles
- The operator should understand the full system state at a glance, without reading text logs
- Real-time monitoring for active work, post-hoc analysis for completed work
- Controls should be immediate — pause takes effect before the next task dispatches
- The system should be visually clear enough that a human can quickly identify problems

### DAG Visualization

**`corc dag`** — render the task dependency graph:

```
corc dag                     # Show full DAG with status colors
corc dag --plan PLAN_ID      # Show DAG for a specific plan
```

Output (terminal with ANSI colors):
```
implement-parser ✅ ──────┐
                          ├──► implement-fts5 ✅ ──► implement-hybrid 🔄 ──► review-search ⬚
implement-schema ✅ ──────┘                                                       │
                                                                                   ▼
scout-embedding ✅ ──► implement-semantic ✅ ────────────────────────────────► implement-cli ⬚

✅ = complete  🔄 = running  ⬚ = ready  ◻ = blocked  ❌ = failed

Progress: 5/8 tasks (62%)  |  Running: 1  |  Ready: 2  |  Blocked: 1
Cost: $4.23  |  Est. remaining: $3.50
```

Also available as Mermaid markdown output (`corc dag --mermaid`) for rendering in Obsidian or GitHub.

### Real-Time Monitoring

**`corc watch`** — live TUI dashboard (built with `textual` or `rich`):

The watch view shows three panels:
1. **DAG view** (top): miniature DAG with live status updates as tasks complete
2. **Event stream** (middle): color-coded events as they happen
3. **Agent detail** (bottom): currently running agents with live checklist progress

```
┌─ DAG ──────────────────────────────────────────────────────┐
│ parser ✅ → fts5 ✅ → hybrid 🔄 [3/6 checklist] → review ◻│
│ schema ✅ ┘            semantic ✅ → cli ⬚                  │
│                                                             │
│ Progress: 5/8 (62%)  Cost: $4.23  Parallel: 1/2            │
├─ Events ───────────────────────────────────────────────────┤
│ 10:10:30  implement-hybrid  ☑ "Score normalization"        │
│ 10:10:15  implement-hybrid  ☑ "Semantic search integrated" │
│ 10:08:12  implement-fts5    ✅ COMPLETED ($0.89)           │
├─ Agents ───────────────────────────────────────────────────┤
│ agent-1  implementer  implement-hybrid  3/6 checklist      │
│          Running 2m15s  ~$0.35 so far                      │
└────────────────────────────────────────────────────────────┘
```

Checklist progress streams live — as the agent marks off items, the operator sees them tick in the dashboard.

**`corc status`** — text snapshot (for scripting or quick checks):
```
Plan: corc-phase-1 (5/8 tasks, 62%)
  ✅ implement-parser, implement-schema, implement-fts5, scout-embedding, implement-semantic
  🔄 implement-hybrid (agent-1, 2m15s, 3/6 checklist)
  ⬚ implement-cli (ready), review-search (ready)
  ◻ (none blocked)

Agents: 1/2 active
Today: 5 tasks completed, 0 failed, $4.23 spent
Pause: inactive
```

### Escalation UX

When the daemon needs the operator, it does three things:

**1. Pauses dispatching** (no new tasks start).

**2. Creates a structured escalation record:**
```json
{
  "escalation_id": "esc-001",
  "task_id": "abc123",
  "task_name": "implement-hybrid-search",
  "type": "retry_exhausted",
  "attempts": 3,
  "last_error": "Assertion failed: hybrid scores not normalized to [0,1]",
  "session_log": "data/sessions/abc123-attempt-3.jsonl",
  "pr_url": "https://github.com/.../pull/7",
  "suggested_actions": [
    "Review the session log for the failing assertion",
    "Check if the normalization formula in the spec is correct",
    "Try adjusting the done_when criteria"
  ]
}
```

**3. Sends notification** via configured channel with a summary:
```
🔴 ESCALATION: implement-hybrid-search failed after 3 attempts
   Error: hybrid scores not normalized to [0,1]
   Run: corc escalation show esc-001
```

The operator resolves escalations:
```
corc escalations                    # List pending escalations
corc escalation show ESC_ID         # Full detail including suggested actions
corc escalation resolve ESC_ID      # Mark resolved, resume daemon
```

### Post-Hoc Analysis

```
corc analyze costs --today
corc analyze costs --project corc --since 2026-03-21
corc analyze failures --since 2026-03-01
corc analyze duration --last 20
corc analyze agent-efficiency --role implementer
corc analyze task TASK_ID          # Full history: dispatches, retries, session logs, findings, PR
corc log --last 20                 # Human-readable formatted log of recent events
corc log --task TASK_ID            # All events for one task, formatted
```

### Notification System

Modular notification backend. Configuration in `.corc/config.yaml`:

```yaml
notifications:
  channels:
    terminal:
      enabled: true
    slack:
      enabled: false
      webhook_url: null
    discord:
      enabled: false
      webhook_url: null
    telegram:
      enabled: false
      bot_token: null
      chat_id: null

  triggers:
    escalation: [terminal]
    task_complete: []               # Empty = no notification (check via corc status)
    task_failure: [terminal]
    cost_threshold: [terminal]
    pause: [terminal]
    daily_summary: [terminal]
```

Interface for new channels:
```python
class NotificationChannel:
    def send(self, event_type: str, title: str, body: str, severity: str) -> bool
```

---

## 10. Rating & Continuous Improvement

### Philosophy
The rating system should be **critical, not generous**. A "10" means flawless execution — rare. The goal is continuous improvement. We don't grade on a curve. Calibration happens during the CORC build itself.

### Scoring Dimensions

| Dimension | Source | Weight | Description |
|---|---|---|---|
| **Correctness** | Automated (tests) + Claude evaluation | 0.25 | Did the output meet the "done when" criteria? |
| **Completeness** | Automated (checklist) | 0.15 | Were all requirements addressed? |
| **Code Quality** | Automated (lint, complexity, test coverage) | 0.15 | Measurable code health metrics |
| **Efficiency** | Automated (tokens, cost, wall clock time) | 0.15 | Resource usage relative to task complexity |
| **Determinism** | Automated (event log analysis) | 0.10 | Did the agent follow the prescribed workflow? Any deviations? |
| **Resilience** | Automated (chaos monkey results) | 0.10 | Did recovery mechanisms work when failures occurred? |
| **Human Intervention** | Automated (escalation count) | 0.10 | How many times did the operator need to step in? |

### Scoring Mechanics

- Each dimension scored 1-10 (integer)
- Overall score = weighted sum
- Scores below 7 on any dimension trigger an investigation flag
- After each workflow run, a separate `claude -p` invocation evaluates the run using the spec as a rubric
- Human override capability for any score

### Tracking

```
corc rate RUN_ID
corc rate --auto
corc ratings --project corc --since 2026-03-21
corc ratings --trend --last 30
corc ratings --dimension correctness
```

Ratings stored as JSONL in `data/ratings/`.

### Rating Feedback Loop

Ratings aren't just for observation — they feed back into system behavior:

**Automated pattern detection** (`corc analyze patterns`):
- Identifies correlations: which roles, task types, or context bundles correlate with low scores
- Example output: "implementer role scores 3.2 avg on correctness for tasks involving SQL. reviewer role scores 8.1. Consider: adjusting implementer system prompt for SQL tasks, or adding a scout phase."
- Runs weekly (or on demand) against accumulated rating data

**Adaptive retry policy:**
- The system tracks first-attempt success rate by task type and role
- For task types where first-attempt success rate >90%, retry count is automatically reduced (default: 2 → 1)
- For task types where first-attempt success rate <50%, retry count is increased and the orchestrator flags the task type for investigation
- Retry adaptation is logged to the audit log so the operator can review changes
- `corc analyze retries` shows retry statistics and current adaptive settings

**Prompt version tracking:**
- Role system prompts are versioned files (`.corc/roles/implementer.yaml` tracked in git)
- Each run records which prompt version was used
- `corc analyze prompts --role implementer` shows quality scores by prompt version
- This lets us A/B test prompt changes with data

**Trust level recommendations:**
- When a role consistently scores >9 on a dimension across 20+ tasks, the system suggests raising trust level for that role/task-type combination
- When scores drop below 5, the system suggests lowering trust level
- Recommendations surfaced in `corc analyze patterns` output and daily summaries

**Planning pattern learning:**
- Track which spec structures lead to better task outcomes
- Track which decomposition granularities work best (too many small tasks vs. too few large ones)
- Track which context bundle sizes and compositions correlate with success
- Surface these patterns in `corc analyze planning` to improve future planning sessions

**Auto-adjustments (with operator confirmation):**
- If the scout phase consistently improves outcomes for a task type, auto-add scout as a dependency for that type (operator confirms)
- If a role's prompt version correlates with worse scores, flag it and suggest rollback
- If certain context bundles are consistently insufficient (agents frequently fall back to RAG search), flag the bundle for expansion

**Project retrospective:**
```
corc retro PROJECT_NAME           # Generate a project-level retrospective
```
Produces a structured analysis: what went well, what didn't, which phases were most/least efficient, total cost vs. estimate, quality trends, top findings, and concrete recommendations for the next project. Stored in the knowledge store as a `task-outcome` document.

---

## 11. Multi-Agent Coordination

### Approach

The orchestrator dispatches tasks as independent `claude -p` invocations. Each agent is a stateless worker — it receives curated context, does its work, and returns structured output. The orchestrator handles all coordination.

For tasks requiring longer interaction (complex implementation, multi-file refactors), agents run as interactive `claude` sessions in tmux panes, still with curated context injected via hooks.

### Concurrency Control

1. **DAG-aware scheduling**: Non-overlapping file sets assigned to parallel tasks.
2. **Git worktrees**: Each agent works in its own worktree.
3. **Optimistic merge**: Merge on completion; retry on conflict.
4. **SQLite WAL + advisory locks**: For knowledge store and work state concurrent access.

### Agent Lifecycle

```
1. Orchestrator identifies ready task (no unblocked dependencies)
2. Orchestrator creates git worktree for agent
3. Orchestrator assembles context bundle (deterministic)
4. Orchestrator dispatches: claude -p with constrained tools/budget/schema
5. Agent works within bounds. Reports findings + output as structured response.
6. Orchestrator validates output against "done when" criteria
7. On pass: agent creates PR, orchestrator merges worktree, updates work state
8. On fail: retry policy kicks in
9. Orchestrator curates findings into knowledge store (if worthy)
10. Orchestrator dispatches next ready tasks
```

### Context Preservation on Agent Failure

When an agent fails or is killed:
1. **Mutation log** records task status
2. **Git worktree** preserves partial changes
3. **Full session log** preserves the agent's entire conversation (every tool call, every response)
4. **Audit log** records events

**Retry with full context:** On retry, the system can reload the failed agent's session log. The retrying agent sees exactly what the previous agent tried, why it failed, and what approaches didn't work. This is not a summary — it's the full transcript, truncated to fit within the context budget.

**Structured handoff on context overflow:** When an agent hits 90% context, it produces a handoff document:

```json
{
  "task_id": "abc123",
  "checklist_completed": ["item 1", "item 2"],
  "checklist_remaining": ["item 3", "item 4"],
  "current_approach": "Using FTS5 with porter tokenizer for stemming",
  "design_decisions": [
    {"decision": "Used WAL mode for SQLite", "reason": "Need concurrent reads from daemon"},
    {"decision": "Chunked by heading not by token count", "reason": "Headings are natural semantic boundaries"}
  ],
  "alternatives_considered": [
    {"approach": "Token-based chunking", "rejected_because": "Splits mid-sentence, loses coherence"}
  ],
  "known_issues": ["FTS5 doesn't handle CJK tokenization well"],
  "files_modified": ["src/corc/search.py", "tests/test_search.py"],
  "pr_url": "https://github.com/.../pull/5",
  "what_to_do_next": "Implement hybrid search score normalization. Start from search.py line 142."
}
```

This preserves not just *what* was done but *why* — design decisions, alternatives considered, known issues. The next agent gets the full qualitative understanding.

Tasks should be scoped small enough that handoffs are rare (target: <5% of tasks). When they happen, they should only happen once per task.

### Daemon Restart & Recovery

The daemon must recover to correct state after any crash. On startup, it runs a reconciliation process:

**1. Rebuild SQLite from mutation log:**
```
corc start → read data/mutations.jsonl → replay into data/state.db
```
This guarantees the materialized view matches the source of truth.

**2. Reconcile "running" tasks:**
For each task marked "running" in the work state:
- Check if the agent process is still alive (PID check + process name verification)
- Check if the git worktree has new commits since dispatch
- Check if the agent produced output files
- **If agent is alive**: let it continue, re-attach monitoring
- **If agent is dead + output exists**: process the output (validate, merge, update state)
- **If agent is dead + no output**: mark task as failed, apply retry policy

**3. Clean up stale worktrees:**
Remove worktrees for agents that are dead and whose tasks have been reconciled.

**4. Resume normal operation:**
After reconciliation, the daemon enters its normal loop. No duplicate dispatches, no lost work.

**Planning session persistence:**
`corc plan` auto-saves the draft spec and partial task list to `.corc/drafts/TIMESTAMP.md` every 60 seconds during the interactive session. If the session crashes:
```
corc plan --resume            # Resume from last auto-save
```

### Repo Merge Policies

Different repos have different merge rules. Configured per-repo in `.corc/repos.yaml`:

```yaml
repos:
  internal-tool:
    merge_policy: auto         # Agent merges via hook after reviewer approval
    protected_branches: [main]
    require_reviewer_approval: true

  production-app:
    merge_policy: human-only   # Only humans can merge. Agent creates PR, human merges.
    protected_branches: [main, staging]
    require_reviewer_approval: true
    block_auto_merge: true     # Prevent agents from enabling auto-merge on PRs
    block_direct_push: true    # Prevent agents from pushing directly to protected branches
```

Merge policies are enforced by `PreToolUse` hooks:
- `block_direct_push`: hook blocks `git push` to protected branches
- `block_auto_merge`: hook blocks `gh pr merge --auto` commands
- `human-only`: the processor creates the PR but does NOT merge it. The PR is left for human review and merge.

### Session Logging

Every agent session is logged in full to `data/sessions/TASK_ID-ATTEMPT.jsonl`. This includes:
- Every tool call and its result
- Every LLM response
- Token counts per turn
- Timing data

Session logs serve three purposes:
1. **Retry context**: failed agent's full session is available to the retrying agent
2. **Audit**: complete record of what happened during every task
3. **Rating**: the evaluator can review the full session, not just the final output

Session logs are local-only (not Git-tracked) due to size, but included in the backup strategy.

### Curation Blacklist

A persistent, growing list of things agents should NOT do. Stored at `.corc/blacklist.md`:

```markdown
# Agent Blacklist

## Architectural
- Never use ORM. Always use raw SQL with parameterized queries. (Reason: ORMs hide query complexity and make debugging harder)
- Never add a new dependency without checking if an existing one covers the use case. (Reason: dependency bloat)

## Code Patterns
- Never use `eval()` or `exec()`. (Reason: security)
- Never catch bare `except:`. Always catch specific exceptions. (Reason: masks bugs)

## Process
- Never merge directly to main. Always create a PR. (Reason: review gate)
- Never skip tests to make a deadline. (Reason: technical debt compounds)
```

The blacklist is:
- Injected into every agent's context (part of the role system prompt)
- Updated when curation rejects a finding (the rejection reason is added as a blacklist entry)
- Updated when recurring mistakes are detected by `corc analyze patterns`
- Reviewed by the operator periodically to prune stale entries

**Curation rejection tracking:** When the orchestrator rejects an agent finding, it logs the rejection with a reason to the mutation log. If the same type of finding is rejected 3+ times, it's auto-added to the blacklist as a candidate (operator confirms).

---

## 12. Implementation Plan

**Test project: CORC builds itself.** Each phase is managed by the system as it's built, providing immediate feedback. See `PLAN.md` for full detail including build lists, tests, gates, and TUI milestones.

Phases are sequential. No time estimates — move as fast as the work allows. Each phase is done when its success criteria are met and the gate passes.

| Phase | Key Deliverable | TUI Milestone |
|---|---|---|
| **0: Skeleton** | Core loop end-to-end | v0: live event stream |
| **1A: Knowledge Store** | Hybrid search | — |
| **1B: Workflow Engine** | Daemon + DAG + planning | v1: + DAG view |
| **2A: Parallel** | Git worktrees + merge | v2: full dashboard |
| **2B: Roles** | Scout → implement → review pipeline | — |
| **3A: Rating** | Scoring + feedback loop | — |
| **3B: Resilience** | Chaos monkey + notifications | + mermaid export |
| **4: Integration** | Quarter-length project test | — |

---

## 13. Adaptability & Trust Levels

The system should get simpler over time as models improve. A configurable trust level controls guardrail intensity.

### Trust Level Dial

| Level | Name | Behavior |
|---|---|---|
| 1 | **Strict** (default) | Full validation on every task. All hooks active. Adversarial review required. |
| 2 | **Standard** | Skip validation for task types with >90% first-attempt success rate. Adversarial review optional for low-risk tasks. |
| 3 | **Relaxed** | Expand agent tool access. Reduce retry counts. Allow agents to self-validate for well-understood task types. |
| 4 | **Autonomous** | Minimal guardrails. Agents can create subtasks, write to knowledge store, and self-review. Human reviews results post-hoc. |

Trust level is configured globally and can be overridden per workflow or per task. The system tracks which trust level was used for each run so we can correlate trust level with quality scores.

### Replace-When-Native Markers

| Module | Replaces When... |
|---|---|
| Knowledge Store | Claude Code gets native persistent memory with search across sessions |
| Work State | Claude Code gets native task tracking with dependency resolution |
| Context Assembly | Claude Code gets native context curation that outperforms curated bundles |
| Observability | Claude Code gets native cost/token tracking and analysis |
| Rating System | Claude Code gets native quality scoring |
| Agent Roles | Claude Code gets native role-based agent configuration |

When a native replacement is available, we test it against our module on real tasks. If it scores comparably on our rating system, we deprecate our module.

### Graceful Deprecation Path

When a native Claude Code feature replaces a CORC module:

1. **Parallel run**: Run both the CORC module and the native feature on the same tasks for 1-2 weeks
2. **Comparative rating**: Score both using our rating system on the same dimensions
3. **Data migration**:
   - Knowledge store → export markdown files (they're already the source of truth, so no migration needed)
   - Work state → mutation log is Git-tracked JSONL, portable to any system that can read it
   - Audit log → JSONL files, standard format, can be ingested by any analytics tool
   - Workflow YAMLs → may need translation to native format; document the mapping
   - Role configs → may need translation; document the mapping
4. **Deprecation**: If native scores within 10% of CORC module, deprecate CORC module. Remove the code. Don't maintain compatibility shims.
5. **Rollback window**: Keep the deprecated code on a git branch for 30 days in case the native feature regresses

The key constraint that makes this work: **every data layer's source of truth is a portable format** (markdown files, JSONL). SQLite is always derived. Nothing is locked in.

---

## 14. Open Questions

1. **Embedding model**: `all-MiniLM-L6-v2` (384-dim) is the default. Should we benchmark alternatives for our specific document types?
2. **Workflow versioning**: When a workflow YAML changes mid-project, what happens to in-flight runs?
3. **Inter-agent communication**: Is pure orchestrator-mediated coordination sufficient, or will some tasks need direct agent-to-agent messaging? Defer decision until Phase 3 experience.
4. **Mutation log compaction**: At what point does the log get large enough to need compaction? Strategy: snapshot SQLite + truncate log.
5. **tmux vs pure `claude -p`**: For complex multi-file tasks, interactive tmux sessions may be needed. When exactly does a task warrant interactive mode vs. single-shot?
6. **Planner quality**: The planner creates task DAGs and context bundles. How do we validate planner output quality before executing? Human review is the current answer, but can we automate this?

---

## 15. What This System Will NOT Do

- Will not replace Claude Code — it augments it
- Will not implement custom LLM inference — uses Claude via `claude -p`
- Will not build a GUI — CLI/TUI only (Obsidian is the optional visual interface)
- Will not use Claude Code Agent Teams initially — orchestrator dispatches independent agents
- Will not build sophisticated browser automation — defer to Claude Code's native capabilities
- Will not attempt to compete with or replace improvements to Claude Code itself
- Will not manage infrastructure or deployment — this is a development orchestration tool
- Will not let agents write directly to the knowledge store — curation is a quality gate
- Will not maintain dead code — when a native feature replaces a module, deprecate immediately
