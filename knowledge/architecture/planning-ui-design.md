---
id: planning-ui-design
type: architecture
project: corc
repos: [corc]
tags: [planning, browser-ui, design, architecture, gui]
created: 2026-03-23T00:00:00Z
updated: 2026-03-23T00:00:00Z
source: human
status: active
---

# Planning UI — Native Browser GUI Design

## Problem

The `corc plan` CLI flow relies on an interactive Claude Code terminal session. This works but limits the planning experience:

- **No visual DAG editing** — operators see ASCII art but can't drag/rearrange tasks.
- **No persistent split-pane** — spec and task DAG can't be viewed side-by-side.
- **No inline diff for draft review** — changes between planning iterations are hard to spot.
- **Limited structured input** — creating tasks with many fields (role, depends_on, checklist, context_bundle) via CLI flags is error-prone.
- **No real-time monitoring integration** — the planning view and the execution dashboard are separate tools (`corc plan` vs `corc watch`).

A native browser GUI provides richer interaction for the three-stage planning flow while reusing the existing data layers and query API.

## Requirements

- [ ] Three-stage planning flow (spec → decomposition → review/approve) as distinct UI states
- [ ] Spec editor with live markdown preview
- [ ] Visual DAG rendering with status colors, matching existing `dag.py` semantics
- [ ] Draft task creation/editing with structured form inputs
- [ ] Inline review of spec + task DAG before commit (approve/reject workflow)
- [ ] Spec ↔ task DAG bidirectional linking visible in the UI
- [ ] Query API endpoints for all read operations; mutation endpoints for writes
- [ ] Real-time updates via server-sent events (SSE) for running plans
- [ ] Reuse existing data layers (WorkState, MutationLog, AuditLog, KnowledgeStore)
- [ ] Works as a local-only server (localhost); no auth required initially

## Non-Requirements

- Multi-user collaboration (single operator)
- Cloud deployment or remote access
- Replacing the CLI planning flow (browser UI is an alternative, not a replacement)
- AI chat interface in the browser (Claude interaction stays in terminal or is a later phase)
- Mobile responsiveness

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                     Browser (localhost:8420)                  │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
│  │ Spec Editor  │  │ DAG Canvas  │  │  Review/Approve      │ │
│  │ (Stage 1)    │  │ (Stage 2)   │  │  Panel (Stage 3)     │ │
│  └──────┬───────┘  └──────┬──────┘  └──────────┬───────────┘ │
│         │                 │                     │             │
│         └────────┬────────┴─────────────────────┘             │
│                  │  fetch / SSE                               │
└──────────────────┼───────────────────────────────────────────┘
                   │
┌──────────────────┼───────────────────────────────────────────┐
│                  │  HTTP API (localhost:8420/api)             │
│  ┌───────────────▼───────────────────────────────────────┐   │
│  │           FastAPI / Starlette Server                   │   │
│  │                                                       │   │
│  │   GET  /api/plan/tasks          (read active plan)    │   │
│  │   GET  /api/plan/tasks/ready    (ready tasks)         │   │
│  │   GET  /api/plan/tasks/blocked  (blocked + reasons)   │   │
│  │   GET  /api/plan/tasks/draft    (draft tasks)         │   │
│  │   GET  /api/plan/dag            (dag structure+layout)│   │
│  │   GET  /api/events              (recent events)       │   │
│  │   GET  /api/events/stream       (SSE live events)     │   │
│  │   GET  /api/knowledge/search    (knowledge search)    │   │
│  │   GET  /api/knowledge/doc/:id   (read document)       │   │
│  │   GET  /api/spec/templates      (spec templates)      │   │
│  │                                                       │   │
│  │   POST /api/draft/spec          (save draft spec)     │   │
│  │   POST /api/draft/tasks         (save draft tasks)    │   │
│  │   PUT  /api/draft/tasks/:id     (edit draft task)     │   │
│  │   DELETE /api/draft/tasks/:id   (remove draft task)   │   │
│  │   POST /api/plan/approve        (commit spec+tasks)   │   │
│  │   POST /api/plan/reject         (discard draft plan)  │   │
│  └───────────────────────────────────────────────────────┘   │
│                           │                                  │
│  ┌────────────────────────▼──────────────────────────────┐   │
│  │              Existing Data Layers                     │   │
│  │                                                       │   │
│  │   QueryAPI  ──  WorkState  ──  MutationLog            │   │
│  │   AuditLog  ──  SessionLogger  ──  KnowledgeStore     │   │
│  └───────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

The server is a thin HTTP layer over the existing Python data layers. It introduces **no new data storage** — all state flows through MutationLog, WorkState, and KnowledgeStore as today.

---

## Design: Planning Flow in the Browser

### Stage 1 — Spec Development

The operator opens `http://localhost:8420/plan` (or `corc plan --gui` launches a browser tab). The UI presents a split-pane view:

**Left pane: Spec editor** — a markdown textarea with live preview. Pre-populated with the spec template from `plan.py`. The operator writes the spec manually or pastes from a seed document.

**Right pane: Context sidebar** — shows:
- Knowledge store search (hits `GET /api/knowledge/search?q=...`)
- Current work state summary (hits `GET /api/plan/tasks`)
- Repository file tree (hits `GET /api/repo/tree`)

The operator develops the spec, searching the knowledge store for prior art. Draft spec is auto-saved to the server every 30 seconds via `POST /api/draft/spec`.

**Transition:** Operator clicks "Decompose →" to advance to Stage 2. The current spec content is sent to the server and persisted as a draft.

### Stage 2 — Task Decomposition

The UI switches to a three-panel layout:

**Left pane: Spec (read-only with section anchors)** — the spec from Stage 1, rendered as HTML with clickable section headers. Each section generates an anchor ID (e.g., `#design`, `#requirements`).

**Center pane: Task DAG** — an interactive directed acyclic graph rendered with SVG/Canvas. Tasks are nodes; `depends_on` relationships are directed edges. Layout uses the same topological leveling algorithm from `dag.py` (`assign_levels` → `assign_rows`).

**Right pane: Task detail / creation form** — when a task node is selected or "Add Task" is clicked, shows a structured form:

| Field | Input Type | Validation |
|---|---|---|
| name | text input | Required, slug format |
| description | textarea | Required |
| role | dropdown: scout, implementer, reviewer, adversarial-reviewer | Required |
| depends_on | multi-select from existing task names | DAG acyclicity check |
| done_when | textarea | Lint for subjectivity (reuse `lint_done_when`) |
| checklist | dynamic list of text inputs | At least 1 item |
| context_bundle | file picker / text list | Files must exist |
| priority | number input (1-100) | Range check |
| task_type | dropdown: implementation, investigation, bugfix | Required |
| spec_section | dropdown: links to spec section anchors | Optional |

**Spec ↔ Task linking (see dedicated section below):** Each task has a `spec_section` field referencing which section of the spec it implements. The spec pane highlights sections that have linked tasks vs. unlinked ones.

All tasks are created in `draft` status via `POST /api/draft/tasks`. The DAG re-renders after each add/edit/delete. The server validates acyclicity before accepting new `depends_on` edges.

**Transition:** Operator clicks "Review →" to advance to Stage 3.

### Stage 3 — Review and Approve

The UI presents a final review layout:

**Top half: Spec with linked tasks** — the spec rendered as HTML with inline task badges next to each section header showing which tasks implement that section. Clicking a badge scrolls to the task in the DAG below.

**Bottom half: Full DAG with task details** — the complete DAG with all task metadata visible on hover/click. Color-coded by draft status:
- 📝 (blue outline): draft, not yet approved
- Task metadata shown inline: role, done_when, depends_on, checklist item count

**Review actions:**
- **Edit**: Click any task node to reopen the edit form (returns to Stage 2 context for that task).
- **Reorder**: Drag edges in the DAG to change `depends_on` (validates acyclicity).
- **Delete**: Remove a draft task (cascades: removes from other tasks' `depends_on`).
- **Approve All** (`POST /api/plan/approve`): Atomically commits the spec to the knowledge store and all draft tasks to the work state as `pending`. The mutation log receives one `task_created` entry per task, plus the spec is written to `knowledge/`.
- **Reject** (`POST /api/plan/reject`): Discards all draft tasks and the draft spec. Confirms via dialog.

After approval, the UI redirects to a "Plan committed" summary showing the spec ID and task IDs, with a link to the monitoring dashboard.

---

## Query API Endpoints

All endpoints are served by a lightweight Python HTTP server (FastAPI recommended for async + automatic OpenAPI docs). The server wraps `QueryAPI`, `WorkState`, `MutationLog`, `AuditLog`, and `KnowledgeStore`.

### Read Endpoints (wrap existing QueryAPI)

| Endpoint | Method | Maps to | Returns |
|---|---|---|---|
| `/api/plan/tasks` | GET | `QueryAPI.get_active_plan_tasks()` | `list[TaskDict]` — active plan tasks sorted by priority |
| `/api/plan/tasks/ready` | GET | `QueryAPI.get_ready_tasks()` | `list[TaskDict]` — tasks ready for dispatch |
| `/api/plan/tasks/blocked` | GET | `QueryAPI.get_blocked_tasks_with_reasons()` | `list[TaskDict]` — blocked tasks with `blocked_by` and `reason` |
| `/api/plan/tasks/draft` | GET | `WorkState.list_tasks(status="draft")` | `list[TaskDict]` — draft tasks awaiting approval |
| `/api/plan/tasks/running` | GET | `QueryAPI.get_running_tasks_with_agents()` | `list[TaskDict]` — running tasks with agent info |
| `/api/plan/task/:id` | GET | `WorkState.get_task(id)` | `TaskDict` — single task detail |
| `/api/plan/dag` | GET | *new* — compute DAG layout | `{nodes: [...], edges: [...], levels: {...}}` |
| `/api/events` | GET | `QueryAPI.get_recent_events(n)` | `list[EventDict]` — recent audit events |
| `/api/events/stream` | GET (SSE) | *new* — poll audit log | Server-Sent Events stream |
| `/api/events/task/:id` | GET | `QueryAPI.get_task_stream_events(id)` | `list[EventDict]` — stream events for task |
| `/api/knowledge/search` | GET | `KnowledgeStore.search(q)` | `list[SearchResult]` — hybrid search results |
| `/api/knowledge/doc/:id` | GET | `KnowledgeStore.get_doc(id)` | `{id, type, content, frontmatter}` |
| `/api/spec/templates` | GET | *new* — read from templates | `{spec_template: str, task_fields: [...]}` |
| `/api/repo/tree` | GET | *new* — list source files | `list[str]` — relative file paths |

### Write Endpoints (new)

| Endpoint | Method | Action | Details |
|---|---|---|---|
| `/api/draft/spec` | POST | Save draft spec | Body: `{content: str, title: str}`. Saves to `.corc/drafts/plan-<slug>.md`. Returns `{draft_id, path}`. |
| `/api/draft/tasks` | POST | Create draft task | Body: `TaskCreatePayload`. Writes `task_created` mutation with `status: "draft"`. Returns `{task_id}`. |
| `/api/draft/tasks/:id` | PUT | Edit draft task | Body: partial `TaskDict`. Writes `task_updated` mutation. Only allowed for draft tasks. Returns `{task_id}`. |
| `/api/draft/tasks/:id` | DELETE | Remove draft task | Writes `task_updated` mutation setting status to a terminal state. Removes from others' `depends_on`. |
| `/api/plan/approve` | POST | Commit plan | Body: `{spec_draft_id, task_ids: [...]}`. Atomically: (1) saves spec to `knowledge/`, (2) writes `task_approved` mutation for each draft task, (3) logs `plan_approved` audit event. Returns `{spec_id, approved_task_ids}`. |
| `/api/plan/reject` | POST | Discard plan | Body: `{spec_draft_id, task_ids: [...]}`. Removes draft tasks and spec. Returns `{discarded}`. |
| `/api/plan/validate-dag` | POST | Check DAG validity | Body: `{tasks: [...]}`. Validates acyclicity and dependency existence. Returns `{valid: bool, errors: [...]}`. |
| `/api/plan/lint-done-when` | POST | Lint criteria | Body: `{done_when: str, task_type: str}`. Returns `{warnings: [...], ok: bool}`. |

### New QueryAPI Methods Required

The following methods must be added to `QueryAPI` (in `src/corc/queries.py`):

```python
def get_draft_tasks(self) -> list[dict]:
    """All tasks in draft status, sorted by creation time."""
    return self.work_state.list_tasks(status="draft")

def get_dag_layout(self) -> dict:
    """Compute DAG node/edge layout for rendering.

    Returns {
        nodes: [{id, name, status, role, level, row, ...}],
        edges: [{from_id, to_id}],
        levels: {task_id: int},
        summary: {total, completed, running, ready, blocked}
    }
    """
    # Delegates to dag.py compute_effective_status + assign_levels + assign_rows

def get_plan_summary(self) -> dict:
    """High-level plan stats for dashboard header.

    Returns {
        total_tasks, completed, running, pending, draft, failed, escalated,
        progress_pct, estimated_cost, actual_cost
    }
    """
```

### SSE Event Stream

`GET /api/events/stream` opens a Server-Sent Events connection. The server polls the audit log every 2 seconds (or uses filesystem watch via `watchdog`) and pushes new events:

```
event: task_completed
data: {"task_id": "impl-search", "name": "Implement search", "timestamp": "..."}

event: task_started
data: {"task_id": "impl-dag", "name": "Implement DAG", "timestamp": "..."}
```

This powers real-time DAG status updates in the browser without page refreshes.

---

## Draft Task Review and Approval Flow

### Draft Lifecycle

```
                ┌──────────┐
                │  (empty)  │
                └─────┬─────┘
                      │ POST /api/draft/tasks
                      ▼
                ┌──────────┐
       ┌───────│   draft   │◄──────────┐
       │       └─────┬─────┘           │
       │             │                 │
       │ DELETE      │ PUT (edit)      │ (reject → re-draft)
       │             │                 │
       ▼             ▼                 │
  ┌──────────┐  ┌──────────┐          │
  │ discarded│  │   draft   │──────────┘
  └──────────┘  │ (updated) │
                └─────┬─────┘
                      │ POST /api/plan/approve
                      ▼
                ┌──────────┐
                │  pending  │ → daemon picks up
                └──────────┘
```

1. **Create drafts**: Each task is created via `POST /api/draft/tasks` with `status: "draft"`. This writes a real `task_created` mutation to the log — drafts are first-class tasks, just not schedulable.

2. **Edit drafts**: `PUT /api/draft/tasks/:id` writes a `task_updated` mutation. The server rejects edits to non-draft tasks (400 error).

3. **Validate**: Before showing the "Approve" button, the client calls `POST /api/plan/validate-dag` to check:
   - All `depends_on` references exist
   - No cycles in the dependency graph
   - All `done_when` criteria pass linting
   - Every task has required fields populated

4. **Approve**: `POST /api/plan/approve` is an atomic operation:
   - Writes the spec markdown to `knowledge/<type>/<slug>.md` with YAML frontmatter including `task_ids: [...]`
   - For each draft task, appends a `task_approved` mutation (transitions draft → pending)
   - Appends a `plan_approved` audit event with spec_id and all task_ids
   - The daemon's next poll picks up newly-pending tasks with satisfied dependencies

5. **Reject**: `POST /api/plan/reject` discards the plan. Draft tasks receive a terminal mutation. The draft spec file is deleted.

### Batch vs. Selective Approval

The UI supports two approval modes:
- **Approve All**: Commits the entire plan (spec + all draft tasks) atomically.
- **Selective Approve**: Operator checks/unchecks individual tasks. Unchecked tasks remain in draft. Useful for phased rollout — approve the first wave, review remaining tasks later.

---

## Spec ↔ Task DAG Linking

### How Linking Works

The spec and task DAG are connected through bidirectional references stored in existing data structures:

**Spec → Tasks (in spec frontmatter):**
```yaml
---
id: feature-semantic-search
type: architecture
task_ids:
  - scout-search-options
  - impl-fts5-index
  - impl-semantic-embed
  - impl-hybrid-ranker
  - review-search-quality
---
```

**Task → Spec (in task data):**
```json
{
  "id": "impl-fts5-index",
  "name": "Implement FTS5 indexing",
  "spec_id": "feature-semantic-search",
  "spec_section": "design.indexing",
  ...
}
```

**Task → Spec Section (granular linking):**
Each task's `spec_section` field is a dot-path into the spec's heading structure:
- `"requirements"` → links to `## Requirements`
- `"design.indexing"` → links to `## Design` → `### Indexing`
- `"testing-strategy"` → links to `## Testing Strategy`

### How the UI Renders Links

**In the Spec pane:**
Each heading shows a badge count of linked tasks. Clicking the badge filters the DAG to show only those tasks. Unlinked sections are highlighted with a warning icon — they represent spec requirements with no implementing task.

```
## Requirements                          [3 tasks ▸]
- [x] Requirement 1                      ← impl-fts5-index
- [x] Requirement 2                      ← impl-semantic-embed
- [ ] Requirement 3                      ← (no task!) ⚠️

## Design
### Indexing                             [1 task ▸]
### Ranking                              [1 task ▸]

## Testing Strategy                      [1 task ▸]
```

**In the DAG pane:**
Each task node shows a small spec-section indicator. Clicking it scrolls the spec pane to that section. Tasks without a `spec_section` are flagged for review.

### Coverage Analysis

The review screen (Stage 3) includes a "Coverage" panel showing:
- Spec sections with 0 linked tasks → **uncovered** (⚠️)
- Spec sections with all linked tasks completed → **done** (✅)
- Spec sections with linked tasks in progress → **in progress** (🔄)

This ensures no spec requirement is accidentally left unimplemented.

---

## Text Wireframes

### Screen 1: Plan Home / New Plan

```
┌─────────────────────────────────────────────────────────────────────┐
│  CORC Planning                                    [Active Plans ▾]  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    Start a New Plan                         │   │
│   │                                                             │   │
│   │   [🆕 Blank Plan]   [📄 From Seed File]   [🔄 Resume]     │   │
│   │                                                             │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│   Active Plans                                                      │
│   ┌────────────────────────────────────────────────────────────┐    │
│   │  📋 semantic-search     5 tasks  ████████░░ 60%  running  │    │
│   │  📋 dag-visualization   3 tasks  ██████████ 100% done     │    │
│   │  📝 notification-system 4 tasks  ░░░░░░░░░░ 0%   draft   │    │
│   └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│   Quick Task                                                        │
│   ┌────────────────────────────────────────────────────────────┐    │
│   │  Name: [________________________]                          │    │
│   │  Done when: [____________________]   [Create Quick Task]   │    │
│   └────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Screen 2: Stage 1 — Spec Editor

```
┌─────────────────────────────────────────────────────────────────────┐
│  CORC Planning > New Plan          Stage: [1·Spec] 2·Tasks 3·Review │
├──────────────────────────────┬──────────────────────────────────────┤
│  Spec Editor                 │  Context                             │
│  ─────────────               │  ────────                            │
│                              │                                      │
│  # Semantic Search           │  🔍 [Search knowledge...________]    │
│                              │                                      │
│  ## Problem                  │  Results:                            │
│  Current keyword search      │  ┌──────────────────────────────┐   │
│  misses semantically         │  │ 📄 three-layer-data-arch     │   │
│  similar content...          │  │    "SQLite FTS5 for keyword   │   │
│                              │  │    search, sentence-transform │   │
│  ## Requirements             │  │    ers for semantic..."       │   │
│  - [ ] Sentence-transformer  │  │ 📄 phase-0-learnings         │   │
│        embeddings            │  │    "Search was keyword-only   │   │
│  - [ ] Hybrid ranking        │  │    in phase 0..."            │   │
│  - [ ] Graceful fallback     │  └──────────────────────────────┘   │
│                              │                                      │
│  ## Non-Requirements         │  Current Work State                  │
│  - GPU acceleration          │  ──────────────────                  │
│  - Real-time indexing        │  ✅ 12 completed                    │
│                              │  🔄  2 running                      │
│  ## Design                   │  ⬚   4 pending                      │
│  ### Indexing                │  📝  0 draft                        │
│  ...                         │                                      │
│                              │  Repository Files                    │
│  ## Testing Strategy         │  ──────────────────                  │
│  ...                         │  src/corc/knowledge.py               │
│                              │  src/corc/search.py                  │
│  ## Rationale                │  src/corc/state.py                   │
│  ...                         │  ...                                 │
│                              │                                      │
├──────────────────────────────┴──────────────────────────────────────┤
│  Auto-saved 10s ago                              [← Back] [→ Decompose]│
└─────────────────────────────────────────────────────────────────────┘
```

### Screen 3: Stage 2 — Task Decomposition

```
┌─────────────────────────────────────────────────────────────────────┐
│  CORC Planning > Semantic Search   Stage: 1·Spec [2·Tasks] 3·Review │
├───────────────┬──────────────────────────┬──────────────────────────┤
│ Spec (ref)    │  Task DAG                │  Task Detail             │
│ ────────────  │  ────────                │  ───────────             │
│               │                          │                          │
│ # Semantic    │  ┌─────────────────┐     │  ✏️  impl-fts5-index    │
│   Search      │  │ scout-search    │     │                          │
│               │  │ ⬚ scout        │     │  Name:                   │
│ ## Problem    │  └────────┬────────┘     │  [impl-fts5-index___]    │
│ [2 tasks ▸]   │           │              │                          │
│               │           ▼              │  Description:            │
│ ## Require-   │  ┌─────────────────┐     │  [Build FTS5 virtual ]   │
│ ments         │  │ impl-fts5-index │     │  [table indexing for ]   │
│ [1 task ▸]    │  │ 📝 implementer ◄── selected                     │
│               │  └────────┬────────┘     │  Role:                   │
│ ## Design     │           │              │  [implementer      ▾]    │
│  ### Indexing  │           ▼              │                          │
│  [1 task ▸]   │  ┌─────────────────┐     │  Depends on:             │
│  ### Ranking  │  │ impl-semantic   │     │  [☑ scout-search    ]    │
│  [1 task ▸]   │  │ 📝 implementer │     │  [☐ impl-semantic   ]    │
│               │  └────────┬────────┘     │                          │
│ ## Testing    │           │              │  Done when:              │
│ [1 task ▸]    │           ▼              │  [FTS5 virtual table ]   │
│               │  ┌─────────────────┐     │  [created; 'corc kn  ]   │
│ ## Rationale  │  │ impl-hybrid     │     │  [search' returns... ]   │
│               │  │ 📝 implementer │     │  ✅ Lint passed          │
│               │  └────────┬────────┘     │                          │
│               │           │              │  Checklist:              │
│               │           ▼              │  [1. Create schema   ]   │
│               │  ┌─────────────────┐     │  [2. Build indexer   ]   │
│               │  │ review-quality  │     │  [+ Add item          ]  │
│               │  │ 📝 reviewer    │     │                          │
│               │  └─────────────────┘     │  Context bundle:         │
│               │                          │  [src/corc/knowledge.]   │
│               │  [+ Add Task]            │  [SPEC.md            ]   │
│               │                          │  [+ Add file          ]  │
│               │                          │                          │
│               │                          │  Spec section:           │
│               │                          │  [design.indexing   ▾]   │
│               │                          │                          │
│               │                          │  Priority: [50_____]     │
│               │                          │                          │
│               │                          │  [Save Draft] [Delete]   │
│               │                          │                          │
├───────────────┴──────────────────────────┴──────────────────────────┤
│  5 draft tasks  |  DAG valid ✅                  [← Spec] [→ Review]│
└─────────────────────────────────────────────────────────────────────┘
```

### Screen 4: Stage 3 — Review and Approve

```
┌─────────────────────────────────────────────────────────────────────┐
│  CORC Planning > Semantic Search   Stage: 1·Spec 2·Tasks [3·Review] │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Spec: Semantic Search                              Coverage: 100%  │
│  ════════════════════                                               │
│                                                                     │
│  ## Problem ──────────────────────────── scout-search ⬚            │
│  Current keyword search misses semantically similar content.        │
│                                                                     │
│  ## Requirements ─────────────────────── impl-fts5-index ⬚         │
│  - Sentence-transformer embeddings       impl-semantic ⬚           │
│  - Hybrid ranking                                                   │
│  - Graceful fallback                                                │
│                                                                     │
│  ## Design > Indexing ────────────────── impl-fts5-index ⬚         │
│  ## Design > Ranking ─────────────────── impl-hybrid ⬚             │
│                                                                     │
│  ## Testing Strategy ─────────────────── review-quality ⬚          │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Task DAG (5 tasks, all draft)                              │    │
│  │                                                             │    │
│  │  ☑ scout-search ──► ☑ impl-fts5-index ──┐                  │    │
│  │                                          ├─► ☑ impl-hybrid  │    │
│  │                     ☑ impl-semantic ─────┘       │          │    │
│  │                                                  │          │    │
│  │                                          ☑ review-quality   │    │
│  │                                                             │    │
│  │  ☑ = selected for approval    ☐ = deselected (stays draft)  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─ Task Summary ──────────────────────────────────────────────┐    │
│  │  Name               │ Role        │ Deps    │ Done When     │    │
│  │  ──────────────────  │ ──────────  │ ─────── │ ──────────── │    │
│  │  scout-search       │ scout       │ (none)  │ Research doc  │    │
│  │  impl-fts5-index    │ implementer │ scout.. │ FTS5 table... │    │
│  │  impl-semantic      │ implementer │ scout.. │ Embeddings... │    │
│  │  impl-hybrid        │ implementer │ fts5,.. │ Hybrid rank.. │    │
│  │  review-quality     │ reviewer    │ hybrid  │ Search qual.. │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  [← Edit Tasks]          [✗ Reject Plan]     [✓ Approve & Commit]  │
└─────────────────────────────────────────────────────────────────────┘
```

### Screen 5: Plan Committed (Success)

```
┌─────────────────────────────────────────────────────────────────────┐
│  CORC Planning                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│                    ✅  Plan Committed Successfully                  │
│                                                                     │
│   Spec saved:   knowledge/architecture/semantic-search.md           │
│   Spec ID:      feature-semantic-search                             │
│                                                                     │
│   Tasks created (5):                                                │
│   ┌────────────────────────────────────────────────────────────┐    │
│   │  ⬚ scout-search          scout        ready               │    │
│   │  ◻ impl-fts5-index      implementer  blocked (scout...)   │    │
│   │  ◻ impl-semantic        implementer  blocked (scout...)   │    │
│   │  ◻ impl-hybrid          implementer  blocked (fts5,sem.)  │    │
│   │  ◻ review-quality       reviewer     blocked (hybrid)     │    │
│   └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│   The daemon will pick up ready tasks automatically.                │
│                                                                     │
│   [📊 View DAG Dashboard]    [📋 Start New Plan]    [🏠 Home]      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Screen 6: Monitoring Dashboard (Post-Approval)

```
┌─────────────────────────────────────────────────────────────────────┐
│  CORC Dashboard                                [Plan ▾] [Settings]  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Progress: ████████░░░░░░░░░░░░ 40%  (2/5 tasks)                   │
│  Running: 1  |  Ready: 1  |  Blocked: 2  |  Done: 2                │
│                                                                     │
│  ┌─ DAG ───────────────────────────────────────────────────────┐    │
│  │                                                             │    │
│  │  scout-search ✅ ──► impl-fts5-index ✅ ──┐                 │    │
│  │                                           ├─► impl-hybrid ⬚ │    │
│  │                      impl-semantic 🔄 ────┘       │         │    │
│  │                                                   │         │    │
│  │                                           review-quality ◻  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─ Live Events ───────────────────────────────────────────────┐    │
│  │  10:15:02  impl-semantic   🔄 Started ($0.00)              │    │
│  │  10:14:58  impl-fts5-index ✅ COMPLETED ($1.23)            │    │
│  │  10:10:30  impl-fts5-index ☑ Checklist: "Create schema"   │    │
│  │  10:08:12  scout-search    ✅ COMPLETED ($0.89)            │    │
│  │  10:05:00  scout-search    🔄 Started ($0.00)              │    │
│  │  10:04:55  plan_approved   📋 5 tasks committed            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─ Running Agents ────────────────────────────────────────────┐    │
│  │  impl-semantic (implementer) — 2m 15s                       │    │
│  │    ☑ Install sentence-transformers                          │    │
│  │    ☐ Create embedding pipeline                              │    │
│  │    ☐ Write tests                                            │    │
│  │    ☐ Integration test with knowledge store                  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Notes

### Technology Choices

| Component | Choice | Rationale |
|---|---|---|
| HTTP server | FastAPI (Starlette) | Async, auto OpenAPI docs, lightweight, Python-native |
| SSE | Starlette `EventSourceResponse` | No WebSocket complexity; sufficient for one-way push |
| Frontend | Vanilla JS + minimal framework (Preact or htmx) | Keeps bundle small; no build step needed for htmx |
| DAG rendering | SVG via D3.js or dagre-d3 | Proven DAG layout library; interactive node selection |
| Markdown preview | marked.js | Lightweight, well-maintained |
| Spec editor | CodeMirror or textarea | CodeMirror for syntax highlighting; textarea for simplicity |

### Server Startup

The server launches via `corc serve` (new CLI command) or `corc plan --gui`:

```bash
corc serve                    # Start HTTP server on localhost:8420
corc serve --port 9000        # Custom port
corc plan --gui               # Start server + open browser to /plan
```

The server process is independent of the daemon — it's a read/write HTTP layer over the same data files. Both can run simultaneously because:
- Reads go through SQLite (concurrent readers are safe)
- Writes go through MutationLog which uses `flock` for atomic appends

### Draft Isolation

Draft tasks are real tasks in the work state with `status: "draft"`. The daemon **ignores** draft tasks (they're not in `_ACTIVE_STATUSES` and `get_ready_tasks()` skips them). This means:
- Drafts are persisted and recoverable (just like any task)
- Drafts appear in `corc task list --draft` (CLI) and `GET /api/plan/tasks/draft` (API)
- Approval is a simple status transition via existing `task_approved` mutation type
- No shadow storage or separate draft database needed

### Error Handling

All API endpoints return structured errors:
```json
{
  "error": "validation_failed",
  "message": "DAG contains a cycle: impl-a → impl-b → impl-a",
  "details": {"cycle": ["impl-a", "impl-b", "impl-a"]}
}
```

HTTP status codes: 200 (success), 400 (validation error), 404 (not found), 409 (conflict — e.g., approving non-draft task), 500 (server error).

---

## Rationale

### Why a local HTTP server rather than Electron or a desktop app?

- **Zero additional dependencies** — Python's HTTP ecosystem is mature; no need for Electron's Chromium bundle.
- **Consistent with CORC's "modules communicate via files and shell commands" philosophy** — the HTTP API is just another interface to the same data layers.
- **Enables future remote access** if needed (add auth later).
- **Browser DevTools** are free debugging infrastructure.

### Why SSE instead of WebSockets?

- **One-way data flow** — the browser needs to receive updates, not send streams. User actions are standard HTTP requests.
- **Simpler server implementation** — no connection upgrade, no ping/pong, no reconnection protocol.
- **Native browser support** — `EventSource` API handles reconnection automatically.

### Why draft tasks in the real work state instead of a separate staging area?

- **Reuses existing mutation infrastructure** — no new storage layer.
- **Drafts are visible via existing CLI commands** (`corc task list --draft`).
- **Approval is a single mutation** (`task_approved`), not a copy operation.
- **Consistent with the "mutation log as source of truth" principle** — all state changes are auditable.

### Why spec_section linking via dot-paths instead of embedding task IDs in the spec markdown?

- **Decoupled** — the spec doesn't need to know about task IDs at authoring time.
- **Stable** — renaming a task doesn't break spec content.
- **Bidirectional** — the spec frontmatter lists task_ids (outbound), and each task lists its spec_section (inbound). Either direction can be queried.

### Why three panels in Stage 2 instead of two?

- The spec must remain visible during decomposition so the operator can verify coverage.
- The DAG must be visible to understand the dependency structure.
- The task form must be visible to edit task details.
- Three panels is the minimum for simultaneous visibility. On smaller screens, the spec collapses to a toggle panel.
