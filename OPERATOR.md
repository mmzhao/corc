# CORC — Operator Guide

## Planning

```
corc plan                    # Interactive planning session
corc plan my-idea.md         # Start with a seed document
corc plan --resume           # Resume a crashed planning session
```

The planning session has full system context:
- Knowledge store (all decisions, research, prior outcomes — searchable)
- Work state (what's in progress, done, blocked)
- Repo context (code structure, recent changes, conventions)
- Rating data (what roles/patterns are working well)
- Curation blacklist (what to avoid)

**Three stages:**

1. **Spec** — you and Claude develop the spec together. Claude can search your knowledge store and the web. The spec includes a "Rationale" section capturing *why* you made the decisions you did.

2. **Decomposition** — Claude breaks the spec into a task DAG. Each task gets a role, "done when" criteria, checklist, and context bundle. Claude determines the right level of formality:
   - **Quick task**: one code change, no decomposition (e.g., "fix the README typo")
   - **Standard**: 2-5 tasks with brief spec (e.g., "add semantic search")
   - **Epic**: full spec, 5+ tasks, scout phases, adversarial review (e.g., "build workflow engine")

3. **Review** — you review spec + DAG together. Adjust anything. Approve → spec saved to knowledge store, tasks written to work state.

The daemon picks up new tasks automatically. No restart needed.

## Running Work

```
corc start                   # Start daemon (default: 1 agent)
corc start --parallel 3      # Up to 3 concurrent agents
corc stop                    # Graceful shutdown
```

Leave the daemon running in a terminal tab. It works through ready tasks, idles when empty, picks up new tasks from your planning sessions.

## Monitoring

**Quick check:**
```
corc status                  # Text snapshot with progress
corc dag                     # Visual DAG with status colors
```

**Live dashboard:**
```
corc watch                   # TUI with DAG, events, and agent progress
```

The watch view shows: DAG progress at the top, live events in the middle, running agents with checklist progress at the bottom. You can see agents ticking off checklist items in real time.

## When It Escalates

The daemon pauses and sends you a notification.

```
corc escalations             # See what needs your attention
corc escalation show ESC_ID  # Full detail: what happened, session log, suggested actions
corc escalation resolve ESC_ID  # Mark resolved, daemon resumes
```

Escalations include: the error, the full session log path, the PR (if any), and suggested actions.

## After Work Completes

```
corc rate --auto             # Score all unscored runs
corc ratings --trend         # Quality over time
corc analyze costs --today   # Cost breakdown
corc analyze patterns        # What's working, what's not, recommendations
corc curate RUN_ID           # Review agent findings, persist good ones
corc retro PROJECT_NAME      # Full project retrospective
```

## Typical Session

**Tab 1:** `corc start --parallel 2` (leave running)

**Tab 2:**
```
corc plan                    # Plan new work
corc dag                     # Check the DAG
corc watch                   # Watch when you want detail
corc escalation show ...     # Handle escalations
corc analyze costs --today   # End of day
```

## Flows

### Plan and execute
```
corc plan → develop spec → decompose into tasks → approve
→ Daemon picks up ready tasks automatically
→ corc dag to see progress
→ corc watch for live detail
```

### Quick fix
```
corc plan → "fix the typo in README" → Claude creates one quick task → approve
→ Daemon dispatches it
```

### Escalation
```
Daemon pauses, notifies you
→ corc escalation show ESC_ID (see error, session log, suggestions)
→ Fix the issue or adjust the plan
→ corc escalation resolve ESC_ID (daemon resumes)
```

### Add work mid-project
```
corc plan (in another terminal)
→ Add new tasks
→ Daemon picks them up, no restart
```

## Essential Commands

| Command | What |
|---|---|
| `corc plan` | Plan new work |
| `corc start` | Run the daemon |
| `corc status` | Quick state check |
| `corc dag` | Visual DAG |
| `corc watch` | Live dashboard |
| `corc pause / resume` | Intervene |
| `corc escalations` | Handle escalations |

## Analysis Commands

| Command | What |
|---|---|
| `corc analyze costs` | Cost breakdown |
| `corc analyze patterns` | Quality recommendations |
| `corc rate --auto` | Score runs |
| `corc ratings --trend` | Quality over time |
| `corc curate RUN_ID` | Persist findings |
| `corc retro PROJECT` | Project retrospective |
| `corc log --last N` | Recent events (human-readable) |
