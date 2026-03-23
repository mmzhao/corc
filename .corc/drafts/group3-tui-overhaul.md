# Group 3: TUI Overhaul

Priority: 30

Design principle: Build as a data layer + rendering layer so a browser GUI can reuse the data layer later. All state queries go through a clean API, not direct SQLite. The TUI is one renderer; a web frontend would be another.

## 3A. Task query API (data layer for TUI and future web GUI)

**Type:** implementation
**Priority:** 30
**Depends on:** 1C (task types), 1D (draft status)
**Done when:** A `corc.queries` module provides functions: `get_active_plan_tasks()`, `get_running_tasks_with_agents()`, `get_ready_tasks()`, `get_blocked_tasks_with_reasons()`, `get_recent_events(n)`, `get_task_stream_events(task_id)`; all return plain dicts (JSON-serializable); tests verify each query returns correct data
**Checklist:**
- New module: src/corc/queries.py
- get_active_plan_tasks(): filter to non-completed tasks + recently completed (last hour)
- get_running_tasks_with_agents(): running tasks + agent info + checklist progress
- get_ready_tasks(): pending tasks with all deps met
- get_blocked_tasks_with_reasons(): pending tasks with unmet deps, show which dep is blocking
- get_recent_events(n): last N events from audit log, formatted
- get_task_stream_events(task_id): streaming events for a running task
- All return plain dicts (no SQLite Row objects, no rich objects)
- Tests for each query

**Context bundle:** src/corc/state.py, src/corc/audit.py, src/corc/sessions.py

---

## 3B. TUI overhaul: active-plan focus

**Type:** implementation
**Priority:** 30
**Depends on:** 3A
**Done when:** `corc watch` shows only active/relevant tasks (not historical completed tasks from old phases); DAG panel shows current plan only; completed tasks shown dimmed at bottom or hidden; running tasks prominent with elapsed time; ready tasks clearly marked; blocked tasks show which dependency they're waiting on; tests verify filtering
**Checklist:**
- DAG panel: only show tasks from current work (not completed old phases)
- Running tasks: prominent, show elapsed time, agent info
- Ready tasks: clearly marked as dispatchable
- Blocked tasks: show which dependency blocks them
- Recently completed (last hour): shown dimmed
- Old completed: hidden (available via corc task list)
- Uses queries.py data layer, not direct SQLite
- Tests verify filtering logic

**Context bundle:** src/corc/tui.py, src/corc/queries.py

---

## 3C. TUI: running task streaming detail

**Type:** implementation
**Priority:** 35
**Depends on:** 3B
**Done when:** Bottom panel shows live streaming output from running agents: tool calls (which files being read/written), assistant reasoning, checklist progress; scrollable; updates in real time from session log; tests verify streaming data appears in panel
**Checklist:**
- Bottom panel reads from session log (stream_event entries)
- Show tool_use events: tool name + file path
- Show assistant messages: truncated reasoning
- Show checklist progress updates
- Scrollable within panel
- Auto-updates as new events arrive
- Uses queries.py get_task_stream_events()
- Tests verify streaming data rendering

**Context bundle:** src/corc/tui.py, src/corc/sessions.py, src/corc/queries.py

---

## 3D. Verify and fix corc plan end-to-end

**Type:** investigation
**Priority:** 30
**Depends on:** none
**Done when:** Root cause of any corc plan issues identified; corc plan launches interactive Claude session with correct system prompt and context; operator can create tasks from within the session; session crashes recover via --resume; findings documented
**Checklist:**
- Run `corc plan` manually and document what happens
- Verify system prompt includes: planner role, spec template, knowledge store summary, work state summary
- Verify tasks can be created from within the session
- Test --resume after simulated crash
- Document any issues found as follow-up tasks
