# Group 1: Data Integrity Fixes

Priority: 10 (critical — everything depends on correct data)

## 1A. Fix mutation log seq race condition

**Type:** bugfix
**Priority:** 10
**Depends on:** none
**Done when:** flock covers both `_next_seq()` read and `append()` write as one atomic operation; concurrent writers from separate processes produce strictly sequential seq numbers with zero duplicates; test spawns 10 concurrent writers doing 10 appends each and verifies all 100 entries have unique sequential seq numbers
**Checklist:**
- Move flock acquisition to before _next_seq() read
- Hold lock through the write
- Release after write completes
- Test: 10 concurrent processes, 10 appends each, verify 100 unique seqs
- Verify existing mutation log has no new duplicates after fix

**Context bundle:** src/corc/mutations.py, tests/test_mutations.py

---

## 1B. Fix task_updated mutation not handling depends_on

**Type:** bugfix
**Priority:** 10
**Depends on:** 1A (clean mutation log needed first)
**Done when:** `_apply_mutation` for `task_updated` type handles all task fields including depends_on, priority, checklist, context_bundle, done_when, description, role, status; test creates a task, updates its depends_on via mutation, and verifies SQLite reflects the change
**Checklist:**
- Add all missing fields to task_updated handler's updatable field list
- JSON-encode list/dict fields before storing
- Test: update depends_on, priority, checklist, context_bundle via task_updated mutation
- Test: verify SQLite matches after each update

**Context bundle:** src/corc/state.py, tests/test_state.py

---

## 1C. Add task type field (investigation vs implementation vs bugfix)

**Type:** implementation
**Priority:** 15
**Depends on:** 1B (schema change, want clean state handling first)
**Done when:** Tasks have a `task_type` field: implementation (default), investigation, bugfix; `corc task create --type investigation` sets the type; done-when linter applies type-specific rules (investigation: must mention "root cause" or "documented"; bugfix: must mention "regression test" or "reproduced"); `corc task list` shows type; tests verify type-specific linting
**Checklist:**
- Add task_type column to tasks table (default: implementation)
- Schema migration for existing tasks (default to implementation)
- `corc task create --type TYPE`
- Done-when linter rules per type:
  - investigation: must reference documentation or root cause identification
  - bugfix: must reference regression test or reproduction
  - implementation: existing rules
- `corc task list` and `corc task status` show task type
- Tests for each type's linting rules
- Backwards compatible with existing tasks

**Context bundle:** src/corc/state.py, src/corc/lint_done_when.py, src/corc/cli.py

---

## 1D. Add draft status for tasks

**Type:** implementation
**Priority:** 15
**Depends on:** 1B
**Done when:** `corc task create --draft` creates tasks with status `draft`; draft tasks are invisible to the scheduler (never dispatched); `corc task approve TASK_ID` flips status from draft to pending; `corc task approve --all` approves all drafts; `corc task list --draft` shows only drafts; tests verify draft tasks are not dispatched and approval flow works
**Checklist:**
- draft status: scheduler ignores it
- `corc task create --draft`
- `corc task approve TASK_ID` (draft → pending)
- `corc task approve --all`
- `corc task list --draft`
- Tests: draft not dispatched, approval changes status, --all approves batch
