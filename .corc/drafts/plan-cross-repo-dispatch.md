# Cross-Repo Task Dispatch

## Problem
CORC dispatches all tasks into worktrees created from the corc repo root, regardless of which repo the task targets. Tasks targeting external repos (e.g., fdp) run in the wrong directory, can't find their context files, and produce no useful work. This is a blocker for any multi-repo orchestration.

The scout task for fdp (8de4581d) "completed" with proof_of_work explicitly stating it couldn't access the target files. The processor marked it done because the agent exited and the worktree merged cleanly (with no changes). The implementer task then auto-dispatched against the same broken worktree.

## Requirements
- [ ] Tasks have a `target_repo` field (string, repo name matching a key in `.corc/config.yaml` repos)
- [ ] `corc task create --target-repo NAME` sets the target repo; omitted defaults to the corc repo (or whichever repo corc is installed in)
- [ ] Executor resolves `target_repo` to the repo's filesystem path via `RepoManager`
- [ ] Worktrees are created in the target repo, not the corc repo (i.e., `target_repo_path/.corc/worktrees/`)
- [ ] Context assembly resolves context_bundle paths relative to the target repo root
- [ ] Context bundle path validation at task creation resolves paths against the target repo root
- [ ] Processor runs validations, posts PR comments, and merges PRs against the target repo
- [ ] `git pull` before worktree creation pulls the target repo's main branch
- [ ] Existing tasks with no `target_repo` field continue to work (backwards compatible, default to corc repo)

## Non-Requirements
- Cross-repo context bundles (files from repo A injected into a task targeting repo B) — use absolute paths for now
- Repo-specific role definitions (all roles are shared)
- Per-repo scheduling policies (all tasks scheduled equally)
- Worktree placement outside the target repo (keep worktrees co-located with their repo)

## Design

### Task schema change
Add `target_repo` (nullable string) to the task record in `state.py`. When null/missing, defaults to the project root (backwards compatible). When set, resolved via `RepoManager.get(name)["path"]`.

### Dispatch flow (changed lines marked)
```
Executor.dispatch(task)
  ├─ target_root = resolve_target_repo(task)     # NEW: resolve via RepoManager
  ├─ pull_main(target_root)                       # CHANGED: was project_root
  ├─ create_worktree(target_root, task_id, ...)   # CHANGED: was project_root
  ├─ assemble_context(task, target_root)           # CHANGED: was project_root
  └─ dispatcher.dispatch(..., cwd=worktree_path)   # unchanged (already parameterized)
```

### Files to change
1. **state.py** — Add `target_repo` column to tasks table; handle in `task_created` and `task_updated` mutations
2. **cli.py** — Add `--target-repo` flag to `corc task create`; validate repo name exists via `RepoManager`; resolve paths for context_bundle validation against target repo
3. **executor.py** — Add `resolve_target_repo(task)` helper that reads `target_repo` from task, looks up path via `RepoManager`, falls back to `self.project_root`; pass resolved path to worktree, context, PR, and processor calls
4. **worktree.py** — `create_worktree` already takes `project_root` as first arg; just pass target repo path instead (no signature change needed)
5. **context.py** — `assemble_context` already takes `project_root`; just pass target repo path instead (no signature change needed)
6. **processor.py** — Same pattern; pass target repo path to validation, PR comment, and merge calls

### Backwards compatibility
- `target_repo` defaults to None in schema
- `resolve_target_repo()` returns `self.project_root` when target_repo is None
- Existing tasks in the mutation log have no `target_repo` field; `_apply_mutation` stores NULL
- ALTER TABLE adds the column with no default (nullable)

## Testing Strategy
- Unit test: create task with `--target-repo fdp`, verify task record has `target_repo = "fdp"`
- Unit test: `resolve_target_repo` returns fdp path for tasks with `target_repo = "fdp"`, returns project_root for tasks with no target_repo
- Unit test: context_bundle validation resolves paths against target repo root
- Integration test: dispatch a task targeting a registered repo, verify worktree is created under that repo's path
- Backwards compat test: existing tasks (no target_repo) still dispatch normally

## Rationale
**Why repo name, not absolute path**: The repo path is already stored in `.corc/config.yaml` via `RepoManager`. Storing the name means the path can change without invalidating tasks. It also validates that the repo is registered.

**Why worktrees in the target repo**: Worktrees need to be in the same git repo they branch from. Creating a corc worktree and trying to work on fdp files doesn't work — they're separate git repos.

**Why no signature changes to worktree/context/processor**: These functions already accept `project_root` as a parameter. The executor just needs to pass the right value. This minimizes the diff and keeps the change focused.
