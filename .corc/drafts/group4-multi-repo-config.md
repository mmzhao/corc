# Group 4: Multi-Repo & Config

Priority: 40 (but scheduled first — foundation for enforcement)

## 4A. Centralized config system

**Type:** implementation
**Priority:** 40
**Depends on:** none
**Done when:** `.corc/config.yaml` is the single source for all settings; `corc config show` displays current config; `corc config set KEY VALUE` updates config; all existing modules read from config instead of hardcoded values; tests verify config loading, defaults, and overrides
**Checklist:**
- Define config schema: repos, notifications, alerts, trust_levels, dispatch provider
- `corc config show` / `corc config set`
- All modules read from config (notification channels, alert thresholds, dispatch provider, parallel limit)
- Defaults for everything (system works with no config file)
- Tests for loading, defaults, override

**Context bundle:** SPEC.md#notification-system, src/corc/config.py, src/corc/cli.py

---

## 4B. Multi-repo registration with per-repo settings

**Type:** implementation
**Priority:** 40
**Depends on:** 4A
**Done when:** `corc repo add NAME --path PATH` registers repos in config; each repo has settings for merge_policy (auto|human-only), protected_branches, enforcement_level (strict|relaxed); `corc repo list` shows all repos with settings; `corc repo show NAME` shows full config; tests verify CRUD and settings persistence
**Checklist:**
- Repo settings in .corc/config.yaml under `repos:` key
- Per-repo fields: path, remote, primary_branch, language, project, merge_policy, protected_branches, enforcement_level
- `corc repo add/remove/list/show` CLI
- Validate repo path exists on add
- Tests for all CRUD operations

**Context bundle:** SPEC.md#repo-merge-policies, src/corc/cli.py, .corc/config.yaml

---

## 4C. Auto-generate enforcement hooks from repo config

**Type:** implementation
**Priority:** 35 (higher — safety critical)
**Depends on:** 4B
**Done when:** Running `corc repo add` or `corc repo update` automatically generates `.claude/settings.json` hooks matching the repo's enforcement level; strict repos get PreToolUse hooks blocking: git push to protected branches, gh pr merge --auto; relaxed repos get only formatting hooks; hook generation is deterministic (same config = same hooks); tests verify hook generation for both strict and relaxed configs
**Checklist:**
- Hook generator reads repo config and produces .claude/settings.json
- Strict: PreToolUse[Bash] blocks `git push` to protected branches, blocks `gh pr merge --auto`
- Relaxed: PostToolUse[Write|Edit] formatting only
- Hook generation runs automatically on `corc repo add` and `corc repo update`
- Hook generation is idempotent (running twice produces same result)
- Tests verify generated hooks for strict and relaxed configs

**Context bundle:** SPEC.md#repo-merge-policies, SPEC.md#hook-enforcement-map, .claude/settings.json

---

## 4D. PR-based workflow (always branch from main, never push to main)

**Type:** implementation
**Priority:** 35
**Depends on:** 4C
**Done when:** Executor always creates worktree from latest main (git pull before worktree creation); agents create PRs via `gh pr create` on task completion; processor posts review summary as PR comment before merging; auto-merge repos: processor merges PR after review comment; human-only repos: PR left open for human merge; no code ever pushed directly to main; tests verify PR creation and review comment flow
**Checklist:**
- Executor: `git pull` on main before creating worktree
- Agent completion: `gh pr create` with task summary as PR body
- Processor: post review/validation summary as PR comment via `gh pr comment`
- Auto-merge repos: `gh pr merge` after review comment
- Human-only repos: leave PR open, notify operator
- PreToolUse hook blocks `git push origin main` for all repos
- Tests for PR creation, review comment, auto-merge, and human-only paths

**Context bundle:** SPEC.md#repo-merge-policies, src/corc/executor.py, src/corc/processor.py

---

## 4E. Register corc and fdp repos

**Type:** implementation
**Priority:** 45
**Depends on:** 4D
**Done when:** Both repos registered in .corc/config.yaml with correct settings; corc: auto-merge, relaxed enforcement; fdp: human-only merge, strict enforcement, protected branches [main]; enforcement hooks generated for both; `corc repo list` shows both repos with settings
**Checklist:**
- Register corc repo: auto-merge, relaxed
- Register fdp repo: human-only, strict, protected_branches=[main]
- Verify enforcement hooks generated correctly for both
- Test by checking .claude/settings.json matches expected hooks
