# Group 2: Enforcement & Safety

Priority: 20

## 2A. Machine-parseable blacklist with auto-generated hooks

**Type:** implementation
**Priority:** 20
**Depends on:** 4C (auto-generate hooks from config)
**Done when:** Blacklist entries support a `block_command:` prefix that auto-generates PreToolUse hooks; when blacklist is updated, hooks are regenerated deterministically; `corc blacklist add "block_command: corc task"` adds an entry and regenerates hooks; regular text entries remain advisory (prompt-only); tests verify hook generation from blacklist entries
**Checklist:**
- Blacklist format extended: plain text = advisory, `block_command: PATTERN` = enforced via hook
- Hook generator reads both repo config AND blacklist for enforcement rules
- `corc blacklist add` / `corc blacklist remove` CLI
- Hook regeneration on blacklist change (same auto-generate mechanism as repo config)
- Tests: add block_command entry, verify hook generated, verify command is blocked

**Context bundle:** .corc/blacklist.md, SPEC.md#curation-blacklist, src/corc/context.py

---

## 2B. Context bundle path validation at task creation

**Type:** bugfix
**Priority:** 25
**Depends on:** none
**Done when:** `corc task create` warns when context_bundle paths don't exist; `corc task create --strict` errors on missing paths and refuses to create the task; section references (file.md#section) validate the file exists (section validation is best-effort); tests verify warning and strict error paths
**Checklist:**
- Validate context_bundle paths on task creation
- Default: warn on missing paths, create task anyway
- --strict: error on missing paths, refuse to create
- Section references: validate file exists, warn if section not found
- Tests for warn path, strict error path, section references
