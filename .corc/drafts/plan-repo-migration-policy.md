# Per-Repo Migration Execution Policy

## Problem
CORC manages repos with different trust levels. For repos like `corc` itself (auto-merge, relaxed enforcement), it's fine for agents to run DB migrations directly. For repos like `fdp` (human-only, strict enforcement), migrations should never be auto-executed — the agent should produce the migration script as an artifact for operator review. Currently there's no per-repo config to control this; the behavior needs to be decided ad-hoc per task.

## Requirements
- [ ] Per-repo `migration_policy` setting in `.corc/config.yaml`: `auto` (agent executes) or `script-only` (agent produces script, operator executes)
- [ ] Default is `script-only` (safe default)
- [ ] `corc repo add` and `corc repo update` accept `--migration-policy` flag
- [ ] `corc repo show` displays migration policy
- [ ] Executor/context assembly injects migration policy into agent system prompt so agents know whether to execute or produce a script
- [ ] Validation rejects unknown policy values

## Non-Requirements
- Enforcing the policy via hooks (advisory for now — agent prompt instruction is sufficient)
- Migration framework integration (CORC doesn't know about drizzle, alembic, etc.)
- Multi-database support

## Design
Add `migration_policy` as a new field alongside `merge_policy`, `enforcement_level`, and `protected_branches` in the repo config. The field flows through the same path: stored in config.yaml, validated on add/update, displayed on show/list, and injected into agent context during dispatch.

Config example:
```yaml
repos:
  corc:
    path: /Users/michaelzhao/corc
    merge_policy: auto
    migration_policy: auto
    enforcement_level: relaxed
  fdp:
    path: /Users/michaelzhao/fdp
    merge_policy: human-only
    migration_policy: script-only
    enforcement_level: strict
```

## Testing Strategy
- Unit tests: add/update/show with migration_policy, validation of invalid values
- Existing tests continue to pass (backwards compatible — missing field defaults to `script-only`)
- Context assembly test verifies policy appears in agent prompt

## Rationale
This is a quick task — single config field addition following the exact pattern of existing repo settings. Keeping it as `script-only` default is the safe choice; auto-execution is the opt-in for trusted repos.
