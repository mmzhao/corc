---
title: CommissionsExpense Reference Scan — v2
task: 430f11b7
scope: workspaces/backend/src/
search_terms:
  - CommissionsExpense
  - commissions_expense
  - getCommissionsExpense
date: 2026-03-24
---

# CommissionsExpense Reference Scan — Findings

## Summary

Searched `workspaces/backend/src/` for all references to `CommissionsExpense`, `commissions_expense`, and `getCommissionsExpense()`. Found exactly **10 files** containing matches. All 10 correspond to the spec-provided file list. No new (unexpected) files were discovered. Commission event files, handler files, and module files were explicitly verified to contain **zero** references to the account — they interact through subledger pipe abstractions only.

## Key Findings

- **10 files match** across the codebase — enum definitions, chart-of-accounts, subledger blueprint, action contracts, test commands, smoke tests, DB schema, and a SQL report
- **0 files are "New"** — no unexpected files reference the account
- **0 omissions** — rg output matches this findings list exactly
- Commission event files (`accrue.ts`, `reverse-accrued.ts`, `reclass-to-deferred.ts`, `amortize.ts`) use pipe IDs from the subledger record (e.g., `subledger.accrue_for_commissions_pipe_id`) rather than directly referencing the CommissionsExpense account
- `CommissionsHandler.ts` and `CommissionsHandler.integration.test.ts` do not reference CommissionsExpense — they operate through `Commission` and `DeferredCommissionsSubledger` abstractions
- `module/initialize.ts` and `module/start.ts` do not reference CommissionsExpense — they create/start the module via `DeferredCommissionsSubledger`

## File-by-File Findings

### Confirmed — In spec, change needed (10 files)

| # | File | References | Notes |
|---|------|-----------|-------|
| 1 | `accounting/account-class.ts` | `CommissionsExpense = 'commissions_expense'` | Enum definition — source of the subclass value |
| 2 | `accounting/chart-of-accounts/chart-of-accounts.ts` | `AccountSubclass.CommissionsExpense` (x3), `getCommissionsExpense()` (definition) | Default account list, getter method, `createDefaultAccounts()` |
| 3 | `accounting/subledgers/blueprints/deferred-commissions.ts` | `coa.getCommissionsExpense()` (x5) | Account key `commissionsExpense`, pipe `from`/`to` for Accrue, ReverseAccrue, Reclass, Amortize |
| 4 | `actions/commands/impl/load-coa.ts` | `coa.getCommissionsExpense().nid` | Exposes account ID in `loadcoa` command output |
| 5 | `actions/commands/impl/commissions/test-commissions.ts` | `coa.data.getCommissionsExpense()`, `args.total_commissions_expense` (x2) | Test assertion on CommissionsExpense account balance |
| 6 | `db/schema/accounting/nid-entities.ts` | `AccountSubclass.CommissionsExpense` | DB enum definition for `account_subclass` column |
| 7 | `codegen/actions/actions__GENERATED.ts` | `commissions_expense: ak.string`, `total_commissions_expense: Arktypes.Decimal.optional()` | Auto-generated action contracts (loadcoa output, testcommissions args) |
| 8 | `smoke-test/types.ts` | `CommissionsExpense = 'commissions_expense'` | `AccountRef` enum for smoke test assertions |
| 9 | `smoke-test/phases/assert.ts` | `AccountRef.CommissionsExpense`, `coa.getCommissionsExpense().nid` | Account ref resolution in `resolveAccountRef()` |
| 10 | `data-lake/dev/config/sql-reports/income-statement.sql` | `'commissions_expense'` | SQL WHEN clause grouping subclass into 'Compensation & Benefits' |

### New — Not in spec, change needed (0 files)

None found.

### No-change-needed — Verified no account reference (8 files checked)

These files were explicitly checked per the task checklist. They reference the commissions *concept* (subledger, pipes, types) but do NOT reference the `CommissionsExpense` account, `commissions_expense` subclass, or `getCommissionsExpense()` getter.

| # | File | Why no change needed |
|---|------|---------------------|
| 1 | `accounting/assets/commissions/events/accrue.ts` | Uses `subledger.accrue_for_commissions_pipe_id` — no account ref |
| 2 | `accounting/assets/commissions/events/reverse-accrued.ts` | Uses `subledger.reverse_accrued_commissions_pipe_id` — no account ref |
| 3 | `accounting/assets/commissions/events/reclass-to-deferred.ts` | Uses `subledger.reclass_deferred_commissions_pipe_id` — no account ref |
| 4 | `accounting/assets/commissions/events/amortize.ts` | Uses `subledger.amortize_commissions_pipe_id` — no account ref |
| 5 | `accounting/assets/commissions/CommissionsHandler.ts` | Operates through `Commission` abstraction — no account ref |
| 6 | `accounting/assets/commissions/CommissionsHandler.integration.test.ts` | Tests through handler/subledger — no account assertions on CommissionsExpense |
| 7 | `accounting/assets/commissions/module/initialize.ts` | Creates subledger via `DeferredCommissionsSubledger.createNew()` — no account ref |
| 8 | `accounting/assets/commissions/module/start.ts` | Starts background jobs — no account ref |

## Reference Details by Search Term

### `CommissionsExpense` (PascalCase — enum/type references)
- `account-class.ts:45` — enum definition
- `chart-of-accounts.ts:37` — default account subclasses array
- `chart-of-accounts.ts:162-163` — getter method definition
- `chart-of-accounts.ts:612,613,618` — `createDefaultAccounts()` logic
- `nid-entities.ts:72` — DB schema enum
- `smoke-test/types.ts:120` — AccountRef enum
- `smoke-test/phases/assert.ts:118` — switch case

### `commissions_expense` (snake_case — DB/field/SQL references)
- `account-class.ts:45` — enum string value
- `codegen/actions/actions__GENERATED.ts:457` — loadcoa output field
- `codegen/actions/actions__GENERATED.ts:536` — testcommissions args field
- `data-lake/dev/config/sql-reports/income-statement.sql:32` — SQL subclass filter
- `smoke-test/types.ts:120` — AccountRef string value

### `getCommissionsExpense()` (method call references)
- `chart-of-accounts.ts:162` — method definition
- `deferred-commissions.ts:64,74,81,89,98` — subledger blueprint pipe configuration (5 calls)
- `load-coa.ts:49` — loadcoa command output
- `test-commissions.ts:50` — test assertion
- `smoke-test/phases/assert.ts:119` — account ref resolution

## Risks and Unknowns

1. **Generated file dependency**: `codegen/actions/actions__GENERATED.ts` is auto-generated. Changes to `commissions_expense` field names require updating the codegen source (likely a TOML or schema file outside `src/`), not the generated file directly.
2. **SQL report**: `income-statement.sql` uses the raw string `'commissions_expense'` — this must be updated if the DB enum value changes, and there is no compile-time check to catch a mismatch.
3. **Database migration**: `nid-entities.ts` defines the Drizzle schema enum. Changing the enum value requires a DB migration to update existing rows and the enum type.
4. **Subledger blueprint is the key integration point**: `deferred-commissions.ts` calls `getCommissionsExpense()` 5 times to wire the account into 4 different pipes. This is the most complex change site.
5. **No runtime references found outside these 10 files**: The event files and handler use pipe IDs from the subledger record, providing good encapsulation. Changes to the account name/subclass will NOT require changes to event logic.

## Recommendations

1. Start changes from the bottom up: `account-class.ts` enum -> `chart-of-accounts.ts` -> `nid-entities.ts` schema -> `deferred-commissions.ts` blueprint -> action contracts -> smoke tests
2. The `codegen/actions/actions__GENERATED.ts` file should NOT be edited directly — find and update the codegen source
3. Plan a DB migration for the `account_subclass` enum change in `nid-entities.ts`
4. The 8 no-change-needed files provide confidence that the account reference is well-encapsulated behind the subledger abstraction

## Verification

Reproduce the full file list with:
```bash
rg -l 'CommissionsExpense|commissions_expense|getCommissionsExpense' workspaces/backend/src/
```

Expected: exactly 10 files matching this findings list, zero omissions.
