# Replace Commissions Expense Account with Payroll Expense

## Problem
The deferred commissions module in fdp uses a dedicated `CommissionsExpense` account, but this is unnecessary — commissions are a type of payroll expense and should use the existing `PayrollExpense` account. The separate account adds complexity to the chart of accounts, the reporting SQL, and the codebase without providing value. Cleaning this up now prevents the divergence from growing as more commission features are built.

## Requirements
- [ ] All code references to `CommissionsExpense` / `commissions_expense` / `getCommissionsExpense()` are removed or replaced with `PayrollExpense` equivalents
- [ ] The `CommissionsExpense` enum value is removed from `AccountSubclass`, `AccountRef`, and the DB schema enum reference
- [ ] The `getCommissionsExpense()` getter and its auto-creation block are removed from `ChartOfAccounts`
- [ ] The deferred commissions blueprint uses `payrollExpense` as the account key and calls `getPayrollExpense()` for all 4 pipe definitions
- [ ] Generated code (`actions__GENERATED.ts`) is regenerated to reflect the removal
- [ ] Smoke tests updated to assert against payroll expense instead of commissions expense
- [ ] A ready-to-run SQL migration script is produced (operator executes manually) that per-tenant:
  - Updates `accounting.core_lines` to point to `payroll_expense` account
  - Updates `accounting.pipes` (`from_account_id` and `to_account_id`) to point to `payroll_expense` account
  - Deletes the `commissions_expense` row from `accounting.accounts`
- [ ] The `commissions_expense` value remains in the Postgres `account_subclass` enum (Postgres enums don't support DROP VALUE; it becomes unused)
- [ ] Income statement SQL report no longer references `commissions_expense` in its classification logic

## Non-Requirements
- Removing the `commissions_expense` value from the Postgres enum itself (can't DROP enum values in Postgres; it's harmless as unused)
- Changing any deferred commissions *business logic* (accrual, reversal, reclassification, amortization flows stay the same — only the account they target changes)
- Auto-executing the migration (fdp is a human-only repo; operator runs it manually after review)
- Renaming any pipes or tables — only the account references within them change

## Design

### Code changes (10 files in `workspaces/backend/src/`)

1. **`accounting/subledgers/blueprints/deferred-commissions.ts`** — Replace `commissionsExpense` key with `payrollExpense` in `deferredCommissionsAccountKeys`; replace all 4 `coa.getCommissionsExpense()` calls with `coa.getPayrollExpense()` in `deferredCommissionsSubledgerContents()`

2. **`accounting/account-class.ts`** — Remove `CommissionsExpense = 'commissions_expense'` from `AccountSubclass` enum

3. **`accounting/chart-of-accounts/chart-of-accounts.ts`** — Remove `AccountSubclass.CommissionsExpense` from `defaultAccountSubclasses` array; remove `getCommissionsExpense()` getter; remove `hasCommissionsExpense` auto-creation block in `createDefaultAccounts()`

4. **`db/schema/accounting/nid-entities.ts`** — Remove `AccountSubclass.CommissionsExpense` from the account subclass enum reference

5. **`actions/commands/impl/load-coa.ts`** — Remove the `commissions_expense: coa.getCommissionsExpense().nid` line

6. **`actions/commands/impl/commissions/test-commissions.ts`** — Change `coa.data.getCommissionsExpense()` to `coa.data.getPayrollExpense()`

7. **`smoke-test/types.ts`** — Remove `CommissionsExpense = 'commissions_expense'` from `AccountRef` enum

8. **`smoke-test/phases/assert.ts`** — Remove `case AccountRef.CommissionsExpense` or remap to `getPayrollExpense()`

9. **`data-lake/dev/config/sql-reports/income-statement.sql`** — Remove `'commissions_expense'` from the `WHEN a.subclass IN (...)` clause

10. **`codegen/actions/actions__GENERATED.ts`** — Regenerated via `pnpm predev` (not hand-edited)

### Additional files to verify (found via codebase search)

These files reference commissions as a *concept* but may also reference the account directly — the scout task confirms whether changes are needed:

- `accounting/assets/commissions/events/accrue.ts` — implements Accrue For Commissions pipe
- `accounting/assets/commissions/events/reverse-accrued.ts` — implements Reverse Accrued pipe
- `accounting/assets/commissions/events/reclass-to-deferred.ts` — implements Reclass pipe
- `accounting/assets/commissions/events/amortize.ts` — implements Amortize pipe
- `accounting/assets/commissions/CommissionsHandler.integration.test.ts` — integration test
- `accounting/assets/commissions/module/initialize.ts` — module setup

### Scout output format

The scout produces a structured findings artifact with three categories:

1. **Confirmed** — files from the spec where the reference is verified and the change pattern matches what the spec describes
2. **New** — files NOT in the spec that contain direct references to `CommissionsExpense` / `commissions_expense` / `getCommissionsExpense()` and need changes
3. **No change needed** — files that reference commissions as a concept but do NOT directly reference the account (e.g., they use pipes or subledger IDs, not account subclass/getter)

After the scout completes, the operator reviews the findings:
- If only **Confirmed** items → approve the implementer task as-is
- If **New** items exist → add those files to the implementer task's context bundle and checklist, then approve
- The scout does NOT make judgment calls about significance; it classifies references mechanically

### DB migration (operator-executed)

A standalone `.sql` script at `workspaces/backend/scripts/_migrate-commissions-to-payroll.sql` (underscore prefix, gitignored). The script:
1. For each tenant, dynamically resolves the `payroll_expense` and `commissions_expense` account nids
2. Updates `accounting.core_lines.account_id` where it matches commissions_expense nid
3. Updates `accounting.pipes.from_account_id` and `to_account_id` where they match
4. Deletes the `accounting.accounts` row with `subclass = 'commissions_expense'`

The operator reviews the script and executes it manually. fdp is configured with `migration_policy: script-only`.

### Order of operations
1. Scout verifies all references (findings feed into implementation)
2. Code changes (make the module use payroll expense)
3. Regenerate codegen (`pnpm predev`)
4. Smoke tests pass
5. Migration script written and reviewed by operator
6. Operator executes migration against target DB

## Testing Strategy
- **Type-check**: `pnpm tsc --noEmit` passes with no errors related to commissions expense
- **Codegen**: `pnpm predev` regenerates cleanly
- **Smoke tests**: All existing smoke test scenarios that referenced `CommissionsExpense` now pass against `PayrollExpense`
- **Grep verification**: `rg 'commissions_expense|CommissionsExpense|getCommissionsExpense' workspaces/backend/src/` returns zero results (excluding the migration file itself)
- **Migration review**: Operator reviews the SQL, runs against staging, verifies:
  - 193 core_lines updated
  - 5 pipes updated
  - 1 account row deleted
  - No orphaned references remain

## Rationale

**Why scout first**: The spec enumerates 10 files, but codebase search found 45+ files referencing commissions. Most are the commissions *module* (business logic that references commissions as a concept, not the account), but a few (event handlers, integration tests) could reference the account directly. A quick scout confirms the exact scope before implementation, preventing missed references that would cause type errors or runtime bugs.

**Why scout findings flow into implementation**: CORC's catch-up injection automatically includes findings from dependency tasks. The scout produces a structured findings artifact with three categories (Confirmed / New / No change needed). The operator reviews findings before approving the implementer task, adding any **New** files to its scope. This keeps the implementer's context accurate without requiring it to re-discover references.

**Why the operator review step between scout and implementation**: The scout classifies references mechanically (does this file reference the account directly — yes or no). The operator decides whether to update the implementation task's scope based on those findings. This is a human-in-the-loop checkpoint. It introduces non-determinism (the operator's judgment), which is a known trade-off we should watch — if we find operators making the same decision repeatedly, that's a signal to codify the pattern into a deterministic rule.

**Why migration is a separate script task**: fdp is configured as `human-only` merge policy. DB migrations are destructive and irreversible at the data level. The implementer produces the `.sql` file; the operator reviews and executes it. This matches how the team already works with fdp.

**Why codegen + smoke tests are completion criteria, not a separate task**: Regenerating codegen and running smoke tests are verification steps for the code changes, not independent deliverables. Including them in `done_when` keeps the task count lean and avoids artificial task boundaries.
