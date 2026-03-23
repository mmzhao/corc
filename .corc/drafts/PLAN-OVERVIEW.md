# Task Plan Overview

## Execution Order (respecting dependencies)

### Wave 1 (no dependencies — dispatch immediately)
- **1A** Fix mutation log seq race condition (P10, bugfix)
- **2B** Context bundle path validation (P25, bugfix)
- **3D** Verify corc plan end-to-end (P30, investigation)
- **4A** Centralized config system (P40, implementation)
- **5A** Session log daily rotation (P50, implementation)

### Wave 2 (depends on Wave 1)
- **1B** Fix task_updated depends_on handling (P10, bugfix) → depends on 1A
- **4B** Multi-repo registration (P40, implementation) → depends on 4A

### Wave 3 (depends on Wave 2)
- **1C** Add task type field (P15, implementation) → depends on 1B
- **1D** Add draft status (P15, implementation) → depends on 1B
- **4C** Auto-generate enforcement hooks from config (P35, implementation) → depends on 4B

### Wave 4 (depends on Wave 3)
- **2A** Machine-parseable blacklist with hooks (P20, implementation) → depends on 4C
- **4D** PR-based workflow (P35, implementation) → depends on 4C
- **3A** Task query API (P30, implementation) → depends on 1C, 1D

### Wave 5 (depends on Wave 4)
- **4E** Register corc + fdp repos (P45, implementation) → depends on 4D
- **3B** TUI: active-plan focus (P30, implementation) → depends on 3A

### Wave 6 (depends on Wave 5)
- **3C** TUI: streaming detail (P35, implementation) → depends on 3B
- **5B** Planning process design (P50, investigation) → depends on 3A

## DAG Visualization

```
1A ──► 1B ──► 1C ──┐
              │     ├──► 3A ──► 3B ──► 3C
              └► 1D ┘     │
                          └──► 5B
4A ──► 4B ──► 4C ──► 4D ──► 4E
              │
              └──► 2A

2B (independent)
3D (independent)
5A (independent)
```

## Summary
- 16 tasks total
- 6 waves of execution
- 5 tasks can start immediately (Wave 1)
- Critical path: 4A → 4B → 4C → 4D → 4E (longest chain, 5 tasks)
- Bug fixes (1A, 1B) are highest priority and unblock everything else

## Task Types
- 5 bugfix tasks (1A, 1B, 2B + implied fixes)
- 2 investigation tasks (3D, 5B)
- 9 implementation tasks (everything else)
