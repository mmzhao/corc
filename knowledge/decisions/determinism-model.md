---
id: decision-determinism-model
type: decision
project: corc
repos: [corc]
tags: [determinism, architecture, hooks, dag, validation, control-flow]
created: 2026-03-21T00:00:00Z
updated: 2026-03-21T00:00:00Z
source: human
status: active
---

# Determinism Model: DAGs + Hooks + Validation

## Context

LLMs are non-deterministic by nature. When given the same prompt, they may produce different outputs, make different tool calls, and take different paths to a solution. This is a feature for fuzzy, generative work (writing code, interpreting errors) but a liability for control flow (deciding what to do next, whether something passed, which agent to dispatch).

Current AI coding tools let the LLM make all decisions — both the fuzzy creative work it's good at and the deterministic control flow it's bad at. This leads to inconsistent behavior, skipped steps, no auditability, and difficulty reproducing or debugging failures.

CORC needs a model that maximizes the LLM's strengths (fuzzy generative work) while enforcing determinism everywhere else.

## Decision

**Three-layer determinism architecture: Structural, Enforcement, and Verification.**

The core principle is: **the orchestrator is deterministic Python code, not an LLM.** Python makes all control flow decisions. The LLM only operates within explicitly scoped steps, doing the fuzzy generative work it's good at.

### Layer 1: Structural Determinism (Workflow DAGs)

All control flow is defined in DAGs. The workflow engine (deterministic Python code) resolves dependencies, determines execution order, and manages parallelism. The agent cannot influence step sequencing.

- Workflow DAGs define all control flow
- DAG engine resolves dependencies via topological sort
- Tasks with no unresolved dependencies are "ready"
- Ready tasks execute in parallel up to the configured limit
- The agent never decides what to do next — the DAG does

### Layer 2: Enforcement (Claude Code Hooks)

Hooks are deterministic guarantees — unlike CLAUDE.md instructions, hooks always execute. They enforce pre/post conditions that the agent cannot bypass.

- `PreToolUse[Write|Edit]` blocks writes outside the step's allowed file paths
- `PreToolUse[Bash]` blocks dangerous commands (rm -rf, force push, etc.)
- `PostToolUse[Write|Edit]` triggers incremental reindex for knowledge files
- `PostToolUse[*]` async-logs every tool call to the audit log
- `Stop` hook runs agent-based validation of "done when" criteria
- `SessionStart` hooks inject curated context bundles deterministically

### Layer 3: Verification ("Done When" Criteria)

Every task has machine-testable completion criteria. These are evaluated programmatically after each step — the agent's claim of completion is never trusted without verification.

- Validation rules: `file_exists`, `file_not_empty`, `tests_pass`, `contains_pattern`, `json_schema`
- "Done when" criteria are linted at plan time to reject subjective language
- Accepted: "All unit tests pass", "FTS5 queries return BM25-ranked results"
- Rejected: "Implementation works correctly", "Code is clean"

### Where Agent Flexibility Is Allowed (and Why It's Safe)

| Flexible Area | Why Allowed | Why Safe |
|---|---|---|
| Code generation | What LLMs are good at | Validated by tests, lint, adversarial review |
| Search queries | Agent knows what it needs | Bad queries → poor results → retry |
| Error interpretation | Novel errors need judgment | Feeds structured retry policy |
| Document content | Writing prose needs creativity | Templates enforce structure |
| Exploration | Investigating code, reading files | Read-only, bounded by timeout |

### What Is Never Flexible

- Step sequencing (DAG)
- Tool availability (`--allowedTools`)
- Output format (`--json-schema`)
- File write permissions (hooks)
- Cost limits (`--max-budget-usd`, `--max-turns`)
- Validation gates ("done when" criteria)
- Knowledge write access (agents report findings only)
- Context injection (deterministic assembly)

## Consequences

### What Becomes Easier
- Reproducibility: same task + same state = same control flow (though LLM output varies)
- Auditability: every decision point is logged, every gate is recorded
- Debugging: failures happen at known checkpoints with structured context
- Safety: agents cannot escape their sandbox, bypass validation, or corrupt state
- Trust calibration: guardrails can be loosened systematically as models improve

### What Becomes Harder
- Development overhead: every task needs "done when" criteria, context bundles, tool restrictions
- Rigidity: truly novel situations may not fit the DAG model (mitigated by escalation to operator)
- Over-engineering risk: too many constraints can slow agents down on simple tasks (mitigated by trust levels and quick-task path)

## Alternatives Considered

- **Full autonomy (let the LLM decide everything)**: Fast to set up but unreliable. No auditability, no reproducibility, can't debug failures. The "LLM as colleague" anti-pattern.
- **Advisory mode (suggest but don't enforce)**: Better than full autonomy but agents routinely ignore suggestions. CLAUDE.md instructions are not guarantees. Hooks provide actual enforcement.
- **Static analysis only (no LLM validation)**: Insufficient for fuzzy criteria. Some "done when" checks require judgment (e.g., "review posted as PR comment with actionable feedback"). Agent-based `Stop` hooks handle these cases.
- **Conversational orchestration (LLM orchestrator)**: Using an LLM to coordinate other LLMs. Doubles the non-determinism. The orchestrator must be deterministic code to provide the stability foundation.
