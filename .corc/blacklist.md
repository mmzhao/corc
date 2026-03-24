# Agent Blacklist

## Core Principle
- **Fail early and loudly.** Never create workarounds, fallbacks, or backdoors that mask failure as success. If something fails, the task should fail. Silent degradation is worse than a visible error. This applies to all work — CORC infrastructure AND work in other repos. (Reason: operator lost visibility into PR failures because the system silently fell back to direct merge)

## Process
- Never create new tasks via `corc task create`. You are scoped to your assigned task only. If you discover additional work needed, report it in your findings. (Reason: agents created unauthorized tasks that consumed resources)
- Never merge directly to main. Always create a PR. If PR creation fails, the task fails. (Reason: review gate, no silent fallbacks)

## Code Patterns
- Never use `eval()` or `exec()`. (Reason: security)
- Never catch bare `except:`. Always catch specific exceptions. (Reason: masks bugs)
- Never swallow errors silently. If a function fails, propagate the error or log it visibly. Return None to signal failure, don't return a default that looks like success. (Reason: fail-early principle)
