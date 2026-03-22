# Agent Blacklist

## Process
- Never create new tasks via `corc task create`. You are scoped to your assigned task only. If you discover additional work needed, report it in your findings. (Reason: agents created unauthorized tasks that consumed resources)
- Never merge directly to main. Always create a PR. (Reason: review gate)

## Code Patterns
- Never use `eval()` or `exec()`. (Reason: security)
- Never catch bare `except:`. Always catch specific exceptions. (Reason: masks bugs)
