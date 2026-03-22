#!/usr/bin/env bash
# PostToolUse hook: auto-format Python files with ruff after Write/Edit.
# Receives JSON on stdin from Claude Code with tool_input.file_path.
# Skips non-Python files gracefully (exit 0).

set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

# If we couldn't extract a file path, nothing to do.
if [ -z "$FILE_PATH" ]; then
  exit 0
fi

# Only format Python files.
if [[ "$FILE_PATH" != *.py ]]; then
  exit 0
fi

# Only format if the file still exists (could have been deleted).
if [ ! -f "$FILE_PATH" ]; then
  exit 0
fi

# Run ruff format. If ruff is not installed, skip gracefully.
if command -v ruff &>/dev/null; then
  ruff format "$FILE_PATH" 2>/dev/null || true
else
  # Try via python -m ruff as fallback.
  python3 -m ruff format "$FILE_PATH" 2>/dev/null || true
fi

exit 0
