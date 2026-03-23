"""Blacklist parsing and enforcement hook generation.

Parses `.corc/blacklist.md` to extract two kinds of entries:

1. **Advisory entries** (plain text): Injected into agent context as prompt
   guidance. Agents see them but they are not machine-enforced.

2. **Block-command entries** (`block_command:` prefix): Both injected into
   context AND auto-generate PreToolUse hooks that block matching Bash
   commands at the tool-use layer.

Block-command entry format in blacklist.md:
    - block_command: <pattern> (Reason: <reason>)

Example:
    - block_command: rm -rf / (Reason: catastrophic deletion)
    - block_command: git push --force (Reason: force push destroys history)

Hook generation is deterministic: the same blacklist content always produces
byte-identical hook scripts and settings entries.

Usage:
    from corc.blacklist import (
        parse_blacklist,
        generate_blacklist_hook_script,
        generate_blacklist_hook_settings,
        add_entry,
        remove_entry,
    )
"""

from __future__ import annotations

import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class BlacklistEntry:
    """A single parsed blacklist entry."""

    __slots__ = ("raw", "is_block_command", "pattern", "reason")

    def __init__(
        self,
        raw: str,
        is_block_command: bool = False,
        pattern: str = "",
        reason: str = "",
    ):
        self.raw = raw
        self.is_block_command = is_block_command
        self.pattern = pattern
        self.reason = reason

    def __repr__(self) -> str:
        if self.is_block_command:
            return f"BlacklistEntry(block_command={self.pattern!r}, reason={self.reason!r})"
        return f"BlacklistEntry(advisory={self.raw!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BlacklistEntry):
            return NotImplemented
        return (
            self.raw == other.raw
            and self.is_block_command == other.is_block_command
            and self.pattern == other.pattern
            and self.reason == other.reason
        )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Regex for extracting (Reason: ...) suffix from an entry line
_REASON_RE = re.compile(r"\s*\(Reason:\s*(.+?)\)\s*$")

# Prefix that marks a block-command entry
_BLOCK_PREFIX = "block_command:"


def parse_blacklist(content: str) -> list[BlacklistEntry]:
    """Parse blacklist markdown content into structured entries.

    Extracts list items (lines starting with ``- ``) and classifies them
    as either block-command entries or advisory entries.

    Args:
        content: Raw markdown content of blacklist.md.

    Returns:
        List of BlacklistEntry objects, preserving file order.
    """
    entries: list[BlacklistEntry] = []

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue

        item_text = stripped[2:].strip()
        if not item_text:
            continue

        # Extract optional (Reason: ...) suffix
        reason = ""
        reason_match = _REASON_RE.search(item_text)
        if reason_match:
            reason = reason_match.group(1).strip()

        # Check for block_command: prefix
        if item_text.lower().startswith(_BLOCK_PREFIX):
            # Extract the command pattern (after prefix, before reason)
            after_prefix = item_text[len(_BLOCK_PREFIX) :].strip()
            pattern = after_prefix
            if reason_match:
                pattern = _REASON_RE.sub("", after_prefix).strip()

            entries.append(
                BlacklistEntry(
                    raw=item_text,
                    is_block_command=True,
                    pattern=pattern,
                    reason=reason,
                )
            )
        else:
            entries.append(
                BlacklistEntry(
                    raw=item_text,
                    is_block_command=False,
                    reason=reason,
                )
            )

    return entries


def load_blacklist(project_root: Path) -> list[BlacklistEntry]:
    """Load and parse the blacklist from .corc/blacklist.md.

    Returns an empty list if the file doesn't exist or can't be read.
    """
    blacklist_path = project_root / ".corc" / "blacklist.md"
    try:
        if blacklist_path.exists():
            content = blacklist_path.read_text()
            return parse_blacklist(content)
    except OSError:
        pass
    return []


def get_block_commands(entries: list[BlacklistEntry]) -> list[BlacklistEntry]:
    """Filter to only block-command entries, sorted by pattern for determinism."""
    blocked = [e for e in entries if e.is_block_command]
    blocked.sort(key=lambda e: e.pattern)
    return blocked


def get_advisory_entries(entries: list[BlacklistEntry]) -> list[BlacklistEntry]:
    """Filter to only advisory (non-block-command) entries."""
    return [e for e in entries if not e.is_block_command]


# ---------------------------------------------------------------------------
# Hook generation
# ---------------------------------------------------------------------------

_BLACKLIST_HOOK_TEMPLATE = r"""#!/usr/bin/env bash
# PreToolUse hook: enforce blacklist block_command entries.
# Blocks Bash commands matching blacklisted patterns.
# Generated by corc — do not edit manually.
# Regenerate with: corc blacklist sync-hooks
set -euo pipefail

INPUT=$(cat)
TOOL_NAME=$(printf '%s' "$INPUT" | jq -r '.tool_name // empty')

# Only check Bash commands
if [ "$TOOL_NAME" != "Bash" ]; then
  exit 0
fi

COMMAND=$(printf '%s' "$INPUT" | jq -r '.tool_input.command // empty')

if [ -z "$COMMAND" ]; then
  exit 0
fi

__PATTERN_CHECKS__
exit 0
"""


def _escape_for_grep(pattern: str) -> str:
    """Escape a pattern for use in grep -F (fixed string match).

    Since we use grep -qF (fixed string), no regex escaping is needed.
    We just need to ensure the pattern is safe for shell single-quoting.
    """
    # Escape single quotes for shell embedding
    return pattern.replace("'", "'\\''")


def generate_blacklist_hook_script(entries: list[BlacklistEntry]) -> str | None:
    """Generate a PreToolUse hook script that blocks blacklisted commands.

    Args:
        entries: All blacklist entries (advisory ones are ignored).

    Returns:
        Shell script content, or None if there are no block-command entries.
    """
    blocked = get_block_commands(entries)
    if not blocked:
        return None

    checks = []
    for entry in blocked:
        escaped = _escape_for_grep(entry.pattern)
        reason = entry.reason or "blocked by blacklist"
        # Use grep -qF for fixed-string matching (no regex interpretation)
        checks.append(
            f"if printf '%s' \"$COMMAND\" | grep -qF '{escaped}'; then\n"
            f'  echo "BLOCKED: {reason}"\n'
            f"  exit 2\n"
            f"fi"
        )

    checks_str = "\n\n".join(checks)
    return _BLACKLIST_HOOK_TEMPLATE.replace("__PATTERN_CHECKS__", checks_str)


def generate_blacklist_hook_settings(
    entries: list[BlacklistEntry],
) -> list[dict] | None:
    """Generate PreToolUse settings entries for blacklist enforcement.

    Returns a list of hook config dicts suitable for .claude/settings.json
    PreToolUse array, or None if there are no block-command entries.
    """
    blocked = get_block_commands(entries)
    if not blocked:
        return None

    return [
        {
            "matcher": "Bash",
            "hooks": [
                {
                    "type": "command",
                    "command": ".claude/hooks/enforce-blacklist.sh",
                }
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Sync hooks to disk
# ---------------------------------------------------------------------------


def sync_blacklist_hooks(project_root: Path) -> list[str]:
    """Write blacklist-derived hooks to the project's .claude/ directory.

    Reads .corc/blacklist.md, parses block_command entries, and generates:
    - .claude/hooks/enforce-blacklist.sh (if block_command entries exist)
    - Updates .claude/settings.json to include the blacklist hook

    If no block_command entries exist, removes the hook script and its
    settings entry (cleanup).

    This function is idempotent and deterministic.

    Args:
        project_root: Path to the project root.

    Returns:
        List of file paths that were written or modified.
    """
    entries = load_blacklist(project_root)
    blocked = get_block_commands(entries)

    claude_dir = project_root / ".claude"
    hooks_dir = claude_dir / "hooks"
    hook_path = hooks_dir / "enforce-blacklist.sh"
    settings_path = claude_dir / "settings.json"

    written: list[str] = []

    if blocked:
        # Ensure directories exist
        claude_dir.mkdir(parents=True, exist_ok=True)
        hooks_dir.mkdir(parents=True, exist_ok=True)

        # Write the hook script
        script = generate_blacklist_hook_script(entries)
        if script is not None:
            hook_path.write_text(script)
            hook_path.chmod(0o755)
            written.append(str(hook_path))

        # Update settings.json to include the blacklist hook
        _update_settings_with_blacklist(settings_path, add=True)
        written.append(str(settings_path))
    else:
        # No block_command entries — clean up if hook exists
        if hook_path.exists():
            hook_path.unlink()
            written.append(str(hook_path))

        if settings_path.exists():
            _update_settings_with_blacklist(settings_path, add=False)
            written.append(str(settings_path))

    return written


def _update_settings_with_blacklist(settings_path: Path, *, add: bool) -> None:
    """Add or remove the blacklist hook from .claude/settings.json.

    Preserves existing hooks (e.g., enforce-policy.sh, format-python.sh).
    """
    import json

    settings: dict = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except (json.JSONDecodeError, OSError):
            settings = {}

    hooks = settings.setdefault("hooks", {})
    pre_hooks: list = hooks.get("PreToolUse", [])

    # Remove any existing blacklist hook entries
    pre_hooks = [h for h in pre_hooks if not _is_blacklist_hook_entry(h)]

    if add:
        # Add the blacklist hook entry
        pre_hooks.append(
            {
                "matcher": "Bash",
                "hooks": [
                    {
                        "type": "command",
                        "command": ".claude/hooks/enforce-blacklist.sh",
                    }
                ],
            }
        )

    if pre_hooks:
        hooks["PreToolUse"] = pre_hooks
    elif "PreToolUse" in hooks:
        del hooks["PreToolUse"]

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")


def _is_blacklist_hook_entry(entry: dict) -> bool:
    """Check if a PreToolUse hook entry is our blacklist hook."""
    if not isinstance(entry, dict):
        return False
    hook_list = entry.get("hooks", [])
    for hook in hook_list:
        if (
            isinstance(hook, dict)
            and hook.get("command") == ".claude/hooks/enforce-blacklist.sh"
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Add / Remove entries
# ---------------------------------------------------------------------------


def add_entry(
    project_root: Path,
    entry_text: str,
    *,
    reason: str = "",
    section: str | None = None,
) -> str:
    """Add an entry to .corc/blacklist.md.

    Args:
        project_root: Path to the project root.
        entry_text: The entry text. If it starts with "block_command:", it
            becomes an enforced entry; otherwise advisory.
        reason: Optional reason string (appended as "(Reason: ...)")
        section: Optional section heading to add under. If None, appends
            at the end of the file.

    Returns:
        The formatted line that was added.
    """
    blacklist_path = project_root / ".corc" / "blacklist.md"
    blacklist_path.parent.mkdir(parents=True, exist_ok=True)

    # Build the formatted line
    line = entry_text.strip()
    if reason:
        line = f"{line} (Reason: {reason})"
    formatted = f"- {line}"

    # Read existing content
    content = ""
    if blacklist_path.exists():
        content = blacklist_path.read_text()

    # Check for duplicate (exact raw text match)
    for existing_line in content.splitlines():
        if existing_line.strip() == formatted:
            return formatted  # Already exists, no-op

    if section and content:
        # Try to find the section and insert at the end of it
        lines = content.splitlines()
        insert_idx = _find_section_end(lines, section)
        if insert_idx is not None:
            lines.insert(insert_idx, formatted)
            content = "\n".join(lines) + "\n"
        else:
            # Section not found — append at end
            if not content.endswith("\n"):
                content += "\n"
            content += formatted + "\n"
    else:
        # Append at end
        if content and not content.endswith("\n"):
            content += "\n"
        content += formatted + "\n"

    blacklist_path.write_text(content)
    return formatted


def remove_entry(project_root: Path, entry_text: str) -> bool:
    """Remove an entry from .corc/blacklist.md.

    Matches entries by checking if the line (after "- ") contains the
    given text as a substring. Removes the first matching line.

    Args:
        project_root: Path to the project root.
        entry_text: Text to match against entry lines.

    Returns:
        True if an entry was removed, False if no match found.
    """
    blacklist_path = project_root / ".corc" / "blacklist.md"

    if not blacklist_path.exists():
        return False

    content = blacklist_path.read_text()
    lines = content.splitlines()
    new_lines = []
    removed = False

    for line in lines:
        stripped = line.strip()
        if not removed and stripped.startswith("- "):
            item_text = stripped[2:].strip()
            if entry_text.strip() in item_text:
                removed = True
                continue
        new_lines.append(line)

    if removed:
        blacklist_path.write_text("\n".join(new_lines) + "\n")

    return removed


def _find_section_end(lines: list[str], section_name: str) -> int | None:
    """Find the line index where content should be inserted at end of a section.

    Returns the index of the line just before the next heading of same or
    higher level, or the end of file if no subsequent heading.
    Returns None if the section is not found.
    """
    section_slug = section_name.lower().replace(" ", "-").replace(":", "")
    in_section = False
    section_level = 0
    last_content_idx = None

    for i, line in enumerate(lines):
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            heading_text = line.lstrip("#").strip()
            slug = (
                heading_text.lower()
                .replace(" ", "-")
                .replace(":", "")
                .replace("(", "")
                .replace(")", "")
            )

            if not in_section and section_slug in slug:
                in_section = True
                section_level = level
                last_content_idx = i + 1
                continue

            if in_section and level <= section_level:
                # Found next section — insert before it
                return last_content_idx if last_content_idx is not None else i

        if in_section:
            last_content_idx = i + 1

    if in_section and last_content_idx is not None:
        return last_content_idx

    return None
