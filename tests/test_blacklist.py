"""Tests for blacklist parsing, hook generation, and CLI commands.

Verifies:
- parse_blacklist() correctly classifies entries (advisory vs block_command)
- block_command entries auto-generate PreToolUse hook scripts
- Hook generation is deterministic (same blacklist → same hooks)
- add_entry() and remove_entry() modify blacklist.md correctly
- sync_blacklist_hooks() writes/removes hook files and settings
- Advisory entries remain prompt-only (no hooks generated)
- CLI commands: corc blacklist add, remove, list, sync-hooks
"""

import json
import os
import stat
from pathlib import Path

import pytest

from corc.blacklist import (
    BlacklistEntry,
    add_entry,
    generate_blacklist_hook_script,
    generate_blacklist_hook_settings,
    get_advisory_entries,
    get_block_commands,
    load_blacklist,
    parse_blacklist,
    remove_entry,
    sync_blacklist_hooks,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal project directory with .corc/."""
    corc_dir = tmp_path / ".corc"
    corc_dir.mkdir()
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
def sample_blacklist():
    """Sample blacklist content with both advisory and block_command entries."""
    return (
        "# Agent Blacklist\n"
        "\n"
        "## Process\n"
        "- Never create new tasks via `corc task create`. (Reason: scope control)\n"
        "- block_command: git push --force (Reason: force push destroys history)\n"
        "\n"
        "## Code Patterns\n"
        "- Never use eval(). (Reason: security)\n"
        "- block_command: rm -rf / (Reason: catastrophic deletion)\n"
        "- block_command: curl | bash (Reason: arbitrary code execution)\n"
    )


@pytest.fixture
def advisory_only_blacklist():
    """Blacklist with only advisory entries (no block_command)."""
    return (
        "# Agent Blacklist\n"
        "\n"
        "## Process\n"
        "- Never create new tasks. (Reason: scope control)\n"
        "- Never merge directly to main. (Reason: review gate)\n"
        "\n"
        "## Code Patterns\n"
        "- Never use eval(). (Reason: security)\n"
    )


# ---------------------------------------------------------------------------
# parse_blacklist
# ---------------------------------------------------------------------------


class TestParseBlacklist:
    """Parsing blacklist markdown into structured entries."""

    def test_parse_mixed_entries(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        assert len(entries) == 5

    def test_advisory_entries_detected(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        advisory = [e for e in entries if not e.is_block_command]
        assert len(advisory) == 2

    def test_block_command_entries_detected(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        blocked = [e for e in entries if e.is_block_command]
        assert len(blocked) == 3

    def test_block_command_pattern_extracted(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        blocked = [e for e in entries if e.is_block_command]
        patterns = [e.pattern for e in blocked]
        assert "git push --force" in patterns
        assert "rm -rf /" in patterns
        assert "curl | bash" in patterns

    def test_block_command_reason_extracted(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        blocked = [e for e in entries if e.is_block_command]
        force_push = [e for e in blocked if e.pattern == "git push --force"][0]
        assert force_push.reason == "force push destroys history"

    def test_advisory_reason_extracted(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        advisory = [e for e in entries if not e.is_block_command]
        eval_entry = [e for e in advisory if "eval" in e.raw][0]
        assert eval_entry.reason == "security"

    def test_empty_content(self):
        entries = parse_blacklist("")
        assert entries == []

    def test_no_list_items(self):
        entries = parse_blacklist("# Heading\n\nSome text\n")
        assert entries == []

    def test_block_command_case_insensitive_prefix(self):
        content = "- Block_Command: git stash drop (Reason: data loss)\n"
        entries = parse_blacklist(content)
        assert len(entries) == 1
        assert entries[0].is_block_command
        assert entries[0].pattern == "git stash drop"

    def test_block_command_no_reason(self):
        content = "- block_command: dangerous_command\n"
        entries = parse_blacklist(content)
        assert len(entries) == 1
        assert entries[0].is_block_command
        assert entries[0].pattern == "dangerous_command"
        assert entries[0].reason == ""

    def test_preserves_order(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        # First entry should be advisory about tasks
        assert not entries[0].is_block_command
        assert "task" in entries[0].raw.lower()
        # Second entry should be block_command for git push
        assert entries[1].is_block_command
        assert entries[1].pattern == "git push --force"

    def test_ignores_non_list_lines(self):
        content = (
            "# Heading\n"
            "Some paragraph text.\n"
            "- Actual entry (Reason: test)\n"
            "## Another heading\n"
        )
        entries = parse_blacklist(content)
        assert len(entries) == 1

    def test_ignores_empty_list_items(self):
        content = "- \n- actual entry\n-\n"
        entries = parse_blacklist(content)
        assert len(entries) == 1
        assert entries[0].raw == "actual entry"


# ---------------------------------------------------------------------------
# get_block_commands / get_advisory_entries
# ---------------------------------------------------------------------------


class TestFilterEntries:
    """Filtering entries by type."""

    def test_get_block_commands_sorted(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        blocked = get_block_commands(entries)
        patterns = [e.pattern for e in blocked]
        assert patterns == sorted(patterns)

    def test_get_block_commands_excludes_advisory(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        blocked = get_block_commands(entries)
        assert all(e.is_block_command for e in blocked)

    def test_get_advisory_entries_excludes_blocked(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        advisory = get_advisory_entries(entries)
        assert all(not e.is_block_command for e in advisory)

    def test_get_block_commands_empty_for_advisory_only(self, advisory_only_blacklist):
        entries = parse_blacklist(advisory_only_blacklist)
        blocked = get_block_commands(entries)
        assert blocked == []


# ---------------------------------------------------------------------------
# load_blacklist
# ---------------------------------------------------------------------------


class TestLoadBlacklist:
    """Loading blacklist from filesystem."""

    def test_load_existing_blacklist(self, tmp_project, sample_blacklist):
        (tmp_project / ".corc" / "blacklist.md").write_text(sample_blacklist)
        entries = load_blacklist(tmp_project)
        assert len(entries) == 5

    def test_load_missing_file_returns_empty(self, tmp_project):
        entries = load_blacklist(tmp_project)
        assert entries == []

    def test_load_missing_corc_dir_returns_empty(self, tmp_path):
        entries = load_blacklist(tmp_path)
        assert entries == []

    def test_load_empty_file_returns_empty(self, tmp_project):
        (tmp_project / ".corc" / "blacklist.md").write_text("")
        entries = load_blacklist(tmp_project)
        assert entries == []


# ---------------------------------------------------------------------------
# generate_blacklist_hook_script
# ---------------------------------------------------------------------------


class TestGenerateBlacklistHookScript:
    """Hook script generation from block_command entries."""

    def test_generates_script_with_block_commands(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        script = generate_blacklist_hook_script(entries)
        assert script is not None
        assert script.startswith("#!/usr/bin/env bash\n")

    def test_returns_none_for_advisory_only(self, advisory_only_blacklist):
        entries = parse_blacklist(advisory_only_blacklist)
        script = generate_blacklist_hook_script(entries)
        assert script is None

    def test_contains_blocked_patterns(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        script = generate_blacklist_hook_script(entries)
        assert "git push --force" in script
        assert "rm -rf /" in script
        assert "curl | bash" in script

    def test_contains_block_reasons(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        script = generate_blacklist_hook_script(entries)
        assert "force push destroys history" in script
        assert "catastrophic deletion" in script

    def test_exits_with_code_2_on_block(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        script = generate_blacklist_hook_script(entries)
        assert "exit 2" in script

    def test_exits_with_code_0_on_allow(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        script = generate_blacklist_hook_script(entries)
        assert "exit 0" in script

    def test_contains_set_euo_pipefail(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        script = generate_blacklist_hook_script(entries)
        assert "set -euo pipefail" in script

    def test_reads_json_from_stdin(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        script = generate_blacklist_hook_script(entries)
        assert "INPUT=$(cat)" in script
        assert "jq" in script
        assert ".tool_input.command" in script

    def test_checks_tool_name_is_bash(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        script = generate_blacklist_hook_script(entries)
        assert "TOOL_NAME" in script
        assert "Bash" in script

    def test_uses_grep_fixed_string(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        script = generate_blacklist_hook_script(entries)
        assert "grep -qF" in script

    def test_generated_by_corc_comment(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        script = generate_blacklist_hook_script(entries)
        assert "Generated by corc" in script

    def test_default_reason_when_none(self):
        content = "- block_command: dangerous_thing\n"
        entries = parse_blacklist(content)
        script = generate_blacklist_hook_script(entries)
        assert "blocked by blacklist" in script

    def test_deterministic(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        s1 = generate_blacklist_hook_script(entries)
        s2 = generate_blacklist_hook_script(entries)
        assert s1 == s2

    def test_deterministic_regardless_of_entry_order(self):
        """block_command entries are sorted by pattern for determinism."""
        content1 = (
            "- block_command: zzz_command (Reason: r1)\n"
            "- block_command: aaa_command (Reason: r2)\n"
        )
        content2 = (
            "- block_command: aaa_command (Reason: r2)\n"
            "- block_command: zzz_command (Reason: r1)\n"
        )
        entries1 = parse_blacklist(content1)
        entries2 = parse_blacklist(content2)
        s1 = generate_blacklist_hook_script(entries1)
        s2 = generate_blacklist_hook_script(entries2)
        assert s1 == s2

    def test_single_quote_in_pattern_escaped(self):
        """Single quotes in patterns should be escaped for shell safety."""
        content = "- block_command: echo 'hello world' (Reason: test)\n"
        entries = parse_blacklist(content)
        script = generate_blacklist_hook_script(entries)
        # Should not break shell syntax
        assert script is not None
        assert "hello" in script


# ---------------------------------------------------------------------------
# generate_blacklist_hook_settings
# ---------------------------------------------------------------------------


class TestGenerateBlacklistHookSettings:
    """Settings generation for blacklist hooks."""

    def test_returns_settings_with_block_commands(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        settings = generate_blacklist_hook_settings(entries)
        assert settings is not None
        assert len(settings) == 1

    def test_returns_none_for_advisory_only(self, advisory_only_blacklist):
        entries = parse_blacklist(advisory_only_blacklist)
        settings = generate_blacklist_hook_settings(entries)
        assert settings is None

    def test_matcher_is_bash(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        settings = generate_blacklist_hook_settings(entries)
        assert settings[0]["matcher"] == "Bash"

    def test_hook_command_is_enforce_blacklist(self, sample_blacklist):
        entries = parse_blacklist(sample_blacklist)
        settings = generate_blacklist_hook_settings(entries)
        hook = settings[0]["hooks"][0]
        assert hook["type"] == "command"
        assert hook["command"] == ".claude/hooks/enforce-blacklist.sh"


# ---------------------------------------------------------------------------
# sync_blacklist_hooks
# ---------------------------------------------------------------------------


class TestSyncBlacklistHooks:
    """Writing blacklist hooks to disk."""

    def test_creates_hook_script(self, tmp_project, sample_blacklist):
        (tmp_project / ".corc" / "blacklist.md").write_text(sample_blacklist)
        written = sync_blacklist_hooks(tmp_project)
        hook_path = tmp_project / ".claude" / "hooks" / "enforce-blacklist.sh"
        assert hook_path.exists()
        assert str(hook_path) in written

    def test_creates_settings_json(self, tmp_project, sample_blacklist):
        (tmp_project / ".corc" / "blacklist.md").write_text(sample_blacklist)
        sync_blacklist_hooks(tmp_project)
        settings_path = tmp_project / ".claude" / "settings.json"
        assert settings_path.exists()

    def test_settings_json_contains_blacklist_hook(self, tmp_project, sample_blacklist):
        (tmp_project / ".corc" / "blacklist.md").write_text(sample_blacklist)
        sync_blacklist_hooks(tmp_project)
        settings_path = tmp_project / ".claude" / "settings.json"
        content = json.loads(settings_path.read_text())
        assert "PreToolUse" in content["hooks"]
        pre_hooks = content["hooks"]["PreToolUse"]
        commands = []
        for h in pre_hooks:
            for sub_h in h.get("hooks", []):
                commands.append(sub_h.get("command"))
        assert ".claude/hooks/enforce-blacklist.sh" in commands

    def test_hook_script_is_executable(self, tmp_project, sample_blacklist):
        (tmp_project / ".corc" / "blacklist.md").write_text(sample_blacklist)
        sync_blacklist_hooks(tmp_project)
        hook_path = tmp_project / ".claude" / "hooks" / "enforce-blacklist.sh"
        mode = os.stat(hook_path).st_mode
        assert mode & stat.S_IXUSR, "Hook script should be executable"

    def test_no_hooks_for_advisory_only(self, tmp_project, advisory_only_blacklist):
        (tmp_project / ".corc" / "blacklist.md").write_text(advisory_only_blacklist)
        written = sync_blacklist_hooks(tmp_project)
        hook_path = tmp_project / ".claude" / "hooks" / "enforce-blacklist.sh"
        assert not hook_path.exists()

    def test_no_hooks_for_missing_blacklist(self, tmp_project):
        written = sync_blacklist_hooks(tmp_project)
        hook_path = tmp_project / ".claude" / "hooks" / "enforce-blacklist.sh"
        assert not hook_path.exists()

    def test_idempotent(self, tmp_project, sample_blacklist):
        (tmp_project / ".corc" / "blacklist.md").write_text(sample_blacklist)
        sync_blacklist_hooks(tmp_project)

        # Read files after first sync
        hook_path = tmp_project / ".claude" / "hooks" / "enforce-blacklist.sh"
        settings_path = tmp_project / ".claude" / "settings.json"
        content1 = hook_path.read_bytes()
        settings1 = settings_path.read_bytes()

        # Run again
        sync_blacklist_hooks(tmp_project)
        content2 = hook_path.read_bytes()
        settings2 = settings_path.read_bytes()

        assert content1 == content2
        assert settings1 == settings2

    def test_cleanup_when_block_commands_removed(self, tmp_project, sample_blacklist):
        """Hooks should be removed when no block_command entries remain."""
        (tmp_project / ".corc" / "blacklist.md").write_text(sample_blacklist)
        sync_blacklist_hooks(tmp_project)

        hook_path = tmp_project / ".claude" / "hooks" / "enforce-blacklist.sh"
        assert hook_path.exists()

        # Replace with advisory-only content
        (tmp_project / ".corc" / "blacklist.md").write_text(
            "# Blacklist\n- Advisory entry only\n"
        )
        sync_blacklist_hooks(tmp_project)

        assert not hook_path.exists()

    def test_cleanup_removes_settings_entry(self, tmp_project, sample_blacklist):
        """PreToolUse entry should be removed from settings when no block commands."""
        (tmp_project / ".corc" / "blacklist.md").write_text(sample_blacklist)
        sync_blacklist_hooks(tmp_project)

        settings_path = tmp_project / ".claude" / "settings.json"
        content = json.loads(settings_path.read_text())
        assert "PreToolUse" in content["hooks"]

        # Remove block commands
        (tmp_project / ".corc" / "blacklist.md").write_text(
            "# Blacklist\n- Advisory only\n"
        )
        sync_blacklist_hooks(tmp_project)

        content = json.loads(settings_path.read_text())
        assert "PreToolUse" not in content.get("hooks", {})

    def test_preserves_existing_hooks(self, tmp_project, sample_blacklist):
        """Existing hooks in settings.json should be preserved."""
        claude_dir = tmp_project / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)
        settings_path = claude_dir / "settings.json"

        # Write existing settings with enforce-policy hook
        existing_settings = {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "Bash",
                        "hooks": [
                            {
                                "type": "command",
                                "command": ".claude/hooks/enforce-policy.sh",
                            }
                        ],
                    }
                ],
                "PostToolUse": [
                    {
                        "matcher": "Write|Edit",
                        "hooks": [
                            {
                                "type": "command",
                                "command": ".claude/hooks/format-python.sh",
                            }
                        ],
                    }
                ],
            }
        }
        settings_path.write_text(json.dumps(existing_settings, indent=2) + "\n")

        # Sync blacklist hooks
        (tmp_project / ".corc" / "blacklist.md").write_text(sample_blacklist)
        sync_blacklist_hooks(tmp_project)

        content = json.loads(settings_path.read_text())

        # Verify existing hooks preserved
        pre_hooks = content["hooks"]["PreToolUse"]
        commands = []
        for h in pre_hooks:
            for sub_h in h.get("hooks", []):
                commands.append(sub_h.get("command"))

        assert ".claude/hooks/enforce-policy.sh" in commands
        assert ".claude/hooks/enforce-blacklist.sh" in commands

        # PostToolUse should be untouched
        assert "PostToolUse" in content["hooks"]

    def test_creates_directories(self, tmp_project, sample_blacklist):
        """Should create .claude/ and .claude/hooks/ if they don't exist."""
        (tmp_project / ".corc" / "blacklist.md").write_text(sample_blacklist)
        assert not (tmp_project / ".claude").exists()
        sync_blacklist_hooks(tmp_project)
        assert (tmp_project / ".claude").is_dir()
        assert (tmp_project / ".claude" / "hooks").is_dir()

    def test_regenerates_on_blacklist_change(self, tmp_project):
        """Hook content changes when blacklist is updated."""
        content1 = "- block_command: dangerous1 (Reason: r1)\n"
        (tmp_project / ".corc" / "blacklist.md").write_text(content1)
        sync_blacklist_hooks(tmp_project)
        hook_path = tmp_project / ".claude" / "hooks" / "enforce-blacklist.sh"
        script1 = hook_path.read_text()

        content2 = (
            "- block_command: dangerous1 (Reason: r1)\n"
            "- block_command: dangerous2 (Reason: r2)\n"
        )
        (tmp_project / ".corc" / "blacklist.md").write_text(content2)
        sync_blacklist_hooks(tmp_project)
        script2 = hook_path.read_text()

        assert script1 != script2
        assert "dangerous2" in script2


# ---------------------------------------------------------------------------
# add_entry
# ---------------------------------------------------------------------------


class TestAddEntry:
    """Adding entries to the blacklist file."""

    def test_add_advisory_entry(self, tmp_project):
        (tmp_project / ".corc" / "blacklist.md").write_text("# Blacklist\n")
        formatted = add_entry(tmp_project, "Never use eval()", reason="security")
        content = (tmp_project / ".corc" / "blacklist.md").read_text()
        assert "- Never use eval() (Reason: security)" in content
        assert formatted == "- Never use eval() (Reason: security)"

    def test_add_block_command_entry(self, tmp_project):
        (tmp_project / ".corc" / "blacklist.md").write_text("# Blacklist\n")
        add_entry(
            tmp_project,
            "block_command: git push --force",
            reason="destroys history",
        )
        content = (tmp_project / ".corc" / "blacklist.md").read_text()
        assert "- block_command: git push --force (Reason: destroys history)" in content

    def test_add_entry_no_reason(self, tmp_project):
        (tmp_project / ".corc" / "blacklist.md").write_text("# Blacklist\n")
        add_entry(tmp_project, "block_command: rm -rf /")
        content = (tmp_project / ".corc" / "blacklist.md").read_text()
        assert "- block_command: rm -rf /" in content
        assert "(Reason:" not in content.split("rm -rf /")[1].split("\n")[0]

    def test_add_entry_creates_file_if_missing(self, tmp_project):
        # Remove the blacklist file
        blacklist_path = tmp_project / ".corc" / "blacklist.md"
        if blacklist_path.exists():
            blacklist_path.unlink()
        add_entry(tmp_project, "Never use eval()", reason="security")
        assert blacklist_path.exists()
        content = blacklist_path.read_text()
        assert "Never use eval()" in content

    def test_add_entry_creates_corc_dir_if_missing(self, tmp_path):
        add_entry(tmp_path, "Never use eval()")
        assert (tmp_path / ".corc" / "blacklist.md").exists()

    def test_add_entry_to_section(self, tmp_project):
        content = (
            "# Blacklist\n"
            "\n"
            "## Process\n"
            "- Existing process entry\n"
            "\n"
            "## Code Patterns\n"
            "- Existing code entry\n"
        )
        (tmp_project / ".corc" / "blacklist.md").write_text(content)
        add_entry(tmp_project, "New process entry", section="Process")

        result = (tmp_project / ".corc" / "blacklist.md").read_text()
        lines = result.splitlines()

        # Find the new entry
        new_idx = None
        process_idx = None
        code_idx = None
        for i, line in enumerate(lines):
            if "New process entry" in line:
                new_idx = i
            if "## Process" in line:
                process_idx = i
            if "## Code Patterns" in line:
                code_idx = i

        assert new_idx is not None
        assert process_idx is not None
        assert code_idx is not None
        # New entry should be between Process heading and Code Patterns heading
        assert process_idx < new_idx < code_idx

    def test_add_duplicate_entry_is_noop(self, tmp_project):
        content = "# Blacklist\n- Never use eval() (Reason: security)\n"
        (tmp_project / ".corc" / "blacklist.md").write_text(content)
        add_entry(tmp_project, "Never use eval()", reason="security")
        result = (tmp_project / ".corc" / "blacklist.md").read_text()
        assert result.count("Never use eval()") == 1


# ---------------------------------------------------------------------------
# remove_entry
# ---------------------------------------------------------------------------


class TestRemoveEntry:
    """Removing entries from the blacklist file."""

    def test_remove_existing_entry(self, tmp_project):
        content = (
            "# Blacklist\n"
            "- Never use eval() (Reason: security)\n"
            "- Never merge directly to main (Reason: review)\n"
        )
        (tmp_project / ".corc" / "blacklist.md").write_text(content)
        removed = remove_entry(tmp_project, "Never use eval()")
        assert removed

        result = (tmp_project / ".corc" / "blacklist.md").read_text()
        assert "Never use eval()" not in result
        assert "Never merge directly to main" in result

    def test_remove_block_command_entry(self, tmp_project):
        content = (
            "# Blacklist\n"
            "- block_command: git push --force (Reason: history)\n"
            "- Advisory entry\n"
        )
        (tmp_project / ".corc" / "blacklist.md").write_text(content)
        removed = remove_entry(tmp_project, "block_command: git push --force")
        assert removed

        result = (tmp_project / ".corc" / "blacklist.md").read_text()
        assert "git push --force" not in result
        assert "Advisory entry" in result

    def test_remove_nonexistent_entry(self, tmp_project):
        content = "# Blacklist\n- Some entry\n"
        (tmp_project / ".corc" / "blacklist.md").write_text(content)
        removed = remove_entry(tmp_project, "Nonexistent entry")
        assert not removed

    def test_remove_from_missing_file(self, tmp_project):
        removed = remove_entry(tmp_project, "Anything")
        assert not removed

    def test_remove_by_partial_match(self, tmp_project):
        content = "# Blacklist\n- block_command: git push --force (Reason: history)\n"
        (tmp_project / ".corc" / "blacklist.md").write_text(content)
        removed = remove_entry(tmp_project, "git push --force")
        assert removed

        result = (tmp_project / ".corc" / "blacklist.md").read_text()
        assert "git push --force" not in result

    def test_remove_only_first_match(self, tmp_project):
        content = "# Blacklist\n- Entry with eval\n- Another entry with eval\n"
        (tmp_project / ".corc" / "blacklist.md").write_text(content)
        removed = remove_entry(tmp_project, "eval")
        assert removed

        result = (tmp_project / ".corc" / "blacklist.md").read_text()
        # First match removed, second remains
        assert result.count("eval") == 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Verify deterministic output for blacklist hook generation."""

    def test_same_content_produces_same_script(self, sample_blacklist):
        entries1 = parse_blacklist(sample_blacklist)
        entries2 = parse_blacklist(sample_blacklist)
        assert generate_blacklist_hook_script(
            entries1
        ) == generate_blacklist_hook_script(entries2)

    def test_entry_order_does_not_affect_script(self):
        """Block commands are sorted by pattern, so order doesn't matter."""
        content_a = (
            "- block_command: zebra_cmd (Reason: z)\n"
            "- block_command: alpha_cmd (Reason: a)\n"
        )
        content_b = (
            "- block_command: alpha_cmd (Reason: a)\n"
            "- block_command: zebra_cmd (Reason: z)\n"
        )
        script_a = generate_blacklist_hook_script(parse_blacklist(content_a))
        script_b = generate_blacklist_hook_script(parse_blacklist(content_b))
        assert script_a == script_b

    def test_full_sync_deterministic(self, tmp_path):
        """Full sync to two different directories produces identical output."""
        proj1 = tmp_path / "proj1"
        proj2 = tmp_path / "proj2"
        for p in [proj1, proj2]:
            p.mkdir()
            (p / ".corc").mkdir()
            (p / ".corc" / "blacklist.md").write_text(
                "- block_command: bad1 (Reason: r1)\n"
                "- block_command: bad2 (Reason: r2)\n"
            )

        sync_blacklist_hooks(proj1)
        sync_blacklist_hooks(proj2)

        hook1 = (proj1 / ".claude" / "hooks" / "enforce-blacklist.sh").read_text()
        hook2 = (proj2 / ".claude" / "hooks" / "enforce-blacklist.sh").read_text()
        assert hook1 == hook2

        settings1 = (proj1 / ".claude" / "settings.json").read_text()
        settings2 = (proj2 / ".claude" / "settings.json").read_text()
        assert settings1 == settings2


# ---------------------------------------------------------------------------
# Advisory entries remain prompt-only
# ---------------------------------------------------------------------------


class TestAdvisoryPromptOnly:
    """Advisory entries should NOT generate hooks."""

    def test_advisory_entries_no_script(self, advisory_only_blacklist):
        entries = parse_blacklist(advisory_only_blacklist)
        assert generate_blacklist_hook_script(entries) is None

    def test_advisory_entries_no_settings(self, advisory_only_blacklist):
        entries = parse_blacklist(advisory_only_blacklist)
        assert generate_blacklist_hook_settings(entries) is None

    def test_advisory_entries_no_hook_files(self, tmp_project, advisory_only_blacklist):
        (tmp_project / ".corc" / "blacklist.md").write_text(advisory_only_blacklist)
        sync_blacklist_hooks(tmp_project)
        hook_path = tmp_project / ".claude" / "hooks" / "enforce-blacklist.sh"
        assert not hook_path.exists()

    def test_advisory_still_in_context(self, tmp_project, advisory_only_blacklist):
        """Advisory entries should still appear in assembled context."""
        from corc.context import assemble_context

        (tmp_project / ".corc" / "blacklist.md").write_text(advisory_only_blacklist)
        task = {"name": "test", "done_when": "done", "context_bundle": []}
        ctx = assemble_context(task, tmp_project)
        assert "Never use eval()" in ctx
        assert "AGENT BLACKLIST" in ctx


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLIBlacklist:
    """Test corc blacklist CLI commands."""

    @pytest.fixture
    def cli_project(self, tmp_path):
        """Create a minimal project directory for CLI tests."""
        corc_dir = tmp_path / ".corc"
        corc_dir.mkdir()
        (tmp_path / ".git").mkdir()
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "events").mkdir()
        (data_dir / "sessions").mkdir()
        (data_dir / "ratings").mkdir()
        (tmp_path / "knowledge").mkdir()
        (corc_dir / "blacklist.md").write_text("# Agent Blacklist\n")
        return tmp_path

    def test_blacklist_add_advisory(self, cli_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(cli_project)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["blacklist", "add", "Never use eval()", "--reason", "security"],
        )
        assert result.exit_code == 0
        assert "Added" in result.output

        content = (cli_project / ".corc" / "blacklist.md").read_text()
        assert "Never use eval() (Reason: security)" in content

    def test_blacklist_add_block_command(self, cli_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(cli_project)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "blacklist",
                "add",
                "block_command: git push --force",
                "--reason",
                "destroys history",
            ],
        )
        assert result.exit_code == 0
        assert "Added" in result.output
        assert "hook" in result.output.lower() or "Generated" in result.output

        # Verify hook was generated
        hook_path = cli_project / ".claude" / "hooks" / "enforce-blacklist.sh"
        assert hook_path.exists()

    def test_blacklist_remove(self, cli_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(cli_project)
        runner = CliRunner()

        # Add first
        runner.invoke(cli, ["blacklist", "add", "Never use eval()"])

        # Then remove
        result = runner.invoke(cli, ["blacklist", "remove", "Never use eval()"])
        assert result.exit_code == 0
        assert "Removed" in result.output

        content = (cli_project / ".corc" / "blacklist.md").read_text()
        assert "Never use eval()" not in content

    def test_blacklist_remove_nonexistent(self, cli_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(cli_project)
        runner = CliRunner()
        result = runner.invoke(cli, ["blacklist", "remove", "nonexistent"])
        assert result.exit_code == 1

    def test_blacklist_list(self, cli_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(cli_project)
        runner = CliRunner()

        # Add some entries
        runner.invoke(
            cli, ["blacklist", "add", "Never use eval()", "--reason", "security"]
        )
        runner.invoke(
            cli,
            [
                "blacklist",
                "add",
                "block_command: git push --force",
                "--reason",
                "history",
            ],
        )

        result = runner.invoke(cli, ["blacklist", "list"])
        assert result.exit_code == 0
        assert "eval" in result.output
        assert "git push --force" in result.output

    def test_blacklist_list_shows_enforced_marker(self, cli_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(cli_project)
        runner = CliRunner()

        runner.invoke(
            cli,
            ["blacklist", "add", "block_command: dangerous_cmd", "--reason", "danger"],
        )
        runner.invoke(cli, ["blacklist", "add", "Advisory entry"])

        result = runner.invoke(cli, ["blacklist", "list"])
        assert "[enforced]" in result.output
        assert "[advisory]" in result.output

    def test_blacklist_remove_block_command_cleans_hooks(
        self, cli_project, monkeypatch
    ):
        """Removing the last block_command entry should clean up hooks."""
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(cli_project)
        runner = CliRunner()

        # Add a block_command entry
        runner.invoke(
            cli,
            [
                "blacklist",
                "add",
                "block_command: git push --force",
                "--reason",
                "history",
            ],
        )
        hook_path = cli_project / ".claude" / "hooks" / "enforce-blacklist.sh"
        assert hook_path.exists()

        # Remove it
        runner.invoke(cli, ["blacklist", "remove", "git push --force"])

        # Hooks should be cleaned up
        assert not hook_path.exists()

    def test_blacklist_sync_hooks(self, cli_project, monkeypatch):
        from click.testing import CliRunner
        from corc.cli import cli

        monkeypatch.chdir(cli_project)
        runner = CliRunner()

        # Write block_command entries directly to file
        (cli_project / ".corc" / "blacklist.md").write_text(
            "# Blacklist\n- block_command: bad_cmd (Reason: bad)\n"
        )

        result = runner.invoke(cli, ["blacklist", "sync-hooks"])
        assert result.exit_code == 0

        hook_path = cli_project / ".claude" / "hooks" / "enforce-blacklist.sh"
        assert hook_path.exists()


# ---------------------------------------------------------------------------
# BlacklistEntry equality
# ---------------------------------------------------------------------------


class TestBlacklistEntry:
    """Tests for BlacklistEntry data class."""

    def test_equality(self):
        e1 = BlacklistEntry("raw text", is_block_command=False)
        e2 = BlacklistEntry("raw text", is_block_command=False)
        assert e1 == e2

    def test_inequality_different_raw(self):
        e1 = BlacklistEntry("raw text", is_block_command=False)
        e2 = BlacklistEntry("other text", is_block_command=False)
        assert e1 != e2

    def test_inequality_different_type(self):
        e1 = BlacklistEntry("raw", is_block_command=True, pattern="cmd")
        e2 = BlacklistEntry("raw", is_block_command=False)
        assert e1 != e2

    def test_repr_advisory(self):
        e = BlacklistEntry("Never use eval()")
        assert "advisory" in repr(e)

    def test_repr_block_command(self):
        e = BlacklistEntry("raw", is_block_command=True, pattern="cmd")
        assert "block_command" in repr(e)

    def test_not_equal_to_non_entry(self):
        e = BlacklistEntry("raw")
        assert e != "not an entry"
