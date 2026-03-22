"""Tests for the PostToolUse code-formatting hook (.claude/hooks/format-python.sh).

Verifies that:
- Unformatted Python files get formatted after a simulated Write/Edit.
- Non-Python files are skipped gracefully.
- Missing file paths are handled gracefully.
- The hook exits 0 in all cases (never blocks the agent).
"""

import json
import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest

# Locate the hook script relative to this test file.
REPO_ROOT = Path(__file__).resolve().parent.parent
HOOK_SCRIPT = REPO_ROOT / ".claude" / "hooks" / "format-python.sh"
SETTINGS_FILE = REPO_ROOT / ".claude" / "settings.json"


def _run_hook(tool_input: dict, tool_name: str = "Write") -> subprocess.CompletedProcess:
    """Simulate a PostToolUse hook invocation by piping JSON to the hook script."""
    payload = json.dumps(
        {
            "session_id": "test-session",
            "hook_event_name": "PostToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_response": {"success": True},
        }
    )
    result = subprocess.run(
        [str(HOOK_SCRIPT)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result


class TestHookExists:
    """Verify the hook infrastructure is in place."""

    def test_hook_script_exists(self):
        assert HOOK_SCRIPT.exists(), f"Hook script not found at {HOOK_SCRIPT}"

    def test_hook_script_is_executable(self):
        assert os.access(HOOK_SCRIPT, os.X_OK), "Hook script is not executable"

    def test_settings_json_exists(self):
        assert SETTINGS_FILE.exists(), f"Settings file not found at {SETTINGS_FILE}"

    def test_settings_json_has_post_tool_use_hook(self):
        settings = json.loads(SETTINGS_FILE.read_text())
        assert "hooks" in settings
        assert "PostToolUse" in settings["hooks"]
        hook_entries = settings["hooks"]["PostToolUse"]
        assert len(hook_entries) >= 1

        # Find the Write|Edit matcher entry.
        write_edit_entry = None
        for entry in hook_entries:
            if entry.get("matcher") in ("Write|Edit", "Edit|Write"):
                write_edit_entry = entry
                break
        assert write_edit_entry is not None, (
            "No PostToolUse hook with matcher 'Write|Edit' found"
        )
        assert len(write_edit_entry["hooks"]) >= 1
        assert write_edit_entry["hooks"][0]["type"] == "command"
        assert "format-python" in write_edit_entry["hooks"][0]["command"]


class TestPythonFormatting:
    """Verify that unformatted Python files get formatted by the hook."""

    def test_unformatted_python_gets_formatted(self, tmp_path):
        """Core test: write badly-formatted Python and verify ruff formats it."""
        ugly_code = textwrap.dedent("""\
            import   os
            import  sys
            x=1
            y =    2
            z=   [1,2,    3,4,     5]
            if x==1:
                 print(  "hello"  )
            def   foo(  a,b,   c  ):
                return a+b+c
        """)
        py_file = tmp_path / "ugly.py"
        py_file.write_text(ugly_code)

        result = _run_hook({"file_path": str(py_file)})
        assert result.returncode == 0, f"Hook failed: {result.stderr}"

        formatted = py_file.read_text()
        # ruff format normalises spacing around operators and commas.
        assert "x=1" not in formatted, "Assignment spacing was not fixed"
        assert "import   os" not in formatted, "Import spacing was not fixed"
        # Verify the file is still valid Python (ruff wouldn't break it).
        compile(formatted, str(py_file), "exec")

    def test_already_formatted_python_unchanged(self, tmp_path):
        """Well-formatted file should not be altered."""
        clean_code = textwrap.dedent("""\
            import os
            import sys

            x = 1
            y = 2
        """)
        py_file = tmp_path / "clean.py"
        py_file.write_text(clean_code)

        result = _run_hook({"file_path": str(py_file)})
        assert result.returncode == 0
        assert py_file.read_text() == clean_code

    def test_formatting_preserves_semantics(self, tmp_path):
        """Formatting must not change runtime behaviour."""
        code = textwrap.dedent("""\
            def add(a,b):
                return a+b

            result=add(1,2)
            assert result==3
        """)
        py_file = tmp_path / "semantics.py"
        py_file.write_text(code)

        result = _run_hook({"file_path": str(py_file)})
        assert result.returncode == 0

        formatted = py_file.read_text()
        # Execute to confirm semantics preserved.
        exec(compile(formatted, str(py_file), "exec"), {})


class TestNonPythonFiles:
    """Verify the hook skips non-Python files gracefully."""

    def test_javascript_file_skipped(self, tmp_path):
        js_file = tmp_path / "app.js"
        original = "const x=1;  const y =  2;"
        js_file.write_text(original)

        result = _run_hook({"file_path": str(js_file)})
        assert result.returncode == 0
        assert js_file.read_text() == original, "JS file should not be modified"

    def test_markdown_file_skipped(self, tmp_path):
        md_file = tmp_path / "README.md"
        original = "#   Bad   Heading\n"
        md_file.write_text(original)

        result = _run_hook({"file_path": str(md_file)})
        assert result.returncode == 0
        assert md_file.read_text() == original, "Markdown file should not be modified"

    def test_yaml_file_skipped(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        original = "key:   value\n"
        yaml_file.write_text(original)

        result = _run_hook({"file_path": str(yaml_file)})
        assert result.returncode == 0
        assert yaml_file.read_text() == original


class TestEdgeCases:
    """Verify graceful handling of edge cases."""

    def test_missing_file_path(self):
        """No file_path in tool_input -> exit 0, no crash."""
        result = _run_hook({})
        assert result.returncode == 0

    def test_empty_file_path(self):
        """Empty string file_path -> exit 0, no crash."""
        result = _run_hook({"file_path": ""})
        assert result.returncode == 0

    def test_nonexistent_file(self):
        """File that doesn't exist -> exit 0, no crash."""
        result = _run_hook({"file_path": "/tmp/does_not_exist_12345.py"})
        assert result.returncode == 0

    def test_edit_tool_also_triggers(self, tmp_path):
        """Hook should also work when tool_name is Edit."""
        ugly = "x=1\n"
        py_file = tmp_path / "edit_test.py"
        py_file.write_text(ugly)

        result = _run_hook({"file_path": str(py_file)}, tool_name="Edit")
        assert result.returncode == 0
        formatted = py_file.read_text()
        assert "x = 1" in formatted

    def test_empty_python_file(self, tmp_path):
        """Empty .py file should not cause errors."""
        py_file = tmp_path / "empty.py"
        py_file.write_text("")

        result = _run_hook({"file_path": str(py_file)})
        assert result.returncode == 0
