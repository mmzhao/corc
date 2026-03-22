"""Done-when validation rules.

Each rule checks a specific condition and returns (passed, detail).
Rules are composable — a task's done_when can reference multiple rules.
"""

import re
import subprocess
from pathlib import Path


def validate_file_exists(path: str, project_root: Path) -> tuple[bool, str]:
    full = project_root / path
    if full.exists():
        return True, f"File exists: {path}"
    return False, f"File not found: {path}"


def validate_file_not_empty(path: str, project_root: Path) -> tuple[bool, str]:
    full = project_root / path
    if not full.exists():
        return False, f"File not found: {path}"
    if full.stat().st_size == 0:
        return False, f"File is empty: {path}"
    return True, f"File has content: {path} ({full.stat().st_size} bytes)"


def validate_tests_pass(test_path: str | None = None, project_root: Path = Path(".")) -> tuple[bool, str]:
    cmd = ["python", "-m", "pytest", "-x", "-q"]
    if test_path:
        cmd.append(test_path)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(project_root),
        )
        if result.returncode == 0:
            return True, f"Tests passed:\n{result.stdout.strip()}"
        return False, f"Tests failed (exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "Tests timed out after 120s"
    except FileNotFoundError:
        return False, "pytest not found"


def validate_contains_pattern(path: str, pattern: str, project_root: Path) -> tuple[bool, str]:
    full = project_root / path
    if not full.exists():
        return False, f"File not found: {path}"
    content = full.read_text()
    if re.search(pattern, content):
        return True, f"Pattern '{pattern}' found in {path}"
    return False, f"Pattern '{pattern}' not found in {path}"


VALIDATORS = {
    "file_exists": validate_file_exists,
    "file_not_empty": validate_file_not_empty,
    "tests_pass": validate_tests_pass,
    "contains_pattern": validate_contains_pattern,
}


def run_validations(rules: list[dict], project_root: Path) -> tuple[bool, list[tuple[bool, str]]]:
    """Run a list of validation rules. Returns (all_passed, [(passed, detail), ...])."""
    results = []
    for rule in rules:
        if isinstance(rule, str):
            # Simple string rule like "tests_pass"
            if rule in VALIDATORS:
                passed, detail = VALIDATORS[rule](project_root=project_root)
                results.append((passed, detail))
            else:
                results.append((False, f"Unknown validation rule: {rule}"))
            continue

        for rule_name, rule_arg in rule.items():
            if rule_name == "file_exists":
                results.append(validate_file_exists(rule_arg, project_root))
            elif rule_name == "file_not_empty":
                results.append(validate_file_not_empty(rule_arg, project_root))
            elif rule_name == "tests_pass":
                results.append(validate_tests_pass(rule_arg, project_root))
            elif rule_name == "contains_pattern":
                if isinstance(rule_arg, dict):
                    results.append(validate_contains_pattern(
                        rule_arg["path"], rule_arg["pattern"], project_root
                    ))
                else:
                    results.append((False, f"contains_pattern expects dict with path and pattern"))
            else:
                results.append((False, f"Unknown validation rule: {rule_name}"))

    all_passed = all(p for p, _ in results)
    return all_passed, results
