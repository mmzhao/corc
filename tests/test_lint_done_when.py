"""Tests for done_when quality linter.

Covers accepted (testable) and rejected (subjective) examples,
edge cases, and CLI integration with --strict flag.
"""

import pytest
from click.testing import CliRunner

from corc.lint_done_when import lint_done_when, LintResult, SUBJECTIVE_PATTERNS, TESTABLE_PATTERNS


# ---------------------------------------------------------------------------
# Accepted examples — testable, measurable criteria (should pass)
# ---------------------------------------------------------------------------

ACCEPTED_EXAMPLES = [
    "All unit tests pass",
    "File src/main.py exists",
    "Output matches JSON schema",
    "Tests pass and coverage > 80%",
    "FTS5 queries return BM25-ranked results",
    "Integration test with 10 sample docs passes",
    "Search returns results within 100ms",
    "PR created with passing CI checks",
    "CLI outputs valid JSON",
    "Error handling returns 404 for missing resources",
    "File contains expected pattern",
    "Schema validation passes for all document types",
    "Database migration creates 3 tables",
    "pytest runs with 0 failures",
    "Exit status is 0",
    "Linter reports no errors",
    "Log file contains the expected entries",
    "Response includes Content-Type header",
    "Commit merged into main branch",
]


@pytest.mark.parametrize("criteria", ACCEPTED_EXAMPLES)
def test_accepted_criteria_pass(criteria):
    """Testable criteria should pass the linter (no warnings)."""
    result = lint_done_when(criteria)
    assert result.passed, f"Expected pass for: {criteria!r}, got warnings: {result.warnings}"
    assert result.has_testable_pattern
    assert result.subjective_words == []


# ---------------------------------------------------------------------------
# Rejected examples — subjective, non-measurable criteria (should fail)
# ---------------------------------------------------------------------------

REJECTED_EXAMPLES = [
    ("Works correctly", "correct"),
    ("Code is clean", "clean"),
    ("Good performance", "good"),
    ("Implementation is proper", "proper"),
    ("Nice code structure", "nice"),
    ("Works well", "well"),
    ("Appropriate error handling", "appropriate"),
    ("Reasonable response time", "reasonable"),
    ("Elegant solution", "elegant"),
    ("Beautiful architecture", "beautiful"),
    ("Code is readable", "readable"),
    ("System is maintainable", "maintainable"),
    ("Robust implementation", "robust"),
    ("Optimal performance", "optimal"),
    ("Code is acceptable", "acceptable"),
    ("Suitable for production", "suitable"),
    ("Intuitive API design", "intuitive"),
    ("Keep it simple", "simple"),
    ("Code is performant", "performant"),
]


@pytest.mark.parametrize("criteria,expected_word", REJECTED_EXAMPLES)
def test_rejected_criteria_fail(criteria, expected_word):
    """Subjective criteria should fail the linter (produce warnings)."""
    result = lint_done_when(criteria)
    assert not result.passed, f"Expected fail for: {criteria!r}"
    assert expected_word in result.subjective_words, (
        f"Expected subjective word '{expected_word}' in {result.subjective_words}"
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_string():
    result = lint_done_when("")
    assert not result.passed
    assert "empty" in result.warnings[0].lower()


def test_whitespace_only():
    result = lint_done_when("   ")
    assert not result.passed
    assert "empty" in result.warnings[0].lower()


def test_well_hyphenated_compound_accepted():
    """'well-tested' and 'well-defined' should NOT trigger the 'well' warning."""
    result = lint_done_when("Code is well-tested and tests pass")
    assert result.passed, f"well-tested should be OK, got warnings: {result.warnings}"
    assert "well" not in result.subjective_words


def test_well_standalone_rejected():
    """Standalone 'well' (not hyphenated) should trigger a warning."""
    result = lint_done_when("System performs well")
    assert not result.passed
    assert "well" in result.subjective_words


def test_mixed_subjective_and_testable():
    """Criteria with both subjective words AND testable patterns should still warn."""
    result = lint_done_when("Tests pass and code is clean")
    assert not result.passed
    assert "clean" in result.subjective_words
    # But it does have testable patterns
    assert result.has_testable_pattern


def test_no_testable_pattern_warning():
    """Criteria without any testable pattern should warn even if not subjective."""
    result = lint_done_when("Everything is done")
    assert not result.passed
    assert any("testable pattern" in w.lower() for w in result.warnings)


def test_case_insensitive_subjective():
    """Subjective word detection should be case-insensitive."""
    result = lint_done_when("GOOD performance")
    assert not result.passed
    assert "good" in result.subjective_words


def test_case_insensitive_testable():
    """Testable pattern detection should be case-insensitive."""
    result = lint_done_when("ALL TESTS PASS")
    assert result.passed


def test_multiple_subjective_words():
    """Multiple subjective words should all be reported."""
    result = lint_done_when("Good, clean, and proper code")
    assert not result.passed
    assert "good" in result.subjective_words
    assert "clean" in result.subjective_words
    assert "proper" in result.subjective_words
    assert len(result.subjective_words) >= 3


def test_numeric_threshold_accepted():
    """Criteria with numeric thresholds should be accepted."""
    result = lint_done_when("Latency under 50ms for 100 requests")
    assert result.passed


def test_correctly_variant_rejected():
    """'correctly' (adverb form) should also be rejected."""
    result = lint_done_when("Handles input correctly")
    assert not result.passed
    assert "correct" in result.subjective_words


def test_properly_variant_rejected():
    """'properly' (adverb form) should also be rejected."""
    result = lint_done_when("Handles errors properly")
    assert not result.passed
    assert "proper" in result.subjective_words


def test_nicely_variant_rejected():
    """'nicely' (adverb form) should also be rejected."""
    result = lint_done_when("Formats output nicely")
    assert not result.passed
    assert "nice" in result.subjective_words


def test_reasonable_reasonably_rejected():
    """Both 'reasonable' and 'reasonably' should be rejected."""
    r1 = lint_done_when("Reasonable speed")
    assert not r1.passed
    r2 = lint_done_when("Reasonably fast")
    assert not r2.passed


def test_lint_result_properties():
    """LintResult dataclass properties work correctly."""
    r = LintResult(criteria="test")
    assert r.passed  # no warnings
    assert r.criteria == "test"
    assert r.warnings == []
    assert r.subjective_words == []
    assert r.has_testable_pattern is False

    r.warnings.append("a warning")
    assert not r.passed


def test_real_world_accepted_criteria():
    """Real-world testable criteria from SPEC.md should pass."""
    specs = [
        "Frontmatter parsing works for all document types; chunking produces ~500 token chunks; unit tests pass",
        "All tables created; migration runs cleanly; schema matches spec",
        "FTS5 queries return BM25-ranked results; integration test with 10 sample docs passes",
        "Review posted as PR comment via gh pr review; no critical issues found",
        "Linter rejects subjective done_when criteria and accepts testable ones; tests cover examples",
    ]
    for criteria in specs:
        result = lint_done_when(criteria)
        assert result.has_testable_pattern, f"Expected testable for: {criteria!r}"


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------

def test_cli_task_create_warns_on_subjective(tmp_path, monkeypatch):
    """corc task create should show warnings for subjective criteria."""
    import corc.config as config_mod
    monkeypatch.setattr(config_mod, "get_project_root", lambda: tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)

    from corc.cli import cli
    runner = CliRunner()
    result = runner.invoke(cli, [
        "task", "create", "my-task",
        "--done-when", "Code is clean and good",
    ])
    # Warnings appear in output (Click mixes stdout/stderr by default)
    assert "clean" in result.output


def test_cli_task_create_strict_rejects(tmp_path, monkeypatch):
    """corc task create --strict should reject subjective criteria."""
    import corc.config as config_mod
    monkeypatch.setattr(config_mod, "get_project_root", lambda: tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)

    from corc.cli import cli
    runner = CliRunner()
    result = runner.invoke(cli, [
        "task", "create", "my-task",
        "--done-when", "Works correctly",
        "--strict",
    ])
    assert result.exit_code != 0
    assert "strict" in result.output.lower() or "subjective" in result.output.lower()


def test_cli_task_create_strict_accepts_testable(tmp_path, monkeypatch):
    """corc task create --strict should accept testable criteria."""
    import corc.config as config_mod
    monkeypatch.setattr(config_mod, "get_project_root", lambda: tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)

    from corc.cli import cli
    runner = CliRunner()
    result = runner.invoke(cli, [
        "task", "create", "my-task",
        "--done-when", "All tests pass and file exists",
        "--strict",
    ])
    assert result.exit_code == 0
    assert "Created task" in result.output


def test_cli_task_create_no_strict_still_creates(tmp_path, monkeypatch):
    """Without --strict, subjective criteria produce warnings but still create the task."""
    import corc.config as config_mod
    monkeypatch.setattr(config_mod, "get_project_root", lambda: tmp_path)
    (tmp_path / "data").mkdir(exist_ok=True)

    from corc.cli import cli
    runner = CliRunner()
    result = runner.invoke(cli, [
        "task", "create", "my-task",
        "--done-when", "Code is clean",
    ])
    # Task is still created despite warning
    assert result.exit_code == 0
    assert "Created task" in result.output


# ---------------------------------------------------------------------------
# Pattern coverage — ensure our pattern lists are non-empty and compiled
# ---------------------------------------------------------------------------

def test_subjective_patterns_list_not_empty():
    assert len(SUBJECTIVE_PATTERNS) >= 10


def test_testable_patterns_list_not_empty():
    assert len(TESTABLE_PATTERNS) >= 10


def test_all_subjective_patterns_compiled():
    for pattern, label in SUBJECTIVE_PATTERNS:
        assert hasattr(pattern, "search"), f"Pattern for '{label}' is not compiled"


def test_all_testable_patterns_compiled():
    for pattern in TESTABLE_PATTERNS:
        assert hasattr(pattern, "search"), f"Pattern is not compiled"
