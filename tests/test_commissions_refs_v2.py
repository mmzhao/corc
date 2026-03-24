"""Tests for commissions-expense-refs-v2 findings artifact.

Verifies that the findings file exists, is well-structured, and that
the listed files match actual grep output with zero omissions.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

# Paths
FINDINGS_PATH = (
    Path(__file__).parent.parent
    / "knowledge"
    / "research"
    / "commissions-expense-refs-v2.md"
)
BACKEND_SRC = (
    Path(os.environ.get("FDP_ROOT", str(Path.home() / "fdp")))
    / "workspaces"
    / "backend"
    / "src"
)

# The 10 confirmed files from findings (relative to workspaces/backend/src/)
CONFIRMED_FILES = {
    "accounting/account-class.ts",
    "accounting/chart-of-accounts/chart-of-accounts.ts",
    "accounting/subledgers/blueprints/deferred-commissions.ts",
    "actions/commands/impl/load-coa.ts",
    "actions/commands/impl/commissions/test-commissions.ts",
    "db/schema/accounting/nid-entities.ts",
    "codegen/actions/actions__GENERATED.ts",
    "smoke-test/types.ts",
    "smoke-test/phases/assert.ts",
    "data-lake/dev/config/sql-reports/income-statement.sql",
}

# Files explicitly verified to have NO references
NO_CHANGE_FILES = {
    "accounting/assets/commissions/events/accrue.ts",
    "accounting/assets/commissions/events/reverse-accrued.ts",
    "accounting/assets/commissions/events/reclass-to-deferred.ts",
    "accounting/assets/commissions/events/amortize.ts",
    "accounting/assets/commissions/CommissionsHandler.ts",
    "accounting/assets/commissions/CommissionsHandler.integration.test.ts",
    "accounting/assets/commissions/module/initialize.ts",
    "accounting/assets/commissions/module/start.ts",
}


class TestFindingsArtifactExists:
    def test_findings_file_exists(self):
        assert FINDINGS_PATH.exists(), f"Findings artifact not found at {FINDINGS_PATH}"

    def test_findings_file_not_empty(self):
        content = FINDINGS_PATH.read_text()
        assert len(content) > 100, "Findings artifact is too short to be valid"


class TestFindingsStructure:
    @pytest.fixture
    def content(self):
        return FINDINGS_PATH.read_text()

    def test_has_frontmatter(self, content):
        assert content.startswith("---"), "Missing YAML frontmatter"
        # Check frontmatter closes
        second_fence = content.index("---", 3)
        assert second_fence > 3, "Frontmatter not properly closed"

    def test_has_summary_section(self, content):
        assert "## Summary" in content

    def test_has_key_findings_section(self, content):
        assert "## Key Findings" in content

    def test_has_confirmed_section(self, content):
        assert "### Confirmed" in content

    def test_has_new_section(self, content):
        assert "### New" in content

    def test_has_no_change_needed_section(self, content):
        assert "### No-change-needed" in content

    def test_has_risks_section(self, content):
        assert "## Risks and Unknowns" in content

    def test_has_recommendations_section(self, content):
        assert "## Recommendations" in content

    def test_has_verification_section(self, content):
        assert "## Verification" in content


class TestConfirmedFilesListed:
    @pytest.fixture
    def content(self):
        return FINDINGS_PATH.read_text()

    @pytest.mark.parametrize("filepath", sorted(CONFIRMED_FILES))
    def test_confirmed_file_in_findings(self, content, filepath):
        # The file path (or its basename) should appear in the findings
        assert filepath in content, f"Confirmed file '{filepath}' not found in findings"


class TestNoChangeFilesListed:
    @pytest.fixture
    def content(self):
        return FINDINGS_PATH.read_text()

    @pytest.mark.parametrize("filepath", sorted(NO_CHANGE_FILES))
    def test_no_change_file_in_findings(self, content, filepath):
        assert filepath in content, f"No-change file '{filepath}' not found in findings"


def _grep_files(pattern: str, search_dir: str) -> set[str]:
    """Run grep -rlE and return matching file paths relative to BACKEND_SRC."""
    result = subprocess.run(
        ["grep", "-rlE", pattern, search_dir],
        capture_output=True,
        text=True,
    )
    # grep returns 0 on match, 1 on no match
    assert result.returncode in (0, 1), f"grep failed: {result.stderr}"
    files = set()
    for line in result.stdout.strip().splitlines():
        if line:
            rel = os.path.relpath(line, BACKEND_SRC)
            files.add(rel)
    return files


@pytest.mark.skipif(
    not BACKEND_SRC.exists(),
    reason=f"Backend source not found at {BACKEND_SRC}",
)
class TestGrepOutputMatchesFindings:
    """Verify findings match actual grep output with zero omissions."""

    @pytest.fixture
    def matched_files(self):
        """Run grep and collect matching file paths relative to backend/src/."""
        return _grep_files(
            "CommissionsExpense|commissions_expense|getCommissionsExpense",
            str(BACKEND_SRC),
        )

    def test_exact_10_files_found(self, matched_files):
        assert len(matched_files) == 10, (
            f"Expected 10 matching files, got {len(matched_files)}. "
            f"Files: {sorted(matched_files)}"
        )

    def test_no_omissions(self, matched_files):
        """Every file found by grep must be in the CONFIRMED_FILES list."""
        omissions = matched_files - CONFIRMED_FILES
        assert not omissions, (
            f"Files found by grep but NOT in findings: {sorted(omissions)}"
        )

    def test_no_phantom_entries(self, matched_files):
        """Every CONFIRMED_FILE must exist in grep output (no phantom entries)."""
        phantoms = CONFIRMED_FILES - matched_files
        assert not phantoms, (
            f"Files listed in findings but NOT found by grep: {sorted(phantoms)}"
        )

    def test_no_change_files_excluded(self, matched_files):
        """No-change files must NOT appear in grep results."""
        unexpected = NO_CHANGE_FILES & matched_files
        assert not unexpected, (
            f"No-change files unexpectedly matched grep: {sorted(unexpected)}"
        )


@pytest.mark.skipif(
    not BACKEND_SRC.exists(),
    reason=f"Backend source not found at {BACKEND_SRC}",
)
class TestSearchTermCoverage:
    """Verify each search term variant matches expected files."""

    def test_pascal_case_CommissionsExpense(self):
        files = _grep_files("CommissionsExpense", str(BACKEND_SRC))
        # Must find at least: account-class.ts, chart-of-accounts.ts,
        # nid-entities.ts, smoke-test/types.ts, smoke-test/phases/assert.ts
        assert len(files) >= 5, f"CommissionsExpense matched only {len(files)} files"
        for f in files:
            assert f in CONFIRMED_FILES, f"Unexpected file for CommissionsExpense: {f}"

    def test_snake_case_commissions_expense(self):
        files = _grep_files("commissions_expense", str(BACKEND_SRC))
        # Must find at least: account-class.ts, actions__GENERATED.ts,
        # income-statement.sql, smoke-test/types.ts
        assert len(files) >= 4, f"commissions_expense matched only {len(files)} files"
        for f in files:
            assert f in CONFIRMED_FILES, f"Unexpected file for commissions_expense: {f}"

    def test_getter_getCommissionsExpense(self):
        files = _grep_files("getCommissionsExpense", str(BACKEND_SRC))
        # Must find at least: chart-of-accounts.ts, deferred-commissions.ts,
        # load-coa.ts, test-commissions.ts, assert.ts
        assert len(files) >= 4, f"getCommissionsExpense matched only {len(files)} files"
        for f in files:
            assert f in CONFIRMED_FILES, (
                f"Unexpected file for getCommissionsExpense: {f}"
            )
