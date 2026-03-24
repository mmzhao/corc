"""Tests for corc.pr module.

Covers merge_pr() behavior when gh pr merge returns non-zero exit codes,
verifying that the actual PR state is checked before declaring failure.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from corc.pr import _check_pr_merged, merge_pr


# ---------------------------------------------------------------------------
# _check_pr_merged helper
# ---------------------------------------------------------------------------


class TestCheckPrMerged:
    """Test _check_pr_merged() helper."""

    @patch("corc.pr.subprocess.run")
    def test_returns_true_when_state_is_merged(self, mock_run):
        """_check_pr_merged returns True when gh pr view reports MERGED."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"state": "MERGED"}),
        )
        assert _check_pr_merged(Path("/tmp/fake"), 42) is True

    @patch("corc.pr.subprocess.run")
    def test_returns_false_when_state_is_open(self, mock_run):
        """_check_pr_merged returns False when gh pr view reports OPEN."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"state": "OPEN"}),
        )
        assert _check_pr_merged(Path("/tmp/fake"), 42) is False

    @patch("corc.pr.subprocess.run")
    def test_returns_false_when_state_is_closed(self, mock_run):
        """_check_pr_merged returns False when state is CLOSED (not merged)."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"state": "CLOSED"}),
        )
        assert _check_pr_merged(Path("/tmp/fake"), 42) is False

    @patch("corc.pr.subprocess.run")
    def test_returns_false_when_gh_pr_view_fails(self, mock_run):
        """_check_pr_merged returns False when gh pr view itself fails."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="not found",
        )
        assert _check_pr_merged(Path("/tmp/fake"), 42) is False

    @patch("corc.pr.subprocess.run")
    def test_returns_false_on_invalid_json(self, mock_run):
        """_check_pr_merged returns False on unparseable JSON output."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="not json",
        )
        assert _check_pr_merged(Path("/tmp/fake"), 42) is False

    @patch("corc.pr.subprocess.run")
    def test_returns_false_on_subprocess_error(self, mock_run):
        """_check_pr_merged returns False when subprocess raises."""
        mock_run.side_effect = OSError("command not found")
        assert _check_pr_merged(Path("/tmp/fake"), 42) is False

    @patch("corc.pr.subprocess.run")
    def test_calls_gh_pr_view_with_json_state(self, mock_run):
        """_check_pr_merged calls gh pr view --json state."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=json.dumps({"state": "OPEN"}),
        )
        _check_pr_merged(Path("/tmp/fake"), 99)

        cmd = mock_run.call_args[0][0]
        assert cmd == ["gh", "pr", "view", "99", "--json", "state"]


# ---------------------------------------------------------------------------
# merge_pr — non-zero exit with verified merge state
# ---------------------------------------------------------------------------


class TestMergePrVerifiesState:
    """Test merge_pr() verifies actual PR state after non-zero gh pr merge."""

    @patch("corc.pr.subprocess.run")
    def test_nonzero_exit_merged_state_returns_true(self, mock_run):
        """merge_pr returns True when gh pr merge fails but PR is actually MERGED.

        This covers the case where gh pr merge returns non-zero because
        --delete-branch failed or a status check race occurred, but the
        merge itself succeeded.
        """
        # First call: gh pr merge returns non-zero
        merge_result = MagicMock(returncode=1, stderr="failed to delete branch")
        # Second call: gh pr view shows MERGED
        view_result = MagicMock(
            returncode=0,
            stdout=json.dumps({"state": "MERGED"}),
        )
        mock_run.side_effect = [merge_result, view_result]

        result = merge_pr(Path("/tmp/fake"), pr_number=42)
        assert result is True

    @patch("corc.pr.subprocess.run")
    def test_nonzero_exit_open_state_returns_false(self, mock_run):
        """merge_pr returns False when gh pr merge fails and PR is still OPEN.

        This is the genuine failure case — the merge didn't happen.
        """
        # First call: gh pr merge returns non-zero
        merge_result = MagicMock(returncode=1, stderr="merge conflict")
        # Second call: gh pr view shows OPEN
        view_result = MagicMock(
            returncode=0,
            stdout=json.dumps({"state": "OPEN"}),
        )
        mock_run.side_effect = [merge_result, view_result]

        result = merge_pr(Path("/tmp/fake"), pr_number=42)
        assert result is False

    @patch("corc.pr.subprocess.run")
    def test_zero_exit_returns_true_without_checking_state(self, mock_run):
        """merge_pr returns True immediately on zero exit (no extra gh call)."""
        mock_run.return_value = MagicMock(returncode=0)

        result = merge_pr(Path("/tmp/fake"), pr_number=42)
        assert result is True
        # Only one subprocess call (gh pr merge), no gh pr view
        assert mock_run.call_count == 1

    @patch("corc.pr.subprocess.run")
    def test_nonzero_exit_view_fails_returns_false(self, mock_run):
        """merge_pr returns False when both gh pr merge and gh pr view fail.

        If we can't verify the state, we conservatively report failure.
        """
        # First call: gh pr merge returns non-zero
        merge_result = MagicMock(returncode=1, stderr="error")
        # Second call: gh pr view also fails
        view_result = MagicMock(returncode=1, stderr="network error")
        mock_run.side_effect = [merge_result, view_result]

        result = merge_pr(Path("/tmp/fake"), pr_number=42)
        assert result is False

    @patch("corc.pr.subprocess.run")
    def test_nonzero_exit_calls_gh_pr_view(self, mock_run):
        """After non-zero exit, merge_pr calls gh pr view to check state."""
        merge_result = MagicMock(returncode=1, stderr="error")
        view_result = MagicMock(
            returncode=0,
            stdout=json.dumps({"state": "OPEN"}),
        )
        mock_run.side_effect = [merge_result, view_result]

        merge_pr(Path("/tmp/fake"), pr_number=42)

        assert mock_run.call_count == 2
        # Second call should be gh pr view
        second_call_cmd = mock_run.call_args_list[1][0][0]
        assert second_call_cmd[0] == "gh"
        assert second_call_cmd[1] == "pr"
        assert second_call_cmd[2] == "view"
        assert second_call_cmd[3] == "42"
        assert "--json" in second_call_cmd
        assert "state" in second_call_cmd

    @patch("corc.pr.subprocess.run")
    def test_subprocess_exception_returns_false(self, mock_run):
        """merge_pr returns False when subprocess.run raises an exception."""
        mock_run.side_effect = OSError("gh not found")
        result = merge_pr(Path("/tmp/fake"), pr_number=42)
        assert result is False
