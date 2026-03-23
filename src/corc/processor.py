"""Processor module — validates output, updates state, collects findings.

Reads work state + agent output, writes task completions.
Can be called standalone: `corc process TASK_ID`

PR workflow: after validation, posts a review comment on the PR via
`gh pr comment`. For auto-merge repos, merges the PR. For human-only
repos, leaves the PR open and notifies the operator.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from corc.adaptive_retry import AdaptiveRetryTracker, TaskOutcome
from corc.audit import AuditLog
from corc.dispatch import AgentResult
from corc.mutations import MutationLog
from corc.notifications import (
    NotificationManager,
    notify_escalation,
    notify_pr_awaiting_human_merge,
    notify_task_failure,
)
from corc.pr import PRInfo, merge_pr, post_review_comment
from corc.repo_policy import get_repo_policy
from corc.retry import create_escalation
from corc.sessions import SessionLogger
from corc.state import WorkState
from corc.validate import run_validations


@dataclass
class ProcessResult:
    """Result of processing a completed task."""

    task_id: str
    passed: bool
    details: list[tuple[bool, str]] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    pr_merged: bool = False
    pr_commented: bool = False


def process_completed(
    task: dict,
    result: AgentResult,
    attempt: int,
    mutation_log: MutationLog,
    state: WorkState,
    audit_log: AuditLog,
    session_logger: SessionLogger,
    project_root: Path,
    notification_manager: NotificationManager | None = None,
    adaptive_tracker: AdaptiveRetryTracker | None = None,
    pr_info: PRInfo | None = None,
) -> ProcessResult:
    """Validate task output and update state.

    If the agent exited with error, marks task as failed or escalated.
    Otherwise, runs done_when validations. If all pass, marks completed.
    If validations fail, marks task as failed or escalated.

    After max_retries are exhausted, sets status to 'escalated' instead
    of 'failed' and creates an escalation record.

    When notification_manager is provided, sends notifications on
    escalation and task failure events.

    When adaptive_tracker is provided, records the task outcome for
    adaptive retry policy adjustments.

    When pr_info is provided, posts a validation summary as a PR comment.
    For auto-merge repos, merges the PR after posting the review comment.
    For human-only repos, leaves the PR open and notifies the operator.
    """
    task_id = task["id"]
    max_retries = task.get("max_retries", 3)

    # Guard: skip if already completed (prevents duplicate completion mutations)
    state.refresh()
    current = state.get_task(task_id)
    if current and current["status"] == "completed":
        audit_log.log(
            "task_completion_skipped", task_id=task_id, reason="already completed"
        )
        return ProcessResult(
            task_id=task_id,
            passed=True,
            details=[(True, "Task already completed — skipped duplicate completion")],
        )

    # Agent crashed or errored
    if result.exit_code != 0:
        error_msg = f"Agent exited with code {result.exit_code}"
        error_detail = result.output[:1000] if result.output else error_msg

        if attempt > max_retries:
            # Retries exhausted — escalate
            _escalate_task(
                task,
                attempt,
                error_detail,
                mutation_log,
                audit_log,
                session_logger,
                notification_manager=notification_manager,
            )
        else:
            # Mark as failed (retriable — scheduler will pick it up)
            mutation_log.append(
                "task_failed",
                {
                    "attempt": attempt,
                    "exit_code": result.exit_code,
                    "attempt_count": attempt,
                },
                reason=error_msg,
                task_id=task_id,
            )
            audit_log.log(
                "task_failed",
                task_id=task_id,
                attempt=attempt,
                exit_code=result.exit_code,
            )
            # Notify on task failure
            if notification_manager:
                notify_task_failure(notification_manager, task, attempt, error_detail)

        # Record failure outcome for adaptive retry tracking
        if adaptive_tracker is not None:
            task_type = task.get("task_type", task.get("type", "general"))
            role = task.get("role", "unknown")
            adaptive_tracker.record_outcome(
                TaskOutcome(
                    task_type=task_type,
                    role=role,
                    attempt=attempt,
                    success=False,
                    task_id=task_id,
                )
            )

        state.refresh()
        return ProcessResult(
            task_id=task_id,
            passed=False,
            details=[(False, error_msg)],
        )

    # Parse and run done_when validations
    done_when = task.get("done_when", "")
    rules = _parse_done_when(done_when)

    if rules:
        all_passed, details = run_validations(rules, project_root)
    else:
        # No machine-parseable rules; if exit code was 0, consider passed
        all_passed = True
        details = [(True, "Agent completed successfully (no validation rules)")]

    # Extract findings from agent output
    findings = _extract_findings(result.output)

    # Log validation result
    session_logger.log_validation(
        task_id,
        attempt,
        all_passed,
        json.dumps([d for _, d in details]),
    )

    # Post validation summary as PR comment (if PR exists)
    pr_commented = False
    pr_merged = False
    if pr_info and pr_info.number:
        pr_commented = post_review_comment(
            project_root,
            pr_info.number,
            all_passed,
            details,
            findings=findings,
        )
        if pr_commented:
            audit_log.log(
                "pr_review_comment_posted",
                task_id=task_id,
                pr_number=pr_info.number,
                passed=all_passed,
            )
        else:
            audit_log.log(
                "pr_review_comment_failed",
                task_id=task_id,
                pr_number=pr_info.number,
            )

    if all_passed:
        # Check merge policy: human-only repos create PR but do not merge
        policy = get_repo_policy(project_root)

        if policy.is_human_only:
            # human-only: mark task as pending_merge (awaiting human merge)
            mutation_log.append(
                "task_pending_merge",
                {
                    "findings": findings,
                    "proof_of_work": {"output_preview": result.output[:1000]},
                    "merge_policy": "human-only",
                    "pr_url": pr_info.url if pr_info else None,
                    "pr_number": pr_info.number if pr_info else None,
                },
                reason="Validation passed; merge policy is human-only — awaiting human merge",
                task_id=task_id,
            )
            audit_log.log(
                "task_pending_merge",
                task_id=task_id,
                attempt=attempt,
                merge_policy="human-only",
                pr_url=pr_info.url if pr_info else None,
            )
            # Notify operator that PR is awaiting human merge
            if notification_manager and pr_info:
                notify_pr_awaiting_human_merge(notification_manager, task, pr_info)
        else:
            # auto: merge PR after review comment, then mark task completed
            if pr_info and pr_info.number:
                pr_merged = merge_pr(project_root, pr_info.number)
                if pr_merged:
                    audit_log.log(
                        "pr_merged",
                        task_id=task_id,
                        pr_number=pr_info.number,
                    )
                else:
                    audit_log.log(
                        "pr_merge_failed",
                        task_id=task_id,
                        pr_number=pr_info.number,
                    )

            mutation_log.append(
                "task_completed",
                {
                    "findings": findings,
                    "proof_of_work": {"output_preview": result.output[:1000]},
                    "pr_url": pr_info.url if pr_info else None,
                    "pr_number": pr_info.number if pr_info else None,
                    "pr_merged": pr_merged,
                },
                reason="Validated by processor",
                task_id=task_id,
            )
            audit_log.log("task_completed", task_id=task_id, attempt=attempt)
    else:
        failed_details = [d for passed, d in details if not passed]

        if attempt > max_retries:
            # Retries exhausted — escalate
            error_msg = f"Validation failed: {'; '.join(failed_details[:3])}"
            _escalate_task(
                task,
                attempt,
                error_msg,
                mutation_log,
                audit_log,
                session_logger,
                notification_manager=notification_manager,
            )
        else:
            mutation_log.append(
                "task_failed",
                {
                    "attempt": attempt,
                    "findings": failed_details,
                    "attempt_count": attempt,
                },
                reason="Validation failed",
                task_id=task_id,
            )
            audit_log.log("task_validation_failed", task_id=task_id, attempt=attempt)
            # Notify on task failure
            if notification_manager:
                error_msg = f"Validation failed: {'; '.join(failed_details[:3])}"
                notify_task_failure(notification_manager, task, attempt, error_msg)

    # Record outcome for adaptive retry tracking
    if adaptive_tracker is not None:
        task_type = task.get("task_type", task.get("type", "general"))
        role = task.get("role", "unknown")
        adaptive_tracker.record_outcome(
            TaskOutcome(
                task_type=task_type,
                role=role,
                attempt=attempt,
                success=all_passed,
                task_id=task_id,
            )
        )

    # Refresh state so subsequent reads see the update
    state.refresh()

    return ProcessResult(
        task_id=task_id,
        passed=all_passed,
        details=details,
        findings=findings,
        pr_merged=pr_merged,
        pr_commented=pr_commented,
    )


def _escalate_task(
    task: dict,
    attempt: int,
    error: str,
    mutation_log: MutationLog,
    audit_log: AuditLog,
    session_logger: SessionLogger,
    notification_manager: NotificationManager | None = None,
) -> None:
    """Mark a task as escalated and create an escalation record.

    Called when max_retries are exhausted. Sets task status to 'escalated'
    and creates a structured escalation for operator review. Sends
    notifications via notification_manager if provided.
    """
    task_id = task["id"]

    # Mark task as escalated
    mutation_log.append(
        "task_escalated",
        {"attempt": attempt, "attempt_count": attempt},
        reason=f"Max retries exhausted after {attempt} attempts",
        task_id=task_id,
    )
    audit_log.log(
        "task_escalated",
        task_id=task_id,
        attempt=attempt,
    )

    # Create escalation record for operator
    escalation = create_escalation(
        task=task,
        attempt=attempt,
        error=error,
        session_logger=session_logger,
        mutation_log=mutation_log,
    )
    audit_log.log(
        "escalation",
        task_id=task_id,
        escalation_id=escalation["escalation_id"],
        attempts=attempt,
    )

    # Send escalation notification
    if notification_manager:
        notify_escalation(notification_manager, task, escalation)


def _parse_done_when(done_when: str) -> list[dict]:
    """Try to parse done_when as JSON validation rules.

    Returns a list of rule dicts/strings if parseable, empty list otherwise.
    A plain text done_when (not valid JSON) returns empty — the processor
    will treat agent exit code 0 as success.
    """
    if not done_when:
        return []
    try:
        parsed = json.loads(done_when)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, (dict, str)):
            return [parsed]
        return []
    except (json.JSONDecodeError, TypeError):
        return []


def _extract_findings(output: str) -> list[str]:
    """Extract findings from agent output.

    Looks for lines prefixed with 'FINDING:' in the output.
    """
    findings = []
    for line in output.split("\n"):
        stripped = line.strip()
        if stripped.upper().startswith("FINDING:"):
            findings.append(stripped[8:].strip())
    return findings
