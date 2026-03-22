"""Processor module — validates output, updates state, collects findings.

Reads work state + agent output, writes task completions.
Can be called standalone: `corc process TASK_ID`
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from corc.audit import AuditLog
from corc.dispatch import AgentResult
from corc.mutations import MutationLog
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


def process_completed(
    task: dict,
    result: AgentResult,
    attempt: int,
    mutation_log: MutationLog,
    state: WorkState,
    audit_log: AuditLog,
    session_logger: SessionLogger,
    project_root: Path,
) -> ProcessResult:
    """Validate task output and update state.

    If the agent exited with error, marks task as failed or escalated.
    Otherwise, runs done_when validations. If all pass, marks completed.
    If validations fail, marks task as failed or escalated.

    After max_retries are exhausted, sets status to 'escalated' instead
    of 'failed' and creates an escalation record.
    """
    task_id = task["id"]
    max_retries = task.get("max_retries", 3)

    # Guard: skip if already completed (prevents duplicate completion mutations)
    state.refresh()
    current = state.get_task(task_id)
    if current and current["status"] == "completed":
        audit_log.log("task_completion_skipped", task_id=task_id,
                      reason="already completed")
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
                task, attempt, error_detail, mutation_log, audit_log, session_logger,
            )
        else:
            # Mark as failed (retriable — scheduler will pick it up)
            mutation_log.append(
                "task_failed",
                {"attempt": attempt, "exit_code": result.exit_code, "attempt_count": attempt},
                reason=error_msg,
                task_id=task_id,
            )
            audit_log.log(
                "task_failed", task_id=task_id,
                attempt=attempt, exit_code=result.exit_code,
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
        task_id, attempt, all_passed,
        json.dumps([d for _, d in details]),
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
                },
                reason="Validation passed; merge policy is human-only — awaiting human merge",
                task_id=task_id,
            )
            audit_log.log(
                "task_pending_merge",
                task_id=task_id,
                attempt=attempt,
                merge_policy="human-only",
            )
        else:
            # auto: mark task as completed (merge already happened in executor)
            mutation_log.append(
                "task_completed",
                {
                    "findings": findings,
                    "proof_of_work": {"output_preview": result.output[:1000]},
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
                task, attempt, error_msg, mutation_log, audit_log, session_logger,
            )
        else:
            mutation_log.append(
                "task_failed",
                {"attempt": attempt, "findings": failed_details, "attempt_count": attempt},
                reason="Validation failed",
                task_id=task_id,
            )
            audit_log.log("task_validation_failed", task_id=task_id, attempt=attempt)

    # Refresh state so subsequent reads see the update
    state.refresh()

    return ProcessResult(
        task_id=task_id,
        passed=all_passed,
        details=details,
        findings=findings,
    )


def _escalate_task(
    task: dict,
    attempt: int,
    error: str,
    mutation_log: MutationLog,
    audit_log: AuditLog,
    session_logger: SessionLogger,
) -> None:
    """Mark a task as escalated and create an escalation record.

    Called when max_retries are exhausted. Sets task status to 'escalated'
    and creates a structured escalation for operator review.
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
        "task_escalated", task_id=task_id, attempt=attempt,
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
