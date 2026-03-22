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

    If the agent exited with error, marks task as failed.
    Otherwise, runs done_when validations. If all pass, marks completed.
    If validations fail, marks task as failed.
    """
    task_id = task["id"]

    # Agent crashed or errored
    if result.exit_code != 0:
        mutation_log.append(
            "task_failed",
            {"attempt": attempt, "exit_code": result.exit_code},
            reason=f"Agent exited with code {result.exit_code}",
            task_id=task_id,
        )
        audit_log.log(
            "task_failed", task_id=task_id,
            attempt=attempt, exit_code=result.exit_code,
        )
        return ProcessResult(
            task_id=task_id,
            passed=False,
            details=[(False, f"Agent exited with code {result.exit_code}")],
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
        mutation_log.append(
            "task_failed",
            {"attempt": attempt, "findings": failed_details},
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
