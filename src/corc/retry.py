"""Retry policies and structured escalation.

Automatic retry with enriched context (previous session log injected).
Escalation records created when retries are exhausted.
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from corc.mutations import MutationLog
from corc.sessions import SessionLogger


@dataclass
class RetryPolicy:
    """Configurable retry policy for task dispatch."""
    max_retries: int = 2  # Default: 2 retries (so up to 3 total attempts)

    def should_retry(self, attempt: int) -> bool:
        """Check if the task should be retried given the current attempt number."""
        return attempt < self.max_retries + 1  # attempt 1 is original, retries are 2..max_retries+1

    def retries_exhausted(self, attempt: int) -> bool:
        """Check if all retries have been used."""
        return attempt >= self.max_retries + 1


def get_retry_context(task_id: str, attempt: int, session_logger: SessionLogger) -> str:
    """Build enriched context from previous attempt's session log.

    Reads the session log from the failed attempt and formats it as
    context for the retry attempt. Includes the dispatch prompt, agent
    output, and any error information.
    """
    prev_attempt = attempt - 1
    if prev_attempt < 1:
        return ""

    entries = session_logger.read_session(task_id, prev_attempt)
    if not entries:
        return ""

    parts = [
        f"\n=== PREVIOUS ATTEMPT ({prev_attempt}) SESSION LOG ===",
        f"This is retry attempt {attempt}. The previous attempt failed.",
        "Review what went wrong below and take a different approach.\n",
    ]

    for entry in entries:
        entry_type = entry.get("type", "unknown")
        content = entry.get("content", "")

        if entry_type == "dispatch":
            parts.append(f"--- Previous Prompt ---")
            # Truncate if very long
            if len(content) > 2000:
                content = content[:2000] + "\n[... truncated ...]"
            parts.append(content)

        elif entry_type == "output":
            exit_code = entry.get("exit_code", "unknown")
            duration = entry.get("duration_s", 0)
            parts.append(f"\n--- Previous Output (exit_code={exit_code}, duration={duration:.1f}s) ---")
            # Truncate output to keep context budget manageable
            if len(content) > 5000:
                content = content[:2500] + "\n[... middle truncated ...]\n" + content[-2500:]
            parts.append(content)

        elif entry_type == "validation":
            passed = entry.get("passed", False)
            parts.append(f"\n--- Validation (passed={passed}) ---")
            parts.append(content)

    parts.append("\n=== END PREVIOUS SESSION LOG ===\n")
    return "\n".join(parts)


def create_escalation(
    task: dict,
    attempt: int,
    error: str,
    session_logger: SessionLogger,
    mutation_log: MutationLog,
) -> dict:
    """Create an escalation record when retries are exhausted.

    Writes an escalation_created mutation to the log with structured
    context and suggested actions for the operator.

    Returns the escalation data dict.
    """
    task_id = task["id"]
    escalation_id = f"esc-{uuid.uuid4().hex[:8]}"
    session_log_path = str(session_logger.session_path(task_id, attempt))

    # Generate suggested actions based on the failure context
    suggested_actions = _suggest_actions(task, error)

    escalation_data = {
        "escalation_id": escalation_id,
        "task_id": task_id,
        "task_name": task.get("name", "unknown"),
        "error": error,
        "attempts": attempt,
        "session_log_path": session_log_path,
        "suggested_actions": suggested_actions,
        "done_when": task.get("done_when", ""),
    }

    mutation_log.append(
        "escalation_created",
        escalation_data,
        reason=f"Retries exhausted after {attempt} attempts",
        task_id=task_id,
    )

    return escalation_data


def resolve_escalation(
    escalation_id: str,
    mutation_log: MutationLog,
    resolution: str = "",
) -> dict:
    """Resolve an escalation, marking it as handled.

    Optionally resets the task back to pending so it can be re-dispatched.
    """
    data = {
        "escalation_id": escalation_id,
        "resolution": resolution,
    }

    entry = mutation_log.append(
        "escalation_resolved",
        data,
        reason=f"Escalation resolved: {resolution or 'no details'}",
    )

    return data


def _suggest_actions(task: dict, error: str) -> list[str]:
    """Generate suggested actions based on failure context."""
    actions = []

    # Always suggest reviewing the session log
    actions.append("Review the session log for detailed error context")

    # Suggest based on error patterns
    error_lower = error.lower() if error else ""

    if "timeout" in error_lower:
        actions.append("Increase timeout or reduce task scope")
    elif "exit code" in error_lower or "exit_code" in error_lower:
        actions.append("Check agent exit code and stderr output")

    if "validation" in error_lower:
        actions.append("Review done_when criteria — may need adjustment")

    # Generic actions
    actions.append("Consider breaking the task into smaller subtasks")
    actions.append("Manually retry with adjusted constraints or context")

    return actions
