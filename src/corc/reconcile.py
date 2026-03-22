"""Daemon restart recovery — reconcile work state on startup.

On startup, the daemon must:
1. Rebuild SQLite from mutation log (ensure consistency)
2. Check all tasks marked 'running' or 'assigned' for PID liveness
3. Process output for dead agents that produced results
4. Mark dead agents with no output as failed (retry policy applies)
5. Clean up stale git worktrees
"""

import os
import subprocess
from pathlib import Path

from corc.audit import AuditLog
from corc.dispatch import AgentResult
from corc.mutations import MutationLog
from corc.processor import process_completed
from corc.sessions import SessionLogger
from corc.state import WorkState


def reconcile_on_startup(
    state: WorkState,
    mutation_log: MutationLog,
    audit_log: AuditLog,
    session_logger: SessionLogger,
    project_root: Path,
    pid_checker=None,
) -> dict:
    """Run full reconciliation on daemon startup.

    Rebuilds SQLite from the mutation log, checks all running/assigned tasks
    for PID liveness, processes dead agent output or marks tasks failed, and
    cleans stale worktrees.

    Args:
        state: The work state (SQLite materialized view).
        mutation_log: The mutation log (source of truth).
        audit_log: For logging reconciliation events.
        session_logger: For reading agent session output.
        project_root: Project root directory.
        pid_checker: Optional callable(pid) -> bool. Returns True if PID is
            alive and is a claude process. Defaults to is_pid_alive() &&
            is_claude_process(). Inject for testing.

    Returns a summary dict with counts of what was reconciled.
    """
    if pid_checker is None:
        pid_checker = _default_pid_checker

    summary = {
        "rebuilt_state": False,
        "running_found": 0,
        "assigned_found": 0,
        "agents_alive": 0,
        "agents_dead_with_output": 0,
        "agents_dead_no_output": 0,
        "worktrees_cleaned": 0,
    }

    # 1. Rebuild SQLite from mutation log
    state.rebuild()
    summary["rebuilt_state"] = True
    audit_log.log("reconcile_state_rebuilt")

    # 2. Find tasks marked as running or assigned
    running_tasks = state.list_tasks(status="running")
    assigned_tasks = state.list_tasks(status="assigned")
    summary["running_found"] = len(running_tasks)
    summary["assigned_found"] = len(assigned_tasks)

    stale_tasks = running_tasks + assigned_tasks
    if not stale_tasks:
        # 3. Still clean worktrees even if no stale tasks
        summary["worktrees_cleaned"] = clean_stale_worktrees(state, project_root)
        audit_log.log("reconcile_complete", **summary)
        return summary

    # 3. For each stale task, check agent liveness
    for task in stale_tasks:
        task_id = task["id"]
        pid = _get_agent_pid(state, task_id)

        if pid and pid_checker(pid):
            # Agent still alive — leave it running
            summary["agents_alive"] += 1
            audit_log.log("reconcile_agent_alive", task_id=task_id, pid=pid)
        else:
            # Agent is dead — check for output in session logs
            output = _get_last_agent_output(session_logger, task_id)

            if output is not None:
                # Process the output through normal validation pipeline
                summary["agents_dead_with_output"] += 1
                attempt = session_logger.get_latest_attempt(task_id)
                process_completed(
                    task=task,
                    result=output,
                    attempt=attempt,
                    mutation_log=mutation_log,
                    state=state,
                    audit_log=audit_log,
                    session_logger=session_logger,
                    project_root=project_root,
                )
                audit_log.log("reconcile_processed_output", task_id=task_id)
            else:
                # No output — mark as failed so scheduler retry can kick in
                summary["agents_dead_no_output"] += 1
                attempt = session_logger.get_latest_attempt(task_id) or 1
                mutation_log.append(
                    "task_failed",
                    {
                        "attempt": attempt,
                        "attempt_count": attempt,
                        "exit_code": -1,
                        "reconciled": True,
                    },
                    reason="Reconciliation: agent process died without producing output",
                    task_id=task_id,
                )
                audit_log.log("reconcile_marked_failed", task_id=task_id)

    # 4. Clean up stale worktrees
    summary["worktrees_cleaned"] = clean_stale_worktrees(state, project_root)

    # Refresh state after all mutations
    state.refresh()

    audit_log.log("reconcile_complete", **summary)
    return summary


def _default_pid_checker(pid: int) -> bool:
    """Check if PID is alive AND is a claude process."""
    return is_pid_alive(pid) and is_claude_process(pid)


def _get_agent_pid(state: WorkState, task_id: str) -> int | None:
    """Get the PID of the agent assigned to a task."""
    agents = state.get_agents_for_task(task_id)
    for agent in agents:
        pid = agent.get("pid")
        if pid:
            return pid
    return None


def _get_last_agent_output(session_logger: SessionLogger, task_id: str) -> AgentResult | None:
    """Check if there's recorded output from the last agent session.

    Returns an AgentResult if output was found in session logs, None otherwise.
    """
    attempt = session_logger.get_latest_attempt(task_id)
    if attempt == 0:
        return None

    session = session_logger.read_session(task_id, attempt)
    for entry in session:
        if entry.get("type") == "output":
            return AgentResult(
                output=entry.get("content", ""),
                exit_code=entry.get("exit_code", -1),
                duration_s=entry.get("duration_s", 0),
            )
    return None


def is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID exists.

    Uses signal 0 which checks for process existence without actually
    sending a signal.
    """
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't have permission to signal it
        return True
    except OSError:
        return False


def is_claude_process(pid: int) -> bool:
    """Check if the process with given PID is a claude-related process.

    Checks the process command name for 'claude' or 'node' (since claude
    CLI runs on Node.js).
    """
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "comm="],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False
        comm = result.stdout.strip().lower()
        return "claude" in comm or "node" in comm
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def clean_stale_worktrees(state: WorkState, project_root: Path) -> int:
    """Remove git worktrees for dead agents.

    Finds agents with worktree_path set, checks if the agent process
    is dead, and removes the worktree if so.

    Returns the number of worktrees cleaned.
    """
    cleaned = 0
    agents = state.list_agents()

    for agent in agents:
        worktree_path = agent.get("worktree_path")
        if not worktree_path:
            continue

        worktree = Path(worktree_path)
        if not worktree.exists():
            continue

        pid = agent.get("pid")
        if pid and is_pid_alive(pid):
            continue  # Agent still alive, don't clean worktree

        # Agent is dead and worktree exists — clean up
        try:
            result = subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                capture_output=True,
                timeout=30,
                cwd=str(project_root),
            )
            if result.returncode == 0:
                cleaned += 1
            else:
                # git worktree remove failed — try manual cleanup
                _remove_dir(worktree)
                cleaned += 1
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            # If git worktree remove fails, try manual cleanup
            _remove_dir(worktree)
            cleaned += 1

    return cleaned


def _remove_dir(path: Path):
    """Remove a directory tree, ignoring errors."""
    import shutil
    try:
        shutil.rmtree(str(path), ignore_errors=True)
    except Exception:
        pass
