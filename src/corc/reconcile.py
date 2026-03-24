"""Daemon restart recovery — reconcile work state on startup.

On startup, the daemon must:
1. Rebuild SQLite from mutation log (ensure consistency)
2. Check all tasks marked 'running' or 'assigned' for PID liveness
3. Scan for running claude processes whose command line contains known task IDs
4. Return alive agent info so the daemon can re-attach monitoring
5. Process output for dead agents that produced results
6. Mark dead agents with no output as failed (retry policy applies)
7. Clean up stale git worktrees
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
        "alive_agents": [],  # Info for daemon to re-attach monitoring
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
        _log_reconcile_complete(audit_log, summary)
        return summary

    # Scan for running claude processes as fallback for missing PIDs.
    # If a task has no PID recorded in its agent record (e.g. old dispatch
    # without PID capture), we can still find the process by scanning ps
    # for claude -p commands whose command line contains the task ID.
    stale_task_ids = {task["id"] for task in stale_tasks}
    scanned_pids = scan_claude_processes(stale_task_ids)

    # 3. For each stale task, check agent liveness
    for task in stale_tasks:
        task_id = task["id"]
        pid = _get_agent_pid(state, task_id)

        # Fallback: check process scan results if no PID in agent record
        if not pid and task_id in scanned_pids:
            pid = scanned_pids[task_id]
            audit_log.log("reconcile_pid_from_scan", task_id=task_id, pid=pid)

        if pid and pid_checker(pid):
            # Agent still alive — collect info for daemon re-attachment
            summary["agents_alive"] += 1
            agents = state.get_agents_for_task(task_id)
            agent = agents[0] if agents else {}
            attempt = session_logger.get_latest_attempt(task_id) or 1
            summary["alive_agents"].append(
                {
                    "task": task,
                    "pid": pid,
                    "attempt": attempt,
                    "worktree_path": agent.get("worktree_path"),
                    "agent_id": agent.get("id"),
                }
            )
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
                        "infrastructure": True,
                    },
                    reason="Reconciliation: agent process died without producing output",
                    task_id=task_id,
                )
                audit_log.log("reconcile_marked_failed", task_id=task_id)

    # 4. Clean up stale worktrees
    summary["worktrees_cleaned"] = clean_stale_worktrees(state, project_root)

    # Refresh state after all mutations
    state.refresh()

    _log_reconcile_complete(audit_log, summary)
    return summary


def _log_reconcile_complete(audit_log: AuditLog, summary: dict):
    """Log reconcile_complete without the alive_agents list (not serializable)."""
    log_summary = {k: v for k, v in summary.items() if k != "alive_agents"}
    audit_log.log("reconcile_complete", **log_summary)


def scan_claude_processes(task_ids: set[str]) -> dict[str, int]:
    """Scan running processes for claude -p commands matching known task IDs.

    Checks ``ps`` output for processes containing 'claude' and '-p' whose
    command line also contains a recognized task_id.  Used as a fallback
    when the agent's PID wasn't recorded in the mutation log.

    Returns mapping of task_id → PID for each match found.
    """
    if not task_ids:
        return {}

    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,args"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {}
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return {}

    found: dict[str, int] = {}
    for line in result.stdout.splitlines()[1:]:  # Skip header
        line = line.strip()
        if not line:
            continue
        if "claude" not in line.lower() or "-p" not in line:
            continue
        # Parse PID from first field
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        # Check if any task_id appears in the command line
        cmd_line = parts[1]
        for task_id in task_ids:
            if task_id in cmd_line:
                found[task_id] = pid
                break

    return found


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


def _get_last_agent_output(
    session_logger: SessionLogger, task_id: str
) -> AgentResult | None:
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
    """Remove git worktrees that are no longer needed.

    Three-pass cleanup:

    1. **Agent-referenced worktrees** — for each agent with a ``worktree_path``,
       remove the worktree if the agent's task is in a terminal state
       (completed / failed / escalated) *or* the agent process is dead.
    2. **Orphaned filesystem worktrees** — scan ``.claude/worktrees/`` for
       directories not associated with any actively-running task's agent.
    3. **Git worktree prune** — ask git to clean up stale internal references.

    Returns the number of worktrees cleaned.
    """
    # Terminal task statuses — worktrees for these tasks are always stale
    _TERMINAL_STATUSES = {"completed", "failed", "escalated", "cancelled"}

    cleaned = 0
    agents = state.list_agents()
    # Track worktree paths that are actively in use (running task, alive agent)
    active_worktree_paths: set[str] = set()

    # --- Pass 1: Agent-referenced worktrees ---
    for agent in agents:
        worktree_path = agent.get("worktree_path")
        if not worktree_path:
            continue

        worktree = Path(worktree_path)

        # Check task status — terminal tasks always get cleaned
        task_id = agent.get("task_id")
        task = state.get_task(task_id) if task_id else None
        task_is_terminal = task is not None and task["status"] in _TERMINAL_STATUSES

        if not worktree.exists():
            continue

        pid = agent.get("pid")
        agent_alive = pid is not None and is_pid_alive(pid)

        # Keep the worktree only if the task is non-terminal AND agent is alive
        if not task_is_terminal and agent_alive:
            active_worktree_paths.add(str(worktree.resolve()))
            continue

        # Stale — remove
        cleaned += _remove_worktree_dir(worktree, project_root)

    # --- Pass 2: Orphaned filesystem worktrees ---
    worktrees_dir = project_root / ".claude" / "worktrees"
    if worktrees_dir.exists():
        for child in worktrees_dir.iterdir():
            if not child.is_dir():
                continue
            if str(child.resolve()) in active_worktree_paths:
                continue
            # Not tracked by any active agent — orphan
            cleaned += _remove_worktree_dir(child, project_root)

    # --- Pass 3: Git worktree prune ---
    _git_worktree_prune(project_root)

    return cleaned


def _remove_worktree_dir(worktree: Path, project_root: Path) -> int:
    """Remove a single worktree directory. Returns 1 on success, 0 on skip."""
    if not worktree.exists():
        return 0
    try:
        result = subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            capture_output=True,
            timeout=30,
            cwd=str(project_root),
        )
        if result.returncode == 0:
            return 1
        # git worktree remove failed — try manual cleanup
        _remove_dir(worktree)
        return 1
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        _remove_dir(worktree)
        return 1


def _git_worktree_prune(project_root: Path):
    """Run ``git worktree prune`` to clean stale internal references."""
    try:
        subprocess.run(
            ["git", "worktree", "prune"],
            capture_output=True,
            cwd=str(project_root),
            timeout=30,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass


def _remove_dir(path: Path):
    """Remove a directory tree, ignoring errors."""
    import shutil

    try:
        shutil.rmtree(str(path), ignore_errors=True)
    except Exception:
        pass
