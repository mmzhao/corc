"""Daemon restart recovery — reconcile work state on startup.

On startup, the daemon must:
1. Rebuild SQLite from mutation log (ensure consistency)
2. Check all tasks marked 'running' or 'assigned' for PID liveness
3. Scan for running claude processes whose command line contains known task IDs
4. Kill orphaned agents that have exceeded agent_timeout_s
5. Return alive agent info so the daemon can re-attach monitoring
6. Process output for dead agents that produced results
7. Mark dead agents with no output as failed (retry policy applies)
8. Clean up stale git worktrees
"""

import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from corc.audit import AuditLog
from corc.config import DEFAULTS
from corc.dispatch import AgentResult, kill_agent_process
from corc.mutations import MutationLog
from corc.pr import create_pr, get_worktree_branch, push_branch
from corc.processor import process_completed

logger = logging.getLogger(__name__)
from corc.sessions import SessionLogger
from corc.state import WorkState


def reconcile_on_startup(
    state: WorkState,
    mutation_log: MutationLog,
    audit_log: AuditLog,
    session_logger: SessionLogger,
    project_root: Path,
    pid_checker=None,
    agent_timeout_s: float | None = None,
) -> dict:
    """Run full reconciliation on daemon startup.

    Rebuilds SQLite from the mutation log, checks all running/assigned tasks
    for PID liveness, processes dead agent output or marks tasks failed, and
    cleans stale worktrees.

    Orphaned agents whose PID is still alive but that have exceeded
    ``agent_timeout_s`` are killed and treated as timed-out failures.

    Args:
        state: The work state (SQLite materialized view).
        mutation_log: The mutation log (source of truth).
        audit_log: For logging reconciliation events.
        session_logger: For reading agent session output.
        project_root: Project root directory.
        pid_checker: Optional callable(pid) -> bool. Returns True if PID is
            alive and is a claude process. Defaults to is_pid_alive() &&
            is_claude_process(). Inject for testing.
        agent_timeout_s: Maximum allowed agent runtime in seconds. Agents
            alive longer than this are killed. Defaults to config value.

    Returns a summary dict with counts of what was reconciled.
    """
    if pid_checker is None:
        pid_checker = _default_pid_checker

    if agent_timeout_s is None:
        agent_timeout_s = DEFAULTS["dispatch"]["agent_timeout_s"]

    summary = {
        "rebuilt_state": False,
        "running_found": 0,
        "assigned_found": 0,
        "agents_alive": 0,
        "agents_dead_with_output": 0,
        "agents_dead_no_output": 0,
        "agents_killed_timeout": 0,
        "worktrees_cleaned": 0,
        "prs_created": 0,
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
            # Agent is alive — check if it has exceeded the timeout.
            # Use the task's 'updated' timestamp (set when task_started fires)
            # to determine how long the agent has been running.
            agent_age = _get_agent_age(task, mutation_log)
            if agent_age is not None and agent_age > agent_timeout_s:
                # Orphaned agent exceeded timeout — kill it
                killed = kill_agent_process(pid)
                summary["agents_killed_timeout"] += 1
                audit_log.log(
                    "reconcile_agent_killed_timeout",
                    task_id=task_id,
                    pid=pid,
                    age_s=round(agent_age, 1),
                    timeout_s=agent_timeout_s,
                    killed=killed,
                )
                # Fall through to dead-agent handling below
            else:
                # Agent still alive and within timeout — collect info for re-attachment
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
                continue

        # Dead or killed agent — check for output in session logs
        output = _get_last_agent_output(session_logger, task_id)

        if output is not None:
            # Process the output through normal validation pipeline
            summary["agents_dead_with_output"] += 1
            attempt = session_logger.get_latest_attempt(task_id)

            # For successful agents with worktrees that have commits ahead
            # of main, create a PR before processing completion.
            pr_info = None
            if output.exit_code == 0:
                worktree_path, branch_name = _get_worktree_info_for_task(
                    state, task_id, project_root
                )
                if (
                    worktree_path is not None
                    and branch_name is not None
                    and _branch_has_commits_ahead(project_root, branch_name)
                ):
                    push_ok, push_err = push_branch(project_root, branch_name)
                    if push_ok:
                        pr_info, pr_err = create_pr(project_root, branch_name, task)
                        if pr_info is None:
                            # PR creation failed — mark as failed (retriable)
                            mutation_log.append(
                                "task_failed",
                                {
                                    "attempt": attempt,
                                    "attempt_count": attempt,
                                    "exit_code": 0,
                                    "reconciled": True,
                                    "infrastructure": True,
                                },
                                reason=f"Reconciliation: PR creation failed: {pr_err}",
                                task_id=task_id,
                            )
                            audit_log.log(
                                "reconcile_pr_creation_failed",
                                task_id=task_id,
                                error=pr_err,
                            )
                            continue
                        else:
                            summary["prs_created"] += 1
                            audit_log.log(
                                "reconcile_pr_created",
                                task_id=task_id,
                                pr_url=pr_info.url,
                                pr_number=pr_info.number,
                            )
                    else:
                        # Push failed — mark as failed (retriable)
                        mutation_log.append(
                            "task_failed",
                            {
                                "attempt": attempt,
                                "attempt_count": attempt,
                                "exit_code": 0,
                                "reconciled": True,
                                "infrastructure": True,
                            },
                            reason=f"Reconciliation: branch push failed: {push_err}",
                            task_id=task_id,
                        )
                        audit_log.log(
                            "reconcile_push_failed",
                            task_id=task_id,
                            error=push_err,
                        )
                        continue

            process_completed(
                task=task,
                result=output,
                attempt=attempt,
                mutation_log=mutation_log,
                state=state,
                audit_log=audit_log,
                session_logger=session_logger,
                project_root=project_root,
                pr_info=pr_info,
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


def _get_agent_age(task: dict, mutation_log: MutationLog) -> float | None:
    """Compute how long an agent has been running, in seconds.

    Uses the task's 'updated' timestamp (set when ``task_started`` fires)
    to determine the start time. Falls back to scanning the mutation log
    for the most recent ``task_started`` entry for the task.

    Returns None if the start time cannot be determined.
    """
    # Try the task's 'updated' field first (set by task_started mutation)
    updated = task.get("updated")
    if updated:
        try:
            start_dt = datetime.fromisoformat(updated)
            return time.time() - start_dt.timestamp()
        except (ValueError, TypeError):
            pass

    # Fallback: scan mutation log for task_started events
    task_id = task.get("id")
    if not task_id:
        return None

    entries = mutation_log.read_all()
    for entry in reversed(entries):
        if entry.get("type") == "task_started" and entry.get("task_id") == task_id:
            try:
                ts = datetime.fromisoformat(entry["ts"])
                return time.time() - ts.timestamp()
            except (ValueError, TypeError, KeyError):
                pass

    return None


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


def _get_worktree_info_for_task(
    state: WorkState, task_id: str, project_root: Path
) -> tuple[Path | None, str | None]:
    """Get worktree path and branch name for a task's agent.

    Looks up the agent record for the given task and checks if it has
    a worktree_path that still exists on disk. If so, resolves the
    branch name from git.

    Returns:
        Tuple of (worktree_path, branch_name). Both are None if no
        valid worktree exists for the task.
    """
    agents = state.get_agents_for_task(task_id)
    for agent in agents:
        worktree_path_str = agent.get("worktree_path")
        if worktree_path_str:
            wt_path = Path(worktree_path_str)
            if wt_path.exists():
                branch = get_worktree_branch(project_root, wt_path)
                if branch:
                    return wt_path, branch
    return None, None


def _branch_has_commits_ahead(project_root: Path, branch_name: str) -> bool:
    """Check if a branch has commits ahead of HEAD (main).

    Runs ``git log HEAD..<branch_name> --oneline`` to see if there are
    any commits on the branch that are not on the current HEAD.

    Returns:
        True if the branch has at least one commit ahead of HEAD.
    """
    try:
        result = subprocess.run(
            ["git", "log", "HEAD.." + branch_name, "--oneline"],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=30,
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
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
