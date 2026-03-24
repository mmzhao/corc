"""CORC CLI — essential commands for Phase 0."""

import json
import sys
import time
import uuid
from pathlib import Path

import click

from corc.config import get_paths, load_config, _parse_value
from corc.mutations import MutationLog
from corc.state import WorkState
from corc.audit import AuditLog
from corc.sessions import SessionLogger
from corc.knowledge import KnowledgeStore
from corc.pause import write_pause_lock, remove_pause_lock, read_pause_lock, is_paused
from corc.dag import render_ascii_dag, render_mermaid
from corc.context import (
    assemble_context,
    record_context_mtimes,
    validate_context_bundle_paths,
)
from corc.daemon import Daemon, stop_daemon
from corc.dispatch import get_dispatcher, Constraints
from corc.roles import RoleLoader, constraints_from_role, get_system_prompt_for_role
from corc.worktree import (
    WorktreeError,
    create_worktree,
    merge_worktree,
    remove_worktree,
)
from corc.validate import run_validations
from corc.templates import get_template, render_template, list_types
from corc.lint_done_when import lint_done_when
from corc.retry import resolve_escalation


def _get_all():
    paths = get_paths()
    ml = MutationLog(paths["mutations"])
    ws = WorkState(paths["state_db"], ml)
    al = AuditLog(paths["events_dir"])
    sl = SessionLogger(paths["sessions_dir"])
    ks = KnowledgeStore(paths["knowledge_dir"], paths["knowledge_db"])
    return paths, ml, ws, al, sl, ks


@click.group()
def cli():
    """CORC — Claude Orchestration System"""
    pass


# --- Task commands ---


@cli.group("config")
def config_cmd():
    """View and update configuration."""
    pass


@config_cmd.command("show")
@click.option("--key", default=None, help="Show a specific config key (dot notation)")
def config_show(key):
    """Display the current configuration.

    Shows the full merged config (defaults + overrides from .corc/config.yaml).
    Use --key to show a specific setting, e.g. --key dispatch.agent_timeout_s
    """
    import yaml as _yaml

    cfg = load_config()
    if key:
        value = cfg.get(key)
        if value is None:
            click.echo(f"Key '{key}' not found in config.")
            sys.exit(1)
        if isinstance(value, dict):
            click.echo(
                _yaml.safe_dump(
                    {key: value}, default_flow_style=False, sort_keys=False
                ).rstrip()
            )
        else:
            click.echo(f"{key}: {value}")
    else:
        click.echo(
            _yaml.safe_dump(
                cfg.as_dict(), default_flow_style=False, sort_keys=False
            ).rstrip()
        )


@config_cmd.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key, value):
    """Update a config value. KEY uses dot notation.

    Examples:
      corc config set dispatch.agent_timeout_s 3600
      corc config set retry.default_retries 3
      corc config set daemon.poll_interval 10.0
      corc config set alerts.cost.daily_limit_usd 100.0
    """
    cfg = load_config()
    parsed_value = _parse_value(value)
    cfg.set(key, parsed_value)
    saved_path = cfg.save()
    click.echo(f"Set {key} = {parsed_value!r}")
    click.echo(f"Saved to {saved_path}")


# --- Task commands ---


@cli.group()
def task():
    """Manage tasks."""
    pass


@task.command("create")
@click.argument("name")
@click.option("--done-when", required=True, help="Testable completion criteria")
@click.option("--description", default="", help="Task description")
@click.option("--role", default="implementer", help="Agent role")
@click.option("--depends-on", default="", help="Comma-separated task IDs")
@click.option(
    "--context",
    "context_bundle",
    default="",
    help="Comma-separated file paths for context bundle",
)
@click.option("--checklist", default="", help="Comma-separated checklist items")
@click.option("--strict", is_flag=True, help="Reject subjective done_when criteria")
@click.option(
    "--priority",
    default=100,
    type=int,
    help="Task priority (lower=higher priority, default 100)",
)
@click.option(
    "--type",
    "task_type",
    default="implementation",
    type=click.Choice(
        ["implementation", "investigation", "bugfix"], case_sensitive=False
    ),
    help="Task type (default: implementation)",
)
@click.option(
    "--draft",
    is_flag=True,
    help="Create task in draft status (not scheduled until approved)",
)
@click.option(
    "--target-repo",
    default=None,
    help="Registered repo name for cross-repo dispatch (default: corc repo)",
)
def task_create(
    name,
    done_when,
    description,
    role,
    depends_on,
    context_bundle,
    checklist,
    strict,
    priority,
    task_type,
    draft,
    target_repo,
):
    """Create a new task."""
    # Lint done_when criteria (with type-specific rules)
    lint_result = lint_done_when(done_when, task_type=task_type)
    if lint_result.warnings:
        for warning in lint_result.warnings:
            click.echo(f"Warning: done_when: {warning}", err=True)
        if strict:
            click.echo(
                "Aborted: --strict rejects subjective done_when criteria.", err=True
            )
            sys.exit(1)

    paths, ml, ws, al, sl, ks = _get_all()
    task_id = str(uuid.uuid4())[:8]
    deps = [d.strip() for d in depends_on.split(",") if d.strip()]
    bundle = [b.strip() for b in context_bundle.split(",") if b.strip()]
    cl = [c.strip() for c in checklist.split(",") if c.strip()]

    # Validate target_repo exists via RepoManager
    project_root = paths["root"] if isinstance(paths, dict) else paths.root
    context_root = project_root  # default: resolve context against corc repo
    if target_repo:
        from corc.repo import RepoManager, RepoNotFoundError

        cfg = load_config()
        mgr = RepoManager(cfg)
        try:
            repo_config = mgr.get(target_repo)
            target_repo_path = Path(repo_config["path"])
            if not target_repo_path.exists():
                click.echo(
                    f"Error: target repo '{target_repo}' path does not exist: {target_repo_path}",
                    err=True,
                )
                sys.exit(1)
            context_root = target_repo_path
        except RepoNotFoundError:
            click.echo(
                f"Error: target repo '{target_repo}' is not registered. "
                f"Use 'corc repo add' to register it first.",
                err=True,
            )
            sys.exit(1)

    # Validate context bundle paths exist (against target repo root)
    if bundle:
        missing = validate_context_bundle_paths(bundle, context_root)
        if missing:
            for m in missing:
                click.echo(
                    f"Warning: context_bundle: {m['ref']}: {m['reason']}",
                    err=True,
                )
            if strict:
                click.echo(
                    "Aborted: --strict rejects missing context_bundle paths.",
                    err=True,
                )
                sys.exit(1)

    # Record file mtimes at creation time for staleness detection
    bundle_mtimes = record_context_mtimes(bundle, context_root)

    task_data = {
        "id": task_id,
        "name": name,
        "description": description,
        "role": role,
        "depends_on": deps,
        "done_when": done_when,
        "checklist": cl,
        "context_bundle": bundle,
        "context_bundle_mtimes": bundle_mtimes,
        "priority": priority,
        "task_type": task_type,
    }
    if target_repo:
        task_data["target_repo"] = target_repo
    if draft:
        task_data["status"] = "draft"

    ml.append(
        "task_created",
        task_data,
        reason=f"Task created via CLI",
    )

    al.log("task_created", task_id=task_id, name=name)
    type_str = f" [{task_type}]" if task_type != "implementation" else ""
    status_str = " [draft]" if draft else ""
    repo_str = f" [repo={target_repo}]" if target_repo else ""
    click.echo(
        f"Created task {task_id}: {name}{type_str}{status_str}{repo_str} (priority {priority})"
    )


@task.command("list")
@click.option("--status", default=None, help="Filter by status")
@click.option("--ready", is_flag=True, help="Show only ready tasks")
@click.option("--draft", "draft_only", is_flag=True, help="Show only draft tasks")
def task_list(status, ready, draft_only):
    """List tasks."""
    _, _, ws, _, _, _ = _get_all()
    if ready:
        tasks = ws.get_ready_tasks()
    elif draft_only:
        tasks = ws.list_tasks(status="draft")
    elif status:
        tasks = ws.list_tasks(status=status)
    else:
        tasks = ws.list_tasks()

    if not tasks:
        click.echo("No tasks found.")
        return

    for t in tasks:
        deps = t.get("depends_on", [])
        dep_str = f" (depends: {', '.join(deps)})" if deps else ""
        pri = t.get("priority", 100)
        pri_str = f" [pri={pri}]" if pri != 100 else ""
        task_type = t.get("task_type", "implementation")
        type_str = f" ({task_type})" if task_type != "implementation" else ""
        click.echo(
            f"  [{t['status']:>10}] {t['id']}  {t['name']}{type_str}{pri_str}{dep_str}"
        )
        click.echo(f"             done_when: {t['done_when']}")


@task.command("prioritize")
@click.argument("task_id")
@click.argument("priority", type=int)
def task_prioritize(task_id, priority):
    """Update the priority of an existing task (lower=higher priority)."""
    _, ml, ws, al, _, _ = _get_all()
    t = ws.get_task(task_id)
    if not t:
        click.echo(f"Task {task_id} not found.")
        sys.exit(1)

    ml.append(
        "task_updated",
        {"priority": priority},
        reason=f"Priority updated to {priority} via CLI",
        task_id=task_id,
    )
    al.log("task_prioritized", task_id=task_id, priority=priority)
    click.echo(f"Task {task_id} priority updated to {priority}.")


@task.command("deprioritize")
@click.argument("task_id")
def task_deprioritize(task_id):
    """Shelve a task: set priority to -1 and kill its agent process.

    Deprioritized tasks are not dispatched by the daemon until
    explicitly re-enabled with 'corc task reprioritize'.

    If the task is currently running, its agent process is killed
    and the task is reset to pending with priority -1.
    """
    import os as _os
    import signal as _signal

    _, ml, ws, al, _, _ = _get_all()
    t = ws.get_task(task_id)
    if not t:
        click.echo(f"Task {task_id} not found.")
        sys.exit(1)

    if t.get("priority") == -1:
        click.echo(f"Task {task_id} is already deprioritized.")
        return

    old_priority = t.get("priority", 100)
    was_running = t["status"] == "running"

    # Kill agent process if running
    if was_running:
        agents = ws.get_agents_for_task(task_id)
        killed = False
        for agent in agents:
            pid = agent.get("pid")
            if pid:
                try:
                    _os.kill(pid, _signal.SIGTERM)
                    killed = True
                    click.echo(f"Killed agent process (PID {pid}).")
                except (ProcessLookupError, PermissionError):
                    pass  # Process already dead

        if not killed:
            click.echo("Warning: no agent process found to kill.", err=True)

    # Set priority to -1 and status to pending (shelved)
    update_data = {"priority": -1, "status": "pending"}
    ml.append(
        "task_updated",
        update_data,
        reason=f"Deprioritized (shelved) via CLI; was priority {old_priority}, status {t['status']}",
        task_id=task_id,
    )
    al.log(
        "task_deprioritized",
        task_id=task_id,
        old_priority=old_priority,
        was_running=was_running,
    )
    click.echo(f"Task {task_id} deprioritized (priority -1, status pending).")
    click.echo("Use 'corc task reprioritize' to re-enable.")


@task.command("reprioritize")
@click.argument("task_id")
@click.option(
    "--priority",
    default=100,
    type=int,
    help="New priority value (default: 100)",
)
def task_reprioritize(task_id, priority):
    """Re-enable a deprioritized task so the daemon can dispatch it.

    Restores the task to a dispatchable priority. By default sets
    priority to 100; use --priority to set a specific value.
    """
    _, ml, ws, al, _, _ = _get_all()
    t = ws.get_task(task_id)
    if not t:
        click.echo(f"Task {task_id} not found.")
        sys.exit(1)

    if t.get("priority") != -1:
        click.echo(
            f"Task {task_id} is not deprioritized (current priority: {t.get('priority', 100)})."
        )
        return

    if priority < 0:
        click.echo("Error: priority must be >= 0.", err=True)
        sys.exit(1)

    ml.append(
        "task_updated",
        {"priority": priority},
        reason=f"Reprioritized to {priority} via CLI (was deprioritized)",
        task_id=task_id,
    )
    al.log("task_reprioritized", task_id=task_id, priority=priority)
    click.echo(
        f"Task {task_id} reprioritized to {priority}. Daemon will dispatch when ready."
    )


@task.command("approve")
@click.argument("task_id", required=False, default=None)
@click.option("--all", "approve_all", is_flag=True, help="Approve all draft tasks")
def task_approve(task_id, approve_all):
    """Approve a draft task (or all drafts) to make it schedulable."""
    _, ml, ws, al, _, _ = _get_all()

    if approve_all:
        drafts = ws.list_tasks(status="draft")
        if not drafts:
            click.echo("No draft tasks to approve.")
            return
        for t in drafts:
            ml.append(
                "task_approved",
                {},
                reason="Draft approved via CLI (--all)",
                task_id=t["id"],
            )
            al.log("task_approved", task_id=t["id"])
            click.echo(f"Approved task {t['id']}: {t['name']}")
        click.echo(f"Approved {len(drafts)} draft task(s).")
        return

    if not task_id:
        click.echo("Error: provide a TASK_ID or use --all.", err=True)
        sys.exit(1)

    t = ws.get_task(task_id)
    if not t:
        click.echo(f"Task {task_id} not found.", err=True)
        sys.exit(1)

    if t["status"] != "draft":
        click.echo(f"Error: task {task_id} is '{t['status']}', not 'draft'.", err=True)
        sys.exit(1)

    ml.append(
        "task_approved",
        {},
        reason="Draft approved via CLI",
        task_id=task_id,
    )
    al.log("task_approved", task_id=task_id)
    click.echo(f"Approved task {task_id}: {t['name']}")


@task.command("status")
@click.argument("task_id")
def task_status(task_id):
    """Show detailed task status."""
    _, _, ws, al, sl, _ = _get_all()
    t = ws.get_task(task_id)
    if not t:
        click.echo(f"Task {task_id} not found.")
        return

    click.echo(f"Task: {t['name']} ({t['id']})")
    click.echo(f"Status: {t['status']}")
    click.echo(f"Type: {t.get('task_type', 'implementation')}")
    click.echo(f"Priority: {t.get('priority', 100)}")
    click.echo(f"Role: {t.get('role', 'unset')}")
    if t.get("target_repo"):
        click.echo(f"Target repo: {t['target_repo']}")
    click.echo(f"Done when: {t['done_when']}")
    if t.get("depends_on"):
        click.echo(f"Depends on: {', '.join(t['depends_on'])}")
    if t.get("checklist"):
        click.echo("Checklist:")
        for item in t["checklist"]:
            if isinstance(item, dict):
                status = "✅" if item.get("done") else "☐"
                click.echo(f"  {status} {item.get('item', item)}")
            else:
                click.echo(f"  ☐ {item}")
    if t.get("context_bundle"):
        click.echo(f"Context: {', '.join(t['context_bundle'])}")
    if t.get("pr_url"):
        click.echo(f"PR: {t['pr_url']}")
    if t.get("findings"):
        click.echo(f"Findings: {json.dumps(t['findings'], indent=2)}")

    events = al.read_for_task(task_id)
    if events:
        click.echo(f"\nEvents ({len(events)}):")
        for e in events[-5:]:
            click.echo(f"  {e['timestamp']} {e['event_type']}")


# --- Plan ---


@cli.command()
@click.argument("file", required=False, type=click.Path(exists=True))
@click.option(
    "--resume", "resume_session", is_flag=True, help="Resume the last planning session"
)
def plan(file, resume_session):
    """Start an interactive planning session.

    Launches Claude Code in interactive mode with full CORC context injected:
    knowledge store summary, work state summary, and repository context.

    Optionally pre-load a seed document: corc plan my-idea.md

    Resume a crashed/interrupted session: corc plan --resume
    """
    import uuid as _uuid
    from pathlib import Path as _Path
    from corc.plan import (
        build_system_prompt,
        save_session_metadata,
        load_latest_draft,
        mark_session_complete,
        launch_interactive_claude,
    )

    paths, ml, ws, al, sl, ks = _get_all()

    seed_content = None
    if file:
        seed_content = _Path(file).read_text()

    draft_content = None
    resume_meta = None
    resume_claude_session_id = None

    if resume_session:
        resume_meta, draft_content = load_latest_draft(paths["corc_dir"])
        if resume_meta:
            click.echo(
                f"Resuming session from {resume_meta.get('timestamp', 'unknown')}"
            )
            resume_claude_session_id = resume_meta.get("claude_session_id")
            if not resume_claude_session_id:
                click.echo(
                    "Warning: no Claude session ID saved — "
                    "falling back to --continue (may resume wrong session).",
                    err=True,
                )
        else:
            click.echo("No previous session found. Starting fresh.")

    # Build system prompt with full context
    system_prompt = build_system_prompt(
        paths,
        ws,
        ks,
        seed_content=seed_content,
        draft_content=draft_content,
        resume_meta=resume_meta,
    )

    # Generate a Claude session UUID so we can resume precisely
    claude_session_id = str(_uuid.uuid4())

    # Save session metadata for crash recovery
    session_id = time.strftime("%Y%m%d-%H%M%S")
    save_session_metadata(
        paths["corc_dir"],
        session_id,
        seed_file=str(file) if file else None,
        claude_session_id=claude_session_id,
    )

    click.echo("Launching planning session...")
    click.echo("Use 'corc task create' to create tasks from within the session.")
    click.echo()

    try:
        exit_code = launch_interactive_claude(
            system_prompt,
            claude_session_id=claude_session_id,
            resume_claude_session_id=resume_claude_session_id,
            continue_session=bool(resume_meta and not resume_claude_session_id),
        )
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Mark session complete on clean exit
    if exit_code == 0:
        mark_session_complete(paths["corc_dir"], session_id)

    sys.exit(exit_code)


# --- Dispatch ---


@cli.command()
@click.argument("task_id")
@click.option("--provider", default="claude-code", help="Dispatch provider")
def dispatch(task_id, provider):
    """Dispatch an agent for a task."""
    paths, ml, ws, al, sl, ks = _get_all()

    # Check pause lock
    if is_paused(paths["corc_dir"]):
        lock = read_pause_lock(paths["corc_dir"])
        click.echo(f"System is paused: {lock.get('reason', 'unknown')}")
        click.echo("Run 'corc resume' to continue dispatch.")
        return

    t = ws.get_task(task_id)
    if not t:
        click.echo(f"Task {task_id} not found.")
        return

    if t["status"] not in ("pending", "failed", "escalated"):
        click.echo(f"Task {task_id} is {t['status']}, cannot dispatch.")
        return

    # Assemble context
    context = assemble_context(t, paths["root"])
    click.echo(f"Assembled context: {len(context)} chars")

    # Determine constraints from role config
    role_name = t.get("role", "implementer")
    role_loader = RoleLoader(paths["root"])
    try:
        role_config = role_loader.load(role_name)
        constraints = constraints_from_role(role_config)
        system_prompt = get_system_prompt_for_role(role_config, t, context)
        click.echo(f"Role: {role_config.name} ({role_config.description})")
    except ValueError:
        # Fallback if role YAML not found
        constraints = Constraints()
        system_prompt = (
            f"You are a {role_name} working on task '{t['name']}'.\n\n{context}"
        )
        click.echo(f"Role: {role_name} (no YAML config found, using defaults)")

    prompt = (
        f"Complete the following task.\n\n"
        f"Done when: {t['done_when']}\n\n"
        f"Work in the current directory. Write tests alongside implementation. "
        f"Commit your changes with a clear message referencing the task."
    )

    # Update state
    attempt = sl.get_latest_attempt(task_id) + 1
    ml.append(
        "task_started",
        {"attempt": attempt},
        reason="Dispatched via CLI",
        task_id=task_id,
    )
    al.log("task_dispatched", task_id=task_id, role=role_name, attempt=attempt)
    sl.log_dispatch(
        task_id,
        attempt,
        prompt,
        system_prompt,
        constraints.allowed_tools,
        constraints.max_budget_usd,
    )

    # Create git worktree for agent isolation
    worktree_path = None
    try:
        worktree_path, branch_name = create_worktree(paths["root"], task_id, attempt)
        click.echo(f"Created worktree: {worktree_path} (branch: {branch_name})")
        al.log(
            "worktree_created",
            task_id=task_id,
            worktree_path=str(worktree_path),
            branch_name=branch_name,
        )
    except (WorktreeError, Exception) as e:
        click.echo(
            f"Warning: Could not create worktree ({e}), running in project root.",
            err=True,
        )

    # Create agent record
    import uuid as _uuid

    agent_id = f"agent-{_uuid.uuid4().hex[:8]}"
    ml.append(
        "agent_created",
        {
            "id": agent_id,
            "role": role_name,
            "task_id": task_id,
            "worktree_path": str(worktree_path) if worktree_path else None,
        },
        reason="Agent created for CLI dispatch",
    )

    cwd = str(worktree_path) if worktree_path else None
    click.echo(f"Dispatching {task_id} (attempt {attempt}) via {provider}...")

    # Build streaming event callback for real-time CLI output
    def event_callback(event):
        event_type = event.get("type", "unknown")

        # Write every event to session log immediately
        sl.log_stream_event(task_id, attempt, event)

        # Write tool_use events to audit log
        if event_type == "tool_use":
            tool = event.get("tool", {})
            al.log(
                "tool_use",
                task_id=task_id,
                tool_name=tool.get("name", "unknown"),
                tool_input=json.dumps(tool.get("input", {}), separators=(",", ":")),
            )
            click.echo(
                f"  > {tool.get('name', '?')}: {json.dumps(tool.get('input', {}))}"
            )

        elif event_type == "assistant":
            message = event.get("message", {})
            for block in message.get("content", []):
                if block.get("type") == "text":
                    click.echo(block["text"])

        elif event_type == "tool_result":
            content = event.get("content", event.get("tool", {}).get("content", ""))
            if isinstance(content, str) and content:
                click.echo(f"  < {content}")

    # Dispatch with streaming — agent runs in worktree if available
    dispatcher = get_dispatcher(provider)
    result = dispatcher.dispatch(
        prompt, system_prompt, constraints, event_callback=event_callback, cwd=cwd
    )

    # Log result
    sl.log_output(task_id, attempt, result.output, result.exit_code, result.duration_s)
    al.log(
        "task_dispatch_complete",
        task_id=task_id,
        exit_code=result.exit_code,
        duration_s=result.duration_s,
        attempt=attempt,
    )

    click.echo(
        f"Agent finished in {result.duration_s:.1f}s (exit code {result.exit_code})"
    )
    click.echo(f"Session log: {sl.session_path(task_id, attempt)}")

    # Merge worktree changes and clean up
    if worktree_path:
        try:
            merged = merge_worktree(paths["root"], worktree_path)
            if merged:
                click.echo(f"Merged worktree changes from {branch_name}")
                al.log(
                    "worktree_merged", task_id=task_id, worktree_path=str(worktree_path)
                )
            else:
                click.echo(
                    "Warning: Merge conflict — manual resolution needed.", err=True
                )
                al.log(
                    "worktree_merge_conflict",
                    task_id=task_id,
                    worktree_path=str(worktree_path),
                )
        except Exception as e:
            click.echo(f"Warning: Merge failed ({e})", err=True)

        try:
            remove_worktree(paths["root"], worktree_path)
            click.echo(f"Cleaned up worktree: {worktree_path}")
            al.log(
                "worktree_removed", task_id=task_id, worktree_path=str(worktree_path)
            )
        except Exception as e:
            click.echo(f"Warning: Worktree cleanup failed ({e})", err=True)

    if result.exit_code != 0:
        click.echo("Agent exited with error.")
        ml.append(
            "task_failed",
            {"attempt": attempt, "exit_code": result.exit_code},
            reason=f"Agent exited with code {result.exit_code}",
            task_id=task_id,
        )
        al.log(
            "task_failed", task_id=task_id, attempt=attempt, exit_code=result.exit_code
        )
        return

    # Output preview
    output_preview = (
        result.output[:500] + "..." if len(result.output) > 500 else result.output
    )
    click.echo(f"\n--- Output ---\n{output_preview}\n---")

    click.echo(
        f"\nTask dispatched. Review output and run: corc task complete {task_id}"
    )


@task.command("complete")
@click.argument("task_id")
@click.option("--pr-url", default=None, help="PR URL")
@click.option("--findings", default=None, help="JSON array of findings")
def task_complete(task_id, pr_url, findings):
    """Mark a task as completed after review."""
    _, ml, ws, al, _, _ = _get_all()
    t = ws.get_task(task_id)
    if not t:
        click.echo(f"Task {task_id} not found.")
        return

    if t["status"] == "completed":
        click.echo(f"Error: task {task_id} is already completed.", err=True)
        sys.exit(1)

    parsed_findings = json.loads(findings) if findings else []
    data = {"pr_url": pr_url, "findings": parsed_findings}

    ml.append("task_completed", data, reason="Marked complete via CLI", task_id=task_id)
    al.log("task_completed", task_id=task_id, pr_url=pr_url)
    click.echo(f"Task {task_id} marked as completed.")

    # Show newly ready tasks
    ws.refresh()
    ready = ws.get_ready_tasks()
    if ready:
        click.echo(f"\nNewly ready tasks:")
        for r in ready:
            click.echo(f"  {r['id']}  {r['name']}")


# --- Context ---


@cli.command("context-for-task")
@click.argument("task_id")
def context_for_task(task_id):
    """Show assembled context for a task (debugging)."""
    paths, _, ws, _, _, _ = _get_all()
    t = ws.get_task(task_id)
    if not t:
        click.echo(f"Task {task_id} not found.")
        return
    ctx = assemble_context(t, paths["root"])
    click.echo(ctx)


# --- Pause / Resume ---


@cli.command()
@click.argument("reason")
@click.option("--source", default=None, help="Source identifier (default: cli:<pid>)")
def pause(reason, source):
    """Pause all new dispatch. In-flight tasks will complete."""
    paths = get_paths()
    corc_dir = paths["corc_dir"]

    if is_paused(corc_dir):
        lock = read_pause_lock(corc_dir)
        click.echo(f"Already paused: {lock.get('reason', 'unknown')}")
        click.echo(f"  since: {lock.get('timestamp', 'unknown')}")
        return

    lock_data = write_pause_lock(corc_dir, reason, source)

    # Log to audit and mutation log
    _, _, _, al, _, _ = _get_all()
    al.log("pause", reason=reason, source=lock_data["source"])

    click.echo(f"Paused: {reason}")
    click.echo("In-flight tasks will complete. No new tasks will be dispatched.")


@cli.command()
def resume():
    """Resume dispatch after a pause."""
    paths = get_paths()
    corc_dir = paths["corc_dir"]

    lock = read_pause_lock(corc_dir)
    if not lock:
        click.echo("Not paused.")
        return

    removed = remove_pause_lock(corc_dir)
    if removed:
        _, _, _, al, _, _ = _get_all()
        al.log("resume", previous_reason=lock.get("reason", ""))
        click.echo(f"Resumed. Previous pause reason: {lock.get('reason', 'unknown')}")
    else:
        click.echo("Not paused.")


# --- Status ---


@cli.command()
def status():
    """Show current system state."""
    paths, _, ws, al, _, _ = _get_all()

    # Show pause state first
    lock = read_pause_lock(paths["corc_dir"])
    if lock:
        click.echo(f"PAUSED: {lock.get('reason', 'unknown')}")
        click.echo(f"  since: {lock.get('timestamp', 'unknown')}")
        click.echo(f"  source: {lock.get('source', 'unknown')}")
        click.echo()

    tasks = ws.list_tasks()
    if not tasks:
        click.echo("No tasks.")
        return

    by_status = {}
    for t in tasks:
        by_status.setdefault(t["status"], []).append(t)

    total = len(tasks)
    completed = len(by_status.get("completed", []))
    click.echo(
        f"Tasks: {completed}/{total} complete ({100 * completed // total if total else 0}%)"
    )

    for status_name in [
        "completed",
        "running",
        "pending",
        "draft",
        "failed",
        "escalated",
        "blocked",
        "handed_off",
    ]:
        group = by_status.get(status_name, [])
        if not group:
            continue
        icon = {
            "completed": "✅",
            "running": "🔄",
            "pending": "⬚",
            "draft": "📝",
            "failed": "❌",
            "escalated": "🚨",
            "blocked": "◻",
            "handed_off": "↗",
        }.get(status_name, "?")
        names = ", ".join(t["name"] for t in group)
        click.echo(f"  {icon} {status_name}: {names}")

    ready = ws.get_ready_tasks()
    if ready:
        click.echo(f"\nReady to dispatch: {', '.join(t['name'] for t in ready)}")

    recent_events = al.read_recent(50)
    if recent_events:
        click.echo(f"\nRecent events: {len(recent_events)}")


# --- DAG Visualization ---


@cli.command()
@click.option(
    "--mermaid", is_flag=True, help="Output Mermaid markdown instead of ASCII"
)
@click.option("--no-color", is_flag=True, help="Disable ANSI colours")
def dag(mermaid, no_color):
    """Render the task dependency graph."""
    _, _, ws, _, _, _ = _get_all()
    tasks = ws.list_tasks()
    if not tasks:
        click.echo("No tasks found.")
        return

    if mermaid:
        click.echo(render_mermaid(tasks), nl=False)
    else:
        click.echo(render_ascii_dag(tasks, use_color=not no_color), nl=False)


# --- Knowledge Store ---


@cli.group()
def knowledge():
    """Knowledge store commands."""
    pass


@knowledge.command("add")
@click.argument("file_path")
@click.option("--type", "doc_type", default="note")
@click.option("--project", default=None)
@click.option("--tags", default="")
def knowledge_add(file_path, doc_type, project, tags):
    """Add a document to the knowledge store."""
    _, _, _, _, _, ks = _get_all()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    doc_id = ks.add(
        file_path=file_path, doc_type=doc_type, project=project, tags=tag_list
    )
    click.echo(f"Added document {doc_id}")


@knowledge.command("search")
@click.argument("query")
@click.option("--limit", default=10)
@click.option("--type", "doc_type", default=None)
@click.option("--project", default=None)
@click.option(
    "--mode",
    type=click.Choice(["hybrid", "keyword", "semantic"]),
    default="hybrid",
    help="Search mode (default: hybrid)",
)
def knowledge_search(query, limit, doc_type, project, mode):
    """Search the knowledge store (hybrid search by default)."""
    _, _, _, _, _, ks = _get_all()
    if mode == "keyword":
        results = ks.search(query, limit=limit, doc_type=doc_type, project=project)
    elif mode == "semantic":
        results = ks.semantic_search(
            query, limit=limit, doc_type=doc_type, project=project
        )
    else:
        results = ks.hybrid_search(
            query, limit=limit, doc_type=doc_type, project=project
        )
    if not results:
        click.echo("No results.")
        return
    for r in results:
        click.echo(
            f"  [{r.get('type', '?'):>12}] {r['id']}  {r['title']}  (score: {r.get('score', 'N/A'):.4f})"
        )


@knowledge.command("get")
@click.argument("doc_id")
def knowledge_get(doc_id):
    """Get a document by ID."""
    _, _, _, _, _, ks = _get_all()
    doc = ks.get(doc_id)
    if not doc:
        click.echo(f"Document {doc_id} not found.")
        return
    click.echo(f"Title: {doc['title']}")
    click.echo(f"Type: {doc['type']}")
    click.echo(f"Path: {doc['file_path']}")
    if doc.get("content"):
        click.echo(f"\n{doc['content']}")


@knowledge.command("reindex")
def knowledge_reindex():
    """Rebuild the knowledge store index."""
    _, _, _, _, _, ks = _get_all()
    ks.reindex()
    stats = ks.stats()
    click.echo(f"Reindexed. {stats['total']} documents.")


@knowledge.command("stats")
def knowledge_stats():
    """Show knowledge store statistics."""
    _, _, _, _, _, ks = _get_all()
    stats = ks.stats()
    click.echo(f"Total: {stats['total']} documents")
    for t, c in stats.get("by_type", {}).items():
        click.echo(f"  {t}: {c}")


# --- Knowledge Curation ---


@cli.command("curate")
@click.argument("task_id")
@click.option(
    "--non-interactive", is_flag=True, help="List findings without interactive prompts"
)
@click.option(
    "--approve-all", is_flag=True, help="Approve all findings without prompts"
)
@click.option("--reject-all", is_flag=True, help="Reject all findings with a reason")
@click.option("--reject-reason", default="", help="Reason for --reject-all")
@click.option(
    "--type", "doc_type", default="note", help="Document type for approved findings"
)
@click.option("--project", default=None, help="Project for approved findings")
def curate_cmd(
    task_id, non_interactive, approve_all, reject_all, reject_reason, doc_type, project
):
    """Curate agent findings from a completed task.

    Shows findings reported by agents during task execution.
    For each finding, the operator can approve (writes to knowledge store)
    or reject (logs to mutation log with reason).

    After 3+ rejections of the same finding type, suggests adding
    that type to a blacklist.
    """
    from corc.curate import CurationEngine, CurationResult

    paths, ml, ws, al, sl, ks = _get_all()
    engine = CurationEngine(ws, ml, al, ks)

    # Load findings
    try:
        findings = engine.get_findings(task_id)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if not findings:
        click.echo(f"No findings for task {task_id}.")
        return

    task = ws.get_task(task_id)
    click.echo(f"Task: {task['name']} ({task_id})")
    click.echo(f"Findings: {len(findings)}")
    click.echo()

    result = CurationResult(task_id=task_id)

    if non_interactive:
        # Just list findings
        for f in findings:
            click.echo(f"  [{f.index}] [{f.finding_type}] {f.content}")
        return

    if approve_all:
        for f in findings:
            doc_id = engine.approve_finding(
                task_id, f, doc_type=doc_type, project=project
            )
            click.echo(f"  Approved [{f.index}]: {f.content[:60]}... -> {doc_id}")
            result.approved += 1
    elif reject_all:
        reason = reject_reason or "Batch rejected"
        for f in findings:
            engine.reject_finding(task_id, f, reason=reason)
            click.echo(f"  Rejected [{f.index}]: {f.content[:60]}...")
            result.rejected += 1
    else:
        # Interactive mode
        for f in findings:
            click.echo(f"--- Finding [{f.index}] ---")
            click.echo(f"Type: {f.finding_type}")
            click.echo(f"Content: {f.content}")
            click.echo()

            while True:
                choice = click.prompt(
                    "Action",
                    type=click.Choice(["approve", "reject", "skip", "a", "r", "s"]),
                    default="s",
                )
                if choice in ("approve", "a"):
                    doc_id = engine.approve_finding(
                        task_id, f, doc_type=doc_type, project=project
                    )
                    click.echo(f"  -> Approved, doc_id: {doc_id}")
                    result.approved += 1
                    break
                elif choice in ("reject", "r"):
                    reason = click.prompt("Rejection reason")
                    engine.reject_finding(task_id, f, reason=reason)
                    click.echo(f"  -> Rejected")
                    result.rejected += 1
                    break
                elif choice in ("skip", "s"):
                    click.echo(f"  -> Skipped")
                    result.skipped += 1
                    break
            click.echo()

    # Summary
    click.echo(f"\nCuration summary:")
    click.echo(f"  Approved: {result.approved}")
    click.echo(f"  Rejected: {result.rejected}")
    click.echo(f"  Skipped:  {result.skipped}")

    # Check for blacklist suggestions
    suggestions = engine.get_blacklist_suggestions()
    if suggestions:
        click.echo(f"\nBlacklist suggestions:")
        for s in suggestions:
            click.echo(f"  {s['finding_type']}: {s['rejection_count']} rejections")
            click.echo(f"    {s['suggestion']}")
            if s["recent_reasons"]:
                click.echo(f"    Recent reasons: {', '.join(s['recent_reasons'][:3])}")


# --- Templates ---


@cli.command("template")
@click.argument("type_name")
@click.option("--title", default=None, help="Document title for rendered template")
@click.option("--project", default=None, help="Project name for rendered template")
@click.option("--render", is_flag=True, help="Render with generated ID and timestamps")
def template_cmd(type_name, title, project, render):
    """Output a document template for TYPE.

    TYPE is one of: decision, task-outcome, architecture, repo-context, research.
    """
    try:
        if render:
            output = render_template(
                type_name,
                title=title or "Untitled",
                project=project or "",
                project_root=get_paths()["root"],
            )
        else:
            output = get_template(type_name, project_root=get_paths()["root"])
        click.echo(output, nl=False)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# --- Watch (TUI v1 — two-panel dashboard) ---


@cli.command()
@click.option("--last", "last_n", default=20, help="Show last N events")
def watch(last_n):
    """Live active-plan dashboard: focused on current work + event stream.

    Top panel shows only active/relevant tasks: running (with elapsed time
    and agent info), ready (dispatchable), blocked (with dependency info),
    and recently completed (dimmed). Historical completed tasks are hidden.
    Bottom panel shows color-coded events as they happen.
    Press 'q' to quit, or Ctrl+C.
    """
    try:
        from corc.tui import run_active_dashboard

        _watch_dashboard(last_n)
    except ImportError:
        _watch_plain(last_n)


def _watch_dashboard(last_n):
    from corc.tui import run_active_dashboard
    from corc.queries import QueryAPI

    paths, ml, ws, al, sl, _ = _get_all()
    query_api = QueryAPI(ws, al, sl)
    run_active_dashboard(query_api, max_events=last_n)


def _watch_plain(last_n):
    paths = get_paths()
    al = AuditLog(paths["events_dir"])
    click.echo("Watching events (install 'rich' for TUI). Ctrl+C to exit.")
    seen_count = 0
    while True:
        events = al.read_recent(last_n)
        if len(events) > seen_count:
            for e in events[seen_count:]:
                ts = e.get("timestamp", "")[:19]
                click.echo(
                    f"{ts} {e.get('event_type', '?'):>25} {e.get('task_id', '')[:8]}"
                )
            seen_count = len(events)
        time.sleep(2)


# --- Daemon ---


@cli.command("start")
@click.option(
    "--parallel", default=1, type=int, help="Max concurrent agents (default: 1)"
)
@click.option(
    "--task", "task_id", default=None, help="Run one specific task, then stop"
)
@click.option("--once", is_flag=True, help="Process one ready task, then stop")
@click.option("--provider", default="claude-code", help="Dispatch provider")
@click.option(
    "--poll-interval",
    default=5.0,
    type=float,
    help="Seconds between polls (default: 5)",
)
def start_cmd(parallel, task_id, once, provider, poll_interval):
    """Start the daemon. Processes all ready work. Idles when empty."""
    paths, ml, ws, al, sl, _ = _get_all()

    # Check for already-running daemon
    pid_file = paths["root"] / ".corc" / "daemon.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            import os

            os.kill(pid, 0)  # Check if process exists
            click.echo(f"Daemon already running (PID {pid}). Use 'corc stop' first.")
            return
        except (ProcessLookupError, ValueError):
            pid_file.unlink(missing_ok=True)

    dispatcher = get_dispatcher(provider)

    click.echo(f"Starting CORC daemon (parallel={parallel}, provider={provider})")
    if task_id:
        click.echo(f"  Target task: {task_id}")
    if once:
        click.echo(f"  Mode: once (will stop after one task)")
    click.echo("Reconciling state from mutation log...")

    daemon = Daemon(
        state=ws,
        mutation_log=ml,
        audit_log=al,
        session_logger=sl,
        dispatcher=dispatcher,
        project_root=paths["root"],
        parallel=parallel,
        poll_interval=poll_interval,
        task_id=task_id,
        once=once,
    )
    daemon.start()

    # Show reconciliation results after start (reconcile runs inside start)
    summary = daemon._reconcile_summary
    if summary:
        stale = summary.get("running_found", 0) + summary.get("assigned_found", 0)
        if stale:
            click.echo(
                f"  Reconciled {stale} stale task(s): "
                f"{summary.get('agents_alive', 0)} alive, "
                f"{summary.get('agents_dead_with_output', 0)} processed output, "
                f"{summary.get('agents_dead_no_output', 0)} marked failed"
            )
        if summary.get("worktrees_cleaned", 0):
            click.echo(f"  Cleaned {summary['worktrees_cleaned']} stale worktree(s)")

    click.echo("Daemon stopped.")


@cli.command("stop")
def stop_cmd():
    """Graceful shutdown (in-flight tasks finish)."""
    paths = get_paths()
    if stop_daemon(paths["root"]):
        click.echo("Stop signal sent to daemon. In-flight tasks will finish.")
    else:
        click.echo("No running daemon found.")


# --- Escalations ---


@cli.command("escalations")
@click.option(
    "--all", "show_all", is_flag=True, help="Show all escalations (including resolved)"
)
def escalations_cmd(show_all):
    """List pending escalations."""
    _, _, ws, _, _, _ = _get_all()
    ws.refresh()
    if show_all:
        escs = ws.list_escalations()
    else:
        escs = ws.list_escalations(status="pending")

    if not escs:
        click.echo("No escalations found.")
        return

    for esc in escs:
        status_icon = "🔴" if esc["status"] == "pending" else "✅"
        click.echo(
            f"  {status_icon} {esc['id']}  task={esc['task_id']}  {esc.get('task_name', '')}  "
            f"attempts={esc.get('attempts', '?')}  status={esc['status']}"
        )


@cli.group()
def escalation():
    """Manage escalations."""
    pass


@escalation.command("show")
@click.argument("escalation_id")
def escalation_show(escalation_id):
    """Show full details of an escalation."""
    _, _, ws, _, sl, _ = _get_all()
    ws.refresh()
    esc = ws.get_escalation(escalation_id)
    if not esc:
        click.echo(f"Escalation {escalation_id} not found.")
        sys.exit(1)
        return

    click.echo(f"Escalation: {esc['id']}")
    click.echo(f"Status: {esc['status']}")
    click.echo(f"Task: {esc.get('task_name', '')} ({esc['task_id']})")
    click.echo(f"Attempts: {esc.get('attempts', '?')}")
    click.echo(f"Created: {esc.get('created', 'unknown')}")
    if esc.get("resolved"):
        click.echo(f"Resolved: {esc['resolved']}")
    if esc.get("resolution"):
        click.echo(f"Resolution: {esc['resolution']}")
    click.echo(f"Done when: {esc.get('done_when', '')}")
    click.echo(f"Session log: {esc.get('session_log_path', '')}")

    click.echo(f"\nError:")
    click.echo(f"  {esc.get('error', 'no error recorded')}")

    actions = esc.get("suggested_actions", [])
    if actions:
        click.echo(f"\nSuggested actions:")
        for action in actions:
            click.echo(f"  • {action}")


@escalation.command("resolve")
@click.argument("escalation_id")
@click.option("--resolution", default="", help="Resolution description")
@click.option(
    "--unblock", is_flag=True, help="Reset the task to pending after resolving"
)
def escalation_resolve(escalation_id, resolution, unblock):
    """Resolve an escalation and optionally unblock the task."""
    paths, ml, ws, al, _, _ = _get_all()
    ws.refresh()
    esc = ws.get_escalation(escalation_id)
    if not esc:
        click.echo(f"Escalation {escalation_id} not found.")
        sys.exit(1)
        return

    if esc["status"] == "resolved":
        click.echo(f"Escalation {escalation_id} is already resolved.")
        return

    resolve_escalation(escalation_id, ml, resolution)
    al.log("escalation_resolved", escalation_id=escalation_id, task_id=esc["task_id"])

    if unblock:
        ml.append(
            "task_updated",
            {"status": "pending"},
            reason=f"Unblocked by escalation resolution {escalation_id}",
            task_id=esc["task_id"],
        )
        al.log("task_unblocked", task_id=esc["task_id"], escalation_id=escalation_id)
        click.echo(f"Task {esc['task_id']} reset to pending.")

    ws.refresh()
    click.echo(f"Escalation {escalation_id} resolved.")


# --- Roles ---


@cli.group()
def role():
    """Manage agent roles."""
    pass


@role.command("list")
def role_list():
    """List all available roles."""
    paths = get_paths()
    loader = RoleLoader(paths["root"])
    roles = loader.list_roles()

    if not roles:
        click.echo("No roles found.")
        return

    for r in roles:
        click.echo(f"  {r['name']:<25} {r['description']:<60} [{r['source']}]")


@role.command("show")
@click.argument("name")
def role_show(name):
    """Show full details of a role."""
    paths = get_paths()
    loader = RoleLoader(paths["root"])

    try:
        rc = loader.load(name)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Name: {rc.name}")
    click.echo(f"Description: {rc.description}")
    click.echo(f"Extends: {rc.extends or '(none)'}")
    click.echo(f"Knowledge write access: {rc.knowledge_write_access}")
    click.echo(f"Allowed tools: {', '.join(rc.allowed_tools)}")
    click.echo(f"Max budget (USD): {rc.max_budget_usd}")
    click.echo(f"Max turns: {rc.max_turns}")
    if rc.source_path:
        click.echo(f"Source: {rc.source_path}")
    click.echo(f"\nSystem prompt:\n{rc.system_prompt}")


@role.command("validate")
@click.argument("name")
def role_validate(name):
    """Validate a role configuration."""
    paths = get_paths()
    loader = RoleLoader(paths["root"])

    result = loader.validate(name)
    if result.valid:
        click.echo(f"Role '{name}' is valid.")
    else:
        click.echo(f"Role '{name}' has errors:", err=True)
        for err in result.errors:
            click.echo(f"  ERROR: {err}", err=True)

    for warn in result.warnings:
        click.echo(f"  WARNING: {warn}", err=True)

    if not result.valid:
        sys.exit(1)


# --- Repo Management ---


@cli.group()
def repo():
    """Manage registered repositories and merge policies."""
    pass


@repo.command("add")
@click.argument("name")
@click.option("--path", required=True, help="Filesystem path to the repository")
@click.option(
    "--merge-policy",
    default="auto",
    type=click.Choice(["auto", "human-only"], case_sensitive=False),
    help="Merge policy (default: auto)",
)
@click.option(
    "--protected-branches",
    default="main",
    help="Comma-separated list of protected branches (default: main)",
)
@click.option(
    "--enforcement-level",
    default="strict",
    type=click.Choice(["strict", "relaxed"], case_sensitive=False),
    help="Enforcement level (default: strict)",
)
def repo_add(name, path, merge_policy, protected_branches, enforcement_level):
    """Register a repository with merge policy settings.

    Examples:
      corc repo add my-app --path /code/my-app
      corc repo add prod --path /code/prod --merge-policy human-only
      corc repo add api --path /code/api --protected-branches main,staging
    """
    from corc.repo import RepoManager, RepoAlreadyExistsError, RepoValidationError

    cfg = load_config()
    mgr = RepoManager(cfg)

    branches = [b.strip() for b in protected_branches.split(",") if b.strip()]

    try:
        repo_config = mgr.add(
            name=name,
            path=path,
            merge_policy=merge_policy,
            protected_branches=branches,
            enforcement_level=enforcement_level,
        )
        cfg.save()
    except RepoAlreadyExistsError:
        click.echo(f"Error: repo '{name}' already exists.", err=True)
        sys.exit(1)
    except RepoValidationError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Auto-generate Claude Code hooks for this repo
    from corc.hook_gen import sync_hooks

    try:
        written = sync_hooks(path, repo_config)
        click.echo(f"Generated {len(written)} hook file(s) in {path}/.claude/")
    except OSError as e:
        click.echo(f"Warning: could not generate hooks: {e}", err=True)

    click.echo(f"Registered repo '{name}'")
    click.echo(f"  path: {repo_config['path']}")
    click.echo(f"  merge_policy: {repo_config['merge_policy']}")
    click.echo(f"  protected_branches: {', '.join(repo_config['protected_branches'])}")
    click.echo(f"  enforcement_level: {repo_config['enforcement_level']}")


@repo.command("list")
def repo_list():
    """List all registered repositories with settings."""
    from corc.repo import RepoManager

    cfg = load_config()
    mgr = RepoManager(cfg)
    repos = mgr.list_repos()

    if not repos:
        click.echo("No repos registered.")
        return

    for r in repos:
        branches = ", ".join(r.get("protected_branches", []))
        click.echo(
            f"  {r['name']:<25} "
            f"policy={r.get('merge_policy', '?'):<12} "
            f"enforcement={r.get('enforcement_level', '?'):<8} "
            f"branches=[{branches}]"
        )
        click.echo(f"  {'':25} path={r.get('path', '?')}")


@repo.command("show")
@click.argument("name")
def repo_show(name):
    """Show full configuration for a repository."""
    from corc.repo import RepoManager, RepoNotFoundError

    cfg = load_config()
    mgr = RepoManager(cfg)

    try:
        repo_config = mgr.get(name)
    except RepoNotFoundError:
        click.echo(f"Error: repo '{name}' not found.", err=True)
        sys.exit(1)

    click.echo(f"Repo: {name}")
    click.echo(f"  path: {repo_config['path']}")
    click.echo(f"  merge_policy: {repo_config['merge_policy']}")
    click.echo(f"  protected_branches: {', '.join(repo_config['protected_branches'])}")
    click.echo(f"  enforcement_level: {repo_config['enforcement_level']}")


@repo.command("remove")
@click.argument("name")
def repo_remove(name):
    """Remove a registered repository."""
    from corc.repo import RepoManager, RepoNotFoundError

    cfg = load_config()
    mgr = RepoManager(cfg)

    try:
        mgr.remove(name)
        cfg.save()
    except RepoNotFoundError:
        click.echo(f"Error: repo '{name}' not found.", err=True)
        sys.exit(1)

    click.echo(f"Removed repo '{name}'.")


@repo.command("update")
@click.argument("name")
@click.option(
    "--merge-policy",
    default=None,
    type=click.Choice(["auto", "human-only"], case_sensitive=False),
    help="Update merge policy",
)
@click.option("--path", default=None, help="Update filesystem path")
@click.option(
    "--protected-branches",
    default=None,
    help="Update protected branches (comma-separated)",
)
@click.option(
    "--enforcement-level",
    default=None,
    type=click.Choice(["strict", "relaxed"], case_sensitive=False),
    help="Update enforcement level",
)
def repo_update(name, merge_policy, path, protected_branches, enforcement_level):
    """Update settings for a registered repository.

    Examples:
      corc repo update my-app --merge-policy human-only
      corc repo update my-app --protected-branches main,staging,prod
      corc repo update my-app --enforcement-level relaxed
    """
    from corc.repo import RepoManager, RepoNotFoundError, RepoValidationError

    cfg = load_config()
    mgr = RepoManager(cfg)

    kwargs = {}
    if merge_policy is not None:
        kwargs["merge_policy"] = merge_policy
    if path is not None:
        kwargs["path"] = path
    if protected_branches is not None:
        kwargs["protected_branches"] = [
            b.strip() for b in protected_branches.split(",") if b.strip()
        ]
    if enforcement_level is not None:
        kwargs["enforcement_level"] = enforcement_level

    if not kwargs:
        click.echo("Nothing to update. Provide at least one option.")
        sys.exit(1)

    try:
        repo_config = mgr.update(name, **kwargs)
        cfg.save()
    except RepoNotFoundError:
        click.echo(f"Error: repo '{name}' not found.", err=True)
        sys.exit(1)
    except RepoValidationError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Re-generate Claude Code hooks after update
    from corc.hook_gen import sync_hooks

    repo_path = repo_config["path"]
    try:
        written = sync_hooks(repo_path, repo_config)
        click.echo(f"Regenerated {len(written)} hook file(s) in {repo_path}/.claude/")
    except OSError as e:
        click.echo(f"Warning: could not regenerate hooks: {e}", err=True)

    click.echo(f"Updated repo '{name}'")
    click.echo(f"  path: {repo_config['path']}")
    click.echo(f"  merge_policy: {repo_config['merge_policy']}")
    click.echo(f"  protected_branches: {', '.join(repo_config['protected_branches'])}")
    click.echo(f"  enforcement_level: {repo_config['enforcement_level']}")


# --- Chaos Monkey ---


@cli.group()
def chaos():
    """Chaos monkey — resilience testing."""
    pass


@chaos.command("enable")
@click.option(
    "--kill-rate",
    default=0.1,
    type=float,
    help="Probability of killing an agent per tick (0.0–1.0)",
)
@click.option(
    "--corrupt-rate",
    default=0.05,
    type=float,
    help="Probability of corrupting state per tick (0.0–1.0)",
)
@click.option("--seed", default=None, type=int, help="RNG seed for reproducibility")
def chaos_enable(kill_rate, corrupt_rate, seed):
    """Enable chaos mode — randomly kills agents and corrupts state."""
    from corc.chaos import ChaosConfig, write_chaos_config

    config = ChaosConfig(
        enabled=True, kill_rate=kill_rate, corrupt_rate=corrupt_rate, seed=seed
    )
    errors = config.validate()
    if errors:
        for err in errors:
            click.echo(f"Error: {err}", err=True)
        sys.exit(1)

    paths = get_paths()
    write_chaos_config(paths["corc_dir"], config)

    _, _, _, al, _, _ = _get_all()
    al.log("chaos_enabled", kill_rate=kill_rate, corrupt_rate=corrupt_rate, seed=seed)

    click.echo(f"Chaos monkey ENABLED")
    click.echo(f"  kill-rate:    {kill_rate}")
    click.echo(f"  corrupt-rate: {corrupt_rate}")
    if seed is not None:
        click.echo(f"  seed:         {seed}")
    click.echo("The daemon will pick up these settings on the next tick.")


@chaos.command("disable")
def chaos_disable():
    """Disable chaos mode."""
    from corc.chaos import read_chaos_config, write_chaos_config, ChaosConfig

    paths = get_paths()
    config = read_chaos_config(paths["corc_dir"])
    config.enabled = False
    write_chaos_config(paths["corc_dir"], config)

    _, _, _, al, _, _ = _get_all()
    al.log("chaos_disabled")

    click.echo("Chaos monkey DISABLED.")


@chaos.command("status")
def chaos_status():
    """Show chaos monkey settings and recovery stats."""
    from corc.chaos import read_chaos_config, get_chaos_stats

    paths = get_paths()
    config = read_chaos_config(paths["corc_dir"])
    stats = get_chaos_stats(paths["corc_dir"])

    if config.enabled:
        click.echo("Chaos monkey: ENABLED")
    else:
        click.echo("Chaos monkey: disabled")

    click.echo(f"  kill-rate:    {config.kill_rate}")
    click.echo(f"  corrupt-rate: {config.corrupt_rate}")
    if config.seed is not None:
        click.echo(f"  seed:         {config.seed}")

    if stats["total_events"] > 0:
        click.echo(f"\nChaos events: {stats['total_events']}")
        click.echo(f"  kills:       {stats['kills']}")
        click.echo(f"  corruptions: {stats['corruptions']}")
        click.echo(f"  recovered:   {stats['recovered']}")
        click.echo(f"  failed:      {stats['failed']}")
        click.echo(f"  pending:     {stats['pending']}")
        click.echo(f"  recovery %%:  {stats['recovery_rate']:.1f}%%")
    else:
        click.echo("\nNo chaos events recorded yet.")


# --- Self-test ---


@cli.command("self-test")
def self_test():
    """Run orchestrator self-tests."""
    import subprocess

    result = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-x", "-q"],
        capture_output=True,
        text=True,
        cwd=str(get_paths()["root"]),
    )
    click.echo(result.stdout)
    if result.stderr:
        click.echo(result.stderr)
    sys.exit(result.returncode)


# --- Log ---


@cli.command("log")
@click.option("--last", "last_n", default=20)
@click.option("--task", "task_id", default=None)
def log_cmd(last_n, task_id):
    """Human-readable event log."""
    _, _, _, al, _, _ = _get_all()
    if task_id:
        events = al.read_for_task(task_id)
    else:
        events = al.read_recent(last_n)

    for e in events:
        ts = e.get("timestamp", "")[:19].replace("T", " ")
        etype = e.get("event_type", "unknown")
        tid = e.get("task_id", "")[:8] if e.get("task_id") else "        "
        extra_parts = []
        for k in ("name", "role", "duration_s", "exit_code", "attempt"):
            if k in e:
                extra_parts.append(f"{k}={e[k]}")
        extra = " ".join(extra_parts)
        click.echo(f"{ts}  {etype:<25} {tid}  {extra}")


# --- Logs (rotation) ---


@cli.group()
def logs():
    """Log management — rotation and archival."""
    pass


@logs.command("rotate")
@click.option(
    "--days",
    default=None,
    type=int,
    help="Move files older than this many days (default: from config or 7)",
)
def logs_rotate(days):
    """Rotate old logs to archive directories.

    Moves session and audit log files older than --days (default 7)
    into date-stamped archive directories:

      data/sessions/archive/YYYY-MM-DD/
      data/events/archive/YYYY-MM-DD/

    Files are never deleted, only moved to archive.
    """
    from corc.rotate import load_rotation_config, rotate_logs

    paths = get_paths()
    events_dir = paths["events_dir"]
    sessions_dir = paths["sessions_dir"]
    corc_dir = paths["corc_dir"]

    config = load_rotation_config(corc_dir)
    rotate_days = days if days is not None else config["rotate_after_days"]

    # Resolve archive paths from config or use defaults
    events_archive = (
        Path(config["events_archive"]) if config.get("events_archive") else None
    )
    sessions_archive = (
        Path(config["session_archive"]) if config.get("session_archive") else None
    )

    result = rotate_logs(
        events_dir=events_dir,
        sessions_dir=sessions_dir,
        rotate_after_days=rotate_days,
        events_archive=events_archive,
        sessions_archive=sessions_archive,
    )

    sessions_moved = result["sessions"]["moved"]
    events_moved = result["events"]["moved"]

    click.echo(f"Log rotation complete (threshold: {rotate_days} days).")
    click.echo(f"  Sessions archived: {sessions_moved}")
    click.echo(f"  Events archived:   {events_moved}")

    if sessions_moved == 0 and events_moved == 0:
        click.echo("  No files needed rotation.")


# --- Analyze ---


@cli.group()
def analyze():
    """Cost analysis, duration trends, and failure reporting."""
    pass


@analyze.command("costs")
@click.option("--today", "today_flag", is_flag=True, help="Show today's cost breakdown")
@click.option("--project", default=None, help="Filter by project name")
@click.option("--since", default=None, help="Filter events since date (ISO format)")
@click.option(
    "--alerts", "alerts_flag", is_flag=True, help="Check cost threshold alerts"
)
def analyze_costs(today_flag, project, since, alerts_flag):
    """Show cost breakdown from audit log data.

    Examples:
      corc analyze costs --today
      corc analyze costs --project myapp --since 2026-03-01
      corc analyze costs --alerts
    """
    from corc.analyze import (
        compute_costs_today,
        compute_costs_project,
        aggregate_costs,
        load_alert_config,
        check_cost_alerts,
        format_cost_breakdown,
        format_alerts,
    )

    paths, _, _, al, _, _ = _get_all()

    if alerts_flag:
        config = load_alert_config(paths["corc_dir"])
        alerts = check_cost_alerts(al, config)
        click.echo(format_alerts(alerts))
        return

    if today_flag:
        breakdown = compute_costs_today(al)
        click.echo(format_cost_breakdown(breakdown, title="Today's Costs"))
    elif project:
        breakdown = compute_costs_project(al, project, since=since)
        title = f"Costs for project '{project}'"
        if since:
            title += f" (since {since})"
        click.echo(format_cost_breakdown(breakdown, title=title))
    else:
        events = al.read_all(since=since)
        breakdown = aggregate_costs(events)
        title = "All Costs"
        if since:
            title += f" (since {since})"
        click.echo(format_cost_breakdown(breakdown, title=title))

    # Always check alerts after showing costs
    config = load_alert_config(paths["corc_dir"])
    if config.enabled:
        alerts = check_cost_alerts(al, config)
        if alerts:
            click.echo()
            click.echo(format_alerts(alerts))


@analyze.command("duration")
@click.option("--last", "last_n", default=20, help="Show last N task durations")
def analyze_duration(last_n):
    """Show task duration trends.

    Examples:
      corc analyze duration --last 10
      corc analyze duration --last 50
    """
    from corc.analyze import compute_duration_trends, format_duration_trends

    _, _, _, al, _, _ = _get_all()
    entries = compute_duration_trends(al, last_n=last_n)
    click.echo(format_duration_trends(entries))


@analyze.command("failures")
@click.option("--since", default=None, help="Filter failures since date (ISO format)")
def analyze_failures(since):
    """Show failure report.

    Examples:
      corc analyze failures --since 2026-03-01
    """
    from corc.analyze import compute_failures, format_failures

    _, _, _, al, _, _ = _get_all()
    entries = compute_failures(al, since=since)
    click.echo(format_failures(entries))


@analyze.command("patterns")
def analyze_patterns_cmd():
    """Identify correlations between roles/task-types/context and scores.

    Detects what's working and what isn't. Produces actionable recommendations.

    Examples:
      corc analyze patterns
    """
    from corc.rating import RatingStore
    from corc.patterns import analyze_patterns, format_pattern_report

    paths, _, _, _, _, _ = _get_all()
    rs = RatingStore(paths["ratings_dir"])
    ratings = rs.read_all()
    report = analyze_patterns(ratings)
    click.echo(format_pattern_report(report))


@analyze.command("prompts")
@click.option("--role", required=True, help="Role name to analyze prompt versions for")
def analyze_prompts_cmd(role):
    """Show quality scores by prompt version for a role.

    Tracks which prompt versions produce better outcomes.

    Examples:
      corc analyze prompts --role implementer
      corc analyze prompts --role reviewer
    """
    from corc.rating import RatingStore
    from corc.patterns import analyze_prompts, format_prompt_report

    paths, _, _, _, _, _ = _get_all()
    rs = RatingStore(paths["ratings_dir"])
    ratings = rs.read_all()
    report = analyze_prompts(ratings, role)
    click.echo(format_prompt_report(report))


@analyze.command("planning")
def analyze_planning_cmd():
    """Show which spec structures produce better outcomes.

    Analyzes checklist size, dependency count, context bundle size,
    and done-when specificity against task quality scores.

    Examples:
      corc analyze planning
    """
    from corc.rating import RatingStore
    from corc.patterns import analyze_planning, format_planning_report

    paths, _, _, _, _, _ = _get_all()
    rs = RatingStore(paths["ratings_dir"])
    ratings = rs.read_all()
    report = analyze_planning(ratings)
    click.echo(format_planning_report(report))


@analyze.command("retries")
@click.option(
    "--flagged-only",
    is_flag=True,
    help="Show only task types flagged for investigation",
)
def analyze_retries_cmd(flagged_only):
    """Show retry statistics and adaptive retry settings.

    Displays first-attempt success rate by task type and role,
    current adaptive retry counts, and flagged task types.

    Examples:
      corc analyze retries
      corc analyze retries --flagged-only
    """
    from corc.adaptive_retry import (
        AdaptiveRetryTracker,
        compute_retry_statistics,
        format_retry_statistics,
    )

    paths = get_paths()
    tracker = AdaptiveRetryTracker(paths["retry_outcomes"])
    report = compute_retry_statistics(tracker)

    if flagged_only:
        flagged = report["flagged"]
        if not flagged:
            click.echo("No task types flagged for investigation.")
            return
        # Show only flagged subset
        report["stats"] = flagged
        click.echo(format_retry_statistics(report))
    else:
        click.echo(format_retry_statistics(report))


# --- Rating ---


@cli.command()
@click.argument("task_id", required=False)
@click.option(
    "--auto", "auto_flag", is_flag=True, help="Score all unscored completed tasks"
)
@click.option(
    "--no-claude",
    "no_claude",
    is_flag=True,
    help="Use heuristic scoring instead of claude -p",
)
def rate(task_id, auto_flag, no_claude):
    """Score a completed task across 7 dimensions.

    Examples:
      corc rate TASK_ID                  Score a specific task
      corc rate TASK_ID --no-claude      Score using heuristics only
      corc rate --auto                   Score all unscored completed tasks
      corc rate --auto --no-claude       Batch score without claude -p
    """
    from corc.rating import RatingEngine, RatingStore, format_rating

    paths, _, ws, al, sl, _ = _get_all()
    rs = RatingStore(paths["ratings_dir"])
    spec_path = paths["root"] / "SPEC.md"

    engine = RatingEngine(
        store=rs,
        work_state=ws,
        audit_log=al,
        session_logger=sl,
        spec_path=spec_path if spec_path.exists() else None,
    )

    if auto_flag:
        use_claude = not no_claude
        # For auto mode, we use evaluator by default but fall back per-task
        all_tasks = ws.list_tasks(status="completed")
        new_ratings = []
        for task in all_tasks:
            tid = task["id"]
            if not rs.is_rated(tid):
                try:
                    rating = engine.rate_task(tid, use_claude=use_claude)
                    new_ratings.append(rating)
                    click.echo(format_rating(rating))
                    click.echo()
                except ValueError as e:
                    click.echo(f"Skipping {tid}: {e}", err=True)

        click.echo(f"Rated {len(new_ratings)} task(s).")
        return

    if not task_id:
        click.echo("Error: provide a TASK_ID or use --auto.", err=True)
        sys.exit(1)

    try:
        rating = engine.rate_task(task_id, use_claude=not no_claude)
        click.echo(format_rating(rating))
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.group()
def ratings():
    """View rating trends and drill-downs."""
    pass


@ratings.command("trend")
@click.option("--last", "last_n", default=30, help="Show last N ratings")
def ratings_trend(last_n):
    """Show quality trends over time.

    Examples:
      corc ratings trend
      corc ratings trend --last 10
    """
    from corc.rating import RatingStore, format_trend

    paths = get_paths()
    rs = RatingStore(paths["ratings_dir"])
    trend = rs.get_trend(last_n=last_n)
    click.echo(format_trend(trend))


@ratings.command("dimension")
@click.argument("name")
def ratings_dimension(name):
    """Drill-down into a specific scoring dimension.

    Examples:
      corc ratings dimension correctness
      corc ratings dimension efficiency
    """
    from corc.rating import RatingStore, format_dimension_drilldown

    paths = get_paths()
    rs = RatingStore(paths["ratings_dir"])
    try:
        entries = rs.get_by_dimension(name)
        click.echo(format_dimension_drilldown(name, entries))
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# --- Retrospective ---


@cli.command()
@click.argument("project_name")
@click.option(
    "--estimate",
    "cost_estimate",
    default=None,
    type=float,
    help="Cost estimate in USD for comparison",
)
def retro(project_name, cost_estimate):
    """Generate a project-level retrospective.

    Produces a structured analysis: what went well, what didn't, cost vs.
    estimate, quality trends, top findings, and recommendations. The
    retrospective is saved to the knowledge store as a task-outcome document.

    Examples:
      corc retro myproject
      corc retro myproject --estimate 50.00
    """
    from corc.rating import RatingStore
    from corc.retro import (
        generate_retrospective,
        format_retrospective,
        save_retrospective,
    )

    paths, _, ws, al, _, ks = _get_all()
    rs = RatingStore(paths["ratings_dir"])

    retro_result = generate_retrospective(
        project_name=project_name,
        work_state=ws,
        audit_log=al,
        rating_store=rs,
        cost_estimate_usd=cost_estimate,
    )

    # Display the retrospective
    click.echo(format_retrospective(retro_result))

    # Save to knowledge store
    doc_id = save_retrospective(retro_result, ks)
    click.echo()
    click.echo(f"Retrospective saved to knowledge store: {doc_id}")


# --- Blacklist Management ---


@cli.group()
def blacklist():
    """Manage the agent blacklist.

    The blacklist contains entries agents should NOT do. Entries come in two forms:

    - Advisory (plain text): Injected into agent context as guidance.
    - Enforced (block_command: prefix): Auto-generates PreToolUse hooks that
      block matching Bash commands at the tool-use layer.
    """
    pass


@blacklist.command("list")
def blacklist_list():
    """List all blacklist entries, showing which are enforced.

    Examples:
      corc blacklist list
    """
    from corc.blacklist import load_blacklist

    paths = get_paths()
    entries = load_blacklist(paths["root"])

    if not entries:
        click.echo("No blacklist entries found.")
        return

    for entry in entries:
        if entry.is_block_command:
            reason_str = f" ({entry.reason})" if entry.reason else ""
            click.echo(f"  [enforced]  block_command: {entry.pattern}{reason_str}")
        else:
            click.echo(f"  [advisory]  {entry.raw}")


@blacklist.command("add")
@click.argument("entry")
@click.option("--reason", default="", help="Reason for this blacklist entry")
@click.option("--section", default=None, help="Section heading to add under")
def blacklist_add(entry, reason, section):
    """Add an entry to the blacklist.

    Use the 'block_command:' prefix for entries that should be machine-enforced
    via PreToolUse hooks. Plain text entries are advisory (prompt-only).

    Examples:
      corc blacklist add "Never use eval()" --reason "security"
      corc blacklist add "block_command: git push --force" --reason "history"
      corc blacklist add "block_command: rm -rf /" --reason "dangerous" --section "Process"
    """
    from corc.blacklist import add_entry, sync_blacklist_hooks, parse_blacklist

    paths = get_paths()
    formatted = add_entry(paths["root"], entry, reason=reason, section=section)
    click.echo(f"Added: {formatted}")

    # If the entry is a block_command, regenerate hooks
    if entry.strip().lower().startswith("block_command:"):
        written = sync_blacklist_hooks(paths["root"])
        if written:
            click.echo(f"Generated/updated {len(written)} hook file(s)")


@blacklist.command("remove")
@click.argument("entry")
def blacklist_remove(entry):
    """Remove an entry from the blacklist.

    Matches by substring — removes the first matching entry.

    Examples:
      corc blacklist remove "Never use eval()"
      corc blacklist remove "git push --force"
    """
    from corc.blacklist import remove_entry, sync_blacklist_hooks

    paths = get_paths()
    removed = remove_entry(paths["root"], entry)

    if not removed:
        click.echo(f"No matching entry found for: {entry}", err=True)
        sys.exit(1)

    click.echo(f"Removed entry matching: {entry}")

    # Regenerate hooks (might need cleanup if a block_command was removed)
    written = sync_blacklist_hooks(paths["root"])
    if written:
        click.echo(f"Updated {len(written)} hook file(s)")


@blacklist.command("sync-hooks")
def blacklist_sync_hooks():
    """Regenerate hooks from current blacklist entries.

    Reads .corc/blacklist.md, finds all block_command: entries, and
    regenerates .claude/hooks/enforce-blacklist.sh and updates
    .claude/settings.json accordingly.

    This is automatically called after 'blacklist add' and 'blacklist remove',
    but can be run manually if the blacklist file was edited directly.

    Examples:
      corc blacklist sync-hooks
    """
    from corc.blacklist import sync_blacklist_hooks, load_blacklist, get_block_commands

    paths = get_paths()
    entries = load_blacklist(paths["root"])
    blocked = get_block_commands(entries)

    written = sync_blacklist_hooks(paths["root"])

    if blocked:
        click.echo(f"Found {len(blocked)} block_command entries")
        for entry in blocked:
            click.echo(f"  • {entry.pattern}")
        click.echo(f"Generated/updated {len(written)} hook file(s)")
    else:
        click.echo("No block_command entries found")
        if written:
            click.echo(f"Cleaned up {len(written)} hook file(s)")


if __name__ == "__main__":
    cli()
