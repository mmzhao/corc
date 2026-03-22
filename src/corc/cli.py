"""CORC CLI — essential commands for Phase 0."""

import json
import sys
import time
import uuid

import click

from corc.config import get_paths
from corc.mutations import MutationLog
from corc.state import WorkState
from corc.audit import AuditLog
from corc.sessions import SessionLogger
from corc.knowledge import KnowledgeStore
from corc.pause import write_pause_lock, remove_pause_lock, read_pause_lock, is_paused
from corc.dag import render_ascii_dag, render_mermaid
from corc.context import assemble_context
from corc.daemon import Daemon, stop_daemon
from corc.dispatch import get_dispatcher, Constraints
from corc.validate import run_validations
from corc.templates import get_template, render_template, list_types
from corc.lint_done_when import lint_done_when


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
@click.option("--context", "context_bundle", default="", help="Comma-separated file paths for context bundle")
@click.option("--checklist", default="", help="Comma-separated checklist items")
@click.option("--strict", is_flag=True, help="Reject subjective done_when criteria")
def task_create(name, done_when, description, role, depends_on, context_bundle, checklist, strict):
    """Create a new task."""
    # Lint done_when criteria
    lint_result = lint_done_when(done_when)
    if lint_result.warnings:
        for warning in lint_result.warnings:
            click.echo(f"Warning: done_when: {warning}", err=True)
        if strict:
            click.echo("Aborted: --strict rejects subjective done_when criteria.", err=True)
            sys.exit(1)

    paths, ml, ws, al, sl, ks = _get_all()
    task_id = str(uuid.uuid4())[:8]
    deps = [d.strip() for d in depends_on.split(",") if d.strip()]
    bundle = [b.strip() for b in context_bundle.split(",") if b.strip()]
    cl = [c.strip() for c in checklist.split(",") if c.strip()]

    ml.append("task_created", {
        "id": task_id,
        "name": name,
        "description": description,
        "role": role,
        "depends_on": deps,
        "done_when": done_when,
        "checklist": cl,
        "context_bundle": bundle,
    }, reason=f"Task created via CLI")

    al.log("task_created", task_id=task_id, name=name)
    click.echo(f"Created task {task_id}: {name}")


@task.command("list")
@click.option("--status", default=None, help="Filter by status")
@click.option("--ready", is_flag=True, help="Show only ready tasks")
def task_list(status, ready):
    """List tasks."""
    _, _, ws, _, _, _ = _get_all()
    if ready:
        tasks = ws.get_ready_tasks()
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
        click.echo(f"  [{t['status']:>10}] {t['id']}  {t['name']}{dep_str}")
        click.echo(f"             done_when: {t['done_when']}")


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
    click.echo(f"Role: {t.get('role', 'unset')}")
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

    if t["status"] not in ("pending", "failed"):
        click.echo(f"Task {task_id} is {t['status']}, cannot dispatch.")
        return

    # Assemble context
    context = assemble_context(t, paths["root"])
    click.echo(f"Assembled context: {len(context)} chars")

    # Determine constraints from role
    constraints = Constraints()
    role = t.get("role", "implementer")
    system_prompt = f"You are a {role} working on task '{t['name']}'.\n\n{context}"

    prompt = (
        f"Complete the following task.\n\n"
        f"Done when: {t['done_when']}\n\n"
        f"Work in the current directory. Write tests alongside implementation. "
        f"Commit your changes with a clear message referencing the task."
    )

    # Update state
    attempt = sl.get_latest_attempt(task_id) + 1
    ml.append("task_started", {"attempt": attempt}, reason="Dispatched via CLI", task_id=task_id)
    al.log("task_dispatched", task_id=task_id, role=role, attempt=attempt)
    sl.log_dispatch(task_id, attempt, prompt, system_prompt, constraints.allowed_tools, constraints.max_budget_usd)

    click.echo(f"Dispatching {task_id} (attempt {attempt}) via {provider}...")

    # Dispatch
    dispatcher = get_dispatcher(provider)
    result = dispatcher.dispatch(prompt, system_prompt, constraints)

    # Log result
    sl.log_output(task_id, attempt, result.output, result.exit_code, result.duration_s)
    al.log("task_dispatch_complete", task_id=task_id, exit_code=result.exit_code,
           duration_s=result.duration_s, attempt=attempt)

    click.echo(f"Agent finished in {result.duration_s:.1f}s (exit code {result.exit_code})")
    click.echo(f"Session log: {sl.session_path(task_id, attempt)}")

    if result.exit_code != 0:
        click.echo("Agent exited with error.")
        ml.append("task_failed", {"attempt": attempt, "exit_code": result.exit_code},
                   reason=f"Agent exited with code {result.exit_code}", task_id=task_id)
        al.log("task_failed", task_id=task_id, attempt=attempt, exit_code=result.exit_code)
        return

    # Output truncated preview
    output_preview = result.output[:500] + "..." if len(result.output) > 500 else result.output
    click.echo(f"\n--- Output preview ---\n{output_preview}\n---")

    click.echo(f"\nTask dispatched. Review output and run: corc task complete {task_id}")


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
    click.echo(f"Tasks: {completed}/{total} complete ({100*completed//total if total else 0}%)")

    for status_name in ["completed", "running", "pending", "failed", "blocked", "handed_off"]:
        group = by_status.get(status_name, [])
        if not group:
            continue
        icon = {"completed": "✅", "running": "🔄", "pending": "⬚", "failed": "❌", "blocked": "◻", "handed_off": "↗"}.get(status_name, "?")
        names = ", ".join(t["name"] for t in group)
        click.echo(f"  {icon} {status_name}: {names}")

    ready = ws.get_ready_tasks()
    if ready:
        click.echo(f"\nReady to dispatch: {', '.join(t['name'] for t in ready)}")

    events_today = al.read_today()
    if events_today:
        click.echo(f"\nEvents today: {len(events_today)}")


# --- DAG Visualization ---

@cli.command()
@click.option("--mermaid", is_flag=True, help="Output Mermaid markdown instead of ASCII")
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
    doc_id = ks.add(file_path=file_path, doc_type=doc_type, project=project, tags=tag_list)
    click.echo(f"Added document {doc_id}")


@knowledge.command("search")
@click.argument("query")
@click.option("--limit", default=10)
@click.option("--type", "doc_type", default=None)
@click.option("--project", default=None)
@click.option("--mode", type=click.Choice(["hybrid", "keyword", "semantic"]),
              default="hybrid", help="Search mode (default: hybrid)")
def knowledge_search(query, limit, doc_type, project, mode):
    """Search the knowledge store (hybrid search by default)."""
    _, _, _, _, _, ks = _get_all()
    if mode == "keyword":
        results = ks.search(query, limit=limit, doc_type=doc_type, project=project)
    elif mode == "semantic":
        results = ks.semantic_search(query, limit=limit, doc_type=doc_type, project=project)
    else:
        results = ks.hybrid_search(query, limit=limit, doc_type=doc_type, project=project)
    if not results:
        click.echo("No results.")
        return
    for r in results:
        click.echo(f"  [{r.get('type', '?'):>12}] {r['id']}  {r['title']}  (score: {r.get('score', 'N/A'):.4f})")


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


# --- Watch (TUI v0) ---

@cli.command()
@click.option("--last", "last_n", default=20, help="Show last N events")
def watch(last_n):
    """Live event stream (TUI v0)."""
    try:
        from rich.live import Live
        from rich.table import Table
        from rich.console import Console
        _watch_rich(last_n)
    except ImportError:
        _watch_plain(last_n)


def _watch_rich(last_n):
    from rich.live import Live
    from rich.text import Text
    from rich.panel import Panel
    from rich.console import Console

    paths = get_paths()
    al = AuditLog(paths["events_dir"])
    console = Console()

    STATUS_COLORS = {
        "task_created": "cyan",
        "task_dispatched": "yellow",
        "task_dispatch_complete": "yellow",
        "task_completed": "green",
        "task_failed": "red",
        "escalation": "red bold",
        "pause": "red",
        "resume": "green",
    }

    seen = set()

    with Live(console=console, refresh_per_second=2) as live:
        while True:
            events = al.read_today()
            display_events = events[-last_n:]

            lines = []
            for e in display_events:
                ts = e.get("timestamp", "")[:19].replace("T", " ")
                etype = e.get("event_type", "unknown")
                tid = e.get("task_id", "")[:8] if e.get("task_id") else ""
                color = STATUS_COLORS.get(etype, "white")

                extra = ""
                if e.get("duration_s"):
                    extra += f" ({e['duration_s']:.1f}s)"
                if e.get("exit_code") is not None and e["exit_code"] != 0:
                    extra += f" exit={e['exit_code']}"

                line = Text()
                line.append(f"{ts} ", style="dim")
                line.append(f"{etype:<25}", style=color)
                line.append(f" {tid}", style="bold")
                line.append(extra, style="dim")
                lines.append(line)

            output = Text("\n").join(lines) if lines else Text("No events yet. Waiting...", style="dim")
            panel = Panel(output, title="[bold]CORC Events[/bold]", subtitle="Ctrl+C to exit")
            live.update(panel)
            time.sleep(2)


def _watch_plain(last_n):
    paths = get_paths()
    al = AuditLog(paths["events_dir"])
    click.echo("Watching events (install 'rich' for TUI). Ctrl+C to exit.")
    seen_count = 0
    while True:
        events = al.read_today()
        if len(events) > seen_count:
            for e in events[seen_count:]:
                ts = e.get("timestamp", "")[:19]
                click.echo(f"{ts} {e.get('event_type', '?'):>25} {e.get('task_id', '')[:8]}")
            seen_count = len(events)
        time.sleep(2)


# --- Daemon ---

@cli.command("start")
@click.option("--parallel", default=1, type=int, help="Max concurrent agents (default: 1)")
@click.option("--task", "task_id", default=None, help="Run one specific task, then stop")
@click.option("--once", is_flag=True, help="Process one ready task, then stop")
@click.option("--provider", default="claude-code", help="Dispatch provider")
@click.option("--poll-interval", default=5.0, type=float, help="Seconds between polls (default: 5)")
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
    click.echo("Press Ctrl+C to stop.\n")

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
    click.echo("Daemon stopped.")


@cli.command("stop")
def stop_cmd():
    """Graceful shutdown (in-flight tasks finish)."""
    paths = get_paths()
    if stop_daemon(paths["root"]):
        click.echo("Stop signal sent to daemon. In-flight tasks will finish.")
    else:
        click.echo("No running daemon found.")


# --- Self-test ---

@cli.command("self-test")
def self_test():
    """Run orchestrator self-tests."""
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-x", "-q"],
        capture_output=True, text=True, cwd=str(get_paths()["root"]),
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
        events = al.read_today()[-last_n:]

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


if __name__ == "__main__":
    cli()
