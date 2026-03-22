"""DAG visualization — render task dependency graph with color-coded status.

Reads work state, computes topological layout, and renders either:
  - ASCII art with ANSI colors for terminal display
  - Mermaid markdown for rendering in Obsidian / GitHub
"""

import json
from collections import defaultdict


# ── Status display configuration ──────────────────────────────────────────

STATUS_ICONS = {
    "completed": "✅",
    "running": "🔄",
    "ready": "⬚",
    "blocked": "◻",
    "failed": "❌",
    "assigned": "🔄",
    "handed_off": "↗",
    "pending": "⬚",
}

STATUS_ANSI = {
    "completed": "\033[32m",   # green
    "running": "\033[33m",     # yellow
    "ready": "\033[0m",        # default/white
    "blocked": "\033[90m",     # gray
    "failed": "\033[31m",      # red
    "assigned": "\033[33m",    # yellow
    "handed_off": "\033[33m",  # yellow
    "pending": "\033[0m",      # default
}

RESET = "\033[0m"


# ── Core computation ──────────────────────────────────────────────────────

def compute_effective_status(tasks: list[dict]) -> dict[str, str]:
    """Map each task to a display status.

    Pending tasks become 'ready' if all deps complete, otherwise 'blocked'.
    """
    task_ids = {t["id"] for t in tasks}
    completed_ids = {t["id"] for t in tasks if t["status"] == "completed"}

    statuses: dict[str, str] = {}
    for t in tasks:
        status = t["status"]
        if status == "pending":
            deps = t.get("depends_on", [])
            if isinstance(deps, str):
                deps = json.loads(deps)
            relevant = [d for d in deps if d in task_ids]
            status = "ready" if all(d in completed_ids for d in relevant) else "blocked"
        statuses[t["id"]] = status
    return statuses


def assign_levels(tasks: list[dict]) -> dict[str, int]:
    """Assign topological level to each task. Level 0 = no dependencies."""
    task_ids = {t["id"] for t in tasks}
    task_map = {t["id"]: t for t in tasks}
    levels: dict[str, int] = {}

    def _level(tid: str) -> int:
        if tid in levels:
            return levels[tid]
        task = task_map.get(tid)
        if not task:
            levels[tid] = 0
            return 0
        deps = task.get("depends_on", [])
        if isinstance(deps, str):
            deps = json.loads(deps)
        valid = [d for d in deps if d in task_ids]
        if not valid:
            levels[tid] = 0
        else:
            levels[tid] = max(_level(d) for d in valid) + 1
        return levels[tid]

    for t in tasks:
        _level(t["id"])
    return levels


def build_adjacency(tasks: list[dict]):
    """Return (forward, reverse) adjacency dicts. Forward: parent→[children]."""
    task_ids = {t["id"] for t in tasks}
    forward: dict[str, list[str]] = defaultdict(list)
    reverse: dict[str, list[str]] = defaultdict(list)
    for t in tasks:
        deps = t.get("depends_on", [])
        if isinstance(deps, str):
            deps = json.loads(deps)
        for dep in deps:
            if dep in task_ids:
                forward[dep].append(t["id"])
                reverse[t["id"]].append(dep)
    return dict(forward), dict(reverse)


def assign_rows(tasks, levels, forward, reverse):
    """Assign display rows to tasks, keeping connected tasks close."""
    task_map = {t["id"]: t for t in tasks}
    level_groups: dict[int, list[str]] = defaultdict(list)
    for t in tasks:
        level_groups[levels[t["id"]]].append(t["id"])

    max_level = max(levels.values()) if levels else 0

    # Sort within levels by name for deterministic output
    for lvl in level_groups:
        level_groups[lvl].sort(key=lambda tid: task_map[tid]["name"])

    task_row: dict[str, int] = {}

    # Level-0 tasks get sequential rows
    for i, tid in enumerate(level_groups.get(0, [])):
        task_row[tid] = i

    # Subsequent levels: place near parents
    for lvl in range(1, max_level + 1):
        group = level_groups.get(lvl, [])
        preferences = []
        for tid in group:
            parents = reverse.get(tid, [])
            parent_rows = [task_row[p] for p in parents if p in task_row]
            pref = sum(parent_rows) / len(parent_rows) if parent_rows else float("inf")
            preferences.append((pref, tid))
        preferences.sort()

        level_taken: set[int] = set()
        for pref_val, tid in preferences:
            target = round(pref_val) if pref_val != float("inf") else (max(task_row.values(), default=-1) + 1)
            if target < 0:
                target = 0
            if target not in level_taken:
                task_row[tid] = target
                level_taken.add(target)
            else:
                for offset in range(1, 200):
                    for cand in (target + offset, target - offset):
                        if cand >= 0 and cand not in level_taken:
                            task_row[tid] = cand
                            level_taken.add(cand)
                            break
                    else:
                        continue
                    break

    # Re-normalise to contiguous 0-indexed rows
    all_rows = sorted(set(task_row.values()))
    remap = {r: i for i, r in enumerate(all_rows)}
    return {tid: remap[r] for tid, r in task_row.items()}


def compute_progress(tasks: list[dict]) -> dict:
    """Compute progress statistics from task list."""
    statuses = compute_effective_status(tasks)
    total = len(tasks)
    counts = {"completed": 0, "running": 0, "ready": 0, "blocked": 0, "failed": 0}
    for s in statuses.values():
        if s in counts:
            counts[s] += 1
        elif s in ("assigned", "handed_off"):
            counts["running"] += 1
    pct = (counts["completed"] * 100 // total) if total else 0
    return {"total": total, "percent": pct, **counts}


def _progress_bar(pct: int, width: int = 20) -> str:
    filled = int(width * pct / 100)
    return "█" * filled + "░" * (width - filled)


# ── ASCII rendering ───────────────────────────────────────────────────────

class _Grid:
    """Mutable 2-D character canvas with optional per-cell ANSI colour."""

    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self.chars = [[" "] * width for _ in range(height)]
        self.color = [[None] * width for _ in range(height)]

    def put(self, r: int, c: int, ch: str, col=None):
        if 0 <= r < self.h and 0 <= c < self.w:
            self.chars[r][c] = ch
            if col is not None:
                self.color[r][c] = col

    def puts(self, r: int, c: int, text: str, col=None):
        for i, ch in enumerate(text):
            self.put(r, c + i, ch, col)

    def get(self, r: int, c: int) -> str:
        if 0 <= r < self.h and 0 <= c < self.w:
            return self.chars[r][c]
        return " "

    def render(self, use_color: bool = True) -> str:
        lines: list[str] = []
        for r in range(self.h):
            parts: list[str] = []
            cur_col = None
            for c in range(self.w):
                cc = self.color[r][c] if use_color else None
                if cc != cur_col:
                    if cur_col is not None:
                        parts.append(RESET)
                    if cc is not None:
                        parts.append(cc)
                    cur_col = cc
                parts.append(self.chars[r][c])
            if cur_col is not None:
                parts.append(RESET)
            lines.append("".join(parts).rstrip())
        # strip trailing blank lines
        while lines and not lines[-1].strip():
            lines.pop()
        return "\n".join(lines)


def render_ascii_dag(tasks: list[dict], use_color: bool = True) -> str:
    """Render a left-to-right ASCII DAG with ANSI status colours.

    Layout: columns = topological levels, rows assigned to minimise edge length.
    Edges drawn with box-drawing characters.
    """
    if not tasks:
        return "No tasks found.\n"

    task_map = {t["id"]: t for t in tasks}
    statuses = compute_effective_status(tasks)
    levels = assign_levels(tasks)
    forward, reverse = build_adjacency(tasks)
    rows = assign_rows(tasks, levels, forward, reverse)

    max_level = max(levels.values()) if levels else 0
    total_rows = (max(rows.values()) + 1) if rows else 1

    # group by level
    level_groups: dict[int, list[str]] = defaultdict(list)
    for t in tasks:
        level_groups[levels[t["id"]]].append(t["id"])

    # ── column geometry ──
    MARGIN = 2
    ICON_DISPLAY = 2  # emoji occupies ~2 columns in terminal
    EDGE_GAP = 6      # chars reserved between task columns for edge routing

    col_name_width: dict[int, int] = {}   # max task-name length in each level
    for lvl in range(max_level + 1):
        w = 0
        for tid in level_groups.get(lvl, []):
            w = max(w, len(task_map[tid]["name"]))
        col_name_width[lvl] = w

    # start-x of each level's task label
    level_x: dict[int, int] = {}
    x = MARGIN
    for lvl in range(max_level + 1):
        level_x[lvl] = x
        x += col_name_width[lvl] + 1 + ICON_DISPLAY  # name + space + icon
        if lvl < max_level:
            x += EDGE_GAP

    grid_w = x + MARGIN
    # Each task-row maps to 2 grid rows (task line + edge-routing line)
    grid_h = total_rows * 2
    g = _Grid(grid_w, grid_h)

    # ── place task labels ──
    for tid, row in rows.items():
        lvl = levels[tid]
        name = task_map[tid]["name"]
        status = statuses[tid]
        icon = STATUS_ICONS.get(status, "?")
        ansi = STATUS_ANSI.get(status)
        gr = row * 2
        gx = level_x[lvl]
        g.puts(gr, gx, name, ansi)
        g.puts(gr, gx + len(name) + 1, icon)

    # ── draw edges ──
    for src_task in tasks:
        src = src_task["id"]
        src_lvl = levels[src]
        src_row = rows[src]
        src_name_len = len(task_map[src]["name"])
        children = forward.get(src, [])
        if not children:
            continue

        # Right edge of source label
        src_right = level_x[src_lvl] + src_name_len + 1 + ICON_DISPLAY

        for child in children:
            dst_lvl = levels[child]
            dst_row = rows[child]
            dst_left = level_x[dst_lvl]

            sgr = src_row * 2
            dgr = dst_row * 2

            if sgr == dgr:
                # ── same-row edge: horizontal arrow ──
                for c in range(src_right, dst_left - 1):
                    if g.get(sgr, c) == " ":
                        g.put(sgr, c, "─")
                g.put(sgr, dst_left - 1, "►")
            else:
                # ── cross-row edge: horizontal ─ corner ─ vertical ─ corner ─ horizontal ──
                # Route through the edge gap after source level
                route_x = level_x[src_lvl] + col_name_width[src_lvl] + 1 + ICON_DISPLAY + 2

                # Horizontal from source to route column
                for c in range(src_right, route_x):
                    ch = g.get(sgr, c)
                    if ch == " ":
                        g.put(sgr, c, "─")
                    elif ch == "│":
                        g.put(sgr, c, "┼")

                # Corner at source row
                existing = g.get(sgr, route_x)
                if dgr > sgr:
                    if existing in ("│", "└", "┌"):
                        g.put(sgr, route_x, "├")
                    elif existing == "┘":
                        g.put(sgr, route_x, "┤")
                    else:
                        g.put(sgr, route_x, "┐")
                else:
                    if existing in ("│", "┌", "└"):
                        g.put(sgr, route_x, "├")
                    elif existing == "┐":
                        g.put(sgr, route_x, "┤")
                    else:
                        g.put(sgr, route_x, "┘")

                # Vertical segment
                lo, hi = (min(sgr, dgr), max(sgr, dgr))
                for r in range(lo + 1, hi):
                    ch = g.get(r, route_x)
                    if ch == " ":
                        g.put(r, route_x, "│")
                    elif ch == "─":
                        g.put(r, route_x, "┼")

                # Corner at destination row
                existing = g.get(dgr, route_x)
                if dgr > sgr:
                    if existing in ("│", "┐", "┘"):
                        g.put(dgr, route_x, "├")
                    else:
                        g.put(dgr, route_x, "└")
                else:
                    if existing in ("│", "┐", "┘"):
                        g.put(dgr, route_x, "├")
                    else:
                        g.put(dgr, route_x, "┌")

                # Horizontal from route column to destination
                for c in range(route_x + 1, dst_left - 1):
                    ch = g.get(dgr, c)
                    if ch == " ":
                        g.put(dgr, c, "─")
                    elif ch == "│":
                        g.put(dgr, c, "┼")
                g.put(dgr, dst_left - 1, "►")

    # ── assemble final output ──
    dag_art = g.render(use_color)

    legend = "  ✅ complete  🔄 running  ⬚ ready  ◻ blocked  ❌ failed"

    progress = compute_progress(tasks)
    bar = _progress_bar(progress["percent"])
    prog = (
        f"  Progress: {progress['completed']}/{progress['total']} tasks "
        f"({progress['percent']}%)  {bar}"
    )
    status_parts: list[str] = []
    if progress["running"]:
        status_parts.append(f"Running: {progress['running']}")
    if progress["ready"]:
        status_parts.append(f"Ready: {progress['ready']}")
    if progress["blocked"]:
        status_parts.append(f"Blocked: {progress['blocked']}")
    if progress["failed"]:
        status_parts.append(f"Failed: {progress['failed']}")
    if status_parts:
        prog += f"\n  {' | '.join(status_parts)}"

    return f"\n{dag_art}\n\n{legend}\n\n{prog}\n"


# ── Mermaid rendering ─────────────────────────────────────────────────────

def render_mermaid(tasks: list[dict]) -> str:
    """Render the task DAG as a Mermaid flowchart (LR direction)."""
    if not tasks:
        return "graph LR\n  empty[No tasks]\n"

    statuses = compute_effective_status(tasks)
    task_ids = {t["id"] for t in tasks}

    MERMAID_STYLE = {
        "completed": "fill:#22c55e,color:#fff",
        "running": "fill:#eab308,color:#000",
        "ready": "fill:#fff,stroke:#333,color:#000",
        "blocked": "fill:#9ca3af,color:#fff",
        "failed": "fill:#ef4444,color:#fff",
        "assigned": "fill:#eab308,color:#000",
        "handed_off": "fill:#eab308,color:#000",
    }

    def safe(tid: str) -> str:
        return tid.replace("-", "_").replace(" ", "_")

    lines = ["graph LR"]

    # Nodes
    for t in tasks:
        tid = t["id"]
        name = t["name"].replace('"', "'")
        icon = STATUS_ICONS.get(statuses[tid], "?")
        lines.append(f'  {safe(tid)}["{name} {icon}"]')

    # Edges
    for t in tasks:
        deps = t.get("depends_on", [])
        if isinstance(deps, str):
            deps = json.loads(deps)
        for dep in deps:
            if dep in task_ids:
                lines.append(f"  {safe(dep)} --> {safe(t['id'])}")

    # Styles
    style_groups: dict[str, list[str]] = defaultdict(list)
    for t in tasks:
        style_groups[statuses[t["id"]]].append(safe(t["id"]))
    for status, nids in style_groups.items():
        style = MERMAID_STYLE.get(status, "")
        if style:
            for nid in nids:
                lines.append(f"  style {nid} {style}")

    # Progress as comment
    progress = compute_progress(tasks)
    lines.append(
        f"  %% Progress: {progress['completed']}/{progress['total']} "
        f"({progress['percent']}%)"
    )

    return "\n".join(lines) + "\n"
