"""Scheduler module — resolves DAG, determines ready tasks, manages parallelism.

Pure query module: reads work state, writes nothing.
Can be called standalone: `corc tasks --ready`

Tasks are sorted by priority (lower number = higher priority, default 100)
before dispatch, so urgent tasks (e.g. bug fixes at priority 10) are
dispatched before normal tasks.
"""

from corc.state import WorkState


def get_ready_tasks(state: WorkState, parallel_limit: int = 1) -> list[dict]:
    """Get tasks ready for dispatch, respecting the parallel limit.

    Calculates available slots as: parallel_limit - currently_running_or_assigned.
    Returns up to that many ready (pending + all deps met) tasks,
    sorted by priority ascending (lower number = higher priority).
    """
    running = state.list_tasks(status="running")
    assigned = state.list_tasks(status="assigned")
    in_flight = len(running) + len(assigned)

    slots = max(0, parallel_limit - in_flight)
    if slots == 0:
        return []

    # get_ready_tasks() already returns tasks sorted by priority
    ready = state.get_ready_tasks()
    return ready[:slots]
