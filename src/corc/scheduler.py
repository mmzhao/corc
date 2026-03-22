"""Scheduler module — resolves DAG, determines ready tasks, manages parallelism.

Pure query module: reads work state, writes nothing.
Can be called standalone: `corc tasks --ready`
"""

from corc.state import WorkState


def get_ready_tasks(state: WorkState, parallel_limit: int = 1) -> list[dict]:
    """Get tasks ready for dispatch, respecting the parallel limit.

    Calculates available slots as: parallel_limit - currently_running_or_assigned.
    Returns up to that many ready (pending + all deps met) tasks.
    """
    running = state.list_tasks(status="running")
    assigned = state.list_tasks(status="assigned")
    in_flight = len(running) + len(assigned)

    slots = max(0, parallel_limit - in_flight)
    if slots == 0:
        return []

    ready = state.get_ready_tasks()
    return ready[:slots]
