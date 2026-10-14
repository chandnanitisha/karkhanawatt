"""
baseline.py
------------
A First-Come-First-Serve (FCFS) baseline scheduler.

This is deliberately NOT an OR model. It exists only so the project can
show "before OR" vs "after OR" and quantify the improvement CP-SAT gives.

Logic:
    - Jobs are topologically sorted so that a predecessor is always
      scheduled before its dependent job, regardless of arrival_order.
    - Among jobs that are currently eligible (all predecessors finished),
      ties are broken by arrival_order (FCFS priority).
    - Each job is placed on its assigned machine as early as that machine
      becomes free. No look-ahead, no optimisation.
    - Precedence is respected: a dependent job starts no earlier than its
      predecessor's completion time.

Dependency validation:
    - Missing predecessor IDs raise ValueError.
    - Self-dependencies raise ValueError.
    - Circular dependencies raise ValueError.

Returns a schedule DataFrame with start_time / end_time per job, plus a
flag for whether each job missed its deadline.
"""

import pandas as pd
from collections import deque


def _validate_and_build_graph(jobs: list) -> tuple[dict, dict]:
    """
    Validate the dependency graph and return adjacency structures.

    Returns:
        predecessor_of: {job_id -> predecessor_id or None}
        successors_of:  {job_id -> [list of dependent job_ids]}
    """
    valid_ids = {str(job["job_id"]) for job in jobs}
    predecessor_of = {}
    successors_of = {str(job["job_id"]): [] for job in jobs}

    for job in jobs:
        jid = str(job["job_id"])
        pred = job.get("predecessor")

        if pred is not None and not pd.isna(pred) and str(pred).strip():
            pred = str(pred).strip()
            if pred == jid:
                raise ValueError(
                    f"Job '{jid}' lists itself as its own predecessor. "
                    "Remove the self-dependency."
                )
            if pred not in valid_ids:
                raise ValueError(
                    f"Job '{jid}' references predecessor '{pred}', "
                    "which does not exist in the job list."
                )
            predecessor_of[jid] = pred
            successors_of[pred].append(jid)
        else:
            predecessor_of[jid] = None

    # Detect cycles using Kahn's algorithm (same traversal used for sort).
    in_degree = {jid: (1 if predecessor_of[jid] is not None else 0) for jid in valid_ids}
    queue = deque(jid for jid in valid_ids if in_degree[jid] == 0)
    visited_count = 0
    while queue:
        node = queue.popleft()
        visited_count += 1
        for successor in successors_of[node]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue.append(successor)

    if visited_count != len(valid_ids):
        # Find which jobs are in the cycle for a helpful message.
        cycle_jobs = [jid for jid in valid_ids if in_degree[jid] > 0]
        raise ValueError(
            f"Circular dependency detected among job(s): {sorted(cycle_jobs)}. "
            "Remove the cycle before scheduling."
        )

    return predecessor_of, successors_of


def _topological_fcfs_order(jobs: list, predecessor_of: dict, successors_of: dict) -> list:
    """
    Return jobs in topological order, breaking ties by arrival_order (FCFS).

    A job is eligible once its predecessor (if any) has been placed in
    the ordering. Among all eligible jobs at any point, the one with the
    lowest arrival_order is chosen next, preserving FCFS priority as
    much as the dependency graph allows.
    """
    arrival = {str(job["job_id"]): job["arrival_order"] for job in jobs}
    job_map = {str(job["job_id"]): job for job in jobs}
    valid_ids = set(job_map.keys())

    in_degree = {jid: (1 if predecessor_of[jid] is not None else 0) for jid in valid_ids}

    # Use a sorted list as a priority queue (small N, so list.sort is fine).
    eligible = sorted(
        [jid for jid in valid_ids if in_degree[jid] == 0],
        key=lambda jid: arrival[jid],
    )

    order = []
    while eligible:
        # Pick the eligible job with the lowest arrival_order.
        chosen = eligible.pop(0)
        order.append(job_map[chosen])

        # Unlock successors whose only predecessor was `chosen`.
        for successor in successors_of[chosen]:
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                eligible.append(successor)
                eligible.sort(key=lambda jid: arrival[jid])

    return order


def run_fcfs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the FCFS baseline scheduler.

    Raises ValueError for invalid dependency graphs (missing predecessors,
    self-dependencies, circular dependencies).
    """
    jobs = df.to_dict("records")

    if not jobs:
        return pd.DataFrame(columns=[
            "job_id", "machine", "start_time", "end_time",
            "processing_time", "deadline", "power_kw", "deadline_violated",
        ])

    # Validate graph and detect problems before touching any schedule state.
    predecessor_of, successors_of = _validate_and_build_graph(jobs)

    # Build a topologically-sorted, FCFS-priority order.
    ordered_jobs = _topological_fcfs_order(jobs, predecessor_of, successors_of)

    machine_free_at = {}
    finish_time: dict[str, float] = {}
    schedule = []

    for job in ordered_jobs:
        machine = job["machine"]
        jid = str(job["job_id"])

        earliest_start = machine_free_at.get(machine, 0.0)

        # Enforce predecessor completion — guaranteed to be in finish_time
        # because we process jobs in topological order.
        pred = predecessor_of.get(jid)
        if pred is not None:
            earliest_start = max(earliest_start, finish_time[pred])

        start = earliest_start
        end = start + job["processing_time"]

        machine_free_at[machine] = end
        finish_time[jid] = end

        schedule.append(
            {
                "job_id": jid,
                "machine": machine,
                "start_time": start,
                "end_time": end,
                "processing_time": job["processing_time"],
                "deadline": job["deadline"],
                "power_kw": job["power_kw"],
                "deadline_violated": end > job["deadline"],
            }
        )

    result = pd.DataFrame(schedule)
    return result.sort_values(["machine", "start_time"]).reset_index(drop=True)
