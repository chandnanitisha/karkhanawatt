"""
baseline.py
------------
A simple First-Come-First-Serve (FCFS) baseline scheduler.

This is deliberately NOT an OR model. It exists only so the project can
show "before OR" vs "after OR" and quantify the improvement CP-SAT gives.

Logic:
    - Jobs are sorted by arrival_order.
    - Each job is placed on its assigned machine as early as that
      machine becomes free (i.e. right after the previous job on that
      machine finishes). No look-ahead, no optimization.
    - Precedence (if present) is respected by also waiting for the
      predecessor job to finish.

Returns a schedule DataFrame with start_time / end_time per job, plus a
flag for whether each job missed its deadline.
"""

import pandas as pd


def run_fcfs(df: pd.DataFrame) -> pd.DataFrame:
    jobs = df.sort_values("arrival_order").to_dict("records")

    machine_free_at = {}
    finish_time = {}  # job_id -> end_time, needed for precedence lookups
    schedule = []

    for job in jobs:
        machine = job["machine"]
        earliest_start = machine_free_at.get(machine, 0.0)

        pred = job["predecessor"]
        if pred is not None and pred in finish_time:
            earliest_start = max(earliest_start, finish_time[pred])

        start = earliest_start
        end = start + job["processing_time"]

        machine_free_at[machine] = end
        finish_time[job["job_id"]] = end

        schedule.append(
            {
                "job_id": job["job_id"],
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
