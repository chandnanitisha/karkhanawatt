"""
Tests for src/baseline.py — FCFS scheduler correctness.

Covers:
1. Predecessor before dependent in arrival_order (normal case).
2. Predecessor AFTER dependent in arrival_order (the bug case).
3. Multi-level dependency chain.
4. Missing predecessor raises ValueError.
5. Self-dependency raises ValueError.
6. Circular dependency raises ValueError.
7. Two jobs on the same machine (non-overlap).
8. Valid schedule with jobs on different machines.
9. Every valid job appears exactly once.
"""

import pytest
import pandas as pd
from src.baseline import run_fcfs


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def make_jobs(**overrides):
    """Return a minimal valid single-job DataFrame with optional overrides."""
    base = {
        "job_id": ["J1"],
        "machine": ["M1"],
        "processing_time": [2],
        "deadline": [20],
        "power_kw": [10.0],
        "arrival_order": [1],
        "predecessor": [None],
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# Test 1: Predecessor has lower arrival_order than dependent (normal order)
# ---------------------------------------------------------------------------
def test_predecessor_before_dependent_in_arrival_order():
    """J1 (arrival=1) → J2 (arrival=2): standard order; J2 must wait for J1."""
    df = pd.DataFrame(
        {
            "job_id": ["J1", "J2"],
            "machine": ["M1", "M1"],
            "processing_time": [3, 2],
            "deadline": [20, 20],
            "power_kw": [10.0, 10.0],
            "arrival_order": [1, 2],
            "predecessor": [None, "J1"],
        }
    )
    sched = run_fcfs(df)
    j1 = sched[sched["job_id"] == "J1"].iloc[0]
    j2 = sched[sched["job_id"] == "J2"].iloc[0]
    # J1 finishes at 3; J2 must start at 3 or later.
    assert j2["start_time"] >= j1["end_time"], (
        f"J2 started at {j2['start_time']} before J1 ended at {j1['end_time']}"
    )
    assert j2["start_time"] == 3  # on same machine, back-to-back


# ---------------------------------------------------------------------------
# Test 2: Predecessor has HIGHER arrival_order than its dependent (bug case)
# ---------------------------------------------------------------------------
def test_predecessor_after_dependent_in_arrival_order():
    """
    J2 (arrival=1) depends on J1 (arrival=2).
    Old code would process J2 first and skip the precedence check.
    Fixed code must still honour J1 → J2 dependency.
    """
    df = pd.DataFrame(
        {
            "job_id": ["J1", "J2"],
            "machine": ["M1", "M1"],
            "processing_time": [3, 2],
            "deadline": [20, 20],
            "power_kw": [10.0, 10.0],
            "arrival_order": [2, 1],      # J2 has lower arrival_order but depends on J1
            "predecessor": [None, "J1"],
        }
    )
    sched = run_fcfs(df)
    j1 = sched[sched["job_id"] == "J1"].iloc[0]
    j2 = sched[sched["job_id"] == "J2"].iloc[0]
    assert j2["start_time"] >= j1["end_time"], (
        f"Precedence violated: J2 started at {j2['start_time']}, "
        f"J1 ended at {j1['end_time']}"
    )


# ---------------------------------------------------------------------------
# Test 3: Multi-level dependency chain
# ---------------------------------------------------------------------------
def test_multi_level_dependency_chain(chained_jobs_df):
    """J1 → J2 → J3. Each must start after its predecessor finishes."""
    sched = run_fcfs(chained_jobs_df)
    j1 = sched[sched["job_id"] == "J1"].iloc[0]
    j2 = sched[sched["job_id"] == "J2"].iloc[0]
    j3 = sched[sched["job_id"] == "J3"].iloc[0]

    assert j2["start_time"] >= j1["end_time"]
    assert j3["start_time"] >= j2["end_time"]
    # All on M1: should be contiguous back-to-back
    assert j1["start_time"] == 0
    assert j2["start_time"] == 2   # J1 finishes at 2
    assert j3["start_time"] == 5   # J2 finishes at 5


# ---------------------------------------------------------------------------
# Test 4: Missing predecessor raises ValueError
# ---------------------------------------------------------------------------
def test_missing_predecessor_raises():
    df = pd.DataFrame(
        {
            "job_id": ["J1"],
            "machine": ["M1"],
            "processing_time": [2],
            "deadline": [20],
            "power_kw": [10.0],
            "arrival_order": [1],
            "predecessor": ["J_NONEXISTENT"],
        }
    )
    with pytest.raises(ValueError, match="nonexistent|not exist|unknown|predecessor"):
        run_fcfs(df)


# ---------------------------------------------------------------------------
# Test 5: Self-dependency raises ValueError
# ---------------------------------------------------------------------------
def test_self_dependency_raises():
    df = pd.DataFrame(
        {
            "job_id": ["J1"],
            "machine": ["M1"],
            "processing_time": [2],
            "deadline": [20],
            "power_kw": [10.0],
            "arrival_order": [1],
            "predecessor": ["J1"],
        }
    )
    with pytest.raises(ValueError, match="self|own predecessor|itself"):
        run_fcfs(df)


# ---------------------------------------------------------------------------
# Test 6: Circular dependency raises ValueError
# ---------------------------------------------------------------------------
def test_circular_dependency_raises():
    """J1 depends on J2, J2 depends on J1 — cycle."""
    df = pd.DataFrame(
        {
            "job_id": ["J1", "J2"],
            "machine": ["M1", "M2"],
            "processing_time": [2, 2],
            "deadline": [20, 20],
            "power_kw": [10.0, 10.0],
            "arrival_order": [1, 2],
            "predecessor": ["J2", "J1"],
        }
    )
    with pytest.raises(ValueError, match="[Cc]ircular|[Cc]ycle"):
        run_fcfs(df)


# ---------------------------------------------------------------------------
# Test 7: Two jobs sharing the same machine must not overlap
# ---------------------------------------------------------------------------
def test_same_machine_no_overlap():
    df = pd.DataFrame(
        {
            "job_id": ["J1", "J2"],
            "machine": ["M1", "M1"],
            "processing_time": [3, 4],
            "deadline": [20, 20],
            "power_kw": [10.0, 10.0],
            "arrival_order": [1, 2],
            "predecessor": [None, None],
        }
    )
    sched = run_fcfs(df)
    j1 = sched[sched["job_id"] == "J1"].iloc[0]
    j2 = sched[sched["job_id"] == "J2"].iloc[0]

    # No overlap: one must finish before the other starts.
    assert (
        j1["end_time"] <= j2["start_time"] or
        j2["end_time"] <= j1["start_time"]
    ), f"Jobs overlap: J1=[{j1['start_time']},{j1['end_time']}] J2=[{j2['start_time']},{j2['end_time']}]"


# ---------------------------------------------------------------------------
# Test 8: Jobs on different machines can run concurrently
# ---------------------------------------------------------------------------
def test_different_machines_concurrent(simple_jobs_df):
    """J1 (M1) and J2 (M2) may overlap; both should start at 0."""
    sched = run_fcfs(simple_jobs_df)
    j1 = sched[sched["job_id"] == "J1"].iloc[0]
    j2 = sched[sched["job_id"] == "J2"].iloc[0]
    # Both start as early as possible (t=0) since their machines are free.
    assert j1["start_time"] == 0
    assert j2["start_time"] == 0


# ---------------------------------------------------------------------------
# Test 9: Every valid job appears exactly once
# ---------------------------------------------------------------------------
def test_every_job_appears_exactly_once(simple_jobs_df):
    sched = run_fcfs(simple_jobs_df)
    assert len(sched) == len(simple_jobs_df), (
        f"Expected {len(simple_jobs_df)} rows, got {len(sched)}"
    )
    assert set(sched["job_id"]) == {"J1", "J2", "J3"}
    # No duplicates
    assert sched["job_id"].nunique() == 3


# ---------------------------------------------------------------------------
# Test 10: Correct end_time = start_time + processing_time for all jobs
# ---------------------------------------------------------------------------
def test_end_time_equals_start_plus_processing(chained_jobs_df):
    sched = run_fcfs(chained_jobs_df)
    for _, row in sched.iterrows():
        assert row["end_time"] == pytest.approx(
            row["start_time"] + row["processing_time"]
        ), f"Job {row['job_id']}: end={row['end_time']} != start+pt"


# ---------------------------------------------------------------------------
# Test 11: Empty dataframe returns empty schedule (no crash)
# ---------------------------------------------------------------------------
def test_empty_dataframe():
    df = pd.DataFrame(
        columns=["job_id", "machine", "processing_time", "deadline",
                 "power_kw", "arrival_order", "predecessor"]
    )
    sched = run_fcfs(df)
    assert sched.empty


# ---------------------------------------------------------------------------
# Test 12: Deadline-violated flag is set correctly
# ---------------------------------------------------------------------------
def test_deadline_violated_flag():
    """J1 takes 5 hours but deadline is 3 — must be flagged as late."""
    df = pd.DataFrame(
        {
            "job_id": ["J1"],
            "machine": ["M1"],
            "processing_time": [5],
            "deadline": [3],
            "power_kw": [10.0],
            "arrival_order": [1],
            "predecessor": [None],
        }
    )
    sched = run_fcfs(df)
    assert sched.iloc[0]["deadline_violated"] is True or sched.iloc[0]["deadline_violated"] == True
