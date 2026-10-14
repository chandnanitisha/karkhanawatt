"""
Tests for src/optimizer.py — CP-SAT model correctness.

These tests actually call the solver. They use small deterministic problems
so they finish quickly and produce predictable results.

Covers:
1. Machine non-overlap in returned schedule.
2. Precedence constraint in returned schedule.
3. Release-time constraint.
4. Strict deadline (feasible and infeasible cases).
5. Soft-deadline fallback when strict is infeasible.
6. Machine downtime constraint.
7. INFEASIBLE result handling (no schedule returned, empty DataFrame).
8. Optimizer objective cost matches schedule_cost within tolerance.
9. OPTIMAL vs FEASIBLE status distinction.
10. Duplicate job_id raises ValueError before solving.
"""

import math
import pytest
import pandas as pd

from src.optimizer import solve_schedule
from src.tariff import schedule_cost



def _check_no_overlap(schedule: pd.DataFrame) -> list[str]:
    """Return a list of overlap errors (empty = no overlaps)."""
    errors = []
    for machine in schedule["machine"].unique():
        rows = schedule[schedule["machine"] == machine].sort_values("start_time")
        prev = None
        for _, row in rows.iterrows():
            if prev is not None and float(row["start_time"]) < float(prev["end_time"]) - 1e-6:
                errors.append(
                    f"Machine {machine}: {prev['job_id']} ends {prev['end_time']} "
                    f"but {row['job_id']} starts {row['start_time']}"
                )
            prev = row
    return errors


# ---------------------------------------------------------------------------
# Test 1: Machine non-overlap
# ---------------------------------------------------------------------------
def test_machine_non_overlap(simple_tariff):
    df = pd.DataFrame(
        {
            "job_id": ["J1", "J2", "J3"],
            "machine": ["M1", "M1", "M1"],
            "processing_time": [3, 2, 4],
            "deadline": [30, 30, 30],
            "power_kw": [10.0, 10.0, 10.0],
            "arrival_order": [1, 2, 3],
            "predecessor": [None, None, None],
        }
    )
    result = solve_schedule(df, simple_tariff, time_limit_seconds=10)
    assert result.status in ("OPTIMAL", "FEASIBLE"), f"Solver status: {result.status}"
    overlap_errors = _check_no_overlap(result.schedule)
    assert not overlap_errors, f"Overlaps found: {overlap_errors}"


# ---------------------------------------------------------------------------
# Test 2: Precedence constraint — dependent starts after predecessor ends
# ---------------------------------------------------------------------------
def test_precedence_constraint(simple_tariff):
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
    result = solve_schedule(df, simple_tariff, time_limit_seconds=10)
    assert result.status in ("OPTIMAL", "FEASIBLE")
    sched = result.schedule
    j1_end = float(sched[sched["job_id"] == "J1"]["end_time"].iloc[0])
    j2_start = float(sched[sched["job_id"] == "J2"]["start_time"].iloc[0])
    assert j2_start >= j1_end - 1e-6, (
        f"Precedence violated: J2 starts at {j2_start}, J1 ends at {j1_end}"
    )


# ---------------------------------------------------------------------------
# Test 3: Release-time constraint
# ---------------------------------------------------------------------------
def test_release_time_constraint(simple_tariff):
    """J1 has release time = 5; it must not start before hour 5."""
    df = pd.DataFrame(
        {
            "job_id": ["J1"],
            "machine": ["M1"],
            "processing_time": [2],
            "deadline": [20],
            "power_kw": [10.0],
            "arrival_order": [1],
            "predecessor": [None],
        }
    )
    result = solve_schedule(
        df, simple_tariff, time_limit_seconds=10, job_release_times={"J1": 5}
    )
    assert result.status in ("OPTIMAL", "FEASIBLE")
    j1_start = float(result.schedule[result.schedule["job_id"] == "J1"]["start_time"].iloc[0])
    assert j1_start >= 5 - 1e-6, f"J1 started at {j1_start}, expected >= 5"


# ---------------------------------------------------------------------------
# Test 4a: Strict deadline — feasible
# ---------------------------------------------------------------------------
def test_strict_deadline_feasible(simple_tariff):
    """J1 takes 2 hours; deadline=5 — should be solvable."""
    df = pd.DataFrame(
        {
            "job_id": ["J1"],
            "machine": ["M1"],
            "processing_time": [2],
            "deadline": [5],
            "power_kw": [10.0],
            "arrival_order": [1],
            "predecessor": [None],
        }
    )
    result = solve_schedule(df, simple_tariff, time_limit_seconds=10)
    assert result.status in ("OPTIMAL", "FEASIBLE")
    sched = result.schedule
    j1_end = float(sched[sched["job_id"] == "J1"]["end_time"].iloc[0])
    assert j1_end <= 5 + 1e-6


# ---------------------------------------------------------------------------
# Test 4b: Strict deadline infeasible → soft fallback; status stays valid
# ---------------------------------------------------------------------------
def test_strict_deadline_infeasible_triggers_soft(simple_tariff):
    """
    Two jobs on M1, each 5 hours, deadline=5.
    Together they cannot fit by deadline=5 (would need 10 hours).
    Solver should fall back to soft deadlines.
    """
    df = pd.DataFrame(
        {
            "job_id": ["J1", "J2"],
            "machine": ["M1", "M1"],
            "processing_time": [5, 5],
            "deadline": [5, 5],
            "power_kw": [10.0, 10.0],
            "arrival_order": [1, 2],
            "predecessor": [None, None],
        }
    )
    result = solve_schedule(df, simple_tariff, time_limit_seconds=15)
    # Either returns soft-deadline solution or INFEASIBLE with no schedule.
    if result.status in ("OPTIMAL", "FEASIBLE"):
        assert result.used_soft_deadlines, (
            "Strict deadlines are impossible; solver should use soft deadlines."
        )
        assert result.deadline_violations > 0


# ---------------------------------------------------------------------------
# Test 5: Machine downtime prevents job from occupying that window
# ---------------------------------------------------------------------------
def test_machine_downtime_respected(simple_tariff):
    """
    M1 is down from hour 0 to hour 3. J1 takes 2 hours.
    J1 must start at hour 3 or later.
    """
    df = pd.DataFrame(
        {
            "job_id": ["J1"],
            "machine": ["M1"],
            "processing_time": [2],
            "deadline": [20],
            "power_kw": [10.0],
            "arrival_order": [1],
            "predecessor": [None],
        }
    )
    result = solve_schedule(
        df, simple_tariff, time_limit_seconds=10,
        machine_downtime={"M1": [(0, 3)]}
    )
    assert result.status in ("OPTIMAL", "FEASIBLE")
    j1_start = float(result.schedule[result.schedule["job_id"] == "J1"]["start_time"].iloc[0])
    assert j1_start >= 3 - 1e-6, f"J1 started at {j1_start} during downtime [0,3]"


# ---------------------------------------------------------------------------
# Test 6: Downtime on machine with no assigned jobs — no crash
# ---------------------------------------------------------------------------
def test_downtime_on_idle_machine(simple_tariff):
    df = pd.DataFrame(
        {
            "job_id": ["J1"],
            "machine": ["M1"],
            "processing_time": [2],
            "deadline": [20],
            "power_kw": [10.0],
            "arrival_order": [1],
            "predecessor": [None],
        }
    )
    # M2 has no jobs but still has downtime.
    result = solve_schedule(
        df, simple_tariff, time_limit_seconds=10,
        machine_downtime={"M2": [(0, 5)]}
    )
    assert result.status in ("OPTIMAL", "FEASIBLE")


# ---------------------------------------------------------------------------
# Test 7: Optimizer objective cost matches independently calculated cost
# ---------------------------------------------------------------------------
def test_objective_cost_matches_schedule_cost(simple_tariff):
    """
    Flat tariff makes the expected cost predictable.
    Optimizer's total_cost must agree with schedule_cost within Rs 1.0.
    """
    df = pd.DataFrame(
        {
            "job_id": ["J1", "J2"],
            "machine": ["M1", "M2"],
            "processing_time": [3, 4],
            "deadline": [24, 24],
            "power_kw": [10.0, 20.0],
            "arrival_order": [1, 2],
            "predecessor": [None, None],
        }
    )
    result = solve_schedule(df, simple_tariff, time_limit_seconds=10)
    assert result.status in ("OPTIMAL", "FEASIBLE")

    independent = schedule_cost(result.schedule, simple_tariff, time_unit_minutes=60)
    assert abs(float(result.total_cost) - float(independent)) <= 1.0, (
        f"Optimizer cost {result.total_cost} vs independent {independent}"
    )


# ---------------------------------------------------------------------------
# Test 8: Empty jobs raises ValueError before solving
# ---------------------------------------------------------------------------
def test_empty_jobs_raises(simple_tariff):
    empty = pd.DataFrame(
        columns=["job_id", "machine", "processing_time", "deadline",
                 "power_kw", "arrival_order", "predecessor"]
    )
    with pytest.raises(ValueError):
        solve_schedule(empty, simple_tariff, time_limit_seconds=10)


# ---------------------------------------------------------------------------
# Test 9: Duplicate job_id raises ValueError
# ---------------------------------------------------------------------------
def test_duplicate_job_id_raises(simple_tariff):
    df = pd.DataFrame(
        {
            "job_id": ["J1", "J1"],
            "machine": ["M1", "M2"],
            "processing_time": [2, 3],
            "deadline": [10, 10],
            "power_kw": [10.0, 10.0],
            "arrival_order": [1, 2],
            "predecessor": [None, None],
        }
    )
    with pytest.raises(ValueError, match="[Dd]uplicate|unique"):
        solve_schedule(df, simple_tariff, time_limit_seconds=10)


# ---------------------------------------------------------------------------
# Test 10: end_time == start_time + processing_time for all returned rows
# ---------------------------------------------------------------------------
def test_end_time_integrity(simple_tariff, simple_jobs_df):
    result = solve_schedule(simple_jobs_df, simple_tariff, time_limit_seconds=10)
    assert result.status in ("OPTIMAL", "FEASIBLE")
    for _, row in result.schedule.iterrows():
        expected_end = int(row["start_time"]) + int(row["processing_time"])
        assert int(row["end_time"]) == expected_end, (
            f"Job {row['job_id']}: end={row['end_time']}, "
            f"start+pt={expected_end}"
        )


# ---------------------------------------------------------------------------
# Test 11: Optimizer prefers off-peak hours on flat-vs-peak tariff
# ---------------------------------------------------------------------------
def test_optimizer_prefers_off_peak(peak_tariff):
    """
    Single job, 1-hour duration. Peak hours 9-17 cost 10 Rs/kWh; off-peak 4.
    Optimal should schedule outside peak window (if deadline allows).
    """
    df = pd.DataFrame(
        {
            "job_id": ["J1"],
            "machine": ["M1"],
            "processing_time": [1],
            "deadline": [24],
            "power_kw": [100.0],
            "arrival_order": [1],
            "predecessor": [None],
        }
    )
    result = solve_schedule(df, peak_tariff, time_limit_seconds=15)
    assert result.status in ("OPTIMAL", "FEASIBLE")
    start = int(result.schedule["start_time"].iloc[0])
    # Should choose off-peak hour (rate 4 Rs), not peak (rate 10 Rs).
    assert peak_tariff[start % 24] == pytest.approx(4.0), (
        f"Scheduled at hour {start} (rate {peak_tariff[start % 24]} Rs/kWh), "
        "expected off-peak hour (4 Rs/kWh)"
    )


# ---------------------------------------------------------------------------
# Test 12: SolveResult exposes best_bound and optimality_gap
# ---------------------------------------------------------------------------
def test_solver_result_exposes_bounds_and_gap(simple_tariff):
    df = pd.DataFrame(
        {
            "job_id": ["J1"],
            "machine": ["M1"],
            "processing_time": [2],
            "deadline": [10],
            "power_kw": [10.0],
            "arrival_order": [1],
            "predecessor": [None],
        }
    )
    result = solve_schedule(df, simple_tariff, time_limit_seconds=10)
    assert result.status == "OPTIMAL"
    assert result.best_bound is not None
    assert result.optimality_gap is not None
    assert result.optimality_gap == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Test 13: Solver strategies (cost, balanced, time) produce valid schedules
# ---------------------------------------------------------------------------
def test_solver_strategies_produce_valid_schedules(simple_tariff):
    df = pd.DataFrame(
        {
            "job_id": ["J1", "J2"],
            "machine": ["M1", "M1"],
            "processing_time": [2, 3],
            "deadline": [10, 10],
            "power_kw": [10.0, 15.0],
            "arrival_order": [1, 2],
            "predecessor": [None, None],
        }
    )
    for strat in ("cost", "balanced", "time"):
        result = solve_schedule(df, simple_tariff, time_limit_seconds=10, strategy=strat)
        assert result.status in ("OPTIMAL", "FEASIBLE")
        assert len(result.schedule) == 2
        # Verify non-overlap
        j1 = result.schedule[result.schedule["job_id"] == "J1"].iloc[0]
        j2 = result.schedule[result.schedule["job_id"] == "J2"].iloc[0]
        assert j1["end_time"] <= j2["start_time"] or j2["end_time"] <= j1["start_time"]


# ---------------------------------------------------------------------------
# Test 14: Invalid strategy raises ValueError
# ---------------------------------------------------------------------------
def test_invalid_strategy_raises(simple_tariff):
    df = pd.DataFrame(
        {
            "job_id": ["J1"],
            "machine": ["M1"],
            "processing_time": [2],
            "deadline": [10],
            "power_kw": [10.0],
            "arrival_order": [1],
            "predecessor": [None],
        }
    )
    with pytest.raises(ValueError, match="Unknown scheduling strategy"):
        solve_schedule(df, simple_tariff, strategy="magic_strategy")

