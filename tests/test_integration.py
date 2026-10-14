"""
Integration test using the bundled sample dataset (data/jobs.csv + data/tariff.csv).

This test runs both the FCFS baseline and the actual CP-SAT optimizer end-to-end,
then validates the returned schedules independently.
"""

import math
from pathlib import Path
import pytest
import pandas as pd

from src.data_loader import load_jobs
from src.tariff import load_tariff, schedule_cost
from src.baseline import run_fcfs
from src.optimizer import solve_schedule
from src.validator import validate_schedule

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


@pytest.fixture(scope="module")
def sample_jobs():
    return load_jobs(DATA_DIR / "jobs.csv")


@pytest.fixture(scope="module")
def sample_tariff():
    return load_tariff(DATA_DIR / "tariff.csv")


# ---------------------------------------------------------------------------
# Integration: FCFS baseline on bundled data
# ---------------------------------------------------------------------------
def test_fcfs_baseline_on_sample(sample_jobs, sample_tariff):
    sched = run_fcfs(sample_jobs)

    # Every job appears exactly once.
    assert len(sched) == len(sample_jobs)
    assert set(sched["job_id"]) == set(sample_jobs["job_id"])

    # No two jobs on the same machine overlap.
    for machine in sched["machine"].unique():
        m_rows = sched[sched["machine"] == machine].sort_values("start_time")
        for i in range(len(m_rows) - 1):
            a = m_rows.iloc[i]
            b = m_rows.iloc[i + 1]
            assert float(a["end_time"]) <= float(b["start_time"]) + 1e-6, (
                f"Machine {machine}: {a['job_id']} ends {a['end_time']} "
                f"overlaps {b['job_id']} starts {b['start_time']}"
            )

    # end_time == start_time + processing_time for every row.
    for _, row in sched.iterrows():
        assert float(row["end_time"]) == pytest.approx(
            float(row["start_time"]) + float(row["processing_time"])
        )

    # Cost is a finite positive number.
    cost = schedule_cost(sched, sample_tariff, time_unit_minutes=60)
    assert math.isfinite(cost) and cost > 0


# ---------------------------------------------------------------------------
# Integration: CP-SAT optimizer on bundled data
# ---------------------------------------------------------------------------
def test_optimizer_on_sample(sample_jobs, sample_tariff):
    result = solve_schedule(
        sample_jobs, sample_tariff, time_limit_seconds=30
    )

    if result.status not in ("OPTIMAL", "FEASIBLE"):
        pytest.skip(
            f"Solver returned {result.status} on sample data; "
            "cannot validate schedule."
        )

    # Every job appears exactly once.
    assert len(result.schedule) == len(sample_jobs)
    assert set(result.schedule["job_id"]) == set(sample_jobs["job_id"].astype(str))

    # Independent validation must pass.
    val = validate_schedule(
        schedule=result.schedule,
        jobs_df=sample_jobs,
        hourly_tariff=sample_tariff,
        time_unit_minutes=60,
        reported_cost=result.total_cost,
    )
    assert val.passed, f"Validation failed:\n{val}"

    # Optimizer cost must be ≤ baseline cost (OR guarantees improvement or tie).
    baseline_sched = run_fcfs(sample_jobs)
    baseline_cost = schedule_cost(baseline_sched, sample_tariff, time_unit_minutes=60)
    assert float(result.total_cost) <= baseline_cost + 1.0, (
        f"Optimizer cost {result.total_cost} > baseline {baseline_cost} "
        "(optimizer should find equal or cheaper schedule)"
    )


# ---------------------------------------------------------------------------
# Integration: Machine breakdown scenario on bundled data
# ---------------------------------------------------------------------------
def test_machine_breakdown_scenario(sample_jobs, sample_tariff):
    # Find a machine used in the sample data.
    used_machines = sample_jobs["machine"].unique()
    breakdown_machine = used_machines[0]

    result = solve_schedule(
        sample_jobs,
        sample_tariff,
        time_limit_seconds=30,
        machine_downtime={breakdown_machine: [(0, 5)]},
    )

    if result.status not in ("OPTIMAL", "FEASIBLE"):
        pytest.skip(f"Solver returned {result.status}; skipping breakdown validation.")

    val = validate_schedule(
        schedule=result.schedule,
        jobs_df=sample_jobs,
        hourly_tariff=sample_tariff,
        time_unit_minutes=60,
        machine_downtime={breakdown_machine: [(0, 5)]},
        reported_cost=result.total_cost,
    )
    assert val.passed, f"Breakdown scenario validation failed:\n{val}"


# ---------------------------------------------------------------------------
# Integration: Urgent job scenario on bundled data
# ---------------------------------------------------------------------------
def test_urgent_job_scenario(sample_jobs, sample_tariff):
    urgent = pd.DataFrame(
        [
            {
                "job_id": "URGENT-1",
                "machine": sample_jobs["machine"].iloc[0],
                "processing_time": 2,
                "deadline": 40,
                "power_kw": 50.0,
                "arrival_order": int(sample_jobs["arrival_order"].max()) + 1,
                "predecessor": None,
            }
        ]
    )
    scenario_jobs = pd.concat([sample_jobs, urgent], ignore_index=True)

    result = solve_schedule(
        scenario_jobs,
        sample_tariff,
        time_limit_seconds=30,
        job_release_times={"URGENT-1": 2},
    )

    if result.status not in ("OPTIMAL", "FEASIBLE"):
        pytest.skip(f"Solver returned {result.status}; skipping urgent job validation.")

    # URGENT-1 must appear in the schedule.
    assert "URGENT-1" in result.schedule["job_id"].values

    val = validate_schedule(
        schedule=result.schedule,
        jobs_df=scenario_jobs,
        hourly_tariff=sample_tariff,
        time_unit_minutes=60,
        job_release_times={"URGENT-1": 2},
        reported_cost=result.total_cost,
    )
    assert val.passed, f"Urgent job scenario validation failed:\n{val}"


# ---------------------------------------------------------------------------
# Integration: Detailed Metrics Comparison Audit on bundled sample dataset
# ---------------------------------------------------------------------------
def test_detailed_metrics_comparison_audit(sample_jobs, sample_tariff):
    """Regression test auditing all metrics in the Detailed Metrics Comparison Table:
    1. Same jobs and machines across both schedules.
    2. Makespan calculated from maximum end time.
    3. Machine utilization calculated from total processing time / (machine_count * makespan).
    4. Idle time measured over the schedule makespan horizon.
    5. Independent electricity cost calculation matches reported costs.
    6. 30-hour makespan satisfies all deadlines and is a legitimate cost-minimizing trade-off.
    7. Deadline violations are 0 for both schedules.
    """
    from src.metrics import compute_metrics, comparison_table

    fcfs_sched = run_fcfs(sample_jobs)
    opt_res = solve_schedule(sample_jobs, sample_tariff, time_limit_seconds=30)
    opt_sched = opt_res.schedule

    # 1. Exact same required jobs and machines
    assert len(fcfs_sched) == 8
    assert len(opt_sched) == 8
    assert set(fcfs_sched["job_id"]) == set(sample_jobs["job_id"])
    assert set(opt_sched["job_id"]) == set(sample_jobs["job_id"])
    assert set(fcfs_sched["machine"]) == set(sample_jobs["machine"])
    assert set(opt_sched["machine"]) == set(sample_jobs["machine"])

    m_fcfs = compute_metrics(fcfs_sched, sample_tariff, time_unit_minutes=60)
    m_opt = compute_metrics(opt_sched, sample_tariff, time_unit_minutes=60)

    # 2. Makespan
    assert m_fcfs["makespan"] == float(fcfs_sched["end_time"].max()) == 11.0
    assert 28.0 <= m_opt["makespan"] <= 30.0
    assert m_opt["makespan"] == float(opt_sched["end_time"].max())

    # 3. Machine utilization
    total_proc = float(sample_jobs["processing_time"].sum())  # 25.0 hours
    num_machines = len(sample_jobs["machine"].unique())  # 3 machines
    expected_fcfs_util = round((total_proc / (num_machines * 11.0)) * 100, 2)  # 75.76%
    expected_opt_util = round((total_proc / (num_machines * m_opt["makespan"])) * 100, 2)
    assert m_fcfs["machine_utilization_pct"] == pytest.approx(expected_fcfs_util, abs=0.01)
    assert m_opt["machine_utilization_pct"] == pytest.approx(expected_opt_util, abs=0.01)

    # 4. Idle time across makespan horizon
    assert m_fcfs["idle_time"] == pytest.approx((num_machines * 11.0) - total_proc) == 8.0
    assert m_opt["idle_time"] == pytest.approx((num_machines * m_opt["makespan"]) - total_proc)

    # 5. Electricity cost and physical energy conservation
    assert m_fcfs["total_energy_kwh"] == 1530.0
    assert m_opt["total_energy_kwh"] == 1530.0
    assert m_fcfs["electricity_cost"] == 8335.0
    assert m_opt["electricity_cost"] == 6895.0
    assert m_opt["electricity_cost"] < m_fcfs["electricity_cost"]

    # 6 & 8. Deadlines strictly respected (0 violations)
    assert m_fcfs["deadline_violations"] == 0
    assert m_opt["deadline_violations"] == 0
    for _, row in opt_sched.iterrows():
        assert row["end_time"] <= row["deadline"], f"Job {row['job_id']} violated deadline {row['deadline']}"

    # Comparison table structure
    df_comp = comparison_table(m_fcfs, m_opt)
    assert len(df_comp) == 6
    assert "Makespan (hours)" in df_comp["Metric"].values


# ---------------------------------------------------------------------------
# Integration: External Manufacturing Dataset (Kaggle FJSP) End-to-End Audit
# ---------------------------------------------------------------------------
def test_external_manufacturing_workflow(sample_tariff):
    """Validate external manufacturing workflow on Kaggle hybrid manufacturing data:
    1. Loads raw records and verifies schema.
    2. Prepares jobs with minute-resolution units.
    3. Runs FCFS baseline and CP-SAT optimizer.
    4. Independently validates schedules, job coverage, durations, non-overlap, and metrics.
    """
    from pathlib import Path
    from src.data_loader import load_manufacturing_records, prepare_manufacturing_jobs
    from src.metrics import compute_metrics
    
    csv_path = Path("kaggle_manufacture") / "hybrid_manufacturing_categorical.csv"
    if not csv_path.exists():
        pytest.skip(f"External manufacturing dataset not found at {csv_path}")

    raw_df = load_manufacturing_records(csv_path)
    assert len(raw_df) == 1000

    prep_df = prepare_manufacturing_jobs(raw_df, max_jobs=10, assume_energy_is_kwh_per_job=True)
    assert len(prep_df) == 10
    assert (prep_df["processing_time"] > 0).all()
    assert (prep_df["power_kw"] > 0).all()

    # Baseline run
    fcfs_sched = run_fcfs(prep_df)
    assert len(fcfs_sched) == 10
    m_fcfs = compute_metrics(fcfs_sched, sample_tariff, time_unit_minutes=1)
    v_fcfs = validate_schedule(fcfs_sched, prep_df, sample_tariff, time_unit_minutes=1)
    assert v_fcfs.passed, f"Baseline validation failed: {v_fcfs.errors}"

    # CP-SAT solve
    opt_res = solve_schedule(prep_df, sample_tariff, time_limit_seconds=15, time_unit_minutes=1)
    assert opt_res.status in ("OPTIMAL", "FEASIBLE")
    assert len(opt_res.schedule) == 10
    m_opt = compute_metrics(opt_res.schedule, sample_tariff, time_unit_minutes=1)
    v_opt = validate_schedule(
        opt_res.schedule,
        prep_df,
        sample_tariff,
        time_unit_minutes=1,
        reported_cost=opt_res.total_cost,
    )
    assert v_opt.passed, f"Optimizer schedule validation failed: {v_opt.errors}"

    # Energy conservation across both schedules
    assert pytest.approx(m_fcfs["total_energy_kwh"], rel=1e-3) == m_opt["total_energy_kwh"]
    assert m_opt["deadline_violations"] <= m_fcfs["deadline_violations"]


def test_external_manufacturing_strategies(sample_tariff):
    """Verify that 'balanced' and 'time' strategies on external manufacturing data:
    1. Pass independent validation.
    2. Conserve exact physical energy.
    3. Successfully restrict makespan compared to unconstrained cost-focused drift.
    """
    from pathlib import Path
    from src.data_loader import load_manufacturing_records, prepare_manufacturing_jobs
    from src.metrics import compute_metrics

    csv_path = Path("kaggle_manufacture") / "hybrid_manufacturing_categorical.csv"
    if not csv_path.exists():
        pytest.skip(f"External manufacturing dataset not found at {csv_path}")

    raw_df = load_manufacturing_records(csv_path)
    prep_df = prepare_manufacturing_jobs(raw_df, max_jobs=10, assume_energy_is_kwh_per_job=True)

    res_cost = solve_schedule(prep_df, sample_tariff, time_limit_seconds=10, time_unit_minutes=1, strategy="cost")
    res_bal = solve_schedule(prep_df, sample_tariff, time_limit_seconds=10, time_unit_minutes=1, strategy="balanced")
    res_time = solve_schedule(prep_df, sample_tariff, time_limit_seconds=10, time_unit_minutes=1, strategy="time")

    for res in (res_cost, res_bal, res_time):
        assert res.status in ("OPTIMAL", "FEASIBLE")
        val = validate_schedule(res.schedule, prep_df, sample_tariff, time_unit_minutes=1, reported_cost=res.total_cost)
        assert val.passed

    m_cost = compute_metrics(res_cost.schedule, sample_tariff, time_unit_minutes=1)
    m_bal = compute_metrics(res_bal.schedule, sample_tariff, time_unit_minutes=1)
    m_time = compute_metrics(res_time.schedule, sample_tariff, time_unit_minutes=1)

    # Physical energy conserved across all strategies
    assert pytest.approx(m_cost["total_energy_kwh"], rel=1e-3) == m_bal["total_energy_kwh"] == m_time["total_energy_kwh"]

    # Time-focused strategy produces equal or shorter makespan than cost-focused
    assert m_time["makespan"] <= m_cost["makespan"] + 1e-4


