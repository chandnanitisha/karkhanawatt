
"""
Schedule metrics for KarkhanaWatt.

The optimizer can use hourly or finer time steps. Metrics are returned
in human-readable units: hours, kWh, rupees, and percentages.
"""

import pandas as pd
from src.tariff import schedule_cost


def compute_metrics(
    schedule: pd.DataFrame,
    hourly_tariff: list,
    time_unit_minutes: int = 60,
) -> dict:
    """
    Compute metrics for a schedule.

    time_unit_minutes=60 preserves the original hourly mode.
    time_unit_minutes=1 supports minute-resolution schedules.
    """
    if (
        isinstance(time_unit_minutes, bool)
        or not isinstance(time_unit_minutes, int)
        or time_unit_minutes <= 0
        or 60 % time_unit_minutes != 0
    ):
        raise ValueError(
            "time_unit_minutes must be a positive integer divisor of 60."
        )

    if schedule.empty:
        return {
            "makespan": None,
            "machine_utilization_pct": None,
            "idle_time": None,
            "deadline_violations": None,
            "total_energy_kwh": None,
            "electricity_cost": None,
        }

    required = {
        "machine",
        "start_time",
        "end_time",
        "processing_time",
        "deadline",
        "deadline_violated",
        "power_kw",
    }
    missing = required - set(schedule.columns)

    if missing:
        raise ValueError(
            f"Schedule is missing metric columns: {sorted(missing)}"
        )

    # Convert model time units into hours for reporting.
    hours_per_step = time_unit_minutes / 60.0

    makespan_steps = float(schedule["end_time"].max())
    makespan_hours = makespan_steps * hours_per_step

    machines = schedule["machine"].unique()
    num_machines = len(machines)

    total_busy_steps = float(
        (schedule["end_time"] - schedule["start_time"]).sum()
    )
    total_busy_hours = total_busy_steps * hours_per_step

    total_available_hours = makespan_hours * num_machines
    idle_time_hours = total_available_hours - total_busy_hours

    utilization_per_machine = []

    for machine in machines:
        machine_jobs = schedule[schedule["machine"] == machine]
        busy_steps = float(
            (
                machine_jobs["end_time"]
                - machine_jobs["start_time"]
            ).sum()
        )

        utilization_per_machine.append(
            busy_steps / makespan_steps if makespan_steps > 0 else 0
        )

    avg_utilization_pct = (
        sum(utilization_per_machine)
        / len(utilization_per_machine)
        * 100
        if utilization_per_machine
        else 0
    )

    deadline_violations = int(
        schedule["deadline_violated"].astype(bool).sum()
    )

    # kW × hours = kWh.
    processing_hours = (
        schedule["processing_time"].astype(float) * hours_per_step
    )
    total_energy_kwh = float(
        (schedule["power_kw"].astype(float) * processing_hours).sum()
    )

    # The tariff function must use the same time resolution as the schedule.
    electricity_cost = schedule_cost(
        schedule,
        hourly_tariff,
        time_unit_minutes=time_unit_minutes,
    )

    return {
        "makespan": round(makespan_hours, 4),
        "machine_utilization_pct": round(
            float(avg_utilization_pct), 2
        ),
        "idle_time": round(float(idle_time_hours), 4),
        "deadline_violations": deadline_violations,
        "total_energy_kwh": round(total_energy_kwh, 4),
        "electricity_cost": float(electricity_cost),
    }


def comparison_table(
    baseline_metrics: dict,
    optimized_metrics: dict,
) -> pd.DataFrame:
    rows = []

    labels = {
        "makespan": "Makespan (hours)",
        "machine_utilization_pct": "Machine Utilization (%)",
        "idle_time": "Idle Time (machine-hours)",
        "deadline_violations": "Deadline Violations",
        "total_energy_kwh": "Total Energy (kWh)",
        "electricity_cost": "Electricity Cost (Rs., time-varying tariff)",
    }

    for key, label in labels.items():
        rows.append(
            {
                "Metric": label,
                "Baseline (FCFS)": baseline_metrics.get(key),
                "OR-Optimized (CP-SAT)": optimized_metrics.get(key),
            }
        )

    return pd.DataFrame(rows)
