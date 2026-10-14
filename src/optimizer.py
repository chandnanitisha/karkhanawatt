"""Energy-cost-aware production scheduling with OR-Tools CP-SAT.

Time units
----------
By default, scheduling times are expressed in whole hours
(`time_unit_minutes=60`). Finer grids can be selected by passing a positive
integer divisor of 60, such as 10, 5, or 1. In that case processing times,
deadlines, release times, downtime intervals, and returned schedule times must
all use the selected grid unit.

The time-varying tariff is integrated over each job's duration. For fine time
grids, the model uses an hourly cumulative tariff table plus a within-hour
remainder representation, rather than creating a cumulative-table entry for
every minute across the entire horizon.
"""

from dataclasses import dataclass
from typing import Optional
import math

import pandas as pd
from ortools.sat.python import cp_model

from src.tariff import (
    build_cumulative_table,
    POWER_SCALE,
    COST_SCALE,
    TARIFF_SCALE,
)


@dataclass
class SolveResult:
    schedule: pd.DataFrame
    status: str
    makespan: float
    total_cost: float
    used_soft_deadlines: bool
    deadline_violations: int
    solver_wall_time: float
    best_bound: Optional[float] = None
    optimality_gap: Optional[float] = None


def _validate_time_unit(time_unit_minutes: int) -> int:
    """Validate that a time step divides one hour exactly."""
    if isinstance(time_unit_minutes, bool) or not isinstance(time_unit_minutes, int):
        raise ValueError("time_unit_minutes must be an integer divisor of 60.")
    if time_unit_minutes <= 0 or 60 % time_unit_minutes != 0:
        raise ValueError(
            "time_unit_minutes must be a positive divisor of 60, such as 60, 10, 5, or 1."
        )
    return time_unit_minutes


def _add_cumulative_tariff_value(
    model: cp_model.CpModel,
    time_var,
    horizon_hours: int,
    hourly_prefix: list,
    step_rate_values: list,
    max_step_rate: int,
    steps_per_hour: int,
    name: str,
):
    """Return scaled cumulative tariff at a fine-grid time variable.

    The result equals tariff accumulated over all complete hours before the
    time, plus the tariff accumulated during the remaining steps in the
    current hour. All tariff inputs are non-negative.
    """
    hour = model.NewIntVar(0, horizon_hours, f"{name}_hour")
    remainder = model.NewIntVar(0, steps_per_hour - 1, f"{name}_remainder")
    model.AddDivisionEquality(hour, time_var, steps_per_hour)
    model.AddModuloEquality(remainder, time_var, steps_per_hour)

    prefix_value = model.NewIntVar(0, hourly_prefix[-1], f"{name}_prefix")
    model.AddElement(hour, hourly_prefix, prefix_value)

    step_rate = model.NewIntVar(0, max_step_rate, f"{name}_step_rate")
    model.AddElement(hour, step_rate_values, step_rate)

    partial_max = (steps_per_hour - 1) * max_step_rate
    partial_value = model.NewIntVar(0, partial_max, f"{name}_partial")
    model.AddMultiplicationEquality(partial_value, [remainder, step_rate])

    result = model.NewIntVar(
        0,
        hourly_prefix[-1] + partial_max,
        f"{name}_cumulative",
    )
    model.Add(result == prefix_value + partial_value)
    return result


def _build_and_solve(
    df: pd.DataFrame,
    hourly_tariff: list,
    soft_deadlines: bool,
    time_limit_seconds: int,
    machine_downtime: Optional[dict] = None,
    job_release_times: Optional[dict] = None,
    time_unit_minutes: int = 60,
    strategy: str = "cost",
) -> SolveResult:
    """Build and solve the scheduling model in the requested time units."""
    if strategy not in ("cost", "balanced", "time"):
        raise ValueError(
            f"Unknown scheduling strategy: '{strategy}'. Supported strategies: 'cost', 'balanced', 'time'."
        )
    time_unit_minutes = _validate_time_unit(time_unit_minutes)

    required = {
        "job_id", "machine", "processing_time", "deadline", "power_kw", "predecessor"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Scheduling dataframe is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Cannot schedule an empty jobs dataframe.")
    if len(hourly_tariff) != 24:
        raise ValueError("hourly_tariff must contain exactly 24 hourly rates.")
    if time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be greater than zero.")
    if df["job_id"].isna().any() or df["job_id"].astype(str).str.strip().eq("").any():
        raise ValueError("job_id must contain a non-empty value for every job.")
    if df["machine"].isna().any() or df["machine"].astype(str).str.strip().eq("").any():
        raise ValueError("machine must contain a non-empty value for every job.")
    if df["job_id"].astype(str).duplicated().any():
        raise ValueError("job_id values must be unique.")

    machine_downtime = machine_downtime or {}
    job_release_times = job_release_times or {}
    jobs = df.to_dict("records")

    # Validate integer time values; never silently round them.
    for col in ("processing_time", "deadline"):
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.isna().any() or not numeric.map(lambda x: math.isfinite(float(x))).all():
            raise ValueError(f"'{col}' must contain finite numeric values.")
        if not numeric.map(lambda x: float(x).is_integer()).all():
            raise ValueError(
                f"'{col}' must contain whole numbers in the selected time unit "
                f"({time_unit_minutes}-minute steps). Convert the input first; do not round silently."
            )
        if (numeric <= 0).any():
            raise ValueError(f"'{col}' must be greater than zero for every job.")

    numeric_power = pd.to_numeric(df["power_kw"], errors="coerce")
    if (
        numeric_power.isna().any()
        or not numeric_power.map(lambda x: math.isfinite(float(x))).all()
        or (numeric_power <= 0).any()
    ):
        raise ValueError("power_kw must contain finite positive numeric values for every job.")

    numeric_tariff = [float(rate) for rate in hourly_tariff]
    if any(not math.isfinite(rate) or rate < 0 for rate in numeric_tariff):
        raise ValueError("Hourly tariff rates must be finite and non-negative.")

    total_processing = int(pd.to_numeric(df["processing_time"]).sum())
    max_deadline = int(pd.to_numeric(df["deadline"]).max())
    horizon = max_deadline + total_processing

    downtime_end = 0
    for machine, blocks in machine_downtime.items():
        for block in blocks:
            if len(block) != 2:
                raise ValueError(f"Downtime for machine {machine} must use (start, end) pairs.")
            block_start, block_end = int(block[0]), int(block[1])
            if block_start < 0 or block_end <= block_start:
                raise ValueError(
                    f"Invalid downtime interval for machine {machine}: ({block_start}, {block_end})."
                )
            downtime_end = max(downtime_end, block_end)

    latest_release = 0
    for jid, value in job_release_times.items():
        release = int(value)
        if release < 0:
            raise ValueError(f"Release time for job {jid} cannot be negative.")
        latest_release = max(latest_release, release)

    horizon = max(
        horizon,
        downtime_end + total_processing,
        latest_release + total_processing,
    )
    if horizon <= 0:
        raise ValueError("Calculated scheduling horizon must be positive.")

    steps_per_hour = 60 // time_unit_minutes
    model = cp_model.CpModel()

    # For the original hourly model, preserve the straightforward table lookup.
    # For finer grids, use an hourly prefix plus a within-hour remainder.
    if steps_per_hour == 1:
        F = build_cumulative_table(hourly_tariff, horizon, time_unit_minutes=60)
        max_cumulative_tariff = F[-1]
        hourly_prefix = None
        step_rate_values = None
        max_step_rate = 0
        horizon_hours = horizon
    else:
        horizon_hours = (horizon + steps_per_hour - 1) // steps_per_hour
        hourly_prefix = build_cumulative_table(
            hourly_tariff,
            horizon_hours,
            time_unit_minutes=60,
        )
        step_rate_values = [
            int(round(float(hourly_tariff[h % 24]) * TARIFF_SCALE / steps_per_hour))
            for h in range(horizon_hours + 1)
        ]
        max_step_rate = max(step_rate_values, default=0)
        max_cumulative_tariff = (
            hourly_prefix[-1] + (steps_per_hour - 1) * max_step_rate
        )
        F = None

    starts, ends, intervals = {}, {}, {}
    violated = {}
    tardiness = {}
    cost_terms = []
    max_possible_cost_scaled = 0
    makespan_var = model.NewIntVar(0, horizon, "makespan")

    for job in jobs:
        jid = str(job["job_id"])
        p = int(job["processing_time"])
        d = int(job["deadline"])
        power_scaled = int(round(float(job["power_kw"]) * POWER_SCALE))
        if power_scaled <= 0:
            raise ValueError(
                f"power_kw for job {jid} is too small for the configured power scale."
            )

        start = model.NewIntVar(0, horizon, f"start_{jid}")
        end = model.NewIntVar(0, horizon, f"end_{jid}")
        release_time = int(job_release_times.get(jid, 0))
        if release_time < 0:
            raise ValueError(f"Release time for job {jid} cannot be negative.")
        model.Add(start >= release_time)
        model.Add(end == start + p)
        model.Add(makespan_var >= end)
        interval = model.NewIntervalVar(start, p, end, f"interval_{jid}")

        tard = model.NewIntVar(0, horizon, f"tard_{jid}")
        model.Add(tard >= end - d)
        model.Add(tard >= 0)
        tardiness[jid] = tard

        if soft_deadlines:
            is_late = model.NewBoolVar(f"late_{jid}")
            model.Add(end <= d + horizon * is_late)
            violated[jid] = is_late
        else:
            model.Add(end <= d)

        starts[jid] = start
        ends[jid] = end
        intervals[jid] = interval

        if steps_per_hour == 1:
            f_start = model.NewIntVar(0, F[-1], f"Fstart_{jid}")
            f_end = model.NewIntVar(0, F[-1], f"Fend_{jid}")
            model.AddElement(start, F, f_start)
            model.AddElement(end, F, f_end)
        else:
            f_start = _add_cumulative_tariff_value(
                model,
                start,
                horizon_hours,
                hourly_prefix,
                step_rate_values,
                max_step_rate,
                steps_per_hour,
                f"Fstart_{jid}",
            )
            f_end = _add_cumulative_tariff_value(
                model,
                end,
                horizon_hours,
                hourly_prefix,
                step_rate_values,
                max_step_rate,
                steps_per_hour,
                f"Fend_{jid}",
            )

        job_cost_scaled = model.NewIntVar(
            0,
            max(0, power_scaled * max_cumulative_tariff),
            f"cost_{jid}",
        )
        model.Add(job_cost_scaled == power_scaled * (f_end - f_start))
        cost_terms.append(job_cost_scaled)
        max_possible_cost_scaled += power_scaled * max_cumulative_tariff

    # Jobs on each machine cannot overlap each other or downtime blocks.
    machines = sorted(set(df["machine"].astype(str).unique()) | set(str(m) for m in machine_downtime))
    for machine_index, machine in enumerate(machines):
        machine_intervals = [
            intervals[str(job["job_id"])]
            for job in jobs
            if str(job["machine"]) == machine
        ]
        # Match downtime dictionary keys by their string representation.
        blocks = []
        for key, value in machine_downtime.items():
            if str(key) == machine:
                blocks = value
                break
        for block_index, block in enumerate(blocks, start=1):
            down_start, down_end = int(block[0]), int(block[1])
            if down_end > horizon:
                raise ValueError(
                    f"Downtime for machine {machine} ends after the scheduling horizon."
                )
            downtime_interval = model.NewIntervalVar(
                down_start,
                down_end - down_start,
                down_end,
                f"downtime_{machine_index}_{block_index}",
            )
            machine_intervals.append(downtime_interval)
        if machine_intervals:
            model.AddNoOverlap(machine_intervals)

    # Optional single-predecessor dependencies.
    valid_ids = {str(job["job_id"]) for job in jobs}
    for job in jobs:
        pred = job.get("predecessor")
        jid = str(job["job_id"])
        if pred is not None and not pd.isna(pred) and str(pred).strip():
            pred = str(pred).strip()
            if pred not in valid_ids:
                raise ValueError(f"Job {jid} references unknown predecessor '{pred}'.")
            if pred == jid:
                raise ValueError(f"Job {jid} cannot be its own predecessor.")
            model.Add(ends[pred] <= starts[jid])

    total_cost_scaled = model.NewIntVar(
        0,
        max(0, max_possible_cost_scaled),
        "total_cost_scaled",
    )
    model.Add(total_cost_scaled == sum(cost_terms))

    if not soft_deadlines:
        if strategy == "time":
            model.Minimize(makespan_var * 500000 + total_cost_scaled)
        elif strategy == "balanced":
            model.Minimize(total_cost_scaled + makespan_var * 5000)
        else:
            model.Minimize(total_cost_scaled)
    else:
        total_violations = sum(violated.values())
        total_tardiness = sum(tardiness.values())
        big_weight = max_possible_cost_scaled + 1

        if strategy == "time":
            model.Minimize(
                total_violations * big_weight
                + total_tardiness * 500000
                + makespan_var * 250000
                + total_cost_scaled
            )
        elif strategy == "balanced":
            # In scaled units (COST_SCALE = 1,000,000 = Rs 1.00):
            # tardiness weight 10,000 corresponds to Rs 0.01 per late minute.
            # makespan weight 5,000 corresponds to Rs 0.005 per minute makespan.
            model.Minimize(
                total_violations * big_weight
                + total_cost_scaled
                + total_tardiness * 10000
                + makespan_var * 5000
            )
        else:
            # Default "cost" strategy preserves exact original formulation
            model.Minimize(total_violations * big_weight + total_cost_scaled)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return SolveResult(
            schedule=pd.DataFrame(),
            status=status_name,
            makespan=float("nan"),
            total_cost=float("nan"),
            used_soft_deadlines=soft_deadlines,
            deadline_violations=-1,
            solver_wall_time=solver.WallTime(),
            best_bound=None,
            optimality_gap=None,
        )

    rows = []
    n_violations = 0
    for job in jobs:
        jid = str(job["job_id"])
        start_value = solver.Value(starts[jid])
        end_value = solver.Value(ends[jid])
        deadline = int(job["deadline"])
        is_violated = end_value > deadline
        n_violations += int(is_violated)
        rows.append(
            {
                "job_id": jid,
                "machine": str(job["machine"]),
                "start_time": start_value,
                "end_time": end_value,
                "processing_time": int(job["processing_time"]),
                "deadline": deadline,
                "power_kw": float(job["power_kw"]),
                "release_time": int(job_release_times.get(jid, 0)),
                "deadline_violated": is_violated,
            }
        )

    schedule = pd.DataFrame(rows).sort_values(
        ["machine", "start_time"]
    ).reset_index(drop=True)
    makespan = float(schedule["end_time"].max())
    total_cost = solver.Value(total_cost_scaled) / COST_SCALE

    best_bound = None
    optimality_gap = None
    try:
        best_bound = float(solver.BestObjectiveBound())
        obj_val = float(solver.ObjectiveValue())
        if obj_val != 0:
            optimality_gap = round(abs(obj_val - best_bound) / abs(obj_val), 4)
        else:
            optimality_gap = 0.0
    except (RuntimeError, ValueError):
        pass

    return SolveResult(
        schedule=schedule,
        status=status_name,
        makespan=makespan,
        total_cost=round(total_cost, 2),
        used_soft_deadlines=soft_deadlines,
        deadline_violations=n_violations,
        solver_wall_time=solver.WallTime(),
        best_bound=best_bound,
        optimality_gap=optimality_gap,
    )


def solve_schedule(
    df: pd.DataFrame,
    hourly_tariff: list,
    time_limit_seconds: int = 10,
    machine_downtime: Optional[dict] = None,
    job_release_times: Optional[dict] = None,
    time_unit_minutes: int = 60,
    strategy: str = "cost",
) -> SolveResult:
    """Try strict deadlines first, then retry infeasible/inconclusive runs softly."""
    time_unit_minutes = _validate_time_unit(time_unit_minutes)
    common_kwargs = {
        "time_limit_seconds": time_limit_seconds,
        "machine_downtime": machine_downtime,
        "job_release_times": job_release_times,
        "time_unit_minutes": time_unit_minutes,
        "strategy": strategy,
    }

    strict_result = _build_and_solve(
        df,
        hourly_tariff,
        soft_deadlines=False,
        **common_kwargs,
    )
    if strict_result.status in ("OPTIMAL", "FEASIBLE"):
        return strict_result

    if strict_result.status not in ("INFEASIBLE", "UNKNOWN"):
        return strict_result

    # UNKNOWN is inconclusive, not proof of infeasibility. A second, soft
    # attempt may find a schedule, but can consume another full time limit.
    return _build_and_solve(
        df,
        hourly_tariff,
        soft_deadlines=True,
        **common_kwargs,
    )
