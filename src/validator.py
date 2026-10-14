"""
validator.py
-------------
Independent post-solve schedule validator for KarkhanaWatt.

This module checks a returned schedule without relying on the solver's
internal constraint objects. It is the authoritative gate before any
schedule is displayed as valid in the UI.

Usage::

    from src.validator import validate_schedule, ValidationResult

    result = validate_schedule(
        schedule=result.schedule,
        jobs_df=df,
        hourly_tariff=hourly_tariff,
        time_unit_minutes=60,
        machine_downtime={},       # optional
        job_release_times={},      # optional
    )
    if not result.passed:
        st.error("Schedule validation failed:")
        for err in result.errors:
            st.write(f"  • {err}")

Design principles:
- Raises nothing; all problems are collected and returned.
- Never crashes the Streamlit app.
- Soft-deadline violations are reported as warnings, not errors.
- Cost discrepancies above tolerance are reported as warnings, not errors,
  because integer scaling in the solver objective may introduce small
  rounding differences.
- Field-level checks compare every scheduled job against its original
  definition. Tolerated unit conversions (minutes ↔ hours) must be
  documented in the caller via the time_unit_minutes parameter; the
  validator only checks values in model units.
"""

from dataclasses import dataclass, field
from typing import Optional
import math

import pandas as pd

from src.tariff import schedule_cost

# Maximum allowed difference (in Rs.) between optimizer-reported cost and
# independently-calculated cost before raising a warning.
COST_TOLERANCE_RS = 1.0

# Relative tolerance for floating-point field comparisons.
_REL_TOL = 1e-6
_ABS_TOL = 1e-6


@dataclass
class ValidationResult:
    """Result of an independent schedule validation."""
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        parts = [f"ValidationResult(passed={self.passed})"]
        for e in self.errors:
            parts.append(f"  ERROR: {e}")
        for w in self.warnings:
            parts.append(f"  WARN:  {w}")
        return "\n".join(parts)


def validate_schedule(
    schedule: pd.DataFrame,
    jobs_df: pd.DataFrame,
    hourly_tariff: list,
    time_unit_minutes: int = 60,
    machine_downtime: Optional[dict] = None,
    job_release_times: Optional[dict] = None,
    reported_cost: Optional[float] = None,
) -> ValidationResult:
    """
    Independently validate a returned schedule against the original job
    definitions and constraints.

    Parameters
    ----------
    schedule : pd.DataFrame
        The schedule returned by the solver or FCFS baseline.
    jobs_df : pd.DataFrame
        The original jobs dataframe used to build the model.
    hourly_tariff : list
        The 24-hour tariff list (Rs/kWh).
    time_unit_minutes : int
        Minutes per model time step (60 = hourly, 1 = per-minute).
        All time values in the schedule and jobs_df are expected in this unit.
    machine_downtime : dict, optional
        {machine_id: [(start, end), ...]} in model time units.
    job_release_times : dict, optional
        {job_id: release_time} in model time units.
    reported_cost : float, optional
        The cost reported by the solver objective (for comparison).

    Returns
    -------
    ValidationResult
        passed=True only if every hard constraint check passes.
        Warnings are recorded for soft issues (deadline violations, cost
        discrepancy) without causing passed=False.
    """
    errors: list[str] = []
    warnings: list[str] = []

    machine_downtime = machine_downtime or {}
    job_release_times = job_release_times or {}

    # ------------------------------------------------------------------
    # Guard: empty or missing schedule
    # ------------------------------------------------------------------
    if schedule is None or schedule.empty:
        errors.append("The schedule is empty; no jobs were scheduled.")
        return ValidationResult(passed=False, errors=errors, warnings=warnings)

    required_schedule_cols = {
        "job_id", "machine", "start_time", "end_time", "processing_time",
        "deadline", "power_kw",
    }
    missing_cols = required_schedule_cols - set(schedule.columns)
    if missing_cols:
        errors.append(
            f"Schedule is missing required columns: {sorted(missing_cols)}."
        )
        return ValidationResult(passed=False, errors=errors, warnings=warnings)

    # ------------------------------------------------------------------
    # CHECK 1: Every required job appears exactly once.
    # CHECK 2: No unexpected jobs appear.
    # ------------------------------------------------------------------
    expected_ids = set(jobs_df["job_id"].astype(str).tolist())
    scheduled_ids_raw = schedule["job_id"].astype(str).tolist()
    scheduled_ids = set(scheduled_ids_raw)

    # Duplicates in schedule
    if len(scheduled_ids_raw) != len(set(scheduled_ids_raw)):
        from collections import Counter
        counts = Counter(scheduled_ids_raw)
        dups = [jid for jid, n in counts.items() if n > 1]
        errors.append(
            f"The following job(s) appear more than once in the schedule: {sorted(dups)}."
        )

    # Missing jobs
    missing_jobs = expected_ids - scheduled_ids
    if missing_jobs:
        errors.append(
            f"The following job(s) are missing from the schedule: {sorted(missing_jobs)}."
        )

    # Unexpected jobs
    unexpected_jobs = scheduled_ids - expected_ids
    if unexpected_jobs:
        errors.append(
            f"The following unexpected job(s) appear in the schedule: {sorted(unexpected_jobs)}."
        )

    # Build a lookup from the original job definitions.
    job_defs = {str(row["job_id"]): row for _, row in jobs_df.iterrows()}

    # ------------------------------------------------------------------
    # Row-level checks (per job)
    # ------------------------------------------------------------------
    for _, row in schedule.iterrows():
        jid = str(row["job_id"])

        # ---- Numeric validity of time fields ----
        try:
            start = float(row["start_time"])
            end = float(row["end_time"])
            pt = float(row["processing_time"])
        except (TypeError, ValueError):
            errors.append(
                f"Job '{jid}': start_time, end_time or processing_time is not numeric."
            )
            continue

        if not (math.isfinite(start) and math.isfinite(end) and math.isfinite(pt)):
            errors.append(
                f"Job '{jid}': start_time, end_time or processing_time is not finite."
            )
            continue

        # CHECK 3: Start times must not be negative.
        if start < 0:
            errors.append(
                f"Job '{jid}': start_time is negative ({start})."
            )

        # CHECK 3b: Processing duration must be positive.
        if pt <= 0:
            errors.append(
                f"Job '{jid}': processing_time is not positive ({pt})."
            )

        # CHECK 4: end_time == start_time + processing_time.
        expected_end = start + pt
        if not math.isclose(end, expected_end, rel_tol=_REL_TOL, abs_tol=_ABS_TOL):
            errors.append(
                f"Job '{jid}': end_time ({end}) != start_time ({start}) "
                f"+ processing_time ({pt}) = {expected_end}."
            )

        machine = str(row["machine"])

        # ---- Numeric validity of power_kw ----
        try:
            power = float(row["power_kw"])
        except (TypeError, ValueError):
            errors.append(f"Job '{jid}': power_kw is not numeric.")
            power = None

        if power is not None and (not math.isfinite(power) or power <= 0):
            errors.append(
                f"Job '{jid}': power_kw must be finite and positive (got {power})."
            )

        # ---- Numeric validity of deadline ----
        try:
            deadline = float(row["deadline"])
        except (TypeError, ValueError):
            errors.append(f"Job '{jid}': deadline is not numeric.")
            deadline = None

        if deadline is not None and not math.isfinite(deadline):
            errors.append(f"Job '{jid}': deadline is not finite.")

        # ---- Compare against original job definition ----
        if jid in job_defs:
            defn = job_defs[jid]
            defined_machine = str(defn["machine"])

            # CHECK 5: Machine assignment matches job definition.
            if machine != defined_machine:
                errors.append(
                    f"Job '{jid}': scheduled on machine '{machine}' "
                    f"but defined on machine '{defined_machine}'."
                )

            # CHECK 5b: processing_time matches original definition.
            try:
                defined_pt = float(defn["processing_time"])
                if not math.isclose(pt, defined_pt, rel_tol=_REL_TOL, abs_tol=_ABS_TOL):
                    errors.append(
                        f"Job '{jid}': scheduled processing_time ({pt}) does not match "
                        f"original definition ({defined_pt})."
                    )
            except (TypeError, ValueError, KeyError):
                pass  # If original is unparseable, skip; load_jobs already validated it.

            # CHECK 5c: power_kw matches original definition.
            if power is not None:
                try:
                    defined_power = float(defn["power_kw"])
                    if not math.isclose(power, defined_power, rel_tol=_REL_TOL, abs_tol=_ABS_TOL):
                        errors.append(
                            f"Job '{jid}': scheduled power_kw ({power}) does not match "
                            f"original definition ({defined_power})."
                        )
                except (TypeError, ValueError, KeyError):
                    pass

            # CHECK 5d: deadline matches original definition (soft: warn only).
            if deadline is not None:
                try:
                    defined_deadline = float(defn["deadline"])
                    if not math.isclose(deadline, defined_deadline, rel_tol=_REL_TOL, abs_tol=_ABS_TOL):
                        warnings.append(
                            f"Job '{jid}': scheduled deadline ({deadline}) differs from "
                            f"original definition ({defined_deadline}). "
                            "This is expected when soft deadlines extend the horizon; "
                            "verify intentionally."
                        )
                except (TypeError, ValueError, KeyError):
                    pass

        # CHECK 8: Release-time constraint.
        release = float(job_release_times.get(jid, 0))
        if start < release - _ABS_TOL:
            errors.append(
                f"Job '{jid}': starts at {start} but release time is {release}."
            )

        # CHECK 10: Deadline violations (soft — report as warning).
        if deadline is not None and end > deadline + _ABS_TOL:
            warnings.append(
                f"Job '{jid}': ends at {end}, past deadline {deadline} "
                "(soft-deadline violation)."
            )

    # ------------------------------------------------------------------
    # CHECK 6: No two jobs on the same machine overlap.
    # ------------------------------------------------------------------
    machine_jobs: dict[str, list] = {}
    for _, row in schedule.iterrows():
        m = str(row["machine"])
        machine_jobs.setdefault(m, []).append(row)

    for machine, m_rows in machine_jobs.items():
        sorted_rows = sorted(m_rows, key=lambda r: float(r["start_time"]))
        for i in range(len(sorted_rows) - 1):
            a = sorted_rows[i]
            b = sorted_rows[i + 1]
            a_end = float(a["end_time"])
            b_start = float(b["start_time"])
            if a_end > b_start + _ABS_TOL:
                errors.append(
                    f"Machine '{machine}': job '{a['job_id']}' "
                    f"(ends {a_end}) overlaps job '{b['job_id']}' "
                    f"(starts {b_start})."
                )

    # ------------------------------------------------------------------
    # CHECK 7: Predecessor constraints.
    # ------------------------------------------------------------------
    finish_times = {
        str(row["job_id"]): float(row["end_time"])
        for _, row in schedule.iterrows()
    }
    start_times = {
        str(row["job_id"]): float(row["start_time"])
        for _, row in schedule.iterrows()
    }

    for _, job_row in jobs_df.iterrows():
        jid = str(job_row["job_id"])
        pred = job_row.get("predecessor")
        if pred is not None and not pd.isna(pred) and str(pred).strip():
            pred = str(pred).strip()
            if pred in finish_times and jid in start_times:
                pred_end = finish_times[pred]
                dep_start = start_times[jid]
                if dep_start < pred_end - _ABS_TOL:
                    errors.append(
                        f"Precedence violated: job '{jid}' starts at {dep_start} "
                        f"but predecessor '{pred}' finishes at {pred_end}."
                    )

    # ------------------------------------------------------------------
    # CHECK 9: Machine downtime constraints.
    # ------------------------------------------------------------------
    for machine, blocks in machine_downtime.items():
        machine_str = str(machine)
        for block in blocks:
            down_start, down_end = float(block[0]), float(block[1])
            for _, row in schedule.iterrows():
                if str(row["machine"]) != machine_str:
                    continue
                jid = str(row["job_id"])
                j_start = float(row["start_time"])
                j_end = float(row["end_time"])
                # Overlap check: intervals [j_start, j_end) and [down_start, down_end)
                if j_start < down_end - _ABS_TOL and j_end > down_start + _ABS_TOL:
                    errors.append(
                        f"Job '{jid}' on machine '{machine_str}' "
                        f"(interval [{j_start}, {j_end}]) overlaps downtime "
                        f"[{down_start}, {down_end}]."
                    )

    # ------------------------------------------------------------------
    # CHECK 11: Reported electricity cost vs independently calculated cost.
    # ------------------------------------------------------------------
    if reported_cost is not None and not math.isnan(reported_cost):
        try:
            independent_cost = schedule_cost(
                schedule, hourly_tariff, time_unit_minutes=time_unit_minutes
            )
            diff = abs(float(reported_cost) - float(independent_cost))
            if diff > COST_TOLERANCE_RS:
                warnings.append(
                    f"Cost discrepancy: optimizer reported ₹{reported_cost:.4f}, "
                    f"independent calculation gives ₹{independent_cost:.4f} "
                    f"(difference ₹{diff:.4f}, tolerance ₹{COST_TOLERANCE_RS})."
                )
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"Could not independently calculate schedule cost: {exc}"
            )

    passed = len(errors) == 0
    return ValidationResult(passed=passed, errors=errors, warnings=warnings)
