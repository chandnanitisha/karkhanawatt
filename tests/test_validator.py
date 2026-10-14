"""
Tests for src/validator.py — independent schedule validator.

Covers:
1. Valid schedule passes all checks.
2. Missing job causes failure.
3. Overlapping jobs on same machine causes failure.
4. Wrong end_time causes failure.
5. Wrong machine assignment causes failure.
6. Predecessor constraint violated causes failure.
7. Release-time violated causes failure.
8. Downtime overlap causes failure.
9. Unexpected job (not in original df) causes failure.
10. Cost discrepancy generates a warning (not an error).
11. Soft-deadline violation generates a warning (not an error).
"""

import pytest
import pandas as pd

from src.validator import validate_schedule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _jobs(rows):
    return pd.DataFrame(rows)


def _sched(rows):
    base = {
        "deadline_violated": False,
    }
    for row in rows:
        for k, v in base.items():
            row.setdefault(k, v)
    return pd.DataFrame(rows)


FLAT_TARIFF = [6.0] * 24


# ---------------------------------------------------------------------------
# Test 1: Valid schedule passes
# ---------------------------------------------------------------------------
def test_valid_schedule_passes():
    jobs = _jobs(
        [
            {"job_id": "J1", "machine": "M1", "processing_time": 3,
             "deadline": 10, "power_kw": 10.0, "predecessor": None},
            {"job_id": "J2", "machine": "M2", "processing_time": 2,
             "deadline": 10, "power_kw": 20.0, "predecessor": None},
        ]
    )
    sched = _sched(
        [
            {"job_id": "J1", "machine": "M1", "start_time": 0, "end_time": 3,
             "processing_time": 3, "deadline": 10, "power_kw": 10.0},
            {"job_id": "J2", "machine": "M2", "start_time": 0, "end_time": 2,
             "processing_time": 2, "deadline": 10, "power_kw": 20.0},
        ]
    )
    result = validate_schedule(sched, jobs, FLAT_TARIFF)
    assert result.passed, f"Expected pass; errors: {result.errors}"
    assert not result.errors


# ---------------------------------------------------------------------------
# Test 2: Missing job causes failure
# ---------------------------------------------------------------------------
def test_missing_job_fails():
    jobs = _jobs(
        [
            {"job_id": "J1", "machine": "M1", "processing_time": 3,
             "deadline": 10, "power_kw": 10.0, "predecessor": None},
            {"job_id": "J2", "machine": "M2", "processing_time": 2,
             "deadline": 10, "power_kw": 20.0, "predecessor": None},
        ]
    )
    # Only J1 in schedule; J2 missing.
    sched = _sched(
        [
            {"job_id": "J1", "machine": "M1", "start_time": 0, "end_time": 3,
             "processing_time": 3, "deadline": 10, "power_kw": 10.0},
        ]
    )
    result = validate_schedule(sched, jobs, FLAT_TARIFF)
    assert not result.passed
    assert any("J2" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Test 3: Overlapping jobs on same machine fails
# ---------------------------------------------------------------------------
def test_overlap_on_same_machine_fails():
    jobs = _jobs(
        [
            {"job_id": "J1", "machine": "M1", "processing_time": 5,
             "deadline": 20, "power_kw": 10.0, "predecessor": None},
            {"job_id": "J2", "machine": "M1", "processing_time": 3,
             "deadline": 20, "power_kw": 10.0, "predecessor": None},
        ]
    )
    sched = _sched(
        [
            {"job_id": "J1", "machine": "M1", "start_time": 0, "end_time": 5,
             "processing_time": 5, "deadline": 20, "power_kw": 10.0},
            {"job_id": "J2", "machine": "M1", "start_time": 2, "end_time": 5,  # overlaps J1
             "processing_time": 3, "deadline": 20, "power_kw": 10.0},
        ]
    )
    result = validate_schedule(sched, jobs, FLAT_TARIFF)
    assert not result.passed
    assert any("overlap" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# Test 4: Wrong end_time fails
# ---------------------------------------------------------------------------
def test_wrong_end_time_fails():
    jobs = _jobs(
        [{"job_id": "J1", "machine": "M1", "processing_time": 3,
          "deadline": 10, "power_kw": 10.0, "predecessor": None}]
    )
    sched = _sched(
        [{"job_id": "J1", "machine": "M1", "start_time": 0,
          "end_time": 5,  # wrong; should be 3
          "processing_time": 3, "deadline": 10, "power_kw": 10.0}]
    )
    result = validate_schedule(sched, jobs, FLAT_TARIFF)
    assert not result.passed
    assert any("end_time" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Test 5: Wrong machine assignment fails
# ---------------------------------------------------------------------------
def test_wrong_machine_fails():
    jobs = _jobs(
        [{"job_id": "J1", "machine": "M1", "processing_time": 3,
          "deadline": 10, "power_kw": 10.0, "predecessor": None}]
    )
    sched = _sched(
        [{"job_id": "J1", "machine": "M2",  # wrong machine
          "start_time": 0, "end_time": 3,
          "processing_time": 3, "deadline": 10, "power_kw": 10.0}]
    )
    result = validate_schedule(sched, jobs, FLAT_TARIFF)
    assert not result.passed
    assert any("machine" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# Test 6: Predecessor constraint violated
# ---------------------------------------------------------------------------
def test_predecessor_violated_fails():
    jobs = _jobs(
        [
            {"job_id": "J1", "machine": "M1", "processing_time": 4,
             "deadline": 20, "power_kw": 10.0, "predecessor": None},
            {"job_id": "J2", "machine": "M1", "processing_time": 2,
             "deadline": 20, "power_kw": 10.0, "predecessor": "J1"},
        ]
    )
    # J2 starts at 2, but J1 ends at 4 — violation.
    sched = _sched(
        [
            {"job_id": "J1", "machine": "M1", "start_time": 0, "end_time": 4,
             "processing_time": 4, "deadline": 20, "power_kw": 10.0},
            {"job_id": "J2", "machine": "M1", "start_time": 2, "end_time": 4,
             "processing_time": 2, "deadline": 20, "power_kw": 10.0},
        ]
    )
    result = validate_schedule(sched, jobs, FLAT_TARIFF)
    assert not result.passed
    assert any("Precedence" in e or "predecessor" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# Test 7: Release-time constraint violated
# ---------------------------------------------------------------------------
def test_release_time_violated_fails():
    jobs = _jobs(
        [{"job_id": "J1", "machine": "M1", "processing_time": 2,
          "deadline": 20, "power_kw": 10.0, "predecessor": None}]
    )
    sched = _sched(
        [{"job_id": "J1", "machine": "M1", "start_time": 1,  # release=5, starts at 1
          "end_time": 3, "processing_time": 2, "deadline": 20, "power_kw": 10.0}]
    )
    result = validate_schedule(sched, jobs, FLAT_TARIFF, job_release_times={"J1": 5})
    assert not result.passed
    assert any("release" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# Test 8: Downtime overlap causes failure
# ---------------------------------------------------------------------------
def test_downtime_overlap_fails():
    jobs = _jobs(
        [{"job_id": "J1", "machine": "M1", "processing_time": 3,
          "deadline": 20, "power_kw": 10.0, "predecessor": None}]
    )
    # J1 runs from 1 to 4; downtime is 2 to 5 — overlap.
    sched = _sched(
        [{"job_id": "J1", "machine": "M1", "start_time": 1, "end_time": 4,
          "processing_time": 3, "deadline": 20, "power_kw": 10.0}]
    )
    result = validate_schedule(
        sched, jobs, FLAT_TARIFF, machine_downtime={"M1": [(2, 5)]}
    )
    assert not result.passed
    assert any("downtime" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# Test 9: Unexpected job causes failure
# ---------------------------------------------------------------------------
def test_unexpected_job_fails():
    jobs = _jobs(
        [{"job_id": "J1", "machine": "M1", "processing_time": 3,
          "deadline": 10, "power_kw": 10.0, "predecessor": None}]
    )
    sched = _sched(
        [
            {"job_id": "J1", "machine": "M1", "start_time": 0, "end_time": 3,
             "processing_time": 3, "deadline": 10, "power_kw": 10.0},
            {"job_id": "J_GHOST", "machine": "M1", "start_time": 3, "end_time": 5,
             "processing_time": 2, "deadline": 10, "power_kw": 10.0},  # not in jobs_df
        ]
    )
    result = validate_schedule(sched, jobs, FLAT_TARIFF)
    assert not result.passed
    assert any("J_GHOST" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Test 10: Cost discrepancy generates warning (not error)
# ---------------------------------------------------------------------------
def test_cost_discrepancy_is_warning_not_error():
    jobs = _jobs(
        [{"job_id": "J1", "machine": "M1", "processing_time": 2,
          "deadline": 10, "power_kw": 10.0, "predecessor": None}]
    )
    sched = _sched(
        [{"job_id": "J1", "machine": "M1", "start_time": 0, "end_time": 2,
          "processing_time": 2, "deadline": 10, "power_kw": 10.0}]
    )
    # Independent cost = 10 × 2 × 6 = 120. Report wildly different value.
    result = validate_schedule(sched, jobs, FLAT_TARIFF, reported_cost=9999.0)
    # Should still pass hard constraints
    assert result.passed
    # But should warn about cost discrepancy
    assert any("discrepancy" in w.lower() or "cost" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# Test 11: Soft-deadline violation is a warning, not an error
# ---------------------------------------------------------------------------
def test_soft_deadline_is_warning_not_error():
    jobs = _jobs(
        [{"job_id": "J1", "machine": "M1", "processing_time": 5,
          "deadline": 3, "power_kw": 10.0, "predecessor": None}]
    )
    # Job ends after deadline — this is a soft-deadline violation.
    sched = _sched(
        [{"job_id": "J1", "machine": "M1", "start_time": 0, "end_time": 5,
          "processing_time": 5, "deadline": 3, "power_kw": 10.0}]
    )
    result = validate_schedule(sched, jobs, FLAT_TARIFF)
    # Hard constraints all pass (no overlap, correct end time, etc.)
    assert result.passed
    assert any("deadline" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# Test 12: Empty schedule fails
# ---------------------------------------------------------------------------
def test_empty_schedule_fails():
    jobs = _jobs(
        [{"job_id": "J1", "machine": "M1", "processing_time": 2,
          "deadline": 10, "power_kw": 10.0, "predecessor": None}]
    )
    empty = pd.DataFrame()
    result = validate_schedule(empty, jobs, FLAT_TARIFF)
    assert not result.passed


# ===========================================================================
# Regression tests — new field-level checks (added in strengthened validator)
# ===========================================================================

# ---------------------------------------------------------------------------
# Test 13: Wrong processing_time in schedule fails
# ---------------------------------------------------------------------------
def test_wrong_processing_time_fails():
    """Schedule processing_time differs from original job definition → error."""
    jobs = _jobs(
        [{"job_id": "J1", "machine": "M1", "processing_time": 3,
          "deadline": 10, "power_kw": 10.0, "predecessor": None}]
    )
    sched = _sched(
        [{"job_id": "J1", "machine": "M1", "start_time": 0,
          "end_time": 5,          # matches the tampered processing_time below
          "processing_time": 5,  # WRONG — original is 3
          "deadline": 10, "power_kw": 10.0}]
    )
    result = validate_schedule(sched, jobs, FLAT_TARIFF)
    assert not result.passed, "Expected failure: processing_time mismatch not caught"
    assert any("processing_time" in e for e in result.errors), (
        f"Expected 'processing_time' in errors; got: {result.errors}"
    )


# ---------------------------------------------------------------------------
# Test 14: Wrong power_kw in schedule fails
# ---------------------------------------------------------------------------
def test_wrong_power_kw_fails():
    """Schedule power_kw differs from original job definition → error."""
    jobs = _jobs(
        [{"job_id": "J1", "machine": "M1", "processing_time": 3,
          "deadline": 10, "power_kw": 10.0, "predecessor": None}]
    )
    sched = _sched(
        [{"job_id": "J1", "machine": "M1", "start_time": 0, "end_time": 3,
          "processing_time": 3, "deadline": 10,
          "power_kw": 999.0}]  # WRONG — original is 10.0
    )
    result = validate_schedule(sched, jobs, FLAT_TARIFF)
    assert not result.passed, "Expected failure: power_kw mismatch not caught"
    assert any("power_kw" in e for e in result.errors), (
        f"Expected 'power_kw' in errors; got: {result.errors}"
    )


# ---------------------------------------------------------------------------
# Test 15: Non-positive processing_time fails
# ---------------------------------------------------------------------------
def test_non_positive_processing_time_fails():
    """A processing_time of 0 must be rejected as a hard error."""
    jobs = _jobs(
        [{"job_id": "J1", "machine": "M1", "processing_time": 3,
          "deadline": 10, "power_kw": 10.0, "predecessor": None}]
    )
    sched = _sched(
        [{"job_id": "J1", "machine": "M1", "start_time": 0, "end_time": 0,
          "processing_time": 0,  # invalid
          "deadline": 10, "power_kw": 10.0}]
    )
    result = validate_schedule(sched, jobs, FLAT_TARIFF)
    assert not result.passed, "Expected failure: zero processing_time not caught"


# ---------------------------------------------------------------------------
# Test 16: Non-finite processing_time fails
# ---------------------------------------------------------------------------
def test_non_finite_time_fails():
    """NaN or infinity in time fields must be rejected."""
    import math
    jobs = _jobs(
        [{"job_id": "J1", "machine": "M1", "processing_time": 3,
          "deadline": 10, "power_kw": 10.0, "predecessor": None}]
    )
    sched = _sched(
        [{"job_id": "J1", "machine": "M1", "start_time": 0, "end_time": float("inf"),
          "processing_time": float("inf"),
          "deadline": 10, "power_kw": 10.0}]
    )
    result = validate_schedule(sched, jobs, FLAT_TARIFF)
    assert not result.passed, "Expected failure: non-finite time values not caught"


# ---------------------------------------------------------------------------
# Test 17: Valid schedule with correct fields still passes
# ---------------------------------------------------------------------------
def test_valid_schedule_with_all_fields_passes():
    """A fully correct schedule that matches all original definitions must pass."""
    jobs = _jobs(
        [
            {"job_id": "J1", "machine": "M1", "processing_time": 3,
             "deadline": 10, "power_kw": 10.0, "predecessor": None},
            {"job_id": "J2", "machine": "M2", "processing_time": 2,
             "deadline": 10, "power_kw": 25.0, "predecessor": None},
        ]
    )
    sched = _sched(
        [
            {"job_id": "J1", "machine": "M1", "start_time": 0, "end_time": 3,
             "processing_time": 3, "deadline": 10, "power_kw": 10.0},
            {"job_id": "J2", "machine": "M2", "start_time": 0, "end_time": 2,
             "processing_time": 2, "deadline": 10, "power_kw": 25.0},
        ]
    )
    result = validate_schedule(sched, jobs, FLAT_TARIFF)
    assert result.passed, f"Expected pass; errors: {result.errors}"
    assert not result.errors


# ---------------------------------------------------------------------------
# Test 18: Invalid schedule produces errors (not just warnings)
# ---------------------------------------------------------------------------
def test_invalid_schedule_produces_errors_not_only_warnings():
    """
    When a schedule has a hard constraint violation (wrong machine), the result
    must have errors and passed=False. The UI must not present this as a success.
    This is the 'blocking' regression test for the app.py gating requirement.
    """
    jobs = _jobs(
        [{"job_id": "J1", "machine": "M1", "processing_time": 3,
          "deadline": 10, "power_kw": 10.0, "predecessor": None}]
    )
    sched = _sched(
        [{"job_id": "J1", "machine": "M2",  # wrong machine
          "start_time": 0, "end_time": 3,
          "processing_time": 3, "deadline": 10, "power_kw": 10.0}]
    )
    result = validate_schedule(sched, jobs, FLAT_TARIFF)
    # Hard check: must fail
    assert not result.passed
    assert len(result.errors) > 0, "Validation must produce errors, not just pass silently"
    # Verify that errors are substantive (not just warning-level text)
    assert not any("warning" in e.lower() or "soft" in e.lower() for e in result.errors), (
        "Errors should describe hard violations, not soft issues"
    )


# ---------------------------------------------------------------------------
# Test 19: Wrong machine AND wrong processing_time both appear in errors
# ---------------------------------------------------------------------------
def test_multiple_field_violations_all_reported():
    """All field-level violations must be collected (not short-circuited)."""
    jobs = _jobs(
        [{"job_id": "J1", "machine": "M1", "processing_time": 3,
          "deadline": 10, "power_kw": 10.0, "predecessor": None}]
    )
    sched = _sched(
        [{"job_id": "J1", "machine": "M2",  # wrong machine
          "start_time": 0, "end_time": 5,
          "processing_time": 5,             # wrong processing_time (vs original 3)
          "deadline": 10, "power_kw": 10.0}]
    )
    result = validate_schedule(sched, jobs, FLAT_TARIFF)
    assert not result.passed
    machine_errors = [e for e in result.errors if "machine" in e.lower()]
    pt_errors = [e for e in result.errors if "processing_time" in e]
    assert machine_errors, f"Expected machine error; got: {result.errors}"
    assert pt_errors, f"Expected processing_time error; got: {result.errors}"

