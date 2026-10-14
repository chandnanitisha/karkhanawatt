"""
Tests for src/what_if.py — schedule change detection and explanation.

Covers:
1. changed_jobs detects a changed start time.
2. changed_jobs detects an added job.
3. changed_jobs detects a removed job.
4. changed_jobs returns empty DataFrame when schedules are identical.
5. build_explanation returns non-empty list for Machine breakdown.
6. build_explanation returns non-empty list for Add urgent job.
7. changed_jobs handles None inputs gracefully.
"""

import pytest
import pandas as pd

from src.what_if import changed_jobs, build_explanation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sched(*rows):
    return pd.DataFrame(rows)


BASE_SCHEDULE = _sched(
    {"job_id": "J1", "machine": "M1", "start_time": 0, "end_time": 3},
    {"job_id": "J2", "machine": "M2", "start_time": 0, "end_time": 2},
)

IDENTICAL_SCHEDULE = _sched(
    {"job_id": "J1", "machine": "M1", "start_time": 0, "end_time": 3},
    {"job_id": "J2", "machine": "M2", "start_time": 0, "end_time": 2},
)


# ---------------------------------------------------------------------------
# Test 1: Changed start time is detected
# ---------------------------------------------------------------------------
def test_changed_start_time_detected():
    revised = _sched(
        {"job_id": "J1", "machine": "M1", "start_time": 5, "end_time": 8},  # moved
        {"job_id": "J2", "machine": "M2", "start_time": 0, "end_time": 2},
    )
    changes = changed_jobs(BASE_SCHEDULE, revised)
    assert "J1" in changes["job_id"].values
    j1_row = changes[changes["job_id"] == "J1"].iloc[0]
    assert "start time" in j1_row["change"]


# ---------------------------------------------------------------------------
# Test 2: Added job is detected
# ---------------------------------------------------------------------------
def test_added_job_detected():
    revised = _sched(
        {"job_id": "J1", "machine": "M1", "start_time": 0, "end_time": 3},
        {"job_id": "J2", "machine": "M2", "start_time": 0, "end_time": 2},
        {"job_id": "URGENT-1", "machine": "M1", "start_time": 3, "end_time": 5},
    )
    changes = changed_jobs(BASE_SCHEDULE, revised)
    added = changes[changes["change"] == "Added job"]
    assert len(added) == 1
    assert added.iloc[0]["job_id"] == "URGENT-1"


# ---------------------------------------------------------------------------
# Test 3: Removed job is detected
# ---------------------------------------------------------------------------
def test_removed_job_detected():
    revised = _sched(
        {"job_id": "J1", "machine": "M1", "start_time": 0, "end_time": 3},
        # J2 is missing
    )
    changes = changed_jobs(BASE_SCHEDULE, revised)
    removed = changes[changes["change"] == "Removed job"]
    assert len(removed) == 1
    assert removed.iloc[0]["job_id"] == "J2"


# ---------------------------------------------------------------------------
# Test 4: Identical schedules produce no changes
# ---------------------------------------------------------------------------
def test_identical_schedules_no_changes():
    changes = changed_jobs(BASE_SCHEDULE, IDENTICAL_SCHEDULE)
    # No rows or all rows should be empty / unchanged
    meaningful = changes[~changes["change"].isin(["Added job", "Removed job"])]
    assert meaningful.empty, f"Unexpected changes: {meaningful}"


# ---------------------------------------------------------------------------
# Test 5: build_explanation for Machine breakdown returns non-empty list
# ---------------------------------------------------------------------------
def test_build_explanation_breakdown():
    revised = _sched(
        {"job_id": "J1", "machine": "M1", "start_time": 5, "end_time": 8},
        {"job_id": "J2", "machine": "M2", "start_time": 0, "end_time": 2},
    )
    details = {"machine": "M1", "start": 0, "end": 5}
    explanations = build_explanation("Machine breakdown", BASE_SCHEDULE, revised, details)
    assert len(explanations) >= 1
    assert any("M1" in e or "breakdown" in e.lower() for e in explanations)


# ---------------------------------------------------------------------------
# Test 6: build_explanation for Add urgent job returns non-empty list
# ---------------------------------------------------------------------------
def test_build_explanation_urgent_job():
    revised = _sched(
        {"job_id": "J1", "machine": "M1", "start_time": 0, "end_time": 3},
        {"job_id": "J2", "machine": "M2", "start_time": 0, "end_time": 2},
        {"job_id": "URGENT-1", "machine": "M1", "start_time": 3, "end_time": 5},
    )
    details = {
        "job": {
            "job_id": "URGENT-1",
            "machine": "M1",
            "processing_time": 2,
            "deadline": 10,
            "release_time": 0,
        }
    }
    explanations = build_explanation("Add urgent job", BASE_SCHEDULE, revised, details)
    assert len(explanations) >= 1
    assert any("URGENT-1" in e for e in explanations)


# ---------------------------------------------------------------------------
# Test 7: changed_jobs with None inputs returns empty DataFrame
# ---------------------------------------------------------------------------
def test_changed_jobs_none_inputs():
    changes = changed_jobs(None, None)
    assert changes.empty


# ---------------------------------------------------------------------------
# Test 8: changed_jobs detects machine reassignment
# ---------------------------------------------------------------------------
def test_changed_machine_detected():
    revised = _sched(
        {"job_id": "J1", "machine": "M2", "start_time": 0, "end_time": 3},  # M1 → M2
        {"job_id": "J2", "machine": "M2", "start_time": 3, "end_time": 5},
    )
    changes = changed_jobs(BASE_SCHEDULE, revised)
    j1_row = changes[changes["job_id"] == "J1"]
    assert not j1_row.empty
    assert "machine" in j1_row.iloc[0]["change"]
