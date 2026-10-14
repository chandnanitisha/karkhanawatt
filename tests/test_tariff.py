"""
Tests for src/tariff.py — cost calculation and tariff boundary correctness.

Covers:
1. Job running entirely within one tariff interval.
2. Job spanning two tariff intervals.
3. Job spanning three or more intervals.
4. Job ending exactly at a tariff boundary.
5. Job beginning exactly at a tariff boundary.
6. Minute-resolution timing.
7. Zero-duration schedule returns 0 cost.
8. schedule_cost vs manually computed cost (exact agreement).
9. build_cumulative_table — correct prefix sums.
10. Negative / out-of-range tariff index wraps correctly via hourly_rate.
"""

import math
import pytest
import pandas as pd

from src.tariff import schedule_cost, build_cumulative_table, hourly_rate, TARIFF_SCALE


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _make_schedule(job_id, machine, start, end, power_kw, deadline=100):
    return pd.DataFrame(
        {
            "job_id": [job_id],
            "machine": [machine],
            "start_time": [start],
            "end_time": [end],
            "processing_time": [end - start],
            "deadline": [deadline],
            "power_kw": [power_kw],
            "deadline_violated": [False],
        }
    )


# ---------------------------------------------------------------------------
# Test 1: Job entirely within one hourly tariff interval (hours 0–1)
# ---------------------------------------------------------------------------
def test_single_interval_whole_hour(simple_tariff):
    """10 kW × 1 h × 6 Rs/kWh = Rs 6.00"""
    sched = _make_schedule("J1", "M1", start=0, end=1, power_kw=10.0)
    cost = schedule_cost(sched, simple_tariff, time_unit_minutes=60)
    assert cost == pytest.approx(60.0, abs=0.01)


# ---------------------------------------------------------------------------
# Test 2: Job spanning two hourly intervals
# ---------------------------------------------------------------------------
def test_two_intervals(peak_tariff):
    """
    Job from hour 8 to hour 10 (2 hours).
    Hour 8 is off-peak (Rs 4.0), hour 9 is peak (Rs 10.0).
    Cost = 10 kW × (4.0 + 10.0) = Rs 140.0
    """
    sched = _make_schedule("J1", "M1", start=8, end=10, power_kw=10.0)
    cost = schedule_cost(sched, peak_tariff, time_unit_minutes=60)
    assert cost == pytest.approx(140.0, abs=0.01)


# ---------------------------------------------------------------------------
# Test 3: Job spanning three hourly intervals
# ---------------------------------------------------------------------------
def test_three_intervals(peak_tariff):
    """
    Job from hour 8 to hour 11 (3 hours).
    Hours 8=4.0, 9=10.0, 10=10.0. Cost = 5 kW × (4+10+10) = Rs 120.0
    """
    sched = _make_schedule("J1", "M1", start=8, end=11, power_kw=5.0)
    cost = schedule_cost(sched, peak_tariff, time_unit_minutes=60)
    assert cost == pytest.approx(120.0, abs=0.01)


# ---------------------------------------------------------------------------
# Test 4: Job ending exactly at a tariff boundary
# ---------------------------------------------------------------------------
def test_ends_at_tariff_boundary(peak_tariff):
    """Job from hour 7 to hour 9 (2 hours). Hours 7=4.0, 8=4.0."""
    sched = _make_schedule("J1", "M1", start=7, end=9, power_kw=10.0)
    cost = schedule_cost(sched, peak_tariff, time_unit_minutes=60)
    assert cost == pytest.approx(80.0, abs=0.01)


# ---------------------------------------------------------------------------
# Test 5: Job beginning exactly at a tariff boundary
# ---------------------------------------------------------------------------
def test_begins_at_tariff_boundary(peak_tariff):
    """Job from hour 9 to hour 11 (peak hours 9,10). Cost = 5 × (10+10) = 100."""
    sched = _make_schedule("J1", "M1", start=9, end=11, power_kw=5.0)
    cost = schedule_cost(sched, peak_tariff, time_unit_minutes=60)
    assert cost == pytest.approx(100.0, abs=0.01)


# ---------------------------------------------------------------------------
# Test 6: Minute-resolution timing (time_unit_minutes=1)
# ---------------------------------------------------------------------------
def test_minute_resolution(simple_tariff):
    """
    60-minute job (start=0, end=60) at 10 kW, flat Rs 6.0/kWh.
    Energy = 10 kW × (60 × 1/60 h) = 10 kWh. Cost = Rs 60.
    Each minute contributes 10 kW × (1/60 h) × 6 Rs/kWh.
    """
    sched = _make_schedule("J1", "M1", start=0, end=60, power_kw=10.0)
    cost = schedule_cost(sched, simple_tariff, time_unit_minutes=1)
    assert cost == pytest.approx(60.0, abs=0.01)


# ---------------------------------------------------------------------------
# Test 7: Empty schedule returns 0.0
# ---------------------------------------------------------------------------
def test_empty_schedule_cost(simple_tariff):
    empty = pd.DataFrame(
        columns=["job_id", "machine", "start_time", "end_time",
                 "processing_time", "deadline", "power_kw", "deadline_violated"]
    )
    assert schedule_cost(empty, simple_tariff) == 0.0


# ---------------------------------------------------------------------------
# Test 8: Cost equals manually computed value (independent check)
# ---------------------------------------------------------------------------
def test_cost_matches_manual_calculation():
    """
    Two jobs, mixed tariff.
    J1: M1, start=0, end=2, power=20 kW. Rate hours 0,1 = 4.0.  Cost = 20×4+20×4=160
    J2: M2, start=1, end=3, power=10 kW. Rate hours 1=4.0, 2=10.0. Cost=10×4+10×10=140
    Total = 300 Rs.
    """
    rates = [4.0] * 24
    rates[2] = 10.0

    sched = pd.DataFrame(
        {
            "job_id": ["J1", "J2"],
            "machine": ["M1", "M2"],
            "start_time": [0, 1],
            "end_time": [2, 3],
            "processing_time": [2, 2],
            "deadline": [10, 10],
            "power_kw": [20.0, 10.0],
            "deadline_violated": [False, False],
        }
    )
    cost = schedule_cost(sched, rates, time_unit_minutes=60)
    expected = 20 * 4.0 + 20 * 4.0 + 10 * 4.0 + 10 * 10.0
    assert cost == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------------------------
# Test 9: build_cumulative_table prefix sums are correct
# ---------------------------------------------------------------------------
def test_cumulative_table_prefix_sums(simple_tariff):
    """
    Flat 6.0 Rs/kWh. TARIFF_SCALE=6000.
    Each hour contributes 6.0 × 6000 = 36000 scaled units.
    F[0]=0, F[1]=36000, F[2]=72000, ...
    """
    F = build_cumulative_table(simple_tariff, horizon=3, time_unit_minutes=60)
    expected_step = round(6.0 * TARIFF_SCALE)  # 36000
    assert F[0] == 0
    assert F[1] == expected_step
    assert F[2] == 2 * expected_step
    assert F[3] == 3 * expected_step


# ---------------------------------------------------------------------------
# Test 10: hourly_rate wraps correctly past 24 hours
# ---------------------------------------------------------------------------
def test_hourly_rate_wraps(peak_tariff):
    """Hour 25 should map to rate at hour 1 (off-peak = 4.0)."""
    assert hourly_rate(peak_tariff, 25) == pytest.approx(4.0)
    assert hourly_rate(peak_tariff, 9) == pytest.approx(10.0)
    assert hourly_rate(peak_tariff, 33) == pytest.approx(10.0)  # 33 % 24 = 9


# ---------------------------------------------------------------------------
# Test 11: Invalid time_unit_minutes raises ValueError
# ---------------------------------------------------------------------------
def test_invalid_time_unit_raises(simple_tariff):
    sched = _make_schedule("J1", "M1", 0, 1, 10.0)
    with pytest.raises(ValueError):
        schedule_cost(sched, simple_tariff, time_unit_minutes=7)
    with pytest.raises(ValueError):
        schedule_cost(sched, simple_tariff, time_unit_minutes=0)
    with pytest.raises(ValueError):
        schedule_cost(sched, simple_tariff, time_unit_minutes=-1)
