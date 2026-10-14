"""
Tests for src/data_loader.py — job loading, validation, and manufacturing data.

Covers:
1. Valid jobs CSV loads correctly.
2. Missing required column raises ValueError.
3. Duplicate job_id raises ValueError.
4. Non-numeric processing_time raises ValueError.
5. Negative processing_time raises ValueError.
6. Circular dependency in predecessor column raises ValueError.
7. Self-dependency raises ValueError.
8. Missing predecessor raises ValueError.
9. Manufacturing CSV loads with correct field mapping.
10. prepare_manufacturing_jobs without confirmation raises ValueError.
11. prepare_manufacturing_jobs respects max_jobs limit.
12. validate_jobs detects infeasible jobs (processing_time > deadline).
"""

import io
import pytest
import pandas as pd

from src.data_loader import load_jobs, validate_jobs, load_manufacturing_records, prepare_manufacturing_jobs


# ---------------------------------------------------------------------------
# Helper: build a CSV string and wrap in a file-like buffer
# ---------------------------------------------------------------------------
def _csv_buf(content: str) -> io.StringIO:
    return io.StringIO(content.strip())


VALID_JOBS_CSV = """
job_id,machine,processing_time,deadline,power_kw,arrival_order,predecessor
J1,M1,3,20,10.0,1,
J2,M2,2,20,20.0,2,
J3,M1,4,20,15.0,3,J1
"""


# ---------------------------------------------------------------------------
# Test 1: Valid jobs CSV loads correctly
# ---------------------------------------------------------------------------
def test_valid_jobs_csv_loads():
    df = load_jobs(_csv_buf(VALID_JOBS_CSV))
    assert len(df) == 3
    assert set(df["job_id"]) == {"J1", "J2", "J3"}
    assert df["processing_time"].dtype in ("int64", "int32", int)
    # Predecessor is None for J1/J2, "J1" for J3
    assert df[df["job_id"] == "J3"]["predecessor"].iloc[0] == "J1"
    assert df[df["job_id"] == "J1"]["predecessor"].iloc[0] is None


# ---------------------------------------------------------------------------
# Test 2: Missing required column raises ValueError
# ---------------------------------------------------------------------------
def test_missing_column_raises():
    csv = "job_id,machine,processing_time,deadline,power_kw\nJ1,M1,3,20,10.0\n"
    with pytest.raises(ValueError, match="arrival_order"):
        load_jobs(_csv_buf(csv))


# ---------------------------------------------------------------------------
# Test 3: Duplicate job_id raises ValueError
# ---------------------------------------------------------------------------
def test_duplicate_job_id_raises():
    csv = """
job_id,machine,processing_time,deadline,power_kw,arrival_order,predecessor
J1,M1,3,20,10.0,1,
J1,M2,2,20,20.0,2,
"""
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        load_jobs(_csv_buf(csv))


# ---------------------------------------------------------------------------
# Test 4: Non-numeric processing_time raises ValueError
# ---------------------------------------------------------------------------
def test_non_numeric_processing_time_raises():
    csv = """
job_id,machine,processing_time,deadline,power_kw,arrival_order,predecessor
J1,M1,three,20,10.0,1,
"""
    with pytest.raises(ValueError):
        load_jobs(_csv_buf(csv))


# ---------------------------------------------------------------------------
# Test 5: Negative / zero processing_time raises ValueError
# ---------------------------------------------------------------------------
def test_zero_processing_time_raises():
    csv = """
job_id,machine,processing_time,deadline,power_kw,arrival_order,predecessor
J1,M1,0,20,10.0,1,
"""
    with pytest.raises(ValueError, match="processing_time"):
        load_jobs(_csv_buf(csv))


# ---------------------------------------------------------------------------
# Test 6: Circular dependency raises ValueError
# ---------------------------------------------------------------------------
def test_circular_dependency_in_csv_raises():
    csv = """
job_id,machine,processing_time,deadline,power_kw,arrival_order,predecessor
J1,M1,3,20,10.0,1,J2
J2,M2,2,20,20.0,2,J1
"""
    with pytest.raises(ValueError, match="[Cc]ircular|[Cc]ycle"):
        load_jobs(_csv_buf(csv))


# ---------------------------------------------------------------------------
# Test 7: Self-dependency raises ValueError
# ---------------------------------------------------------------------------
def test_self_dependency_raises():
    csv = """
job_id,machine,processing_time,deadline,power_kw,arrival_order,predecessor
J1,M1,3,20,10.0,1,J1
"""
    with pytest.raises(ValueError, match="own predecessor|self|itself"):
        load_jobs(_csv_buf(csv))


# ---------------------------------------------------------------------------
# Test 8: Missing predecessor reference raises ValueError
# ---------------------------------------------------------------------------
def test_missing_predecessor_raises():
    csv = """
job_id,machine,processing_time,deadline,power_kw,arrival_order,predecessor
J1,M1,3,20,10.0,1,J_MISSING
"""
    with pytest.raises(ValueError, match="unknown|not exist|predecessor"):
        load_jobs(_csv_buf(csv))


# ---------------------------------------------------------------------------
# Test 9: validate_jobs detects processing_time > deadline
# ---------------------------------------------------------------------------
def test_infeasible_job_raises():
    df = pd.DataFrame(
        {
            "job_id": ["J1"],
            "machine": ["M1"],
            "processing_time": [10],
            "deadline": [5],   # impossible: pt > deadline
            "power_kw": [10.0],
            "arrival_order": [1],
            "predecessor": [None],
        }
    )
    with pytest.raises(ValueError, match="cannot finish"):
        validate_jobs(df)


# ---------------------------------------------------------------------------
# Test 10: Manufacturing CSV loads with field mapping
# ---------------------------------------------------------------------------
def test_manufacturing_csv_loads():
    csv = """
Job_ID,Machine_ID,Processing_Time,Energy_Consumption,Scheduled_Start,Scheduled_End,Job_Status
JA,MA,60,5.0,2024-01-01 08:00,2024-01-01 09:00,Completed
JB,MB,120,8.0,2024-01-01 09:00,2024-01-01 11:00,Completed
"""
    df = load_manufacturing_records(_csv_buf(csv))
    assert len(df) == 2
    assert "processing_time_raw" in df.columns
    assert "energy_consumption_raw" in df.columns
    assert "machine_id" in df.columns


# ---------------------------------------------------------------------------
# Test 11: prepare_manufacturing_jobs without confirmation raises ValueError
# ---------------------------------------------------------------------------
def test_prepare_without_confirmation_raises():
    df = pd.DataFrame(
        {
            "job_id": ["JA"],
            "machine_id": ["MA"],
            "processing_time_raw": [60.0],
            "energy_consumption_raw": [5.0],
            "scheduled_start": pd.to_datetime(["2024-01-01 08:00"]),
            "scheduled_end": pd.to_datetime(["2024-01-01 09:00"]),
        }
    )
    with pytest.raises(ValueError, match="[Ee]nergy units|[Ee]nergy.*confirm"):
        prepare_manufacturing_jobs(df, assume_energy_is_kwh_per_job=False)


# ---------------------------------------------------------------------------
# Test 12: prepare_manufacturing_jobs respects max_jobs limit
# ---------------------------------------------------------------------------
def test_prepare_manufacturing_jobs_max_jobs():
    rows = {
        "job_id": [f"J{i}" for i in range(10)],
        "machine_id": ["MA"] * 10,
        "processing_time_raw": [60.0] * 10,
        "energy_consumption_raw": [5.0] * 10,
        "scheduled_start": pd.date_range("2024-01-01 08:00", periods=10, freq="1h"),
        "scheduled_end": pd.date_range("2024-01-01 09:00", periods=10, freq="1h"),
    }
    df = pd.DataFrame(rows)
    result = prepare_manufacturing_jobs(df, max_jobs=3, assume_energy_is_kwh_per_job=True)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# Test 13: load_jobs — predecessor column is optional (no column = all None)
# ---------------------------------------------------------------------------
def test_predecessor_column_optional():
    csv = """
job_id,machine,processing_time,deadline,power_kw,arrival_order
J1,M1,3,20,10.0,1
J2,M2,2,20,20.0,2
"""
    df = load_jobs(_csv_buf(csv))
    assert "predecessor" in df.columns
    assert df["predecessor"].isna().all() or (df["predecessor"] == None).all()
