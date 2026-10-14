"""
Shared pytest fixtures for KarkhanaWatt tests.
"""

import pandas as pd
import pytest


@pytest.fixture
def simple_tariff():
    """Flat tariff: Rs 6.00/kWh for all 24 hours."""
    return [6.0] * 24


@pytest.fixture
def peak_tariff():
    """Off-peak Rs 4.0, peak (hours 9-17) Rs 10.0."""
    rates = [4.0] * 24
    for h in range(9, 18):
        rates[h] = 10.0
    return rates


@pytest.fixture
def simple_jobs_df():
    """Three independent jobs on two machines."""
    return pd.DataFrame(
        {
            "job_id": ["J1", "J2", "J3"],
            "machine": ["M1", "M2", "M1"],
            "processing_time": [3, 2, 4],
            "deadline": [20, 20, 20],
            "power_kw": [10.0, 20.0, 15.0],
            "arrival_order": [1, 2, 3],
            "predecessor": [None, None, None],
        }
    )


@pytest.fixture
def chained_jobs_df():
    """J1 → J2 → J3 dependency chain on one machine."""
    return pd.DataFrame(
        {
            "job_id": ["J1", "J2", "J3"],
            "machine": ["M1", "M1", "M1"],
            "processing_time": [2, 3, 1],
            "deadline": [20, 20, 20],
            "power_kw": [10.0, 10.0, 10.0],
            "arrival_order": [1, 2, 3],
            "predecessor": [None, "J1", "J2"],
        }
    )


@pytest.fixture
def sample_schedule_df():
    """A minimal valid schedule for two jobs."""
    return pd.DataFrame(
        {
            "job_id": ["J1", "J2"],
            "machine": ["M1", "M2"],
            "start_time": [0, 0],
            "end_time": [3, 2],
            "processing_time": [3, 2],
            "deadline": [10, 10],
            "power_kw": [10.0, 20.0],
            "deadline_violated": [False, False],
        }
    )
