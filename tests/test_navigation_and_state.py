"""Regression tests for KarkhanaWatt navigation, session state persistence,
input validation, stale result invalidation, and independent schedule gating.
"""

from pathlib import Path
import copy
import pandas as pd
import pytest
import streamlit.testing.v1 as st_test

from src.data_loader import load_jobs
from src.tariff import load_tariff
from src.validator import validate_schedule
from src.optimizer import solve_schedule
from src.baseline import run_fcfs

APP_FILE = str(Path(__file__).resolve().parent.parent / "app.py")


def test_custom_jobs_persist_across_navigation():
    """Requirement 1 & 3: Upload custom jobs, navigate away and return;

    the custom dataset remains active and does not fall back to bundled sample data.
    """
    at = st_test.AppTest.from_file(APP_FILE)
    at.run()
    assert not at.exception

    # Create a custom jobs dataset with 3 unique jobs
    custom_df = pd.DataFrame([
        {"job_id": "CUSTOM_1", "processing_time": 2, "power_kw": 12.0, "deadline": 8, "machine": "M1"},
        {"job_id": "CUSTOM_2", "processing_time": 3, "power_kw": 15.0, "deadline": 10, "machine": "M2"},
        {"job_id": "CUSTOM_3", "processing_time": 1, "power_kw": 8.0, "deadline": 6, "machine": "M1"},
    ])

    # Inject into session state representing an uploaded custom dataset
    at.session_state["active_jobs_df"] = custom_df
    at.session_state["jobs_source"] = "Uploaded: my_custom_jobs.csv"
    at.session_state["schedule_time_unit"] = 60

    # Navigate to Dashboard
    at.sidebar.radio[0].set_value("Dashboard").run()
    assert not at.exception
    assert at.session_state["jobs_source"] == "Uploaded: my_custom_jobs.csv"
    assert len(at.session_state["active_jobs_df"]) == 3
    assert "CUSTOM_1" in at.session_state["active_jobs_df"]["job_id"].values

    # Navigate to Optimize & Compare
    at.sidebar.radio[0].set_value("Optimize & Compare").run()
    assert not at.exception
    assert at.session_state["jobs_source"] == "Uploaded: my_custom_jobs.csv"
    assert len(at.session_state["active_jobs_df"]) == 3

    # Navigate to What-If Analysis
    at.sidebar.radio[0].set_value("What-If Analysis").run()
    assert not at.exception
    assert at.session_state["jobs_source"] == "Uploaded: my_custom_jobs.csv"

    # Return to Setup & Data
    at.sidebar.radio[0].set_value("Setup & Data").run()
    assert not at.exception
    assert at.session_state["jobs_source"] == "Uploaded: my_custom_jobs.csv"
    assert len(at.session_state["active_jobs_df"]) == 3


def test_custom_tariff_persists_across_navigation():
    """Requirement 2: Upload custom tariff, navigate away and return;

    the custom tariff remains active and does not fall back to bundled data.
    """
    at = st_test.AppTest.from_file(APP_FILE)
    at.run()
    assert not at.exception

    # Custom flat 24-hour tariff of 11.5 Rs/kWh
    custom_tariff = [11.5] * 24
    at.session_state["active_tariff"] = custom_tariff
    at.session_state["tariff_source"] = "Uploaded: custom_industrial_tariff.csv"

    # Navigate across multiple sections
    for sec in ["Dashboard", "Optimize & Compare", "What-If Analysis", "Setup & Data"]:
        at.sidebar.radio[0].set_value(sec).run()
        assert not at.exception
        assert at.session_state["tariff_source"] == "Uploaded: custom_industrial_tariff.csv"
        assert at.session_state["active_tariff"] == custom_tariff


def test_change_inputs_after_optimization_marks_results_outdated():
    """Requirement 4: Changing inputs (jobs, tariff, or solver limit) after

    optimization marks prior results as outdated.
    """
    at = st_test.AppTest.from_file(APP_FILE)
    at.run()
    assert not at.exception

    # Run optimization on Optimize & Compare
    at.sidebar.radio[0].set_value("Optimize & Compare").run()
    run_btns = [b for b in at.button if b.key == "sched_page_run_btn"]
    assert len(run_btns) == 1
    run_btns[0].click().run()
    assert not at.exception
    assert "result" in at.session_state

    # Modify solver time limit
    at.sidebar.slider[0].set_value(30).run()
    assert not at.exception
    # Now the current run signature does not match original_sig
    # The app should show the outdated warning
    warnings = [w.value for w in at.warning]
    assert any("outdated" in w.lower() for w in warnings)


def test_invalid_upload_blocks_optimization_and_shows_error():
    """Requirement 5: If an uploaded dataset is invalid, optimization is blocked

    with a clear error; the app never silently substitutes sample data.
    """
    at = st_test.AppTest.from_file(APP_FILE)
    at.run()
    assert not at.exception

    # Simulate invalid jobs state (e.g. malformed CSV or validation failure)
    at.session_state["active_jobs_df"] = None
    at.session_state["jobs_source"] = "Invalid: broken.csv"
    at.session_state["upload_error_jobs"] = "Missing required column 'power_kw'"

    # Check Setup & Data shows the error
    at.sidebar.radio[0].set_value("Setup & Data").run()
    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("Missing required column 'power_kw'" in e for e in errors)

    # Check Optimize & Compare blocks run and does NOT fall back
    at.sidebar.radio[0].set_value("Optimize & Compare").run()
    assert not at.exception
    run_btn = [b for b in at.button if b.key == "sched_page_run_btn"][0]
    run_btn.click().run()
    errors = [e.value for e in at.error]
    assert any("jobs dataset is invalid" in e for e in errors)
    # Ensure active_jobs_df was NOT silently replaced with sample data
    assert at.session_state["active_jobs_df"] is None


def test_reset_restores_bundled_data_and_clears_old_results():
    """Requirement 6: Reset to Sample Data restores both bundled jobs and

    bundled tariff, and invalidates previous results.
    """
    at = st_test.AppTest.from_file(APP_FILE)
    at.run()
    assert not at.exception

    # First run optimization
    at.sidebar.radio[0].set_value("Optimize & Compare").run()
    run_btn = [b for b in at.button if b.key == "sched_page_run_btn"][0]
    run_btn.click().run()
    assert "result" in at.session_state

    # Modify jobs to a custom dataset
    at.session_state["active_jobs_df"] = pd.DataFrame([
        {"job_id": "X1", "processing_time": 1, "power_kw": 5.0, "deadline": 4, "machine": "M1"}
    ])
    at.session_state["jobs_source"] = "Custom X"

    # Click sidebar Reset to Sample Data button
    reset_btn = [b for b in at.button if b.key == "sidebar_reset_btn"][0]
    reset_btn.click().run()
    assert not at.exception

    # Verify bundled data is restored
    assert "Bundled" in at.session_state["jobs_source"]
    assert "Bundled" in at.session_state["tariff_source"]
    assert len(at.session_state["active_jobs_df"]) == 8  # default bundled jobs count (J1..J8)
    # Verify previous results are cleared
    assert "result" not in at.session_state
    assert "baseline_schedule" not in at.session_state
    assert "what_if_result" not in at.session_state


def test_what_if_requires_valid_current_reference_schedule():
    """Requirement 7: What-If Analysis requires a current valid reference schedule

    and informs the user when no results or outdated results exist.
    """
    at = st_test.AppTest.from_file(APP_FILE)
    at.run()
    assert not at.exception

    # Case A: No optimization run yet
    at.sidebar.radio[0].set_value("What-If Analysis").run()
    assert not at.exception
    infos = [i.value for i in at.info]
    assert any("Run baseline optimization" in i for i in infos)
    # Form submit button should not be present
    assert not any("Simulate and re-optimize" in b.label for b in at.button)

    # Case B: Outdated results
    at.session_state["result"] = "dummy_result"
    at.session_state["run_signature"] = "outdated_signature"
    at.sidebar.radio[0].set_value("What-If Analysis").run()
    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert any("outdated" in w.lower() for w in warnings)
    assert not any("Simulate and re-optimize" in b.label for b in at.button)


def test_url_query_params_and_in_app_navigation():
    """Requirement 8: URL query params synchronize with navigation selection,

    and in-app navigation buttons transition cleanly between sections.
    """
    at = st_test.AppTest.from_file(APP_FILE)
    at.run()
    assert not at.exception

    # Set URL query parameter to "Setup & Data"
    at.query_params["section"] = "Setup & Data"
    at.run()
    assert not at.exception
    assert at.session_state["app_navigation_selection"] == "Setup & Data"

    # Click in-app navigation button to Optimize & Compare
    next_btn = [b for b in at.button if b.key == "setup_to_compare_btn"][0]
    next_btn.click().run()
    assert not at.exception
    assert at.session_state["app_navigation_selection"] == "Optimize & Compare"
    assert at.query_params.get("section") == "Optimize & Compare"


def test_optimized_schedule_passes_independent_validation():
    """Requirement 9: The optimized schedule produced by CP-SAT passes

    independent schedule validation.
    """
    sample_jobs = load_jobs("data/jobs.csv")
    sample_tariff = load_tariff("data/tariff.csv")
    res = solve_schedule(sample_jobs, sample_tariff, time_limit_seconds=10)
    assert res.status in ("OPTIMAL", "FEASIBLE")

    val = validate_schedule(
        schedule=res.schedule,
        jobs_df=sample_jobs,
        hourly_tariff=sample_tariff,
        time_unit_minutes=60,
        reported_cost=res.total_cost,
    )
    assert val.passed, f"Independent validation failed: {val.errors}"


def test_full_workflow_optimization_downloads_and_whatif():
    """Requirement 10: Complete workflow from optimization, CSV downloads,

    to What-If machine breakdown and urgent job simulations.
    """
    at = st_test.AppTest.from_file(APP_FILE)
    at.run()
    assert not at.exception

    # 1. Optimize on Optimize & Compare
    at.sidebar.radio[0].set_value("Optimize & Compare").run()
    run_btn = [b for b in at.button if b.key == "sched_page_run_btn"][0]
    run_btn.click().run()
    assert not at.exception
    assert len(at.dataframe) > 0
    # Download button present
    dl_btns = [b for b in at.download_button if b.key == "download_sched_page_csv"]
    assert len(dl_btns) == 1

    # 2. What-If Machine Breakdown
    at.sidebar.radio[0].set_value("What-If Analysis").run()
    assert not at.exception
    sim_btn = [b for b in at.button if "Simulate and re-optimize" in b.label][0]
    sim_btn.click().run()
    assert not at.exception
    assert "what_if_result" in at.session_state
    assert len(at.download_button) > 0

    # 3. What-If Urgent Job
    whatif_radio = [r for r in at.radio if r.key == "what_if_scenario_type"][0]
    whatif_radio.set_value("Add urgent job").run()
    assert not at.exception
    sim_btn = [b for b in at.button if "Simulate and re-optimize" in b.label][0]
    sim_btn.click().run()
    assert not at.exception
    assert at.session_state["what_if_result"]["type"] == "Add urgent job"
