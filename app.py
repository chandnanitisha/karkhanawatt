"""KarkhanaWatt: energy-aware production scheduling with OR-Tools CP-SAT.

Run from the project folder with:
    python -m streamlit run app.py

Consolidated into four core workflow sections:
1. Dashboard: Key performance indicators, executive summaries, Gantt timeline, and analytics.
2. Setup & Data: Standard jobs, external manufacturing data, electricity tariffs, and validation.
3. Optimize & Compare: FCFS baseline vs CP-SAT energy-aware optimization, metrics, and schedules.
4. What-If Analysis: Disruption simulation (machine breakdowns & urgent jobs) against reference plans.
"""

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import (
    load_jobs,
    get_machines,
    load_manufacturing_records,
    prepare_manufacturing_jobs,
)
from src.baseline import run_fcfs
from src.optimizer import solve_schedule
from src.metrics import compute_metrics, comparison_table
from src.visualization import make_gantt_chart, make_tariff_chart
from src.tariff import load_tariff, schedule_cost
from src.what_if import changed_jobs, build_explanation
from src.validator import validate_schedule


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

SECTIONS = [
    "Dashboard",
    "Setup & Data",
    "Optimize & Compare",
    "What-If Analysis",
]


# ======================================================================
# UI THEME INJECTION (Consistent Dark-Navy, Teal & Amber styling)
# ======================================================================
def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
          --navy: #142b45;
          --navy-2: #1d3b5a;
          --green: #15966a;
          --bg: #f5f7fa;
          --card: #ffffff;
          --text: #172b3a;
          --muted: #64748b;
          --border: #e2e8f0;
          --warning: #d97706;
          --error: #dc2626;
          --info: #2563eb;
          --radius: 11px;
        }

        /* App base styling */
        html, body, [data-testid="stAppViewContainer"] {
          background-color: var(--bg) !important;
          color: var(--text) !important;
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }

        [data-testid="stAppViewBlockContainer"] {
          max-width: 1440px !important;
          padding-top: 1.2rem !important;
          padding-bottom: 3.5rem !important;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
          background-color: var(--navy) !important;
          border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        }
        [data-testid="stSidebar"] * {
          color: #bac7d3;
        }
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
          color: #ffffff !important;
        }
        [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
          color: #bac7d3 !important;
        }

        /* Brand section in sidebar */
        .kw-brand {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 8px 0 20px 0;
          border-bottom: 1px solid rgba(255, 255, 255, 0.08);
          margin-bottom: 18px;
        }
        .kw-brand-mark {
          width: 38px;
          height: 38px;
          border-radius: 9px;
          background: linear-gradient(135deg, #10b981 0%, #059669 100%);
          display: flex;
          align-items: center;
          justify-content: center;
          color: white;
          font-weight: 700;
          box-shadow: 0 4px 10px rgba(16, 185, 129, 0.35);
        }
        .kw-brand-copy b {
          display: block;
          font-size: 16px;
          color: #ffffff;
          line-height: 1.2;
        }
        .kw-brand-copy b span {
          color: #34d399;
        }
        .kw-brand-copy small {
          display: block;
          font-size: 11px;
          color: #7f97ab;
        }

        /* Top Bar & Breadcrumb */
        .kw-topbar {
          background: var(--card);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 10px 18px;
          margin-bottom: 20px;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }
        .kw-breadcrumbs {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 13px;
          color: var(--muted);
        }
        .kw-breadcrumbs b {
          color: var(--navy);
        }
        .kw-topbar-actions {
          display: flex;
          align-items: center;
          gap: 10px;
        }
        .kw-badge {
          display: inline-block;
          font-size: 11px;
          font-weight: 600;
          padding: 3px 8px;
          border-radius: 6px;
        }
        .kw-badge-success { background: #dcfce7; color: #15803d; }
        .kw-badge-info    { background: #e0f2fe; color: #0369a1; }
        .kw-badge-neutral { background: #f1f5f9; color: #475569; }
        .kw-badge-warning { background: #fef3c7; color: #b45309; }
        .kw-badge-danger  { background: #fee2e2; color: #b91c1c; }

        /* KPI cards grid */
        .kw-kpi-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 14px;
          margin-bottom: 20px;
        }
        .kw-kpi {
          background: var(--card);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 16px 14px;
          display: flex;
          gap: 12px;
          align-items: flex-start;
          transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .kw-kpi:hover {
          transform: translateY(-2px);
          box-shadow: 0 6px 14px rgba(0,0,0,0.04);
        }
        .kw-kpi-icon {
          width: 36px;
          height: 36px;
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 16px;
          flex-shrink: 0;
        }
        .kw-kpi-icon.navy   { background: #e2e8f0; color: #1e293b; }
        .kw-kpi-icon.green  { background: #dcfce7; color: #15803d; }
        .kw-kpi-icon.orange { background: #ffedd5; color: #c2410c; }
        .kw-kpi-icon.blue   { background: #e0f2fe; color: #0284c7; }

        .kw-kpi-copy p {
          margin: 0;
          font-size: 11px;
          font-weight: 600;
          color: var(--muted);
          text-transform: uppercase;
          letter-spacing: 0.4px;
        }
        .kw-kpi-copy strong {
          display: block;
          font-size: 20px;
          font-weight: 700;
          color: var(--navy);
          line-height: 1.25;
          margin: 3px 0 2px 0;
        }
        .kw-kpi-copy span {
          font-size: 11px;
          color: var(--muted);
        }

        /* Generic Card */
        .kw-card {
          background: var(--card);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 20px;
          margin-bottom: 20px;
        }
        .kw-section-header {
          margin-bottom: 14px;
        }
        .kw-section-header h2, .kw-section-header h3 {
          margin: 0 0 4px 0 !important;
          color: var(--navy) !important;
          font-weight: 700 !important;
        }
        .kw-section-header p {
          margin: 0 !important;
          font-size: 12px !important;
          color: var(--muted) !important;
        }

        /* Comparison head */
        .kw-comparison-head {
          display: grid;
          grid-template-columns: 1fr auto 1fr 1fr;
          gap: 16px;
          align-items: center;
          margin-bottom: 18px;
        }
        .kw-card-baseline, .kw-card-optimized {
          background: var(--card);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 18px;
          display: flex;
          flex-direction: column;
        }
        .kw-card-saving {
          background: linear-gradient(135deg, #142b45 0%, #1a3c61 100%);
          color: white;
          border-radius: var(--radius);
          padding: 18px;
          display: flex;
          flex-direction: column;
        }
        .kw-card-saving strong {
          font-size: 28px;
          color: #34d399;
          margin: 4px 0;
        }
        .kw-versus {
          font-size: 11px;
          font-weight: 700;
          color: var(--muted);
          background: #f1f5f9;
          padding: 4px 8px;
          border-radius: 5px;
          text-align: center;
        }

        /* Metric strip */
        .kw-metric-strip {
          background: #f8fafc;
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 14px 18px;
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
          gap: 14px;
          margin-bottom: 20px;
        }
        .kw-metric-strip > div {
          display: flex;
          flex-direction: column;
        }
        .kw-metric-strip span {
          font-size: 11px;
          color: var(--muted);
          font-weight: 600;
          text-transform: uppercase;
        }
        .kw-metric-strip b {
          font-size: 15px;
          color: var(--navy);
          margin-top: 2px;
        }
        .kw-metric-strip small {
          font-size: 10px;
          color: var(--muted);
        }

        /* Sidebar Footer */
        .kw-sidebar-footer {
          margin-top: 30px;
          padding-top: 15px;
          border-top: 1px solid rgba(255, 255, 255, 0.08);
        }
        .kw-project-badge {
          display: flex;
          align-items: center;
          gap: 10px;
        }
        .kw-project-badge span {
          background: rgba(255, 255, 255, 0.1);
          color: #fff;
          font-size: 11px;
          font-weight: 700;
          padding: 4px 6px;
          border-radius: 5px;
        }
        .kw-project-badge b {
          color: #fff;
          font-size: 12px;
          display: block;
        }
        .kw-project-badge small {
          color: #7f97ab;
          font-size: 10px;
          display: block;
        }

        /* Page header */
        .kw-page-header {
          margin-bottom: 22px;
        }
        .kw-page-header h1 {
          font-size: 24px !important;
          color: var(--navy) !important;
          font-weight: 700 !important;
          margin: 0 0 6px 0 !important;
        }
        .kw-page-header p {
          font-size: 13px !important;
          color: var(--muted) !important;
          margin: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ======================================================================
# DATA FORMATTING & HELPER FUNCTIONS
# ======================================================================
def schedule_as_hours(schedule: pd.DataFrame, time_unit_minutes: int) -> pd.DataFrame:
    """Format schedule times in hours for display and charting."""
    if schedule is None or schedule.empty:
        return schedule
    out = schedule.copy()
    if time_unit_minutes == 60:
        return out
    units_per_hour = 60 // time_unit_minutes
    for col in ("start_time", "end_time", "processing_time", "deadline", "release_time"):
        if col in out.columns:
            out[col] = (out[col] / units_per_hour).round(3)
    return out


def display_schedule_table(schedule_hours: pd.DataFrame) -> pd.DataFrame:
    """Format table column names for user display."""
    if schedule_hours is None or schedule_hours.empty:
        return pd.DataFrame()
    col_order = [
        c for c in [
            "job_id", "machine", "start_time", "end_time", "processing_time",
            "deadline", "release_time", "power_kw", "deadline_violated", "predecessor"
        ] if c in schedule_hours.columns
    ]
    return schedule_hours[col_order].rename(
        columns={
            "start_time": "start_hour",
            "end_time": "end_hour",
            "processing_time": "processing_time_hours",
            "deadline": "reference_deadline_hours",
            "release_time": "release_time_hours",
        }
    )


def signature_for_run(df: pd.DataFrame | None, tariff: list | None, time_limit: int,
                      time_unit_minutes: int, source: str) -> str:
    """Create a deterministic hash of current inputs to detect stale optimization results."""
    if df is None or tariff is None:
        return ""
    payload = {
        "jobs": df.to_json(orient="records", date_format="iso"),
        "tariff": tariff,
        "time_limit": time_limit,
        "time_unit_minutes": time_unit_minutes,
        "source": source,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def render_topbar(current_page: str, dataset_status: str = "Ready"):
    st.markdown(
        f"""
        <div class="kw-topbar">
          <div class="kw-breadcrumbs">
            <span>KarkhanaWatt</span>
            <span style="color:#cbd5e1;">›</span>
            <b>{current_page}</b>
          </div>
          <div class="kw-topbar-actions">
            <span class="kw-badge kw-badge-info">{dataset_status}</span>
            <span class="kw-badge kw-badge-success">CP-SAT Optimizer</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def execute_optimization(
    jobs_df: pd.DataFrame,
    tariff: list,
    t_limit: int,
    unit_minutes: int,
    run_sig: str,
    strategy: str = "cost",
):
    """Run baseline FCFS and CP-SAT optimizer and persist in session state."""
    with st.spinner("Building the FCFS baseline..."):
        baseline_sched = run_fcfs(jobs_df)
        baseline_mets = compute_metrics(
            baseline_sched,
            tariff,
            time_unit_minutes=unit_minutes,
        )

    with st.spinner("Solving with OR-Tools CP-SAT..."):
        res = solve_schedule(
            jobs_df,
            tariff,
            time_limit_seconds=t_limit,
            time_unit_minutes=unit_minutes,
            strategy=strategy,
        )

    st.session_state["baseline_schedule"] = baseline_sched
    st.session_state["baseline_metrics"] = baseline_mets
    st.session_state["result"] = res
    st.session_state["run_signature"] = run_sig
    st.session_state["active_strategy"] = strategy
    st.session_state.pop("what_if_result", None)


def reset_to_sample_data():
    """Explicitly restore bundled jobs and tariff, reset units, and invalidate previous results."""
    st.session_state["active_jobs_df"] = load_jobs(DATA_DIR / "jobs.csv")
    st.session_state["jobs_source"] = "Bundled sample: data/jobs.csv"
    st.session_state["schedule_time_unit"] = 60
    st.session_state["upload_error_jobs"] = None
    st.session_state["_last_jobs_upload_key"] = None
    st.session_state["raw_manufacturing_df"] = None
    st.session_state["_last_mfg_upload_key"] = None

    st.session_state["active_tariff"] = load_tariff(DATA_DIR / "tariff.csv")
    st.session_state["tariff_source"] = "Bundled tariff: data/tariff.csv"
    st.session_state["upload_error_tariff"] = None
    st.session_state["_last_tariff_upload_key"] = None

    st.session_state.pop("result", None)
    st.session_state.pop("baseline_schedule", None)
    st.session_state.pop("baseline_metrics", None)
    st.session_state.pop("run_signature", None)
    st.session_state.pop("what_if_result", None)


def navigate_to(section_name: str):
    """Synchronize URL query params and session state for predictable in-app navigation."""
    if section_name in SECTIONS:
        st.session_state["pending_nav"] = section_name
        st.query_params["section"] = section_name
        st.rerun()


# ======================================================================
# PAGE CONFIGURATION & INITIALIZATION
# ======================================================================
st.set_page_config(page_title="KarkhanaWatt — Energy-Aware Production Scheduler", layout="wide")
inject_custom_css()

# Initialize authoritative session state
if "active_jobs_df" not in st.session_state:
    try:
        st.session_state["active_jobs_df"] = load_jobs(DATA_DIR / "jobs.csv")
        st.session_state["jobs_source"] = "Bundled sample: data/jobs.csv"
        st.session_state["schedule_time_unit"] = 60
        st.session_state["upload_error_jobs"] = None
    except Exception as exc:
        st.session_state["active_jobs_df"] = None
        st.session_state["jobs_source"] = "None"
        st.session_state["schedule_time_unit"] = 60
        st.session_state["upload_error_jobs"] = str(exc)

if "active_tariff" not in st.session_state:
    try:
        st.session_state["active_tariff"] = load_tariff(DATA_DIR / "tariff.csv")
        st.session_state["tariff_source"] = "Bundled tariff: data/tariff.csv"
        st.session_state["upload_error_tariff"] = None
    except Exception as exc:
        st.session_state["active_tariff"] = None
        st.session_state["tariff_source"] = "None"
        st.session_state["upload_error_tariff"] = str(exc)

# Sync URL query parameters and pending navigation before widget creation
url_section = st.query_params.get("section")
if "last_synced_url" not in st.session_state:
    init_nav = url_section if url_section in SECTIONS else "Dashboard"
    st.session_state["sidebar_nav_radio"] = init_nav
    st.session_state["last_synced_url"] = init_nav
    st.query_params["section"] = init_nav
elif url_section in SECTIONS and url_section != st.session_state.get("last_synced_url"):
    st.session_state["sidebar_nav_radio"] = url_section
    st.session_state["last_synced_url"] = url_section

if "pending_nav" in st.session_state and st.session_state["pending_nav"] in SECTIONS:
    target = st.session_state.pop("pending_nav")
    st.session_state["sidebar_nav_radio"] = target
    st.session_state["last_synced_url"] = target
    st.query_params["section"] = target

# ======================================================================
# SIDEBAR
# ======================================================================
st.sidebar.markdown(
    """
    <div class="kw-brand">
      <div class="kw-brand-mark">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M3 21V9l6 3V8l6 4V4h6v17z"/>
          <path d="M7 17h2M13 17h2M18 9h3"/>
        </svg>
      </div>
      <div class="kw-brand-copy">
        <b>Karkhana<span>Watt</span></b>
        <small>Energy-aware production</small>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

nav_selection = st.sidebar.radio(
    "Navigation",
    SECTIONS,
    label_visibility="collapsed",
    key="sidebar_nav_radio",
)

st.session_state["app_navigation_selection"] = nav_selection
st.session_state["last_synced_url"] = nav_selection
st.query_params["section"] = nav_selection

st.sidebar.markdown("---")
st.sidebar.markdown("<small style='color:#7f97ab; font-weight:600; text-transform:uppercase;'>Solver Settings</small>", unsafe_allow_html=True)
time_limit = st.sidebar.slider(
    "Solver time limit (seconds)",
    min_value=5,
    max_value=120,
    value=60,
    key="main_solver_time_limit",
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
if st.sidebar.button("🔄 Reset to Sample Data", use_container_width=True, key="sidebar_reset_btn"):
    reset_to_sample_data()
    st.sidebar.success("Reset to bundled sample data.")
    st.rerun()

st.sidebar.markdown(
    """
    <div class="kw-sidebar-footer">
      <div class="kw-project-badge">
        <span>KW</span>
        <div>
          <b>College Project</b>
          <small>Sample environment</small>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ======================================================================
# ACTIVE DATA & STATE ACCESS
# ======================================================================
df = st.session_state.get("active_jobs_df")
hourly_tariff = st.session_state.get("active_tariff")
jobs_source = st.session_state.get("jobs_source", "None")
tariff_source = st.session_state.get("tariff_source", "None")
schedule_time_unit = st.session_state.get("schedule_time_unit", 60)
upload_error_jobs = st.session_state.get("upload_error_jobs")
upload_error_tariff = st.session_state.get("upload_error_tariff")

machines = get_machines(df) if df is not None and not df.empty else []

current_run_signature = signature_for_run(
    df, hourly_tariff, time_limit, schedule_time_unit, str(jobs_source)
)

has_results = "result" in st.session_state and st.session_state["result"] is not None
is_current = has_results and (st.session_state.get("run_signature") == current_run_signature)
is_outdated = has_results and not is_current


# ======================================================================
# SECTION 1: DASHBOARD
# ======================================================================
if nav_selection == "Dashboard":
    status_label = "Sample Dataset" if "Bundled" in str(jobs_source) else "Ready"
    render_topbar("Operations Dashboard", status_label)

    st.markdown(
        """
        <div class="kw-page-header">
          <h1>Operations Dashboard</h1>
          <p>Key performance indicators, electricity cost summaries, machine allocation, and operational analytics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df is None:
        st.error(f"⚠️ Active jobs dataset is invalid: {upload_error_jobs}. Please visit 'Setup & Data' to repair or reset.")
        if st.button("⚙️ Go to Setup & Data", key="dash_goto_setup_err"):
            navigate_to("Setup & Data")
    elif hourly_tariff is None:
        st.error(f"⚠️ Active electricity tariff is invalid: {upload_error_tariff}. Please visit 'Setup & Data' to repair or reset.")
        if st.button("⚙️ Go to Setup & Data", key="dash_goto_setup_err_t"):
            navigate_to("Setup & Data")
    elif is_outdated:
        st.warning(
            "⚠️ Current inputs (jobs data, tariff, or solver settings) have changed since the last optimization run. "
            "Prior results are outdated. Re-run optimization to update schedules and analytics."
        )
        c_btn1, c_btn2 = st.columns([1, 4])
        with c_btn1:
            if st.button("🚀 Re-run Optimization Now", type="primary", key="dash_rerun_opt_btn"):
                execute_optimization(df, hourly_tariff, time_limit, schedule_time_unit, current_run_signature)
                st.rerun()
        with c_btn2:
            if st.button("⚙️ Review Inputs in Setup & Data", key="dash_goto_setup_outdated"):
                navigate_to("Setup & Data")
    elif not has_results:
        st.markdown(
            """
            <div class="kw-card" style="text-align: center; padding: 36px 20px;">
              <h3 style="color: var(--navy); margin-bottom: 8px;">Optimizer Ready to Run</h3>
              <p style="color: var(--muted); margin-bottom: 18px; font-size: 13px;">
                Production dataset and 24-hour tariff profile are loaded. Run CP-SAT to calculate schedules, cost savings, and machine allocation.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c_btn1, c_btn2, _ = st.columns([2, 2, 4])
        with c_btn1:
            if st.button("🚀 Run Baseline + OR Optimization", type="primary", key="overview_run_opt_btn"):
                execute_optimization(df, hourly_tariff, time_limit, schedule_time_unit, current_run_signature)
                st.rerun()
        with c_btn2:
            if st.button("⚙️ Review Data in Setup & Data", key="dash_goto_setup_btn"):
                navigate_to("Setup & Data")
    else:
        result = st.session_state["result"]
        baseline_schedule = st.session_state["baseline_schedule"]
        baseline_metrics = st.session_state["baseline_metrics"]

        # Validate schedule independently
        _val = validate_schedule(
            schedule=result.schedule,
            jobs_df=df,
            hourly_tariff=hourly_tariff,
            time_unit_minutes=schedule_time_unit,
            reported_cost=result.total_cost,
        )

        if not _val.passed:
            st.error("⚠️ Independent schedule validation failed. Results cannot be presented:")
            for _err in _val.errors:
                st.write(f"  • {_err}")
        else:
            optimized_schedule = result.schedule.copy()
            optimized_metrics = compute_metrics(
                optimized_schedule,
                hourly_tariff,
                time_unit_minutes=schedule_time_unit,
            )
            optimized_display = schedule_as_hours(optimized_schedule, schedule_time_unit)

            baseline_cost = baseline_metrics["electricity_cost"]
            opt_cost = optimized_metrics["electricity_cost"]
            savings_rs = baseline_cost - opt_cost
            savings_pct = (savings_rs / baseline_cost * 100) if baseline_cost else 0.0

            # 6 Executive KPI cards
            st.markdown(
                f"""
                <div class="kw-kpi-grid">
                  <div class="kw-kpi">
                    <div class="kw-kpi-icon navy">⚡</div>
                    <div class="kw-kpi-copy">
                      <p>FCFS Baseline Cost</p>
                      <strong>₹{baseline_cost:,.2f}</strong>
                      <span>Arrival-order plan</span>
                    </div>
                  </div>
                  <div class="kw-kpi">
                    <div class="kw-kpi-icon green">⚡</div>
                    <div class="kw-kpi-copy">
                      <p>CP-SAT Optimized Cost</p>
                      <strong>₹{opt_cost:,.2f}</strong>
                      <span>Tariff-aware plan</span>
                    </div>
                  </div>
                  <div class="kw-kpi">
                    <div class="kw-kpi-icon green">📉</div>
                    <div class="kw-kpi-copy">
                      <p>Estimated Cost Savings</p>
                      <strong>₹{savings_rs:,.2f}</strong>
                      <span>{savings_pct:.1f}% reduction</span>
                    </div>
                  </div>
                  <div class="kw-kpi">
                    <div class="kw-kpi-icon blue">🏭</div>
                    <div class="kw-kpi-copy">
                      <p>Total Estimated Energy</p>
                      <strong>{optimized_metrics['total_energy_kwh']:.1f} kWh</strong>
                      <span>Calculated consumption</span>
                    </div>
                  </div>
                  <div class="kw-kpi">
                    <div class="kw-kpi-icon orange">⏱️</div>
                    <div class="kw-kpi-copy">
                      <p>Production Makespan</p>
                      <strong>{optimized_metrics['makespan']:.1f} hours</strong>
                      <span>Overall schedule span</span>
                    </div>
                  </div>
                  <div class="kw-kpi">
                    <div class="kw-kpi-icon orange">⚠️</div>
                    <div class="kw-kpi-copy">
                      <p>Deadline Violations</p>
                      <strong>{optimized_metrics['deadline_violations']} job(s)</strong>
                      <span>Soft completion targets</span>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col_chart, col_summary = st.columns([3, 2], gap="large")
            with col_chart:
                st.markdown(
                    f"""
                    <div class="kw-card">
                      <div class="kw-section-header">
                        <h3>Electricity Cost Comparison</h3>
                        <p>Arrival-order baseline vs. energy-aware CP-SAT schedule</p>
                      </div>
                      <div style="margin-top: 10px;">
                        <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px;">
                          <span>FCFS Baseline</span>
                          <b>₹{baseline_cost:,.2f}</b>
                        </div>
                        <div style="background: #e2e8f0; height: 18px; border-radius: 9px; overflow: hidden; margin-bottom: 12px;">
                          <div style="background: #64748b; width: 100%; height: 100%;"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px;">
                          <span>CP-SAT Optimized</span>
                          <b style="color: var(--green);">₹{opt_cost:,.2f}</b>
                        </div>
                        <div style="background: #e2e8f0; height: 18px; border-radius: 9px; overflow: hidden;">
                          <div style="background: var(--green); width: {max(5.0, min(100.0, (opt_cost / baseline_cost * 100) if baseline_cost else 100)):.1f}%; height: 100%;"></div>
                        </div>
                        <div style="margin-top: 14px; font-size: 12px; color: var(--green); font-weight: 600;">
                          ✓ Projected Savings: ₹{savings_rs:,.2f} ({savings_pct:.1f}%) without altering physical work requirements.
                        </div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with col_summary:
                status_tone = "kw-badge-success" if result.status == "OPTIMAL" else "kw-badge-info"
                st.markdown(
                    f"""
                    <div class="kw-card">
                      <div class="kw-section-header">
                        <h3>Optimization Summary</h3>
                        <p>CP-SAT solve parameters & validator status</p>
                      </div>
                      <div style="display: grid; gap: 10px; font-size: 12px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #f1f5f9; padding-bottom: 6px;">
                          <span style="color: var(--muted);">Solver Status</span>
                          <span class="kw-badge {status_tone}">{result.status}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #f1f5f9; padding-bottom: 6px;">
                          <span style="color: var(--muted);">Solver Engine</span>
                          <b style="color: var(--navy);">OR-Tools CP-SAT</b>
                        </div>
                        <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #f1f5f9; padding-bottom: 6px;">
                          <span style="color: var(--muted);">Machine Utilization</span>
                          <b style="color: var(--navy);">{optimized_metrics['machine_utilization_pct']:.1f}%</b>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                          <span style="color: var(--muted);">Validation Status</span>
                          <b style="color: var(--green);">Passed Independent Check ✓</b>
                        </div>
                      </div>
                      <div style="background: #f8fafc; padding: 10px; border-radius: 7px; font-size: 10px; color: var(--muted); line-height: 1.4;">
                        ℹ️ Independent validator verified non-overlap, precedence, release times, and unit integrity.
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Timeline
            st.markdown(
                """
                <div class="kw-card">
                  <div class="kw-section-header">
                    <h3>Optimized Production Timeline</h3>
                    <p>Machine allocation across the planning horizon</p>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                make_gantt_chart(optimized_display, "OR-Optimized Production Schedule"),
                use_container_width=True,
            )

            # Integrated Operational Analytics
            st.markdown(
                """
                <div class="kw-card">
                  <div class="kw-section-header">
                    <h3>Operational Analytics & Energy Allocation</h3>
                    <p>Workload distribution across machines and high-cost job concentration</p>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            insights = optimized_schedule.copy()
            insights["processing_hours"] = insights["processing_time"] * schedule_time_unit / 60.0
            insights["estimated_energy_kwh"] = insights["power_kw"] * insights["processing_hours"]
            insights["estimated_cost_rs"] = [
                schedule_cost(optimized_schedule.iloc[[idx]], hourly_tariff, time_unit_minutes=schedule_time_unit)
                for idx in range(len(optimized_schedule))
            ]

            machine_load = insights.groupby("machine")["processing_hours"].sum().sort_values(ascending=False)
            top_cost_row = insights.loc[insights["estimated_cost_rs"].idxmax()]

            col_ana1, col_ana2 = st.columns(2, gap="medium")
            with col_ana1:
                st.metric("Busiest Machine", str(machine_load.idxmax()), help="Highest scheduled processing hours")
                st.subheader("Machine Workload (scheduled hours)")
                st.bar_chart(machine_load)

            with col_ana2:
                st.metric(
                    "Highest Single Job Cost",
                    f"₹{top_cost_row['estimated_cost_rs']:,.2f}",
                    help=f"Job ID: {top_cost_row['job_id']}",
                )
                st.subheader("Top Jobs by Electricity Cost")
                top_jobs = insights.sort_values("estimated_cost_rs", ascending=False).head(5)
                st.dataframe(
                    top_jobs[
                        ["job_id", "machine", "processing_hours", "estimated_energy_kwh", "estimated_cost_rs", "deadline_violated"]
                    ].rename(columns={
                        "job_id": "Job ID",
                        "machine": "Machine",
                        "processing_hours": "Duration (h)",
                        "estimated_energy_kwh": "Energy (kWh)",
                        "estimated_cost_rs": "Cost (₹)",
                        "deadline_violated": "Late?",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)
            c_nav1, c_nav2 = st.columns([1, 1])
            with c_nav1:
                if st.button("Inspect Full Comparison in Optimize & Compare ➜", key="dash_to_compare_btn", use_container_width=True):
                    navigate_to("Optimize & Compare")
            with c_nav2:
                if st.button("Simulate Disruptions in What-If Analysis ➜", key="dash_to_whatif_btn", use_container_width=True):
                    navigate_to("What-If Analysis")


# ======================================================================
# SECTION 2: SETUP & DATA
# ======================================================================
elif nav_selection == "Setup & Data":
    render_topbar("Setup & Data", f"{len(df) if df is not None else 0} Jobs Active")

    st.markdown(
        """
        <div class="kw-page-header">
          <h1>Setup & Data Management</h1>
          <p>Configure jobs, machine capabilities, electricity tariffs, and scheduling assumptions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Active configuration status card
    c_status1, c_status2, c_reset = st.columns([3, 3, 2])
    with c_status1:
        st.markdown(f"**Active Jobs Source**: `{jobs_source}`")
        if df is not None:
            st.caption(f"{len(df)} jobs across {len(machines)} machine(s). Time unit: {'Hours' if schedule_time_unit == 60 else 'Minutes'}.")
    with c_status2:
        st.markdown(f"**Active Tariff Source**: `{tariff_source}`")
        if hourly_tariff is not None:
            avg_rate = sum(hourly_tariff) / len(hourly_tariff)
            st.caption(f"24-hour tariff loaded. Average rate: ₹{avg_rate:.2f}/kWh.")
    with c_reset:
        if st.button("🔄 Reset to Sample Data", key="setup_reset_sample_btn", use_container_width=True):
            reset_to_sample_data()
            st.success("Reset to bundled sample data.")
            st.rerun()

    job_col, tariff_col = st.columns(2, gap="large")

    with job_col:
        with st.container(border=True):
            st.markdown(
                """
                <div class="kw-section-header">
                  <h3>📦 Production Jobs</h3>
                  <p>Load jobs from standard CSV or external manufacturing records.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            job_tab1, job_tab2 = st.tabs(["Standard Jobs CSV", "External Manufacturing (Kaggle/MES)"])

            with job_tab1:
                uploaded_jobs = st.file_uploader(
                    "Upload jobs CSV",
                    type=["csv"],
                    key="setup_standard_jobs_upload",
                    help="Columns: job_id, processing_time, power_kw, deadline, machine. Optional: release_time, predecessor.",
                )
                if uploaded_jobs is not None:
                    upload_key = f"{uploaded_jobs.name}_{uploaded_jobs.size}"
                    if upload_key != st.session_state.get("_last_jobs_upload_key"):
                        try:
                            uploaded_jobs.seek(0)
                            loaded_df = load_jobs(uploaded_jobs)
                            st.session_state["active_jobs_df"] = loaded_df
                            st.session_state["jobs_source"] = f"Uploaded: {uploaded_jobs.name}"
                            st.session_state["schedule_time_unit"] = 60
                            st.session_state["upload_error_jobs"] = None
                            st.session_state["_last_jobs_upload_key"] = upload_key
                            st.session_state.pop("result", None)
                            st.session_state.pop("what_if_result", None)
                            st.rerun()
                        except Exception as exc:
                            st.session_state["active_jobs_df"] = None
                            st.session_state["jobs_source"] = f"Invalid: {uploaded_jobs.name}"
                            st.session_state["upload_error_jobs"] = str(exc)
                            st.session_state["_last_jobs_upload_key"] = upload_key
                            st.session_state.pop("result", None)
                            st.session_state.pop("what_if_result", None)
                            st.rerun()

                if upload_error_jobs and "Standard" not in str(upload_error_jobs):
                    st.error(f"Error loading jobs CSV: {upload_error_jobs}")

            with job_tab2:
                mfg_file = st.file_uploader(
                    "Upload external manufacturing CSV",
                    type=["csv"],
                    key="setup_mfg_jobs_upload",
                )
                if mfg_file is not None:
                    mfg_key = f"{mfg_file.name}_{mfg_file.size}"
                    if mfg_key != st.session_state.get("_last_mfg_upload_key"):
                        try:
                            mfg_file.seek(0)
                            raw_records = load_manufacturing_records(mfg_file)
                            st.session_state["raw_manufacturing_df"] = raw_records
                            st.session_state["_last_mfg_upload_key"] = mfg_key
                        except Exception as exc:
                            st.session_state["raw_manufacturing_df"] = None
                            st.error(f"Could not load manufacturing records: {exc}")

                raw_mfg = st.session_state.get("raw_manufacturing_df")
                if raw_mfg is not None and not raw_mfg.empty:
                    with st.expander("Inspect Raw Manufacturing Records", expanded=False):
                        m_col = "machine_id" if "machine_id" in raw_mfg.columns else ("machine_code" if "machine_code" in raw_mfg.columns else raw_mfg.columns[1])
                        machine_counts = raw_mfg[m_col].astype(str).value_counts()
                        st.bar_chart(machine_counts)
                        st.dataframe(raw_mfg.head(50), use_container_width=True, hide_index=True)

                    max_jobs = st.slider(
                        "Records to schedule (prototype limit: 30)",
                        min_value=1,
                        max_value=min(30, len(raw_mfg)),
                        value=min(10, len(raw_mfg)),
                        key="setup_mfg_slider_max_jobs",
                    )
                    units_confirmed = st.checkbox(
                        "I accept the assumption that Processing_Time is in minutes and Energy_Consumption is kWh per job.",
                        value=False,
                        key="setup_mfg_units_confirmed",
                    )
                    if units_confirmed:
                        try:
                            prep_df = prepare_manufacturing_jobs(
                                raw_mfg,
                                max_jobs=max_jobs,
                                assume_energy_is_kwh_per_job=True,
                            )
                            st.session_state["active_jobs_df"] = prep_df
                            st.session_state["jobs_source"] = f"External manufacturing data: {mfg_file.name if mfg_file else 'uploaded'}"
                            st.session_state["schedule_time_unit"] = 1
                            st.session_state["upload_error_jobs"] = None
                            st.success(f"✓ Prepared {len(prep_df)} jobs for minute-resolution scheduling.")
                            st.download_button(
                                "Download prepared scheduling data",
                                data=prep_df.to_csv(index=False).encode("utf-8"),
                                file_name="prepared_manufacturing_jobs.csv",
                                mime="text/csv",
                                key="download_prep_jobs_csv",
                            )
                        except Exception as exc:
                            st.error(f"Error preparing manufacturing jobs: {exc}")
                    else:
                        st.info("Check the confirmation box above to convert and activate this dataset.")

    with tariff_col:
        with st.container(border=True):
            st.markdown(
                """
                <div class="kw-section-header">
                  <h3>⚡ Electricity Tariff</h3>
                  <p>Hourly rates (₹/kWh) influencing scheduling decisions.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            uploaded_tariff = st.file_uploader(
                "Upload tariff CSV",
                type=["csv"],
                key="setup_tariff_upload",
                help="Requires columns: hour and tariff_rs_per_kwh (24 rows).",
            )
            if uploaded_tariff is not None:
                t_key = f"{uploaded_tariff.name}_{uploaded_tariff.size}"
                if t_key != st.session_state.get("_last_tariff_upload_key"):
                    try:
                        uploaded_tariff.seek(0)
                        loaded_t = load_tariff(uploaded_tariff)
                        st.session_state["active_tariff"] = loaded_t
                        st.session_state["tariff_source"] = f"Uploaded: {uploaded_tariff.name}"
                        st.session_state["upload_error_tariff"] = None
                        st.session_state["_last_tariff_upload_key"] = t_key
                        st.session_state.pop("result", None)
                        st.session_state.pop("what_if_result", None)
                        st.rerun()
                    except Exception as exc:
                        st.session_state["active_tariff"] = None
                        st.session_state["tariff_source"] = f"Invalid: {uploaded_tariff.name}"
                        st.session_state["upload_error_tariff"] = str(exc)
                        st.session_state["_last_tariff_upload_key"] = t_key
                        st.session_state.pop("result", None)
                        st.session_state.pop("what_if_result", None)
                        st.rerun()

            if upload_error_tariff:
                st.error(f"Invalid tariff CSV: {upload_error_tariff}")
            elif hourly_tariff is not None:
                st.plotly_chart(make_tariff_chart(hourly_tariff), use_container_width=True)

    # Input review table
    st.markdown(
        """
        <div class="kw-section-header" style="margin-top: 18px;">
          <h2>Review Active Scheduling Inputs</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df is not None:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"{len(df)} jobs across {len(machines)} machine(s): {', '.join(machines)}")
        with c2:
            st.markdown("**Time units & Assumptions**")
            if schedule_time_unit == 60:
                st.info("Standard dataset: all processing times and deadlines are in hours.")
            else:
                st.info(
                    "External dataset: the optimizer uses minutes internally. "
                    "Results are displayed in hours."
                )
                st.warning(
                    "External-data costs depend on the selected energy assumption, "
                    "reference completion targets and tariff. Costs are estimates."
                )

            # Tariff Table preview
            if hourly_tariff is not None:
                st.markdown("**24-Hour Tariff Breakdown**")
                t_df = pd.DataFrame({
                    "Hour": [f"{h:02d}:00" for h in range(24)],
                    "Rate (₹/kWh)": [f"₹{r:.2f}" for r in hourly_tariff],
                    "Category": [
                        "Peak" if r >= 12.0 else ("Low" if r <= 7.0 else "Standard")
                        for r in hourly_tariff
                    ],
                })
                st.dataframe(t_df, use_container_width=True, hide_index=True)
    else:
        st.error(f"Cannot display inputs: {upload_error_jobs}")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Next: Optimize & Compare ➜", type="primary", key="setup_to_compare_btn"):
        navigate_to("Optimize & Compare")


# ======================================================================
# SECTION 3: OPTIMIZE & COMPARE
# ======================================================================
elif nav_selection == "Optimize & Compare":
    render_topbar("Optimize & Compare", "Baseline vs CP-SAT")

    st.markdown(
        """
        <div class="kw-page-header">
          <h1>Optimize & Compare Production Schedules</h1>
          <p>Solve production schedules using Google OR-Tools CP-SAT and compare against the FCFS baseline.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 3])
    with btn_col1:
        run_btn = st.button("🚀 Run Baseline + OR Optimization", type="primary", key="sched_page_run_btn")
    with btn_col2:
        strategy_label = st.selectbox(
            "Scheduling Strategy",
            ["Cost-Focused", "Balanced (Cost + Time)", "Time-Focused"],
            index=0,
            key="sched_page_strategy_select",
            help="Cost-Focused: pure tariff-driven cost minimization. Balanced: saves electricity while controlling tardiness and makespan. Time-Focused: compact schedule with lowest makespan.",
        )
        strategy_map = {
            "Cost-Focused": "cost",
            "Balanced (Cost + Time)": "balanced",
            "Time-Focused": "time",
        }
        chosen_strategy = strategy_map[strategy_label]
    with btn_col3:
        st.caption(
            f"Active: {jobs_source} | Strategy: {strategy_label} | "
            f"Jobs: {len(df) if df is not None else 0} | Machines: {len(machines)}"
        )

    if run_btn:
        if df is None:
            st.error(f"Cannot run optimization: jobs dataset is invalid ({upload_error_jobs}). Please review Setup & Data.")
        elif hourly_tariff is None:
            st.error(f"Cannot run optimization: tariff is invalid ({upload_error_tariff}). Please review Setup & Data.")
        else:
            execute_optimization(df, hourly_tariff, time_limit, schedule_time_unit, current_run_signature, strategy=chosen_strategy)
            st.rerun()

    if is_outdated:
        st.warning(
            "⚠️ Current inputs (dataset, tariff, or solver settings) have changed since the last optimization. "
            "Prior results are outdated. Click 'Run Baseline + OR Optimization' above to update."
        )
    elif not has_results:
        st.info("Click 'Run Baseline + OR Optimization' above to generate schedules and view comparisons.")
    else:
        result = st.session_state["result"]
        baseline_schedule = st.session_state["baseline_schedule"]
        baseline_metrics = st.session_state["baseline_metrics"]

        if result.status not in ("OPTIMAL", "FEASIBLE") or result.schedule.empty:
            st.error(f"Solver returned status: {result.status}. No feasible schedule could be found.")
        else:
            _val = validate_schedule(
                schedule=result.schedule,
                jobs_df=df,
                hourly_tariff=hourly_tariff,
                time_unit_minutes=schedule_time_unit,
                reported_cost=result.total_cost,
            )

            if not _val.passed:
                st.error("⚠️ Independent schedule validation failed. Results cannot be presented:")
                for _err in _val.errors:
                    st.write(f"  • {_err}")
            else:
                if result.used_soft_deadlines:
                    st.warning(f"The optimizer used soft deadlines. {result.deadline_violations} job(s) finish after deadline.")
                if result.status == "OPTIMAL":
                    gap_info = " (Proven optimal, 0.0% gap)" if result.optimality_gap is not None else ""
                    st.success(f"✓ Optimal schedule found in {result.solver_wall_time:.2f}s and verified by independent validator{gap_info}.")
                else:
                    gap_str = f" (Optimality gap: {result.optimality_gap*100:.1f}%)" if result.optimality_gap is not None else ""
                    st.info(f"✓ Feasible schedule found within {result.solver_wall_time:.2f}s and verified by independent validator{gap_str}.")


                optimized_schedule = result.schedule.copy()
                optimized_metrics = compute_metrics(optimized_schedule, hourly_tariff, time_unit_minutes=schedule_time_unit)
                optimized_display = schedule_as_hours(optimized_schedule, schedule_time_unit)
                baseline_display = schedule_as_hours(baseline_schedule, schedule_time_unit)

                b_cost = baseline_metrics["electricity_cost"]
                o_cost = optimized_metrics["electricity_cost"]
                savings_val = b_cost - o_cost
                savings_pct = (savings_val / b_cost * 100) if b_cost else 0.0

                # Comparison Cards Head
                st.markdown(
                    f"""
                    <div class="kw-comparison-head">
                      <div class="kw-card-baseline">
                        <span class="kw-badge kw-badge-neutral">Baseline · FCFS</span>
                        <strong style="font-size: 22px; color: var(--navy); margin: 6px 0;">₹{b_cost:,.2f}</strong>
                        <span style="color: var(--muted); font-size: 11px;">Arrival-order electricity cost</span>
                        <div style="font-size: 10px; color: var(--muted); margin-top: 6px;">
                          {baseline_metrics['makespan']:.1f}h makespan · {baseline_metrics['deadline_violations']} violations
                        </div>
                      </div>
                      <div class="kw-versus">VS</div>
                      <div class="kw-card-optimized">
                        <span class="kw-badge kw-badge-success">Optimized · CP-SAT</span>
                        <strong style="font-size: 22px; color: var(--navy); margin: 6px 0;">₹{o_cost:,.2f}</strong>
                        <span style="color: var(--muted); font-size: 11px;">Tariff-aware electricity cost</span>
                        <div style="font-size: 10px; color: var(--muted); margin-top: 6px;">
                          {optimized_metrics['makespan']:.1f}h makespan · {optimized_metrics['deadline_violations']} violations
                        </div>
                      </div>
                      <div class="kw-card-saving">
                        <span style="font-size: 11px; color: #a1d9c4;">Estimated Savings</span>
                        <strong>{savings_pct:.1f}%</strong>
                        <span style="font-size: 11px; color: #cbeade;">₹{savings_val:,.2f} cost reduction</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Metric Strip
                st.markdown(
                    f"""
                    <div class="kw-metric-strip">
                      <div>
                        <span>Total Energy</span>
                        <b>{optimized_metrics['total_energy_kwh']:.1f} kWh</b>
                        <small>Physical work conserved</small>
                      </div>
                      <div>
                        <span>Makespan Difference</span>
                        <b>{optimized_metrics['makespan'] - baseline_metrics['makespan']:+.1f} hours</b>
                        <small>Optimized vs Baseline</small>
                      </div>
                      <div>
                        <span>Deadline Violations</span>
                        <b>{baseline_metrics['deadline_violations']} → {optimized_metrics['deadline_violations']}</b>
                        <small>Soft completion targets</small>
                      </div>
                      <div>
                        <span>Machine Utilization</span>
                        <b>{optimized_metrics['machine_utilization_pct']:.1f}%</b>
                        <small>Average scheduled load</small>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Detailed Metrics Comparison Table
                st.markdown(
                    """
                    <div class="kw-section-header">
                      <h3>Detailed Metrics Comparison Table</h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.dataframe(comparison_table(baseline_metrics, optimized_metrics), use_container_width=True, hide_index=True)

                # Interactive Gantt Charts
                st.markdown(
                    """
                    <div class="kw-section-header" style="margin-top: 20px;">
                      <h3>Production Timeline Comparison</h3>
                      <p>Inspect machine allocations between the FCFS baseline and CP-SAT optimized schedule.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                gantt_tab1, gantt_tab2 = st.tabs(["Optimized Schedule (CP-SAT)", "Baseline Schedule (FCFS)"])
                with gantt_tab1:
                    st.plotly_chart(make_gantt_chart(optimized_display, "CP-SAT Optimized Schedule"), use_container_width=True)
                with gantt_tab2:
                    st.plotly_chart(make_gantt_chart(baseline_display, "FCFS Baseline Schedule"), use_container_width=True)

                # Schedule Table & Download
                st.markdown(
                    """
                    <div class="kw-section-header" style="margin-top: 20px;">
                      <h3>Scheduled Jobs Table</h3>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                disp_table = display_schedule_table(optimized_display)
                st.dataframe(disp_table, use_container_width=True, hide_index=True)

                st.download_button(
                    "📥 Download optimized schedule (CSV)",
                    data=disp_table.to_csv(index=False).encode("utf-8"),
                    file_name="karkhanawatt_optimized_schedule_hours.csv",
                    mime="text/csv",
                    key="download_sched_page_csv",
                )

    st.markdown("<br>", unsafe_allow_html=True)
    c_back, c_next = st.columns([1, 1])
    with c_back:
        if st.button("⬅ Back to Setup & Data", key="compare_back_to_setup"):
            navigate_to("Setup & Data")
    with c_next:
        if st.button("Next: What-If Analysis ➜", type="primary", key="compare_to_whatif_btn"):
            navigate_to("What-If Analysis")


# ======================================================================
# SECTION 4: WHAT-IF ANALYSIS
# ======================================================================
elif nav_selection == "What-If Analysis":
    render_topbar("What-If Analysis", "Disruption Simulator")

    st.markdown(
        """
        <div class="kw-page-header">
          <h1>Smart Factory What-If Analysis</h1>
          <p>Simulate unexpected operational disruptions (machine breakdowns or rush orders) and evaluate revised plans against the current reference schedule.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not has_results:
        st.info("Run baseline optimization in 'Optimize & Compare' or 'Dashboard' first before simulating what-if scenarios.")
        if st.button("Go to Optimize & Compare ➜", key="whatif_goto_opt_first"):
            navigate_to("Optimize & Compare")
    elif is_outdated:
        st.warning("⚠️ The reference schedule is outdated because active inputs have changed. Please re-run optimization in 'Optimize & Compare' before simulating scenarios.")
        if st.button("Go to Optimize & Compare ➜", key="whatif_goto_opt_outdated"):
            navigate_to("Optimize & Compare")
    else:
        result = st.session_state["result"]
        optimized_schedule = result.schedule.copy()
        optimized_metrics = compute_metrics(optimized_schedule, hourly_tariff, time_unit_minutes=schedule_time_unit)
        optimized_display = schedule_as_hours(optimized_schedule, schedule_time_unit)

        st.caption(
            f"Reference Plan: CP-SAT schedule with ₹{optimized_metrics['electricity_cost']:,.2f} electricity cost, "
            f"{optimized_metrics['makespan']:.1f}h makespan ({len(df)} jobs on {len(machines)} machines)."
        )

        units_per_hour = 60 // schedule_time_unit
        max_model_hour = max(
            24,
            int(math.ceil(
                float(df["deadline"].max()) / units_per_hour
                + float(df["processing_time"].sum()) / units_per_hour
            )),
        )
        hour_step = 1.0 if schedule_time_unit == 60 else 0.5
        min_processing_hours = 1.0 if schedule_time_unit == 60 else 0.25

        scenario_type = st.radio(
            "Choose scenario type",
            ["Machine breakdown", "Add urgent job"],
            horizontal=True,
            key="what_if_scenario_type",
        )

        with st.form("what_if_scenario_form"):
            scenario_details = {}
            if scenario_type == "Machine breakdown":
                scenario_machine = st.selectbox("Machine experiencing breakdown", machines)
                breakdown_start_hours = st.number_input(
                    "Breakdown starts at hour",
                    min_value=0.0,
                    max_value=float(max_model_hour),
                    value=min(10.0, float(max_model_hour)),
                    step=hour_step,
                )
                breakdown_duration_hours = st.number_input(
                    "Breakdown duration (hours)",
                    min_value=hour_step,
                    max_value=float(max_model_hour),
                    value=max(hour_step, min(4.0, float(max_model_hour))),
                    step=hour_step,
                )
                scenario_details = {
                    "machine": scenario_machine,
                    "start": breakdown_start_hours,
                    "end": breakdown_start_hours + breakdown_duration_hours,
                    "duration": breakdown_duration_hours,
                }
            else:
                uj_id = st.text_input("Urgent Job ID", value="JOB_URGENT")
                scenario_machine = st.selectbox("Machine for the urgent job", machines)
                uj_processing_hours = st.number_input(
                    "Processing duration (hours)",
                    min_value=min_processing_hours,
                    max_value=float(max_model_hour),
                    value=min(2.0, float(max_model_hour)),
                    step=hour_step,
                )
                uj_power = st.number_input(
                    "Power demand (kW)",
                    min_value=1.0,
                    max_value=200.0,
                    value=15.0,
                    step=1.0,
                )
                uj_deadline_hours = st.number_input(
                    "Deadline (hour)",
                    min_value=hour_step,
                    max_value=float(max_model_hour * 2),
                    value=min(12.0, float(max_model_hour)),
                    step=hour_step,
                )
                uj_release_hours = st.number_input(
                    "Release time (hour)",
                    min_value=0.0,
                    max_value=float(max_model_hour),
                    value=0.0,
                    step=hour_step,
                )
                predecessor_choices = ["None"] + sorted(list(df["job_id"].astype(str).unique()))
                scenario_predecessor = st.selectbox("Predecessor job (optional)", predecessor_choices)

                job_dict = {
                    "job_id": uj_id,
                    "machine": scenario_machine,
                    "processing_time": uj_processing_hours,
                    "power_kw": uj_power,
                    "deadline": uj_deadline_hours,
                    "release_time": uj_release_hours,
                    "predecessor": None if scenario_predecessor == "None" else scenario_predecessor,
                }
                scenario_details = {
                    "job": job_dict,
                    **job_dict,
                }

            simulate_scenario = st.form_submit_button("Simulate and re-optimize", type="primary")

        if simulate_scenario:
            downtime_blocks = {}
            augmented_df = df.copy()

            if scenario_type == "Machine breakdown":
                st_b = int(round(scenario_details["start"] * units_per_hour))
                dur_b = int(round(scenario_details["duration"] * units_per_hour))
                downtime_blocks = {scenario_details["machine"]: [(st_b, st_b + dur_b)]}
            else:
                uj_row = {
                    "job_id": str(scenario_details["job_id"]).strip(),
                    "processing_time": int(round(scenario_details["processing_time"] * units_per_hour)),
                    "power_kw": float(scenario_details["power_kw"]),
                    "deadline": int(round(scenario_details["deadline"] * units_per_hour)),
                    "machine": str(scenario_details["machine"]),
                    "release_time": int(round(scenario_details["release_time"] * units_per_hour)),
                    "predecessor": scenario_details["predecessor"],
                }
                augmented_df = pd.concat([augmented_df, pd.DataFrame([uj_row])], ignore_index=True)

            with st.spinner("Re-optimizing schedule for scenario..."):
                s_result = solve_schedule(
                    augmented_df,
                    hourly_tariff,
                    machine_downtime=downtime_blocks,
                    time_limit_seconds=time_limit,
                    time_unit_minutes=schedule_time_unit,
                )

            st.session_state["what_if_result"] = {
                "type": scenario_type,
                "result": s_result,
                "details": scenario_details,
                "downtime": downtime_blocks,
                "base_run": current_run_signature,
                "augmented_df": augmented_df,
            }

        saved_scenario = st.session_state.get("what_if_result")
        if saved_scenario and saved_scenario.get("base_run") == current_run_signature:
            s_res = saved_scenario["result"]
            aug_df = saved_scenario.get("augmented_df", df)

            if s_res.status not in ("OPTIMAL", "FEASIBLE") or s_res.schedule.empty:
                st.error(f"Scenario optimization returned status: {s_res.status}. No feasible revised schedule was found.")
            else:
                _val = validate_schedule(
                    schedule=s_res.schedule,
                    jobs_df=aug_df,
                    hourly_tariff=hourly_tariff,
                    time_unit_minutes=schedule_time_unit,
                    reported_cost=s_res.total_cost,
                    machine_downtime=saved_scenario.get("downtime"),
                )

                if not _val.passed:
                    st.error("⚠️ Revised schedule failed independent validation:")
                    for _err in _val.errors:
                        st.write(f"  • {_err}")
                else:
                    s_schedule = s_res.schedule.copy()
                    s_metrics = compute_metrics(s_schedule, hourly_tariff, time_unit_minutes=schedule_time_unit)

                    st.markdown("### Revised Scenario Impact")
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric(
                        "Electricity Cost",
                        f"₹{s_metrics['electricity_cost']:,.2f}",
                        delta=f"₹{s_metrics['electricity_cost'] - optimized_metrics['electricity_cost']:+,.2f}",
                        delta_color="inverse",
                    )
                    k2.metric(
                        "Total Energy",
                        f"{s_metrics['total_energy_kwh']:.1f} kWh",
                        delta=f"{s_metrics['total_energy_kwh'] - optimized_metrics['total_energy_kwh']:+.1f} kWh",
                        delta_color="inverse",
                    )
                    k3.metric(
                        "Makespan",
                        f"{s_metrics['makespan']:.1f} hrs",
                        delta=f"{s_metrics['makespan'] - optimized_metrics['makespan']:+.1f} hrs",
                        delta_color="inverse",
                    )
                    k4.metric(
                        "Machine Utilization",
                        f"{s_metrics['machine_utilization_pct']:.1f}%",
                        delta=f"{s_metrics['machine_utilization_pct'] - optimized_metrics['machine_utilization_pct']:+.1f} pp",
                    )

                    st.subheader("Why did the schedule change?")
                    for exp in build_explanation(
                        saved_scenario["type"],
                        optimized_schedule,
                        s_schedule,
                        saved_scenario["details"],
                    ):
                        st.write(f"• {exp}")

                    changes_df = changed_jobs(optimized_schedule, s_schedule)
                    if not changes_df.empty and schedule_time_unit != 60:
                        for c in ("original_start", "revised_start", "original_end", "revised_end"):
                            if c in changes_df.columns:
                                changes_df[c] = (changes_df[c] / units_per_hour).round(3)

                    st.subheader("Jobs that shifted")
                    if changes_df.empty:
                        st.info("No existing job's recorded timing changed.")
                    else:
                        st.dataframe(changes_df, use_container_width=True, hide_index=True)

                    st.subheader("Revised vs Original Plan Gantt Comparison")
                    otab, rtab = st.tabs(["Original Plan", "Revised Scenario Plan"])
                    with otab:
                        st.plotly_chart(make_gantt_chart(optimized_display, "Original Optimized Plan"), use_container_width=True)
                    with rtab:
                        s_display = schedule_as_hours(s_schedule, schedule_time_unit)
                        dt_for_chart = {
                            m: [(st_b / units_per_hour, end_b / units_per_hour) for st_b, end_b in blocks]
                            for m, blocks in saved_scenario["downtime"].items()
                        }
                        st.plotly_chart(make_gantt_chart(s_display, "Revised Scenario Plan", downtime=dt_for_chart), use_container_width=True)

                    st.download_button(
                        "📥 Download revised scenario schedule (CSV)",
                        data=display_schedule_table(schedule_as_hours(s_schedule, schedule_time_unit)).to_csv(index=False).encode("utf-8"),
                        file_name="karkhanawatt_what_if_schedule_hours.csv",
                        mime="text/csv",
                        key="download_whatif_tab_csv",
                    )

    st.markdown("<br>", unsafe_allow_html=True)
    c_back, c_home = st.columns([1, 1])
    with c_back:
        if st.button("⬅ Back to Optimize & Compare", key="whatif_back_to_compare"):
            navigate_to("Optimize & Compare")
    with c_home:
        if st.button("Go to Dashboard 📊", key="whatif_to_dashboard"):
            navigate_to("Dashboard")
