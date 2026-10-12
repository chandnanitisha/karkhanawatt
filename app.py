"""KarkhanaWatt: energy-aware production scheduling with OR-Tools CP-SAT.

Run from the project folder with:
    python -m streamlit run app.py

The app has one integrated Production Scheduler with two data sources:
1. Standard KarkhanaWatt jobs CSV (time unit: hours).
2. Compatible external manufacturing CSV (prepared from the supported
   manufacturing schema; time unit: minutes).

External-dataset energy costs are estimates when energy units or deadlines
are inferred. The app labels those assumptions rather than presenting them
as verified factory electricity costs.
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
    validate_jobs,
    load_manufacturing_records,
    prepare_manufacturing_jobs,
)
from src.baseline import run_fcfs
from src.optimizer import solve_schedule
from src.metrics import compute_metrics, comparison_table
from src.visualization import make_gantt_chart, make_tariff_chart
from src.tariff import load_tariff, schedule_cost
from src.what_if import changed_jobs, build_explanation


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def schedule_as_hours(schedule: pd.DataFrame, time_unit_minutes: int) -> pd.DataFrame:
    """Return a copy with all schedule time fields represented in hours."""
    result = schedule.copy()
    units_per_hour = 60 / time_unit_minutes
    for column in (
        "start_time",
        "end_time",
        "processing_time",
        "deadline",
        "release_time",
    ):
        if column in result.columns:
            result[column] = (result[column] / units_per_hour).round(3)
    return result


def display_schedule_table(schedule_hours: pd.DataFrame) -> pd.DataFrame:
    """Rename time columns so the displayed units are explicit."""
    return schedule_hours.rename(
        columns={
            "start_time": "start_time_hours",
            "end_time": "end_time_hours",
            "processing_time": "processing_time_hours",
            "deadline": "reference_deadline_hours",
            "release_time": "release_time_hours",
        }
    )


def signature_for_run(df: pd.DataFrame, tariff: list, time_limit: int,
                      time_unit_minutes: int, source: str) -> str:
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


st.set_page_config(page_title="KarkhanaWatt", layout="wide")
st.title("⚙️ KarkhanaWatt — Production Scheduler")
st.caption(
    "Energy-aware production scheduling using OR-Tools CP-SAT. "
    "Choose a supported data source, validate its scheduling inputs, "
    "and compare the FCFS baseline with an optimized schedule."
)

# ----------------------------------------------------------------------
# 1. Select data source and tariff
# ----------------------------------------------------------------------
st.header("1. Upload Production Data and Electricity Tariff")

job_col, tariff_col = st.columns(2, gap="large")

df = None
jobs_source = None
schedule_time_unit = 60  # minutes per model step; default is whole hours
raw_manufacturing_df = None

with job_col:
    with st.container(border=True):
        st.subheader("🏭 Production Data")
        data_source = st.radio(
            "Choose data source",
            ["Standard Jobs CSV", "External Manufacturing CSV"],
            key="scheduler_data_source",
            help=(
                "Both sources use the same OR-Tools optimizer. External data "
                "must contain the supported manufacturing fields and valid units."
            ),
        )

        if data_source == "Standard Jobs CSV":
            uploaded_jobs = st.file_uploader(
                "Choose jobs CSV",
                type=["csv"],
                key="jobs_csv_upload",
                help=(
                    "Required columns: job_id, machine, processing_time, "
                    "deadline, power_kw, arrival_order. predecessor is optional. "
                    "Time values are whole hours."
                ),
            )
            try:
                if uploaded_jobs is not None:
                    df = load_jobs(uploaded_jobs)
                    jobs_source = f"Uploaded: {uploaded_jobs.name}"
                else:
                    df = load_jobs(DATA_DIR / "jobs.csv")
                    jobs_source = "Bundled sample: data/jobs.csv"
                schedule_time_unit = 60
                st.success(f"✓ {jobs_source}")
                st.caption(f"{len(df)} jobs validated. Time unit: hours.")
            except (ValueError, KeyError, TypeError, OSError) as exc:
                st.error(f"Invalid jobs CSV: {exc}")
                df = None

        else:
            manufacturing_file = st.file_uploader(
                "Choose manufacturing CSV",
                type=["csv"],
                key="scheduler_manufacturing_upload",
                help=(
                    "Upload a CSV matching the supported manufacturing schema "
                    "(for example, the provided Kaggle manufacturing file), not a ZIP."
                ),
            )

            if manufacturing_file is None:
                st.info("Upload a compatible manufacturing CSV to continue.")
            else:
                try:
                    raw_manufacturing_df = load_manufacturing_records(
                        manufacturing_file
                    )
                    st.success(
                        f"✓ Loaded {len(raw_manufacturing_df):,} manufacturing records."
                    )

                    with st.expander("Inspect source manufacturing data", expanded=False):
                        info1, info2, info3 = st.columns(3)
                        info1.metric("Records", f"{len(raw_manufacturing_df):,}")
                        info2.metric(
                            "Machines",
                            f"{raw_manufacturing_df['machine_id'].nunique():,}",
                        )
                        info3.metric(
                            "Average processing-time value",
                            f"{raw_manufacturing_df['processing_time_raw'].mean():.2f}",
                        )
                        machine_counts = (
                            raw_manufacturing_df["machine_id"]
                            .value_counts()
                            .sort_index()
                        )
                        st.bar_chart(machine_counts)
                        if "job_status" in raw_manufacturing_df.columns:
                            status_counts = (
                                raw_manufacturing_df["job_status"]
                                .fillna("Missing")
                                .astype(str)
                                .value_counts()
                            )
                            st.bar_chart(status_counts)
                        st.dataframe(
                            raw_manufacturing_df.head(100),
                            use_container_width=True,
                            hide_index=True,
                        )

                    max_jobs = st.slider(
                        "Records to schedule (prototype limit: 30)",
                        min_value=1,
                        max_value=min(30, len(raw_manufacturing_df)),
                        value=min(10, len(raw_manufacturing_df)),
                        key="scheduler_manufacturing_max_jobs",
                    )
                    units_confirmed = st.checkbox(
                        "I accept the assumption that Processing_Time is in minutes "
                        "and Energy_Consumption is kWh per job.",
                        value=False,
                        key="scheduler_manufacturing_units_confirmed",
                        help=(
                            "Use only if this interpretation is appropriate for your dataset. "
                            "The dataset ZIP does not verify these units."
                        ),
                    )
                    st.caption(
                        "Scheduled_End is treated as a reference completion target, not a "
                        "verified contractual deadline. Average power is estimated from the "
                        "energy assumption."
                    )

                    if units_confirmed:
                        df = prepare_manufacturing_jobs(
                            raw_manufacturing_df,
                            max_jobs=max_jobs,
                            assume_energy_is_kwh_per_job=True,
                        )
                        jobs_source = f"External manufacturing data: {manufacturing_file.name}"
                        schedule_time_unit = 1
                        st.success(
                            f"✓ Prepared {len(df)} jobs for minute-resolution scheduling."
                        )
                        st.dataframe(
                            df.head(10),
                            use_container_width=True,
                            hide_index=True,
                        )
                        st.download_button(
                            "Download prepared scheduling data",
                            data=df.to_csv(index=False).encode("utf-8"),
                            file_name="prepared_manufacturing_jobs.csv",
                            mime="text/csv",
                            key="download_prepared_from_scheduler",
                        )
                    else:
                        st.info(
                            "Confirm the processing-time and energy assumptions to prepare "
                            "this dataset for scheduling."
                        )
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    st.error(f"Could not prepare manufacturing data: {exc}")
                    df = None

with tariff_col:
    with st.container(border=True):
        st.subheader("⚡ Electricity Tariff")
        uploaded_tariff = st.file_uploader(
            "Choose tariff CSV",
            type=["csv"],
            key="tariff_csv_upload",
            help=(
                "Required columns: hour and tariff_rs_per_kwh. Provide exactly "
                "one row for each hour from 0 through 23."
            ),
        )
        try:
            if uploaded_tariff is not None:
                hourly_tariff = load_tariff(uploaded_tariff)
                tariff_source = f"Uploaded: {uploaded_tariff.name}"
            else:
                hourly_tariff = load_tariff(DATA_DIR / "tariff.csv")
                tariff_source = "Bundled tariff: data/tariff.csv"
            st.success(f"✓ {tariff_source}")
            st.plotly_chart(
                make_tariff_chart(hourly_tariff),
                use_container_width=True,
            )
            st.caption(
                "The bundled rates are illustrative, not a live electricity-provider feed."
            )
        except (ValueError, KeyError, TypeError, OSError) as exc:
            st.error(f"Invalid tariff CSV: {exc}")
            st.stop()

if df is None:
    st.info("Choose a valid data source and complete any required data preparation above.")
    st.stop()

# ----------------------------------------------------------------------
# 2. Review data and configure the solver
# ----------------------------------------------------------------------
st.header("2. Review Scheduling Inputs")
st.caption(f"Source: {jobs_source}")

c1, c2 = st.columns([3, 2])
with c1:
    st.dataframe(df, use_container_width=True, hide_index=True)
    machines = get_machines(df)
    st.caption(f"{len(df)} jobs across {len(machines)} machines: {', '.join(machines)}")
with c2:
    st.markdown("**Time units**")
    if schedule_time_unit == 60:
        st.info("Standard dataset: all time values are in hours.")
    else:
        st.info(
            "External dataset: the optimizer uses minutes internally. "
            "Results are displayed in hours."
        )
    if schedule_time_unit == 1:
        st.warning(
            "External-data results depend on the selected energy assumption, "
            "reference completion targets and tariff. Costs are estimates."
        )

st.sidebar.header("Solver Settings")
time_limit = st.sidebar.slider(
    "Solver time limit (seconds)",
    min_value=5,
    max_value=120,
    value=60,
    key="main_solver_time_limit",
)

current_run_signature = signature_for_run(
    df, hourly_tariff, time_limit, schedule_time_unit, str(jobs_source)
)

# ----------------------------------------------------------------------
# 3. Run FCFS baseline and OR-Tools optimization
# ----------------------------------------------------------------------
st.header("3. Run Scheduling")
run = st.button("Run Baseline + OR Optimization", type="primary")

if run:
    try:
        with st.spinner("Building the FCFS baseline..."):
            baseline_schedule = run_fcfs(df)
            baseline_metrics = compute_metrics(
                baseline_schedule,
                hourly_tariff,
                time_unit_minutes=schedule_time_unit,
            )

        with st.spinner("Solving with OR-Tools CP-SAT..."):
            result = solve_schedule(
                df,
                hourly_tariff,
                time_limit_seconds=time_limit,
                time_unit_minutes=schedule_time_unit,
            )

        st.session_state["baseline_schedule"] = baseline_schedule
        st.session_state["baseline_metrics"] = baseline_metrics
        st.session_state["result"] = result
        st.session_state["run_signature"] = current_run_signature
        st.session_state.pop("what_if_result", None)
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"Could not run scheduling: {exc}")
    except Exception as exc:
        st.error(f"Unexpected scheduling error: {exc}")

has_current_results = (
    "result" in st.session_state
    and st.session_state.get("run_signature") == current_run_signature
)

if has_current_results:
    result = st.session_state["result"]
    baseline_schedule = st.session_state["baseline_schedule"]
    baseline_metrics = st.session_state["baseline_metrics"]

    if result.status not in ("OPTIMAL", "FEASIBLE") or result.schedule.empty:
        st.error(
            f"Solver returned {result.status}; no feasible schedule was returned. "
            "UNKNOWN means the solver did not find a feasible solution or prove "
            "infeasibility within its time limit. Try fewer jobs or a longer limit."
        )
    else:
        if result.used_soft_deadlines:
            st.warning(
                "The optimizer used soft deadlines. "
                f"{result.deadline_violations} job(s) finish after their completion targets."
            )
        elif result.status == "OPTIMAL":
            st.success("Optimal schedule found for the selected model and assumptions.")
        else:
            st.success("A feasible schedule was found; optimality was not proven within the time limit.")

        optimized_schedule = result.schedule.copy()
        optimized_metrics = compute_metrics(
            optimized_schedule,
            hourly_tariff,
            time_unit_minutes=schedule_time_unit,
        )

        optimized_display = schedule_as_hours(optimized_schedule, schedule_time_unit)
        baseline_display = schedule_as_hours(baseline_schedule, schedule_time_unit)

        st.header("4. Optimized Schedule")
        st.dataframe(
            display_schedule_table(optimized_display),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download optimized schedule (times in hours)",
            data=display_schedule_table(optimized_display).to_csv(index=False).encode("utf-8"),
            file_name="karkhanawatt_optimized_schedule_hours.csv",
            mime="text/csv",
            key="download_main_optimized_schedule",
        )

        st.header("5. Gantt Chart")
        tab1, tab2 = st.tabs(["OR-Optimized", "Baseline (FCFS)"])
        with tab1:
            st.plotly_chart(
                make_gantt_chart(
                    optimized_display,
                    "OR-Optimized Production Schedule",
                ),
                use_container_width=True,
            )
        with tab2:
            st.plotly_chart(
                make_gantt_chart(
                    baseline_display,
                    "Baseline Schedule (FCFS)",
                ),
                use_container_width=True,
            )

        st.header("6. Performance Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated Electricity Cost", f"₹{optimized_metrics['electricity_cost']:,.2f}")
        m2.metric("Makespan", f"{optimized_metrics['makespan']:.2f} hours")
        m3.metric("Deadline / Target Violations", optimized_metrics["deadline_violations"])
        m4, m5 = st.columns(2)
        m4.metric("Average Machine Utilization", f"{optimized_metrics['machine_utilization_pct']:.1f}%")
        m5.metric("Estimated Energy", f"{optimized_metrics['total_energy_kwh']:.2f} kWh")

        if abs(float(result.total_cost) - float(optimized_metrics["electricity_cost"])) > 0.05:
            st.warning(
                "The optimizer's objective cost differs from the independently calculated schedule cost. "
                "Check tariff/power scaling before interpreting cost values."
            )

        st.header("7. Baseline vs Optimized Comparison")
        st.dataframe(
            comparison_table(baseline_metrics, optimized_metrics),
            use_container_width=True,
            hide_index=True,
        )
        baseline_cost = baseline_metrics["electricity_cost"]
        if baseline_cost:
            savings_pct = (
                (baseline_cost - optimized_metrics["electricity_cost"])
                / baseline_cost * 100
            )
            st.caption(
                f"Cost difference compared with FCFS baseline: {savings_pct:.1f}%. "
                "This is calculated from the selected jobs and tariff."
            )

        # --------------------------------------------------------------
        # Smart Optimization Insights
        # --------------------------------------------------------------
        st.header("8. Smart Optimization Insights")
        insights = optimized_schedule.copy()
        insights["processing_hours"] = (
            insights["processing_time"] * schedule_time_unit / 60.0
        )
        insights["estimated_energy_kwh"] = (
            insights["power_kw"] * insights["processing_hours"]
        )
        insights["estimated_cost_rs"] = [
            schedule_cost(
                optimized_schedule.iloc[[idx]],
                hourly_tariff,
                time_unit_minutes=schedule_time_unit,
            )
            for idx in range(len(optimized_schedule))
        ]

        insight1, insight2 = st.columns(2)
        top_cost_row = insights.loc[insights["estimated_cost_rs"].idxmax()]
        machine_load = (
            insights.groupby("machine")["processing_hours"]
            .sum()
            .sort_values(ascending=False)
        )
        insight1.metric(
            "Highest Estimated Job Cost",
            f"₹{top_cost_row['estimated_cost_rs']:,.2f}",
            help=f"Job ID: {top_cost_row['job_id']}; based on the selected tariff and energy assumptions.",
        )
        insight2.metric("Busiest Machine", str(machine_load.idxmax()))
        st.subheader("Machine Workload (scheduled hours)")
        st.bar_chart(machine_load)
        st.subheader("Top 5 Jobs by Estimated Electricity Cost")
        top_jobs = insights.sort_values("estimated_cost_rs", ascending=False).head(5)
        st.dataframe(
            top_jobs[
                ["job_id", "machine", "processing_hours", "estimated_energy_kwh",
                 "estimated_cost_rs", "deadline_violated"]
            ].rename(columns={
                "job_id": "Job ID",
                "machine": "Machine",
                "processing_hours": "Duration (hours)",
                "estimated_energy_kwh": "Estimated Energy (kWh)",
                "estimated_cost_rs": "Estimated Cost (Rs.)",
                "deadline_violated": "Late?",
            }),
            use_container_width=True,
            hide_index=True,
        )
        if schedule_time_unit == 1:
            st.caption(
                "External-data power, energy and cost figures are assumption-based estimates. "
                "The completion targets originate from Scheduled_End and are not verified contractual deadlines."
            )

        # --------------------------------------------------------------
        # Smart Factory What-If Lab: inputs are displayed in hours and
        # converted to model units before being passed to CP-SAT.
        # --------------------------------------------------------------
        st.header("9. Smart Factory What-If Lab")
        st.write(
            "Simulate a machine breakdown or an urgent order, then compare the revised schedule. "
            "Scenario controls are displayed in hours; the app converts them to the selected model units."
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
            "Choose a scenario",
            ["Machine breakdown", "Add urgent job"],
            horizontal=True,
            key="what_if_scenario_type",
        )

        with st.form("what_if_scenario_form"):
            scenario_details = {}
            if scenario_type == "Machine breakdown":
                scenario_machine = st.selectbox(
                    "Machine that becomes unavailable", machines
                )
                breakdown_start_hours = st.number_input(
                    "Breakdown starts at hour",
                    min_value=0.0,
                    max_value=float(max_model_hour),
                    value=min(10.0, float(max_model_hour)),
                    step=hour_step,
                )
                breakdown_duration_hours = st.number_input(
                    "Downtime duration (hours)",
                    min_value=hour_step,
                    max_value=48.0,
                    value=3.0,
                    step=hour_step,
                )
                scenario_details = {
                    "machine": str(scenario_machine),
                    "start": float(breakdown_start_hours),
                    "end": float(breakdown_start_hours + breakdown_duration_hours),
                }
            else:
                scenario_job_id = st.text_input("Urgent job ID", value="URGENT-1").strip()
                scenario_machine = st.selectbox(
                    "Machine for the urgent job", machines
                )
                scenario_processing_hours = st.number_input(
                    "Urgent-job processing time (hours)",
                    min_value=min_processing_hours,
                    max_value=72.0,
                    value=2.0,
                    step=hour_step,
                )
                release_hours = st.number_input(
                    "Order becomes available at hour",
                    min_value=0.0,
                    max_value=float(max_model_hour),
                    value=min(2.0, float(max_model_hour)),
                    step=hour_step,
                )
                deadline_hours = st.number_input(
                    "Reference deadline (hour from planning start)",
                    min_value=min_processing_hours,
                    max_value=max(500.0, float(max_model_hour)),
                    value=max(8.0, float(df["deadline"].min()) / units_per_hour),
                    step=hour_step,
                )
                scenario_power = st.number_input(
                    "Power draw (kW)",
                    min_value=0.1,
                    max_value=1000.0,
                    value=30.0,
                    step=5.0,
                )
                predecessor_choices = ["No predecessor"] + df["job_id"].astype(str).tolist()
                scenario_predecessor = st.selectbox(
                    "Predecessor job (optional)", predecessor_choices
                )
                scenario_details = {
                    "job": {
                        "job_id": scenario_job_id,
                        "machine": str(scenario_machine),
                        "processing_time": float(scenario_processing_hours),
                        "deadline": float(deadline_hours),
                        "power_kw": float(scenario_power),
                        "arrival_order": int(df["arrival_order"].max()) + 1,
                        "predecessor": (
                            None if scenario_predecessor == "No predecessor"
                            else scenario_predecessor
                        ),
                        "release_time": float(release_hours),
                    }
                }

            simulate_scenario = st.form_submit_button(
                "Simulate and re-optimize", type="primary"
            )

        scenario_payload = {
            "base_run": current_run_signature,
            "scenario_type": scenario_type,
            "details": scenario_details,
            "time_limit": time_limit,
            "time_unit_minutes": schedule_time_unit,
        }
        scenario_signature = hashlib.sha256(
            json.dumps(scenario_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

        if simulate_scenario:
            try:
                scenario_jobs = df.copy(deep=True)
                scenario_downtime = {}
                scenario_release_times = {}

                if scenario_type == "Machine breakdown":
                    down_start_units = int(round(scenario_details["start"] * units_per_hour))
                    down_end_units = int(round(scenario_details["end"] * units_per_hour))
                    scenario_downtime = {
                        scenario_details["machine"]: [(down_start_units, down_end_units)]
                    }
                else:
                    job_details = scenario_details["job"]
                    if not job_details["job_id"]:
                        raise ValueError("Urgent job ID cannot be blank.")
                    if job_details["job_id"] in set(df["job_id"].astype(str)):
                        raise ValueError("Urgent job ID already exists in the dataset.")

                    new_job_row = {
                        "job_id": job_details["job_id"],
                        "machine": job_details["machine"],
                        "processing_time": int(round(job_details["processing_time"] * units_per_hour)),
                        "deadline": int(round(job_details["deadline"] * units_per_hour)),
                        "power_kw": float(job_details["power_kw"]),
                        "arrival_order": int(job_details["arrival_order"]),
                        "predecessor": job_details["predecessor"],
                    }
                    scenario_jobs = pd.concat(
                        [scenario_jobs, pd.DataFrame([new_job_row])],
                        ignore_index=True,
                    )
                    validate_jobs(scenario_jobs)
                    scenario_release_times = {
                        job_details["job_id"]: int(round(job_details["release_time"] * units_per_hour))
                    }

                with st.spinner("Re-optimizing the scenario with OR-Tools CP-SAT..."):
                    scenario_result = solve_schedule(
                        scenario_jobs,
                        hourly_tariff,
                        time_limit_seconds=time_limit,
                        machine_downtime=scenario_downtime,
                        job_release_times=scenario_release_times,
                        time_unit_minutes=schedule_time_unit,
                    )

                if (
                    scenario_result.status not in ("OPTIMAL", "FEASIBLE")
                    or scenario_result.schedule.empty
                ):
                    st.error(
                        f"Scenario solver returned {scenario_result.status}; no feasible schedule was returned."
                    )
                else:
                    scenario_metrics = compute_metrics(
                        scenario_result.schedule,
                        hourly_tariff,
                        time_unit_minutes=schedule_time_unit,
                    )
                    st.session_state["what_if_result"] = {
                        "signature": scenario_signature,
                        "type": scenario_type,
                        "details": scenario_details,
                        "jobs": scenario_jobs,
                        "downtime": scenario_downtime,
                        "result": scenario_result,
                        "metrics": scenario_metrics,
                    }
            except (ValueError, KeyError, TypeError) as exc:
                st.error(f"This scenario could not be simulated: {exc}")
            except Exception as exc:
                st.error(f"Unexpected scenario error: {exc}")

        saved_scenario = st.session_state.get("what_if_result")
        if saved_scenario and saved_scenario.get("signature") == scenario_signature:
            scenario_result = saved_scenario["result"]
            scenario_metrics = saved_scenario["metrics"]
            scenario_schedule = scenario_result.schedule
            original_cost = optimized_metrics["electricity_cost"]
            revised_cost = scenario_metrics["electricity_cost"]
            cost_delta = revised_cost - original_cost

            if scenario_result.status == "OPTIMAL":
                st.success("Scenario solved to optimality for the configured model.")
            else:
                st.info(
                    f"Scenario status: {scenario_result.status}. A feasible schedule was found, "
                    "but optimality was not proven within the time limit."
                )
            if scenario_result.used_soft_deadlines:
                st.warning(
                    f"{scenario_result.deadline_violations} job(s) finish after their completion targets."
                )

            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "Revised electricity cost",
                f"₹{revised_cost:,.2f}",
                delta=f"₹{cost_delta:+,.2f} vs original",
                delta_color="inverse",
            )
            k2.metric(
                "Revised makespan",
                f"{scenario_metrics['makespan']:.2f} h",
                delta=f"{scenario_metrics['makespan'] - optimized_metrics['makespan']:+.2f} h",
                delta_color="inverse",
            )
            k3.metric(
                "Late jobs",
                int(scenario_metrics["deadline_violations"]),
                delta=f"{scenario_metrics['deadline_violations'] - optimized_metrics['deadline_violations']:+d} vs original",
            )
            k4.metric(
                "Machine utilization",
                f"{scenario_metrics['machine_utilization_pct']:.1f}%",
                delta=f"{scenario_metrics['machine_utilization_pct'] - optimized_metrics['machine_utilization_pct']:+.1f} pp",
            )

            if saved_scenario["type"] == "Add urgent job":
                st.warning(
                    "The revised plan includes additional production work; its total cost is not a direct cost-saving comparison."
                )

            st.subheader("Why did the schedule change?")
            for explanation in build_explanation(
                saved_scenario["type"],
                optimized_schedule,
                scenario_schedule,
                saved_scenario["details"],
            ):
                st.write(f"• {explanation}")

            changes_df = changed_jobs(optimized_schedule, scenario_schedule)
            if not changes_df.empty and schedule_time_unit != 60:
                for column in (
                    "original_start", "revised_start", "original_end", "revised_end"
                ):
                    if column in changes_df.columns:
                        changes_df[column] = (
                            changes_df[column] / units_per_hour
                        ).round(3)
            st.subheader("Jobs that changed")
            if changes_df.empty:
                st.info("No existing job's recorded machine or timing changed.")
            else:
                st.dataframe(changes_df, use_container_width=True, hide_index=True)

            st.subheader("Original vs revised schedule")
            original_tab, revised_tab = st.tabs(["Original optimized plan", "Revised scenario plan"])
            with original_tab:
                st.plotly_chart(
                    make_gantt_chart(
                        optimized_display,
                        "Original optimized production plan",
                    ),
                    use_container_width=True,
                )
            with revised_tab:
                scenario_display = schedule_as_hours(scenario_schedule, schedule_time_unit)
                downtime_for_chart = {
                    machine: [
                        (start / units_per_hour, end / units_per_hour)
                        for start, end in blocks
                    ]
                    for machine, blocks in saved_scenario["downtime"].items()
                }
                st.plotly_chart(
                    make_gantt_chart(
                        scenario_display,
                        "Revised scenario production plan",
                        downtime=downtime_for_chart,
                    ),
                    use_container_width=True,
                )
            st.download_button(
                "Download revised scenario schedule (times in hours)",
                data=display_schedule_table(
                    schedule_as_hours(scenario_schedule, schedule_time_unit)
                ).to_csv(index=False).encode("utf-8"),
                file_name="karkhanawatt_what_if_schedule_hours.csv",
                mime="text/csv",
                key="download_what_if_schedule",
            )
        elif st.session_state.get("what_if_result"):
            st.info("Scenario or dataset settings changed. Run the scenario again for current inputs.")

elif "result" in st.session_state:
    st.info(
        "Inputs changed since the last optimization. Click 'Run Baseline + OR Optimization' "
        "to recalculate the schedule for the current dataset and tariff."
    )
else:
    st.info("Review your data and tariff above, then run the optimizer to generate results.")
