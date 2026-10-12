# KarkhanaWatt — Energy-Aware Production Scheduling

KarkhanaWatt is a Streamlit application for production scheduling using **Operations Research and Google OR-Tools CP-SAT**. It schedules jobs assigned to machines, considers deadlines or reference completion targets, estimates electricity cost under a time-varying tariff, and compares the optimized result with an FCFS (First-Come, First-Served) baseline.

The application has one **Production Scheduler** interface with two supported data-source options:

1. **Standard Jobs CSV** — uses the project's standard scheduling schema and whole-hour time units.
2. **External Manufacturing CSV** — prepares a compatible manufacturing dataset into scheduling inputs and uses minute-level time steps. This path currently supports the known manufacturing CSV structure described below; it is not a universal CSV-to-scheduler converter.

> **Important:** The external manufacturing workflow uses documented assumptions because the uploaded Kaggle CSV does not establish every scheduling field or measurement unit. In particular, `Scheduled_End` is treated as a *reference completion target*, and `Energy_Consumption` is interpreted as kWh per job only when the user explicitly accepts that assumption. The resulting power, energy and electricity-cost values are estimates, not verified factory bills or proven factory savings. The bundled tariff is illustrative, not a live utility feed.

---

## Features

- Interactive Streamlit interface with separate upload controls for production data and electricity tariffs.
- CSV validation and a preview of the selected scheduling inputs.
- OR-Tools CP-SAT optimization that minimizes estimated electricity cost subject to machine and timing constraints.
- FCFS baseline schedule for comparison.
- Gantt charts for the baseline and optimized schedule.
- Metrics including makespan, machine utilization, idle time, deadline/reference-target violations, estimated energy use and estimated electricity cost.
- **Smart Optimization Insights**, including machine workload and jobs with the highest estimated electricity cost.
- **Smart Factory What-If Lab** to simulate a machine breakdown or add an urgent job, then compare the revised schedule with the original plan.
- Downloadable optimized and revised schedules in CSV format.
- Time-of-use tariff upload support; the bundled tariff is used when no tariff file is provided.

## Technology stack

- Python
- Streamlit
- Pandas
- Google OR-Tools CP-SAT
- Plotly

---

## Project structure

```text
karkhanawatt/
├── app.py                      # Main Streamlit application
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── data/
│   ├── jobs.csv                # Illustrative standard-format jobs
│   └── tariff.csv              # Illustrative 24-hour tariff
└── src/
    ├── __init__.py
    ├── data_loader.py          # CSV loading, validation and external-data preparation
    ├── baseline.py             # FCFS baseline scheduler
    ├── optimizer.py            # OR-Tools CP-SAT scheduling model
    ├── tariff.py               # Tariff loading and cost calculations
    ├── metrics.py              # Schedule metrics and comparison table
    ├── visualization.py        # Gantt and tariff charts
    └── what_if.py              # Scenario comparison and explanations
```

The `src/what_if.py` module must exist because `app.py` imports `changed_jobs` and `build_explanation` from it.

---

## Setup in VS Code on Windows

### 1. Open the project

Extract the project folder and open the folder containing `app.py` in VS Code. Open **Terminal → New Terminal** and check that the terminal is in the project directory:

```powershell
dir
```

You should see `app.py`, `requirements.txt`, `data`, and `src`.

### 2. Create a virtual environment

```powershell
py -m venv .venv
```

### 3. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

The terminal prompt should show `(.venv)`. If PowerShell blocks activation, use the Python interpreter in `.venv` directly or follow your institution's approved PowerShell policy; avoid installing project dependencies globally just to bypass the environment.

### 4. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 5. Run the app

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

Stop the server with `Ctrl+C` in the terminal.

---

## Using the Production Scheduler

### A. Standard Jobs CSV

Choose **Standard Jobs CSV** in the data-source selector. If you do not upload a file, the application uses `data/jobs.csv`.

The standard jobs CSV must contain these columns:

| Column | Meaning | Requirements |
|---|---|---|
| `job_id` | Unique job identifier | Required; non-empty and unique |
| `machine` | Machine assigned to the job | Required; non-empty |
| `processing_time` | Job duration | Required; positive whole number of hours |
| `deadline` | Latest completion time, measured from planning hour 0 | Required; positive whole number of hours |
| `power_kw` | Power draw while the job runs | Required; positive numeric value in kW |
| `arrival_order` | Ordering used by the FCFS baseline | Required; whole number |
| `predecessor` | Job that must finish before this job starts | Optional; must reference another existing `job_id` |

Example:

```csv
job_id,machine,processing_time,deadline,power_kw,arrival_order,predecessor
J1,M1,4,40,90,1,
J2,M2,4,40,85,2,
J3,M1,3,20,20,3,J1
```

For this standard input mode, `processing_time` and `deadline` are in hours. A deadline of `10` means hour 10 from the planning start; a value such as `34` means hour 10 on the following day. The model uses fixed machine assignments, so each job runs on the machine specified in its input row.

### B. External Manufacturing CSV

Choose **External Manufacturing CSV** to upload the manufacturing CSV. Upload the extracted `.csv` file, not the ZIP archive.

The current adapter recognises these manufacturing fields after basic column-name normalization:

- `Job_ID`
- `Machine_ID`
- `Processing_Time`
- `Energy_Consumption`
- `Scheduled_Start`
- `Scheduled_End`

Additional available fields such as `Actual_Start`, `Actual_End`, `Job_Status`, `Operation_Type`, and `Machine_Availability` may be retained for dataset inspection when present. The adapter does not currently map every arbitrary manufacturing dataset automatically; incompatible column structures will be rejected and need a proper adapter or column-mapping step.

Before the external data is sent to the optimizer, the app asks the user to accept the assumptions below:

1. `Processing_Time` is measured in minutes.
2. `Energy_Consumption` represents kWh consumed per job.
3. `Scheduled_End` can be used as a reference completion target. It is **not** claimed to be a contractual customer deadline.

The adapter derives an average-power estimate from the accepted energy assumption:

```text
estimated average power (kW) = energy (kWh per job) / processing duration (hours)
```

The planning-time origin is derived from the earliest valid `Scheduled_Start`, and the reference targets are expressed as elapsed minutes from that origin. A small record limit is available for the prototype because minute-resolution optimization can be more computationally demanding. The current UI caps this mode at 30 records.

If the source dataset does not genuinely use these units or `Scheduled_End` does not represent a meaningful reference target, do not accept the assumptions for a real-world claim. Update the adapter to match the dataset documentation instead.

### C. Electricity tariff CSV

The tariff uploader accepts a CSV with exactly 24 unique rows, one for each hour from 0 to 23, and these required columns:

| Column | Meaning |
|---|---|
| `hour` | Hour index from 0 through 23 |
| `tariff_rs_per_kwh` | Electricity price in Rs/kWh |
| `band` | Optional tariff label, such as `off_peak`, `shoulder`, or `peak` |

Example:

```csv
hour,tariff_rs_per_kwh,band
0,4.5,off_peak
1,4.5,off_peak
2,4.5,off_peak
```

The full file must have all 24 hourly entries. The bundled `data/tariff.csv` is **illustrative** and is not automatically refreshed from an electricity provider. Use verified current rates for any deployment or real cost claim.

---

## Optimization model

For each job, the model has start time `S_i` and completion time `E_i`:

```text
E_i = S_i + P_i
```

The model enforces:

- **Machine no-overlap:** jobs assigned to the same machine cannot run at the same time.
- **Deadline or reference-target constraints:** the model first tries a strict schedule. When the strict model is infeasible (or inconclusive, depending on solver status handling), it can retry with soft deadlines and report violations.
- **Optional precedence:** a job's predecessor must finish before that job starts.
- **Release times:** an urgent job cannot start before its configured availability time.
- **Machine downtime:** simulated downtime intervals are included in machine no-overlap constraints.

The objective minimizes estimated electricity cost under the 24-hour time-of-use tariff. The optimizer reports `OPTIMAL` when it proves optimality, `FEASIBLE` when it finds a valid schedule without proving optimality, and `UNKNOWN` when it cannot establish a solution or prove infeasibility within the allowed time. `UNKNOWN` does not mean that a valid schedule is mathematically impossible.

The app's solver time limit is configurable. Larger datasets or minute-level schedules can require more solving time, so start with a small subset when testing external data.

---

## Results shown in the app

- Optimized schedule table and downloadable CSV (display times in hours).
- FCFS baseline schedule and optimized schedule Gantt charts.
- Estimated electricity cost under the selected tariff.
- Makespan in hours.
- Deadline or reference-target violation count.
- Average machine utilization and estimated energy.
- Baseline-versus-optimized comparison.
- Smart Optimization Insights: busiest machine, machine workload, and jobs with the largest estimated electricity cost.
- What-If comparisons for a machine breakdown or an added urgent job.

For external manufacturing data, these metrics remain assumption-based until the source units, energy interpretation, target times and tariff are verified. An `OPTIMAL` solver status means optimal for the encoded model and assumptions, not that the assumptions themselves are verified.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'streamlit'`

Activate `.venv` and run:

```powershell
python -m pip install -r requirements.txt
```

### `ModuleNotFoundError: No module named 'src.what_if'`

Check that `src/what_if.py` exists in the project and defines `changed_jobs` and `build_explanation`.

### A jobs CSV is rejected

The Standard Jobs CSV option requires the standard field names documented above. An external manufacturing CSV should be uploaded using the **External Manufacturing CSV** option. The current external adapter supports a specific manufacturing schema and is not a universal parser for any CSV.

### The solver returns `UNKNOWN`

Try fewer jobs or increase the solver time limit. `UNKNOWN` means the solver did not find a solution or prove infeasibility within the allotted time; it is not proof that the input is impossible.

### The optimizer and independent cost metrics differ

Check `POWER_SCALE`, `TARIFF_SCALE`, `COST_SCALE`, the `build_cumulative_table()` implementation, the selected time unit, and the `schedule_cost()` calculation in `src/tariff.py`. Do not present costs as trustworthy until the objective and independently calculated cost agree within the expected numeric tolerance.

---

## Current limitations

- The external manufacturing adapter is tailored to the supplied manufacturing CSV structure; arbitrary CSVs with unrelated or unknown columns are not automatically schedulable.
- The external path depends on user-confirmed assumptions about processing-time units, energy units and reference targets.
- The example tariff is illustrative and must be replaced with verified current rates for real-world cost reporting.
- The current optimization model uses fixed machine assignments; it does not automatically choose among alternative eligible machines unless the model is extended.
- The What-If Lab and metrics must use the selected time resolution consistently. Test both standard-hour and minute-resolution paths before relying on scenario outputs.
- The Kaggle data's provenance and units should be checked against its source documentation before claiming that the measurements represent verified factory records.

---

## Data source

The external manufacturing CSV used during development was downloaded from Kaggle:

- [Manufacturing Production Data — Kaggle](https://www.kaggle.com/datasets/ziya07/manufacturing-production-data/data)

The application can process the uploaded file structure, but the dataset's units and provenance should be verified from the dataset author's documentation before describing the results as real industrial measurements. The internal sample `data/jobs.csv` and bundled `data/tariff.csv` are illustrative project inputs.

---

## Project objective

KarkhanaWatt demonstrates how Operations Research can be applied to production scheduling under time-varying electricity prices. Its main contribution as a mini-project is the integrated workflow: data validation, CP-SAT optimization, FCFS comparison, schedule visualization, performance metrics, and What-If scenario analysis.
