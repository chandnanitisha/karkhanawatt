"""
Load and validate production-job CSV data for KarkhanaWatt.

Required column names are deliberately strict so a different/unrelated CSV
cannot be mistaken for a jobs file. Additional columns are allowed and ignored
by the scheduler. Processing time and deadline use whole-hour units because the
current CP-SAT model discretizes time into integer hours.
"""

import math
import pandas as pd

REQUIRED_COLUMNS = [
    "job_id",
    "machine",
    "processing_time",
    "deadline",
    "power_kw",
    "arrival_order",
]
OPTIONAL_COLUMNS = ["predecessor"]


def _read_csv(csv_path_or_buffer, kind: str) -> pd.DataFrame:
    """Read a path or uploaded file and translate common parsing errors."""
    try:
        if hasattr(csv_path_or_buffer, "seek"):
            csv_path_or_buffer.seek(0)
        df = pd.read_csv(csv_path_or_buffer)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"The {kind} CSV is empty. Upload a CSV with a header row and data.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise ValueError(f"Could not read the {kind} CSV. Check that it is a valid, readable CSV file.") from exc

    if df.empty:
        raise ValueError(f"The {kind} CSV has a header but contains no data rows.")

    df.columns = [str(c).strip() for c in df.columns]
    if any(not c for c in df.columns):
        raise ValueError(f"The {kind} CSV contains a blank column name. Rename the column and try again.")
    duplicates = df.columns[df.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"The {kind} CSV contains duplicate column name(s): {duplicates}.")
    return df


def load_jobs(csv_path_or_buffer) -> pd.DataFrame:
    """Load a jobs CSV and validate all fields required by the scheduler."""
    df = _read_csv(csv_path_or_buffer, "jobs")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "This file does not look like a KarkhanaWatt jobs CSV. "
            f"Missing required column(s): {missing}. "
            f"Required columns: {REQUIRED_COLUMNS}. Column names must match exactly. "
            "Extra columns are allowed."
        )

    if "predecessor" not in df.columns:
        df["predecessor"] = None

    # Reject empty identifiers before converting values to strings (otherwise NaN becomes 'nan').
    for col in ("job_id", "machine"):
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"'{col}' must contain a non-empty value for every job.")
        df[col] = df[col].astype(str).str.strip()

    numeric_columns = ["processing_time", "deadline", "power_kw", "arrival_order"]
    for col in numeric_columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Column '{col}' must contain numeric values only.") from exc
        if df[col].isna().any() or not df[col].map(lambda x: math.isfinite(float(x))).all():
            raise ValueError(f"Column '{col}' contains a blank or non-finite value.")

    # The existing CP-SAT model uses integer-hour time slots; fail clearly rather than rounding silently.
    for col in ("processing_time", "deadline"):
        if not (df[col] % 1 == 0).all():
            raise ValueError(
                f"Column '{col}' must use whole hours in the current model. "
                "Decimal-hour scheduling is not supported yet."
            )

    if not (df["arrival_order"] % 1 == 0).all():
        raise ValueError("Column 'arrival_order' must contain whole-number values.")

    df["processing_time"] = df["processing_time"].astype(int)
    df["deadline"] = df["deadline"].astype(int)
    df["arrival_order"] = df["arrival_order"].astype(int)
    df["power_kw"] = df["power_kw"].astype(float)

    df["predecessor"] = df["predecessor"].apply(
        lambda v: None if pd.isna(v) or str(v).strip() == "" else str(v).strip()
    )

    validate_jobs(df)
    return df.reset_index(drop=True)


def validate_jobs(df: pd.DataFrame) -> None:
    """Raise ValueError with a clear message if input data cannot be used."""
    if df["job_id"].duplicated().any():
        dups = df.loc[df["job_id"].duplicated(keep=False), "job_id"].unique().tolist()
        raise ValueError(f"Duplicate job_id value(s) found: {dups}. Each job_id must be unique.")

    if (df["processing_time"] <= 0).any():
        raise ValueError("processing_time must be greater than 0 for every job.")
    if (df["deadline"] <= 0).any():
        raise ValueError("deadline must be greater than 0 for every job.")
    if (df["power_kw"] <= 0).any():
        raise ValueError("power_kw must be greater than 0 for every job.")
    if (df["arrival_order"] < 0).any():
        raise ValueError("arrival_order must be zero or a positive whole number.")

    infeasible = df[df["processing_time"] > df["deadline"]]
    if not infeasible.empty:
        raise ValueError(
            "These jobs cannot finish by their deadlines even if started at t=0: "
            f"{infeasible['job_id'].tolist()}"
        )

    valid_ids = set(df["job_id"])
    predecessor_by_job = {}
    for _, row in df.iterrows():
        pred = row["predecessor"]
        job_id = row["job_id"]
        if pred is not None and pred not in valid_ids:
            raise ValueError(f"Job {job_id} references unknown predecessor '{pred}'.")
        if pred == job_id:
            raise ValueError(f"Job {job_id} cannot be its own predecessor.")
        predecessor_by_job[job_id] = pred

    # Detect cycles in the single-predecessor dependency graph.
    for job_id in predecessor_by_job:
        seen = set()
        current = job_id
        while current is not None:
            if current in seen:
                raise ValueError(f"Predecessor cycle detected involving job '{current}'. Remove the circular dependency.")
            seen.add(current)
            current = predecessor_by_job.get(current)


def get_machines(df: pd.DataFrame):
    """Return sorted unique machine identifiers."""
    return sorted(df["machine"].unique().tolist())


def load_manufacturing_records(csv_path_or_buffer) -> pd.DataFrame:
    """
    Load the Kaggle manufacturing dataset for inspection and preprocessing.

    This function preserves the meaning of the original measurements.
    It does NOT invent deadlines, convert energy to power, or assume
    undocumented units.
    """
    df = _read_csv(csv_path_or_buffer, "manufacturing")

    # Normalize column names for matching only.
    def normalize_column(name):
        return (
            str(name)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

    normalized_columns = {
        normalize_column(column): column
        for column in df.columns
    }

    # Map the known Kaggle columns to consistent internal names.
    aliases = {
        "job_id": ["job_id"],
        "machine_id": ["machine_id"],
        "processing_time_raw": ["processing_time"],
        "energy_consumption_raw": ["energy_consumption"],
        "scheduled_start": ["scheduled_start"],
        "scheduled_end": ["scheduled_end"],
        "actual_start": ["actual_start"],
        "actual_end": ["actual_end"],
        "job_status": ["job_status"],
        "operation_type": ["operation_type"],
        "machine_availability": ["machine_availability"],
    }

    rename_map = {}

    for standard_name, candidates in aliases.items():
        for candidate in candidates:
            source_column = normalized_columns.get(candidate)
            if source_column is not None:
                rename_map[source_column] = standard_name
                break

    df = df.rename(columns=rename_map)

    # These fields identify a production record and its measurements.
    required_source_fields = [
        "job_id",
        "machine_id",
        "processing_time_raw",
        "energy_consumption_raw",
    ]

    missing = [
        column
        for column in required_source_fields
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "The manufacturing CSV is missing required source fields: "
            f"{missing}. "
            "Expected fields include Job_ID, Machine_ID, "
            "Processing_Time and Energy_Consumption."
        )

    # Identifiers must not be empty.
    for column in ("job_id", "machine_id"):
        if (
            df[column].isna().any()
            or df[column].astype(str).str.strip().eq("").any()
        ):
            raise ValueError(
                f"Manufacturing column '{column}' contains blank identifiers."
            )

        df[column] = df[column].astype(str).str.strip()

    # Validate numeric source values without assuming their units.
    for column in (
        "processing_time_raw",
        "energy_consumption_raw",
    ):
        df[column] = pd.to_numeric(df[column], errors="coerce")

        if (
            df[column].isna().any()
            or not df[column].map(
                lambda value: math.isfinite(float(value))
            ).all()
        ):
            raise ValueError(
                f"Manufacturing column '{column}' contains missing "
                "or invalid numeric values. Inspect and clean the data."
            )

    if (df["processing_time_raw"] <= 0).any():
        raise ValueError(
            "Processing-time measurements must be greater than zero."
        )

    if (df["energy_consumption_raw"] < 0).any():
        raise ValueError(
            "Energy-consumption measurements cannot be negative."
        )

    return df.reset_index(drop=True)


def prepare_manufacturing_jobs(
    df: pd.DataFrame,
    max_jobs: int = 30,
    assume_energy_is_kwh_per_job: bool = False,
) -> pd.DataFrame:
    """
    Convert manufacturing records into KarkhanaWatt scheduling inputs.

    Assumptions that must be explicitly enabled:
    - Scheduled_End is used as a reference completion target, not a
      verified contractual deadline.
    - Energy_Consumption is interpreted as kWh consumed per job.
      Average power is estimated as energy / processing duration.

    Processing times and deadlines in the returned dataframe are in minutes.
    The optimizer must therefore be called with time_unit_minutes=1.
    """
    if not assume_energy_is_kwh_per_job:
        raise ValueError(
            "Energy units have not been confirmed. Enable the kWh-per-job "
            "assumption only if you accept this interpretation of the data."
        )

    if max_jobs < 1:
        raise ValueError("max_jobs must be at least 1.")

    required = [
        "job_id",
        "machine_id",
        "processing_time_raw",
        "energy_consumption_raw",
        "scheduled_start",
        "scheduled_end",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Manufacturing data is missing required fields: {missing}"
        )

    work = df.copy()

    work["scheduled_start"] = pd.to_datetime(
        work["scheduled_start"], errors="coerce"
    )
    work["scheduled_end"] = pd.to_datetime(
        work["scheduled_end"], errors="coerce"
    )

    if work["scheduled_start"].isna().any():
        raise ValueError("Some Scheduled_Start values are missing or invalid.")

    if work["scheduled_end"].isna().any():
        raise ValueError("Some Scheduled_End values are missing or invalid.")

    work = work.sort_values(
        ["scheduled_start", "job_id"]
    ).head(max_jobs).copy()

    # Retain only rows with valid positive durations and energy values.
    work["processing_time_raw"] = pd.to_numeric(
        work["processing_time_raw"], errors="coerce"
    )
    work["energy_consumption_raw"] = pd.to_numeric(
        work["energy_consumption_raw"], errors="coerce"
    )

    if (
        work["processing_time_raw"].isna().any()
        or (work["processing_time_raw"] <= 0).any()
        or not work["processing_time_raw"].map(
            lambda value: float(value).is_integer()
        ).all()
    ):
        raise ValueError(
            "Processing_Time must contain positive whole-number minutes."
        )

    if (
        work["energy_consumption_raw"].isna().any()
        or (work["energy_consumption_raw"] <= 0).any()
    ):
        raise ValueError(
            "Energy_Consumption must contain positive numeric values."
        )

    origin = work["scheduled_start"].min()

    processing_minutes = work["processing_time_raw"].astype(int)

    # Use Scheduled_End only as a clearly labelled reference target.
    reference_deadline_minutes = (
        (work["scheduled_end"] - origin).dt.total_seconds() / 60
    )

    if (
        (reference_deadline_minutes <= 0).any()
        or not reference_deadline_minutes.map(
            lambda value: float(value).is_integer()
        ).all()
    ):
        raise ValueError(
            "Scheduled_End cannot be converted into valid whole-minute "
            "reference targets."
        )

    duration_hours = processing_minutes / 60.0

    prepared = pd.DataFrame({
        "job_id": work["job_id"].astype(str).str.strip().values,
        "machine": work["machine_id"].astype(str).str.strip().values,
        "processing_time": processing_minutes.values,
        "deadline": reference_deadline_minutes.astype(int).values,

        # This is an estimate based on the stated assumption,
        # not a directly measured power rating.
        "power_kw": (
            work["energy_consumption_raw"].to_numpy()
            / duration_hours.to_numpy()
        ),

        "arrival_order": range(1, len(work) + 1),
        "predecessor": None,
    })

    validate_jobs(prepared)

    return prepared.reset_index(drop=True)
