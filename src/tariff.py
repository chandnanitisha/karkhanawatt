"""
Time-of-Use (ToU) electricity tariff handling for KarkhanaWatt.

A tariff CSV must contain 24 rows, one for every hour (0-23), and the columns
'hour' and 'tariff_rs_per_kwh'. An optional 'band' column and additional
columns are allowed. The bundled rates are illustrative, not live DISCOM data.
"""

import math
import pandas as pd

TARIFF_SCALE = 6000   # tariff stored as integer paise/kWh internally
POWER_SCALE = 1000    # power stored as integer deci-kW internally
COST_SCALE = TARIFF_SCALE * POWER_SCALE


def load_tariff(csv_path_or_buffer="data/tariff.csv") -> list:
    """
    Load and validate a 24-hour tariff table.

    Accepts a path or a file-like object (including a Streamlit file uploader).
    Returns 24 floats in hour order: index 0 is 00:00-01:00, index 23 is
    23:00-00:00.
    """
    try:
        if hasattr(csv_path_or_buffer, "seek"):
            csv_path_or_buffer.seek(0)
        df = pd.read_csv(csv_path_or_buffer)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The tariff CSV is empty. Upload a CSV with hourly rates.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        raise ValueError("Could not read the tariff CSV. Check that it is a valid, readable CSV file.") from exc

    if df.empty:
        raise ValueError("The tariff CSV has a header but contains no data rows.")
    df.columns = [str(c).strip() for c in df.columns]
    required = ["hour", "tariff_rs_per_kwh"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "This file does not look like a KarkhanaWatt tariff CSV. "
            f"Missing required column(s): {missing}. Required columns: {required}. "
            "Column names must match exactly. Extra columns, such as 'band', are allowed."
        )

    for col in required:
        try:
            df[col] = pd.to_numeric(df[col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Tariff column '{col}' must contain numeric values only.") from exc
        if df[col].isna().any() or not df[col].map(lambda x: math.isfinite(float(x))).all():
            raise ValueError(f"Tariff column '{col}' contains a blank or non-finite value.")

    if not (df["hour"] % 1 == 0).all():
        raise ValueError("Tariff 'hour' values must be whole numbers from 0 through 23.")
    df["hour"] = df["hour"].astype(int)

    if len(df) != 24 or sorted(df["hour"].tolist()) != list(range(24)):
        raise ValueError(
            "The tariff CSV must contain exactly 24 rows: one unique row for each hour 0-23. "
            "Check for missing or duplicate hours."
        )
    if (df["tariff_rs_per_kwh"] < 0).any():
        raise ValueError("Tariff rates cannot be negative. Use rates in Rs/kWh.")

    df = df.sort_values("hour")
    return df["tariff_rs_per_kwh"].astype(float).tolist()


def hourly_rate(hourly_tariff: list, t: int) -> float:
    """Tariff rate (Rs/kWh) applicable during hour t; wraps after 24 hours."""
    return hourly_tariff[t % 24]



def build_cumulative_table(
    hourly_tariff: list,
    horizon: int,
    time_unit_minutes: int = 60,
) -> list:
    """
    Build a scaled cumulative tariff table for CP-SAT.

    time_unit_minutes=60: each time step represents one hour.
    time_unit_minutes=1: each time step represents one minute.

    The default preserves compatibility with the existing hourly optimizer.
    """
    if time_unit_minutes <= 0 or 60 % time_unit_minutes != 0:
        raise ValueError(
            "time_unit_minutes must be a positive divisor of 60."
        )

    F = [0]
    running = 0

    for t in range(horizon):
        minute_from_start = t * time_unit_minutes
        hour_index = minute_from_start // 60

        rate = hourly_rate(hourly_tariff, hour_index)

        contribution = round(
            rate * TARIFF_SCALE * time_unit_minutes / 60
        )

        running += contribution
        F.append(running)

    return F



def schedule_cost(
    schedule: pd.DataFrame,
    hourly_tariff: list,
    time_unit_minutes: int = 60,
) -> float:
    """Calculate electricity cost using hour or minute time steps."""
    if schedule.empty:
        return 0.0

    if time_unit_minutes <= 0 or 60 % time_unit_minutes != 0:
        raise ValueError(
            "time_unit_minutes must be a positive divisor of 60."
        )

    total = 0.0

    for _, row in schedule.iterrows():
        start = int(row["start_time"])
        end = int(row["end_time"])

        energy_price_sum = 0.0

        for t in range(start, end):
            hour_index = (t * time_unit_minutes) // 60

            energy_price_sum += (
                hourly_rate(hourly_tariff, hour_index)
                * time_unit_minutes / 60
            )

        total += float(row["power_kw"]) * energy_price_sum

    return round(total, 2)

