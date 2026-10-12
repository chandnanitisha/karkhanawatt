
import pandas as pd


def _find_column(df, candidates):
    """Return the first matching column name."""
    return next(
        (column for column in candidates if column in df.columns),
        None
    )


def _get_value(row, column):
    """Safely read a value from a schedule row."""
    if column is None:
        return None

    value = row.get(column)

    if value is None or pd.isna(value):
        return None

    return value


def _same_value(a, b):
    """Compare two values, including numeric schedule times."""
    if a is None and b is None:
        return True

    if a is None or b is None:
        return False

    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return str(a) == str(b)


def changed_jobs(original_schedule, revised_schedule):
    """Return jobs whose machine or timing changed."""
    output_columns = [
        "job_id",
        "original_machine",
        "revised_machine",
        "original_start",
        "revised_start",
        "original_end",
        "revised_end",
        "change",
    ]

    if original_schedule is None or revised_schedule is None:
        return pd.DataFrame(columns=output_columns)

    id_col = _find_column(
        original_schedule,
        ["job_id", "job", "Job_ID"]
    )

    revised_id_col = _find_column(
        revised_schedule,
        ["job_id", "job", "Job_ID"]
    )

    if id_col is None or revised_id_col is None:
        raise ValueError(
            "The original and revised schedules must contain a job_id column."
        )

    machine_col = _find_column(
        original_schedule, ["machine", "machine_id"]
    )
    revised_machine_col = _find_column(
        revised_schedule, ["machine", "machine_id"]
    )

    start_col = _find_column(
        original_schedule, ["start", "start_time", "start_hour"]
    )
    revised_start_col = _find_column(
        revised_schedule, ["start", "start_time", "start_hour"]
    )

    end_col = _find_column(
        original_schedule, ["end", "end_time", "end_hour"]
    )
    revised_end_col = _find_column(
        revised_schedule, ["end", "end_time", "end_hour"]
    )

    original_rows = {
        str(row[id_col]): row
        for _, row in original_schedule.iterrows()
    }

    revised_rows = {
        str(row[revised_id_col]): row
        for _, row in revised_schedule.iterrows()
    }

    results = []

    all_job_ids = list(original_rows)

    for job_id in revised_rows:
        if job_id not in original_rows:
            all_job_ids.append(job_id)

    for job_id in all_job_ids:
        old = original_rows.get(job_id)
        new = revised_rows.get(job_id)

        if old is None:
            results.append({
                "job_id": job_id,
                "original_machine": None,
                "revised_machine": _get_value(new, revised_machine_col),
                "original_start": None,
                "revised_start": _get_value(new, revised_start_col),
                "original_end": None,
                "revised_end": _get_value(new, revised_end_col),
                "change": "Added job",
            })
            continue

        if new is None:
            results.append({
                "job_id": job_id,
                "original_machine": _get_value(old, machine_col),
                "revised_machine": None,
                "original_start": _get_value(old, start_col),
                "revised_start": None,
                "original_end": _get_value(old, end_col),
                "revised_end": None,
                "change": "Removed job",
            })
            continue

        old_machine = _get_value(old, machine_col)
        new_machine = _get_value(new, revised_machine_col)
        old_start = _get_value(old, start_col)
        new_start = _get_value(new, revised_start_col)
        old_end = _get_value(old, end_col)
        new_end = _get_value(new, revised_end_col)

        changes = []

        if not _same_value(old_machine, new_machine):
            changes.append("machine")

        if not _same_value(old_start, new_start):
            changes.append("start time")

        if not _same_value(old_end, new_end):
            changes.append("end time")

        if changes:
            results.append({
                "job_id": job_id,
                "original_machine": old_machine,
                "revised_machine": new_machine,
                "original_start": old_start,
                "revised_start": new_start,
                "original_end": old_end,
                "revised_end": new_end,
                "change": ", ".join(changes),
            })

    return pd.DataFrame(results, columns=output_columns)


def build_explanation(
    scenario_type,
    original_schedule,
    revised_schedule,
    details
):
    """Explain the constraints and schedule changes in a scenario."""
    changes_df = changed_jobs(original_schedule, revised_schedule)

    existing_changes = changes_df[
        ~changes_df["change"].isin(["Added job", "Removed job"])
    ]

    explanations = []

    if scenario_type == "Machine breakdown":
        machine = details["machine"]
        start = details["start"]
        end = details["end"]

        explanations.append(
            f"Machine {machine} is unavailable from hour {start} "
            f"to hour {end}. The optimizer must avoid scheduling "
            "jobs on that machine during this interval."
        )

        count = len(existing_changes)

        if count:
            explanations.append(
                f"{count} existing job(s) changed machine assignment "
                "and/or start or end time compared with the original plan."
            )
        else:
            explanations.append(
                "No existing job's recorded machine assignment or "
                "start/end time changed. The breakdown interval may not "
                "conflict with the original schedule."
            )

    elif scenario_type == "Add urgent job":
        job = details["job"]

        explanations.append(
            f"Job {job['job_id']} was added on machine "
            f"{job['machine']}, with a processing time of "
            f"{job['processing_time']} hour(s)."
        )

        explanations.append(
            f"The new job cannot start before hour "
            f"{job['release_time']} and has a deadline at hour "
            f"{job['deadline']}."
        )

        count = len(existing_changes)

        explanations.append(
            f"{count} existing job(s) changed machine assignment "
            "and/or start or end time compared with the original plan."
        )

        explanations.append(
            "The revised plan includes additional production work, "
            "so its total electricity cost should not be interpreted "
            "as a direct cost-saving comparison."
        )

    else:
        explanations.append(
            "The scenario type is not recognized."
        )

    return explanations
