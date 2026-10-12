"""
visualization.py
------------------
Builds the Gantt chart for a schedule using Plotly.

Times in the schedule are plain numeric hours (e.g. 0, 3.5, 18), not
calendar dates, so we draw the chart manually with horizontal bars
rather than px.timeline (which requires datetime values).
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd



def make_gantt_chart(
    schedule: pd.DataFrame,
    title: str = "Production Schedule",
    downtime: dict | None = None,
) -> go.Figure:
    downtime = downtime or {}

    if schedule.empty:
        fig = go.Figure()
        fig.update_layout(title=f"{title} (no feasible schedule)")
        return fig

    machines = sorted(
        set(schedule["machine"].unique()) | set(downtime.keys())
    )

    job_ids = schedule["job_id"].unique().tolist()
    colors = px.colors.qualitative.Set3
    color_map = {
        jid: colors[i % len(colors)]
        for i, jid in enumerate(job_ids)
    }

    fig = go.Figure()

    # Draw each scheduled production job.
    for _, row in schedule.iterrows():
        duration = row["end_time"] - row["start_time"]
        late = row.get("deadline_violated", False)

        fig.add_trace(
            go.Bar(
                x=[duration],
                y=[row["machine"]],
                base=[row["start_time"]],
                orientation="h",
                name=str(row["job_id"]),
                marker=dict(
                    color=color_map.get(row["job_id"], "steelblue"),
                    line=dict(
                        color="crimson" if late else "black",
                        width=2 if late else 1,
                    ),
                ),
                text=[
                    str(row["job_id"]) + (" (LATE)" if late else "")
                ],
                textposition="inside",
                hovertemplate=(
                    f"Job: {row['job_id']}<br>"
                    f"Machine: {row['machine']}<br>"
                    f"Start: {row['start_time']}<br>"
                    f"End: {row['end_time']}<br>"
                    f"Deadline: {row['deadline']}<br>"
                    f"Power: {row['power_kw']} kW"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    # Draw machine downtime as grey blocks.
    for machine, blocks in downtime.items():
        for block_index, (down_start, down_end) in enumerate(
            blocks, start=1
        ):
            fig.add_trace(
                go.Bar(
                    x=[down_end - down_start],
                    y=[machine],
                    base=[down_start],
                    orientation="h",
                    name=f"{machine} downtime",
                    marker=dict(
                        color="rgba(95, 99, 104, 0.85)",
                        line=dict(color="#444444", width=1),
                    ),
                    text=["BREAKDOWN"],
                    textposition="inside",
                    hovertemplate=(
                        f"Machine: {machine}<br>"
                        f"Unavailable: hour {down_start} to "
                        f"{down_end}<extra></extra>"
                    ),
                    showlegend=(block_index == 1),
                )
            )

    fig.update_layout(
        title=title,
        barmode="overlay",
        xaxis_title="Time (hours)",
        yaxis_title="Machine",
        yaxis=dict(
            categoryorder="array",
            categoryarray=machines,
        ),
        height=120 + 70 * len(machines),
    )

    return fig



def make_tariff_chart(hourly_tariff: list, title: str = "Electricity Tariff (24h)") -> go.Figure:
    """Bar chart of the Rs/kWh tariff rate by hour of day, for context."""
    hours = list(range(24))
    fig = go.Figure(
        go.Bar(
            x=hours,
            y=hourly_tariff,
            marker=dict(color=hourly_tariff, colorscale="OrRd"),
            hovertemplate="Hour %{x}:00<br>Rs %{y}/kWh<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Hour of day",
        yaxis_title="Rs / kWh",
        xaxis=dict(tickmode="linear", dtick=2),
        height=260,
        margin=dict(t=40, b=30),
    )
    return fig
