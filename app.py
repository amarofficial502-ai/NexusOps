
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px


# ==================================================
# 1. CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="NexusOps AI",
    page_icon="⚙️",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "tasks.csv"

st.title("NexusOps AI")
st.subheader("Autonomous Management & Operations Intelligence")
st.caption("Synthetic business data | Rule-based intelligence engine")


# ==================================================
# 2. DATA LOADING
# ==================================================

@st.cache_data
def load_tasks(file_path, modified_time):
    return pd.read_csv(file_path)


if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE}")
    st.stop()

try:
    df = load_tasks(str(DATA_FILE), DATA_FILE.stat().st_mtime)
except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(f"Unable to read dataset: {exc}")
    st.stop()


# ==================================================
# 3. SCHEMA VALIDATION
# ==================================================

required_columns = [
    "task_id",
    "department",
    "task_name",
    "owner",
    "status",
    "priority",
    "due_date",
    "estimated_hours",
    "actual_hours",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()


# ==================================================
# 4. DATA NORMALIZATION
# ==================================================

df["due_date"] = pd.to_datetime(
    df["due_date"],
    errors="coerce",
).dt.normalize()

for column in ["estimated_hours", "actual_hours"]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

for column in ["department", "task_name", "owner", "status", "priority"]:
    df[column] = df[column].astype("string").str.strip()

status_norm = df["status"].str.casefold()
priority_norm = df["priority"].str.casefold()

today = pd.Timestamp.now().normalize()

completed_mask = status_norm.eq("completed").fillna(False)
blocked_mask = status_norm.eq("blocked").fillna(False)

overdue_mask = (
    df["due_date"].notna()
    & df["due_date"].lt(today)
    & ~completed_mask
)


# ==================================================
# 5. DATA QUALITY
# ==================================================

missing_mask = df[required_columns].isna().any(axis=1)

task_id = df["task_id"].astype("string").str.strip()

blank_id_mask = (
    df["task_id"].isna()
    | task_id.eq("").fillna(True)
)

duplicate_id_mask = df["task_id"].duplicated(keep=False)

invalid_dates_mask = df["due_date"].isna()

invalid_hours_mask = (
    df[["estimated_hours", "actual_hours"]].isna().any(axis=1)
    | df[["estimated_hours", "actual_hours"]].lt(0).any(axis=1)
)

allowed_statuses = {
    "completed", "in progress", "overdue", "blocked"
}

allowed_priorities = {"high", "medium", "low"}

invalid_status_mask = (
    status_norm.isna()
    | ~status_norm.isin(allowed_statuses)
)

invalid_priority_mask = (
    priority_norm.isna()
    | ~priority_norm.isin(allowed_priorities)
)

quality_mask = (
    missing_mask
    | blank_id_mask
    | duplicate_id_mask
    | invalid_dates_mask
    | invalid_hours_mask
    | invalid_status_mask
    | invalid_priority_mask
)

quality_details = df.loc[quality_mask].copy()
quality_count = int(quality_mask.sum())


# ==================================================
# 6. TASK INTELLIGENCE ENGINE
# ==================================================

# These are transparent business rules, not an ML prediction.
priority_points = priority_norm.map({
    "high": 50,
    "medium": 30,
    "low": 10,
}).fillna(0)

days_remaining = (df["due_date"] - today).dt.days

deadline_points = pd.Series(0, index=df.index, dtype="int64")

deadline_points.loc[days_remaining.le(1).fillna(False)] = 30

deadline_points.loc[
    days_remaining.between(2, 3).fillna(False)
] = 20

deadline_points.loc[
    days_remaining.between(4, 7).fillna(False)
] = 10

blocked_points = blocked_mask.astype(int) * 20

# Completed tasks do not need a new-action priority.
df["priority_score"] = (
    priority_points + deadline_points + blocked_points
).clip(upper=100).astype(int)

df.loc[completed_mask, "priority_score"] = 0

df["days_remaining"] = days_remaining

df["sla_risk"] = "On Track"

df.loc[
    days_remaining.between(0, 2).fillna(False) & ~completed_mask,
    "sla_risk",
] = "At Risk"

df.loc[overdue_mask, "sla_risk"] = "Overdue"

df.loc[completed_mask, "sla_risk"] = "Completed"

df.loc[
    df["due_date"].isna(),
    "sla_risk",
] = "Unknown"

df.loc[
    blocked_mask & ~overdue_mask & ~completed_mask,
    "sla_risk",
] = "Blocked"

def recommend_action(row):
    """Provide a transparent recommendation using business rules."""
    if pd.isna(row["due_date"]):
        return "Validate the due date"

    if str(row["status"]).casefold() == "completed":
        return "No action required"

    if str(row["status"]).casefold() == "blocked":
        return "Investigate blocker and assign an owner"

    if row["due_date"] < today:
        return "Review delay and agree a recovery plan"

    if (row["due_date"] - today).days <= 2:
        return "Confirm progress and check dependencies"

    if str(row["priority"]).casefold() == "high":
        return "Review progress during the next status check"

    return "Continue routine monitoring"


df["recommended_action"] = df.apply(
    recommend_action,
    axis=1,
)


# ==================================================
# 7. NAVIGATION
# ==================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Task Intelligence",
        "Workload Intelligence",
        "Data Quality",
        "Root Cause Analysis",
        "Cost Simulator",
        "AI Actions",
        "Executive Reports",
    ],
)

st.sidebar.caption(f"Reporting date: {today.date()}")


# ==================================================
# 8. DASHBOARD
# ==================================================

if page == "Dashboard":
    st.header("Operations Control Tower")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Tasks", len(df))
    c2.metric("Overdue Tasks", int(overdue_mask.sum()))
    c3.metric("Blocked Tasks", int(blocked_mask.sum()))
    c4.metric("Data Quality Issues", quality_count)

    chart1, chart2 = st.columns(2)

    with chart1:
        department_counts = (
            df.groupby("department", dropna=False)
            .size()
            .reset_index(name="tasks")
        )

        st.plotly_chart(
            px.bar(
                department_counts,
                x="department",
                y="tasks",
                title="Tasks by Department",
            ),
            use_container_width=True,
        )

    with chart2:
        status_counts = (
            df["status"]
            .fillna("Missing")
            .value_counts()
            .rename_axis("status")
            .reset_index(name="tasks")
        )

        st.plotly_chart(
            px.pie(
                status_counts,
                names="status",
                values="tasks",
                title="Task Status Distribution",
            ),
            use_container_width=True,
        )

    st.subheader("Highest Priority Open Tasks")

    open_tasks = df.loc[~completed_mask].sort_values(
        "priority_score",
        ascending=False,
    )

    st.dataframe(
        open_tasks[
            [
                "task_id",
                "department",
                "task_name",
                "owner",
                "priority",
                "status",
                "sla_risk",
                "priority_score",
                "recommended_action",
            ]
        ].head(10),
        use_container_width=True,
        hide_index=True,
    )


# ==================================================
# 9. TASK INTELLIGENCE
# ==================================================

elif page == "Task Intelligence":
    st.header("Task Intelligence & SLA Risk")

    filter1, filter2 = st.columns(2)

    with filter1:
        risk_options = sorted(df["sla_risk"].dropna().unique().tolist())

        selected_risks = st.multiselect(
            "Filter by SLA Risk",
            risk_options,
            default=risk_options,
        )

    with filter2:
        priority_options = sorted(
            df["priority"].dropna().unique().tolist()
        )

        selected_priorities = st.multiselect(
            "Filter by Priority",
            priority_options,
            default=priority_options,
        )

    intelligence_df = df.loc[
        df["sla_risk"].isin(selected_risks)
        & df["priority"].isin(selected_priorities)
    ].copy()

    a, b, c = st.columns(3)

    a.metric("Matching Tasks", len(intelligence_df))

    a_risk = int(
        intelligence_df["sla_risk"].eq("At Risk").sum()
    )
    b.metric("At Risk", a_risk)

    overdue_count = int(
        intelligence_df["sla_risk"].eq("Overdue").sum()
    )
    c.metric("Overdue", overdue_count)

    st.subheader("Ranked Tasks")

    intelligence_df = intelligence_df.sort_values(
        "priority_score",
        ascending=False,
    )

    st.dataframe(
        intelligence_df[
            [
                "task_id",
                "department",
                "task_name",
                "owner",
                "due_date",
                "days_remaining",
                "priority",
                "status",
                "sla_risk",
                "priority_score",
                "recommended_action",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Priority scores use predefined rules. They are not trained "
        "ML predictions and do not guarantee task outcomes."
    )


# ==================================================
# 10. WORKLOAD INTELLIGENCE
# ==================================================

elif page == "Workload Intelligence":
    st.header("Workload Intelligence")

    open_df = df.loc[~completed_mask].copy()

    if open_df.empty:
        st.success("No open tasks in the current dataset.")
    else:
        workload = (
            open_df.groupby("owner", dropna=False)
            .agg(
                open_tasks=("task_id", "size"),
                estimated_hours=("estimated_hours", "sum"),
                blocked_tasks=("status", lambda x: x.str.casefold().eq("blocked").sum()),
            )
            .reset_index()
            .sort_values("estimated_hours", ascending=False)
        )

        st.plotly_chart(
            px.bar(
                workload,
                x="owner",
                y="estimated_hours",
                title="Estimated Open Workload by Owner",
            ),
            use_container_width=True,
        )

        st.dataframe(
            workload,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Estimated hours and task counts indicate workload volume, "
            "not employee performance or productivity."
        )


# ==================================================
# 11. DATA QUALITY
# ==================================================

elif page == "Data Quality":
    st.header("Data Quality & Validation")

    a, b, c, d = st.columns(4)

    a.metric("Missing Values", int(missing_mask.sum()))
    b.metric("Invalid Dates", int(invalid_dates_mask.sum()))
    c.metric("Invalid Hours", int(invalid_hours_mask.sum()))
    d.metric("Invalid Statuses", int(invalid_status_mask.sum()))

    e, f, g = st.columns(3)

    e.metric("Duplicate IDs", int(duplicate_id_mask.sum()))
    f.metric("Invalid Priorities", int(invalid_priority_mask.sum()))
    g.metric("Rows Requiring Review", quality_count)

    if quality_details.empty:
        st.success("No data quality issues detected.")
    else:
        st.dataframe(
            quality_details,
            use_container_width=True,
            hide_index=True,
        )


# ==================================================
# 12. ROOT CAUSE ANALYSIS
# ==================================================

elif page == "Root Cause Analysis":
    st.header("Root Cause Analysis")

    overdue_df = df.loc[overdue_mask].copy()

    if overdue_df.empty:
        st.success("No overdue tasks detected.")
    else:
        root_summary = (
            overdue_df.groupby("department", dropna=False)
            .size()
            .reset_index(name="overdue_tasks")
            .sort_values("overdue_tasks", ascending=False)
        )

        st.plotly_chart(
            px.bar(
                root_summary,
                x="department",
                y="overdue_tasks",
                title="Overdue Tasks by Department",
            ),
            use_container_width=True,
        )

        st.dataframe(
            overdue_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Department-level counts are investigation signals, "
            "not proof of root cause."
        )


# ==================================================
# 13. COST SIMULATOR
# ==================================================

elif page == "Cost Simulator":
    st.header("Cost & What-if Simulator")

    valid_hours = df[
        ["estimated_hours", "actual_hours"]
    ].dropna()

    valid_hours = valid_hours.loc[
        (valid_hours["estimated_hours"] >= 0)
        & (valid_hours["actual_hours"] >= 0)
    ]

    estimated = float(valid_hours["estimated_hours"].sum())
    actual = float(valid_hours["actual_hours"].sum())
    extra = max(actual - estimated, 0)

    hourly_rate = st.number_input(
        "Assumed cost per hour (₹)",
        min_value=0,
        value=500,
        step=50,
    )

    reduction = st.slider(
        "Hypothetical rework reduction (%)",
        0,
        100,
        20,
    )

    saved_hours = extra * reduction / 100
    saved_cost = saved_hours * hourly_rate

    c1, c2, c3 = st.columns(3)

    c1.metric("Estimated Hours", f"{estimated:.1f}")
    c2.metric("Actual Hours", f"{actual:.1f}")
    c3.metric("Extra Hours", f"{extra:.1f}")

    st.metric("Hypothetical Hours Saved", f"{saved_hours:.1f}")
    st.metric("Hypothetical Cost Saving", f"₹{saved_cost:,.0f}")

    st.caption(
        "Illustrative scenario only. Savings are not verified company results."
    )


# ==================================================
# 14. AI ACTIONS
# ==================================================

elif page == "AI Actions":
    st.header("AI Actions & Approvals")

    st.info(
        "AI execution and approval workflow are not implemented yet. "
        "This module does not perform external actions."
    )


# ==================================================
# 15. EXECUTIVE REPORTS
# ==================================================

elif page == "Executive Reports":
    st.header("Executive Reports")

    report = (
        "NexusOps AI — Operations Summary\n"
        f"Reporting date: {today.date()}\n"
        f"Total tasks: {len(df)}\n"
        f"Overdue tasks: {int(overdue_mask.sum())}\n"
        f"Blocked tasks: {int(blocked_mask.sum())}\n"
        f"Rows requiring review: {quality_count}\n"
        "\nSynthetic demonstration data only.\n"
    )

    st.text_area("Generated Summary", report, height=220)

    st.download_button(
        "Download Executive Summary",
        data=report,
        file_name="nexusops_executive_summary.txt",
        mime="text/plain",
    )
