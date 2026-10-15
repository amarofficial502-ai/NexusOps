
from pathlib import Path
import sqlite3
import io
import re
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
import plotly.express as px

from src.security import require_login, logout, has_role

# =========================================================
# 1. CONFIGURATION AND AUTHENTICATION
# =========================================================

st.set_page_config(
    page_title="NexusOps AI",
    page_icon="⚙️",
    layout="wide",
)

# Do not load or display dashboard data before login.
require_login()

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "tasks.csv"
DB_FILE = BASE_DIR / "data" / "nexusops.db"

username = st.session_state.get("username", "Unknown")
role = st.session_state.get("role", "Viewer")

st.sidebar.caption(f"Logged in as: {username}")
st.sidebar.caption(f"Role: {role}")

if st.sidebar.button("Log out", use_container_width=True):
    logout()

st.title("NexusOps AI")
st.subheader("Autonomous Management & Operations Intelligence")
st.caption(
    "Synthetic business data | Transparent rule-based intelligence engine"
)

REQUIRED_COLUMNS = [
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

ALLOWED_STATUSES = {
    "completed",
    "in progress",
    "overdue",
    "blocked",
}

ALLOWED_PRIORITIES = {"high", "medium", "low"}


# =========================================================
# 2. SQLITE DATABASE AND DECISION HISTORY
# =========================================================

def get_connection():
    """Create the local database and approval table if needed."""
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        DB_FILE,
        timeout=10,
    )

    connection.execute("""
        CREATE TABLE IF NOT EXISTS action_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            task_name TEXT NOT NULL,
            owner TEXT,
            recommendation TEXT NOT NULL,
            decision TEXT NOT NULL
                CHECK(decision IN ('Approved', 'Rejected')),
            decided_by TEXT NOT NULL,
            notes TEXT,
            decided_at TEXT NOT NULL,
            UNIQUE(task_id, recommendation)
        )
    """)

    connection.commit()
    return connection


def get_decisions():
    """Read all saved approval and rejection records."""
    with get_connection() as connection:
        return pd.read_sql_query(
            """
            SELECT
                id,
                task_id,
                task_name,
                owner,
                recommendation,
                decision,
                decided_by,
                notes,
                decided_at
            FROM action_decisions
            ORDER BY id DESC
            """,
            connection,
        )


def save_decision(
    task,
    recommendation,
    decision,
    decided_by,
    notes,
):
    """Save a human decision in SQLite."""
    timestamp = (
        datetime.now(timezone.utc)
        .astimezone()
        .isoformat(timespec="seconds")
    )

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO action_decisions (
                task_id,
                task_name,
                owner,
                recommendation,
                decision,
                decided_by,
                notes,
                decided_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(task["task_id"]),
                str(task["task_name"]),
                str(task["owner"]),
                str(recommendation),
                str(decision),
                str(decided_by).strip(),
                str(notes).strip(),
                timestamp,
            ),
        )

        connection.commit()


# =========================================================
# 3. LOAD CSV DATA
# =========================================================

if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE}")
    st.stop()


@st.cache_data
def load_tasks(file_path, modified_time):
    """Load tasks from CSV."""
    return pd.read_csv(file_path)


try:
    df = load_tasks(
        str(DATA_FILE),
        DATA_FILE.stat().st_mtime,
    )
except (
    OSError,
    pd.errors.ParserError,
    UnicodeDecodeError,
) as exc:
    st.error(f"Unable to read dataset: {exc}")
    st.stop()


missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()


# =========================================================
# 4. DATA NORMALIZATION
# =========================================================

df["due_date"] = pd.to_datetime(
    df["due_date"],
    errors="coerce",
).dt.normalize()

for column in ["estimated_hours", "actual_hours"]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )

for column in [
    "task_id",
    "department",
    "task_name",
    "owner",
    "status",
    "priority",
]:
    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
    )


status_norm = df["status"].str.casefold()
priority_norm = df["priority"].str.casefold()

today = pd.Timestamp.now().normalize()

completed_mask = (
    status_norm.eq("completed").fillna(False)
)

blocked_mask = (
    status_norm.eq("blocked").fillna(False)
)

overdue_mask = (
    df["due_date"].notna()
    & df["due_date"].lt(today)
    & ~completed_mask
)


# =========================================================
# 5. DATA QUALITY CHECKS
# =========================================================

missing_mask = df[REQUIRED_COLUMNS].isna().any(axis=1)

blank_required_mask = pd.Series(
    False,
    index=df.index,
)

for column in [
    "task_id",
    "department",
    "task_name",
    "owner",
    "status",
    "priority",
]:
    blank_required_mask |= (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )


task_ids = df["task_id"].astype("string").str.strip()

blank_id_mask = (
    task_ids.isna()
    | task_ids.eq("").fillna(True)
)

duplicate_id_mask = (
    task_ids.duplicated(keep=False)
    & ~blank_id_mask
)

invalid_dates_mask = df["due_date"].isna()

hours_df = df[
    ["estimated_hours", "actual_hours"]
]

invalid_hours_mask = (
    hours_df.isna().any(axis=1)
    | hours_df.lt(0).any(axis=1)
)

invalid_status_mask = (
    status_norm.isna()
    | ~status_norm.isin(ALLOWED_STATUSES)
)

invalid_priority_mask = (
    priority_norm.isna()
    | ~priority_norm.isin(ALLOWED_PRIORITIES)
)

quality_mask = (
    missing_mask
    | blank_required_mask
    | blank_id_mask
    | duplicate_id_mask
    | invalid_dates_mask
    | invalid_hours_mask
    | invalid_status_mask
    | invalid_priority_mask
)

quality_details = df.loc[quality_mask].copy()
quality_count = int(quality_mask.sum())


# =========================================================
# 6. TASK INTELLIGENCE ENGINE
# =========================================================

# Transparent business rules, not trained ML predictions.

priority_points = (
    priority_norm
    .map({
        "high": 50,
        "medium": 30,
        "low": 10,
    })
    .fillna(0)
)

days_remaining = (
    df["due_date"] - today
).dt.days

deadline_points = pd.Series(
    0,
    index=df.index,
    dtype="int64",
)

deadline_points.loc[
    days_remaining.le(1).fillna(False)
] = 30

deadline_points.loc[
    days_remaining.between(2, 3).fillna(False)
] = 20

deadline_points.loc[
    days_remaining.between(4, 7).fillna(False)
] = 10

blocked_points = blocked_mask.astype(int) * 20

df["priority_score"] = (
    priority_points
    + deadline_points
    + blocked_points
).clip(upper=100).astype(int)

df.loc[
    completed_mask,
    "priority_score",
] = 0

df["days_remaining"] = days_remaining

df["sla_risk"] = "On Track"

df.loc[
    days_remaining.between(0, 2).fillna(False)
    & ~completed_mask,
    "sla_risk",
] = "At Risk"

df.loc[
    overdue_mask,
    "sla_risk",
] = "Overdue"

df.loc[
    completed_mask,
    "sla_risk",
] = "Completed"

df.loc[
    df["due_date"].isna(),
    "sla_risk",
] = "Unknown"

df.loc[
    blocked_mask & ~overdue_mask & ~completed_mask,
    "sla_risk",
] = "Blocked"


def recommend_action(row):
    """Suggest an action using transparent business rules."""

    if pd.isna(row["due_date"]):
        return "Validate the due date"

    task_status = str(row["status"]).casefold()

    if task_status == "completed":
        return "No action required"

    if task_status == "blocked":
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


# =========================================================
# 7. NAVIGATION
# =========================================================

all_pages = [
    "Dashboard",
    "Task Intelligence",
    "Workload Intelligence",
    "Data Quality",
    "Smart Data Cleaner",
    "Root Cause Analysis",
    "Cost Simulator",
    "AI Actions",
    "Executive Reports",
]

# Only Admin and Manager can record decisions.
# Viewers can inspect dashboard information.
if role == "Viewer":
    pages = [
        page for page in all_pages
        if page != "AI Actions"
    ]
else:
    pages = all_pages

page = st.sidebar.radio(
    "Navigation",
    pages,
)

st.sidebar.caption(
    f"Reporting date: {today.date()}"
)


# =========================================================
# 8. DASHBOARD
# =========================================================

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

    open_tasks = (
        df.loc[~completed_mask]
        .sort_values("priority_score", ascending=False)
    )

    display_columns = [
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

    st.dataframe(
        open_tasks[display_columns].head(10),
        use_container_width=True,
        hide_index=True,
    )

# =========================================================
# 9. TASK INTELLIGENCE
# =========================================================

elif page == "Task Intelligence":

    st.header("Task Intelligence & SLA Risk")

    filter1, filter2 = st.columns(2)

    risk_options = sorted(
        df["sla_risk"].dropna().unique().tolist()
    )

    priority_options = sorted(
        df["priority"].dropna().unique().tolist()
    )

    with filter1:
        selected_risks = st.multiselect(
            "Filter by SLA Risk",
            risk_options,
            default=risk_options,
        )

    with filter2:
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

    b.metric(
        "At Risk",
        int(intelligence_df["sla_risk"].eq("At Risk").sum()),
    )

    c.metric(
        "Overdue",
        int(intelligence_df["sla_risk"].eq("Overdue").sum()),
    )

    st.subheader("Ranked Tasks")

    intelligence_df = intelligence_df.sort_values(
        "priority_score",
        ascending=False,
    )

    columns = [
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

    st.dataframe(
        intelligence_df[columns],
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Priority scores use predefined rules, not trained ML predictions."
    )


# =========================================================
# 10. WORKLOAD INTELLIGENCE
# =========================================================

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
                blocked_tasks=(
                    "status",
                    lambda values:
                        values.str.casefold().eq("blocked").sum(),
                ),
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

        st.subheader("Team Workload Summary")

        st.dataframe(
            workload,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Workload volume indicates task allocation, "
            "not employee performance or productivity."
        )


# =========================================================
# 11. DATA QUALITY
# =========================================================

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
        st.warning(
            "Rows with validation issues are listed below."
        )

        st.dataframe(
            quality_details,
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# 12. SMART DATA CLEANER
# =========================================================

elif page == "Smart Data Cleaner":

    st.header("Smart Data Cleaner")

    st.write(
        "Upload CSV or Excel, inspect data quality, clean a copy, "
        "and download the result. Original file is not overwritten."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        key="smart_cleaner_upload",
    )

    if uploaded_file is not None:

        if uploaded_file.size > 50 * 1024 * 1024:
            st.error("File exceeds 50 MB. Upload a smaller file.")
            st.stop()

        try:
            file_bytes = uploaded_file.getvalue()

            if uploaded_file.name.lower().endswith(".csv"):
                try:
                    original_df = pd.read_csv(
                        io.BytesIO(file_bytes)
                    )
                except UnicodeDecodeError:
                    original_df = pd.read_csv(
                        io.BytesIO(file_bytes),
                        encoding="latin-1",
                    )
            else:
                original_df = pd.read_excel(
                    io.BytesIO(file_bytes)
                )

        except Exception as exc:
            st.error(f"Unable to read file: {exc}")
            st.stop()

        if len(original_df.columns) == 0:
            st.error("The file contains no readable columns.")
            st.stop()

        st.subheader("Original Data Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Rows", len(original_df))
        c2.metric("Columns", len(original_df.columns))
        c3.metric(
            "Missing Cells",
            int(original_df.isna().sum().sum()),
        )
        c4.metric(
            "Duplicate Rows",
            int(original_df.duplicated().sum()),
        )

        st.dataframe(
            original_df.head(100),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Choose Cleaning Rules")

        c1, c2 = st.columns(2)

        with c1:
            trim_spaces = st.checkbox(
                "Trim spaces in text",
                value=True,
            )

            normalize_headers = st.checkbox(
                "Normalize column names",
                value=True,
            )

            remove_empty_rows = st.checkbox(
                "Remove completely empty rows",
                value=True,
            )

            remove_empty_columns = st.checkbox(
                "Remove completely empty columns",
                value=True,
            )

        with c2:
            remove_duplicates = st.checkbox(
                "Remove duplicate rows",
                value=True,
            )

            missing_strategy = st.selectbox(
                "Handle missing values",
                [
                    "Keep missing values",
                    "Drop rows with missing values",
                    "Fill text with Unknown and numbers with median",
                ],
            )

        if st.button("Clean Dataset", type="primary"):

            cleaned_df = original_df.copy()
            changes = []

            if normalize_headers:
                names = []
                seen = {}

                for column in cleaned_df.columns:

                    name = re.sub(
                        r"[^0-9a-zA-Z]+",
                        "_",
                        str(column).strip().lower(),
                    ).strip("_") or "column"

                    seen[name] = seen.get(name, 0) + 1

                    if seen[name] > 1:
                        name = f"{name}_{seen[name]}"

                    names.append(name)

                cleaned_df.columns = names
                changes.append("Normalized column names")

            if trim_spaces:

                for column in cleaned_df.select_dtypes(
                    include=["object", "string"]
                ).columns:

                    cleaned_df[column] = cleaned_df[column].map(
                        lambda value: (
                            value.strip()
                            if isinstance(value, str)
                            else value
                        )
                    )

                changes.append("Trimmed text spaces")

            if remove_empty_columns:

                before = len(cleaned_df.columns)

                cleaned_df = cleaned_df.dropna(
                    axis=1,
                    how="all",
                )

                removed = before - len(cleaned_df.columns)

                if removed:
                    changes.append(
                        f"Removed {removed} empty columns"
                    )

            if remove_empty_rows:

                before = len(cleaned_df)

                cleaned_df = cleaned_df.dropna(
                    axis=0,
                    how="all",
                )

                removed = before - len(cleaned_df)

                if removed:
                    changes.append(
                        f"Removed {removed} empty rows"
                    )

            if remove_duplicates:

                before = len(cleaned_df)
                cleaned_df = cleaned_df.drop_duplicates()
                removed = before - len(cleaned_df)

                if removed:
                    changes.append(
                        f"Removed {removed} duplicate rows"
                    )

            if missing_strategy == "Drop rows with missing values":

                before = len(cleaned_df)
                cleaned_df = cleaned_df.dropna()
                removed = before - len(cleaned_df)

                changes.append(
                    f"Removed {removed} rows with missing values"
                )

            elif missing_strategy.startswith("Fill text"):

                for column in cleaned_df.columns:

                    if pd.api.types.is_numeric_dtype(
                        cleaned_df[column]
                    ):

                        median = cleaned_df[column].median()

                        if pd.notna(median):
                            cleaned_df[column] = (
                                cleaned_df[column].fillna(median)
                            )

                    else:
                        cleaned_df[column] = (
                            cleaned_df[column].fillna("Unknown")
                        )

                changes.append("Filled missing values")

            st.session_state["cleaned_dataset"] = cleaned_df
            st.session_state["cleaning_changes"] = changes
            st.session_state["cleaning_source"] = uploaded_file.name

            st.session_state["cleaning_before_shape"] = (
                len(original_df),
                len(original_df.columns),
            )

        if (
            "cleaned_dataset" in st.session_state
            and st.session_state.get("cleaning_source")
            == uploaded_file.name
        ):

            cleaned_df = st.session_state["cleaned_dataset"]

            changes = st.session_state.get(
                "cleaning_changes",
                [],
            )

            st.subheader("Cleaning Report")

            before_rows, before_cols = st.session_state[
                "cleaning_before_shape"
            ]

            after_rows, after_cols = cleaned_df.shape

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Rows",
                f"{before_rows} → {after_rows}",
            )

            c2.metric(
                "Columns",
                f"{before_cols} → {after_cols}",
            )

            c3.metric(
                "Missing Cells After",
                int(cleaned_df.isna().sum().sum()),
            )

            c4.metric(
                "Duplicates After",
                int(cleaned_df.duplicated().sum()),
            )

            if changes:
                for change in changes:
                    st.write(f"- {change}")
            else:
                st.info("No changes were required.")

            st.subheader("Cleaned Data Preview")

            st.dataframe(
                cleaned_df.head(100),
                use_container_width=True,
                hide_index=True,
            )

            csv_data = cleaned_df.to_csv(
                index=False,
            ).encode("utf-8-sig")

            base_name = Path(uploaded_file.name).stem

            st.download_button(
                "Download Cleaned CSV",
                data=csv_data,
                file_name=f"{base_name}_cleaned.csv",
                mime="text/csv",
            )

            try:
                excel_buffer = io.BytesIO()

                with pd.ExcelWriter(
                    excel_buffer,
                    engine="openpyxl",
                ) as writer:

                    cleaned_df.to_excel(
                        writer,
                        index=False,
                        sheet_name="Cleaned Data",
                    )

                st.download_button(
                    "Download Cleaned Excel",
                    data=excel_buffer.getvalue(),
                    file_name=f"{base_name}_cleaned.xlsx",
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                )

            except ImportError:
                st.warning(
                    "Install openpyxl to enable Excel download."
                )

            st.caption(
                "The original uploaded file is not overwritten. "
                "Review cleaned data before using it."
            )

# =========================================================
# 13. ROOT CAUSE ANALYSIS
# =========================================================

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

        st.subheader("Overdue Task Details")

        st.dataframe(
            overdue_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Department-level counts are investigation signals, "
            "not proof of the actual root cause."
        )


# =========================================================
# 14. COST SIMULATOR
# =========================================================

elif page == "Cost Simulator":

    st.header("Cost & What-if Simulator")

    valid_hours = df[
        ["estimated_hours", "actual_hours"]
    ].dropna()

    valid_hours = valid_hours.loc[
        (valid_hours["estimated_hours"] >= 0)
        & (valid_hours["actual_hours"] >= 0)
    ]

    estimated = float(
        valid_hours["estimated_hours"].sum()
    )

    actual = float(
        valid_hours["actual_hours"].sum()
    )

    extra = max(actual - estimated, 0)

    hourly_rate = st.number_input(
        "Assumed cost per hour (₹)",
        min_value=0,
        value=500,
        step=50,
    )

    reduction = st.slider(
        "Hypothetical rework reduction (%)",
        min_value=0,
        max_value=100,
        value=20,
    )

    saved_hours = extra * reduction / 100
    saved_cost = saved_hours * hourly_rate

    c1, c2, c3 = st.columns(3)

    c1.metric("Estimated Hours", f"{estimated:.1f}")
    c2.metric("Actual Hours", f"{actual:.1f}")
    c3.metric("Extra Hours", f"{extra:.1f}")

    st.metric(
        "Hypothetical Hours Saved",
        f"{saved_hours:.1f}",
    )

    st.metric(
        "Hypothetical Cost Saving",
        f"₹{saved_cost:,.0f}",
    )

    st.caption(
        "Illustrative scenario only. Savings are not verified company results."
    )

# =========================================================
# 15. AI ACTIONS AND HUMAN APPROVAL
# =========================================================

elif page == "AI Actions":

    # Enforce permission again at the action boundary.
    if not has_role("Admin", "Manager"):
        st.error("You do not have permission to review or record decisions.")
        st.stop()

    st.header("AI Actions & Human Approval")

    st.info(
        "Recommendations use transparent business rules. "
        "Approving or rejecting records a decision only; "
        "no external action is executed."
    )

    open_df = df.loc[~completed_mask].copy()

    open_df = open_df.sort_values(
        ["priority_score", "days_remaining"],
        ascending=[False, True],
        na_position="last",
    )

    try:
        saved = get_decisions()

    except sqlite3.Error as exc:
        st.error(f"Unable to load decision history: {exc}")
        st.stop()

    saved_keys = set()

    if not saved.empty:
        saved_keys = set(
            zip(
                saved["task_id"].astype(str),
                saved["recommendation"].astype(str),
            )
        )

    if open_df.empty:
        st.success("No open tasks need action.")

    else:
        st.subheader("Recommended Actions")

        action_rows = []

        for _, task in open_df.iterrows():
            task_id = str(task["task_id"])
            recommendation = str(task["recommended_action"])

            action_rows.append({
                "Task ID": task_id,
                "Department": str(task["department"]),
                "Task": str(task["task_name"]),
                "Owner": str(task["owner"]),
                "Status": str(task["status"]),
                "Priority": str(task["priority"]),
                "SLA Risk": str(task["sla_risk"]),
                "Score": int(task["priority_score"]),
                "Recommended Action": recommendation,
                "Decision Recorded": (
                    "Yes"
                    if (task_id, recommendation) in saved_keys
                    else "No"
                ),
            })

        st.dataframe(
            pd.DataFrame(action_rows),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Review a Recommendation")

        task_options = [
            (
                f'{row["task_id"]} — '
                f'{row["task_name"]} '
                f'({row["owner"]})'
            )
            for _, row in open_df.iterrows()
        ]

        selected_label = st.selectbox(
            "Select task",
            task_options,
        )

        selected_index = task_options.index(selected_label)
        selected_task = open_df.iloc[selected_index]

        selected_task_id = str(selected_task["task_id"])
        recommendation = str(selected_task["recommended_action"])

        st.write(f"**Recommendation:** {recommendation}")

        st.write(
            f'**Reason:** status is "{selected_task["status"]}", '
            f'SLA risk is "{selected_task["sla_risk"]}", and '
            f'the rule-based priority score is '
            f'{int(selected_task["priority_score"])}/100.'
        )

        already_recorded = (
            selected_task_id,
            recommendation,
        ) in saved_keys

        if already_recorded:
            st.warning(
                "A decision for this task and recommendation "
                "has already been recorded."
            )

        with st.form("action_decision_form"):

            # The identity comes from the authenticated session.
            decided_by = st.text_input(
                "Authenticated Reviewer",
                value=username,
                disabled=True,
            )

            decision = st.radio(
                "Decision",
                ["Approved", "Rejected"],
                horizontal=True,
            )

            notes = st.text_area(
                "Decision Notes (Optional)",
                placeholder="Reason or follow-up notes",
            )

            submitted = st.form_submit_button(
                "Save Decision",
                type="primary",
            )

            if submitted:

                if already_recorded:
                    st.error(
                        "This task recommendation already has a saved decision."
                    )

                else:
                    try:
                        save_decision(
                            selected_task,
                            recommendation,
                            decision,
                            username,
                            notes,
                        )

                        st.success(
                            f"{decision} decision saved "
                            f"for task {selected_task_id}."
                        )

                        st.rerun()

                    except sqlite3.IntegrityError:
                        st.error(
                            "A decision for this task and recommendation "
                            "already exists."
                        )

                    except sqlite3.Error as exc:
                        st.error(f"Could not save decision: {exc}")

    st.divider()
    st.subheader("Decision History / Audit Trail")

    try:
        history = get_decisions()

        if history.empty:
            st.caption(
                "No decisions recorded yet. "
                "Submit an approval or rejection above."
            )

        else:
            st.dataframe(
                history,
                use_container_width=True,
                hide_index=True,
            )

            csv_data = history.to_csv(
                index=False,
            ).encode("utf-8")

            st.download_button(
                "Download Decision History (CSV)",
                data=csv_data,
                file_name="nexusops_decision_history.csv",
                mime="text/csv",
            )

    except sqlite3.Error as exc:
        st.error(
            f"Unable to read decision history: {exc}"
        )

# =========================================================
# 16. EXECUTIVE REPORTS
# =========================================================

elif page == "Executive Reports":

    st.header("Executive Reports")

    try:
        decision_history = get_decisions()

        if decision_history.empty:
            approved_count = 0
            rejected_count = 0

        else:
            approved_count = int(
                decision_history["decision"]
                .eq("Approved")
                .sum()
            )

            rejected_count = int(
                decision_history["decision"]
                .eq("Rejected")
                .sum()
            )

    except sqlite3.Error:
        approved_count = 0
        rejected_count = 0

    report = (
        "NexusOps AI — Operations Summary\n"
        f"Reporting date: {today.date()}\n"
        f"Total tasks: {len(df)}\n"
        f"Overdue tasks: {int(overdue_mask.sum())}\n"
        f"Blocked tasks: {int(blocked_mask.sum())}\n"
        f"Rows requiring review: {quality_count}\n"
        f"Approved recommendations recorded: {approved_count}\n"
        f"Rejected recommendations recorded: {rejected_count}\n"
        "\n"
        "Synthetic demonstration data only. "
        "Recommendations and cost estimates are illustrative.\n"
    )

    st.text_area(
        "Generated Summary",
        report,
        height=260,
    )

    st.download_button(
        "Download Executive Summary",
        data=report,
        file_name="nexusops_executive_summary.txt",
        mime="text/plain",
    )
