
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "tasks.csv"


def test_tasks_csv_exists():
    """The demo task dataset should exist."""
    assert DATA_FILE.exists()


def test_tasks_csv_has_required_columns():
    """The dataset should contain the columns used by NexusOps."""
    df = pd.read_csv(DATA_FILE)

    required_columns = {
        "task_id",
        "department",
        "task_name",
        "owner",
        "status",
        "priority",
        "due_date",
        "estimated_hours",
        "actual_hours",
    }

    assert required_columns.issubset(set(df.columns))


def test_task_ids_are_not_blank():
    """Every task should have a task ID."""
    df = pd.read_csv(DATA_FILE)

    assert df["task_id"].notna().all()
    assert df["task_id"].astype(str).str.strip().ne("").all()


def test_task_ids_are_unique():
    """Each task should have a unique ID."""
    df = pd.read_csv(DATA_FILE)

    assert df["task_id"].is_unique


def test_task_hours_are_non_negative():
    """Estimated and actual hours cannot be negative."""
    df = pd.read_csv(DATA_FILE)

    for column in ["estimated_hours", "actual_hours"]:
        values = pd.to_numeric(df[column], errors="coerce")

        assert values.notna().all()
        assert values.ge(0).all()


def test_task_status_values_are_valid():
    """Tasks should use supported status values."""
    df = pd.read_csv(DATA_FILE)

    allowed_statuses = {
        "Completed",
        "In Progress",
        "Overdue",
        "Blocked",
    }

    assert df["status"].isin(allowed_statuses).all()

import sqlite3


DB_FILE = PROJECT_ROOT / "data" / "nexusops.db"


def test_decision_database_exists():
    """The decision database should exist after using AI Actions."""
    assert DB_FILE.exists()


def test_decision_table_exists():
    """The database should contain the decision audit table."""
    with sqlite3.connect(DB_FILE) as connection:
        result = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'action_decisions'
            """
        ).fetchone()

    assert result is not None


def test_saved_decision_has_required_fields():
    """The test approval should have been recorded."""
    with sqlite3.connect(DB_FILE) as connection:
        row = connection.execute(
            """
            SELECT task_id, decision, decided_by, notes
            FROM action_decisions
            WHERE task_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            ("T004",),
        ).fetchone()

    assert row is not None
    assert row[0] == "T004"
    assert row[1] == "Approved"
    assert row[2] == "Test Manager"
    assert row[3] == "Testing decision logging and audit trail."
