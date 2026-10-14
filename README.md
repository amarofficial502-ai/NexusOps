
# NexusOps AI — Operations Intelligence Dashboard

NexusOps AI is a Streamlit-based operations intelligence demo that helps users review task status, identify operational risks, understand workload distribution, and prepare management summaries.

> **Disclaimer:** This project uses synthetic demonstration data and transparent rule-based recommendations. It is not an official company system, and approving or rejecting a recommendation does not execute an external business action.

## Features

1. **Dashboard** — Overview of tasks, overdue items, blocked tasks, and data quality.
2. **Task Intelligence** — Task-level risk scores and rule-based recommendations.
3. **Workload Intelligence** — Workload distribution across owners and departments.
4. **Data Quality** — Checks for missing or invalid task information.
5. **Root Cause Analysis** — Highlights patterns associated with delays and blockers.
6. **Cost Simulator** — Illustrative estimates based on configurable assumptions.
7. **AI Actions & Human Approval** — Records a reviewer's decision and optional notes.
8. **Executive Reports** — Generates a management summary for download.

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- SQLite
- Pytest

## Project Structure

```text
NexusOps/
├── app.py
├── app_new.py
├── data/
│   └── tasks.csv
├── src/
│   └── tests/
│       └── test_nexusops.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.13 or a compatible Python 3 version
- pip

## Setup

Open a terminal in the project directory.

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install pytest
```

### 3. Run the application

The current development version is `app_new.py`:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app_new.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

## Run Tests

```powershell
.\.venv\Scripts\python.exe -m pytest src\tests\test_nexusops.py -v
```

The tests validate the task dataset and the local decision database. Tests that inspect previously saved decisions depend on the local demo database state.

## Data and Limitations

- The included task records are synthetic demonstration data.
- Risk scores and recommendations are generated using business rules; they are not evidence of a trained machine-learning model.
- Cost estimates depend on illustrative assumptions and should not be treated as actual savings.
- Reviewer names are entered by the user and are not authenticated identities.
- The local SQLite database is intended for demonstration and development, not as a hardened production audit system.
- Before public deployment, add appropriate authentication, authorization, input validation, database persistence, and security controls.

## Future Improvements

- Automated test coverage for business logic and edge cases
- Authentication and role-based permissions
- Improved audit controls
- Production database integration
- Secure online deployment
- CI checks and dependency security scanning

## Author

Developed as a portfolio project to demonstrate operations reporting, data validation, rule-based decision support, and dashboard development.
