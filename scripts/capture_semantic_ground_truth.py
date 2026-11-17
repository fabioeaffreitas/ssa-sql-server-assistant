"""Capture reviewed semantic-evaluation reference results from SQL Server.

This script is intentionally separate from the live assistant evaluation. It
executes only the two approved, single-statement SELECT queries and writes a
review artifact; it never calls the assistant, Ollama, or ChromaDB.
"""

from datetime import date, datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
from typing import Any, Callable

from ssa.config_validation import SQLConfiguration, validate_sql_configuration
from ssa.db import get_connection
from ssa.sql_validator import validate_sql


EXPECTED_DATABASE = "AdventureWorks2025"
EXPECTED_LOGIN = "SSA_ReadOnly"
SAFETY_ROW_LIMIT = 1000
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "semantic_ground_truth"
IDENTITY_SQL = (
    "SELECT DB_NAME() AS DatabaseName, "
    "SUSER_SNAME() AS LoginName, "
    "CAST(SERVERPROPERTY('ProductVersion') AS nvarchar(128)) AS SQLServerVersion"
)

REFERENCE_CASES = (
    {
        "case_id": "AW2025-004",
        "question": "Show the 10 most expensive products by list price.",
        "sql": """SELECT TOP (10)
    p.ProductID,
    p.Name,
    p.ListPrice
FROM Production.Product AS p
ORDER BY
    p.ListPrice DESC,
    p.ProductID ASC;""",
        "ordering_policy": "ListPrice descending, ProductID ascending tie-breaker.",
        "top_n": 10,
    },
    {
        "case_id": "AW2025-012",
        "question": "Show customers with more than 10 sales orders.",
        "sql": """SELECT
    c.CustomerID,
    COUNT(soh.SalesOrderID) AS SalesOrderCount
FROM Sales.Customer AS c
INNER JOIN Sales.SalesOrderHeader AS soh
    ON soh.CustomerID = c.CustomerID
GROUP BY
    c.CustomerID
HAVING COUNT(soh.SalesOrderID) > 10
ORDER BY
    COUNT(soh.SalesOrderID) DESC,
    c.CustomerID ASC;""",
        "ordering_policy": "SalesOrderCount descending, CustomerID ascending tie-breaker; semantic comparison may be order-insensitive.",
        "top_n": None,
    },
)


class GroundTruthCaptureError(RuntimeError):
    """Raised when safe, complete ground-truth capture cannot finish."""


def _json_value(value: Any) -> Any:
    """Convert supported SQL values without losing numeric/date precision."""

    if value is None or isinstance(value, (str, int, bool, float)):
        return value
    if isinstance(value, Decimal):
        return {"__type__": "decimal", "value": format(value, "f")}
    if isinstance(value, datetime):
        return {"__type__": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"__type__": "date", "value": value.isoformat()}
    raise GroundTruthCaptureError(
        f"Unsupported result value type: {type(value).__name__}."
    )


def _row_mapping(cursor: Any, row: Any) -> dict[str, Any]:
    columns = [column[0] for column in cursor.description]
    return {column: _json_value(value) for column, value in zip(columns, row)}


def _capture_query(cursor: Any, case: dict[str, Any]) -> dict[str, Any]:
    """Execute one approved query and fail instead of silently truncating."""

    sql = validate_sql(case["sql"])
    cursor.execute(sql)
    rows = cursor.fetchmany(SAFETY_ROW_LIMIT + 1)
    if len(rows) > SAFETY_ROW_LIMIT:
        raise GroundTruthCaptureError(
            f"{case['case_id']} exceeded the capture safety limit of "
            f"{SAFETY_ROW_LIMIT} rows."
        )

    columns = [column[0] for column in cursor.description]
    serialized_rows = [
        [_json_value(value) for value in row]
        for row in rows
    ]
    top_n = case["top_n"]
    return {
        "columns": columns,
        "rows": serialized_rows,
        "row_count": len(serialized_rows),
        "truncated": False,
        "top_n_complete": None if top_n is None else len(serialized_rows) >= top_n,
    }


def _validate_identity(cursor: Any, configuration: SQLConfiguration) -> dict[str, str]:
    """Verify the connected database and login without exposing the login."""

    cursor.execute(validate_sql(IDENTITY_SQL))
    row = cursor.fetchone()
    if row is None:
        raise GroundTruthCaptureError("SQL Server identity check returned no row.")

    identity = _row_mapping(cursor, row)
    database = identity.get("DatabaseName")
    login = identity.get("LoginName")
    version = identity.get("SQLServerVersion")
    if database != EXPECTED_DATABASE:
        raise GroundTruthCaptureError(
            "Connected database is not the required AdventureWorks2025 database."
        )
    if configuration.trusted_connection or login != EXPECTED_LOGIN:
        raise GroundTruthCaptureError(
            "Connected login is not the required read-only SQL authentication account."
        )
    return {
        "database_name": database,
        "sql_server_version": version,
    }


def capture_ground_truth(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    connection_factory: Callable[[], Any] = get_connection,
    configuration_validator: Callable[[], SQLConfiguration] = validate_sql_configuration,
    captured_at: datetime | None = None,
) -> Path:
    """Capture both reference results and return the written artifact path."""

    configuration = configuration_validator()
    if configuration.database != EXPECTED_DATABASE:
        raise GroundTruthCaptureError(
            "SQL_DATABASE must be AdventureWorks2025 for this capture."
        )
    if configuration.trusted_connection:
        raise GroundTruthCaptureError(
            "Ground-truth capture requires SQL authentication with SSA_ReadOnly."
        )
    if os.getenv("SQL_USERNAME", "").strip() != EXPECTED_LOGIN:
        raise GroundTruthCaptureError(
            "SQL_USERNAME must be SSA_ReadOnly for this capture."
        )

    connection = None
    cursor = None
    try:
        connection = connection_factory()
        cursor = connection.cursor()
        identity = _validate_identity(cursor, configuration)
        captured_timestamp = captured_at or datetime.now(timezone.utc)
        cases = []
        for case in REFERENCE_CASES:
            cases.append({
                "case_id": case["case_id"],
                "question": case["question"],
                "reference_sql": case["sql"],
                "ordering_policy": case["ordering_policy"],
                "captured_result": _capture_query(cursor, case),
            })

        output = {
            "capture_schema_version": 1,
            "captured_at_utc": captured_timestamp.astimezone(timezone.utc).isoformat(),
            "database": identity,
            "cases": cases,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"semantic_ground_truth_{captured_timestamp.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}.json"
        output_path = output_dir / filename
        output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return output_path
    except GroundTruthCaptureError:
        raise
    except Exception as exc:
        raise GroundTruthCaptureError(
            "Ground-truth capture failed before a complete artifact was written."
        ) from exc
    finally:
        try:
            if cursor is not None:
                cursor.close()
        finally:
            if connection is not None:
                try:
                    connection.rollback()
                finally:
                    connection.close()


def main() -> int:
    try:
        output_path = capture_ground_truth()
    except Exception as exc:
        print(f"Ground-truth capture failed: {type(exc).__name__}.")
        return 1
    print(f"Ground-truth capture written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
