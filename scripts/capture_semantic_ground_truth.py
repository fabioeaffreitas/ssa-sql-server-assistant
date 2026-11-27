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
import sys
from typing import Any, Callable, Sequence

# When run by file path, Python puts ``scripts`` on sys.path instead of the
# repository root. Keep module execution unchanged while making the documented
# direct command resolve the local package without installation or PYTHONPATH.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    {
        "case_id": "AW2025-007",
        "question": "Show total sales by sales territory.",
        "sql": """SELECT
    st.TerritoryID,
    CASE
        WHEN st.TerritoryID IS NULL THEN N'[Unassigned]'
        ELSE st.Name
    END AS TerritoryName,
    SUM(COALESCE(soh.TotalDue, CONVERT(money, 0))) AS TotalSales
FROM Sales.SalesTerritory AS st
FULL OUTER JOIN Sales.SalesOrderHeader AS soh
    ON soh.TerritoryID = st.TerritoryID
GROUP BY
    st.TerritoryID,
    st.Name
ORDER BY
    CASE WHEN st.TerritoryID IS NULL THEN 1 ELSE 0 END,
    st.TerritoryID ASC;""",
        "ordering_policy": "Deterministic territory ordering; semantic comparison is order-insensitive.",
        "top_n": None,
        "expected_columns": ["TerritoryID", "TerritoryName", "TotalSales"],
        "required_sql_invariants": [
            "table:Sales.SalesOrderHeader",
            "table:Sales.SalesTerritory",
            "aggregate:SUM",
        ],
        "forbidden_sql_invariants": [],
        "order_sensitive": False,
        "numeric_tolerance": None,
        "business_definition": (
            "TotalDue includes tax and freight; all order statuses are included; "
            "territories without orders have zero sales; NULL or unmatched territory "
            "attribution is preserved as [Unassigned] without an invented ID."
        ),
    },
    {
        "case_id": "AW2025-008",
        "question": "Show the number of sales orders per year.",
        "sql": """SELECT
    DATEPART(YEAR, soh.OrderDate) AS OrderYear,
    COUNT(soh.SalesOrderID) AS SalesOrderCount
FROM Sales.SalesOrderHeader AS soh
GROUP BY DATEPART(YEAR, soh.OrderDate)
ORDER BY OrderYear ASC;""",
        "ordering_policy": "Calendar years ascending; semantic comparison is order-insensitive.",
        "top_n": None,
        "expected_columns": ["OrderYear", "SalesOrderCount"],
        "required_sql_invariants": [
            "table:Sales.SalesOrderHeader",
            "aggregate:COUNT",
        ],
        "forbidden_sql_invariants": [],
        "order_sensitive": False,
        "numeric_tolerance": None,
        "business_definition": (
            "Calendar year is derived from OrderDate; SalesOrderID is counted; "
            "all represented years and all order statuses are included."
        ),
    },
    {
        "case_id": "AW2025-010",
        "question": "Show total sales for orders placed in 2024.",
        "sql": """SELECT
    SUM(soh.TotalDue) AS TotalSales
FROM Sales.SalesOrderHeader AS soh
WHERE soh.OrderDate >= CONVERT(datetime, '20240101', 112)
  AND soh.OrderDate < CONVERT(datetime, '20250101', 112);""",
        "ordering_policy": "Single aggregate row; no ordering applies.",
        "top_n": None,
        "expected_columns": ["TotalSales"],
        "expected_row_count": 1,
        "required_sql_invariants": [
            "table:Sales.SalesOrderHeader",
            "aggregate:SUM",
        ],
        "forbidden_sql_invariants": [],
        "order_sensitive": False,
        "numeric_tolerance": None,
        "business_definition": (
            "TotalDue includes tax and freight; OrderDate uses the half-open 2024 "
            "calendar interval; all order statuses are included; no-match results "
            "preserve NULL."
        ),
    },
    {
        "case_id": "AW2025-001",
        "question": "Show the top 10 customers by total sales.",
        "sql": """SELECT TOP (10)
    soh.CustomerID,
    SUM(soh.TotalDue) AS TotalSales
FROM Sales.SalesOrderHeader AS soh
GROUP BY
    soh.CustomerID
ORDER BY
    TotalSales DESC,
    soh.CustomerID ASC;""",
        "ordering_policy": "Deterministic ranked reference result; semantic comparison is order-sensitive.",
        "top_n": 10,
        "expected_columns": ["CustomerID", "TotalSales"],
        "expected_row_count": 10,
        "required_sql_invariants": [
            "table:Sales.SalesOrderHeader", "aggregate:SUM", "top:10"
        ],
        "forbidden_sql_invariants": [],
        "order_sensitive": True,
        "numeric_tolerance": None,
        "business_definition": (
            "Total sales means TotalDue, including tax and freight. Include all "
            "order statuses and customers represented by recorded orders. Customer "
            "identity is CustomerID; customer names are not required. Rank by total "
            "sales descending, with CustomerID ascending as the deterministic tie-breaker."
        ),
    },
    {
        "case_id": "AW2025-009",
        "question": "Show the top 10 products by total quantity sold.",
        "sql": """SELECT TOP (10)
    p.ProductID,
    p.Name,
    SUM(CONVERT(bigint, sod.OrderQty)) AS TotalQuantity
FROM Sales.SalesOrderDetail AS sod
INNER JOIN Production.Product AS p
    ON p.ProductID = sod.ProductID
GROUP BY
    p.ProductID,
    p.Name
ORDER BY
    TotalQuantity DESC,
    p.ProductID ASC;""",
        "ordering_policy": "Deterministic ranked reference result; semantic comparison is order-sensitive.",
        "top_n": 10,
        "expected_columns": ["ProductID", "Name", "TotalQuantity"],
        "expected_row_count": 10,
        "required_sql_invariants": [
            "table:Sales.SalesOrderDetail", "table:Production.Product",
            "aggregate:SUM", "top:10"
        ],
        "forbidden_sql_invariants": [],
        "order_sensitive": True,
        "numeric_tolerance": None,
        "business_definition": (
            "Product identity is ProductID. Include product name. Total quantity is "
            "the sum of recorded OrderQty values across all order-detail rows, "
            "regardless of order status. Convert OrderQty to bigint before aggregation "
            "to avoid the ordinary int aggregation limit. Rank by total quantity "
            "descending, with ProductID ascending as the deterministic tie-breaker."
        ),
    },
    {
        "case_id": "AW2025-011",
        "question": "Show the top 5 products by total sales.",
        "sql": """SELECT TOP (5)
    p.ProductID,
    p.Name,
    SUM(sod.LineTotal) AS TotalSales
FROM Sales.SalesOrderDetail AS sod
INNER JOIN Production.Product AS p
    ON p.ProductID = sod.ProductID
GROUP BY
    p.ProductID,
    p.Name
ORDER BY
    TotalSales DESC,
    p.ProductID ASC;""",
        "ordering_policy": "Deterministic ranked reference result; semantic comparison is order-sensitive.",
        "top_n": 5,
        "expected_columns": ["ProductID", "Name", "TotalSales"],
        "expected_row_count": 5,
        "required_sql_invariants": [
            "table:Sales.SalesOrderDetail", "table:Production.Product",
            "aggregate:SUM", "top:5"
        ],
        "forbidden_sql_invariants": [],
        "order_sensitive": True,
        "numeric_tolerance": None,
        "business_definition": (
            "Product-level sales means the sum of Sales.SalesOrderDetail.LineTotal, "
            "representing recorded line amounts after applicable line discounts. Do "
            "not allocate header-level tax or freight to products. Include all "
            "recorded order-detail rows regardless of order status. Rank by total sales "
            "descending, with ProductID ascending as the deterministic tie-breaker. "
            "The metadata describes the LineTotal expression but does not independently "
            "establish its executable computed-column definition."
        ),
    },
    {
        "case_id": "AW2025-014",
        "question": "Show total sales by month.",
        "sql": """SELECT
    DATEPART(YEAR, soh.OrderDate) AS OrderYear,
    DATEPART(MONTH, soh.OrderDate) AS OrderMonth,
    SUM(soh.TotalDue) AS TotalSales
FROM Sales.SalesOrderHeader AS soh
GROUP BY
    DATEPART(YEAR, soh.OrderDate),
    DATEPART(MONTH, soh.OrderDate)
ORDER BY
    OrderYear ASC,
    OrderMonth ASC;""",
        "ordering_policy": "Deterministic chronological reference result; semantic comparison is order-sensitive.",
        "top_n": None,
        "expected_columns": ["OrderYear", "OrderMonth", "TotalSales"],
        "required_sql_invariants": ["table:Sales.SalesOrderHeader", "aggregate:SUM"],
        "forbidden_sql_invariants": [],
        "order_sensitive": True,
        "numeric_tolerance": None,
        "business_definition": (
            "Group by distinct calendar year and month derived from OrderDate. Total "
            "sales means TotalDue, including tax and freight. Include all order statuses "
            "and represented calendar months only; do not fabricate missing months or "
            "combine the same month across different years. Order chronologically by year "
            "and month."
        ),
    },
)

REFERENCE_CASES_BY_ID = {case["case_id"]: case for case in REFERENCE_CASES}
PILOT_CASE_IDS = ("AW2025-004", "AW2025-012")


def select_reference_cases(case_ids: Sequence[str] | None = None) -> tuple[dict[str, Any], ...]:
    """Select registered cases; omitted selection preserves the two-case pilot."""

    selected_ids = list(PILOT_CASE_IDS if case_ids is None else case_ids)
    if not selected_ids:
        raise GroundTruthCaptureError("At least one reference case must be selected.")
    if len(set(selected_ids)) != len(selected_ids):
        raise GroundTruthCaptureError("Duplicate reference case IDs are not allowed.")
    unknown = [case_id for case_id in selected_ids if case_id not in REFERENCE_CASES_BY_ID]
    if unknown:
        raise GroundTruthCaptureError(f"Unknown reference case ID: {unknown[0]}.")
    return tuple(REFERENCE_CASES_BY_ID[case_id] for case_id in selected_ids)


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
    expected_columns = case.get("expected_columns")
    if expected_columns is not None and columns != expected_columns:
        raise GroundTruthCaptureError(
            f"{case['case_id']} returned unexpected columns."
        )
    expected_row_count = case.get("expected_row_count")
    if expected_row_count is not None and len(rows) != expected_row_count:
        if case["case_id"] == "AW2025-010":
            raise GroundTruthCaptureError(
                "AW2025-010 must return exactly one aggregate row."
            )
        raise GroundTruthCaptureError(
            f"{case['case_id']} returned {len(rows)} rows; expected "
            f"exactly {expected_row_count}."
        )
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
    case_ids: Sequence[str] | None = None,
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

    selected_cases = select_reference_cases(case_ids)
    connection = None
    cursor = None
    try:
        connection = connection_factory()
        cursor = connection.cursor()
        identity = _validate_identity(cursor, configuration)
        captured_timestamp = captured_at or datetime.now(timezone.utc)
        cases = []
        for case in selected_cases:
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
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="Registered case ID; repeat for a batch. Omit for the two-case pilot.",
    )
    args = parser.parse_args()
    try:
        output_path = capture_ground_truth(case_ids=args.case_ids)
    except Exception as exc:
        print(f"Ground-truth capture failed: {type(exc).__name__}.")
        return 1
    print(f"Ground-truth capture written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
