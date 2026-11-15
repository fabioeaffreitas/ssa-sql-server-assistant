
"""
SSA - SQL Executor
Executes schema-validated SQL and returns query results.
"""

import pyodbc

from ssa.db import DatabaseConnectionError, get_connection, get_query_timeout
from ssa.sql_schema_validator import validate_sql_schema

MAX_ROWS = 100


class SQLExecutionError(RuntimeError):
    """Raised when SQL Server cannot execute an otherwise validated query."""


class SQLQueryTimeoutError(SQLExecutionError):
    """Raised when SQL Server exceeds the configured query timeout."""


class SQLQueryCancelledError(SQLExecutionError):
    """Raised when SQL Server reports a cancelled query."""


class SQLExecutionConnectionError(SQLExecutionError):
    """Raised when execution cannot open its SQL Server connection."""


def _execution_error(exc: Exception) -> SQLExecutionError:
    """Translate driver errors into stable execution-layer exceptions."""

    message = str(exc).lower()
    sql_states = {str(value).upper() for value in getattr(exc, "args", ())}

    if {"HYT00", "HYT01"}.intersection(sql_states) or "timeout" in message:
        return SQLQueryTimeoutError("SQL query timed out.")

    if {"HY008", "57014"}.intersection(sql_states) or any(
        phrase in message for phrase in ("cancelled", "canceled", "canceling")
    ):
        return SQLQueryCancelledError("SQL query was cancelled.")

    return SQLExecutionError(f"SQL execution failed: {exc}")


def execute_sql(
    sql: str,
    max_rows: int = MAX_ROWS,
    query_timeout: int | None = None,
) -> dict:
    """
    Validate SQL against database metadata and execute it.

    Results are limited to max_rows. The executor checks whether
    additional rows exist beyond the returned result set.
    """

    if max_rows < 1:
        raise ValueError("max_rows must be greater than zero.")

    timeout = get_query_timeout(query_timeout)
    validated_sql = validate_sql_schema(sql, query_timeout=timeout)

    try:
        connection = get_connection()
    except DatabaseConnectionError as exc:
        raise SQLExecutionConnectionError(
            "SQL Server connection failed before query execution."
        ) from exc

    cursor = None
    try:
        connection.timeout = timeout
        cursor = connection.cursor()
        try:
            cursor.execute(validated_sql)
            columns = [column[0] for column in cursor.description]

            # Fetch one extra row to determine whether results are truncated.
            rows = cursor.fetchmany(max_rows + 1)
        except pyodbc.Error as exc:
            raise _execution_error(exc) from exc

        truncated = len(rows) > max_rows
        rows = rows[:max_rows]

        results = [
            dict(zip(columns, row))
            for row in rows
        ]

        return {
            "columns": columns,
            "rows": results,
            "row_count": len(results),
            "max_rows": max_rows,
            "truncated": truncated,
        }

    finally:
        try:
            if cursor is not None:
                cursor.close()
        finally:
            try:
                connection.rollback()
            finally:
                connection.close()
