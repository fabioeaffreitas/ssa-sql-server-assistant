
"""
SSA - SQL Executor
Executes schema-validated SQL and returns query results.
"""

import time

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


def _add_timing(timings: dict | None, key: str, started_at: float) -> None:
    """Accumulate one completed timing segment in milliseconds."""

    if timings is None:
        return
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    timings[key] = float(timings.get(key) or 0) + elapsed_ms


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

    return SQLExecutionError("SQL execution failed.")


def execute_sql(
    sql: str,
    max_rows: int = MAX_ROWS,
    query_timeout: int | None = None,
    timings: dict | None = None,
) -> dict:
    """
    Validate SQL against database metadata and execute it.

    Results are limited to max_rows. The executor checks whether
    additional rows exist beyond the returned result set.
    """

    if max_rows < 1:
        raise ValueError("max_rows must be greater than zero.")

    timeout = get_query_timeout(query_timeout)
    validation_started = time.perf_counter()
    try:
        validated_sql = validate_sql_schema(sql, query_timeout=timeout)
    finally:
        _add_timing(timings, "validation_ms", validation_started)

    execution_started = time.perf_counter()
    connection = None
    cursor = None
    try:
        try:
            connection = get_connection()
        except DatabaseConnectionError as exc:
            raise SQLExecutionConnectionError(
                "SQL Server connection failed before query execution."
            ) from exc

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
            if connection is not None:
                try:
                    connection.rollback()
                finally:
                    connection.close()
            _add_timing(timings, "execution_ms", execution_started)
