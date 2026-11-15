"""Integration coverage for SQL Server query-timeout enforcement."""

from time import monotonic

import pytest

import ssa.sql_executor as executor
from ssa.sql_executor import SQLQueryTimeoutError, execute_sql


pytestmark = pytest.mark.integration


def test_query_timeout_interrupts_execution_and_server_remains_usable(monkeypatch):
    """A one-second ODBC timeout interrupts a controlled five-second delay."""

    timeout_query = "WAITFOR DELAY '00:00:05'; SELECT 1 AS HealthCheck"

    # This test-only boundary bypass is required because production validation
    # intentionally permits SELECT statements only and rejects WAITFOR.
    monkeypatch.setattr(
        executor,
        "validate_sql_schema",
        lambda sql, query_timeout: timeout_query,
    )

    started_at = monotonic()

    with pytest.raises(SQLQueryTimeoutError):
        execute_sql("SELECT 1 AS HealthCheck", query_timeout=1)

    assert monotonic() - started_at < 10

    # Restore real schema validation before checking database health.
    monkeypatch.undo()

    assert execute_sql("SELECT 1 AS HealthCheck", query_timeout=5)["rows"] == [
        {"HealthCheck": 1}
    ]
