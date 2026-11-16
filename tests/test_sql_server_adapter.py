from unittest.mock import Mock

import pytest

import ssa.sql_server_adapter as adapter_module
from ssa.contracts import (
    QueryResult,
    RelationalDatabaseAdapter,
    ValidatedQuery,
)
from ssa.db import DatabaseConnectionError
from ssa.sql_executor import (
    SQLExecutionError,
    SQLQueryCancelledError,
    SQLQueryTimeoutError,
)
from ssa.sql_validator import SQLValidationError


def test_adapter_matches_relational_database_protocol():
    adapter: RelationalDatabaseAdapter = adapter_module.SQLServerDatabaseAdapter()

    assert callable(adapter.validate_read_query)
    assert callable(adapter.execute_read)


def test_validate_read_query_returns_validated_query_and_forwards_query(monkeypatch):
    validate = Mock(return_value="SELECT 1")
    monkeypatch.setattr(adapter_module, "validate_sql_schema", validate)
    adapter = adapter_module.SQLServerDatabaseAdapter()

    result = adapter.validate_read_query("SELECT 1", timeout=7)

    assert result == ValidatedQuery("SELECT 1")
    validate.assert_called_once_with("SELECT 1", query_timeout=7)


def test_validate_read_query_forwards_default_timeout(monkeypatch):
    validate = Mock(return_value="SELECT 1")
    monkeypatch.setattr(adapter_module, "validate_sql_schema", validate)

    adapter_module.SQLServerDatabaseAdapter().validate_read_query("SELECT 1")

    validate.assert_called_once_with("SELECT 1", query_timeout=None)


def test_validation_errors_propagate_unchanged(monkeypatch):
    error = SQLValidationError("invalid query")
    monkeypatch.setattr(
        adapter_module,
        "validate_sql_schema",
        Mock(side_effect=error),
    )

    with pytest.raises(SQLValidationError) as raised:
        adapter_module.SQLServerDatabaseAdapter().validate_read_query("DELETE")

    assert raised.value is error


def test_database_connection_errors_propagate_unchanged(monkeypatch):
    error = DatabaseConnectionError("connection unavailable")
    monkeypatch.setattr(
        adapter_module,
        "validate_sql_schema",
        Mock(side_effect=error),
    )

    with pytest.raises(DatabaseConnectionError) as raised:
        adapter_module.SQLServerDatabaseAdapter().validate_read_query("SELECT 1")

    assert raised.value is error


def test_execute_read_converts_all_result_fields_and_preserves_order(monkeypatch):
    legacy_result = {
        "columns": ["id", "name"],
        "rows": [{"id": 2, "name": "second"}, {"id": 1, "name": "first"}],
        "row_count": 2,
        "max_rows": 2,
        "truncated": True,
    }
    execute = Mock(return_value=legacy_result)
    monkeypatch.setattr(adapter_module, "execute_sql", execute)
    query = ValidatedQuery("SELECT id, name FROM items")

    result = adapter_module.SQLServerDatabaseAdapter().execute_read(
        query,
        max_rows=2,
        timeout=9,
    )

    assert result == QueryResult(**legacy_result)
    assert result.rows == legacy_result["rows"]
    execute.assert_called_once_with(
        query.sql,
        max_rows=2,
        query_timeout=9,
    )


def test_execute_read_forwards_default_max_rows_and_timeout(monkeypatch):
    execute = Mock(return_value={
        "columns": [],
        "rows": [],
        "row_count": 0,
        "max_rows": 100,
        "truncated": False,
    })
    monkeypatch.setattr(adapter_module, "execute_sql", execute)

    adapter_module.SQLServerDatabaseAdapter().execute_read(ValidatedQuery("SELECT 1"))

    execute.assert_called_once_with(
        "SELECT 1",
        max_rows=100,
        query_timeout=None,
    )


@pytest.mark.parametrize(
    "error",
    [
        SQLQueryTimeoutError("timed out"),
        SQLQueryCancelledError("cancelled"),
        SQLExecutionError("execution failed"),
    ],
)
def test_execution_errors_propagate_unchanged(monkeypatch, error):
    monkeypatch.setattr(
        adapter_module,
        "execute_sql",
        Mock(side_effect=error),
    )

    with pytest.raises(type(error)) as raised:
        adapter_module.SQLServerDatabaseAdapter().execute_read(
            ValidatedQuery("SELECT 1")
        )

    assert raised.value is error


def test_execution_delegates_to_executor_not_pyodbc(monkeypatch):
    execute = Mock(return_value={
        "columns": ["value"],
        "rows": [{"value": 1}],
        "row_count": 1,
        "max_rows": 100,
        "truncated": False,
    })
    monkeypatch.setattr(adapter_module, "execute_sql", execute)

    adapter_module.SQLServerDatabaseAdapter().execute_read(
        ValidatedQuery("SELECT 1")
    )

    execute.assert_called_once()
    assert not hasattr(adapter_module, "pyodbc")


def test_validation_then_execution_keeps_executor_validation_boundary(monkeypatch):
    validate = Mock(return_value="SELECT 1")
    execute = Mock(return_value={
        "columns": ["value"],
        "rows": [{"value": 1}],
        "row_count": 1,
        "max_rows": 100,
        "truncated": False,
    })
    monkeypatch.setattr(adapter_module, "validate_sql_schema", validate)
    monkeypatch.setattr(adapter_module, "execute_sql", execute)
    adapter = adapter_module.SQLServerDatabaseAdapter()

    validated = adapter.validate_read_query("SELECT 1", timeout=5)
    adapter.execute_read(validated, timeout=5)

    validate.assert_called_once_with("SELECT 1", query_timeout=5)
    execute.assert_called_once_with(
        "SELECT 1",
        max_rows=100,
        query_timeout=5,
    )
