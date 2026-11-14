
import pytest
import pyodbc

import ssa.sql_executor as executor


class FakeCursor:
    description = [("ProductID",)]

    def execute(self, sql):
        pass

    def fetchmany(self, size):
        return [(1,), (2,), (3,)][:size]


class FakeConnection:
    def cursor(self):
        return FakeCursor()

    def rollback(self):
        pass

    def close(self):
        pass


def test_rejects_invalid_max_rows():
    with pytest.raises(ValueError):
        executor.execute_sql("SELECT 1", max_rows=0)


def test_rejects_negative_max_rows():
    with pytest.raises(ValueError):
        executor.execute_sql("SELECT 1", max_rows=-1)


def test_result_truncation(monkeypatch):
    validation_calls = []

    monkeypatch.setattr(
        executor,
        "validate_sql_schema",
        lambda sql, query_timeout: validation_calls.append((sql, query_timeout)) or sql,
    )

    monkeypatch.setattr(
        executor,
        "get_connection",
        lambda: FakeConnection(),
    )

    result = executor.execute_sql("SELECT ProductID", max_rows=2)

    assert result["row_count"] == 2
    assert result["rows"] == [
        {"ProductID": 1},
        {"ProductID": 2},
    ]
    assert result["truncated"] is True
    assert result["max_rows"] == 2
    assert validation_calls == [("SELECT ProductID", 30)]


def test_result_not_truncated(monkeypatch):
    class TwoRowCursor(FakeCursor):
        def fetchmany(self, size):
            return [(1,), (2,)]

    class TwoRowConnection(FakeConnection):
        def cursor(self):
            return TwoRowCursor()

    monkeypatch.setattr(
        executor,
        "validate_sql_schema",
        lambda sql, query_timeout: sql,
    )

    monkeypatch.setattr(
        executor,
        "get_connection",
        lambda: TwoRowConnection(),
    )

    result = executor.execute_sql("SELECT ProductID", max_rows=2)

    assert result["row_count"] == 2
    assert result["truncated"] is False


@pytest.mark.parametrize(
    ("driver_error", "error_type"),
    [
        (pyodbc.OperationalError("HYT00", "Query timeout expired"), executor.SQLQueryTimeoutError),
        (pyodbc.OperationalError("HY008", "Operation canceled"), executor.SQLQueryCancelledError),
        (pyodbc.ProgrammingError("42000", "Syntax error"), executor.SQLExecutionError),
    ],
)
def test_classifies_execution_driver_errors(monkeypatch, driver_error, error_type):
    class FailingCursor(FakeCursor):
        def execute(self, sql):
            raise driver_error

    class FailingConnection(FakeConnection):
        def cursor(self):
            return FailingCursor()

    monkeypatch.setattr(
        executor,
        "validate_sql_schema",
        lambda sql, query_timeout: sql,
    )
    monkeypatch.setattr(executor, "get_connection", lambda: FailingConnection())

    with pytest.raises(error_type):
        executor.execute_sql("SELECT ProductID")


def test_wraps_execution_connection_error(monkeypatch):
    monkeypatch.setattr(
        executor,
        "validate_sql_schema",
        lambda sql, query_timeout: sql,
    )

    def fail_connection():
        raise executor.DatabaseConnectionError("connection unavailable")

    monkeypatch.setattr(executor, "get_connection", fail_connection)

    with pytest.raises(executor.SQLExecutionConnectionError):
        executor.execute_sql("SELECT ProductID")
