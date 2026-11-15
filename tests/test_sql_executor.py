
import pytest
import pyodbc

import ssa.sql_executor as executor


class FakeCursor:
    description = [("ProductID",)]

    def __init__(self):
        self.timeout = None
        self.closed = False

    def execute(self, sql):
        pass

    def fetchmany(self, size):
        return [(1,), (2,), (3,)][:size]

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self):
        self.cursor_instance = FakeCursor()
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return self.cursor_instance

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


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

    connection = FakeConnection()
    monkeypatch.setattr(executor, "get_connection", lambda: connection)

    result = executor.execute_sql("SELECT ProductID", max_rows=2)

    assert result["row_count"] == 2
    assert result["rows"] == [
        {"ProductID": 1},
        {"ProductID": 2},
    ]
    assert result["truncated"] is True
    assert result["max_rows"] == 2
    assert validation_calls == [("SELECT ProductID", 30)]
    assert connection.timeout == 30
    assert connection.cursor_instance.closed is True
    assert connection.rolled_back is True
    assert connection.closed is True


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


def test_records_validation_and_execution_milliseconds(monkeypatch):
    clock = iter([10.0, 10.2, 20.0, 20.35])
    connection = FakeConnection()

    monkeypatch.setattr(
        executor,
        "validate_sql_schema",
        lambda sql, query_timeout: sql,
    )
    monkeypatch.setattr(executor, "get_connection", lambda: connection)
    monkeypatch.setattr(executor.time, "perf_counter", lambda: next(clock))

    timings = {}
    executor.execute_sql("SELECT ProductID", timings=timings)

    assert timings["validation_ms"] == pytest.approx(200.0)
    assert timings["execution_ms"] == pytest.approx(350.0)


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


def test_timeout_releases_cursor_and_connection(monkeypatch):
    class TimeoutCursor(FakeCursor):
        def execute(self, sql):
            raise pyodbc.OperationalError("HYT00", "Query timeout expired")

    class TimeoutConnection(FakeConnection):
        def __init__(self):
            super().__init__()
            self.cursor_instance = TimeoutCursor()

    connection = TimeoutConnection()
    validation_calls = []
    monkeypatch.setattr(
        executor,
        "validate_sql_schema",
        lambda sql, query_timeout: validation_calls.append(query_timeout) or sql,
    )
    monkeypatch.setattr(executor, "get_connection", lambda: connection)

    with pytest.raises(executor.SQLQueryTimeoutError):
        executor.execute_sql("SELECT ProductID", query_timeout=7)

    assert validation_calls == [7]
    assert connection.timeout == 7
    assert connection.cursor_instance.closed is True
    assert connection.rolled_back is True
    assert connection.closed is True


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
