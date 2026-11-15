import pytest

import ssa.sql_schema_validator as validator


class FakeCursor:
    def __init__(self, error=None):
        self.error = error
        self.executed = []
        self.closed = False

    def execute(self, sql):
        self.executed.append(sql)
        if self.error:
            raise self.error

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, cursor_error=None):
        self.cursor_instance = FakeCursor(cursor_error)
        self.timeout = None
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return self.cursor_instance

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def test_schema_validation_sets_timeout_and_closes_resources(monkeypatch):
    connection = FakeConnection()
    monkeypatch.setattr(validator, "get_connection", lambda: connection)

    sql = "SELECT 1"
    assert validator.validate_sql_schema(sql, query_timeout=7) == sql

    assert connection.timeout == 7
    assert connection.cursor_instance.closed is True
    assert connection.rolled_back is True
    assert connection.closed is True


def test_cursor_open_failure_is_wrapped_without_masking_cleanup(monkeypatch):
    connection = FakeConnection()
    connection.cursor = lambda: (_ for _ in ()).throw(RuntimeError("cursor failed"))
    monkeypatch.setattr(validator, "get_connection", lambda: connection)

    with pytest.raises(
        validator.SQLValidationError,
        match="SQL schema validation failed",
    ) as error:
        validator.validate_sql_schema("SELECT 1")

    assert "cursor failed" not in str(error.value)
    assert connection.rolled_back is True
    assert connection.closed is True
