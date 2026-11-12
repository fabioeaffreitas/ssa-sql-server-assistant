
import pytest

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
    monkeypatch.setattr(
        executor,
        "validate_sql_schema",
        lambda sql: sql,
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
        lambda sql: sql,
    )

    monkeypatch.setattr(
        executor,
        "get_connection",
        lambda: TwoRowConnection(),
    )

    result = executor.execute_sql("SELECT ProductID", max_rows=2)

    assert result["row_count"] == 2
    assert result["truncated"] is False
