from datetime import date, datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import scripts.capture_semantic_ground_truth as capture
from ssa.config_validation import SQLConfiguration
from ssa.sql_validator import validate_sql


def configuration(database="AdventureWorks2025", trusted_connection=False):
    return SQLConfiguration(
        server="test-server",
        database=database,
        driver="ODBC Driver 18 for SQL Server",
        trusted_connection=trusted_connection,
        encrypt=True,
        trust_server_certificate=True,
        connection_timeout=10,
        query_timeout=30,
    )


class FakeCursor:
    def __init__(self, result_rows=None, identity=None, failure=None):
        self.result_rows = list(result_rows or [
            [(1, "Widget", Decimal("12.3400"))],
            [],
        ])
        self.identity = identity or ("AdventureWorks2025", "SSA_ReadOnly", "16.0")
        self.failure = failure
        self.description = []
        self.executed = []
        self.closed = False
        self._current_rows = []
        self._identity_pending = False

    def execute(self, sql):
        if self.failure is not None:
            raise self.failure
        self.executed.append(sql)
        if "DB_NAME()" in sql:
            self.description = [("DatabaseName",), ("LoginName",), ("SQLServerVersion",)]
            self._current_rows = [self.identity]
            self._identity_pending = True
            return
        if len(self.executed) == 2:
            self.description = [("ProductID",), ("Name",), ("ListPrice",)]
        else:
            self.description = [("CustomerID",), ("SalesOrderCount",)]
        index = len(self.executed) - 2
        self._current_rows = self.result_rows[index]

    def fetchone(self):
        return self._current_rows[0] if self._current_rows else None

    def fetchmany(self, size):
        return self._current_rows[:size]

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, cursor):
        self.fake_cursor = cursor
        self.rollback_called = False
        self.closed = False

    def cursor(self):
        return self.fake_cursor

    def rollback(self):
        self.rollback_called = True

    def close(self):
        self.closed = True


def run_capture(tmp_path, monkeypatch, cursor):
    monkeypatch.setenv("SQL_USERNAME", "SSA_ReadOnly")
    connection = FakeConnection(cursor)
    path = capture.capture_ground_truth(
        tmp_path,
        connection_factory=lambda: connection,
        configuration_validator=lambda: configuration(),
        captured_at=datetime(2026, 11, 17, 12, 0, tzinfo=timezone.utc),
    )
    return path, connection


def test_reference_cases_are_exact_and_select_only():
    assert [case["case_id"] for case in capture.REFERENCE_CASES] == [
        "AW2025-004",
        "AW2025-012",
    ]
    assert [case["question"] for case in capture.REFERENCE_CASES] == [
        "Show the 10 most expensive products by list price.",
        "Show customers with more than 10 sales orders.",
    ]
    for case in capture.REFERENCE_CASES:
        assert validate_sql(case["sql"]) == case["sql"].strip()
        assert case["sql"].lstrip().upper().startswith("SELECT")


def test_capture_writes_complete_results_and_serializes_decimal_dates(tmp_path, monkeypatch):
    cursor = FakeCursor(result_rows=[
        [
            (1, "Widget", Decimal("12.3400")),
            (2, "Gadget", Decimal("9.5000")),
        ],
        [
            (7, 11),
        ],
    ])
    path, connection = run_capture(tmp_path, monkeypatch, cursor)

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["database"] == {
        "database_name": "AdventureWorks2025",
        "sql_server_version": "16.0",
    }
    assert len(document["cases"]) == 2
    assert document["cases"][0]["captured_result"]["row_count"] == 2
    assert document["cases"][0]["captured_result"]["truncated"] is False
    assert document["cases"][0]["captured_result"]["top_n_complete"] is False
    assert document["cases"][0]["captured_result"]["rows"][0][2] == {
        "__type__": "decimal",
        "value": "12.3400",
    }
    assert document["cases"][1]["captured_result"]["rows"] == [[7, 11]]
    assert "SSA_ReadOnly" not in path.read_text(encoding="utf-8")
    assert connection.fake_cursor.closed is True
    assert connection.rollback_called is True
    assert connection.closed is True


def test_empty_results_are_captured_without_being_confused_with_unknown(tmp_path, monkeypatch):
    cursor = FakeCursor(result_rows=[[], []])
    path, _ = run_capture(tmp_path, monkeypatch, cursor)

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["cases"][0]["captured_result"]["rows"] == []
    assert document["cases"][0]["captured_result"]["row_count"] == 0
    assert document["cases"][1]["captured_result"]["rows"] == []


def test_safety_limit_fails_without_silent_truncation(tmp_path, monkeypatch):
    cursor = FakeCursor(result_rows=[[], [(index, 11) for index in range(capture.SAFETY_ROW_LIMIT + 1)]])
    monkeypatch.setenv("SQL_USERNAME", "SSA_ReadOnly")
    connection = FakeConnection(cursor)

    with pytest.raises(capture.GroundTruthCaptureError, match="safety limit"):
        capture.capture_ground_truth(
            tmp_path,
            connection_factory=lambda: connection,
            configuration_validator=lambda: configuration(),
        )

    assert list(tmp_path.iterdir()) == []
    assert cursor.closed is True
    assert connection.closed is True


def test_wrong_database_is_rejected_before_connection(tmp_path, monkeypatch):
    monkeypatch.setenv("SQL_USERNAME", "SSA_ReadOnly")
    calls = []

    with pytest.raises(capture.GroundTruthCaptureError, match="AdventureWorks2025"):
        capture.capture_ground_truth(
            tmp_path,
            connection_factory=lambda: calls.append(True),
            configuration_validator=lambda: configuration(database="OtherDatabase"),
        )

    assert calls == []


def test_wrong_authentication_mode_is_rejected_before_connection(tmp_path, monkeypatch):
    monkeypatch.setenv("SQL_USERNAME", "SSA_ReadOnly")
    calls = []

    with pytest.raises(capture.GroundTruthCaptureError, match="SQL authentication"):
        capture.capture_ground_truth(
            tmp_path,
            connection_factory=lambda: calls.append(True),
            configuration_validator=lambda: configuration(trusted_connection=True),
        )

    assert calls == []


def test_wrong_read_only_login_is_rejected_before_connection(tmp_path, monkeypatch):
    monkeypatch.setenv("SQL_USERNAME", "other_user")
    calls = []

    with pytest.raises(capture.GroundTruthCaptureError, match="SQL_USERNAME"):
        capture.capture_ground_truth(
            tmp_path,
            connection_factory=lambda: calls.append(True),
            configuration_validator=lambda: configuration(),
        )

    assert calls == []


def test_identity_mismatch_fails_and_cleans_up(tmp_path, monkeypatch):
    monkeypatch.setenv("SQL_USERNAME", "SSA_ReadOnly")
    cursor = FakeCursor(identity=("OtherDatabase", "SSA_ReadOnly", "16.0"))
    connection = FakeConnection(cursor)

    with pytest.raises(capture.GroundTruthCaptureError, match="required AdventureWorks2025"):
        capture.capture_ground_truth(
            tmp_path,
            connection_factory=lambda: connection,
            configuration_validator=lambda: configuration(),
        )

    assert cursor.closed is True
    assert connection.rollback_called is True
    assert connection.closed is True
    assert list(tmp_path.iterdir()) == []


def test_sql_failure_is_safe_and_does_not_write_partial_output(tmp_path, monkeypatch):
    monkeypatch.setenv("SQL_USERNAME", "SSA_ReadOnly")
    cursor = FakeCursor(failure=RuntimeError("driver details must not escape"))
    connection = FakeConnection(cursor)

    with pytest.raises(capture.GroundTruthCaptureError, match="capture failed"):
        capture.capture_ground_truth(
            tmp_path,
            connection_factory=lambda: connection,
            configuration_validator=lambda: configuration(),
        )

    assert list(tmp_path.iterdir()) == []
    assert cursor.closed is True
    assert connection.closed is True


def test_datetime_serialization_is_json_compatible():
    assert capture._json_value(date(2025, 1, 2)) == {
        "__type__": "date",
        "value": "2025-01-02",
    }
    assert capture._json_value(datetime(2025, 1, 2, 3, 4)) == {
        "__type__": "datetime",
        "value": "2025-01-02T03:04:00",
    }


def test_script_has_no_assistant_or_provider_entry_point():
    assert not hasattr(capture, "ask")
    assert not hasattr(capture, "OllamaProvider")
    assert not hasattr(capture, "ChromaSchemaRetriever")


def test_direct_script_import_bootstrap_resolves_local_ssa_without_capture(tmp_path):
    script = Path(capture.__file__).resolve()
    code = (
        "import runpy, sys; "
        "namespace = runpy.run_path(sys.argv[1], run_name='direct_import_test'); "
        "assert namespace['EXPECTED_DATABASE'] == 'AdventureWorks2025'; "
        "assert 'ssa.config_validation' in sys.modules"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [sys.executable, "-c", code, str(script)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Ground-truth capture written" not in completed.stdout
