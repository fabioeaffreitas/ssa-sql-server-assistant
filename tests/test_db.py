import pyodbc
import pytest

import ssa.db as db


def test_connection_uses_default_timeout(monkeypatch):
    captured = {}

    def fake_connect(connection_string, timeout):
        captured["connection_string"] = connection_string
        captured["timeout"] = timeout
        return object()

    monkeypatch.delenv("SQL_CONNECTION_TIMEOUT", raising=False)
    monkeypatch.setattr(db.pyodbc, "connect", fake_connect)

    db.get_connection()

    assert captured["timeout"] == 10


def test_connection_uses_configured_timeout(monkeypatch):
    captured = {}
    monkeypatch.setenv("SQL_CONNECTION_TIMEOUT", "7")
    monkeypatch.setattr(
        db.pyodbc,
        "connect",
        lambda connection_string, timeout: captured.setdefault("timeout", timeout),
    )

    db.get_connection()

    assert captured["timeout"] == 7


def test_rejects_invalid_query_timeout(monkeypatch):
    monkeypatch.setenv("SQL_QUERY_TIMEOUT", "0")

    with pytest.raises(ValueError, match="SQL_QUERY_TIMEOUT must be a positive integer"):
        db.get_query_timeout()


def test_wraps_driver_connection_errors(monkeypatch):
    def fail_connect(connection_string, timeout):
        raise pyodbc.OperationalError("08001", "Server unavailable")

    monkeypatch.setattr(db.pyodbc, "connect", fail_connect)

    with pytest.raises(db.DatabaseConnectionError, match="connection failed after 10 seconds"):
        db.get_connection()
