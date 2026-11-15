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


@pytest.mark.parametrize(
    "environment_name",
    ["SQL_TRUSTED_CONNECTION", "SQL_ENCRYPT", "SQL_TRUST_SERVER_CERTIFICATE"],
)
def test_rejects_invalid_boolean_configuration(monkeypatch, environment_name):
    monkeypatch.setenv(environment_name, "maybe")

    with pytest.raises(db.ConfigurationError, match=environment_name):
        db.get_connection()


@pytest.mark.parametrize("environment_name", ["SQL_SERVER", "SQL_DATABASE", "SQL_DRIVER"])
def test_rejects_empty_required_connection_setting(monkeypatch, environment_name):
    monkeypatch.setenv(environment_name, "   ")

    with pytest.raises(db.ConfigurationError, match=environment_name):
        db.get_connection()


def test_missing_sql_authentication_credentials_is_safe(monkeypatch):
    secret = "super-secret-password"
    monkeypatch.setenv("SQL_TRUSTED_CONNECTION", "no")
    monkeypatch.delenv("SQL_USERNAME", raising=False)
    monkeypatch.delenv("SQL_PASSWORD", raising=False)

    with pytest.raises(db.ConfigurationError) as error:
        db.get_connection()

    assert secret not in str(error.value)
    assert "PWD=" not in str(error.value)


def test_wraps_driver_connection_errors(monkeypatch):
    def fail_connect(connection_string, timeout):
        raise pyodbc.OperationalError("08001", "Server unavailable")

    monkeypatch.setattr(db.pyodbc, "connect", fail_connect)

    with pytest.raises(db.DatabaseConnectionError, match="connection failed after 10 seconds"):
        db.get_connection()
