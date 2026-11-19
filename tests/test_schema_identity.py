from dataclasses import FrozenInstanceError

import pytest

from ssa.schema_identity import (
    IDENTITY_QUERY,
    SQLServerIdentityCapture,
    SQLServerIdentityReadError,
    read_sql_server_identity,
)


class FakeCursor:
    def __init__(self, row=None, *, execute_error=None, fetch_error=None, close_error=None):
        self.row = row
        self.execute_error = execute_error
        self.fetch_error = fetch_error
        self.close_error = close_error
        self.executed = []
        self.closed = False

    def execute(self, query, *parameters):
        self.executed.append((query, parameters))
        if self.execute_error:
            raise self.execute_error

    def fetchone(self):
        if self.fetch_error:
            raise self.fetch_error
        return self.row

    def close(self):
        self.closed = True
        if self.close_error:
            raise self.close_error


class FakeConnection:
    def __init__(self, cursor=None, *, cursor_error=None):
        self.fake_cursor = cursor
        self.cursor_error = cursor_error
        self.closed = False
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        if self.cursor_error:
            raise self.cursor_error
        return self.fake_cursor

    def close(self):
        self.closed = True

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def run_reader(row, **cursor_options):
    cursor = FakeCursor(row, **cursor_options)
    connection = FakeConnection(cursor)
    result = read_sql_server_identity(connection)
    return result, connection, cursor


def test_reads_named_instance_identity_and_product_version():
    result, connection, cursor = run_reader(
        ("SQLHOST\\SQLEXPRESS", "SQLEXPRESS", 7, "17.0.1000.7")
    )

    assert isinstance(result, SQLServerIdentityCapture)
    assert result.identity.server_name == "SQLHOST\\SQLEXPRESS"
    assert result.identity.instance_name == "SQLEXPRESS"
    assert result.identity.database_id == 7
    assert result.identity.database_guid is None
    assert result.server_version == "17.0.1000.7"
    assert cursor.closed is True
    assert connection.closed is False


def test_default_instance_and_null_optional_values_are_preserved():
    result, _, _ = run_reader(("SQLHOST", None, None, None))

    assert result.identity.instance_name is None
    assert result.identity.database_id is None
    assert result.server_version is None


@pytest.mark.parametrize("server_name", [None, "", "   "])
def test_invalid_server_name_is_rejected(server_name):
    cursor = FakeCursor((server_name, None, None, None))
    connection = FakeConnection(cursor)

    with pytest.raises(SQLServerIdentityReadError):
        read_sql_server_identity(connection)
    assert cursor.closed is True


@pytest.mark.parametrize("instance_name", ["", "   ", 123])
def test_invalid_instance_name_is_rejected(instance_name):
    with pytest.raises(SQLServerIdentityReadError):
        run_reader(("SQLHOST", instance_name, None, None))


@pytest.mark.parametrize("database_id", [-1, "7", 7.0, True])
def test_invalid_database_id_is_rejected(database_id):
    with pytest.raises(SQLServerIdentityReadError):
        run_reader(("SQLHOST", None, database_id, None))


@pytest.mark.parametrize("server_version", ["", "   ", 17, "17.0", "17.x.1000.7"])
def test_invalid_product_version_is_rejected(server_version):
    with pytest.raises(SQLServerIdentityReadError):
        run_reader(("SQLHOST", None, None, server_version))


@pytest.mark.parametrize(
    "row",
    [None, ("SQLHOST", None, None), ("SQLHOST", None, None, None, "extra")],
)
def test_missing_or_wrong_width_row_is_rejected(row):
    with pytest.raises(SQLServerIdentityReadError):
        run_reader(row)


def test_cursor_acquisition_failure_is_wrapped_without_connection_cleanup():
    connection = FakeConnection(cursor_error=RuntimeError("driver details"))

    with pytest.raises(SQLServerIdentityReadError, match="acquire"):
        read_sql_server_identity(connection)
    assert connection.closed is False


def test_execution_failure_closes_cursor_and_preserves_connection():
    cursor = FakeCursor(execute_error=RuntimeError("driver details"))
    connection = FakeConnection(cursor)

    with pytest.raises(SQLServerIdentityReadError, match="execute"):
        read_sql_server_identity(connection)
    assert cursor.closed is True
    assert connection.closed is False


def test_fetch_failure_closes_cursor():
    cursor = FakeCursor(fetch_error=RuntimeError("driver details"))
    connection = FakeConnection(cursor)

    with pytest.raises(SQLServerIdentityReadError, match="fetch"):
        read_sql_server_identity(connection)
    assert cursor.closed is True


def test_cursor_closes_on_validation_failure():
    cursor = FakeCursor(("", None, None, None))
    connection = FakeConnection(cursor)

    with pytest.raises(SQLServerIdentityReadError):
        read_sql_server_identity(connection)
    assert cursor.closed is True


def test_cursor_close_failure_is_reported_after_success():
    cursor = FakeCursor(("SQLHOST", None, None, None), close_error=OSError("close"))
    connection = FakeConnection(cursor)

    with pytest.raises(SQLServerIdentityReadError, match="close"):
        read_sql_server_identity(connection)


def test_cursor_close_failure_does_not_mask_primary_failure():
    cursor = FakeCursor(
        execute_error=RuntimeError("driver details"),
        close_error=OSError("close details"),
    )
    connection = FakeConnection(cursor)

    with pytest.raises(SQLServerIdentityReadError, match="execute") as error:
        read_sql_server_identity(connection)
    assert "close details" not in str(error.value)


def test_exact_query_once_without_parameters_or_transaction_control():
    _, connection, cursor = run_reader(("SQLHOST", None, None, None))

    assert cursor.executed == [(IDENTITY_QUERY, ())]
    assert "SERVERPROPERTY('ServerName')" in IDENTITY_QUERY
    assert "SERVERPROPERTY('InstanceName')" in IDENTITY_QUERY
    assert "SERVERPROPERTY('ProductVersion')" in IDENTITY_QUERY
    assert "DB_ID()" in IDENTITY_QUERY
    assert "sys." not in IDENTITY_QUERY.lower()
    assert "insert" not in IDENTITY_QUERY.lower()
    assert "update" not in IDENTITY_QUERY.lower()
    assert "delete" not in IDENTITY_QUERY.lower()
    assert "commit" not in IDENTITY_QUERY.lower()
    assert "rollback" not in IDENTITY_QUERY.lower()
    assert connection.closed is False
    assert connection.committed is False
    assert connection.rolled_back is False


def test_capture_is_frozen():
    result, _, _ = run_reader(("SQLHOST", None, None, None))

    with pytest.raises(FrozenInstanceError):
        result.server_version = "17.0.1000.7"


def test_reader_does_not_access_environment_or_credentials(monkeypatch):
    def fail_getenv(*args, **kwargs):
        raise AssertionError("environment access is not allowed")

    monkeypatch.setattr("os.getenv", fail_getenv)
    result, _, _ = run_reader(("SQLHOST", None, None, None))

    assert result.identity.database_guid is None
