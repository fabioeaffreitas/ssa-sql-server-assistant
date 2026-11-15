
"""
SSA - Database Connection
Supports Windows Authentication and SQL Authentication.
"""

import os

import pyodbc
from dotenv import load_dotenv

load_dotenv()

DEFAULT_CONNECTION_TIMEOUT = 10
DEFAULT_QUERY_TIMEOUT = 30


class DatabaseConnectionError(RuntimeError):
    """Raised when a connection to SQL Server cannot be established."""


class ConfigurationError(ValueError):
    """Raised when SSA configuration is missing or invalid."""


def _resolve_timeout(value: int | None, environment_name: str, default: int) -> int:
    """Return a positive timeout from an explicit value or environment setting."""

    if value is None:
        value = os.getenv(environment_name, str(default))

    try:
        timeout = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{environment_name} must be a positive integer."
        ) from exc

    if timeout < 1:
        raise ConfigurationError(f"{environment_name} must be a positive integer.")

    return timeout


def _resolve_bool(value: str | None, environment_name: str, default: str) -> bool:
    """Return a configured boolean, rejecting ambiguous values."""

    configured = default if value is None else value
    normalized = configured.strip().lower()
    if normalized in {"yes", "true", "1"}:
        return True
    if normalized in {"no", "false", "0"}:
        return False
    raise ConfigurationError(
        f"{environment_name} must be one of yes, no, true, false, 1, or 0."
    )


def _required_setting(environment_name: str, default: str | None = None) -> str:
    """Return a non-empty environment setting without exposing its value."""

    value = os.getenv(environment_name, default)
    if value is None or not value.strip():
        raise ConfigurationError(f"{environment_name} must not be empty.")
    return value.strip()


def get_connection_timeout(connection_timeout: int | None = None) -> int:
    """Return the configured SQL Server connection timeout in seconds."""

    return _resolve_timeout(
        connection_timeout,
        "SQL_CONNECTION_TIMEOUT",
        DEFAULT_CONNECTION_TIMEOUT,
    )


def get_query_timeout(query_timeout: int | None = None) -> int:
    """Return the configured SQL Server query timeout in seconds."""

    return _resolve_timeout(
        query_timeout,
        "SQL_QUERY_TIMEOUT",
        DEFAULT_QUERY_TIMEOUT,
    )


def get_connection(connection_timeout: int | None = None):
    """Open a SQL Server connection with a bounded connection timeout."""

    server = _required_setting("SQL_SERVER", "localhost")
    database = _required_setting("SQL_DATABASE", "AdventureWorks2025")
    driver = _required_setting("SQL_DRIVER", "ODBC Driver 18 for SQL Server")

    trusted_connection = _resolve_bool(
        os.getenv("SQL_TRUSTED_CONNECTION"),
        "SQL_TRUSTED_CONNECTION",
        "yes",
    )
    encrypt = _resolve_bool(os.getenv("SQL_ENCRYPT"), "SQL_ENCRYPT", "yes")
    trust_server_certificate = _resolve_bool(
        os.getenv("SQL_TRUST_SERVER_CERTIFICATE"),
        "SQL_TRUST_SERVER_CERTIFICATE",
        "yes",
    )

    connection_string = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"Encrypt={'yes' if encrypt else 'no'};"
        f"TrustServerCertificate={'yes' if trust_server_certificate else 'no'};"
    )

    if trusted_connection:
        connection_string += "Trusted_Connection=yes;"
    else:
        username = os.getenv("SQL_USERNAME")
        password = os.getenv("SQL_PASSWORD")

        if (
            not username
            or not username.strip()
            or not password
            or not password.strip()
        ):
            raise ConfigurationError(
                "SQL_USERNAME and SQL_PASSWORD are required "
                "when SQL_TRUSTED_CONNECTION=no."
            )

        connection_string += (
            f"UID={username};"
            f"PWD={password};"
        )

    timeout = get_connection_timeout(connection_timeout)

    try:
        return pyodbc.connect(connection_string, timeout=timeout)
    except pyodbc.Error as exc:
        raise DatabaseConnectionError(
            f"SQL Server connection failed after {timeout} seconds."
        ) from exc
