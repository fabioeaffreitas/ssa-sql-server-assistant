
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


def _resolve_timeout(value: int | None, environment_name: str, default: int) -> int:
    """Return a positive timeout from an explicit value or environment setting."""

    if value is None:
        value = os.getenv(environment_name, str(default))

    try:
        timeout = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{environment_name} must be a positive integer.") from exc

    if timeout < 1:
        raise ValueError(f"{environment_name} must be a positive integer.")

    return timeout


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

    server = os.getenv("SQL_SERVER", "localhost")
    database = os.getenv("SQL_DATABASE", "AdventureWorks2025")
    driver = os.getenv(
        "SQL_DRIVER",
        "ODBC Driver 18 for SQL Server"
    )

    trusted_connection = os.getenv(
        "SQL_TRUSTED_CONNECTION", "yes"
    ).lower()

    encrypt = os.getenv("SQL_ENCRYPT", "yes")
    trust_server_certificate = os.getenv(
        "SQL_TRUST_SERVER_CERTIFICATE", "yes"
    )

    connection_string = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"Encrypt={encrypt};"
        f"TrustServerCertificate={trust_server_certificate};"
    )

    if trusted_connection in ("yes", "true", "1"):
        connection_string += "Trusted_Connection=yes;"
    else:
        username = os.getenv("SQL_USERNAME")
        password = os.getenv("SQL_PASSWORD")

        if not username or not password:
            raise ValueError(
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
