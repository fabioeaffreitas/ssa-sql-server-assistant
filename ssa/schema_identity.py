"""Read-only SQL Server identity evidence from an existing connection."""

from dataclasses import dataclass
import re
from typing import Any

from ssa.schema_capture_provenance import SQLServerIdentityEvidence


IDENTITY_QUERY = """
SELECT
    CONVERT(nvarchar(256), SERVERPROPERTY('ServerName')) AS ServerName,
    CONVERT(nvarchar(128), SERVERPROPERTY('InstanceName')) AS InstanceName,
    DB_ID() AS DatabaseId,
    CONVERT(nvarchar(128), SERVERPROPERTY('ProductVersion')) AS ProductVersion;
"""

_PRODUCT_VERSION_PATTERN = re.compile(r"^\d+(?:\.\d+){3}$")


class SQLServerIdentityReadError(RuntimeError):
    """Raised when SQL Server identity evidence cannot be read safely."""


@dataclass(frozen=True)
class SQLServerIdentityCapture:
    """Identity evidence and optional product version from one SQL read."""

    identity: SQLServerIdentityEvidence
    server_version: str | None


def read_sql_server_identity(connection: Any) -> SQLServerIdentityCapture:
    """Read SQL Server identity evidence without taking ownership of connection."""

    try:
        cursor = connection.cursor()
    except Exception as exc:
        raise SQLServerIdentityReadError(
            "could not acquire SQL Server identity cursor"
        ) from exc

    primary_error: BaseException | None = None
    try:
        try:
            cursor.execute(IDENTITY_QUERY)
        except Exception as exc:
            raise SQLServerIdentityReadError(
                "could not execute SQL Server identity query"
            ) from exc

        try:
            row = cursor.fetchone()
        except Exception as exc:
            raise SQLServerIdentityReadError(
                "could not fetch SQL Server identity row"
            ) from exc

        if row is None:
            raise SQLServerIdentityReadError(
                "SQL Server identity query returned no row"
            )

        try:
            values = tuple(row)
        except Exception as exc:
            raise SQLServerIdentityReadError(
                "SQL Server identity row is not readable"
            ) from exc

        if len(values) != 4:
            raise SQLServerIdentityReadError(
                "SQL Server identity row must contain exactly four values"
            )

        server_name, instance_name, database_id, server_version = values
        _validate_server_version(server_version)

        try:
            identity = SQLServerIdentityEvidence(
                server_name=server_name,
                instance_name=instance_name,
                database_id=database_id,
            )
        except Exception as exc:
            raise SQLServerIdentityReadError(
                "SQL Server identity evidence is invalid"
            ) from exc

        return SQLServerIdentityCapture(
            identity=identity,
            server_version=server_version,
        )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            cursor.close()
        except Exception as exc:
            if primary_error is None:
                raise SQLServerIdentityReadError(
                    "could not close SQL Server identity cursor"
                ) from exc


def _validate_server_version(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not _PRODUCT_VERSION_PATTERN.fullmatch(value):
        raise SQLServerIdentityReadError(
            "SQL Server product version is invalid"
        )


__all__ = [
    "IDENTITY_QUERY",
    "SQLServerIdentityCapture",
    "SQLServerIdentityReadError",
    "read_sql_server_identity",
]
