"""Pure versioned provenance contracts for schema metadata captures.

This module performs validation and JSON-compatible serialization only. It does
not obtain timestamps, generate identifiers, access files, or contact SQL
Server, ChromaDB, or Ollama.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re
from collections.abc import Mapping
from typing import Any


METADATA_FORMAT_VERSION = 2


class SchemaCaptureProvenanceError(ValueError):
    """Base error for invalid versioned capture provenance."""


class MissingProvenanceError(SchemaCaptureProvenanceError):
    """Required version or provenance data is absent."""


class UnsupportedMetadataVersionError(SchemaCaptureProvenanceError):
    """The metadata document uses an unsupported format version."""


class DatabaseIdentityMismatchError(SchemaCaptureProvenanceError):
    """The root and provenance database identities do not match."""


class InvalidTimestampError(SchemaCaptureProvenanceError):
    """The capture timestamp is not canonical UTC."""


class InvalidServerIdentityError(SchemaCaptureProvenanceError):
    """SQL Server identity evidence is missing or malformed."""


class InvalidFingerprintError(SchemaCaptureProvenanceError):
    """The semantic content fingerprint is malformed."""


@dataclass(frozen=True)
class SQLServerIdentityEvidence:
    """Captured SQL Server identity evidence, not a uniqueness guarantee."""

    server_name: str
    instance_name: str | None = None
    database_id: int | None = None
    database_guid: str | None = None

    def __post_init__(self) -> None:
        _identity_text(self.server_name, "server_name")
        _optional_identity_text(self.instance_name, "instance_name")
        if self.database_id is not None and (
            isinstance(self.database_id, bool)
            or not isinstance(self.database_id, int)
            or self.database_id < 0
        ):
            raise InvalidServerIdentityError("database_id must be a nonnegative integer")
        _optional_identity_text(self.database_guid, "database_guid")


@dataclass(frozen=True)
class SchemaCaptureProvenance:
    """Immutable provenance for one schema capture event."""

    snapshot_id: str
    captured_at_utc: datetime
    database: str
    server_identity: SQLServerIdentityEvidence
    content_fingerprint: str
    server_version: str | None = None
    capture_tool_version: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.snapshot_id, "snapshot_id")
        _require_text(self.database, "database")
        _validate_utc_timestamp(self.captured_at_utc)
        if not isinstance(self.server_identity, SQLServerIdentityEvidence):
            raise InvalidServerIdentityError(
                "server_identity must be SQLServerIdentityEvidence"
            )
        _validate_fingerprint(self.content_fingerprint)
        _optional_text(self.server_version, "server_version")
        _optional_text(self.capture_tool_version, "capture_tool_version")


def serialize_capture_provenance(
    provenance: SchemaCaptureProvenance,
) -> dict[str, object]:
    """Serialize provenance as a mergeable versioned metadata-root fragment."""

    if not isinstance(provenance, SchemaCaptureProvenance):
        raise TypeError("provenance must be SchemaCaptureProvenance")

    server_identity: dict[str, object] = {
        "server_name": provenance.server_identity.server_name,
    }
    if provenance.server_identity.instance_name is not None:
        server_identity["instance_name"] = provenance.server_identity.instance_name
    if provenance.server_identity.database_id is not None:
        server_identity["database_id"] = provenance.server_identity.database_id
    if provenance.server_identity.database_guid is not None:
        server_identity["database_guid"] = provenance.server_identity.database_guid

    nested: dict[str, object] = {
        "snapshot_id": provenance.snapshot_id,
        "captured_at_utc": _serialize_timestamp(provenance.captured_at_utc),
        "database": provenance.database,
        "server_identity": server_identity,
        "content_fingerprint": provenance.content_fingerprint,
    }
    if provenance.server_version is not None:
        nested["server_version"] = provenance.server_version
    if provenance.capture_tool_version is not None:
        nested["capture_tool_version"] = provenance.capture_tool_version

    return {
        "metadata_format_version": METADATA_FORMAT_VERSION,
        "database": provenance.database,
        "provenance": nested,
    }


def parse_capture_provenance(
    metadata: Mapping[str, Any],
) -> SchemaCaptureProvenance:
    """Parse provenance from a complete or partial versioned metadata root.

    Unknown root and provenance fields are ignored for forward compatibility.
    Required known fields are never coerced. Legacy unversioned metadata is
    rejected as unavailable provenance rather than assigned invented values.
    """

    if not isinstance(metadata, Mapping):
        raise SchemaCaptureProvenanceError("metadata root must be an object")

    if "metadata_format_version" not in metadata:
        raise MissingProvenanceError("metadata format version is missing")
    version = metadata["metadata_format_version"]
    if version != METADATA_FORMAT_VERSION:
        raise UnsupportedMetadataVersionError(
            f"unsupported metadata format version: {version!r}"
        )

    root_database = metadata.get("database")
    if not isinstance(root_database, str) or not root_database.strip():
        raise SchemaCaptureProvenanceError("root database is missing or invalid")

    nested = metadata.get("provenance")
    if not isinstance(nested, Mapping):
        raise MissingProvenanceError("provenance object is missing or invalid")

    provenance_database = nested.get("database")
    if not isinstance(provenance_database, str) or not provenance_database.strip():
        raise SchemaCaptureProvenanceError("provenance database is missing or invalid")
    if provenance_database.strip().casefold() != root_database.strip().casefold():
        raise DatabaseIdentityMismatchError(
            "root and provenance database identities do not match"
        )

    snapshot_id = nested.get("snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise SchemaCaptureProvenanceError("snapshot_id is missing or invalid")

    captured_at = nested.get("captured_at_utc")
    parsed_timestamp = _parse_timestamp(captured_at)

    server_identity = _parse_server_identity(nested.get("server_identity"))

    content_fingerprint = nested.get("content_fingerprint")
    _validate_fingerprint(content_fingerprint)

    server_version = _parse_optional_text(nested, "server_version")
    capture_tool_version = _parse_optional_text(nested, "capture_tool_version")

    return SchemaCaptureProvenance(
        snapshot_id=snapshot_id,
        captured_at_utc=parsed_timestamp,
        database=provenance_database,
        server_identity=server_identity,
        content_fingerprint=content_fingerprint,
        server_version=server_version,
        capture_tool_version=capture_tool_version,
    )


def _require_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SchemaCaptureProvenanceError(
            f"{field_name} must be a nonempty string without surrounding whitespace"
        )


def _optional_text(value: Any, field_name: str) -> None:
    if value is not None:
        _require_text(value, field_name)


def _identity_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise InvalidServerIdentityError(
            f"{field_name} must be a nonempty string without surrounding whitespace"
        )


def _optional_identity_text(value: Any, field_name: str) -> None:
    if value is not None:
        _identity_text(value, field_name)


def _parse_optional_text(
    mapping: Mapping[str, Any],
    field_name: str,
) -> str | None:
    value = mapping.get(field_name)
    _optional_text(value, field_name)
    return value


def _validate_utc_timestamp(value: Any) -> None:
    if not isinstance(value, datetime):
        raise InvalidTimestampError("captured_at_utc must be a datetime")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise InvalidTimestampError("captured_at_utc must be timezone-aware UTC")


def _serialize_timestamp(value: datetime) -> str:
    _validate_utc_timestamp(value)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$"
)


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not _TIMESTAMP_PATTERN.fullmatch(value):
        raise InvalidTimestampError("captured_at_utc must use canonical UTC format")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise InvalidTimestampError("captured_at_utc is invalid") from exc
    return parsed.replace(tzinfo=timezone.utc)


def _validate_fingerprint(value: Any) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise InvalidFingerprintError(
            "content_fingerprint must be 64 lowercase hexadecimal characters"
        )


def _parse_server_identity(value: Any) -> SQLServerIdentityEvidence:
    if not isinstance(value, Mapping):
        raise InvalidServerIdentityError("server_identity is missing or invalid")

    return SQLServerIdentityEvidence(
        server_name=value.get("server_name"),
        instance_name=value.get("instance_name"),
        database_id=value.get("database_id"),
        database_guid=value.get("database_guid"),
    )


__all__ = [
    "DatabaseIdentityMismatchError",
    "InvalidFingerprintError",
    "InvalidServerIdentityError",
    "InvalidTimestampError",
    "METADATA_FORMAT_VERSION",
    "MissingProvenanceError",
    "SQLServerIdentityEvidence",
    "SchemaCaptureProvenance",
    "SchemaCaptureProvenanceError",
    "UnsupportedMetadataVersionError",
    "parse_capture_provenance",
    "serialize_capture_provenance",
]
