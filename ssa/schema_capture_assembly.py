"""Pure assembly of versioned schema metadata from supplied evidence."""

from copy import deepcopy
from datetime import datetime
from collections.abc import Mapping
from typing import Any

from ssa.schema_capture_provenance import (
    METADATA_FORMAT_VERSION,
    SQLServerIdentityEvidence,
    SchemaCaptureProvenance,
    parse_capture_provenance,
    serialize_capture_provenance,
)
from ssa.schema_key_metadata import (
    fingerprint_schema_metadata,
    fingerprint_schema_semantic_payload,
)


_RESERVED_ROOT_FIELDS = frozenset(
    {
        "metadata_format_version",
        "provenance",
        "captured_at_utc",
        "captured_at",
        "capture_timestamp",
        "capture_time_utc",
        "export_run_id",
        "snapshot_id",
        "content_fingerprint",
    }
)


def assemble_versioned_schema_metadata(
    metadata: Mapping[str, Any],
    *,
    snapshot_id: str,
    captured_at_utc: datetime,
    server_identity: SQLServerIdentityEvidence,
    server_version: str | None = None,
    capture_tool_version: str | None = None,
) -> dict[str, Any]:
    """Assemble and verify one complete version-2 metadata document.

    The input must be legacy semantic metadata. Capture identity is supplied
    explicitly by the caller and is never inferred or generated here. The
    returned mapping is a defensive copy. This function performs no
    filesystem, database, ChromaDB, or network I/O.
    """

    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    _reject_reserved_root_fields(metadata)

    # The existing complete-document API is used only for legacy structural
    # validation here. No legacy digest is reused after version 2 is added.
    fingerprint_schema_metadata(metadata)
    semantic = deepcopy(dict(metadata))
    semantic["metadata_format_version"] = METADATA_FORMAT_VERSION

    content_fingerprint = fingerprint_schema_semantic_payload(semantic)
    database = semantic["database"]

    provenance = SchemaCaptureProvenance(
        snapshot_id=snapshot_id,
        captured_at_utc=captured_at_utc,
        database=database,
        server_identity=server_identity,
        content_fingerprint=content_fingerprint,
        server_version=server_version,
        capture_tool_version=capture_tool_version,
    )
    assembled = semantic.copy()
    assembled.update(serialize_capture_provenance(provenance))

    parse_capture_provenance(assembled)
    verified_fingerprint = fingerprint_schema_metadata(assembled)
    if verified_fingerprint != content_fingerprint:
        raise ValueError("assembled semantic fingerprint does not match provenance")

    for key, value in semantic.items():
        if assembled.get(key) != value:
            raise ValueError(f"assembled semantic field changed: {key}")

    return assembled


def _reject_reserved_root_fields(metadata: Mapping[str, Any]) -> None:
    for field_name in _RESERVED_ROOT_FIELDS:
        if field_name not in metadata:
            continue
        if field_name == "metadata_format_version":
            raise ValueError("input already contains metadata_format_version")
        raise ValueError(f"reserved root field is not accepted: {field_name}")


__all__ = ["assemble_versioned_schema_metadata"]
