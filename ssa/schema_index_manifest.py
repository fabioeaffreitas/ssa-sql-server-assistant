"""Pure contracts for aligning schema-index documents with source metadata."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import re
from typing import Any

from ssa.schema_capture_provenance import parse_capture_provenance
from ssa.schema_key_metadata import (
    fingerprint_schema_metadata,
    fingerprint_schema_metadata_json,
)


INDEX_MANIFEST_FORMAT_VERSION = 1
SUPPORTED_METADATA_FORMAT_VERSION = 2
DOCUMENT_GENERATION_FORMAT_VERSION = 1
_IDENTITY_DIGEST_DOMAIN = "ssa-schema-index-document-identities-v1"
_CONTENT_DIGEST_DOMAIN = "ssa-schema-index-document-content-v1"
_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$"
)
_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_BUILD_STATES = frozenset({"incomplete", "complete"})


class SchemaIndexManifestError(ValueError):
    """Raised when schema-index alignment evidence is invalid."""


@dataclass(frozen=True)
class SchemaIndexSource:
    """Validated version-2 metadata identity for one source capture."""

    database: str
    metadata_format_version: int
    content_fingerprint: str
    snapshot_id: str
    captured_at_utc: datetime

    def __post_init__(self) -> None:
        _require_identifier(self.database, "database")
        if self.metadata_format_version != SUPPORTED_METADATA_FORMAT_VERSION:
            raise SchemaIndexManifestError(
                "source metadata_format_version must be 2"
            )
        _require_digest(self.content_fingerprint, "content_fingerprint")
        _require_identifier(self.snapshot_id, "snapshot_id")
        _validate_utc_datetime(self.captured_at_utc, "captured_at_utc")


@dataclass(frozen=True)
class SchemaIndexDocumentEvidence:
    """Expected document inventory evidence for one index build."""

    generation_format_version: int
    expected_count: int
    id_digest: str
    content_digest: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.generation_format_version, bool)
            or not isinstance(self.generation_format_version, int)
            or self.generation_format_version < 1
        ):
            raise SchemaIndexManifestError(
                "generation_format_version must be a positive integer"
            )
        if (
            isinstance(self.expected_count, bool)
            or not isinstance(self.expected_count, int)
            or self.expected_count < 0
        ):
            raise SchemaIndexManifestError(
                "expected_count must be a nonnegative integer"
            )
        _require_digest(self.id_digest, "id_digest")
        _require_digest(self.content_digest, "content_digest")


@dataclass(frozen=True)
class DocumentInventoryComparison:
    """Immutable comparison result for expected and observed documents."""

    matches: bool
    expected_count: int
    observed_count: int
    missing_ids: tuple[str, ...] = ()
    unexpected_ids: tuple[str, ...] = ()
    changed_document_ids: tuple[str, ...] = ()
    duplicate_ids: tuple[str, ...] = ()
    duplicate_expected_ids: tuple[str, ...] = ()
    duplicate_observed_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SchemaIndexManifest:
    """Strict, immutable manifest evidence for one index build."""

    status: str
    source: SchemaIndexSource
    documents: SchemaIndexDocumentEvidence
    build_id: str
    started_at_utc: datetime
    completed_at_utc: datetime | None = None
    embedding_identity: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _BUILD_STATES:
            raise SchemaIndexManifestError(
                "status must be incomplete or complete"
            )
        if not isinstance(self.source, SchemaIndexSource):
            raise SchemaIndexManifestError("source must be SchemaIndexSource")
        if not isinstance(self.documents, SchemaIndexDocumentEvidence):
            raise SchemaIndexManifestError(
                "documents must be SchemaIndexDocumentEvidence"
            )
        _require_identifier(self.build_id, "build_id")
        _validate_utc_datetime(self.started_at_utc, "started_at_utc")

        if self.embedding_identity is not None:
            _require_identifier(self.embedding_identity, "embedding_identity")

        if self.status == "incomplete":
            if self.completed_at_utc is not None:
                raise SchemaIndexManifestError(
                    "incomplete manifest must not have completed_at_utc"
                )
            return

        if self.documents.expected_count < 1:
            raise SchemaIndexManifestError(
                "complete manifest requires a positive expected_count"
            )
        if self.completed_at_utc is None:
            raise SchemaIndexManifestError(
                "complete manifest requires completed_at_utc"
            )
        _validate_utc_datetime(self.completed_at_utc, "completed_at_utc")
        if self.completed_at_utc < self.started_at_utc:
            raise SchemaIndexManifestError(
                "completed_at_utc must not precede started_at_utc"
            )


def validate_schema_index_source(
    metadata: Mapping[str, Any] | str,
    expected_database: str | None = None,
) -> SchemaIndexSource:
    """Validate version-2 metadata and return immutable source evidence."""

    try:
        if isinstance(metadata, str):
            fingerprint = fingerprint_schema_metadata_json(metadata)
            parsed = _parse_json_object(metadata)
        else:
            if not isinstance(metadata, Mapping):
                raise SchemaIndexManifestError("metadata must be a mapping or JSON text")
            parsed = dict(metadata)
            fingerprint = fingerprint_schema_metadata(parsed)
    except SchemaIndexManifestError:
        raise
    except Exception as exc:
        raise SchemaIndexManifestError("invalid version-2 metadata source") from exc

    try:
        provenance = parse_capture_provenance(parsed)
    except Exception as exc:
        raise SchemaIndexManifestError(
            "metadata does not contain valid version-2 provenance"
        ) from exc

    if fingerprint != provenance.content_fingerprint:
        raise SchemaIndexManifestError(
            "metadata content fingerprint does not match provenance"
        )

    if expected_database is not None:
        _require_identifier(expected_database, "expected_database")
        if provenance.database.casefold() != expected_database.casefold():
            raise SchemaIndexManifestError("metadata database does not match expected database")

    return SchemaIndexSource(
        database=provenance.database,
        metadata_format_version=2,
        content_fingerprint=provenance.content_fingerprint,
        snapshot_id=provenance.snapshot_id,
        captured_at_utc=provenance.captured_at_utc,
    )


def compute_document_identity_digest(
    document_ids: Iterable[str],
) -> str:
    """Return a domain-separated digest of an exact document-ID inventory."""

    ids = _validate_document_ids(document_ids)
    payload = sorted(ids)
    return _digest(_IDENTITY_DIGEST_DOMAIN, payload)


def compute_document_content_digest(
    documents: Iterable[tuple[str, str]] | Mapping[str, str],
) -> str:
    """Return a domain-separated digest of exact document ID/text pairs."""

    records, duplicates = _validate_document_records(documents)
    if duplicates:
        raise SchemaIndexManifestError(
            "document content contains duplicate IDs"
        )
    payload = [[document_id, text] for document_id, text in sorted(records)]
    return _digest(_CONTENT_DIGEST_DOMAIN, payload)


def compare_document_inventories(
    expected: Iterable[tuple[str, str]] | Mapping[str, str],
    observed: Iterable[tuple[str, str]] | Mapping[str, str],
) -> DocumentInventoryComparison:
    """Compare document IDs and exact text without silently deduplicating."""

    expected_records, expected_duplicates = _validate_document_records(expected)
    observed_records, observed_duplicates = _validate_document_records(observed)

    expected_map = dict(expected_records)
    observed_map = dict(observed_records)
    expected_ids = set(expected_map)
    observed_ids = set(observed_map)
    missing = tuple(sorted(expected_ids - observed_ids))
    unexpected = tuple(sorted(observed_ids - expected_ids))
    changed = tuple(
        sorted(
            document_id
            for document_id in expected_ids & observed_ids
            if expected_map[document_id] != observed_map[document_id]
        )
    )
    duplicate_expected = tuple(sorted(set(expected_duplicates)))
    duplicate_observed = tuple(sorted(set(observed_duplicates)))
    duplicates = tuple(sorted(set(duplicate_expected) | set(duplicate_observed)))

    return DocumentInventoryComparison(
        matches=(
            not duplicate_expected
            and not duplicate_observed
            and len(expected_records) == len(observed_records)
            and not missing
            and not unexpected
            and not changed
        ),
        expected_count=len(expected_records),
        observed_count=len(observed_records),
        missing_ids=missing,
        unexpected_ids=unexpected,
        changed_document_ids=changed,
        duplicate_ids=duplicates,
        duplicate_expected_ids=duplicate_expected,
        duplicate_observed_ids=duplicate_observed,
    )


def serialize_index_manifest(
    manifest: SchemaIndexManifest,
) -> dict[str, Any]:
    """Serialize a manifest into a deterministic JSON-compatible mapping."""

    if not isinstance(manifest, SchemaIndexManifest):
        raise SchemaIndexManifestError("manifest must be SchemaIndexManifest")

    build: dict[str, Any] = {
        "build_id": manifest.build_id,
        "started_at_utc": _serialize_timestamp(manifest.started_at_utc),
    }
    if manifest.completed_at_utc is not None:
        build["completed_at_utc"] = _serialize_timestamp(manifest.completed_at_utc)

    return {
        "manifest_format_version": INDEX_MANIFEST_FORMAT_VERSION,
        "status": manifest.status,
        "source": {
            "database": manifest.source.database,
            "metadata_format_version": manifest.source.metadata_format_version,
            "content_fingerprint": manifest.source.content_fingerprint,
            "snapshot_id": manifest.source.snapshot_id,
            "captured_at_utc": _serialize_timestamp(
                manifest.source.captured_at_utc
            ),
        },
        "documents": {
            "generation_format_version": manifest.documents.generation_format_version,
            "expected_count": manifest.documents.expected_count,
            "id_digest": manifest.documents.id_digest,
            "content_digest": manifest.documents.content_digest,
        },
        "embedding": {"identity": manifest.embedding_identity},
        "build": build,
    }


def parse_index_manifest(
    value: Mapping[str, Any],
) -> SchemaIndexManifest:
    """Strictly parse and validate a manifest mapping."""

    if not isinstance(value, Mapping):
        raise SchemaIndexManifestError("manifest must be an object")
    _require_exact_keys(
        value,
        {
            "manifest_format_version",
            "status",
            "source",
            "documents",
            "embedding",
            "build",
        },
        "manifest",
    )
    if value["manifest_format_version"] != INDEX_MANIFEST_FORMAT_VERSION:
        raise SchemaIndexManifestError("unsupported manifest format version")
    if not isinstance(value["status"], str):
        raise SchemaIndexManifestError("manifest status must be a string")

    source = value["source"]
    documents = value["documents"]
    embedding = value["embedding"]
    build = value["build"]
    if not isinstance(source, Mapping):
        raise SchemaIndexManifestError("manifest source must be an object")
    if not isinstance(documents, Mapping):
        raise SchemaIndexManifestError("manifest documents must be an object")
    if not isinstance(embedding, Mapping):
        raise SchemaIndexManifestError("manifest embedding must be an object")
    if not isinstance(build, Mapping):
        raise SchemaIndexManifestError("manifest build must be an object")

    _require_exact_keys(
        source,
        {"database", "metadata_format_version", "content_fingerprint", "snapshot_id", "captured_at_utc"},
        "source",
    )
    _require_exact_keys(
        documents,
        {"generation_format_version", "expected_count", "id_digest", "content_digest"},
        "documents",
    )
    _require_exact_keys(embedding, {"identity"}, "embedding")

    build_keys = {"build_id", "started_at_utc"}
    if value["status"] == "complete":
        build_keys.add("completed_at_utc")
    _require_exact_keys(build, build_keys, "build")

    try:
        parsed_source = SchemaIndexSource(
            database=source["database"],
            metadata_format_version=source["metadata_format_version"],
            content_fingerprint=source["content_fingerprint"],
            snapshot_id=source["snapshot_id"],
            captured_at_utc=_parse_timestamp(source["captured_at_utc"], "captured_at_utc"),
        )
        parsed_documents = SchemaIndexDocumentEvidence(
            generation_format_version=documents["generation_format_version"],
            expected_count=documents["expected_count"],
            id_digest=documents["id_digest"],
            content_digest=documents["content_digest"],
        )
        completed = (
            _parse_timestamp(build["completed_at_utc"], "completed_at_utc")
            if "completed_at_utc" in build
            else None
        )
        return SchemaIndexManifest(
            status=value["status"],
            source=parsed_source,
            documents=parsed_documents,
            embedding_identity=embedding["identity"],
            build_id=build["build_id"],
            started_at_utc=_parse_timestamp(build["started_at_utc"], "started_at_utc"),
            completed_at_utc=completed,
        )
    except SchemaIndexManifestError:
        raise
    except Exception as exc:
        raise SchemaIndexManifestError("invalid manifest field") from exc


def parse_index_manifest_json(document: str) -> SchemaIndexManifest:
    """Strictly parse a manifest JSON document, rejecting duplicate keys."""

    return parse_index_manifest(_parse_json_object(document))


def _validate_document_ids(document_ids: Iterable[str]) -> list[str]:
    if isinstance(document_ids, (str, bytes)):
        raise SchemaIndexManifestError("document IDs must be an iterable of IDs")
    try:
        values = list(document_ids)
    except TypeError as exc:
        raise SchemaIndexManifestError("document IDs must be iterable") from exc

    for document_id in values:
        _require_document_identifier(document_id)
    if len(set(values)) != len(values):
        raise SchemaIndexManifestError("document IDs must be unique")
    return values


def _validate_document_records(
    documents: Iterable[tuple[str, str]] | Mapping[str, str],
) -> tuple[list[tuple[str, str]], list[str]]:
    if isinstance(documents, Mapping):
        iterable = list(documents.items())
    else:
        if isinstance(documents, (str, bytes)):
            raise SchemaIndexManifestError("documents must be document records")
        try:
            iterable = list(documents)
        except TypeError as exc:
            raise SchemaIndexManifestError("documents must be iterable") from exc

    records: list[tuple[str, str]] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    for record in iterable:
        if not isinstance(record, (tuple, list)) or len(record) != 2:
            raise SchemaIndexManifestError("document records must be ID/text pairs")
        document_id, text = record
        _require_document_identifier(document_id)
        if not isinstance(text, str) or not text.strip():
            raise SchemaIndexManifestError(
                f"invalid document text for document ID {document_id!r}"
            )
        if document_id in seen:
            duplicates.append(document_id)
        else:
            seen.add(document_id)
        records.append((document_id, text))
    return records, duplicates


def _digest(domain: str, payload: Any) -> str:
    encoded = json.dumps(
        {"domain": domain, "payload": payload},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_identifier(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SchemaIndexManifestError(
            f"{field_name} must be a nonempty string without surrounding whitespace"
        )


def _require_document_identifier(value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaIndexManifestError(
            "document ID must be a nonempty, non-whitespace string"
        )


def _require_digest(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or _DIGEST_PATTERN.fullmatch(value) is None:
        raise SchemaIndexManifestError(
            f"{field_name} must be 64 lowercase hexadecimal characters"
        )


def _validate_utc_datetime(value: Any, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise SchemaIndexManifestError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise SchemaIndexManifestError(f"{field_name} must be timezone-aware UTC")


def _serialize_timestamp(value: datetime) -> str:
    _validate_utc_datetime(value, "timestamp")
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_timestamp(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str) or _TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise SchemaIndexManifestError(
            f"{field_name} must use canonical UTC timestamp format"
        )
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise SchemaIndexManifestError(f"{field_name} is invalid") from exc
    return parsed.replace(tzinfo=timezone.utc)


def _parse_json_object(document: str) -> dict[str, Any]:
    if not isinstance(document, str):
        raise SchemaIndexManifestError("JSON document must be text")
    try:
        parsed = json.loads(
            document,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise SchemaIndexManifestError("invalid JSON document") from exc
    if not isinstance(parsed, dict):
        raise SchemaIndexManifestError("JSON document root must be an object")
    return parsed


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON value: {value}")


def _require_exact_keys(
    mapping: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    actual = set(mapping)
    missing = expected - actual
    unexpected = actual - expected
    if missing or unexpected:
        raise SchemaIndexManifestError(
            f"invalid {label} fields"
        )


__all__ = [
    "DOCUMENT_GENERATION_FORMAT_VERSION",
    "DocumentInventoryComparison",
    "INDEX_MANIFEST_FORMAT_VERSION",
    "SchemaIndexDocumentEvidence",
    "SchemaIndexManifest",
    "SchemaIndexManifestError",
    "SchemaIndexSource",
    "compare_document_inventories",
    "compute_document_content_digest",
    "compute_document_identity_digest",
    "parse_index_manifest",
    "parse_index_manifest_json",
    "serialize_index_manifest",
    "validate_schema_index_source",
]
