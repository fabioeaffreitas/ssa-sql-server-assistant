"""Load authoritative primary-key metadata for optional SQL advisories."""

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from collections.abc import Mapping
from typing import Any


_FINGERPRINT_DOMAIN = "ssa-schema-metadata-fingerprint-v1"
_VOLATILE_FINGERPRINT_FIELDS = frozenset(
    {
        "captured_at_utc",
        "captured_at",
        "capture_timestamp",
        "capture_time_utc",
        "export_run_id",
        "snapshot_id",
        "content_fingerprint",
    }
)


@dataclass(frozen=True)
class SchemaKeyMetadata:
    """Validated primary-key metadata or a safe unavailable outcome."""

    available: bool
    reason: str
    database: str | None = None
    primary_keys: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


def load_primary_keys(
    path: str | Path,
    expected_database: str | None,
) -> SchemaKeyMetadata:
    """Load primary keys from the existing exported schema JSON.

    The loader performs no database or Chroma access. A matching database name
    establishes identity consistency only; it does not establish freshness.
    """

    if not isinstance(expected_database, str) or not expected_database.strip():
        return SchemaKeyMetadata(False, "missing_expected_database")

    metadata_path = Path(path)
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return SchemaKeyMetadata(False, "metadata_unavailable")
    except json.JSONDecodeError:
        return SchemaKeyMetadata(False, "malformed_json")

    if not isinstance(payload, dict):
        return SchemaKeyMetadata(False, "invalid_root")

    database = payload.get("database")
    if not isinstance(database, str) or not database.strip():
        return SchemaKeyMetadata(False, "invalid_database")
    if database.strip().casefold() != expected_database.strip().casefold():
        return SchemaKeyMetadata(False, "database_mismatch", database.strip())

    tables = payload.get("tables")
    if not isinstance(tables, dict):
        return SchemaKeyMetadata(False, "invalid_tables", database.strip())

    primary_keys: dict[str, tuple[str, ...]] = {}
    normalized_names: set[str] = set()
    for table_name, table in tables.items():
        if not _valid_table_name(table_name):
            return SchemaKeyMetadata(False, "invalid_table_identifier", database.strip())
        if not isinstance(table, dict):
            return SchemaKeyMetadata(False, "invalid_table_mapping", database.strip())
        if "primary_keys" not in table:
            return SchemaKeyMetadata(False, "missing_primary_keys", database.strip())

        keys = table["primary_keys"]
        if not isinstance(keys, list):
            return SchemaKeyMetadata(False, "invalid_primary_keys", database.strip())

        normalized_name = _normalize_identifier(table_name)
        if normalized_name in normalized_names:
            return SchemaKeyMetadata(False, "duplicate_table_identifier", database.strip())
        normalized_names.add(normalized_name)

        if not keys:
            continue
        if not all(isinstance(key, str) and key.strip() for key in keys):
            return SchemaKeyMetadata(False, "invalid_primary_keys", database.strip())

        normalized_keys = tuple(key.strip() for key in keys)
        normalized_key_names = [_normalize_identifier(key) for key in normalized_keys]
        if len(set(normalized_key_names)) != len(normalized_key_names):
            return SchemaKeyMetadata(False, "duplicate_primary_key", database.strip())
        primary_keys[table_name] = normalized_keys

    return SchemaKeyMetadata(
        True,
        "available",
        database.strip(),
        primary_keys,
    )


def fingerprint_schema_metadata(metadata: Mapping[str, Any]) -> str:
    """Return a canonical SHA-256 fingerprint for schema metadata.

    The fingerprint covers the complete JSON-compatible semantic metadata
    document, including database identity, tables, views, columns, keys, and
    any other non-volatile fields supplied by the caller. Object keys are
    sorted recursively, while array order is preserved because catalog and
    constraint ordering can be meaningful. The versioned domain separates
    this canonicalization from future algorithms.

    Only these explicitly named root-level provenance fields are excluded:
    ``captured_at_utc``, ``captured_at``, ``capture_timestamp``,
    ``capture_time_utc``, ``export_run_id``, ``snapshot_id``, and
    ``content_fingerprint``. Unknown fields are retained and therefore cannot
    be silently omitted from the fingerprint.

    This validates an offline metadata shape; it does not prove SQL Server
    truth, provenance, or freshness. A mapping cannot reveal duplicate keys
    from JSON that has already been parsed. Use
    :func:`fingerprint_schema_metadata_json` at the JSON-text boundary when
    duplicate-key rejection is required.
    """

    canonical = _validated_metadata_for_fingerprint(metadata)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    domain = _FINGERPRINT_DOMAIN.encode("utf-8")
    return hashlib.sha256(domain + b"\n" + encoded).hexdigest()


def fingerprint_schema_metadata_json(document: str) -> str:
    """Parse JSON with duplicate-key rejection and fingerprint the metadata."""

    if not isinstance(document, str):
        raise ValueError("metadata JSON must be a string")

    try:
        metadata = json.loads(document, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid schema metadata JSON") from exc

    return fingerprint_schema_metadata(metadata)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _validated_metadata_for_fingerprint(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("schema metadata root must be an object")

    canonical = _validate_json_value(metadata, "$")
    if not isinstance(canonical, dict):
        raise ValueError("schema metadata root must be an object")

    database = canonical.get("database")
    if not isinstance(database, str) or not database.strip():
        raise ValueError("schema metadata database must be a nonempty string")

    for collection_name in ("tables", "views"):
        collection = canonical.get(collection_name)
        if not isinstance(collection, dict):
            raise ValueError(f"schema metadata {collection_name} must be an object")
        for object_name, obj in collection.items():
            _validate_schema_object(collection_name, object_name, obj)

    _validate_optional_count(canonical, "table_count", len(canonical["tables"]))
    _validate_optional_count(canonical, "view_count", len(canonical["views"]))
    foreign_key_count = sum(
        len(obj.get("foreign_keys", []))
        for obj in canonical["tables"].values()
        if isinstance(obj, dict) and isinstance(obj.get("foreign_keys", []), list)
    )
    _validate_optional_count(canonical, "foreign_key_count", foreign_key_count)

    return {
        key: value
        for key, value in canonical.items()
        if key not in _VOLATILE_FINGERPRINT_FIELDS
    }


def _validate_json_value(value: Any, path: str) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"non-string metadata key at {path}")
            result[key] = _validate_json_value(child, f"{path}.{key}")
        return result

    if isinstance(value, list):
        return [
            _validate_json_value(child, f"{path}[{index}]")
            for index, child in enumerate(value)
        ]

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite number at {path}")
        return value

    raise ValueError(f"unsupported metadata value at {path}")


def _validate_schema_object(
    object_type: str,
    object_name: str,
    obj: Any,
) -> None:
    if not isinstance(object_name, str) or not object_name.strip():
        raise ValueError(f"invalid {object_type} identifier")
    if not isinstance(obj, dict):
        raise ValueError(f"invalid {object_type} mapping for {object_name}")

    columns = obj.get("columns")
    if not isinstance(columns, list):
        raise ValueError(f"invalid columns for {object_name}")
    for column in columns:
        if not isinstance(column, dict):
            raise ValueError(f"invalid column in {object_name}")
        name = column.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"invalid column name in {object_name}")
        if "data_type" in column and (
            not isinstance(column["data_type"], str)
            or not column["data_type"].strip()
        ):
            raise ValueError(f"invalid column data type in {object_name}")

    if object_type == "tables" and "primary_keys" not in obj:
        raise ValueError(f"missing primary_keys for {object_name}")

    if "primary_keys" in obj:
        _validate_string_list(obj["primary_keys"], "primary_keys", object_name)

    if "foreign_keys" in obj:
        foreign_keys = obj["foreign_keys"]
        if not isinstance(foreign_keys, list):
            raise ValueError(f"invalid foreign_keys for {object_name}")
        for foreign_key in foreign_keys:
            if not isinstance(foreign_key, dict):
                raise ValueError(f"invalid foreign key in {object_name}")
            for field_name in (
                "name",
                "referenced_schema",
                "referenced_table",
            ):
                field_value = foreign_key.get(field_name)
                if not isinstance(field_value, str) or not field_value.strip():
                    raise ValueError(
                        f"invalid foreign key {field_name} in {object_name}"
                    )
            columns = foreign_key.get("columns")
            referenced_columns = foreign_key.get("referenced_columns")
            _validate_string_list(columns, "columns", object_name)
            _validate_string_list(
                referenced_columns,
                "referenced_columns",
                object_name,
            )
            if len(columns) != len(referenced_columns):
                raise ValueError(f"foreign key column count mismatch in {object_name}")


def _validate_string_list(value: Any, field_name: str, object_name: str) -> None:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"invalid {field_name} in {object_name}")


def _validate_optional_count(
    metadata: Mapping[str, Any],
    field_name: str,
    expected: int,
) -> None:
    if field_name not in metadata:
        return
    value = metadata[field_name]
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"invalid {field_name}")


def _valid_table_name(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    parts = [part.strip() for part in value.split(".")]
    return len(parts) == 2 and all(parts)


def _normalize_identifier(value: str) -> str:
    return value.strip().strip("[]").replace("[", "").replace("]", "").casefold()


__all__ = [
    "SchemaKeyMetadata",
    "fingerprint_schema_metadata",
    "fingerprint_schema_metadata_json",
    "load_primary_keys",
]
