"""Load authoritative primary-key metadata for optional SQL advisories."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from collections.abc import Mapping


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


def _valid_table_name(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    parts = [part.strip() for part in value.split(".")]
    return len(parts) == 2 and all(parts)


def _normalize_identifier(value: str) -> str:
    return value.strip().strip("[]").replace("[", "").replace("]", "").casefold()


__all__ = ["SchemaKeyMetadata", "load_primary_keys"]
