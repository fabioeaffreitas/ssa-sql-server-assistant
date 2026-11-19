"""Atomic writer for already assembled versioned schema metadata."""

import json
import os
from pathlib import Path
from collections.abc import Mapping
import tempfile
from typing import Any

from ssa.schema_capture_provenance import parse_capture_provenance
from ssa.schema_key_metadata import fingerprint_schema_metadata


class SchemaMetadataWriteError(RuntimeError):
    """Raised when versioned schema metadata cannot be safely written."""


def write_versioned_schema_metadata(
    metadata: Mapping[str, Any],
    destination: str | Path,
) -> Path:
    """Atomically write a validated versioned metadata document.

    Validation, deterministic serialization, temporary-file readback, and
    fingerprint verification all occur before ``os.replace``. The temporary
    file is created beside the destination so replacement is same-volume and
    Windows-compatible. A successful return means replacement completed; it
    does not guarantee power-loss durability.
    """

    destination_path = Path(destination)
    try:
        validated, serialized = _validate_and_serialize(metadata)
    except SchemaMetadataWriteError:
        raise
    except Exception as exc:
        raise SchemaMetadataWriteError(
            f"schema metadata validation failed for {destination_path}"
        ) from exc

    temporary_path: Path | None = None
    descriptor: int | None = None
    primary_error: BaseException | None = None

    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination_path.name}.",
            suffix=".tmp",
            dir=str(destination_path.parent),
        )
        temporary_path = Path(temporary_name)

        _write_temporary_file(descriptor, serialized)
        descriptor = None

        readback = _readback_document(temporary_path)
        _validate_readback(readback, validated)

        os.replace(temporary_path, destination_path)
        temporary_path = None
        return destination_path
    except Exception as exc:
        primary_error = exc
        if isinstance(exc, SchemaMetadataWriteError):
            raise
        raise SchemaMetadataWriteError(
            f"could not atomically write schema metadata to {destination_path}"
        ) from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except Exception:
                if primary_error is None:
                    raise SchemaMetadataWriteError(
                        "could not close temporary schema metadata file"
                    )
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except Exception:
                # Cleanup failure must never replace the operation's primary
                # error. On a successful operation the path is already None.
                if primary_error is None:
                    raise SchemaMetadataWriteError(
                        "could not remove temporary schema metadata file"
                    )


def _validate_and_serialize(
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    if not isinstance(metadata, Mapping):
        raise SchemaMetadataWriteError("metadata must be a mapping")

    provenance = parse_capture_provenance(metadata)
    computed = fingerprint_schema_metadata(metadata)
    if computed != provenance.content_fingerprint:
        raise SchemaMetadataWriteError(
            "metadata content fingerprint does not match provenance"
        )

    validated = _json_compatible_copy(metadata)
    serialized = _serialize_document(validated)
    return validated, serialized


def _serialize_document(metadata: Mapping[str, Any]) -> str:
    return json.dumps(
        metadata,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )


def _json_compatible_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _json_compatible_copy(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_json_compatible_copy(child) for child in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError("metadata contains a non-JSON-compatible value")


def _write_temporary_file(descriptor: int, serialized: str) -> None:
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())


def _readback_document(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    parsed = json.loads(text, object_pairs_hook=_unique_object)
    if not isinstance(parsed, dict):
        raise ValueError("readback metadata root must be an object")
    return parsed


def _validate_readback(
    readback: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    provenance = parse_capture_provenance(readback)
    computed = fingerprint_schema_metadata(readback)
    if computed != provenance.content_fingerprint:
        raise ValueError("readback content fingerprint mismatch")
    if dict(readback) != dict(expected):
        raise ValueError("readback metadata differs from serialized input")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key in readback")
        result[key] = value
    return result


__all__ = ["SchemaMetadataWriteError", "write_versioned_schema_metadata"]
