"""Atomic persistence for validated schema-index manifest records."""

from collections.abc import Mapping
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from ssa.schema_index_manifest import (
    SchemaIndexManifest,
    SchemaIndexManifestError,
    parse_index_manifest,
    parse_index_manifest_json,
    serialize_index_manifest,
)


class SchemaIndexManifestStoreError(RuntimeError):
    """Raised when a schema-index manifest cannot be read or written safely."""


class SchemaIndexManifestNotFoundError(FileNotFoundError, SchemaIndexManifestStoreError):
    """Raised when the requested manifest file does not exist."""


class SchemaIndexManifestReadError(SchemaIndexManifestStoreError):
    """Raised when a manifest exists but cannot be read or validated."""


def write_index_manifest(
    manifest: SchemaIndexManifest,
    destination: str | Path,
) -> Path:
    """Validate and atomically persist a schema-index manifest."""

    destination_path = Path(destination)
    validated, serialized = _validate_and_serialize(manifest, destination_path)
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

        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = None
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())

        readback = read_index_manifest(temporary_path)
        if readback != validated:
            raise SchemaIndexManifestStoreError(
                f"temporary manifest readback differs from intended manifest "
                f"for {destination_path}"
            )

        os.replace(temporary_path, destination_path)
        temporary_path = None
        return destination_path
    except Exception as exc:
        primary_error = exc
        if isinstance(exc, SchemaIndexManifestStoreError):
            raise
        raise SchemaIndexManifestStoreError(
            f"could not atomically write schema index manifest to {destination_path}"
        ) from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except Exception:
                if primary_error is None:
                    raise SchemaIndexManifestStoreError(
                        "could not close temporary schema index manifest file"
                    )
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except Exception:
                if primary_error is None:
                    raise SchemaIndexManifestStoreError(
                        "could not remove temporary schema index manifest file"
                    )


def read_index_manifest(source: str | Path) -> SchemaIndexManifest:
    """Read and strictly validate a schema-index manifest file."""

    source_path = Path(source)
    try:
        document = source_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SchemaIndexManifestNotFoundError(
            f"schema index manifest does not exist: {source_path}"
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise SchemaIndexManifestReadError(
            f"could not read schema index manifest: {source_path}"
        ) from exc

    try:
        return parse_index_manifest_json(document)
    except Exception as exc:
        if isinstance(exc, SchemaIndexManifestStoreError):
            raise
        raise SchemaIndexManifestReadError(
            f"schema index manifest is invalid: {source_path}"
        ) from exc


def _validate_and_serialize(
    manifest: SchemaIndexManifest,
    destination: Path,
) -> tuple[SchemaIndexManifest, str]:
    try:
        serialized_mapping = serialize_index_manifest(manifest)
        validated = parse_index_manifest(serialized_mapping)
        serialized = json.dumps(
            serialized_mapping,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            separators=(",", ": "),
        )
    except SchemaIndexManifestError as exc:
        raise SchemaIndexManifestStoreError(
            f"schema index manifest validation failed for {destination}"
        ) from exc
    except Exception as exc:
        raise SchemaIndexManifestStoreError(
            f"schema index manifest serialization failed for {destination}"
        ) from exc
    return validated, serialized


__all__ = [
    "SchemaIndexManifestNotFoundError",
    "SchemaIndexManifestReadError",
    "SchemaIndexManifestStoreError",
    "read_index_manifest",
    "write_index_manifest",
]
