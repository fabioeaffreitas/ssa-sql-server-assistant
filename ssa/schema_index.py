
"""
SSA - Schema Indexing
"""

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

import chromadb
from chromadb.errors import NotFoundError

from ssa.schema_embedding import (
    create_schema_embedding_function,
    schema_embedding_identity,
)
from ssa.schema_index_manifest import (
    SchemaIndexDocumentEvidence,
    SchemaIndexManifest,
    SchemaIndexSource,
    compare_document_inventories,
    compute_document_content_digest,
    compute_document_identity_digest,
    validate_schema_index_source,
)
from ssa.schema_index_manifest_store import (
    SchemaIndexManifestNotFoundError,
    SchemaIndexManifestReadError,
    read_index_manifest,
    write_index_manifest,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
METADATA_PATH = ROOT_DIR / "data" / "schema_metadata.json"
CHROMA_PATH = ROOT_DIR / "data" / "chroma_db"

COLLECTION_NAME = "ssa_schema"
DOCUMENT_GENERATION_FORMAT_VERSION = 1
_GUARD_NAME = ".ssa-schema-index-build.lock"


def _key_lines(obj):
    """Return conservative, human-readable declarations for table keys."""

    lines = []

    primary_keys = obj.get("primary_keys")
    if isinstance(primary_keys, list) and primary_keys and all(
        isinstance(column, str) and column.strip()
        for column in primary_keys
    ):
        columns = [column.strip() for column in primary_keys]
        primary = columns[0] if len(columns) == 1 else f"({', '.join(columns)})"
        lines.append(f"- Primary key: {primary}")

    foreign_keys = obj.get("foreign_keys")
    if isinstance(foreign_keys, list) and foreign_keys:
        formatted_foreign_keys = []
        for foreign_key in foreign_keys:
            if not isinstance(foreign_key, dict):
                formatted_foreign_keys = []
                break

            columns = foreign_key.get("columns")
            referenced_columns = foreign_key.get("referenced_columns")
            referenced_schema = foreign_key.get("referenced_schema")
            referenced_table = foreign_key.get("referenced_table")

            if (
                not isinstance(columns, list)
                or not columns
                or not all(
                    isinstance(column, str) and column.strip()
                    for column in columns
                )
                or not isinstance(referenced_columns, list)
                or len(referenced_columns) != len(columns)
                or not all(
                    isinstance(column, str) and column.strip()
                    for column in referenced_columns
                )
                or not isinstance(referenced_schema, str)
                or not referenced_schema.strip()
                or not isinstance(referenced_table, str)
                or not referenced_table.strip()
            ):
                formatted_foreign_keys = []
                break

            parent = [column.strip() for column in columns]
            referenced = [column.strip() for column in referenced_columns]
            parent_text = parent[0] if len(parent) == 1 else f"({', '.join(parent)})"
            referenced_text = ", ".join(referenced)
            formatted_foreign_keys.append(
                "- Foreign key: "
                f"{parent_text} -> "
                f"{referenced_schema.strip()}.{referenced_table.strip()}"
                f"({referenced_text})"
            )

        if formatted_foreign_keys:
            lines.extend(formatted_foreign_keys)

    return lines


def create_schema_documents(metadata):
    documents = []
    ids = []

    for object_type in ("tables", "views"):
        for object_name, obj in metadata.get(object_type, {}).items():
            lines = [
                f"Object type: {object_type[:-1]}",
                f"Name: {object_name}",
            ]

            description = obj.get("description")
            if description:
                lines.append(f"Description: {description}")

            lines.append("Columns:")

            for column in obj.get("columns", []):
                column_name = column.get("name", "")
                data_type = column.get("data_type", "")
                column_description = column.get("description", "")

                line = f"- {column_name} ({data_type})"

                if column_description:
                    line += f": {column_description}"

                lines.append(line)

            if object_type == "tables":
                key_lines = _key_lines(obj)
                if key_lines:
                    lines.append("Keys:")
                    lines.extend(key_lines)

            document = "\n".join(lines)

            documents.append(document)
            ids.append(f"{object_type[:-1]}:{object_name}")

    return documents, ids


def schema_index_manifest_path(
    chroma_path: str | Path,
    collection_name: str,
) -> Path:
    """Return a safe, persistence-directory-local manifest path."""

    if not isinstance(collection_name, str) or not collection_name:
        raise ValueError("collection_name must be a nonempty string")
    digest = hashlib.sha256(collection_name.encode("utf-8")).hexdigest()[:24]
    return Path(chroma_path) / f"ssa_index_manifest_{digest}.json"


class _ExclusiveBuildGuard:
    def __init__(self, directory: Path):
        self.path = directory / _GUARD_NAME
        self._owned = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            raise RuntimeError(
                f"SSA schema indexing is already active for {self.path.parent}"
            ) from exc
        try:
            with os.fdopen(descriptor, "w", encoding="ascii") as handle:
                handle.write(f"pid={os.getpid()}\n")
            self._owned = True
        except Exception:
            try:
                self.path.unlink()
            except OSError:
                pass
            raise

    def release(self) -> None:
        if not self._owned:
            return
        self.path.unlink()
        self._owned = False


def _strict_json_object(document: str) -> dict:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    parsed = json.loads(
        document,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    if not isinstance(parsed, dict):
        raise ValueError("schema metadata root must be an object")
    return parsed


def _load_metadata(
    path: Path,
    expected_database: str | None = None,
) -> tuple[dict, SchemaIndexSource | None]:
    if not path.exists():
        raise FileNotFoundError(f"Schema metadata not found: {path}")
    metadata = _strict_json_object(path.read_text(encoding="utf-8"))
    if metadata.get("metadata_format_version") == 2:
        return metadata, validate_schema_index_source(
            metadata,
            expected_database=expected_database,
        )
    return metadata, None


def _resolve_expected_database(explicit: str | None) -> str | None:
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise ValueError("expected_database must not be empty")
        return explicit.strip()
    configured = os.getenv("SQL_DATABASE")
    if configured is None or not configured.strip():
        return None
    return configured.strip()


def _document_evidence(
    documents: list[str],
    ids: list[str],
) -> SchemaIndexDocumentEvidence:
    if not ids:
        raise ValueError("schema index document inventory must not be empty")
    records = list(zip(ids, documents, strict=True))
    return SchemaIndexDocumentEvidence(
        generation_format_version=DOCUMENT_GENERATION_FORMAT_VERSION,
        expected_count=len(ids),
        id_digest=compute_document_identity_digest(ids),
        content_digest=compute_document_content_digest(records),
    )


def _collection_configuration(collection) -> Mapping:
    configuration = getattr(collection, "configuration_json", None)
    if callable(configuration):
        configuration = configuration()
    if not isinstance(configuration, Mapping):
        raise RuntimeError("collection configuration is unavailable")
    return configuration


def _verify_collection_configuration(collection) -> None:
    configuration = _collection_configuration(collection)
    embedding = configuration.get("embedding_function")
    hnsw = configuration.get("hnsw")
    metadata_space = configuration.get("hnsw:space")
    space = metadata_space
    if isinstance(hnsw, Mapping):
        configured_space = hnsw.get("space")
        if (
            configured_space is not None
            and metadata_space is not None
            and configured_space != metadata_space
        ):
            raise RuntimeError("collection distance configuration is contradictory")
        space = configured_space if configured_space is not None else metadata_space
    if not isinstance(embedding, Mapping) or embedding.get("name") != "default":
        raise RuntimeError("collection embedding configuration is not the SSA default")
    if "type" in embedding and embedding["type"] != "known":
        raise RuntimeError("collection embedding configuration type is unsupported")
    if "config" in embedding and embedding["config"] != {}:
        raise RuntimeError("collection default embedding configuration is incompatible")
    if space != "l2":
        raise RuntimeError("collection distance space is not l2")


def _read_inventory(collection) -> list[tuple[str, str]]:
    result = collection.get(include=["documents"])
    if not isinstance(result, Mapping):
        raise RuntimeError("collection inventory response is invalid")
    ids = result.get("ids")
    documents = result.get("documents")
    if (
        not isinstance(ids, list)
        or not isinstance(documents, list)
        or len(ids) != len(documents)
    ):
        raise RuntimeError("collection inventory response is incomplete")
    if not all(isinstance(item, str) for item in ids) or not all(
        isinstance(item, str) for item in documents
    ):
        raise RuntimeError("collection inventory contains invalid records")
    return list(zip(ids, documents, strict=True))


def _get_existing_collection(client, name: str):
    try:
        return client.get_collection(
            name=name,
            embedding_function=create_schema_embedding_function(),
        )
    except (AttributeError, NotFoundError):
        return None


def _get_or_create_collection(client, name: str):
    return client.get_or_create_collection(
        name=name,
        embedding_function=create_schema_embedding_function(),
        metadata={"hnsw:space": "l2"},
    )


def _invalidate_legacy_manifest(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RuntimeError(f"could not invalidate legacy index manifest: {path}") from exc


def index_schema(
    *,
    metadata_path: str | Path | None = None,
    chroma_path: str | Path | None = None,
    collection_name: str | None = None,
    client_factory: Callable[[str], object] | None = None,
    manifest_writer: Callable[[SchemaIndexManifest, str | Path], Path] | None = None,
    clock: Callable[[], datetime] | None = None,
    build_id_factory: Callable[[], uuid.UUID | str] | None = None,
    expected_database: str | None = None,
    rebuild: bool = False,
) -> None:
    metadata_file = Path(metadata_path) if metadata_path is not None else METADATA_PATH
    persistence_path = Path(chroma_path) if chroma_path is not None else CHROMA_PATH
    name = collection_name if collection_name is not None else COLLECTION_NAME
    manifest_path = schema_index_manifest_path(persistence_path, name)
    guard = _ExclusiveBuildGuard(persistence_path)
    guard.acquire()
    primary_error: BaseException | None = None
    try:
        publish_manifest = manifest_writer or write_index_manifest
        effective_expected_database = _resolve_expected_database(expected_database)
        metadata, source = _load_metadata(
            metadata_file,
            expected_database=effective_expected_database,
        )
        documents, ids = create_schema_documents(metadata)

        if source is None:
            if rebuild:
                raise ValueError(
                    "rebuild=True is supported only for version-2 metadata"
                )
            _invalidate_legacy_manifest(manifest_path)
            client = (
                client_factory
                or (lambda path: chromadb.PersistentClient(path=path))
            )(str(persistence_path))
            collection = _get_existing_collection(client, name)
            if collection is None:
                collection = _get_or_create_collection(client, name)
            collection.upsert(documents=documents, ids=ids)
            _report_success(len(documents), name, persistence_path)
            return

        if effective_expected_database is None:
            raise ValueError(
                "SQL_DATABASE must be configured for version-2 schema indexing"
            )

        evidence = _document_evidence(documents, ids)
        embedding_identity = schema_embedding_identity()
        prior_manifest = None
        try:
            prior_manifest = read_index_manifest(manifest_path)
        except (SchemaIndexManifestNotFoundError, SchemaIndexManifestReadError):
            pass
        now = clock or (lambda: datetime.now(timezone.utc))
        started_at = now()
        build_factory = build_id_factory or uuid.uuid4
        build_id = str(build_factory())
        incomplete = SchemaIndexManifest(
            status="incomplete",
            source=source,
            documents=evidence,
            build_id=build_id,
            started_at_utc=started_at,
            embedding_identity=embedding_identity,
        )
        publish_manifest(incomplete, manifest_path)

        client = (
            client_factory
            or (lambda path: chromadb.PersistentClient(path=path))
        )(str(persistence_path))
        existing = _get_existing_collection(client, name)
        if existing is not None and not rebuild:
            if (
                prior_manifest is None
                or prior_manifest.status != "complete"
                or prior_manifest.source.database != source.database
                or (
                    prior_manifest.documents.generation_format_version
                    != DOCUMENT_GENERATION_FORMAT_VERSION
                )
                or prior_manifest.embedding_identity != embedding_identity
            ):
                raise RuntimeError("existing collection requires explicit rebuild")
            _verify_collection_configuration(existing)
            previous = _read_inventory(existing)
            expected_ids = set(ids)
            stale_ids = [document_id for document_id, _ in previous if document_id not in expected_ids]
            if stale_ids:
                existing.delete(ids=stale_ids)
        if existing is not None and rebuild:
            client.delete_collection(name=name)
        collection = _get_or_create_collection(client, name)
        collection.upsert(documents=documents, ids=ids)

        actual = _read_inventory(collection)
        comparison = compare_document_inventories(
            list(zip(ids, documents, strict=True)),
            actual,
        )
        if not comparison.matches:
            raise RuntimeError(
                "persisted schema collection inventory does not match expected documents"
            )
        if compute_document_identity_digest([item[0] for item in actual]) != evidence.id_digest:
            raise RuntimeError("persisted schema collection identity digest mismatch")
        if compute_document_content_digest(actual) != evidence.content_digest:
            raise RuntimeError("persisted schema collection content digest mismatch")
        _verify_collection_configuration(collection)

        _, current_source = _load_metadata(
            metadata_file,
            expected_database=effective_expected_database,
        )
        if current_source != source:
            raise RuntimeError("schema metadata source changed during indexing")
        completed_at = now()
        complete = SchemaIndexManifest(
            status="complete",
            source=source,
            documents=evidence,
            build_id=build_id,
            started_at_utc=started_at,
            completed_at_utc=completed_at,
            embedding_identity=embedding_identity,
        )
        publish_manifest(complete, manifest_path)
        _report_success(len(documents), name, persistence_path)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            guard.release()
        except Exception:
            if primary_error is None:
                raise


def _report_success(count: int, name: str, persistence_path: Path) -> None:
    print("SSA Schema Indexing")
    print("------------------")
    print(f"Documents indexed: {count}")
    print(f"Collection: {name}")
    print(f"Database: {persistence_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Index SSA schema documents")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="explicitly rebuild an existing Chroma collection",
    )
    index_schema(rebuild=parser.parse_args().rebuild)
