"""Read-only readiness inspection for the local ChromaDB schema index."""

from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError

from ssa import schema_retrieval
from ssa import schema_index
from ssa.schema_embedding import schema_embedding_identity
from ssa.schema_index_manifest_store import (
    SchemaIndexManifestNotFoundError,
    SchemaIndexManifestReadError,
    read_index_manifest,
)
from ssa.schema_index_manifest import (
    compare_document_inventories,
    compute_document_content_digest,
    compute_document_identity_digest,
)


@dataclass(frozen=True)
class SchemaIndexStatus:
    """Whether the configured schema collection is available and populated."""

    available: bool
    document_count: int
    message: str | None = None
    alignment: str = "unknown"


ClientFactory = Callable[[str], object]


def inspect_schema_index(
    *,
    client_factory: ClientFactory | None = None,
    metadata_path: str | Path | None = None,
    expected_database: str | None = None,
) -> SchemaIndexStatus:
    """Inspect the configured ChromaDB collection without creating or changing it."""

    chroma_path = Path(schema_retrieval.CHROMA_PATH)
    if not chroma_path.is_dir():
        return SchemaIndexStatus(
            available=False,
            document_count=0,
            message=(
                "The schema index directory is missing. Run the schema export, "
                "context, and indexing commands."
            ),
        )

    create_client = client_factory or (
        lambda path: chromadb.PersistentClient(path=path)
    )

    guard_path = chroma_path / ".ssa-schema-index-build.lock"
    if guard_path.exists():
        return SchemaIndexStatus(
            available=False,
            document_count=0,
            alignment="incomplete",
            message="Schema indexing is in progress or an indexing lock remains; alignment is unverified.",
        )

    metadata_file = Path(metadata_path) if metadata_path is not None else schema_index.METADATA_PATH
    effective_database = schema_index._resolve_expected_database(expected_database)
    metadata_source = None
    metadata_error = None
    if metadata_file.exists():
        try:
            metadata, metadata_source = schema_index._load_metadata(
                metadata_file,
                expected_database=effective_database,
            )
        except Exception as exc:
            metadata = None
            metadata_error = exc
    else:
        metadata = None
        metadata_error = FileNotFoundError("schema metadata is missing")

    if (
        metadata_error is None
        and isinstance(metadata, dict)
        and "metadata_format_version" in metadata
        and metadata["metadata_format_version"] != 2
    ):
        metadata_error = ValueError("unsupported metadata format version")
        metadata_source = None

    if metadata_error is None and metadata_source is not None and effective_database is None:
        return SchemaIndexStatus(
            available=False,
            document_count=0,
            alignment="invalid",
            message="SQL_DATABASE must be configured to verify version-2 schema alignment.",
        )

    try:
        client = create_client(str(chroma_path))
        collection = client.get_collection(
            name=schema_retrieval.COLLECTION_NAME
        )
        document_count = collection.count()
    except NotFoundError:
        return SchemaIndexStatus(
            available=False,
            document_count=0,
            alignment="unknown",
            message=(
                "The schema collection is missing. Run the schema export, "
                "context, and indexing commands."
            ),
        )
    except Exception:
        return SchemaIndexStatus(
            available=False,
            document_count=0,
            alignment="invalid" if metadata_error else "unknown",
            message=(
                "The schema index could not be inspected because ChromaDB "
                "returned an unexpected error."
            ),
        )

    if document_count < 1:
        return SchemaIndexStatus(
            available=False,
            document_count=0,
            alignment="unknown",
            message=(
                "The schema collection is empty. Run the schema export, "
                "context, and indexing commands."
            ),
        )

    if metadata_error is not None:
        return SchemaIndexStatus(
            available=True,
            document_count=document_count,
            alignment="invalid",
            message="Schema index is available, but metadata validation failed; alignment is unverified.",
        )
    if metadata_source is None:
        return SchemaIndexStatus(
            available=True,
            document_count=document_count,
            alignment="legacy",
            message="Schema index is available; legacy metadata has no verified version-2 alignment evidence.",
        )

    manifest_path = schema_index.schema_index_manifest_path(
        chroma_path,
        schema_retrieval.COLLECTION_NAME,
    )
    try:
        manifest = read_index_manifest(manifest_path)
    except SchemaIndexManifestNotFoundError:
        return SchemaIndexStatus(
            available=True,
            document_count=document_count,
            alignment="unknown",
            message="Schema index is available, but its alignment manifest is missing.",
        )
    except SchemaIndexManifestReadError:
        return SchemaIndexStatus(
            available=True,
            document_count=document_count,
            alignment="invalid",
            message="Schema index is available, but its alignment manifest is invalid.",
        )

    if manifest.status != "complete":
        return SchemaIndexStatus(
            available=True,
            document_count=document_count,
            alignment="incomplete",
            message="Schema index build evidence is incomplete; alignment is unverified.",
        )

    if (
        manifest.source.database != metadata_source.database
        or manifest.source.metadata_format_version != metadata_source.metadata_format_version
        or manifest.source.content_fingerprint != metadata_source.content_fingerprint
        or manifest.embedding_identity != schema_embedding_identity()
        or manifest.documents.generation_format_version != schema_index.DOCUMENT_GENERATION_FORMAT_VERSION
    ):
        return SchemaIndexStatus(
            available=True,
            document_count=document_count,
            alignment="mismatch",
            message="Schema metadata and index manifest do not describe the same verified source.",
        )

    documents, ids = schema_index.create_schema_documents(metadata)
    expected_records = list(zip(ids, documents, strict=True))
    if not ids:
        return SchemaIndexStatus(
            available=True,
            document_count=document_count,
            alignment="invalid",
            message="Validated metadata produced an empty schema document inventory.",
        )
    try:
        actual_records = schema_index._read_inventory(collection)
        comparison = compare_document_inventories(expected_records, actual_records)
        identity_digest = compute_document_identity_digest([item[0] for item in actual_records])
        content_digest = compute_document_content_digest(actual_records)
        schema_index._verify_collection_configuration(collection)
    except Exception:
        return SchemaIndexStatus(
            available=True,
            document_count=document_count,
            alignment="mismatch",
            message="Schema collection contents or configuration could not be verified.",
        )

    if (
        not comparison.matches
        or document_count != len(actual_records)
        or len(actual_records) != manifest.documents.expected_count
        or identity_digest != manifest.documents.id_digest
        or content_digest != manifest.documents.content_digest
    ):
        return SchemaIndexStatus(
            available=True,
            document_count=document_count,
            alignment="mismatch",
            message="Schema collection inventory does not match the verified manifest.",
        )

    if guard_path.exists():
        return SchemaIndexStatus(
            available=False,
            document_count=document_count,
            alignment="incomplete",
            message="Schema indexing activity was detected during inspection; alignment is unverified.",
        )

    try:
        _, final_source = schema_index._load_metadata(
            metadata_file,
            expected_database=effective_database,
        )
        final_manifest = read_index_manifest(manifest_path)
    except Exception:
        return SchemaIndexStatus(
            available=True,
            document_count=document_count,
            alignment="invalid",
            message="Schema evidence changed or became invalid during inspection.",
        )
    if final_source != metadata_source or final_manifest != manifest:
        return SchemaIndexStatus(
            available=True,
            document_count=document_count,
            alignment="mismatch",
            message="Schema metadata or manifest changed during inspection; alignment is unverified.",
        )

    return SchemaIndexStatus(
        available=True,
        document_count=document_count,
        alignment="verified",
        message="Schema metadata and persisted documents are verified as semantically aligned.",
    )
