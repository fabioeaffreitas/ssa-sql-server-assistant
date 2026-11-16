"""Read-only readiness inspection for the local ChromaDB schema index."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError

from ssa import schema_retrieval


@dataclass(frozen=True)
class SchemaIndexStatus:
    """Whether the configured schema collection is available and populated."""

    available: bool
    document_count: int
    message: str | None = None


ClientFactory = Callable[[str], object]


def inspect_schema_index(
    *,
    client_factory: ClientFactory | None = None,
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
            message=(
                "The schema collection is missing. Run the schema export, "
                "context, and indexing commands."
            ),
        )
    except Exception:
        return SchemaIndexStatus(
            available=False,
            document_count=0,
            message=(
                "The schema index could not be inspected because ChromaDB "
                "returned an unexpected error."
            ),
        )

    if document_count < 1:
        return SchemaIndexStatus(
            available=False,
            document_count=0,
            message=(
                "The schema collection is empty. Run the schema export, "
                "context, and indexing commands."
            ),
        )

    return SchemaIndexStatus(
        available=True,
        document_count=document_count,
    )
