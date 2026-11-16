from pathlib import Path

import chromadb

from ssa import schema_readiness, schema_retrieval
from ssa.schema_readiness import SchemaIndexStatus, inspect_schema_index


class FakeCollection:
    def __init__(self, count: int):
        self.document_count = count
        self.count_calls = 0

    def count(self) -> int:
        self.count_calls += 1
        return self.document_count


class FakeClient:
    def __init__(self, collection=None, error=None):
        self.collection = collection
        self.error = error
        self.collection_calls: list[str] = []

    def get_collection(self, *, name: str):
        self.collection_calls.append(name)
        if self.error is not None:
            raise self.error
        return self.collection


def test_missing_directory_does_not_construct_chroma_client(tmp_path, monkeypatch):
    missing_path = tmp_path / "missing-index"
    monkeypatch.setattr(schema_retrieval, "CHROMA_PATH", missing_path)

    def fail_factory(path):
        raise AssertionError("Chroma client should not be constructed")

    status = inspect_schema_index(client_factory=fail_factory)

    assert status == SchemaIndexStatus(
        available=False,
        document_count=0,
        message="The schema index directory is missing. Run the schema export, context, and indexing commands.",
    )
    assert not missing_path.exists()


def test_missing_collection_returns_unavailable_status(tmp_path, monkeypatch):
    monkeypatch.setattr(schema_retrieval, "CHROMA_PATH", tmp_path)
    client = FakeClient(error=chromadb.errors.NotFoundError("missing"))
    paths = []

    status = inspect_schema_index(
        client_factory=lambda path: (paths.append(path) or client)
    )

    assert status.available is False
    assert status.document_count == 0
    assert "collection is missing" in status.message
    assert paths == [str(tmp_path)]
    assert client.collection_calls == [schema_retrieval.COLLECTION_NAME]


def test_empty_collection_is_not_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(schema_retrieval, "CHROMA_PATH", tmp_path)
    collection = FakeCollection(0)
    client = FakeClient(collection=collection)

    status = inspect_schema_index(client_factory=lambda path: client)

    assert status.available is False
    assert status.document_count == 0
    assert "collection is empty" in status.message
    assert collection.count_calls == 1


def test_populated_collection_is_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(schema_retrieval, "CHROMA_PATH", tmp_path)
    collection = FakeCollection(12)
    client = FakeClient(collection=collection)

    status = inspect_schema_index(client_factory=lambda path: client)

    assert status == SchemaIndexStatus(available=True, document_count=12)
    assert collection.count_calls == 1


def test_unexpected_chroma_error_is_not_classified_as_empty(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(schema_retrieval, "CHROMA_PATH", tmp_path)
    client = FakeClient(error=RuntimeError("unexpected failure"))

    status = inspect_schema_index(client_factory=lambda path: client)

    assert status.available is False
    assert status.document_count == 0
    assert "unexpected error" in status.message
    assert "empty" not in status.message


def test_inspection_does_not_mutate_existing_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(schema_retrieval, "CHROMA_PATH", tmp_path)
    marker = tmp_path / "existing.marker"
    marker.write_text("unchanged", encoding="utf-8")
    client = FakeClient(collection=FakeCollection(1))

    status = inspect_schema_index(client_factory=lambda path: client)

    assert status.available is True
    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert sorted(path.name for path in Path(tmp_path).iterdir()) == [
        "existing.marker"
    ]


def test_readiness_uses_existing_path_and_collection_configuration(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(schema_retrieval, "CHROMA_PATH", tmp_path)
    client = FakeClient(collection=FakeCollection(1))
    captured = []

    inspect_schema_index(client_factory=lambda path: (captured.append(path) or client))

    assert captured == [str(tmp_path)]
    assert client.collection_calls == [schema_retrieval.COLLECTION_NAME]
