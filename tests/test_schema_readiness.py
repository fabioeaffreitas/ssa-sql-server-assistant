from pathlib import Path
from datetime import datetime, timezone
import json

import chromadb
import pytest

from ssa import schema_readiness, schema_retrieval
from ssa import schema_index
from ssa.schema_capture_assembly import assemble_versioned_schema_metadata
from ssa.schema_capture_provenance import SQLServerIdentityEvidence
from ssa.schema_index_manifest import SchemaIndexManifest, validate_schema_index_source
from ssa.schema_index_manifest_store import write_index_manifest
from ssa.schema_embedding import schema_embedding_identity
from ssa.schema_readiness import SchemaIndexStatus, inspect_schema_index


class FakeCollection:
    configuration_json = {
        "embedding_function": {"name": "default", "type": "known", "config": {}},
        "hnsw": {"space": "l2"},
    }

    def __init__(self, count: int, records=None):
        self.document_count = count
        self.count_calls = 0
        self.records = records or []
        self.mutations = []

    def count(self) -> int:
        self.count_calls += 1
        return self.document_count

    def get(self, **kwargs):
        return {
            "ids": [item[0] for item in self.records],
            "documents": [item[1] for item in self.records],
        }

    def upsert(self, **kwargs):
        self.mutations.append("upsert")

    def delete(self, **kwargs):
        self.mutations.append("delete")


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
    monkeypatch.setattr(schema_index, "METADATA_PATH", tmp_path / "missing-schema.json")
    collection = FakeCollection(12)
    client = FakeClient(collection=collection)

    status = inspect_schema_index(client_factory=lambda path: client)

    assert status.available is True
    assert status.document_count == 12
    assert status.alignment == "invalid"
    assert "metadata validation failed" in status.message
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


def _versioned_metadata():
    return assemble_versioned_schema_metadata(
        {
            "database": "AdventureWorks2025",
            "tables": {
                "dbo.Item": {
                    "schema": "dbo",
                    "table": "Item",
                    "description": None,
                    "columns": [{"name": "ID", "data_type": "int", "description": None}],
                    "primary_keys": ["ID"],
                    "foreign_keys": [],
                }
            },
            "views": {},
            "table_count": 1,
            "view_count": 0,
            "foreign_key_count": 0,
        },
        snapshot_id="capture-1",
        captured_at_utc=datetime(2026, 11, 26, 12, 0, tzinfo=timezone.utc),
        server_identity=SQLServerIdentityEvidence(server_name="SQLHOST"),
        server_version="17.0.1000.7",
    )


def _write_verified_fixture(tmp_path, *, records=None):
    metadata = _versioned_metadata()
    metadata_path = tmp_path / "schema.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    documents, ids = schema_index.create_schema_documents(metadata)
    source = validate_schema_index_source(metadata, expected_database="AdventureWorks2025")
    evidence = schema_index._document_evidence(documents, ids)
    manifest = SchemaIndexManifest(
        status="complete",
        source=source,
        documents=evidence,
        build_id="build-1",
        started_at_utc=datetime(2026, 11, 26, 12, 1, tzinfo=timezone.utc),
        completed_at_utc=datetime(2026, 11, 26, 12, 2, tzinfo=timezone.utc),
        embedding_identity=schema_embedding_identity(),
    )
    chroma_path = tmp_path / "chroma"
    chroma_path.mkdir()
    write_index_manifest(
        manifest,
        schema_index.schema_index_manifest_path(chroma_path, schema_retrieval.COLLECTION_NAME),
    )
    return metadata_path, chroma_path, list(zip(ids, documents)) if records is None else records


def test_verified_versioned_alignment(tmp_path, monkeypatch):
    metadata_path, chroma_path, records = _write_verified_fixture(tmp_path)
    monkeypatch.setattr(schema_retrieval, "CHROMA_PATH", chroma_path)
    status = inspect_schema_index(
        metadata_path=metadata_path,
        expected_database="AdventureWorks2025",
        client_factory=lambda path: FakeClient(FakeCollection(len(records), records)),
    )
    assert status.available is True
    assert status.alignment == "verified"


def test_legacy_alignment_is_explicitly_unverified(tmp_path, monkeypatch):
    metadata_path = tmp_path / "schema.json"
    metadata_path.write_text(json.dumps({"tables": {"dbo.Item": {"columns": []}}}), encoding="utf-8")
    monkeypatch.setattr(schema_retrieval, "CHROMA_PATH", tmp_path)
    status = inspect_schema_index(
        metadata_path=metadata_path,
        client_factory=lambda path: FakeClient(FakeCollection(1)),
    )
    assert status.available is True
    assert status.alignment == "legacy"
    assert "no verified" in status.message


def test_missing_manifest_is_unverified(tmp_path, monkeypatch):
    metadata_path = _versioned_metadata()
    metadata_file = tmp_path / "schema.json"
    metadata_file.write_text(json.dumps(metadata_path), encoding="utf-8")
    monkeypatch.setattr(schema_retrieval, "CHROMA_PATH", tmp_path)
    status = inspect_schema_index(
        metadata_path=metadata_file,
        expected_database="AdventureWorks2025",
        client_factory=lambda path: FakeClient(FakeCollection(1)),
    )
    assert status.alignment == "unknown"
    assert "manifest is missing" in status.message


def test_build_guard_blocks_verified_readiness(tmp_path, monkeypatch):
    _, chroma_path, _ = _write_verified_fixture(tmp_path)
    (chroma_path / ".ssa-schema-index-build.lock").write_text("active", encoding="ascii")
    monkeypatch.setattr(schema_retrieval, "CHROMA_PATH", chroma_path)
    status = inspect_schema_index(client_factory=lambda path: pytest.fail("client constructed"))
    assert status.alignment == "incomplete"
    assert status.available is False


def test_inventory_mismatch_is_not_verified_and_is_read_only(tmp_path, monkeypatch):
    metadata_path, chroma_path, records = _write_verified_fixture(tmp_path)
    collection = FakeCollection(len(records), [(records[0][0], "changed")])
    monkeypatch.setattr(schema_retrieval, "CHROMA_PATH", chroma_path)
    status = inspect_schema_index(
        metadata_path=metadata_path,
        expected_database="AdventureWorks2025",
        client_factory=lambda path: FakeClient(collection),
    )
    assert status.alignment == "mismatch"
    assert collection.mutations == []


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(metadata_format_version=3),
    lambda value: value["provenance"].update(content_fingerprint="a" * 64),
])
def test_invalid_metadata_prevents_verified_alignment_and_client_inspection(
    tmp_path, monkeypatch, mutation
):
    metadata = _versioned_metadata()
    mutation(metadata)
    metadata_file = tmp_path / "schema.json"
    metadata_file.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(schema_retrieval, "CHROMA_PATH", tmp_path)
    calls = []
    status = inspect_schema_index(
        metadata_path=metadata_file,
        expected_database="AdventureWorks2025",
        client_factory=lambda path: calls.append(path),
    )
    assert status.alignment == "invalid"
    assert calls == [str(tmp_path)]


@pytest.mark.parametrize("document", [
    '{"metadata_format_version": 2, "metadata_format_version": 2}',
    '{"metadata_format_version": 2, "value": NaN}',
    "not-json",
])
def test_strict_metadata_json_failures_are_unverified(tmp_path, monkeypatch, document):
    metadata_file = tmp_path / "schema.json"
    metadata_file.write_text(document, encoding="utf-8")
    monkeypatch.setattr(schema_retrieval, "CHROMA_PATH", tmp_path)
    status = inspect_schema_index(
        metadata_path=metadata_file,
        expected_database="AdventureWorks2025",
        client_factory=lambda path: FakeClient(FakeCollection(1)),
    )
    assert status.alignment == "invalid"


def test_versioned_readiness_requires_configured_database(tmp_path, monkeypatch):
    metadata_file = tmp_path / "schema.json"
    metadata_file.write_text(json.dumps(_versioned_metadata()), encoding="utf-8")
    monkeypatch.delenv("SQL_DATABASE", raising=False)
    monkeypatch.setattr(schema_retrieval, "CHROMA_PATH", tmp_path)
    status = inspect_schema_index(
        metadata_path=metadata_file,
        client_factory=lambda path: pytest.fail("client should not be constructed"),
    )
    assert status.alignment == "invalid"
    assert "SQL_DATABASE" in status.message


def test_expected_database_mismatch_is_not_verified(tmp_path, monkeypatch):
    metadata_file, chroma_path, records = _write_verified_fixture(tmp_path)
    monkeypatch.setattr(schema_retrieval, "CHROMA_PATH", chroma_path)
    status = inspect_schema_index(
        metadata_path=metadata_file,
        expected_database="OtherDatabase",
        client_factory=lambda path: FakeClient(FakeCollection(1)),
    )
    assert status.alignment == "invalid"


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(status="incomplete"),
    lambda value: value["source"].update(database="OtherDatabase"),
    lambda value: value["source"].update(content_fingerprint="b" * 64),
    lambda value: value["documents"].update(generation_format_version=2),
    lambda value: value["documents"].update(expected_count=99),
    lambda value: value["documents"].update(id_digest="d" * 64),
    lambda value: value["documents"].update(content_digest="e" * 64),
    lambda value: value["embedding"].update(identity="unsupported"),
])
def test_manifest_evidence_failures_are_not_verified(tmp_path, monkeypatch, mutation):
    metadata_file, chroma_path, records = _write_verified_fixture(tmp_path)
    manifest_file = schema_index.schema_index_manifest_path(
        chroma_path, schema_retrieval.COLLECTION_NAME
    )
    value = json.loads(manifest_file.read_text(encoding="utf-8"))
    mutation(value)
    manifest_file.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(schema_retrieval, "CHROMA_PATH", chroma_path)
    status = inspect_schema_index(
        metadata_path=metadata_file,
        expected_database="AdventureWorks2025",
        client_factory=lambda path: FakeClient(FakeCollection(len(records), records)),
    )
    assert status.alignment in {"invalid", "incomplete", "mismatch"}
    assert status.alignment != "verified"


def test_invalid_manifest_json_is_not_verified(tmp_path, monkeypatch):
    metadata_file, chroma_path, records = _write_verified_fixture(tmp_path)
    manifest_file = schema_index.schema_index_manifest_path(
        chroma_path, schema_retrieval.COLLECTION_NAME
    )
    manifest_file.write_text("{", encoding="utf-8")
    monkeypatch.setattr(schema_retrieval, "CHROMA_PATH", chroma_path)
    status = inspect_schema_index(
        metadata_path=metadata_file,
        expected_database="AdventureWorks2025",
        client_factory=lambda path: FakeClient(FakeCollection(len(records), records)),
    )
    assert status.alignment == "invalid"


@pytest.mark.parametrize("case", [
    "missing",
    "unexpected",
    "duplicate",
    "changed",
    "count",
    "malformed",
    "configuration",
])
def test_actual_collection_evidence_failures_are_not_verified(tmp_path, monkeypatch, case):
    metadata_file, chroma_path, records = _write_verified_fixture(tmp_path)
    actual = list(records)
    if case == "missing":
        actual = []
    elif case == "unexpected":
        actual.append(("table:dbo.Other", "other"))
    elif case == "duplicate":
        actual.append(records[0])
    elif case == "changed":
        actual[0] = (actual[0][0], "changed")
    collection = FakeCollection(len(actual), actual)
    if case == "count":
        collection.document_count += 1
    if case == "configuration":
        collection.configuration_json = {}
    if case == "malformed":
        collection.get = lambda **kwargs: {"ids": "bad", "documents": []}
    monkeypatch.setattr(schema_retrieval, "CHROMA_PATH", chroma_path)
    status = inspect_schema_index(
        metadata_path=metadata_file,
        expected_database="AdventureWorks2025",
        client_factory=lambda path: FakeClient(collection),
    )
    assert status.alignment != "verified"


def test_guard_appearing_during_inventory_inspection_prevents_verified_status(tmp_path, monkeypatch):
    metadata_file, chroma_path, records = _write_verified_fixture(tmp_path)
    guard = chroma_path / ".ssa-schema-index-build.lock"
    original_get = FakeCollection.get

    def get_and_create_guard(self, **kwargs):
        guard.write_text("appeared", encoding="ascii")
        return original_get(self, **kwargs)

    monkeypatch.setattr(FakeCollection, "get", get_and_create_guard)
    monkeypatch.setattr(schema_retrieval, "CHROMA_PATH", chroma_path)
    status = inspect_schema_index(
        metadata_path=metadata_file,
        expected_database="AdventureWorks2025",
        client_factory=lambda path: FakeClient(FakeCollection(len(records), records)),
    )
    assert status.alignment == "incomplete"


@pytest.mark.parametrize("configuration", [
    {"embedding_function": {"name": "other"}, "hnsw": {"space": "l2"}},
    {"embedding_function": {"name": "default"}, "hnsw": {"space": "cosine"}},
    {"embedding_function": {"name": "default"}, "hnsw": {"space": "l2"}, "hnsw:space": "cosine"},
])
def test_incompatible_collection_configuration_is_not_verified(tmp_path, monkeypatch, configuration):
    metadata_file, chroma_path, records = _write_verified_fixture(tmp_path)
    collection = FakeCollection(len(records), records)
    collection.configuration_json = configuration
    monkeypatch.setattr(schema_retrieval, "CHROMA_PATH", chroma_path)
    status = inspect_schema_index(
        metadata_path=metadata_file,
        expected_database="AdventureWorks2025",
        client_factory=lambda path: FakeClient(collection),
    )
    assert status.alignment == "mismatch"


def test_inventory_retrieval_failure_is_not_verified(tmp_path, monkeypatch):
    metadata_file, chroma_path, records = _write_verified_fixture(tmp_path)

    class FailingCollection(FakeCollection):
        def get(self, **kwargs):
            raise RuntimeError("inventory unavailable")

    monkeypatch.setattr(schema_retrieval, "CHROMA_PATH", chroma_path)
    status = inspect_schema_index(
        metadata_path=metadata_file,
        expected_database="AdventureWorks2025",
        client_factory=lambda path: FakeClient(FailingCollection(len(records), records)),
    )
    assert status.alignment == "mismatch"


def test_manifest_change_during_inspection_is_not_verified(tmp_path, monkeypatch):
    metadata_file, chroma_path, records = _write_verified_fixture(tmp_path)
    manifest_file = schema_index.schema_index_manifest_path(
        chroma_path, schema_retrieval.COLLECTION_NAME
    )
    original_get = FakeCollection.get

    def mutate_manifest(self, **kwargs):
        value = json.loads(manifest_file.read_text(encoding="utf-8"))
        value["build"]["build_id"] = "changed-build"
        manifest_file.write_text(json.dumps(value), encoding="utf-8")
        return original_get(self, **kwargs)

    monkeypatch.setattr(FakeCollection, "get", mutate_manifest)
    monkeypatch.setattr(schema_retrieval, "CHROMA_PATH", chroma_path)
    status = inspect_schema_index(
        metadata_path=metadata_file,
        expected_database="AdventureWorks2025",
        client_factory=lambda path: FakeClient(FakeCollection(len(records), records)),
    )
    assert status.alignment != "verified"
