from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from ssa.schema_capture_assembly import assemble_versioned_schema_metadata
from ssa.schema_capture_provenance import SQLServerIdentityEvidence
import ssa.schema_index as index
from ssa.schema_index_manifest import SchemaIndexManifest
from ssa.schema_index_manifest_store import read_index_manifest, write_index_manifest


CAPTURED = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
STARTED = datetime(2026, 11, 25, 12, 1, tzinfo=timezone.utc)
COMPLETED = datetime(2026, 11, 25, 12, 2, tzinfo=timezone.utc)


def metadata(snapshot="capture-1", database="AdventureWorks2025"):
    semantic = {
        "database": database,
        "tables": {
            "dbo.Item": {
                "schema": "dbo",
                "table": "Item",
                "description": None,
                "columns": [{
                    "name": "ID", "data_type": "int", "description": None,
                }],
                "primary_keys": ["ID"],
                "foreign_keys": [],
            }
        },
        "views": {},
        "table_count": 1,
        "view_count": 0,
        "foreign_key_count": 0,
    }
    return assemble_versioned_schema_metadata(
        semantic,
        snapshot_id=snapshot,
        captured_at_utc=CAPTURED,
        server_identity=SQLServerIdentityEvidence(server_name="SQLHOST"),
        server_version="17.0.1000.7",
    )


class FakeEmbedding:
    @staticmethod
    def name():
        return "default"


class FakeCollection:
    configuration_json = {
        "embedding_function": {"name": "default"},
        "hnsw": {"space": "l2"},
    }

    def __init__(self, events, mismatch=False, fail_get=None, fail_upsert=None, fail_delete=None):
        self.events = events
        self.records = {}
        self.mismatch = mismatch
        self.fail_get = fail_get
        self.fail_upsert = fail_upsert
        self.fail_delete = fail_delete

    def upsert(self, *, documents, ids):
        self.events.append("upsert")
        if self.fail_upsert:
            raise RuntimeError(self.fail_upsert)
        self.records.update(dict(zip(ids, documents, strict=True)))

    def get(self, **kwargs):
        self.events.append("get")
        if self.fail_get:
            raise RuntimeError(self.fail_get)
        if self.mismatch:
            return {"ids": list(self.records), "documents": ["wrong"]}
        return {
            "ids": list(self.records),
            "documents": list(self.records.values()),
        }

    def delete(self, **kwargs):
        self.events.append("delete")
        if self.fail_delete:
            raise RuntimeError(self.fail_delete)
        for document_id in kwargs["ids"]:
            self.records.pop(document_id, None)


class FakeClient:
    def __init__(self, collection, events, existing=False, fail_inspect=None, fail_create=None, fail_drop=None):
        self.collection = collection
        self.events = events
        self.existing = existing
        self.fail_inspect = fail_inspect
        self.fail_create = fail_create
        self.fail_drop = fail_drop

    def get_collection(self, **kwargs):
        self.events.append("inspect")
        if self.fail_inspect:
            raise RuntimeError(self.fail_inspect)
        if not self.existing:
            raise index.NotFoundError("missing")
        return self.collection

    def get_or_create_collection(self, **kwargs):
        self.events.append("create")
        if self.fail_create:
            raise RuntimeError(self.fail_create)
        return self.collection

    def delete_collection(self, **kwargs):
        self.events.append("drop")
        if self.fail_drop:
            raise RuntimeError(self.fail_drop)
        self.collection.records.clear()


def run_index(tmp_path, monkeypatch, *, existing=False, rebuild=False, mismatch=False, source=metadata(), expected_database="AdventureWorks2025"):
    metadata_path = tmp_path / "schema.json"
    metadata_path.write_text(json.dumps(source), encoding="utf-8")
    events = []
    collection = FakeCollection(events, mismatch=mismatch)
    client = FakeClient(collection, events, existing=existing)
    manifests = []

    def writer(value, destination):
        manifests.append(value)
        events.append(value.status)
        return Path(destination)

    monkeypatch.setattr(index, "create_schema_embedding_function", lambda: FakeEmbedding())
    monkeypatch.setattr(index, "schema_embedding_identity", lambda: "test-embedding")
    index.index_schema(
        metadata_path=metadata_path,
        chroma_path=tmp_path / "chroma",
        client_factory=lambda path: client,
        manifest_writer=writer,
        clock=lambda: STARTED if not manifests else COMPLETED,
        build_id_factory=lambda: "build-1",
        expected_database=expected_database,
        rebuild=rebuild,
    )
    return events, manifests, collection


def test_incomplete_manifest_precedes_client_and_complete_follows_verification(tmp_path, monkeypatch):
    events, manifests, _ = run_index(tmp_path, monkeypatch, rebuild=False)
    assert manifests[0].status == "incomplete"
    assert manifests[1].status == "complete"
    assert manifests[0].build_id == manifests[1].build_id == "build-1"
    assert events.index("incomplete") < events.index("create")
    assert events.index("complete") > events.index("get")


def test_existing_collection_requires_explicit_rebuild(tmp_path, monkeypatch):
    metadata_path = tmp_path / "schema.json"
    metadata_path.write_text(json.dumps(metadata()), encoding="utf-8")
    events = []
    collection = FakeCollection(events)
    client = FakeClient(collection, events, existing=True)
    manifests = []
    monkeypatch.setattr(index, "create_schema_embedding_function", lambda: FakeEmbedding())
    monkeypatch.setattr(index, "schema_embedding_identity", lambda: "test-embedding")
    with pytest.raises(RuntimeError, match="explicit rebuild"):
        index.index_schema(
            metadata_path=metadata_path,
            chroma_path=tmp_path / "chroma",
            client_factory=lambda path: client,
            manifest_writer=lambda value, path: manifests.append(value),
            clock=lambda: STARTED,
            build_id_factory=lambda: "build-1",
            expected_database="AdventureWorks2025",
        )
    assert [item.status for item in manifests] == ["incomplete"]
    assert "upsert" not in events


def test_rebuild_drops_existing_collection_only_after_incomplete_manifest(tmp_path, monkeypatch):
    events, manifests, _ = run_index(tmp_path, monkeypatch, existing=True, rebuild=True)
    assert events.index("incomplete") < events.index("drop")
    assert manifests[-1].status == "complete"


def test_inventory_mismatch_does_not_publish_complete(tmp_path, monkeypatch):
    metadata_path = tmp_path / "schema.json"
    metadata_path.write_text(json.dumps(metadata()), encoding="utf-8")
    events = []
    collection = FakeCollection(events, mismatch=True)
    client = FakeClient(collection, events)
    manifests = []
    monkeypatch.setattr(index, "create_schema_embedding_function", lambda: FakeEmbedding())
    monkeypatch.setattr(index, "schema_embedding_identity", lambda: "test-embedding")
    with pytest.raises(RuntimeError, match="inventory"):
        index.index_schema(
            metadata_path=metadata_path,
            chroma_path=tmp_path / "chroma",
            client_factory=lambda path: client,
            manifest_writer=lambda value, path: manifests.append(value),
            clock=lambda: STARTED,
            build_id_factory=lambda: "build-1",
            expected_database="AdventureWorks2025",
        )
    assert [item.status for item in manifests] == ["incomplete"]


def test_source_replacement_does_not_publish_complete(tmp_path, monkeypatch):
    source_path = tmp_path / "schema.json"
    source_path.write_text(json.dumps(metadata()), encoding="utf-8")
    events = []
    collection = FakeCollection(events)
    client = FakeClient(collection, events)
    manifests = []
    original_writer = lambda value, path: manifests.append(value)

    def writer(value, path):
        original_writer(value, path)
        if value.status == "incomplete":
            source_path.write_text(json.dumps(metadata("capture-2")), encoding="utf-8")

    monkeypatch.setattr(index, "create_schema_embedding_function", lambda: FakeEmbedding())
    monkeypatch.setattr(index, "schema_embedding_identity", lambda: "test-embedding")
    with pytest.raises(RuntimeError, match="source changed"):
        index.index_schema(
            metadata_path=source_path,
            chroma_path=tmp_path / "chroma",
            client_factory=lambda path: client,
            manifest_writer=writer,
            clock=lambda: STARTED,
            build_id_factory=lambda: "build-1",
            expected_database="AdventureWorks2025",
        )
    assert [item.status for item in manifests] == ["incomplete"]


def test_legacy_indexing_invalidates_old_manifest_and_keeps_legacy_output(tmp_path, monkeypatch):
    metadata_path = tmp_path / "schema.json"
    metadata_path.write_text(json.dumps({"tables": {"dbo.Item": {"columns": []}}}), encoding="utf-8")
    chroma_path = tmp_path / "chroma"
    manifest_path = index.schema_index_manifest_path(chroma_path, "ssa_schema")
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("old", encoding="utf-8")
    events = []
    collection = FakeCollection(events)
    client = FakeClient(collection, events)
    monkeypatch.setattr(index, "create_schema_embedding_function", lambda: FakeEmbedding())
    index.index_schema(
        metadata_path=metadata_path,
        chroma_path=chroma_path,
        client_factory=lambda path: client,
    )
    assert not manifest_path.exists()
    assert "upsert" in events


def test_matching_expected_database_is_used_for_initial_and_final_validation(tmp_path, monkeypatch):
    events, manifests, _ = run_index(tmp_path, monkeypatch)
    assert [item.status for item in manifests] == ["incomplete", "complete"]


def test_mismatched_expected_database_prevents_manifest_and_client(tmp_path, monkeypatch):
    metadata_path = tmp_path / "schema.json"
    metadata_path.write_text(json.dumps(metadata()), encoding="utf-8")
    calls = []
    with pytest.raises(Exception, match="database"):
        index.index_schema(
            metadata_path=metadata_path,
            chroma_path=tmp_path / "chroma",
            expected_database="OtherDatabase",
            client_factory=lambda path: calls.append(path),
            manifest_writer=lambda value, path: pytest.fail("manifest was published"),
        )
    assert calls == []


def test_missing_configured_database_prevents_versioned_indexing(tmp_path, monkeypatch):
    metadata_path = tmp_path / "schema.json"
    metadata_path.write_text(json.dumps(metadata()), encoding="utf-8")
    monkeypatch.delenv("SQL_DATABASE", raising=False)
    with pytest.raises(ValueError, match="SQL_DATABASE"):
        index.index_schema(
            metadata_path=metadata_path,
            chroma_path=tmp_path / "chroma",
            client_factory=lambda path: pytest.fail("client was constructed"),
        )


def test_configured_database_is_used_when_explicit_value_is_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("SQL_DATABASE", "AdventureWorks2025")
    events, manifests, _ = run_index(tmp_path, monkeypatch, expected_database=None)
    assert manifests[-1].status == "complete"


def test_final_database_replacement_prevents_complete_publication(tmp_path, monkeypatch):
    source_path = tmp_path / "schema.json"
    source_path.write_text(json.dumps(metadata()), encoding="utf-8")
    events = []
    collection = FakeCollection(events)
    client = FakeClient(collection, events)
    manifests = []

    def writer(value, path):
        manifests.append(value)
        if value.status == "incomplete":
            source_path.write_text(
                json.dumps(metadata(database="OtherDatabase")),
                encoding="utf-8",
            )

    monkeypatch.setattr(index, "create_schema_embedding_function", lambda: FakeEmbedding())
    monkeypatch.setattr(index, "schema_embedding_identity", lambda: "test-embedding")
    with pytest.raises(Exception, match="database"):
        index.index_schema(
            metadata_path=source_path,
            chroma_path=tmp_path / "chroma",
            expected_database="AdventureWorks2025",
            client_factory=lambda path: client,
            manifest_writer=writer,
            clock=lambda: STARTED,
            build_id_factory=lambda: "build-1",
        )
    assert [item.status for item in manifests] == ["incomplete"]


def test_existing_guard_blocks_manifest_and_client(tmp_path, monkeypatch):
    metadata_path = tmp_path / "schema.json"
    metadata_path.write_text(json.dumps(metadata()), encoding="utf-8")
    guard_path = tmp_path / "chroma" / ".ssa-schema-index-build.lock"
    guard_path.parent.mkdir()
    guard_path.write_text("other-process\n", encoding="ascii")
    calls = []
    with pytest.raises(RuntimeError, match="already active"):
        index.index_schema(
            metadata_path=metadata_path,
            chroma_path=guard_path.parent,
            expected_database="AdventureWorks2025",
            client_factory=lambda path: calls.append(path),
            manifest_writer=lambda value, path: pytest.fail("manifest was published"),
        )
    assert calls == []
    assert guard_path.read_text(encoding="ascii") == "other-process\n"


def test_incomplete_publication_failure_prevents_client_and_releases_guard(tmp_path, monkeypatch):
    metadata_path = tmp_path / "schema.json"
    metadata_path.write_text(json.dumps(metadata()), encoding="utf-8")
    chroma_path = tmp_path / "chroma"
    calls = []
    with pytest.raises(RuntimeError, match="writer failure"):
        index.index_schema(
            metadata_path=metadata_path,
            chroma_path=chroma_path,
            expected_database="AdventureWorks2025",
            client_factory=lambda path: calls.append(path),
            manifest_writer=lambda value, path: (_ for _ in ()).throw(RuntimeError("writer failure")),
        )
    assert calls == []
    assert not (chroma_path / ".ssa-schema-index-build.lock").exists()


def test_complete_publication_failure_leaves_only_incomplete_evidence(tmp_path, monkeypatch):
    statuses = []

    def writer(value, path):
        statuses.append(value.status)
        if value.status == "complete":
            raise RuntimeError("complete writer failure")
        write_index_manifest(value, path)

    metadata_path = tmp_path / "schema.json"
    metadata_path.write_text(json.dumps(metadata()), encoding="utf-8")
    events = []
    collection = FakeCollection(events)
    client = FakeClient(collection, events)
    monkeypatch.setattr(index, "create_schema_embedding_function", lambda: FakeEmbedding())
    monkeypatch.setattr(index, "schema_embedding_identity", lambda: "test-embedding")
    with pytest.raises(RuntimeError, match="complete writer failure"):
        index.index_schema(
            metadata_path=metadata_path,
            chroma_path=tmp_path / "chroma",
            expected_database="AdventureWorks2025",
            client_factory=lambda path: client,
            manifest_writer=writer,
            clock=lambda: STARTED,
            build_id_factory=lambda: "build-1",
        )
    assert statuses == ["incomplete", "complete"]
    assert read_index_manifest(index.schema_index_manifest_path(tmp_path / "chroma", "ssa_schema")).status == "incomplete"
    assert not (tmp_path / "chroma" / ".ssa-schema-index-build.lock").exists()


@pytest.mark.parametrize("failure", ["inspect", "create", "upsert", "get"])
def test_collection_operation_failures_leave_incomplete_manifest(tmp_path, monkeypatch, failure):
    metadata_path = tmp_path / "schema.json"
    metadata_path.write_text(json.dumps(metadata()), encoding="utf-8")
    events = []
    collection = FakeCollection(events, fail_get="inventory failure" if failure == "get" else None,
                                fail_upsert="upsert failure" if failure == "upsert" else None)
    client = FakeClient(collection, events, fail_inspect="inspect failure" if failure == "inspect" else None,
                        fail_create="create failure" if failure == "create" else None)
    statuses = []
    monkeypatch.setattr(index, "create_schema_embedding_function", lambda: FakeEmbedding())
    monkeypatch.setattr(index, "schema_embedding_identity", lambda: "test-embedding")
    with pytest.raises(RuntimeError):
        index.index_schema(
            metadata_path=metadata_path,
            chroma_path=tmp_path / "chroma",
            expected_database="AdventureWorks2025",
            client_factory=lambda path: client,
            manifest_writer=lambda value, path: statuses.append(value.status),
            clock=lambda: STARTED,
            build_id_factory=lambda: "build-1",
            rebuild=False,
        )
    assert statuses == ["incomplete"]


@pytest.mark.parametrize("configuration", [
    {},
    {"embedding_function": {"name": "other"}, "hnsw": {"space": "l2"}},
    {"embedding_function": {"name": "default"}, "hnsw": {"space": "cosine"}},
    {"embedding_function": {"name": "default", "config": {"model": "other"}}, "hnsw": {"space": "l2"}},
    {"embedding_function": {"name": "default"}, "hnsw": {"space": "l2"}, "hnsw:space": "cosine"},
])
def test_incompatible_or_missing_collection_configuration_fails(configuration):
    class Collection:
        configuration_json = configuration

    with pytest.raises(RuntimeError):
        index._verify_collection_configuration(Collection())


def test_legacy_rebuild_is_rejected_without_manifest_or_collection_mutation(tmp_path, monkeypatch):
    metadata_path = tmp_path / "schema.json"
    metadata_path.write_text(json.dumps({"tables": {"dbo.Item": {"columns": []}}}), encoding="utf-8")
    chroma_path = tmp_path / "chroma"
    manifest_path = index.schema_index_manifest_path(chroma_path, "ssa_schema")
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("old", encoding="utf-8")
    with pytest.raises(ValueError, match="version-2"):
        index.index_schema(
            metadata_path=metadata_path,
            chroma_path=chroma_path,
            rebuild=True,
            client_factory=lambda path: pytest.fail("client was constructed"),
        )
    assert manifest_path.read_text(encoding="utf-8") == "old"


def test_legacy_manifest_invalidation_failure_prevents_collection_mutation(tmp_path, monkeypatch):
    metadata_path = tmp_path / "schema.json"
    metadata_path.write_text(json.dumps({"tables": {"dbo.Item": {"columns": []}}}), encoding="utf-8")
    events = []
    client = FakeClient(FakeCollection(events), events)
    monkeypatch.setattr(index, "_invalidate_legacy_manifest", lambda path: (_ for _ in ()).throw(RuntimeError("invalidate failure")))
    with pytest.raises(RuntimeError, match="invalidate failure"):
        index.index_schema(
            metadata_path=metadata_path,
            chroma_path=tmp_path / "chroma",
            client_factory=lambda path: client,
        )
    assert events == []


def test_guard_cleanup_failure_does_not_mask_primary_failure(tmp_path, monkeypatch):
    metadata_path = tmp_path / "schema.json"
    metadata_path.write_text(json.dumps(metadata()), encoding="utf-8")
    monkeypatch.setattr(index._ExclusiveBuildGuard, "release", lambda self: (_ for _ in ()).throw(RuntimeError("cleanup failure")))
    with pytest.raises(RuntimeError, match="primary failure") as error:
        index.index_schema(
            metadata_path=metadata_path,
            chroma_path=tmp_path / "chroma",
            expected_database="AdventureWorks2025",
            client_factory=lambda path: (_ for _ in ()).throw(RuntimeError("primary failure")),
            manifest_writer=lambda value, path: None,
        )
    assert "primary failure" in str(error.value)


@pytest.mark.parametrize("response", [
    {"ids": [], "documents": ["missing"]},
    {"ids": ["unexpected"], "documents": ["text"]},
    {"ids": ["table:dbo.Item", "table:dbo.Item"], "documents": ["a", "a"]},
    {"ids": ["table:dbo.Item"], "documents": ["wrong"]},
    {"ids": ["table:dbo.Item"], "documents": []},
    {"ids": "bad", "documents": ["bad"]},
])
def test_exact_inventory_failures_prevent_complete_publication(tmp_path, monkeypatch, response):
    class ResponseCollection(FakeCollection):
        def get(self, **kwargs):
            self.events.append("get")
            return response

    metadata_path = tmp_path / "schema.json"
    metadata_path.write_text(json.dumps(metadata()), encoding="utf-8")
    events = []
    client = FakeClient(ResponseCollection(events), events)
    statuses = []
    monkeypatch.setattr(index, "create_schema_embedding_function", lambda: FakeEmbedding())
    monkeypatch.setattr(index, "schema_embedding_identity", lambda: "test-embedding")
    with pytest.raises(Exception):
        index.index_schema(
            metadata_path=metadata_path,
            chroma_path=tmp_path / "chroma",
            expected_database="AdventureWorks2025",
            client_factory=lambda path: client,
            manifest_writer=lambda value, path: statuses.append(value.status),
            clock=lambda: STARTED,
            build_id_factory=lambda: "build-1",
        )
    assert statuses == ["incomplete"]


def test_compatible_reuse_removes_stale_ids_and_refreshes_inventory(tmp_path, monkeypatch):
    events = []
    collection = FakeCollection(events)
    client = FakeClient(collection, events, existing=False)
    monkeypatch.setattr(index, "create_schema_embedding_function", lambda: FakeEmbedding())
    monkeypatch.setattr(index, "schema_embedding_identity", lambda: "test-embedding")

    # Establish the source and a complete prior manifest through the same
    # fake-only lifecycle, then reuse the collection with an obsolete record.
    (tmp_path / "schema.json").write_text(json.dumps(metadata()), encoding="utf-8")
    index.index_schema(
        metadata_path=tmp_path / "schema.json",
        chroma_path=tmp_path / "chroma",
        expected_database="AdventureWorks2025",
        client_factory=lambda path: client,
        manifest_writer=write_index_manifest,
        clock=lambda: STARTED,
        build_id_factory=lambda: "build-1",
        rebuild=True,
    )
    collection.records["stale"] = "stale document"
    client.existing = True
    index.index_schema(
        metadata_path=tmp_path / "schema.json",
        chroma_path=tmp_path / "chroma",
        expected_database="AdventureWorks2025",
        client_factory=lambda path: client,
        manifest_writer=write_index_manifest,
        clock=lambda: COMPLETED,
        build_id_factory=lambda: "build-2",
        rebuild=False,
    )
    assert "stale" not in collection.records


def test_incompatible_prior_source_or_generation_requires_rebuild(tmp_path, monkeypatch):
    events = []
    collection = FakeCollection(events)
    client = FakeClient(collection, events)
    monkeypatch.setattr(index, "create_schema_embedding_function", lambda: FakeEmbedding())
    monkeypatch.setattr(index, "schema_embedding_identity", lambda: "test-embedding")
    source_path = tmp_path / "schema.json"
    source_path.write_text(json.dumps(metadata()), encoding="utf-8")
    index.index_schema(
        metadata_path=source_path,
        chroma_path=tmp_path / "chroma",
        expected_database="AdventureWorks2025",
        client_factory=lambda path: client,
        manifest_writer=write_index_manifest,
        clock=lambda: STARTED,
        build_id_factory=lambda: "build-1",
        rebuild=True,
    )
    manifest_path = index.schema_index_manifest_path(tmp_path / "chroma", "ssa_schema")
    prior = json.loads(manifest_path.read_text(encoding="utf-8"))
    prior["documents"]["generation_format_version"] = 2
    manifest_path.write_text(json.dumps(prior), encoding="utf-8")
    client.existing = True
    with pytest.raises(RuntimeError, match="explicit rebuild"):
        index.index_schema(
            metadata_path=source_path,
            chroma_path=tmp_path / "chroma",
            expected_database="AdventureWorks2025",
            client_factory=lambda path: client,
            manifest_writer=write_index_manifest,
            clock=lambda: COMPLETED,
            build_id_factory=lambda: "build-2",
        )


def test_stale_document_deletion_failure_leaves_incomplete_manifest(tmp_path, monkeypatch):
    events = []
    collection = FakeCollection(events)
    client = FakeClient(collection, events)
    monkeypatch.setattr(index, "create_schema_embedding_function", lambda: FakeEmbedding())
    monkeypatch.setattr(index, "schema_embedding_identity", lambda: "test-embedding")
    source_path = tmp_path / "schema.json"
    source_path.write_text(json.dumps(metadata()), encoding="utf-8")
    index.index_schema(
        metadata_path=source_path,
        chroma_path=tmp_path / "chroma",
        expected_database="AdventureWorks2025",
        client_factory=lambda path: client,
        manifest_writer=write_index_manifest,
        clock=lambda: STARTED,
        build_id_factory=lambda: "build-1",
        rebuild=True,
    )
    collection.records["stale"] = "stale"
    collection.fail_delete = "stale deletion failure"
    client.existing = True
    with pytest.raises(RuntimeError, match="stale deletion failure"):
        index.index_schema(
            metadata_path=source_path,
            chroma_path=tmp_path / "chroma",
            expected_database="AdventureWorks2025",
            client_factory=lambda path: client,
            manifest_writer=write_index_manifest,
            clock=lambda: COMPLETED,
            build_id_factory=lambda: "build-2",
        )
    assert read_index_manifest(index.schema_index_manifest_path(tmp_path / "chroma", "ssa_schema")).status == "incomplete"


def test_collection_deletion_failure_requires_rebuild_and_leaves_incomplete(tmp_path, monkeypatch):
    events = []
    collection = FakeCollection(events)
    client = FakeClient(collection, events, existing=True, fail_drop="drop failure")
    monkeypatch.setattr(index, "create_schema_embedding_function", lambda: FakeEmbedding())
    monkeypatch.setattr(index, "schema_embedding_identity", lambda: "test-embedding")
    source_path = tmp_path / "schema.json"
    source_path.write_text(json.dumps(metadata()), encoding="utf-8")
    statuses = []
    with pytest.raises(RuntimeError, match="drop failure"):
        index.index_schema(
            metadata_path=source_path,
            chroma_path=tmp_path / "chroma",
            expected_database="AdventureWorks2025",
            client_factory=lambda path: client,
            manifest_writer=lambda value, path: statuses.append(value.status),
            clock=lambda: STARTED,
            build_id_factory=lambda: "build-1",
            rebuild=True,
        )
    assert statuses == ["incomplete"]
