from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

from ssa.schema_capture_assembly import assemble_versioned_schema_metadata
from ssa.schema_capture_provenance import SQLServerIdentityEvidence
from ssa.schema_index_manifest import (
    DocumentInventoryComparison,
    INDEX_MANIFEST_FORMAT_VERSION,
    SchemaIndexDocumentEvidence,
    SchemaIndexManifest,
    SchemaIndexManifestError,
    SchemaIndexSource,
    compare_document_inventories,
    compute_document_content_digest,
    compute_document_identity_digest,
    parse_index_manifest,
    parse_index_manifest_json,
    serialize_index_manifest,
    validate_schema_index_source,
)
from ssa.schema_key_metadata import fingerprint_schema_metadata


CAPTURED_AT = datetime(2026, 11, 25, 12, 0, 0, 123456, tzinfo=timezone.utc)
STARTED_AT = datetime(2026, 11, 25, 12, 1, 0, 123456, tzinfo=timezone.utc)
COMPLETED_AT = datetime(2026, 11, 25, 12, 2, 0, 123456, tzinfo=timezone.utc)


def semantic_metadata():
    return {
        "database": "AdventureWorks2025",
        "tables": {
            "dbo.Item": {
                "schema": "dbo",
                "table": "Item",
                "description": None,
                "columns": [
                    {
                        "name": "ItemID",
                        "data_type": "int",
                        "max_length": 4,
                        "precision": 10,
                        "scale": 0,
                        "nullable": False,
                        "identity": True,
                        "description": None,
                    }
                ],
                "primary_keys": ["ItemID"],
                "foreign_keys": [],
            }
        },
        "views": {},
        "table_count": 1,
        "view_count": 0,
        "foreign_key_count": 0,
    }


def versioned_metadata(**overrides):
    metadata = assemble_versioned_schema_metadata(
        semantic_metadata(),
        snapshot_id="capture-1",
        captured_at_utc=CAPTURED_AT,
        server_identity=SQLServerIdentityEvidence(server_name="SQLHOST"),
        server_version="17.0.1000.7",
    )
    metadata.update(overrides)
    return metadata


def source_evidence(**overrides):
    values = {
        "database": "AdventureWorks2025",
        "metadata_format_version": 2,
        "content_fingerprint": "a" * 64,
        "snapshot_id": "capture-1",
        "captured_at_utc": CAPTURED_AT,
    }
    values.update(overrides)
    return SchemaIndexSource(**values)


def document_evidence(**overrides):
    values = {
        "generation_format_version": 1,
        "expected_count": 2,
        "id_digest": "b" * 64,
        "content_digest": "c" * 64,
    }
    values.update(overrides)
    return SchemaIndexDocumentEvidence(**values)


def manifest(status="complete", **overrides):
    values = {
        "status": status,
        "source": source_evidence(),
        "documents": document_evidence(),
        "build_id": "build-1",
        "started_at_utc": STARTED_AT,
        "completed_at_utc": COMPLETED_AT if status == "complete" else None,
        "embedding_identity": None,
    }
    values.update(overrides)
    return SchemaIndexManifest(**values)


def test_valid_source_evidence_and_expected_database():
    metadata = versioned_metadata()
    before = deepcopy(metadata)
    source = validate_schema_index_source(metadata, "adventureworks2025")

    assert source.database == "AdventureWorks2025"
    assert source.metadata_format_version == 2
    assert source.snapshot_id == "capture-1"
    assert source.captured_at_utc == CAPTURED_AT
    assert source.content_fingerprint == metadata["provenance"]["content_fingerprint"]
    assert metadata == before


def test_source_json_validation_rejects_malformed_duplicate_and_nonfinite_json():
    with pytest.raises(SchemaIndexManifestError):
        validate_schema_index_source("{")
    with pytest.raises(SchemaIndexManifestError):
        validate_schema_index_source('{"database":"A","database":"B"}')
    with pytest.raises(SchemaIndexManifestError):
        validate_schema_index_source('{"value":NaN}')


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("metadata_format_version", 3),
        lambda value: value["provenance"].__setitem__("content_fingerprint", "a" * 64),
        lambda value: value["provenance"].__setitem__("captured_at_utc", "bad"),
    ],
)
def test_source_validation_rejects_invalid_version_provenance_or_fingerprint(mutation):
    metadata = versioned_metadata()
    mutation(metadata)
    with pytest.raises(SchemaIndexManifestError):
        validate_schema_index_source(metadata)


def test_source_validation_rejects_legacy_and_database_mismatch():
    legacy = semantic_metadata()
    with pytest.raises(SchemaIndexManifestError):
        validate_schema_index_source(legacy)

    with pytest.raises(SchemaIndexManifestError):
        validate_schema_index_source(versioned_metadata(), "OtherDatabase")


def test_identity_digest_is_order_independent_and_domain_separated():
    first = compute_document_identity_digest(["view:v", "table:t"])
    second = compute_document_identity_digest(["table:t", "view:v"])
    content = compute_document_content_digest([("table:t", "text"), ("view:v", "text")])

    assert first == second
    assert first != content
    assert len(first) == 64
    assert first == first.lower()


def test_identity_digest_preserves_exact_ids_and_changes_for_set_changes():
    assert compute_document_identity_digest(["é:one"]) != compute_document_identity_digest(["e:one"])
    assert compute_document_identity_digest([]) != compute_document_identity_digest(["table:t"])


@pytest.mark.parametrize("ids", [["table:t", "table:t"], ["   "], ["table:t", ""]])
def test_identity_digest_rejects_invalid_or_duplicate_ids(ids):
    with pytest.raises(SchemaIndexManifestError):
        compute_document_identity_digest(ids)


def test_content_digest_is_order_independent_and_preserves_unicode_whitespace():
    first = compute_document_content_digest([
        ("b", "line 1\nline 2"),
        ("a", "é  text"),
    ])
    second = compute_document_content_digest([
        ("a", "é  text"),
        ("b", "line 1\nline 2"),
    ])
    assert first == second
    assert first != compute_document_content_digest([
        ("a", "é text"),
        ("b", "line 1\nline 2"),
    ])


@pytest.mark.parametrize(
    "documents",
    [
        [("a", "one"), ("a", "two")],
        [("a", "")],
        [("a", "   ")],
        [("a", 1)],
        [("", "text")],
    ],
)
def test_content_digest_rejects_invalid_records(documents):
    with pytest.raises(SchemaIndexManifestError):
        compute_document_content_digest(documents)


def test_inventory_exact_match_is_immutable():
    result = compare_document_inventories(
        [("a", "one"), ("b", "two")],
        [("b", "two"), ("a", "one")],
    )
    assert result == DocumentInventoryComparison(
        matches=True,
        expected_count=2,
        observed_count=2,
    )
    with pytest.raises(AttributeError):
        result.matches = False


def test_inventory_detects_missing_unexpected_changed_and_count_equal_cases():
    result = compare_document_inventories(
        [("a", "one"), ("b", "two")],
        [("a", "changed"), ("c", "three")],
    )
    assert result.matches is False
    assert result.expected_count == result.observed_count == 2
    assert result.missing_ids == ("b",)
    assert result.unexpected_ids == ("c",)
    assert result.changed_document_ids == ("a",)


def test_inventory_reports_duplicate_ids_without_deduplicating():
    result = compare_document_inventories(
        [("a", "one"), ("a", "one")],
        [("a", "one")],
    )
    assert result.matches is False
    assert result.expected_count == 2
    assert result.observed_count == 1
    assert result.duplicate_ids == ("a",)
    assert result.duplicate_expected_ids == ("a",)


def test_inventory_rejects_malformed_records():
    with pytest.raises(SchemaIndexManifestError):
        compare_document_inventories([("a", "one")], [("a",)])


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: SchemaIndexDocumentEvidence(0, 1, "a" * 64, "b" * 64),
        lambda: SchemaIndexDocumentEvidence(1, True, "a" * 64, "b" * 64),
        lambda: SchemaIndexDocumentEvidence(1, 1, "A" * 64, "b" * 64),
    ],
)
def test_document_evidence_rejects_invalid_fields(constructor):
    with pytest.raises(SchemaIndexManifestError):
        constructor()


def test_incomplete_manifest_shape_and_round_trip():
    value = manifest("incomplete")
    serialized = serialize_index_manifest(value)
    assert serialized["manifest_format_version"] == INDEX_MANIFEST_FORMAT_VERSION
    assert "completed_at_utc" not in serialized["build"]
    assert parse_index_manifest(serialized) == value
    assert parse_index_manifest_json(json.dumps(serialized)) == value


def test_complete_manifest_shape_and_round_trip():
    value = manifest(embedding_identity="embedding-v1")
    serialized = serialize_index_manifest(value)
    assert serialized["embedding"] == {"identity": "embedding-v1"}
    assert serialized["build"]["completed_at_utc"] == "2026-11-25T12:02:00.123456Z"
    assert parse_index_manifest(serialized) == value


def test_complete_manifest_rejects_empty_inventory():
    with pytest.raises(SchemaIndexManifestError):
        SchemaIndexManifest(
            status="complete",
            source=source_evidence(),
            documents=SchemaIndexDocumentEvidence(
                1, 0, "a" * 64, "b" * 64
            ),
            build_id="build-1",
            started_at_utc=STARTED_AT,
            completed_at_utc=COMPLETED_AT,
        )
def test_complete_manifest_requires_positive_count_and_completion_time():
    with pytest.raises(SchemaIndexManifestError):
        manifest("complete", documents=document_evidence(expected_count=0))
    with pytest.raises(SchemaIndexManifestError):
        manifest("complete", completed_at_utc=None)


def test_incomplete_manifest_rejects_completion_time():
    with pytest.raises(SchemaIndexManifestError):
        manifest("incomplete", completed_at_utc=COMPLETED_AT)


@pytest.mark.parametrize(
    "started,completed",
    [
        (datetime(2026, 11, 25), COMPLETED_AT),
        (STARTED_AT, datetime(2026, 11, 25, 11, 0, tzinfo=timezone.utc)),
        (STARTED_AT, datetime(2026, 11, 25, 12, 2, tzinfo=timezone(timedelta(hours=1)))),
    ],
)
def test_manifest_rejects_invalid_timestamps(started, completed):
    with pytest.raises(SchemaIndexManifestError):
        manifest(started_at_utc=started, completed_at_utc=completed)


def test_manifest_rejects_unknown_fields_and_versions():
    value = serialize_index_manifest(manifest())
    value["unexpected"] = True
    with pytest.raises(SchemaIndexManifestError):
        parse_index_manifest(value)

    value = serialize_index_manifest(manifest())
    value["manifest_format_version"] = 2
    with pytest.raises(SchemaIndexManifestError):
        parse_index_manifest(value)


def test_manifest_rejects_invalid_embedding_identity_and_generation_version():
    with pytest.raises(SchemaIndexManifestError):
        manifest(embedding_identity=" ")

    value = serialize_index_manifest(manifest())
    value["documents"]["generation_format_version"] = True
    with pytest.raises(SchemaIndexManifestError):
        parse_index_manifest(value)


def test_manifest_rejects_duplicate_keys_and_nonfinite_json():
    valid = json.dumps(serialize_index_manifest(manifest()))
    duplicate = valid[:-1] + ',"status":"complete"}'
    with pytest.raises(SchemaIndexManifestError):
        parse_index_manifest_json(duplicate)
    with pytest.raises(SchemaIndexManifestError):
        parse_index_manifest_json('{"manifest_format_version":NaN}')


def test_source_snapshot_changes_are_not_semantic_changes():
    first = source_evidence(snapshot_id="capture-1")
    second = source_evidence(snapshot_id="capture-2", captured_at_utc=COMPLETED_AT)
    assert first.content_fingerprint == second.content_fingerprint
    assert first.snapshot_id != second.snapshot_id


def test_generation_and_embedding_evidence_are_distinguishable():
    first = manifest()
    second = manifest(
        documents=document_evidence(generation_format_version=2),
        embedding_identity="embedding-v2",
    )
    assert first != second
