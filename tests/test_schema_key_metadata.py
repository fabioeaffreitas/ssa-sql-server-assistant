import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from ssa.schema_capture_provenance import (
    DatabaseIdentityMismatchError,
    InvalidFingerprintError,
    InvalidServerIdentityError,
    InvalidTimestampError,
    SQLServerIdentityEvidence,
    SchemaCaptureProvenance,
    MissingProvenanceError,
    serialize_capture_provenance,
    UnsupportedMetadataVersionError,
)
from ssa.schema_key_metadata import (
    fingerprint_schema_metadata,
    fingerprint_schema_metadata_json,
    load_primary_keys,
)


def write_metadata(tmp_path, payload):
    path = tmp_path / "schema_metadata.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def valid_payload():
    return {
        "database": "AdventureWorks2025",
        "tables": {
            "Catalog.Item": {
                "primary_keys": ["ItemKey"],
                "foreign_keys": [{"columns": ["CategoryID"]}],
            },
            "Catalog.Composite": {
                "primary_keys": ["PartA", "PartB"],
            },
            "Catalog.NoKey": {
                "primary_keys": [],
            },
        },
        "views": {
            "Catalog.ItemView": {
                "primary_keys": ["ItemKey"],
            },
        },
    }


def fingerprint_payload():
    return {
        "database": "AdventureWorks2025",
        "tables": {
            "Catalog.Item": {
                "schema": "Catalog",
                "table": "Item",
                "description": "Items",
                "columns": [
                    {
                        "name": "ItemKey",
                        "data_type": "int",
                        "max_length": 4,
                        "precision": 10,
                        "scale": 0,
                        "nullable": False,
                        "identity": True,
                        "description": "Identifier",
                    },
                    {
                        "name": "ItemName",
                        "data_type": "nvarchar",
                        "max_length": 200,
                        "precision": 0,
                        "scale": 0,
                        "nullable": False,
                        "identity": False,
                        "description": "名称",
                    },
                ],
                "primary_keys": ["ItemKey"],
                "foreign_keys": [
                    {
                        "name": "FK_Item_Category",
                        "referenced_schema": "Catalog",
                        "referenced_table": "Category",
                        "columns": ["CategoryKey"],
                        "referenced_columns": ["CategoryKey"],
                    }
                ],
            },
            "Catalog.Composite": {
                "schema": "Catalog",
                "table": "Composite",
                "description": None,
                "columns": [
                    {
                        "name": "PartA",
                        "data_type": "int",
                        "max_length": 4,
                        "precision": 10,
                        "scale": 0,
                        "nullable": False,
                        "identity": False,
                        "description": None,
                    },
                    {
                        "name": "PartB",
                        "data_type": "int",
                        "max_length": 4,
                        "precision": 10,
                        "scale": 0,
                        "nullable": False,
                        "identity": False,
                        "description": None,
                    },
                ],
                "primary_keys": ["PartA", "PartB"],
                "foreign_keys": [],
            },
        },
        "views": {
            "Catalog.ItemView": {
                "schema": "Catalog",
                "view": "ItemView",
                "description": "View",
                "columns": [
                    {
                        "name": "ItemKey",
                        "data_type": "int",
                        "description": None,
                    }
                ],
            }
        },
        "table_count": 2,
        "view_count": 1,
        "foreign_key_count": 1,
    }


def versioned_payload():
    payload = fingerprint_payload()
    provenance = serialize_capture_provenance(
        SchemaCaptureProvenance(
            snapshot_id="capture-1",
            captured_at_utc=datetime(
                2026,
                11,
                19,
                12,
                34,
                56,
                123456,
                tzinfo=timezone.utc,
            ),
            database="AdventureWorks2025",
            server_identity=SQLServerIdentityEvidence(
                server_name="SQLHOST\\SQLEXPRESS",
                instance_name="SQLEXPRESS",
                database_id=7,
                database_guid="11111111-2222-3333-4444-555555555555",
            ),
            content_fingerprint="a" * 64,
            server_version="17.0.1000.7",
            capture_tool_version="ssa-schema-export/1",
        )
    )
    payload.update(provenance)
    return payload


def test_valid_metadata_extracts_ordered_table_keys_only(tmp_path):
    result = load_primary_keys(
        write_metadata(tmp_path, valid_payload()),
        "adventureworks2025",
    )

    assert result.available is True
    assert result.reason == "available"
    assert result.database == "AdventureWorks2025"
    assert dict(result.primary_keys) == {
        "Catalog.Item": ("ItemKey",),
        "Catalog.Composite": ("PartA", "PartB"),
    }
    assert "Catalog.ItemView" not in result.primary_keys


def test_missing_file_is_unavailable_without_exposing_path(tmp_path):
    result = load_primary_keys(tmp_path / "missing.json", "AdventureWorks2025")

    assert result.available is False
    assert result.reason == "metadata_unavailable"
    assert str(tmp_path) not in result.reason


def test_malformed_json_is_unavailable(tmp_path):
    path = tmp_path / "schema_metadata.json"
    path.write_text("{not-json", encoding="utf-8")

    result = load_primary_keys(path, "AdventureWorks2025")

    assert (result.available, result.reason) == (False, "malformed_json")


def test_database_identity_is_case_insensitive_but_required(tmp_path):
    path = write_metadata(tmp_path, valid_payload())

    assert load_primary_keys(path, "ADVENTUREWORKS2025").available is True
    assert load_primary_keys(path, "OtherDatabase").reason == "database_mismatch"
    assert load_primary_keys(path, None).reason == "missing_expected_database"
    assert load_primary_keys(path, " ").reason == "missing_expected_database"


def test_invalid_root_and_tables_are_rejected(tmp_path):
    path = write_metadata(tmp_path, [])
    assert load_primary_keys(path, "AdventureWorks2025").reason == "invalid_root"

    path = write_metadata(tmp_path, {
        "database": "AdventureWorks2025",
        "tables": [],
    })
    assert load_primary_keys(path, "AdventureWorks2025").reason == "invalid_tables"


def test_invalid_table_and_primary_key_shapes_are_rejected(tmp_path):
    base = {"database": "AdventureWorks2025", "tables": {}}
    cases = [
        {"": {"primary_keys": []}},
        {"Catalog.Item": []},
        {"Catalog.Item": {}},
        {"Catalog.Item": {"primary_keys": "ItemKey"}},
        {"Catalog.Item": {"primary_keys": ["ItemKey", "itemkey"]}},
        {"Catalog.Item": {"primary_keys": ["ItemKey", 1]}},
    ]

    for table_mapping in cases:
        payload = {**base, "tables": table_mapping}
        result = load_primary_keys(
            write_metadata(tmp_path, payload),
            "AdventureWorks2025",
        )
        assert result.available is False


def test_empty_primary_key_list_is_valid_but_provides_no_uniqueness(tmp_path):
    payload = {
        "database": "AdventureWorks2025",
        "tables": {"Catalog.NoKey": {"primary_keys": []}},
    }

    result = load_primary_keys(
        write_metadata(tmp_path, payload),
        "AdventureWorks2025",
    )

    assert result.available is True
    assert dict(result.primary_keys) == {}


def test_fingerprint_is_stable_for_object_order_and_json_whitespace():
    payload = fingerprint_payload()
    reordered = {
        "foreign_key_count": 1,
        "views": payload["views"],
        "database": payload["database"],
        "table_count": 2,
        "tables": payload["tables"],
        "view_count": 1,
    }

    first = fingerprint_schema_metadata_json(json.dumps(payload, indent=2))
    second = fingerprint_schema_metadata_json(
        json.dumps(reordered, ensure_ascii=False, separators=(",", ":"))
    )

    assert first == second
    assert len(first) == 64
    assert first == first.lower()


def test_legacy_fingerprint_digest_remains_unchanged():
    assert fingerprint_schema_metadata(fingerprint_payload()) == (
        "503a12b67162ee645187e1aa6fec115800ca45cf97112c51c5e5e2cef7795829"
    )


def test_versioned_metadata_fingerprints_successfully():
    assert len(fingerprint_schema_metadata(versioned_payload())) == 64


@pytest.mark.parametrize(
    "provenance_update",
    [
        {"snapshot_id": "capture-2"},
        {"captured_at_utc": datetime(2027, 1, 2, 3, 4, 5, tzinfo=timezone.utc)},
        {
            "server_identity": SQLServerIdentityEvidence(
                server_name="OTHER-SERVER",
                instance_name="OTHER",
                database_id=8,
                database_guid="66666666-7777-8888-9999-000000000000",
            )
        },
        {"server_version": "18.0.1"},
        {"capture_tool_version": "ssa-schema-export/2"},
        {"content_fingerprint": "b" * 64},
    ],
)
def test_capture_provenance_changes_do_not_change_semantic_fingerprint(
    provenance_update,
):
    first = versioned_payload()
    base = SchemaCaptureProvenance(
        snapshot_id="capture-1",
        captured_at_utc=datetime(
            2026,
            11,
            19,
            12,
            34,
            56,
            123456,
            tzinfo=timezone.utc,
        ),
        database="AdventureWorks2025",
        server_identity=SQLServerIdentityEvidence(
            server_name="SQLHOST\\SQLEXPRESS",
            instance_name="SQLEXPRESS",
            database_id=7,
            database_guid="11111111-2222-3333-4444-555555555555",
        ),
        content_fingerprint="a" * 64,
        server_version="17.0.1000.7",
        capture_tool_version="ssa-schema-export/1",
    )
    changed = replace(base, **provenance_update)
    second = versioned_payload()
    second.update(serialize_capture_provenance(changed))

    assert fingerprint_schema_metadata(first) == fingerprint_schema_metadata(second)


def test_root_database_change_changes_fingerprint():
    first = versioned_payload()
    second = versioned_payload()
    second["database"] = "OtherDatabase"
    second["provenance"]["database"] = "OtherDatabase"
    assert fingerprint_schema_metadata(first) != fingerprint_schema_metadata(second)


def test_root_format_version_is_semantically_significant():
    legacy = fingerprint_payload()
    versioned = versioned_payload()
    assert fingerprint_schema_metadata(legacy) != fingerprint_schema_metadata(versioned)


def test_nested_semantic_provenance_field_is_not_excluded():
    first = versioned_payload()
    second = versioned_payload()
    first["tables"]["Catalog.Item"]["columns"][0]["provenance"] = {
        "source": "catalog-a"
    }
    second["tables"]["Catalog.Item"]["columns"][0]["provenance"] = {
        "source": "catalog-b"
    }
    assert fingerprint_schema_metadata(first) != fingerprint_schema_metadata(second)


def test_invalid_versioned_provenance_is_rejected():
    unsupported = versioned_payload()
    unsupported["metadata_format_version"] = 3
    with pytest.raises(UnsupportedMetadataVersionError):
        fingerprint_schema_metadata(unsupported)

    invalid_type = versioned_payload()
    invalid_type["metadata_format_version"] = "2"
    with pytest.raises(UnsupportedMetadataVersionError):
        fingerprint_schema_metadata(invalid_type)

    missing = versioned_payload()
    del missing["provenance"]
    with pytest.raises(MissingProvenanceError):
        fingerprint_schema_metadata(missing)

    malformed = versioned_payload()
    malformed["provenance"] = "invalid"
    with pytest.raises(MissingProvenanceError):
        fingerprint_schema_metadata(malformed)


def test_invalid_versioned_provenance_fields_are_rejected():
    database_mismatch = versioned_payload()
    database_mismatch["provenance"]["database"] = "OtherDatabase"
    with pytest.raises(DatabaseIdentityMismatchError):
        fingerprint_schema_metadata(database_mismatch)

    invalid_timestamp = versioned_payload()
    invalid_timestamp["provenance"]["captured_at_utc"] = "not-a-timestamp"
    with pytest.raises(InvalidTimestampError):
        fingerprint_schema_metadata(invalid_timestamp)

    invalid_fingerprint = versioned_payload()
    invalid_fingerprint["provenance"]["content_fingerprint"] = "A" * 64
    with pytest.raises(InvalidFingerprintError):
        fingerprint_schema_metadata(invalid_fingerprint)

    invalid_identity = versioned_payload()
    invalid_identity["provenance"]["server_identity"] = {"database_id": 7}
    with pytest.raises(InvalidServerIdentityError):
        fingerprint_schema_metadata(invalid_identity)


def test_legacy_root_provenance_field_is_not_silently_excluded():
    first = fingerprint_payload()
    second = fingerprint_payload()
    first["provenance"] = {"semantic_value": "one"}
    second["provenance"] = {"semantic_value": "two"}
    assert fingerprint_schema_metadata(first) != fingerprint_schema_metadata(second)


def test_fingerprint_changes_for_semantic_fields():
    baseline = fingerprint_schema_metadata(fingerprint_payload())

    changed_table = fingerprint_payload()
    changed_table["tables"]["Catalog.Item"]["description"] = "Changed"
    assert fingerprint_schema_metadata(changed_table) != baseline

    changed_column = fingerprint_payload()
    changed_column["tables"]["Catalog.Item"]["columns"][0]["data_type"] = "bigint"
    assert fingerprint_schema_metadata(changed_column) != baseline

    changed_key = fingerprint_payload()
    changed_key["tables"]["Catalog.Item"]["primary_keys"] = ["ItemName"]
    assert fingerprint_schema_metadata(changed_key) != baseline

    changed_foreign_key = fingerprint_payload()
    changed_foreign_key["tables"]["Catalog.Item"]["foreign_keys"][0][
        "referenced_table"
    ] = "OtherCategory"
    assert fingerprint_schema_metadata(changed_foreign_key) != baseline

    changed_database = fingerprint_payload()
    changed_database["database"] = "OtherDatabase"
    assert fingerprint_schema_metadata(changed_database) != baseline


def test_array_order_is_semantically_preserved():
    payload = fingerprint_payload()
    first = fingerprint_schema_metadata(payload)

    payload["tables"]["Catalog.Composite"]["primary_keys"] = ["PartB", "PartA"]
    assert fingerprint_schema_metadata(payload) != first


def test_unicode_and_numeric_types_are_deterministic():
    payload = fingerprint_payload()
    first = fingerprint_schema_metadata(payload)
    second = fingerprint_schema_metadata(json.loads(json.dumps(payload, ensure_ascii=False)))
    assert first == second

    integer_payload = fingerprint_payload()
    float_payload = fingerprint_payload()
    integer_payload["tables"]["Catalog.Item"]["columns"][0]["precision"] = 10
    float_payload["tables"]["Catalog.Item"]["columns"][0]["precision"] = 10.0
    assert fingerprint_schema_metadata(integer_payload) != fingerprint_schema_metadata(
        float_payload
    )


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"database": "DB", "tables": {}, "views": []},
        {"database": "DB", "tables": {"Catalog.Item": []}, "views": {}},
        {
            "database": "DB",
            "tables": {
                "Catalog.Item": {
                    "columns": [{"name": "ItemKey"}],
                    "primary_keys": "ItemKey",
                }
            },
            "views": {},
        },
        {
            "database": "DB",
            "tables": {
                "Catalog.Item": {
                    "columns": [{"name": "ItemKey"}],
                    "primary_keys": ["ItemKey"],
                    "foreign_keys": [{"columns": ["A"]}],
                }
            },
            "views": {},
        },
    ],
)
def test_invalid_metadata_shapes_are_rejected(payload):
    with pytest.raises(ValueError):
        fingerprint_schema_metadata(payload)


def test_non_string_keys_and_non_json_values_are_rejected():
    payload = fingerprint_payload()
    payload["tables"]["Catalog.Item"]["unexpected"] = {1: "invalid"}
    with pytest.raises(ValueError):
        fingerprint_schema_metadata(payload)

    payload = fingerprint_payload()
    payload["unexpected"] = {"value": float("nan")}
    with pytest.raises(ValueError):
        fingerprint_schema_metadata(payload)

    payload = fingerprint_payload()
    payload["unexpected"] = {"value": float("inf")}
    with pytest.raises(ValueError):
        fingerprint_schema_metadata(payload)

    payload = fingerprint_payload()
    payload["unexpected"] = {"value": object()}
    with pytest.raises(ValueError):
        fingerprint_schema_metadata(payload)


def test_unknown_semantic_fields_are_included():
    first = fingerprint_payload()
    second = fingerprint_payload()
    first["future_semantic_field"] = "one"
    second["future_semantic_field"] = "two"
    assert fingerprint_schema_metadata(first) != fingerprint_schema_metadata(second)


def test_explicit_volatile_provenance_fields_are_excluded():
    first = fingerprint_payload()
    second = fingerprint_payload()
    first.update(
        {
            "captured_at_utc": "2026-01-01T00:00:00Z",
            "capture_timestamp": "old",
            "export_run_id": "run-1",
            "content_fingerprint": "old-fingerprint",
        }
    )
    second.update(
        {
            "captured_at_utc": "2027-01-01T00:00:00Z",
            "capture_timestamp": "new",
            "export_run_id": "run-2",
            "content_fingerprint": "new-fingerprint",
        }
    )
    assert fingerprint_schema_metadata(first) == fingerprint_schema_metadata(second)


def test_duplicate_json_keys_are_rejected_at_text_boundary():
    duplicate = '{"database":"DB","database":"OtherDB","tables":{},"views":{}}'
    with pytest.raises(ValueError):
        fingerprint_schema_metadata_json(duplicate)


def test_equivalent_json_text_is_fingerprinted_without_writing_files():
    payload = fingerprint_payload()
    compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    pretty = json.dumps(payload, ensure_ascii=False, indent=4)
    assert fingerprint_schema_metadata_json(compact) == fingerprint_schema_metadata_json(pretty)


def test_loader_behavior_remains_independent_of_fingerprint_api(tmp_path):
    result = load_primary_keys(
        write_metadata(tmp_path, valid_payload()),
        "AdventureWorks2025",
    )
    assert result.available is True
    assert dict(result.primary_keys)["Catalog.Composite"] == ("PartA", "PartB")
