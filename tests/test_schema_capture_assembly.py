from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from ssa.schema_capture_assembly import assemble_versioned_schema_metadata
from ssa.schema_capture_provenance import SQLServerIdentityEvidence
from ssa.schema_key_metadata import (
    fingerprint_schema_metadata,
    fingerprint_schema_semantic_payload,
)


CAPTURED_AT = datetime(2026, 11, 19, 12, 34, 56, 123456, tzinfo=timezone.utc)


def semantic_metadata():
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
                        "description": "Name",
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
                    {"name": "ItemKey", "data_type": "int", "description": None}
                ],
            }
        },
        "table_count": 2,
        "view_count": 1,
        "foreign_key_count": 1,
    }


def identity(server_name="SQLHOST\\SQLEXPRESS"):
    return SQLServerIdentityEvidence(
        server_name=server_name,
        instance_name="SQLEXPRESS",
        database_id=7,
        database_guid="11111111-2222-3333-4444-555555555555",
    )


def assemble(metadata=None, **overrides):
    arguments = {
        "snapshot_id": "capture-1",
        "captured_at_utc": CAPTURED_AT,
        "server_identity": identity(),
        "server_version": "17.0.1000.7",
        "capture_tool_version": "ssa-schema-export/1",
    }
    arguments.update(overrides)
    source = semantic_metadata() if metadata is None else metadata
    return assemble_versioned_schema_metadata(source, **arguments)


def test_valid_legacy_input_produces_complete_version_two_document():
    output = assemble()

    assert output["metadata_format_version"] == 2
    assert output["database"] == "AdventureWorks2025"
    assert set(output) == {
        "database",
        "tables",
        "views",
        "table_count",
        "view_count",
        "foreign_key_count",
        "metadata_format_version",
        "provenance",
    }


def test_provenance_shape_and_optional_diagnostics():
    provenance = assemble()["provenance"]
    assert provenance["snapshot_id"] == "capture-1"
    assert provenance["captured_at_utc"] == "2026-11-19T12:34:56.123456Z"
    assert provenance["database"] == "AdventureWorks2025"
    assert provenance["server_identity"]["server_name"] == "SQLHOST\\SQLEXPRESS"
    assert provenance["server_version"] == "17.0.1000.7"
    assert provenance["capture_tool_version"] == "ssa-schema-export/1"

    without_diagnostics = assemble(
        server_version=None,
        capture_tool_version=None,
    )["provenance"]
    assert "server_version" not in without_diagnostics
    assert "capture_tool_version" not in without_diagnostics


def test_stored_fingerprint_matches_both_fingerprint_apis():
    output = assemble()
    stored = output["provenance"]["content_fingerprint"]
    assert stored == fingerprint_schema_semantic_payload(
        {key: value for key, value in output.items() if key != "provenance"}
    )
    assert stored == fingerprint_schema_metadata(output)


@pytest.mark.parametrize(
    "override",
    [
        {"snapshot_id": "capture-2"},
        {"captured_at_utc": datetime(2027, 1, 2, tzinfo=timezone.utc)},
        {"server_identity": identity("OTHER-SERVER")},
    ],
)
def test_capture_identity_changes_do_not_change_semantic_digest(override):
    assert assemble()["provenance"]["content_fingerprint"] == assemble(**override)[
        "provenance"
    ]["content_fingerprint"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["tables"]["Catalog.Item"].update(
            {"description": "Changed"}
        ),
        lambda payload: payload["tables"]["Catalog.Item"]["columns"][0].update(
            {"data_type": "bigint"}
        ),
        lambda payload: payload["tables"]["Catalog.Item"].update(
            {"primary_keys": ["ItemName"]}
        ),
        lambda payload: payload["tables"]["Catalog.Item"]["foreign_keys"][0].update(
            {"referenced_table": "OtherCategory"}
        ),
        lambda payload: payload.update({"future_semantic_field": "changed"}),
    ],
)
def test_semantic_changes_change_digest(mutation):
    changed = semantic_metadata()
    mutation(changed)
    assert assemble()["provenance"]["content_fingerprint"] != assemble(changed)[
        "provenance"
    ]["content_fingerprint"]


def test_input_and_nested_values_are_not_aliased():
    source = semantic_metadata()
    before = deepcopy(source)
    output = assemble(source)

    assert source == before
    output["tables"]["Catalog.Item"]["columns"][0]["name"] = "Changed"
    assert source == before

    source["tables"]["Catalog.Item"]["columns"][0]["name"] = "Changed Again"
    assert output["tables"]["Catalog.Item"]["columns"][0]["name"] == "Changed"


def test_array_order_is_preserved():
    source = semantic_metadata()
    source["tables"]["Catalog.Composite"]["primary_keys"] = ["PartB", "PartA"]
    output = assemble(source)
    assert output["tables"]["Catalog.Composite"]["primary_keys"] == [
        "PartB",
        "PartA",
    ]


@pytest.mark.parametrize(
    "field",
    [
        "metadata_format_version",
        "provenance",
        "captured_at_utc",
        "captured_at",
        "capture_timestamp",
        "capture_time_utc",
        "export_run_id",
        "snapshot_id",
        "content_fingerprint",
    ],
)
def test_existing_version_or_capture_fields_are_rejected(field):
    source = semantic_metadata()
    source[field] = 3 if field == "metadata_format_version" else "ambiguous"
    with pytest.raises(ValueError):
        assemble(source)


def test_existing_versioned_document_is_rejected():
    with pytest.raises(ValueError):
        assemble(assemble())


@pytest.mark.parametrize(
    "source",
    [
        {"tables": {}, "views": {}},
        {"database": "DB", "tables": [], "views": {}},
        {"database": "DB", "tables": {}, "views": {}, "bad": object()},
        {"database": "DB", "tables": {}, "views": {}, "bad": float("nan")},
    ],
)
def test_invalid_semantic_input_is_rejected(source):
    with pytest.raises(ValueError):
        assemble(source)


def test_invalid_capture_inputs_are_rejected():
    with pytest.raises(ValueError):
        assemble(snapshot_id="")
    with pytest.raises(ValueError):
        assemble(captured_at_utc=datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        assemble(
            captured_at_utc=datetime(
                2026,
                1,
                1,
                tzinfo=timezone(timedelta(hours=1)),
            )
        )
    with pytest.raises(ValueError):
        assemble(server_identity=None)


def test_output_is_deterministic_for_identical_inputs():
    assert assemble() == assemble()
