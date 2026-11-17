import json

from ssa.schema_key_metadata import load_primary_keys


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
