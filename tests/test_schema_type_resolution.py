from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import ssa.schema as schema
from ssa.schema_capture_assembly import assemble_versioned_schema_metadata
from ssa.schema_capture_provenance import SQLServerIdentityEvidence
from ssa.schema_key_metadata import fingerprint_schema_metadata


def column_row(
    *,
    schema_name="dbo",
    object_name="Example",
    object_type="U",
    column_name="ID",
    column_id=1,
    declared_count=1,
    declared_name="int",
    system_count=1,
    system_name="int",
    description=None,
    **attributes,
):
    values = {
        "schema_name": schema_name,
        "object_name": object_name,
        "object_type": object_type,
        "column_name": column_name,
        "column_id": column_id,
        "declared_type_match_count": declared_count,
        "declared_type_name": declared_name,
        "system_type_match_count": system_count,
        "system_type_name": system_name,
        "max_length": 4,
        "precision": 10,
        "scale": 0,
        "is_nullable": False,
        "is_identity": False,
        "description": description,
    }
    values.update(attributes)
    return SimpleNamespace(**values)


def resolve(row, object_key="dbo.Example"):
    return schema._resolve_column_type(row, object_key)


@pytest.mark.parametrize(
    ("declared_name", "system_name"),
    [("int", "int"), ("AliasCustomerName", "nvarchar")],
)
def test_unique_declared_type_wins(declared_name, system_name):
    assert resolve(column_row(declared_name=declared_name, system_name=system_name)) == declared_name


@pytest.mark.parametrize(
    ("system_name", "column_name"),
    [("nvarchar", "Name"), ("bit", "MakeFlag")],
)
def test_unavailable_declared_type_uses_unique_system_fallback(system_name, column_name):
    row = column_row(
        column_name=column_name,
        declared_count=0,
        declared_name=None,
        system_count=1,
        system_name=system_name,
    )
    assert resolve(row) == system_name


def test_declared_type_takes_precedence_over_ambiguous_system_lookup():
    assert resolve(column_row(system_count=2, system_name="int")) == "int"


@pytest.mark.parametrize(
    ("declared_count", "system_count", "message"),
    [
        (0, 0, "unresolved type"),
        (2, 1, "ambiguous declared type"),
        (2, 0, "ambiguous declared type"),
        (0, 2, "ambiguous system type"),
    ],
)
def test_unresolved_or_ambiguous_type_fails(declared_count, system_count, message):
    row = column_row(
        declared_count=declared_count,
        declared_name=None,
        system_count=system_count,
        system_name=None,
    )
    with pytest.raises(schema.SchemaColumnResolutionError, match=message):
        resolve(row)


@pytest.mark.parametrize(
    "row",
    [
        column_row(declared_count=1, declared_name="", system_count=1),
        column_row(declared_count=1, declared_name="   ", system_count=1),
        column_row(declared_count=1, declared_name=None, system_count=1),
        column_row(declared_count=0, declared_name=None, system_count=1, system_name=""),
        column_row(declared_count=0, declared_name=None, system_count=1, system_name="   "),
        column_row(declared_count=0, declared_name=None, system_count=1, system_name=None),
    ],
)
def test_invalid_selected_type_name_fails(row):
    with pytest.raises(schema.SchemaColumnResolutionError, match="type name"):
        resolve(row)


@pytest.mark.parametrize("field", ["declared_type_match_count", "system_type_match_count"])
def test_invalid_match_count_fails(field):
    row = column_row()
    setattr(row, field, True)
    with pytest.raises(schema.SchemaColumnResolutionError, match="match count"):
        resolve(row)

    setattr(row, field, -1)
    with pytest.raises(schema.SchemaColumnResolutionError, match="match count"):
        resolve(row)


class FakeCursor:
    def __init__(self, result_sets):
        self.result_sets = iter(result_sets)
        self.current = []
        self.executed = []

    def execute(self, query):
        self.executed.append(query)
        self.current = next(self.result_sets)

    def fetchall(self):
        return self.current

    def fetchone(self):
        return self.current[0]


class FakeConnection:
    def __init__(self, result_sets):
        self.cursor_instance = FakeCursor(result_sets)

    def cursor(self):
        return self.cursor_instance


def description_row(object_name, column_name, object_type="U", description=None):
    return SimpleNamespace(
        schema_name="dbo",
        object_name=object_name,
        column_name=column_name,
        object_description=None,
        column_description=description,
        object_type=object_type,
    )


def extracted_metadata(object_rows, description_rows=None, pk_rows=None, fk_rows=None):
    connection = FakeConnection([
        description_rows or [],
        object_rows,
        pk_rows or [],
        fk_rows or [],
        [("SyntheticDB",)],
    ])
    return schema.get_schema_metadata(connection), connection


def test_extraction_preserves_tables_views_order_descriptions_and_attributes():
    rows = [
        column_row(column_name="First", column_id=1, description="first"),
        column_row(column_name="Second", column_id=2, declared_count=0, declared_name=None, system_name="nvarchar"),
        column_row(object_name="ExampleView", object_type="V", column_name="Value", column_id=1),
    ]
    descriptions = [description_row("Example", "First", description="first")]
    metadata, connection = extracted_metadata(rows, descriptions)

    assert list(metadata["tables"]) == ["dbo.Example"]
    assert list(metadata["views"]) == ["dbo.ExampleView"]
    assert [column["name"] for column in metadata["tables"]["dbo.Example"]["columns"]] == [
        "First", "Second"
    ]
    assert metadata["tables"]["dbo.Example"]["columns"][0] == {
        "name": "First",
        "data_type": "int",
        "max_length": 4,
        "precision": 10,
        "scale": 0,
        "nullable": False,
        "identity": False,
        "description": "first",
    }
    assert "declared_type_match_count" not in metadata["tables"]["dbo.Example"]["columns"][0]
    assert len(connection.cursor_instance.executed) == 5
    assert "LEFT JOIN" in connection.cursor_instance.executed[1]
    assert "GROUP BY user_type_id" in connection.cursor_instance.executed[1]
    assert "GROUP BY system_type_id" in connection.cursor_instance.executed[1]
    assert "is_user_defined = 0" in connection.cursor_instance.executed[1]
    assert "is_assembly_type = 0" in connection.cursor_instance.executed[1]
    assert "ORDER BY s.name, o.name, c.column_id" in connection.cursor_instance.executed[1]


def test_missing_descriptions_do_not_remove_columns():
    rows = [
        column_row(column_name="A", column_id=1),
        column_row(column_name="B", column_id=2),
    ]
    metadata, _ = extracted_metadata(rows, description_rows=[])
    assert [column["name"] for column in metadata["tables"]["dbo.Example"]["columns"]] == ["A", "B"]
    assert all(column["description"] is None for column in metadata["tables"]["dbo.Example"]["columns"])


@pytest.mark.parametrize(
    "rows",
    [
        [column_row(column_name="A", column_id=1), column_row(column_name="A", column_id=2)],
        [column_row(column_name="A", column_id=1), column_row(column_name="B", column_id=1)],
    ],
)
def test_duplicate_column_identity_fails(rows):
    with pytest.raises(schema.SchemaColumnResolutionError, match="duplicate column"):
        extracted_metadata(rows)


def test_unsupported_or_padded_object_type_is_handled():
    row = column_row(object_type="U ")
    metadata, _ = extracted_metadata([row])
    assert "dbo.Example" in metadata["tables"]

    with pytest.raises(schema.SchemaColumnResolutionError, match="unsupported object type"):
        extracted_metadata([column_row(object_type="X")])


def test_keys_remain_intact_when_columns_use_fallback_types():
    rows = [
        column_row(column_name="ID", column_id=1, declared_count=0, declared_name=None, system_name="int"),
        column_row(column_name="ParentID", column_id=2, declared_count=0, declared_name=None, system_name="int"),
    ]
    pk_rows = [SimpleNamespace(schema_name="dbo", table_name="Example", column_name="ID")]
    fk_rows = [SimpleNamespace(
        parent_schema="dbo",
        parent_table="Example",
        foreign_key_name="FK_Example_Parent",
        parent_column="ParentID",
        referenced_schema="dbo",
        referenced_table="Parent",
        referenced_column="ID",
    )]
    metadata, _ = extracted_metadata(rows, pk_rows=pk_rows, fk_rows=fk_rows)
    table = metadata["tables"]["dbo.Example"]
    assert table["primary_keys"] == ["ID"]
    assert table["foreign_keys"][0]["columns"] == ["ParentID"]


def test_extraction_failure_is_not_partial_metadata():
    rows = [
        column_row(column_name="Good", column_id=1),
        column_row(column_name="Bad", column_id=2, declared_count=0, declared_name=None, system_count=0, system_name=None),
    ]
    with pytest.raises(schema.SchemaColumnResolutionError, match="dbo.Example.Bad"):
        extracted_metadata(rows)


def test_extracted_shape_is_compatible_with_version_two_assembly_and_fingerprint():
    metadata, _ = extracted_metadata([column_row()])
    versioned = assemble_versioned_schema_metadata(
        metadata,
        snapshot_id="capture-1",
        captured_at_utc=datetime(2026, 11, 24, tzinfo=timezone.utc),
        server_identity=SQLServerIdentityEvidence(server_name="SQLHOST"),
        server_version="17.0.1000.7",
    )
    assert versioned["metadata_format_version"] == 2
    assert fingerprint_schema_metadata(versioned) == versioned["provenance"]["content_fingerprint"]


def test_changing_data_type_changes_semantic_fingerprint():
    metadata, _ = extracted_metadata([column_row()])
    changed = deepcopy(metadata)
    changed["tables"]["dbo.Example"]["columns"][0]["data_type"] = "nvarchar"
    assert fingerprint_schema_metadata(metadata) != fingerprint_schema_metadata(changed)
