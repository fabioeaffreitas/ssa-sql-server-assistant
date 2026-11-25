import json

import ssa.schema_index as schema_index


def _table(**overrides):
    value = {
        "description": "A test table.",
        "columns": [
            {"name": "ID", "data_type": "int", "description": "Identifier."},
            {"name": "Value", "data_type": "money", "description": "Amount."},
        ],
    }
    value.update(overrides)
    return value


def _documents(table):
    documents, ids = schema_index.create_schema_documents({"tables": table})
    return documents, ids


def test_single_primary_key_is_declared_without_inventing_uniqueness():
    documents, ids = _documents({"dbo.Example": _table(primary_keys=["ID"])})

    assert ids == ["table:dbo.Example"]
    assert documents[0].endswith(
        "Keys:\n- Primary key: ID"
    )
    assert "Foreign key" not in documents[0]


def test_composite_primary_key_preserves_catalog_order():
    documents, _ = _documents({
        "dbo.Example": _table(primary_keys=["Value", "ID"]),
    })

    assert documents[0].endswith(
        "Keys:\n- Primary key: (Value, ID)"
    )


def test_foreign_key_preserves_single_column_mapping():
    documents, _ = _documents({
        "dbo.Example": _table(
            foreign_keys=[{
                "name": "FK_Example_Parent",
                "referenced_schema": "dbo",
                "referenced_table": "Parent",
                "columns": ["ID"],
                "referenced_columns": ["ParentID"],
            }],
        ),
    })

    assert documents[0].endswith(
        "Keys:\n- Foreign key: ID -> dbo.Parent(ParentID)"
    )
    assert "unique" not in documents[0].lower()


def test_composite_foreign_key_preserves_column_mapping_order():
    documents, _ = _documents({
        "dbo.Example": _table(
            foreign_keys=[{
                "name": "FK_Example_Parent",
                "referenced_schema": "dbo",
                "referenced_table": "Parent",
                "columns": ["Value", "ID"],
                "referenced_columns": ["ParentValue", "ParentID"],
            }],
        ),
    })

    assert documents[0].endswith(
        "Keys:\n- Foreign key: (Value, ID) -> "
        "dbo.Parent(ParentValue, ParentID)"
    )


def test_primary_and_foreign_keys_are_distinguished():
    documents, _ = _documents({
        "dbo.Example": _table(
            primary_keys=["ID"],
            foreign_keys=[{
                "name": "FK_Example_Parent",
                "referenced_schema": "dbo",
                "referenced_table": "Parent",
                "columns": ["Value"],
                "referenced_columns": ["ParentValue"],
            }],
        ),
    })

    assert "- Primary key: ID" in documents[0]
    assert "- Foreign key: Value -> dbo.Parent(ParentValue)" in documents[0]


def test_identity_or_id_named_columns_without_declared_keys_are_not_unique():
    documents, _ = _documents({
        "dbo.Example": _table(),
    })

    assert "Keys:" not in documents[0]
    assert "Primary key" not in documents[0]
    assert "unique" not in documents[0].lower()


def test_missing_and_malformed_key_metadata_is_omitted_conservatively():
    documents, _ = _documents({
        "dbo.Missing": _table(),
        "dbo.Malformed": _table(
            primary_keys=["ID", 1],
            foreign_keys=[{"columns": ["ID"]}],
        ),
    })

    assert all("Keys:" not in document for document in documents)


def test_views_do_not_receive_key_declarations():
    documents, ids = schema_index.create_schema_documents({
        "views": {
            "dbo.ExampleView": _table(
                primary_keys=["ID"],
                foreign_keys=[{"columns": ["ID"]}],
            ),
        },
    })

    assert ids == ["view:dbo.ExampleView"]
    assert "Keys:" not in documents[0]
    assert "Primary key" not in documents[0]


def test_existing_text_ids_and_document_order_are_preserved():
    metadata = {
        "tables": {
            "dbo.First": _table(description="First."),
            "dbo.Second": _table(description="Second"),
        },
        "views": {
            "dbo.View": _table(description="View."),
        },
    }

    documents_one, ids_one = schema_index.create_schema_documents(metadata)
    documents_two, ids_two = schema_index.create_schema_documents(metadata)

    assert ids_one == ["table:dbo.First", "table:dbo.Second", "view:dbo.View"]
    assert ids_one == ids_two
    assert documents_one == documents_two
    assert "Object type: table\nName: dbo.First" in documents_one[0]
    assert "Description: First." in documents_one[0]
    assert "- ID (int): Identifier." in documents_one[0]
    assert "- Value (money): Amount." in documents_one[0]


def test_index_schema_preserves_configuration_and_uses_upsert(monkeypatch, tmp_path):
    metadata_path = tmp_path / "schema_metadata.json"
    metadata_path.write_text(
        json.dumps({"tables": {"dbo.Example": _table(primary_keys=["ID"])} }),
        encoding="utf-8",
    )
    captured = {}

    class FakeEmbedding:
        @staticmethod
        def name():
            return "default"

    monkeypatch.setattr(schema_index, "create_schema_embedding_function", lambda: FakeEmbedding())

    class FakeCollection:
        def upsert(self, **kwargs):
            captured["upsert"] = kwargs

    class FakeClient:
        def get_collection(self, **kwargs):
            raise schema_index.NotFoundError("missing")

        def get_or_create_collection(self, **kwargs):
            captured["collection"] = kwargs
            return FakeCollection()

    def fake_persistent_client(**kwargs):
        captured["client"] = kwargs
        return FakeClient()

    monkeypatch.setattr(schema_index, "METADATA_PATH", metadata_path)
    monkeypatch.setattr(schema_index, "CHROMA_PATH", tmp_path / "chroma_db")
    monkeypatch.setattr(
        schema_index.chromadb,
        "PersistentClient",
        fake_persistent_client,
    )

    schema_index.index_schema()

    assert captured["client"] == {"path": str(tmp_path / "chroma_db")}
    assert captured["collection"]["name"] == "ssa_schema"
    assert captured["collection"]["metadata"] == {"hnsw:space": "l2"}
    assert captured["collection"]["embedding_function"].name() == "default"
    assert captured["upsert"]["ids"] == ["table:dbo.Example"]
    assert "Primary key: ID" in captured["upsert"]["documents"][0]
