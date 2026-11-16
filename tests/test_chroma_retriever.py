from pathlib import Path

import pytest

import ssa.chroma_retriever as retriever_module
from ssa.chroma_retriever import ChromaSchemaDocument, ChromaSchemaRetriever
from ssa.contracts import SchemaDocument, SchemaRetriever


def test_default_configuration_and_legacy_delegation(monkeypatch):
    captured = {}

    def fake_search(question, n_results):
        captured["question"] = question
        captured["n_results"] = n_results
        return [{
            "id": "table:Sales.Customer",
            "document": "customer schema",
            "distance": 0.25,
            "final_score": 0.9,
        }]

    monkeypatch.setattr(retriever_module.schema_retrieval, "search_schema", fake_search)
    retriever = ChromaSchemaRetriever()
    documents = retriever.retrieve("customer question", limit=3)

    assert retriever.chroma_path == retriever_module.schema_retrieval.CHROMA_PATH
    assert retriever.collection_name == retriever_module.schema_retrieval.COLLECTION_NAME
    assert captured == {"question": "customer question", "n_results": 3}
    assert documents == [
        ChromaSchemaDocument(
            id="table:Sales.Customer",
            document="customer schema",
            distance=0.25,
            metadata={"final_score": 0.9},
        )
    ]


class FakeCollection:
    def __init__(self, semantic_results):
        self.semantic_results = semantic_results
        self.query_args = None

    def count(self):
        return len(self.semantic_results["ids"][0])

    def query(self, **kwargs):
        self.query_args = kwargs
        return self.semantic_results


class FakeClient:
    def __init__(self, collection):
        self.collection = collection
        self.collection_args = None

    def get_collection(self, **kwargs):
        self.collection_args = kwargs
        return self.collection


def test_custom_configuration_forwards_path_collection_and_full_query(monkeypatch, tmp_path):
    collection = FakeCollection({
        "ids": [["table:Sales.Customer"]],
        "documents": [["customer names"]],
        "distances": [[0.2]],
    })
    client = FakeClient(collection)
    monkeypatch.setattr(
        retriever_module.chromadb,
        "PersistentClient",
        lambda **kwargs: client,
    )

    retriever = ChromaSchemaRetriever(tmp_path / "custom", "custom_collection")
    documents = retriever.retrieve("customer names", limit=1)

    assert client.collection_args == {"name": "custom_collection"}
    assert collection.query_args == {
        "query_texts": ["customer names"],
        "n_results": 1,
    }
    assert documents[0].id == "table:Sales.Customer"


def test_hybrid_ranking_preserves_boosts_and_applies_limit_after_sorting(
    monkeypatch, tmp_path
):
    collection = FakeCollection({
        "ids": [[
            "table:Production.Product",
            "table:Sales.Customer",
            "table:Sales.SalesOrderHeader",
        ]],
        "documents": [[
            "unrelated product data",
            "customer names",
            "order header",
        ]],
        "distances": [[0.0, 0.2, 0.01]],
    })
    client = FakeClient(collection)
    monkeypatch.setattr(
        retriever_module.chromadb,
        "PersistentClient",
        lambda **kwargs: client,
    )

    documents = ChromaSchemaRetriever(tmp_path, "ranking").retrieve(
        "show sales order customers by customer names",
        limit=2,
    )

    assert [document.id for document in documents] == [
        "table:Sales.SalesOrderHeader",
        "table:Sales.Customer",
    ]
    assert collection.query_args["n_results"] == 3
    assert documents[0].metadata["name_boost"] == 0.40
    assert documents[0].metadata["final_score"] > documents[1].metadata["final_score"]
    assert documents[0].distance == 0.01


def test_conversion_preserves_legacy_fields_and_schema_document_contract():
    candidate = {
        "id": "table:Sales.Customer",
        "document": "customer schema",
        "distance": 0.4,
        "keyword_score": 0.2,
        "semantic_score": 0.7,
        "name_boost": 0.4,
        "final_score": 0.96,
    }

    document = ChromaSchemaRetriever._to_document(candidate)

    assert isinstance(document, SchemaDocument)
    assert document.id == candidate["id"]
    assert document.document == candidate["document"]
    assert document.distance == candidate["distance"]
    assert document.metadata == {
        "keyword_score": 0.2,
        "semantic_score": 0.7,
        "name_boost": 0.4,
        "final_score": 0.96,
    }


def test_chroma_errors_propagate(monkeypatch, tmp_path):
    def fail_client(**kwargs):
        raise RuntimeError("Chroma unavailable")

    monkeypatch.setattr(retriever_module.chromadb, "PersistentClient", fail_client)

    with pytest.raises(RuntimeError, match="Chroma unavailable"):
        ChromaSchemaRetriever(tmp_path, "collection").retrieve("question")


def test_adapter_instances_do_not_share_client_state(tmp_path):
    first = ChromaSchemaRetriever(tmp_path / "first", "one")
    second = ChromaSchemaRetriever(tmp_path / "second", "two")

    assert first is not second
    assert first.chroma_path != second.chroma_path
    assert first.collection_name != second.collection_name
    assert not hasattr(first, "client")
    assert not hasattr(second, "client")


def test_adapter_has_schema_retriever_protocol_shape():
    retriever: SchemaRetriever = ChromaSchemaRetriever()

    assert callable(retriever.retrieve)
