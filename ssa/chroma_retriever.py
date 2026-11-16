"""Provider-neutral adapter for the existing ChromaDB schema retrieval path."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import chromadb

from ssa import schema_retrieval
from ssa.contracts import SchemaDocument, SchemaRetriever


@dataclass(frozen=True)
class ChromaSchemaDocument(SchemaDocument):
    """Schema document retaining legacy hybrid-ranking metadata."""

    metadata: Mapping[str, object] = field(default_factory=dict)


class ChromaSchemaRetriever:
    """Expose the existing hybrid ChromaDB search through ``SchemaRetriever``."""

    def __init__(
        self,
        chroma_path: str | Path = schema_retrieval.CHROMA_PATH,
        collection_name: str = schema_retrieval.COLLECTION_NAME,
    ) -> None:
        self.chroma_path = Path(chroma_path)
        self.collection_name = collection_name

    def retrieve(
        self,
        question: str,
        limit: int = 5,
    ) -> list[SchemaDocument]:
        """Return ranked schema documents without retaining client state."""

        if (
            self.chroma_path == schema_retrieval.CHROMA_PATH
            and self.collection_name == schema_retrieval.COLLECTION_NAME
        ):
            candidates = schema_retrieval.search_schema(
                question,
                n_results=limit,
            )
        else:
            candidates = self._search_with_configuration(question, limit)

        return [self._to_document(candidate) for candidate in candidates]

    def _search_with_configuration(
        self,
        question: str,
        limit: int,
    ) -> list[dict]:
        """Run the existing hybrid algorithm against custom Chroma settings."""

        client = chromadb.PersistentClient(path=str(self.chroma_path))
        collection = client.get_collection(name=self.collection_name)
        semantic_results = collection.query(
            query_texts=[question],
            n_results=collection.count(),
        )

        candidates = []
        for document_id, document, distance in zip(
            semantic_results["ids"][0],
            semantic_results["documents"][0],
            semantic_results["distances"][0],
        ):
            candidates.append({
                "id": document_id,
                "document": document,
                "distance": distance,
            })

        question_tokens = schema_retrieval.tokenize(question)
        for candidate in candidates:
            document_tokens = schema_retrieval.tokenize(
                candidate["id"] + " " + candidate["document"]
            )
            overlap = question_tokens.intersection(document_tokens)
            keyword_score = len(overlap) / max(len(question_tokens), 1)
            semantic_score = 1 / (1 + candidate["distance"])
            object_name = candidate["id"].split(":", 1)[1].lower()
            name_boost = 0.0
            question_lower = question.lower()

            if "sales order" in question_lower and "salesorder" in object_name:
                name_boost = 0.20

            if any(term in question_lower for term in [
                "customer name",
                "customer names",
                "customers by name",
                "including their names",
                "with their names",
            ]):
                if object_name in {
                    "sales.customer",
                    "person.person",
                    "sales.store",
                    "sales.salesorderheader",
                }:
                    name_boost = 0.40

            if (
                "product" in question_lower
                and "list price" in question_lower
                and object_name == "production.product"
            ):
                name_boost = 0.30

            candidate["keyword_score"] = keyword_score
            candidate["semantic_score"] = semantic_score
            candidate["name_boost"] = name_boost
            candidate["final_score"] = (
                0.6 * semantic_score
                + 0.2 * keyword_score
                + name_boost
            )

        candidates.sort(
            key=lambda item: item["final_score"],
            reverse=True,
        )
        return candidates[:limit]

    @staticmethod
    def _to_document(candidate: dict) -> ChromaSchemaDocument:
        metadata = {
            key: candidate[key]
            for key in (
                "keyword_score",
                "semantic_score",
                "name_boost",
                "final_score",
            )
            if key in candidate
        }
        return ChromaSchemaDocument(
            id=candidate["id"],
            document=candidate["document"],
            distance=candidate.get("distance"),
            metadata=metadata,
        )


def _protocol_check(retriever: ChromaSchemaRetriever) -> SchemaRetriever:
    """Keep the adapter's protocol compatibility visible to type checkers."""

    return retriever
