
"""
SSA - Hybrid Schema Retrieval
"""

import re
import logging
from pathlib import Path

import chromadb

from ssa.schema_embedding import create_schema_embedding_function


ROOT_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = ROOT_DIR / "data" / "chroma_db"
COLLECTION_NAME = "ssa_schema"
logger = logging.getLogger(__name__)


def tokenize(text):
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def search_schema(question, n_results=5):
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=create_schema_embedding_function(),
    )

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

    question_tokens = tokenize(question)

    
    question_tokens = tokenize(question)

    for candidate in candidates:
        document_tokens = tokenize(
            candidate["id"] + " " + candidate["document"]
        )

        overlap = question_tokens.intersection(document_tokens)

        keyword_score = len(overlap) / max(len(question_tokens), 1)

        semantic_score = 1 / (1 + candidate["distance"])

        object_name = candidate["id"].split(":", 1)[1].lower()

        name_boost = 0.0

        question_lower = question.lower()

        # Sales order relevance
        if "sales order" in question_lower and "salesorder" in object_name:
            name_boost = 0.20

                # Customer name relevance
        if any(term in question_lower for term in [
            "customer name",
            "customer names",
            "customers by name",
            "including their names",
            "with their names",
        ]):
            if object_name == "sales.customer":
                name_boost = 0.40
            elif object_name == "person.person":
                name_boost = 0.40
            elif object_name == "sales.store":
                name_boost = 0.40
            elif object_name == "sales.salesorderheader":
                name_boost = 0.40

        # Product list price relevance
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

    logger.info("Schema retrieval completed: candidates=%d", len(candidates[:n_results]))

    return candidates[:n_results]


if __name__ == "__main__":
    search_schema(
        "Which tables store sales orders and order details?"
    )

    
