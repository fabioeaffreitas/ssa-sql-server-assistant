
"""
SSA - Semantic Schema Search
"""

from pathlib import Path

import chromadb


ROOT_DIR = Path(__file__).resolve().parent.parent
CHROMA_PATH = ROOT_DIR / "data" / "chroma_db"
COLLECTION_NAME = "ssa_schema"


def search_schema(question, n_results=5):
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    results = collection.query(
        query_texts=[question],
        n_results=n_results,
    )

    documents = results["documents"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    print(f"\nSchema Search: {question}")
    print("-" * 60)

    for document_id, document, distance in zip(
        ids, documents, distances
    ):
        print(f"\nID: {document_id}")
        print(f"Distance: {distance:.4f}")
        print(document)


if __name__ == "__main__":
    questions = [
        "Which tables contain customer information?",
        "Which tables store sales orders and order details?",
        "Which tables contain product information and prices?",
    ]

    for question in questions:
        search_schema(question)
