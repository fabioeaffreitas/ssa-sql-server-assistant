
"""
SSA - Schema Indexing
"""

import json
from pathlib import Path

import chromadb


ROOT_DIR = Path(__file__).resolve().parent.parent
METADATA_PATH = ROOT_DIR / "data" / "schema_metadata.json"
CHROMA_PATH = ROOT_DIR / "data" / "chroma_db"

COLLECTION_NAME = "ssa_schema"


def create_schema_documents(metadata):
    documents = []
    ids = []

    for object_type in ("tables", "views"):
        for object_name, obj in metadata.get(object_type, {}).items():
            lines = [
                f"Object type: {object_type[:-1]}",
                f"Name: {object_name}",
            ]

            description = obj.get("description")
            if description:
                lines.append(f"Description: {description}")

            lines.append("Columns:")

            for column in obj.get("columns", []):
                column_name = column.get("name", "")
                data_type = column.get("data_type", "")
                column_description = column.get("description", "")

                line = f"- {column_name} ({data_type})"

                if column_description:
                    line += f": {column_description}"

                lines.append(line)

            document = "\n".join(lines)

            documents.append(document)
            ids.append(f"{object_type[:-1]}:{object_name}")

    return documents, ids


def index_schema():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Schema metadata not found: {METADATA_PATH}"
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    documents, ids = create_schema_documents(metadata)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    collection.upsert(
        documents=documents,
        ids=ids,
    )

    print("SSA Schema Indexing")
    print("------------------")
    print(f"Documents indexed: {len(documents)}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Database: {CHROMA_PATH}")


if __name__ == "__main__":
    index_schema()
