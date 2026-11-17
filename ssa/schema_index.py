
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


def _key_lines(obj):
    """Return conservative, human-readable declarations for table keys."""

    lines = []

    primary_keys = obj.get("primary_keys")
    if isinstance(primary_keys, list) and primary_keys and all(
        isinstance(column, str) and column.strip()
        for column in primary_keys
    ):
        columns = [column.strip() for column in primary_keys]
        primary = columns[0] if len(columns) == 1 else f"({', '.join(columns)})"
        lines.append(f"- Primary key: {primary}")

    foreign_keys = obj.get("foreign_keys")
    if isinstance(foreign_keys, list) and foreign_keys:
        formatted_foreign_keys = []
        for foreign_key in foreign_keys:
            if not isinstance(foreign_key, dict):
                formatted_foreign_keys = []
                break

            columns = foreign_key.get("columns")
            referenced_columns = foreign_key.get("referenced_columns")
            referenced_schema = foreign_key.get("referenced_schema")
            referenced_table = foreign_key.get("referenced_table")

            if (
                not isinstance(columns, list)
                or not columns
                or not all(
                    isinstance(column, str) and column.strip()
                    for column in columns
                )
                or not isinstance(referenced_columns, list)
                or len(referenced_columns) != len(columns)
                or not all(
                    isinstance(column, str) and column.strip()
                    for column in referenced_columns
                )
                or not isinstance(referenced_schema, str)
                or not referenced_schema.strip()
                or not isinstance(referenced_table, str)
                or not referenced_table.strip()
            ):
                formatted_foreign_keys = []
                break

            parent = [column.strip() for column in columns]
            referenced = [column.strip() for column in referenced_columns]
            parent_text = parent[0] if len(parent) == 1 else f"({', '.join(parent)})"
            referenced_text = ", ".join(referenced)
            formatted_foreign_keys.append(
                "- Foreign key: "
                f"{parent_text} -> "
                f"{referenced_schema.strip()}.{referenced_table.strip()}"
                f"({referenced_text})"
            )

        if formatted_foreign_keys:
            lines.extend(formatted_foreign_keys)

    return lines


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

            if object_type == "tables":
                key_lines = _key_lines(obj)
                if key_lines:
                    lines.append("Keys:")
                    lines.extend(key_lines)

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
