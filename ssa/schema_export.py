
"""
SSA - Schema Metadata Export
"""

import json
import os

from ssa.db import get_connection
from ssa.schema import get_schema_metadata


def export_schema(output_path="data/schema_metadata.json"):
    """Export SQL Server schema metadata to JSON."""
    with get_connection() as connection:
        metadata = get_schema_metadata(connection)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print(f"Schema exported to: {output_path}")
    print(f"Tables: {metadata['table_count']}")
    print(f"Foreign keys: {metadata['foreign_key_count']}")


if __name__ == "__main__":
    export_schema()
