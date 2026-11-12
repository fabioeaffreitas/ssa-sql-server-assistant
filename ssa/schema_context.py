
"""
SSA - Schema Context Generator

Generates a readable schema context from exported SQL Server metadata.
Includes tables, views, columns, descriptions, primary keys, and foreign keys.
"""

import json
from pathlib import Path


INPUT_FILE = Path("data/schema_metadata.json")
OUTPUT_FILE = Path("data/schema_context.txt")


def format_data_type(column):
    """Format a SQL Server column data type."""

    data_type = column["data_type"].lower()
    max_length = column.get("max_length")
    precision = column.get("precision")
    scale = column.get("scale")

    if data_type in ("varchar", "char", "varbinary", "binary"):
        if max_length == -1:
            return f"{data_type}(max)"
        return f"{data_type}({max_length})"

    if data_type in ("nvarchar", "nchar"):
        if max_length == -1:
            return f"{data_type}(max)"
        return f"{data_type}({max_length // 2})"

    if data_type in ("decimal", "numeric"):
        return f"{data_type}({precision},{scale})"

    if data_type in ("datetime2", "datetimeoffset", "time"):
        return f"{data_type}({scale})"

    return data_type


def format_columns(columns):
    """Format column information."""

    lines = []

    for column in columns:
        data_type = format_data_type(column)
        nullable = "NULL" if column["nullable"] else "NOT NULL"
        identity = " IDENTITY" if column.get("identity") else ""

        line = (
            f"  - {column['name']}: "
            f"{data_type} {nullable}{identity}"
        )

        if column.get("description"):
            line += f" | {column['description']}"

        lines.append(line)

    return lines


def format_table(table_name, table):
    """Format a table with columns and keys."""

    lines = [f"TABLE: {table_name}"]

    if table.get("description"):
        lines.append(f"Description: {table['description']}")

    lines.append("Columns:")
    lines.extend(format_columns(table.get("columns", [])))

    primary_keys = table.get("primary_keys", [])

    if primary_keys:
        lines.append(f"Primary Key: {', '.join(primary_keys)}")

    foreign_keys = table.get("foreign_keys", [])

    if foreign_keys:
        lines.append("Foreign Keys:")

        for fk in foreign_keys:
            lines.append(
                f"  - {fk['name']}: "
                f"{', '.join(fk['columns'])} -> "
                f"{fk['referenced_table']} "
                f"({', '.join(fk['referenced_columns'])})"
            )

    lines.append("")
    return lines


def format_view(view_name, view):
    """Format a view with columns."""

    lines = [f"VIEW: {view_name}"]

    if view.get("description"):
        lines.append(f"Description: {view['description']}")

    lines.append("Columns:")
    lines.extend(format_columns(view.get("columns", [])))
    lines.append("")

    return lines


def generate_schema_context(metadata):
    """Generate readable context from schema metadata."""

    lines = [
        f"Database: {metadata['database']}",
        f"Tables: {metadata.get('table_count', len(metadata.get('tables', {})))}",
        f"Views: {metadata.get('view_count', len(metadata.get('views', {})))}",
        "",
        "=" * 80,
        "TABLES",
        "=" * 80,
        "",
    ]

    for table_name, table in metadata.get("tables", {}).items():
        lines.extend(format_table(table_name, table))

    lines.extend([
        "=" * 80,
        "VIEWS",
        "=" * 80,
        "",
    ])

    for view_name, view in metadata.get("views", {}).items():
        lines.extend(format_view(view_name, view))

    return "\n".join(lines)


def export_schema_context():
    """Read JSON metadata and write schema context."""

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    context = generate_schema_context(metadata)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        file.write(context)

    print("Schema context generated successfully.")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Tables included: {metadata.get('table_count', len(metadata.get('tables', {})))}")
    print(f"Views included: {metadata.get('view_count', len(metadata.get('views', {})))}")
    print(f"Characters: {len(context):,}")


if __name__ == "__main__":
    export_schema_context()
