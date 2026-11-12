
"""
SSA - SQL Executor
Executes schema-validated SQL and returns query results.
"""

from ssa.db import get_connection
from ssa.sql_schema_validator import validate_sql_schema

MAX_ROWS = 100


def execute_sql(sql: str, max_rows: int = MAX_ROWS) -> dict:
    """
    Validate SQL against database metadata and execute it.

    Results are limited to max_rows. The executor checks whether
    additional rows exist beyond the returned result set.
    """

    if max_rows < 1:
        raise ValueError("max_rows must be greater than zero.")

    validated_sql = validate_sql_schema(sql)
    connection = get_connection()

    try:
        cursor = connection.cursor()
        cursor.execute(validated_sql)

        columns = [column[0] for column in cursor.description]

        # Fetch one extra row to determine whether results are truncated.
        rows = cursor.fetchmany(max_rows + 1)

        truncated = len(rows) > max_rows
        rows = rows[:max_rows]

        results = [
            dict(zip(columns, row))
            for row in rows
        ]

        return {
            "columns": columns,
            "rows": results,
            "row_count": len(results),
            "max_rows": max_rows,
            "truncated": truncated,
        }

    finally:
        connection.rollback()
        connection.close()
