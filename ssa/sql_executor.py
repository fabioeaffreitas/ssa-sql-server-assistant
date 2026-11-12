
"""
SSA - SQL Executor
Executes validated SQL and returns query results.
"""

from ssa.db import get_connection
from ssa.sql_validator import validate_sql


MAX_ROWS = 100


def execute_sql(sql: str, max_rows: int = MAX_ROWS) -> dict:
    """
    Execute validated SQL and return columns and rows.

    Database permissions must enforce read-only access.
    """

    validated_sql = validate_sql(sql)

    connection = get_connection()

    try:
        cursor = connection.cursor()
        cursor.execute(validated_sql)

        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchmany(max_rows)

        results = [
            dict(zip(columns, row))
            for row in rows
        ]

        return {
            "columns": columns,
            "rows": results,
            "row_count": len(results),
            "max_rows": max_rows,
        }

    finally:
        connection.rollback()
        connection.close()


if __name__ == "__main__":
    query = """
    SELECT TOP 10 ProductID, Name, ListPrice
    FROM Production.Product
    ORDER BY ListPrice DESC;
    """

    result = execute_sql(query)

    print("\nCOLUMNS:")
    print(result["columns"])

    print("\nROWS:")
    for row in result["rows"]:
        print(row)

    print("\nReturned rows:", result["row_count"])
