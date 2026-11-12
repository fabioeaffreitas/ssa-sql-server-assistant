
"""
SSA - SQL Schema Validator
Validates SQL table and column references against database metadata.
"""

from ssa.db import get_connection
from ssa.sql_validator import validate_sql, SQLValidationError


def validate_sql_schema(sql: str) -> str:
    """
    Validate SQL syntax and referenced tables against SQL Server metadata.
    """

    validated_sql = validate_sql(sql)

    connection = get_connection()

    try:
        cursor = connection.cursor()

        # Ask SQL Server to compile the query without executing it.
        cursor.execute(
            """
            SET NOEXEC ON;
            """
            + validated_sql
            + """
            ;
            SET NOEXEC OFF;
            """
        )

        return validated_sql

    except Exception as exc:
        try:
            cursor.execute("SET NOEXEC OFF;")
        except Exception:
            pass

        raise SQLValidationError(
            f"SQL schema validation failed: {exc}"
        ) from exc

    finally:
        connection.rollback()
        connection.close()


if __name__ == "__main__":
    tests = [
        "SELECT TOP 10 ProductID FROM Production.Product",
        "SELECT CustomerName FROM Sales.Customer",
    ]

    for query in tests:
        print("\nQuery:", query)

        try:
            validate_sql_schema(query)
            print("VALID")
        except SQLValidationError as exc:
            print("REJECTED:", exc)
