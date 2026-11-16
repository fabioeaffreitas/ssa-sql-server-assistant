
"""
SSA - SQL Schema Validator
Validates SQL table and column references against database metadata.
"""

import logging

from ssa.db import get_connection, get_query_timeout
from ssa.sql_validator import validate_sql, SQLValidationError


logger = logging.getLogger(__name__)

def validate_sql_schema(sql: str, query_timeout: int | None = None) -> str:
    """
    Validate SQL syntax and referenced tables against SQL Server metadata.
    """

    validated_sql = validate_sql(sql)

    timeout = get_query_timeout(query_timeout)
    connection = get_connection()
    cursor = None

    try:
        connection.timeout = timeout
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
        if cursor is not None:
            try:
                cursor.execute("SET NOEXEC OFF;")
            except Exception:
                pass

        raise SQLValidationError(
            "SQL schema validation failed. Check the referenced database schema."
        ) from exc

    finally:
        try:
            if cursor is not None:
                cursor.close()
        finally:
            try:
                connection.rollback()
            finally:
                connection.close()


if __name__ == "__main__":
    tests = [
        "SELECT TOP 10 ProductID FROM Production.Product",
        "SELECT CustomerName FROM Sales.Customer",
    ]

    for query in tests:
        try:
            validate_sql_schema(query)
            logger.info("SQL schema validation succeeded")
        except SQLValidationError as exc:
            logger.warning("SQL schema validation failed: error_type=%s", type(exc).__name__)
