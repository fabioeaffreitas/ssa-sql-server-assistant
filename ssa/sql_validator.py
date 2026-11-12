
"""
SSA - SQL Validator
Validates generated SQL before execution.
"""

import sqlglot
from sqlglot import exp


class SQLValidationError(Exception):
    """Raised when SQL fails SSA validation."""


def validate_sql(sql: str) -> str:
    """
    Validate SQL Server SQL.
    Accept exactly one SELECT statement.
    """

    if not sql or not sql.strip():
        raise SQLValidationError("SQL query is empty.")

    try:
        statements = sqlglot.parse(sql, read="tsql")
    except sqlglot.errors.ParseError as exc:
        raise SQLValidationError(f"SQL parsing failed: {exc}") from exc

    # Reject multiple statements
    if len(statements) != 1:
        raise SQLValidationError(
            "Only one SQL statement is allowed."
        )

    statement = statements[0]

    # Only SELECT queries are permitted
    if not isinstance(statement, exp.Query):
        raise SQLValidationError(
            "Only SELECT queries are allowed."
        )

    # Reject SELECT INTO, which creates a table
    if statement.find(exp.Into):
        raise SQLValidationError(
            "SELECT INTO is not allowed."
        )

    return sql.strip()


if __name__ == "__main__":
    test_queries = [
        "SELECT TOP 10 ProductID FROM Production.Product",
        "DELETE FROM Production.Product",
        "SELECT ProductID FROM Production.Product; "
        "DELETE FROM Production.Product",
        "SELECT ProductID INTO NewTable FROM Production.Product",
    ]

    for query in test_queries:
        print("\nQuery:", query)

        try:
            validated = validate_sql(query)
            print("VALID:", validated)
        except SQLValidationError as exc:
            print("REJECTED:", exc)
