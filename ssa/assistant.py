
"""
SSA - End-to-End Assistant
Natural language -> schema -> SQL -> validation -> execution -> results.
"""

from ssa.sql_generator import generate_sql
from ssa.sql_executor import (
    SQLExecutionConnectionError,
    SQLExecutionError,
    SQLQueryCancelledError,
    SQLQueryTimeoutError,
    execute_sql,
)
from ssa.sql_validator import SQLValidationError
from ssa.db import DatabaseConnectionError


def ask(question: str) -> dict:
    """Answer a natural-language question using the database."""

    print("\nQUESTION:")
    print(question)

    token_usage = None
    generation_succeeded = False

    try:
        print("\nGENERATING SQL...")
        # ``return_usage`` lets the UI show Ollama's actual token accounting.
        # The fallback keeps compatibility with older/test doubles of generate_sql.
        try:
            generated = generate_sql(question, return_usage=True)
        except TypeError as exc:
            if "return_usage" not in str(exc):
                raise
            generated = generate_sql(question)

        if isinstance(generated, tuple):
            sql, token_usage = generated
        else:
            sql, token_usage = generated, None
        generation_succeeded = True

        print("\nGENERATED SQL:")
        print(sql)

        print("\nVALIDATING SQL SYNTAX, DATABASE SCHEMA, AND EXECUTING SQL...")
        result = execute_sql(sql)

        print("\nRESULTS:")
        for row in result["rows"]:
            print(row)

        print("\nRESULT SUMMARY:")
        print(f"Returned rows: {result['row_count']}")
        print(f"Maximum rows: {result['max_rows']}")

        if result["truncated"]:
            print(
                f"WARNING: Results truncated. "
                f"More than {result['max_rows']} rows are available."
            )
        else:
            print("Results complete: no additional rows detected.")

        generation = {"success": True}
        if token_usage:
            generation["usage"] = token_usage

        return {
            "question": question,
            "sql": sql,
            "generation": generation,
            "validation": {
                "syntax": True,
                "schema": True,
                "success": True,
            },
            "execution": {
                "success": True,
            },
            "result": result,
            "success": True,
        }

    except SQLValidationError as exc:
        print(f"\nSQL VALIDATION ERROR: {exc}")

        generation = {"success": generation_succeeded}
        if token_usage:
            generation["usage"] = token_usage

        return {
            "question": question,
            "sql": None,
            "generation": generation,
            "validation": {
                "syntax": False,
                "schema": False,
                "success": False,
            },
            "execution": {
                "success": False,
            },
            "result": None,
            "success": False,
            "error_stage": "validation",
            "error": str(exc),
        }

    except DatabaseConnectionError as exc:
        print(f"\nSQL CONNECTION ERROR: {exc}")

        generation = {"success": generation_succeeded}
        if token_usage:
            generation["usage"] = token_usage

        return {
            "question": question,
            "sql": None,
            "generation": generation,
            "validation": None,
            "execution": {"success": False, "error_type": "connection"},
            "result": None,
            "success": False,
            "error_stage": "connection",
            "error": str(exc),
        }

    except SQLExecutionError as exc:
        print(f"\nSQL EXECUTION ERROR: {exc}")

        if isinstance(exc, SQLQueryTimeoutError):
            error_type = "timeout"
        elif isinstance(exc, SQLQueryCancelledError):
            error_type = "cancelled"
        elif isinstance(exc, SQLExecutionConnectionError):
            error_type = "connection"
        else:
            error_type = "execution"

        generation = {"success": generation_succeeded}
        if token_usage:
            generation["usage"] = token_usage

        return {
            "question": question,
            "sql": sql if generation_succeeded else None,
            "generation": generation,
            "validation": {
                "syntax": True,
                "schema": True,
                "success": True,
            },
            "execution": {"success": False, "error_type": error_type},
            "result": None,
            "success": False,
            "error_stage": "execution",
            "error": str(exc),
        }

    except Exception as exc:
        print(f"\nSSA ERROR: {exc}")

        generation = {"success": generation_succeeded}
        if token_usage:
            generation["usage"] = token_usage

        return {
            "question": question,
            "sql": None,
            "generation": generation,
            "validation": None,
            "execution": {
                "success": False,
            },
            "result": None,
            "success": False,
            "error_stage": "execution_or_generation",
            "error": str(exc),
        }


if __name__ == "__main__":
    question = (
        "Show the top 10 customers by total sales "
        "from sales orders, using CustomerID only."
    )

    ask(question)
