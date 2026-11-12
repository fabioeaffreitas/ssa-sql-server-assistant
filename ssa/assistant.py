
"""
SSA - End-to-End Assistant
Natural language -> schema -> SQL -> validation -> execution -> results.
"""

from ssa.sql_generator import generate_sql
from ssa.sql_executor import execute_sql
from ssa.sql_validator import SQLValidationError
from ssa.sql_schema_validator import validate_sql_schema


def ask(question: str) -> dict:
    """Answer a natural-language question using the database."""

    print("\nQUESTION:")
    print(question)

    try:
        print("\nGENERATING SQL...")
        sql = generate_sql(question)

        print("\nGENERATED SQL:")
        print(sql)

        print("\nVALIDATING SQL SYNTAX AND DATABASE SCHEMA...")
        validated_sql = validate_sql_schema(sql)

        print("\nSQL VALIDATION SUCCESSFUL.")

        print("\nEXECUTING SQL...")
        result = execute_sql(validated_sql)

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

        return {
            "question": question,
            "sql": validated_sql,
            "generation": {
                "success": True,
            },
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

        return {
            "question": question,
            "sql": None,
            "generation": {
                "success": True,
            },
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

    except Exception as exc:
        print(f"\nSSA ERROR: {exc}")

        return {
            "question": question,
            "sql": None,
            "generation": {
                "success": False,
            },
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
