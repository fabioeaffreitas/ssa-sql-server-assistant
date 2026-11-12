
"""
SSA - End-to-End Assistant
Natural language -> schema -> SQL -> validation -> execution -> results.
"""

from ssa.sql_generator import generate_sql
from ssa.sql_executor import execute_sql
from ssa.sql_validator import SQLValidationError


def ask(question: str) -> dict:
    """Answer a natural-language question using the database."""

    print("\nQUESTION:")
    print(question)

    try:
        print("\nGENERATING SQL...")
        sql = generate_sql(question)

        print("\nGENERATED SQL:")
        print(sql)

        print("\nVALIDATING AND EXECUTING SQL...")
        result = execute_sql(sql)

        print("\nRESULTS:")
        for row in result["rows"]:
            print(row)

        print(f"\nReturned rows: {result['row_count']}")

        return {
            "question": question,
            "sql": sql,
            "result": result,
            "success": True,
        }

    except SQLValidationError as exc:
        print(f"\nSQL VALIDATION ERROR: {exc}")

        return {
            "question": question,
            "sql": None,
            "result": None,
            "success": False,
            "error": str(exc),
        }

    except Exception as exc:
        print(f"\nSSA ERROR: {exc}")

        return {
            "question": question,
            "sql": None,
            "result": None,
            "success": False,
            "error": str(exc),
        }


if __name__ == "__main__":
    question = (
        "Show the top 10 customers by total sales "
        "from sales orders, using CustomerID only."
    )

    ask(question)
