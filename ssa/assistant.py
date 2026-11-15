
"""
SSA - End-to-End Assistant
Natural language -> schema -> SQL -> validation -> execution -> results.
"""

import time

from ssa.sql_generator import generate_sql
from ssa.sql_generator import (
    OllamaConnectionError,
    OllamaGenerationError,
    OllamaGenerationTimeoutError,
)
from ssa.sql_executor import (
    SQLExecutionConnectionError,
    SQLExecutionError,
    SQLQueryCancelledError,
    SQLQueryTimeoutError,
    execute_sql,
)
from ssa.sql_validator import SQLValidationError
from ssa.db import ConfigurationError, DatabaseConnectionError


TIMING_KEYS = (
    "retrieval_ms",
    "generation_ms",
    "validation_ms",
    "execution_ms",
    "total_ms",
)


def _new_timings() -> dict:
    """Return the stable provider-neutral timing shape."""

    return {key: None for key in TIMING_KEYS}


def _finish_response(response: dict, timings: dict, started_at: float) -> dict:
    """Attach diagnostics while preserving the existing response fields."""

    timings["total_ms"] = (time.perf_counter() - started_at) * 1000
    response["diagnostics"] = {
        "timings_ms": {key: timings.get(key) for key in TIMING_KEYS},
        "failure_stage": response.get("error_stage"),
    }
    return response


def ask(question: str) -> dict:
    """Answer a natural-language question using the database."""

    started_at = time.perf_counter()
    timings = _new_timings()

    print("\nQUESTION:")
    print(question)

    token_usage = None
    generation_succeeded = False

    try:
        print("\nGENERATING SQL...")
        # ``return_usage`` lets the UI show Ollama's actual token accounting.
        # The fallback keeps compatibility with older/test doubles of generate_sql.
        try:
            generated = generate_sql(
                question,
                return_usage=True,
                timings=timings,
            )
        except TypeError as exc:
            if "timings" not in str(exc):
                raise
            try:
                generated = generate_sql(question, return_usage=True)
            except TypeError as fallback_exc:
                if "return_usage" not in str(fallback_exc):
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
        try:
            result = execute_sql(sql, timings=timings)
        except TypeError as exc:
            if "timings" not in str(exc):
                raise
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

        return _finish_response({
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
        }, timings, started_at)

    except SQLValidationError as exc:
        error = "SQL validation failed. Check the query syntax and referenced schema."
        print(f"\nSQL VALIDATION ERROR: {type(exc).__name__}")

        generation = {"success": generation_succeeded}
        if token_usage:
            generation["usage"] = token_usage

        return _finish_response({
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
            "error": error,
        }, timings, started_at)

    except ConfigurationError:
        print("\nSSA CONFIGURATION ERROR")

        generation = {"success": generation_succeeded}
        if token_usage:
            generation["usage"] = token_usage

        return _finish_response({
            "question": question,
            "sql": None,
            "generation": generation,
            "validation": None,
            "execution": {"success": False},
            "result": None,
            "success": False,
            "error_stage": "configuration",
            "error": "SSA configuration is invalid. Check the configured environment variables.",
        }, timings, started_at)

    except OllamaConnectionError:
        error = "Ollama is unavailable. Start Ollama and verify the local service."
        print(f"\nSQL GENERATION ERROR: {error}")
        return _finish_response(
            _generation_failure(question, error, token_usage),
            timings,
            started_at,
        )

    except OllamaGenerationTimeoutError:
        error = "Ollama generation timed out. Check Ollama or increase OLLAMA_REQUEST_TIMEOUT."
        print(f"\nSQL GENERATION ERROR: {error}")
        return _finish_response(
            _generation_failure(question, error, token_usage),
            timings,
            started_at,
        )

    except OllamaGenerationError:
        error = "Ollama could not generate SQL. Verify the configured model is available."
        print(f"\nSQL GENERATION ERROR: {error}")
        return _finish_response(
            _generation_failure(question, error, token_usage),
            timings,
            started_at,
        )

    except DatabaseConnectionError as exc:
        print(f"\nSQL CONNECTION ERROR: {exc}")

        generation = {"success": generation_succeeded}
        if token_usage:
            generation["usage"] = token_usage

        return _finish_response({
            "question": question,
            "sql": None,
            "generation": generation,
            "validation": None,
            "execution": {"success": False, "error_type": "connection"},
            "result": None,
            "success": False,
            "error_stage": "connection",
            "error": str(exc),
        }, timings, started_at)

    except SQLExecutionError as exc:
        if isinstance(exc, SQLQueryTimeoutError):
            error_type = "timeout"
            error = "SQL query timed out. Try a narrower question or increase SQL_QUERY_TIMEOUT."
        elif isinstance(exc, SQLQueryCancelledError):
            error_type = "cancelled"
            error = "SQL query was cancelled."
        elif isinstance(exc, SQLExecutionConnectionError):
            error_type = "connection"
            error = "SQL Server is unavailable. Check the database connection settings."
        else:
            error_type = "execution"
            error = "SQL execution failed. Check the database and generated query."

        print(f"\nSQL EXECUTION ERROR: {error_type}")

        generation = {"success": generation_succeeded}
        if token_usage:
            generation["usage"] = token_usage

        return _finish_response({
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
            "error": error,
        }, timings, started_at)

    except Exception as exc:
        error_stage = "execution" if generation_succeeded else "generation"
        error = "SSA could not complete the request. Check the application logs."
        print(f"\nSSA ERROR ({error_stage}): {type(exc).__name__}")

        generation = {"success": generation_succeeded}
        if token_usage:
            generation["usage"] = token_usage

        return _finish_response({
            "question": question,
            "sql": sql if generation_succeeded else None,
            "generation": generation,
            "validation": None,
            "execution": {
                "success": False,
            },
            "result": None,
            "success": False,
            "error_stage": error_stage,
            "error": error,
        }, timings, started_at)


def _generation_failure(question: str, error: str, token_usage: dict | None) -> dict:
    """Build a stable, safe response for Ollama generation failures."""

    generation = {"success": False}
    if token_usage:
        generation["usage"] = token_usage
    return {
        "question": question,
        "sql": None,
        "generation": generation,
        "validation": None,
        "execution": {"success": False},
        "result": None,
        "success": False,
        "error_stage": "generation",
        "error": error,
    }


if __name__ == "__main__":
    question = (
        "Show the top 10 customers by total sales "
        "from sales orders, using CustomerID only."
    )

    ask(question)
