
"""
SSA - End-to-End Assistant
Natural language -> schema -> SQL -> validation -> execution -> results.
"""

from dataclasses import replace
import time

from ssa.contracts import PipelineDiagnostics, QueryResult, ValidatedQuery
from ssa.pipeline import PipelineExecutionError, PipelineOrchestrator, TIMING_KEYS
from ssa.sql_server_adapter import SQLServerDatabaseAdapter
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
)
from ssa.sql_validator import SQLValidationError
from ssa.db import ConfigurationError, DatabaseConnectionError


sql_server_adapter = SQLServerDatabaseAdapter()


def _finish_response(
    response: dict,
    diagnostics: PipelineDiagnostics,
) -> dict:
    """Attach diagnostics while preserving the existing response fields."""

    response["diagnostics"] = {
        "timings_ms": {
            "retrieval_ms": diagnostics.retrieval_ms,
            "generation_ms": diagnostics.generation_ms,
            "validation_ms": diagnostics.validation_ms,
            "execution_ms": diagnostics.execution_ms,
            "total_ms": diagnostics.total_ms,
        },
        "failure_stage": response.get("error_stage"),
    }
    return response


def _failure_stage(cause: Exception, fallback: str) -> str:
    """Preserve the legacy public stage for known exception classes."""

    if isinstance(cause, SQLValidationError):
        return "validation"
    if isinstance(cause, DatabaseConnectionError):
        return "connection"
    if isinstance(
        cause,
        (
            OllamaConnectionError,
            OllamaGenerationError,
            OllamaGenerationTimeoutError,
        ),
    ):
        return "generation"
    return fallback


def _legacy_result(result: QueryResult | dict) -> dict:
    """Convert the typed database result to the existing public shape."""

    if isinstance(result, dict):
        return result

    return {
        "columns": result.columns,
        "rows": result.rows,
        "row_count": result.row_count,
        "max_rows": result.max_rows,
        "truncated": result.truncated,
    }


def ask(question: str) -> dict:
    """Answer a natural-language question using the database."""

    print("\nQUESTION:")
    print(question)

    token_usage = None
    generation_succeeded = False
    diagnostics = None

    try:
        try:
            print("\nGENERATING SQL...")
            pipeline_result = PipelineOrchestrator(
                generate_sql,
                sql_server_adapter,
                max_rows=100,
                timing_aware=True,
            ).run(question)
            sql = pipeline_result.sql
            token_usage = pipeline_result.usage
            diagnostics = pipeline_result.diagnostics
        except PipelineExecutionError as pipeline_error:
            sql = pipeline_error.sql
            token_usage = pipeline_error.usage
            generation_succeeded = pipeline_error.stage != "generation"
            diagnostics = replace(
                pipeline_error.diagnostics,
                failure_stage=_failure_stage(
                    pipeline_error.cause,
                    pipeline_error.stage,
                ),
            )
            raise pipeline_error.cause

        generation_succeeded = True

        print("\nGENERATED SQL:")
        print(sql)

        print("\nVALIDATING SQL SYNTAX, DATABASE SCHEMA, AND EXECUTING SQL...")
        result = _legacy_result(pipeline_result.result)

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
        }, diagnostics)

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
        }, diagnostics)

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
        }, diagnostics)

    except OllamaConnectionError:
        error = "Ollama is unavailable. Start Ollama and verify the local service."
        print(f"\nSQL GENERATION ERROR: {error}")
        return _finish_response(
            _generation_failure(question, error, token_usage),
            diagnostics,
        )

    except OllamaGenerationTimeoutError:
        error = "Ollama generation timed out. Check Ollama or increase OLLAMA_REQUEST_TIMEOUT."
        print(f"\nSQL GENERATION ERROR: {error}")
        return _finish_response(
            _generation_failure(question, error, token_usage),
            diagnostics,
        )

    except OllamaGenerationError:
        error = "Ollama could not generate SQL. Verify the configured model is available."
        print(f"\nSQL GENERATION ERROR: {error}")
        return _finish_response(
            _generation_failure(question, error, token_usage),
            diagnostics,
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
        }, diagnostics)

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
        }, diagnostics)

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
        }, diagnostics)


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
