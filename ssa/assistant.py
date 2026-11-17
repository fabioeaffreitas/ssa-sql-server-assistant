
"""
SSA - End-to-End Assistant
Natural language -> schema -> SQL -> validation -> execution -> results.
"""

from dataclasses import replace
import logging
import os
from pathlib import Path
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
from ssa.schema_key_metadata import load_primary_keys
from ssa.sql_ordering_safeguard import OrderingRisk


logger = logging.getLogger(__name__)
sql_server_adapter = SQLServerDatabaseAdapter()
SCHEMA_METADATA_PATH = Path(__file__).resolve().parent.parent / "data" / "schema_metadata.json"


def _finish_response(
    response: dict,
    diagnostics: PipelineDiagnostics,
    ordering_advisory: OrderingRisk | None = None,
) -> dict:
    """Attach diagnostics while preserving the existing response fields."""

    serialized = {
        "timings_ms": {
            "retrieval_ms": diagnostics.retrieval_ms,
            "generation_ms": diagnostics.generation_ms,
            "validation_ms": diagnostics.validation_ms,
            "execution_ms": diagnostics.execution_ms,
            "total_ms": diagnostics.total_ms,
        },
        "failure_stage": response.get("error_stage"),
    }
    if ordering_advisory is not None:
        serialized["ordering"] = {
            "status": ordering_advisory.status,
            "reason": ordering_advisory.reason,
            "table": ordering_advisory.table,
            "primary_key_columns": list(ordering_advisory.primary_key_columns),
            "missing_key_columns": list(ordering_advisory.missing_key_columns),
        }
    response["diagnostics"] = serialized
    return response


def _load_ordering_keys() -> dict[str, tuple[str, ...]] | None:
    """Load optional advisory metadata without affecting request execution."""

    expected_database = os.getenv("SQL_DATABASE")
    metadata = load_primary_keys(SCHEMA_METADATA_PATH, expected_database)
    if not metadata.available or not metadata.primary_keys:
        logger.info(
            "Ordering metadata unavailable: reason=%s",
            metadata.reason if metadata.primary_keys else "no_primary_keys",
        )
        return None
    return dict(metadata.primary_keys)


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

    logger.info("SSA request received")

    token_usage = None
    generation_succeeded = False
    diagnostics = None
    ordering_advisory = None

    try:
        try:
            logger.info("SQL generation started")
            pipeline_result = PipelineOrchestrator(
                generate_sql,
                sql_server_adapter,
                max_rows=100,
                timing_aware=True,
                primary_keys=_load_ordering_keys(),
            ).run(question)
            sql = pipeline_result.sql
            token_usage = pipeline_result.usage
            diagnostics = pipeline_result.diagnostics
            ordering_advisory = pipeline_result.ordering_advisory
        except PipelineExecutionError as pipeline_error:
            sql = pipeline_error.sql
            token_usage = pipeline_error.usage
            ordering_advisory = pipeline_error.ordering_advisory
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

        result = _legacy_result(pipeline_result.result)
        logger.info(
            "Query execution completed: rows=%d max_rows=%d truncated=%s",
            result["row_count"],
            result["max_rows"],
            result["truncated"],
        )

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
        }, diagnostics, ordering_advisory)

    except SQLValidationError as exc:
        error = "SQL validation failed. Check the query syntax and referenced schema."
        logger.warning("SQL validation failed: error_type=%s", type(exc).__name__)

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
        }, diagnostics, ordering_advisory)

    except ConfigurationError:
        logger.error("SSA configuration validation failed")

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
        }, diagnostics, ordering_advisory)

    except OllamaConnectionError:
        error = "Ollama is unavailable. Start Ollama and verify the local service."
        logger.error("SQL generation failed: category=connection")
        return _finish_response(
            _generation_failure(question, error, token_usage),
            diagnostics,
            ordering_advisory,
        )

    except OllamaGenerationTimeoutError:
        error = "Ollama generation timed out. Check Ollama or increase OLLAMA_REQUEST_TIMEOUT."
        logger.error("SQL generation failed: category=timeout")
        return _finish_response(
            _generation_failure(question, error, token_usage),
            diagnostics,
            ordering_advisory,
        )

    except OllamaGenerationError:
        error = "Ollama could not generate SQL. Verify the configured model is available."
        logger.error("SQL generation failed: category=provider")
        return _finish_response(
            _generation_failure(question, error, token_usage),
            diagnostics,
            ordering_advisory,
        )

    except DatabaseConnectionError as exc:
        logger.error("SQL connection failed")

        generation = {"success": generation_succeeded}
        if token_usage:
            generation["usage"] = token_usage

        return _finish_response({
            "question": question,
            "sql": sql if generation_succeeded else None,
            "generation": generation,
            "validation": None,
            "execution": {"success": False, "error_type": "connection"},
            "result": None,
            "success": False,
            "error_stage": "connection",
            "error": str(exc),
        }, diagnostics, ordering_advisory)

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

        logger.error("SQL execution failed: category=%s", error_type)

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
        }, diagnostics, ordering_advisory)

    except Exception as exc:
        error_stage = "execution" if generation_succeeded else "generation"
        error = "SSA could not complete the request. Check the application logs."
        logger.error(
            "SSA request failed: stage=%s error_type=%s",
            error_stage,
            type(exc).__name__,
        )

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
        }, diagnostics, ordering_advisory)


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
