
from unittest.mock import Mock

import pytest

from ssa import assistant
from ssa.contracts import QueryResult


def assert_generation_call(mock, question):
    mock.assert_called_once()
    assert mock.call_args.args == (question,)
    assert mock.call_args.kwargs["return_usage"] is True
    assert set(mock.call_args.kwargs["timings"]) == set(assistant.TIMING_KEYS)


def assert_execution_call(mock, sql):
    mock.assert_called_once()
    assert mock.call_args.args == (assistant.ValidatedQuery(sql),)
    assert set(mock.call_args.kwargs["timings"]) == set(assistant.TIMING_KEYS)


def test_ask_success(monkeypatch):
    question = "Test question"
    expected_sql = "SELECT 1"

    expected_result = {
        "columns": ["value"],
        "rows": [{"value": 1}],
        "row_count": 1,
        "max_rows": 100,
        "truncated": False,
    }

    generate_sql = Mock(return_value=expected_sql)
    execute_sql = Mock(return_value=expected_result)
    monkeypatch.setattr(assistant, "generate_sql", generate_sql)
    monkeypatch.setattr(assistant.sql_server_adapter, "execute_read", execute_sql)

    response = assistant.ask(question)

    assert response["success"] is True
    assert response["question"] == question
    assert response["sql"] == expected_sql
    assert response["generation"] == {
        "success": True,
    }
    assert response["validation"] == {
        "syntax": True,
        "schema": True,
        "success": True,
    }
    assert response["execution"] == {
        "success": True,
    }
    assert response["result"] == expected_result
    assert "error" not in response
    assert_generation_call(generate_sql, question)
    assert_execution_call(execute_sql, expected_sql)


def test_ask_uses_adapter_execution_without_redundant_validation(monkeypatch):
    execute_read = Mock(return_value=QueryResult(
        columns=["value"],
        rows=[{"value": 1}],
        row_count=1,
        max_rows=100,
        truncated=False,
    ))
    validate_read_query = Mock()
    database_adapter = Mock(
        execute_read=execute_read,
        validate_read_query=validate_read_query,
    )
    monkeypatch.setattr(assistant, "generate_sql", Mock(return_value="SELECT 1"))
    monkeypatch.setattr(assistant, "sql_server_adapter", database_adapter)

    response = assistant.ask("Test adapter activation")

    assert response["success"] is True
    assert response["result"] == {
        "columns": ["value"],
        "rows": [{"value": 1}],
        "row_count": 1,
        "max_rows": 100,
        "truncated": False,
    }
    execute_read.assert_called_once_with(
        assistant.ValidatedQuery("SELECT 1"),
        timings=response["diagnostics"]["timings_ms"] | {},
    )
    validate_read_query.assert_not_called()


def test_ask_collects_provider_neutral_timings(monkeypatch):
    clock = iter([10.0, 10.125])

    def fake_generate(question, return_usage, timings):
        timings["retrieval_ms"] = 12.5
        timings["generation_ms"] = 250.0
        timings["validation_ms"] = 4.0
        return "SELECT 1", {"total_tokens": 3}

    def fake_execute(sql, timings):
        timings["validation_ms"] += 6.0
        timings["execution_ms"] = 18.75
        return {
            "columns": ["value"],
            "rows": [{"value": 1}],
            "row_count": 1,
            "max_rows": 100,
            "truncated": False,
        }

    monkeypatch.setattr(assistant.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(assistant, "generate_sql", fake_generate)
    monkeypatch.setattr(assistant.sql_server_adapter, "execute_read", fake_execute)

    response = assistant.ask("Test timing")

    assert response["diagnostics"] == {
        "timings_ms": {
            "retrieval_ms": 12.5,
            "generation_ms": 250.0,
            "validation_ms": 10.0,
            "execution_ms": 18.75,
            "total_ms": 125.0,
        },
        "failure_stage": None,
    }
    assert response["generation"]["usage"] == {"total_tokens": 3}


def test_ask_preserves_completed_timings_on_generation_failure(monkeypatch):
    clock = iter([2.0, 2.4])

    def fail_generate(question, return_usage, timings):
        timings["retrieval_ms"] = 100.0
        timings["generation_ms"] = 300.0
        raise assistant.OllamaGenerationError("model unavailable")

    monkeypatch.setattr(assistant.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(assistant, "generate_sql", fail_generate)

    response = assistant.ask("Test generation timing failure")
    timings = response["diagnostics"]["timings_ms"]

    assert response["diagnostics"]["failure_stage"] == "generation"
    assert timings["retrieval_ms"] == 100.0
    assert timings["generation_ms"] == 300.0
    assert timings["validation_ms"] is None
    assert timings["execution_ms"] is None
    assert timings["total_ms"] == pytest.approx(400.0)


def test_ask_validation_error(monkeypatch):
    question = "Test invalid query"
    sql = "SELECT CustomerName FROM Sales.Customer"
    generate_sql = Mock(return_value=sql)
    execute_sql = Mock(
        side_effect=assistant.SQLValidationError("Invalid column")
    )
    monkeypatch.setattr(assistant, "generate_sql", generate_sql)
    monkeypatch.setattr(assistant.sql_server_adapter, "execute_read", execute_sql)

    response = assistant.ask(question)

    assert response["success"] is False
    assert response["sql"] is None
    assert response["generation"] == {
        "success": True,
    }
    assert response["validation"] == {
        "syntax": False,
        "schema": False,
        "success": False,
    }
    assert response["execution"] == {
        "success": False,
    }
    assert response["result"] is None
    assert response["error_stage"] == "validation"
    assert response["error"] == (
        "SQL validation failed. Check the query syntax and referenced schema."
    )
    assert_generation_call(generate_sql, question)
    assert_execution_call(execute_sql, sql)


def test_ask_schema_validation_error_is_reported_as_validation(monkeypatch):
    question = "Test schema validation failure"
    sql = "SELECT UnknownColumn FROM Sales.Customer"
    generate_sql = Mock(return_value=sql)
    execute_sql = Mock(
        side_effect=assistant.SQLValidationError("Invalid column in schema")
    )
    monkeypatch.setattr(assistant, "generate_sql", generate_sql)
    monkeypatch.setattr(assistant.sql_server_adapter, "execute_read", execute_sql)

    response = assistant.ask(question)

    assert response["success"] is False
    assert response["generation"] == {"success": True}
    assert response["validation"]["success"] is False
    assert response["execution"] == {"success": False}
    assert response["error_stage"] == "validation"
    assert response["error"] == (
        "SQL validation failed. Check the query syntax and referenced schema."
    )
    assert_generation_call(generate_sql, question)
    assert_execution_call(execute_sql, sql)


def test_ask_generation_error(monkeypatch):
    question = "Test generation failure"
    generate_sql = Mock(side_effect=RuntimeError("Model unavailable"))
    execute_sql = Mock()
    monkeypatch.setattr(assistant, "generate_sql", generate_sql)
    monkeypatch.setattr(assistant.sql_server_adapter, "execute_read", execute_sql)

    response = assistant.ask(question)

    assert response["success"] is False
    assert response["sql"] is None
    assert response["generation"] == {
        "success": False,
    }
    assert response["validation"] is None
    assert response["execution"] == {
        "success": False,
    }
    assert response["result"] is None
    assert response["error_stage"] == "generation"
    assert response["error"] == (
        "SSA could not complete the request. Check the application logs."
    )
    assert_generation_call(generate_sql, question)
    execute_sql.assert_not_called()


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (
            assistant.OllamaConnectionError("raw URL and credentials"),
            "Ollama is unavailable.",
        ),
        (
            assistant.OllamaGenerationTimeoutError("raw timeout details"),
            "Ollama generation timed out.",
        ),
        (
            assistant.OllamaGenerationError("raw model response"),
            "Ollama could not generate SQL.",
        ),
    ],
)
def test_ask_classifies_generation_failures_safely(monkeypatch, failure, message):
    generate_sql = Mock(side_effect=failure)
    monkeypatch.setattr(assistant, "generate_sql", generate_sql)

    response = assistant.ask("Test generation failure")

    assert response["error_stage"] == "generation"
    assert response["generation"] == {"success": False}
    assert response["error"].startswith(message)
    assert "raw" not in response["error"]
    assert "credentials" not in response["error"]


def test_ask_reports_execution_timeout_after_validation(monkeypatch):
    question = "Test timeout"
    sql = "SELECT 1"
    generate_sql = Mock(return_value=sql)
    execute_sql = Mock(
        side_effect=assistant.SQLQueryTimeoutError("SQL query timed out.")
    )
    monkeypatch.setattr(assistant, "generate_sql", generate_sql)
    monkeypatch.setattr(assistant.sql_server_adapter, "execute_read", execute_sql)

    response = assistant.ask(question)

    assert response["success"] is False
    assert response["validation"] == {
        "syntax": True,
        "schema": True,
        "success": True,
    }
    assert response["execution"] == {
        "success": False,
        "error_type": "timeout",
    }
    assert response["error_stage"] == "execution"
    assert_generation_call(generate_sql, question)
    assert_execution_call(execute_sql, sql)


def test_ask_reports_execution_error_after_validation(monkeypatch):
    question = "Test execution failure"
    sql = "SELECT 1"
    generate_sql = Mock(return_value=sql)
    execute_sql = Mock(
        side_effect=assistant.SQLExecutionError("Query execution failed")
    )
    monkeypatch.setattr(assistant, "generate_sql", generate_sql)
    monkeypatch.setattr(assistant.sql_server_adapter, "execute_read", execute_sql)

    response = assistant.ask(question)

    assert response["success"] is False
    assert response["validation"] == {
        "syntax": True,
        "schema": True,
        "success": True,
    }
    assert response["execution"] == {
        "success": False,
        "error_type": "execution",
    }
    assert response["error_stage"] == "execution"
    assert response["error"] == (
        "SQL execution failed. Check the database and generated query."
    )
    assert_generation_call(generate_sql, question)
    assert_execution_call(execute_sql, sql)


def test_ask_reports_execution_connection_error(monkeypatch):
    question = "Test execution connection error"
    sql = "SELECT 1"
    generate_sql = Mock(return_value=sql)
    execute_sql = Mock(
        side_effect=assistant.SQLExecutionConnectionError("Connection unavailable")
    )
    monkeypatch.setattr(assistant, "generate_sql", generate_sql)
    monkeypatch.setattr(assistant.sql_server_adapter, "execute_read", execute_sql)

    response = assistant.ask(question)

    assert response["validation"]["success"] is True
    assert response["execution"] == {
        "success": False,
        "error_type": "connection",
    }
    assert response["error_stage"] == "execution"
    assert_generation_call(generate_sql, question)
    assert_execution_call(execute_sql, sql)
