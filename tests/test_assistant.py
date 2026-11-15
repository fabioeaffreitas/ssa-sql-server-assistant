
from unittest.mock import Mock

from ssa import assistant


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
    monkeypatch.setattr(assistant, "execute_sql", execute_sql)

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
    generate_sql.assert_called_once_with(question, return_usage=True)
    execute_sql.assert_called_once_with(expected_sql)


def test_ask_validation_error(monkeypatch):
    question = "Test invalid query"
    sql = "SELECT CustomerName FROM Sales.Customer"
    generate_sql = Mock(return_value=sql)
    execute_sql = Mock(
        side_effect=assistant.SQLValidationError("Invalid column")
    )
    monkeypatch.setattr(assistant, "generate_sql", generate_sql)
    monkeypatch.setattr(assistant, "execute_sql", execute_sql)

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
    assert "Invalid column" in response["error"]
    generate_sql.assert_called_once_with(question, return_usage=True)
    execute_sql.assert_called_once_with(sql)


def test_ask_schema_validation_error_is_reported_as_validation(monkeypatch):
    question = "Test schema validation failure"
    sql = "SELECT UnknownColumn FROM Sales.Customer"
    generate_sql = Mock(return_value=sql)
    execute_sql = Mock(
        side_effect=assistant.SQLValidationError("Invalid column in schema")
    )
    monkeypatch.setattr(assistant, "generate_sql", generate_sql)
    monkeypatch.setattr(assistant, "execute_sql", execute_sql)

    response = assistant.ask(question)

    assert response["success"] is False
    assert response["generation"] == {"success": True}
    assert response["validation"]["success"] is False
    assert response["execution"] == {"success": False}
    assert response["error_stage"] == "validation"
    assert "schema" in response["error"]
    generate_sql.assert_called_once_with(question, return_usage=True)
    execute_sql.assert_called_once_with(sql)


def test_ask_generation_error(monkeypatch):
    question = "Test generation failure"
    generate_sql = Mock(side_effect=RuntimeError("Model unavailable"))
    execute_sql = Mock()
    monkeypatch.setattr(assistant, "generate_sql", generate_sql)
    monkeypatch.setattr(assistant, "execute_sql", execute_sql)

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
    assert response["error_stage"] == "execution_or_generation"
    assert "Model unavailable" in response["error"]
    generate_sql.assert_called_once_with(question, return_usage=True)
    execute_sql.assert_not_called()


def test_ask_reports_execution_timeout_after_validation(monkeypatch):
    question = "Test timeout"
    sql = "SELECT 1"
    generate_sql = Mock(return_value=sql)
    execute_sql = Mock(
        side_effect=assistant.SQLQueryTimeoutError("SQL query timed out.")
    )
    monkeypatch.setattr(assistant, "generate_sql", generate_sql)
    monkeypatch.setattr(assistant, "execute_sql", execute_sql)

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
    generate_sql.assert_called_once_with(question, return_usage=True)
    execute_sql.assert_called_once_with(sql)


def test_ask_reports_execution_error_after_validation(monkeypatch):
    question = "Test execution failure"
    sql = "SELECT 1"
    generate_sql = Mock(return_value=sql)
    execute_sql = Mock(
        side_effect=assistant.SQLExecutionError("Query execution failed")
    )
    monkeypatch.setattr(assistant, "generate_sql", generate_sql)
    monkeypatch.setattr(assistant, "execute_sql", execute_sql)

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
    assert "Query execution failed" in response["error"]
    generate_sql.assert_called_once_with(question, return_usage=True)
    execute_sql.assert_called_once_with(sql)


def test_ask_reports_execution_connection_error(monkeypatch):
    question = "Test execution connection error"
    sql = "SELECT 1"
    generate_sql = Mock(return_value=sql)
    execute_sql = Mock(
        side_effect=assistant.SQLExecutionConnectionError("Connection unavailable")
    )
    monkeypatch.setattr(assistant, "generate_sql", generate_sql)
    monkeypatch.setattr(assistant, "execute_sql", execute_sql)

    response = assistant.ask(question)

    assert response["validation"]["success"] is True
    assert response["execution"] == {
        "success": False,
        "error_type": "connection",
    }
    assert response["error_stage"] == "execution"
    generate_sql.assert_called_once_with(question, return_usage=True)
    execute_sql.assert_called_once_with(sql)
