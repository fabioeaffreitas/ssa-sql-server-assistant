
from ssa import assistant


def test_ask_success(monkeypatch):
    expected_sql = "SELECT 1"

    expected_result = {
        "columns": ["value"],
        "rows": [{"value": 1}],
        "row_count": 1,
        "max_rows": 100,
        "truncated": False,
    }

    monkeypatch.setattr(
        assistant,
        "generate_sql",
        lambda question: expected_sql,
    )

    monkeypatch.setattr(
        assistant,
        "execute_sql",
        lambda sql: expected_result,
    )

    response = assistant.ask("Test question")

    assert response["success"] is True
    assert response["question"] == "Test question"
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


def test_ask_validation_error(monkeypatch):
    monkeypatch.setattr(
        assistant,
        "generate_sql",
        lambda question: "SELECT CustomerName FROM Sales.Customer",
    )

    monkeypatch.setattr(
        assistant,
        "execute_sql",
        lambda sql: (_ for _ in ()).throw(
            assistant.SQLValidationError("Invalid column")
        ),
    )

    response = assistant.ask("Test invalid query")

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


def test_ask_generation_error(monkeypatch):
    def fake_generate(question):
        raise RuntimeError("Model unavailable")

    monkeypatch.setattr(
        assistant,
        "generate_sql",
        fake_generate,
    )

    response = assistant.ask("Test generation failure")

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


def test_ask_reports_execution_timeout_after_validation(monkeypatch):
    monkeypatch.setattr(
        assistant,
        "generate_sql",
        lambda question: "SELECT 1",
    )

    monkeypatch.setattr(
        assistant,
        "execute_sql",
        lambda sql: (_ for _ in ()).throw(
            assistant.SQLQueryTimeoutError("SQL query timed out.")
        ),
    )

    response = assistant.ask("Test timeout")

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
