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
    assert response["result"] == expected_result
    assert response["error"] if "error" in response else True


def test_ask_validation_error(monkeypatch):
    monkeypatch.setattr(
        assistant,
        "generate_sql",
        lambda question: "SELECT CustomerName FROM Sales.Customer",
    )

    def fake_execute(sql):
        raise assistant.SQLValidationError("Invalid column")

    monkeypatch.setattr(
        assistant,
        "execute_sql",
        fake_execute,
    )

    response = assistant.ask("Test invalid query")

    assert response["success"] is False
    assert response["sql"] is None
    assert response["result"] is None
    assert "Invalid column" in response["error"]