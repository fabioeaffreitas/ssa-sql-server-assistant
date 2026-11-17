from unittest.mock import Mock
import logging
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ssa.app_ordering_advisory import ORDERING_RISK_MESSAGE
from ssa.app_readiness import ReadinessCheck
from ssa import app_readiness, assistant


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture(autouse=True)
def restore_ssa_logger_after_app_test():
    logger = logging.getLogger("ssa")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    yield
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    for handler in original_handlers:
        logger.addHandler(handler)
    logger.setLevel(original_level)
    logger.propagate = original_propagate


def _response(
    *,
    ordering=None,
    diagnostics=True,
    validation_success=True,
    execution_success=True,
    truncated=False,
):
    response = {
        "question": "show top items",
        "sql": "SELECT TOP 10 ItemKey, ItemName FROM Catalog.Item ORDER BY Price DESC;",
        "generation": {
            "success": True,
            "usage": {
                "model": "fake-model",
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
            },
        },
        "validation": {
            "syntax": validation_success,
            "schema": validation_success,
            "success": validation_success,
        },
        "execution": {"success": execution_success},
        "result": (
            {
                "columns": ["ItemKey", "ItemName"],
                "rows": [{"ItemKey": 1, "ItemName": "One"}],
                "row_count": 1,
                "max_rows": 1 if truncated else 100,
                "truncated": truncated,
            }
            if execution_success
            else None
        ),
        "success": validation_success and execution_success,
    }
    if not validation_success:
        response.update({
            "error_stage": "validation",
            "error": "SQL validation failed.",
        })
    elif not execution_success:
        response.update({
            "error_stage": "execution",
            "error": "SQL execution failed.",
        })

    if diagnostics is True:
        response["diagnostics"] = {
            "timings_ms": {
                "retrieval_ms": 1.0,
                "generation_ms": 2.0,
                "validation_ms": 3.0,
                "execution_ms": 4.0,
                "total_ms": 10.0,
            },
            "failure_stage": None if response["success"] else response["error_stage"],
        }
        if ordering is not None:
            response["diagnostics"]["ordering"] = ordering
    elif diagnostics is not False:
        response["diagnostics"] = diagnostics

    return response


def _readiness_checks():
    return (
        ReadinessCheck("Schema index", "Ready", "1 document available."),
        ReadinessCheck(
            "SQL Server configuration",
            "Ready",
            "Configuration is valid; connectivity is not checked here.",
        ),
        ReadinessCheck(
            "Ollama configuration",
            "Ready",
            "Configuration is valid; service and model availability are not checked here.",
        ),
    )


def _run_app(monkeypatch, response):
    ask = Mock(return_value=response)
    monkeypatch.setattr(assistant, "ask", ask)
    monkeypatch.setattr(app_readiness, "collect_readiness", _readiness_checks)

    app = AppTest.from_file(APP_PATH, default_timeout=10)
    app.run()
    app.text_area[0].input("show top items")
    app.run()
    next(button for button in app.button if button.label == "Ask SSA").click()
    app.run()
    return app, ask


def _texts(elements):
    return [element.value for element in elements]


def test_risk_advisory_is_shown_once_with_successful_results(monkeypatch):
    app, ask = _run_app(
        monkeypatch,
        _response(ordering={"status": "risk"}),
    )

    assert ask.call_count == 1
    assert _texts(app.info).count(ORDERING_RISK_MESSAGE) == 1
    assert any("SELECT TOP 10 ItemKey" in text for text in _texts(app.code))
    assert app.success
    assert any("Query executed successfully" in text for text in _texts(app.success))
    assert len(app.dataframe) == 1
    assert app.dataframe[0].value.iloc[0]["ItemName"] == "One"
    assert not app.error


@pytest.mark.parametrize("status", ["sufficient", "unsupported"])
def test_nonrisk_advisories_do_not_show_ordering_note(monkeypatch, status):
    app, ask = _run_app(
        monkeypatch,
        _response(ordering={"status": status}),
    )

    assert ask.call_count == 1
    assert ORDERING_RISK_MESSAGE not in _texts(app.info)
    assert any("SELECT TOP 10 ItemKey" in text for text in _texts(app.code))
    assert len(app.dataframe) == 1


def test_validation_failure_with_risk_shows_validation_error_only(monkeypatch):
    app, ask = _run_app(
        monkeypatch,
        _response(
            ordering={"status": "risk"},
            validation_success=False,
            execution_success=False,
        ),
    )

    assert ask.call_count == 1
    assert ORDERING_RISK_MESSAGE not in _texts(app.info)
    assert "SQL validation failed." in _texts(app.error)
    assert len(app.dataframe) == 0


def test_execution_failure_with_risk_shows_execution_error_only(monkeypatch):
    app, ask = _run_app(
        monkeypatch,
        _response(ordering={"status": "risk"}, execution_success=False),
    )

    assert ask.call_count == 1
    assert ORDERING_RISK_MESSAGE not in _texts(app.info)
    assert "SQL execution failed." in _texts(app.error)
    assert len(app.dataframe) == 0


@pytest.mark.parametrize(
    "diagnostics",
    [
        False,
        None,
        "unexpected value",
        ["unexpected"],
        {"timings_ms": "invalid"},
        {"ordering": "invalid"},
        {"ordering": {"status": "unknown"}},
    ],
)
def test_missing_or_malformed_diagnostics_do_not_crash_or_warn(
    monkeypatch,
    diagnostics,
):
    app, ask = _run_app(
        monkeypatch,
        _response(diagnostics=diagnostics),
    )

    assert ask.call_count == 1
    assert ORDERING_RISK_MESSAGE not in _texts(app.info)
    assert len(app.dataframe) == 1
    assert not app.exception


def test_valid_timings_and_truncation_warning_remain_visible(monkeypatch):
    app, ask = _run_app(
        monkeypatch,
        _response(ordering={"status": "risk"}, truncated=True),
    )

    assert ask.call_count == 1
    assert ORDERING_RISK_MESSAGE in _texts(app.info)
    assert any("Results limited to the configured maximum row count." in text
               for text in _texts(app.warning))
    assert len(app.dataframe) == 1
    diagnostics = app.expander[0]
    assert diagnostics is not None
