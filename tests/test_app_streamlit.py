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
    empty_result=False,
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
                "rows": [] if empty_result else [{"ItemKey": 1, "ItemName": "One"}],
                "row_count": 0 if empty_result else 1,
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


def _run_app(monkeypatch, response, *, ask_side_effect=None):
    ask = Mock(
        side_effect=ask_side_effect,
    ) if ask_side_effect is not None else Mock(return_value=response)
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


def _result_dataframes(app):
    return [
        element.value
        for element in app.dataframe
        if list(element.value.columns) == ["ItemKey", "ItemName"]
    ]


def test_app_uses_supported_streamlit_width_api():
    source = APP_PATH.read_text(encoding="utf-8")

    assert "use_container_width" not in source
    assert source.count('width="stretch"') == 3


def test_initial_page_has_no_stored_response(monkeypatch):
    monkeypatch.setattr(assistant, "ask", Mock())
    monkeypatch.setattr(app_readiness, "collect_readiness", _readiness_checks)

    app = AppTest.from_file(APP_PATH, default_timeout=10)
    app.run()

    assert not app.code
    assert not app.dataframe
    assert app.session_state["last_response"] is None


def test_harmless_rerun_preserves_response_without_asking_again(monkeypatch):
    app, ask = _run_app(monkeypatch, _response(ordering={"status": "risk"}))
    app.run()

    assert ask.call_count == 1
    assert any("SELECT TOP 10 ItemKey" in text for text in _texts(app.code))
    assert len(_result_dataframes(app)) == 1
    assert ORDERING_RISK_MESSAGE in _texts(app.info)


def test_editing_question_does_not_regenerate_or_change_response(monkeypatch):
    app, ask = _run_app(monkeypatch, _response())
    app.text_area[0].input("a different question")
    app.run()

    assert ask.call_count == 1
    assert len(_result_dataframes(app)) == 1
    assert "Question: show top items" in _texts(app.caption)


def test_sidebar_budget_change_does_not_regenerate(monkeypatch):
    app, ask = _run_app(monkeypatch, _response())
    app.number_input[0].set_value(200_000)
    app.run()

    assert ask.call_count == 1
    assert len(_result_dataframes(app)) == 1


def test_reset_token_tracker_preserves_response(monkeypatch):
    app, ask = _run_app(monkeypatch, _response())
    assert app.session_state["token_total"] == 7

    next(button for button in app.button if button.label == "Reset token tracker").click()
    app.run()

    assert ask.call_count == 1
    assert app.session_state["token_total"] == 0
    assert len(_result_dataframes(app)) == 1


def test_new_question_replaces_previous_response(monkeypatch):
    first = _response()
    second = _response(validation_success=False, execution_success=False)
    app, ask = _run_app(
        monkeypatch,
        first,
        ask_side_effect=[first, second],
    )

    app.text_area[0].input("new question")
    app.run()
    next(button for button in app.button if button.label == "Ask SSA").click()
    app.run()

    assert ask.call_count == 2
    assert "Question: new question" in _texts(app.caption)
    assert "SQL validation failed." in _texts(app.error)
    assert not _result_dataframes(app)


def test_failure_response_persists_across_rerun(monkeypatch):
    app, ask = _run_app(
        monkeypatch,
        _response(validation_success=False, execution_success=False),
    )
    app.run()

    assert ask.call_count == 1
    assert "SQL validation failed." in _texts(app.error)
    assert not _result_dataframes(app)


def test_execution_failure_persists_across_rerun(monkeypatch):
    app, ask = _run_app(
        monkeypatch,
        _response(execution_success=False),
    )
    app.run()

    assert ask.call_count == 1
    assert "SQL execution failed." in _texts(app.error)
    assert not _result_dataframes(app)


def test_token_usage_is_not_recorded_again_on_rerender(monkeypatch):
    app, ask = _run_app(monkeypatch, _response())
    recorded_total = app.session_state["token_total"]
    app.run()

    assert ask.call_count == 1
    assert app.session_state["token_total"] == recorded_total
    assert len(app.session_state["token_history"]) == 1


def test_empty_result_response_persists(monkeypatch):
    app, ask = _run_app(monkeypatch, _response(empty_result=True))
    app.run()

    assert ask.call_count == 1
    result_frames = _result_dataframes(app)
    assert len(result_frames) == 1
    assert result_frames[0].empty


def test_same_question_runs_again_only_after_explicit_click(monkeypatch):
    response = _response()
    app, ask = _run_app(
        monkeypatch,
        response,
        ask_side_effect=[response, response],
    )
    app.run()
    assert ask.call_count == 1

    next(button for button in app.button if button.label == "Ask SSA").click()
    app.run()

    assert ask.call_count == 2
    assert len(_result_dataframes(app)) == 1


def test_unexpected_assistant_exception_does_not_leave_running_state(monkeypatch):
    app, ask = _run_app(
        monkeypatch,
        None,
        ask_side_effect=RuntimeError("synthetic failure"),
    )

    assert ask.call_count == 1
    assert app.session_state["request_state"] == "failed"
    assert "SSA could not complete the request. Check the application logs." in _texts(app.error)

    app.run()
    assert ask.call_count == 1
    assert "SSA could not complete the request. Check the application logs." in _texts(app.error)


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
