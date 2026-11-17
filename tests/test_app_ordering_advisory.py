from copy import deepcopy

import pytest

from ssa.app_ordering_advisory import (
    ORDERING_RISK_MESSAGE,
    ordering_advisory_message,
)


def test_valid_risk_returns_exact_fixed_message():
    diagnostics = {
        "ordering": {
            "status": "risk",
            "reason": "primary_key_missing_from_root_order",
            "table": "Production.Product",
            "missing_key_columns": ["ProductID"],
        }
    }

    assert ordering_advisory_message(diagnostics) == ORDERING_RISK_MESSAGE
    assert ordering_advisory_message(diagnostics) == (
        "Ordering note: This TOP query may select different rows when sort values "
        "are tied. This is an advisory about result stability, not a SQL safety "
        "or execution error."
    )


@pytest.mark.parametrize(
    "diagnostics",
    [
        {"ordering": {"status": "sufficient"}},
        {"ordering": {"status": "unsupported"}},
        {},
        None,
        [],
        {"ordering": None},
        {"ordering": []},
        {"ordering": {}},
        {"ordering": {"status": "Risk"}},
        {"ordering": {"status": "unknown"}},
    ],
)
def test_only_exact_risk_status_returns_a_message(diagnostics):
    assert ordering_advisory_message(diagnostics) is None


@pytest.mark.parametrize(
    "diagnostics",
    [
        {"ordering": {"status": "risk", "reason": object()}},
        {"ordering": {"status": "risk", "missing_key_columns": object()}},
        {"ordering": {"status": "risk", "table": object()}},
        {"ordering": {"status": "risk", "reason": None}},
    ],
)
def test_malformed_optional_fields_do_not_crash_or_change_message(diagnostics):
    assert ordering_advisory_message(diagnostics) == ORDERING_RISK_MESSAGE


def test_risk_without_optional_fields_returns_fixed_message():
    assert ordering_advisory_message({"ordering": {"status": "risk"}}) == (
        ORDERING_RISK_MESSAGE
    )


def test_diagnostics_are_not_mutated_and_sensitive_details_are_not_exposed():
    diagnostics = {
        "ordering": {
            "status": "risk",
            "reason": "secret_reason",
            "table": "Secret.Table",
            "primary_key_columns": ["SecretKey"],
            "missing_key_columns": ["SecretKey"],
        }
    }
    original = deepcopy(diagnostics)

    message = ordering_advisory_message(diagnostics)

    assert diagnostics == original
    assert "secret_reason" not in message
    assert "Secret.Table" not in message
    assert "SecretKey" not in message
