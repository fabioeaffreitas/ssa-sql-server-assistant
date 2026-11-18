from dataclasses import FrozenInstanceError

import pytest

from ssa.sql_ordering_safeguard import OrderingRisk
from ssa.sql_preview import SQLPreviewService
from ssa.sql_validator import SQLValidationError


RISK_SQL = "SELECT TOP 10 ItemKey, ItemName, Price FROM Catalog.Item ORDER BY Price DESC"
SUFFICIENT_SQL = (
    "SELECT TOP 10 ItemKey, ItemName, Price FROM Catalog.Item "
    "ORDER BY Price DESC, ItemKey ASC"
)
KEYS = {"Catalog.Item": ("ItemKey",)}


class FakeGenerator:
    def __init__(self, generated=RISK_SQL, error=None, timings=None):
        self.generated = generated
        self.error = error
        self.timings = timings or {
            "retrieval_ms": 1.5,
            "generation_ms": 2.5,
            "validation_ms": 0.5,
        }
        self.calls = []

    def __call__(self, question, *, return_usage, timings):
        self.calls.append((question, return_usage, timings))
        timings.update(self.timings)
        if self.error is not None:
            raise self.error
        return self.generated, {
            "model": "fake-model",
            "total_tokens": 9,
        }


class FakeClock:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return float(self.calls * 10)


def service(generator, primary_keys=KEYS, inspector=None, clock=None):
    kwargs = {}
    if inspector is not None:
        kwargs["ordering_inspector"] = inspector
    if clock is not None:
        kwargs["clock"] = clock
    return SQLPreviewService(generator, primary_keys, **kwargs)


def test_valid_preview_returns_clean_sql_without_rows_or_execution():
    generator = FakeGenerator()
    result = service(generator).preview("  show top items  ")

    assert result.status == "preview_ready"
    assert result.question == "show top items"
    assert result.sql == RISK_SQL
    assert result.validation_passed is True
    assert result.execution_attempted is False
    assert result.usage == {"model": "fake-model", "total_tokens": 9}
    assert result.retrieval_ms == 1.5
    assert result.generation_ms == 2.5
    assert result.validation_ms == 0.5
    assert result.total_ms is not None
    assert not hasattr(result, "rows")
    assert not hasattr(result, "result")
    assert len(generator.calls) == 1


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_question_is_rejected_without_generation(question):
    generator = FakeGenerator()
    result = service(generator).preview(question)

    assert result.status == "invalid_request"
    assert result.error == "Question must not be empty."
    assert generator.calls == []
    assert result.ordering_advisory is None
    assert result.execution_attempted is False


def test_missing_metadata_is_unsupported_not_a_preview_failure():
    result = service(FakeGenerator(), primary_keys=None).preview("items")

    assert result.status == "preview_ready"
    assert result.ordering_advisory.status == "unsupported"
    assert result.ordering_advisory.reason == "unsupported_primary_key_metadata"
    assert result.ordering_advisory.primary_key_columns == ()


def test_risk_ordering_is_reported_without_rewriting_sql():
    result = service(FakeGenerator()).preview("items")

    assert result.sql == RISK_SQL
    assert result.ordering_advisory == OrderingRisk(
        "risk",
        "primary_key_missing_from_root_order",
        "Catalog.Item",
        ("ItemKey",),
        ("ItemKey",),
    )


def test_sufficient_ordering_is_reported_without_rewriting_sql():
    result = service(FakeGenerator(SUFFICIENT_SQL)).preview("items")

    assert result.status == "preview_ready"
    assert result.sql == SUFFICIENT_SQL
    assert result.ordering_advisory.status == "sufficient"
    assert result.ordering_advisory.reason == "primary_key_in_root_order"


def test_unsupported_ordering_is_not_preview_failure():
    joined = (
        "SELECT TOP 10 i.ItemKey FROM Catalog.Item i "
        "JOIN Catalog.Other o ON o.ItemKey = i.ItemKey ORDER BY i.Price DESC"
    )
    result = service(FakeGenerator(joined)).preview("items")

    assert result.status == "preview_ready"
    assert result.ordering_advisory.status == "unsupported"
    assert result.ordering_advisory.reason == "unsupported_join"


def test_sql_validation_failure_is_classified_without_execution():
    generator = FakeGenerator(error=SQLValidationError("invalid SQL"))
    result = service(generator).preview("items")

    assert result.status == "validation_failed"
    assert result.validation_passed is False
    assert result.error == "Generated SQL failed local validation."
    assert result.sql is None
    assert result.execution_attempted is False


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("retriever failed"),
        type("OllamaGenerationTimeoutError", (RuntimeError,), {})(),
    ],
)
def test_generation_and_unexpected_failures_are_classified(error):
    result = service(FakeGenerator(error=error)).preview("items")

    expected = (
        "generation_failed"
        if type(error).__name__.startswith("Ollama")
        else "preview_failed"
    )
    assert result.status == expected
    assert result.sql is None
    assert result.ordering_advisory is None
    assert result.execution_attempted is False


def test_unexpected_ordering_inspector_failure_is_nonblocking():
    def failing_inspector(sql, primary_keys):
        raise RuntimeError("synthetic advisory failure")

    result = service(FakeGenerator(), inspector=failing_inspector).preview("items")

    assert result.status == "preview_ready"
    assert result.sql == RISK_SQL
    assert result.ordering_advisory is None
    assert result.ordering_error == "ordering_inspection_unavailable"
    assert result.execution_attempted is False


def test_usage_is_immutable_and_result_is_frozen():
    result = service(FakeGenerator()).preview("items")

    with pytest.raises(TypeError):
        result.usage["total_tokens"] = 10
    with pytest.raises(FrozenInstanceError):
        result.sql = "SELECT 1"


def test_preview_has_no_repair_or_approval_state():
    result = service(FakeGenerator()).preview("items")

    assert not hasattr(result, "proposal")
    assert not hasattr(result, "candidate")
    assert not hasattr(result, "approval")


def test_clock_is_used_for_deterministic_total_timing():
    clock = FakeClock()
    result = service(FakeGenerator(), clock=clock).preview("items")

    assert clock.calls == 2
    assert result.total_ms == 10_000.0


def test_service_constructor_has_no_database_execution_dependency():
    import inspect

    signature = inspect.signature(SQLPreviewService)
    assert "database" not in signature.parameters
    assert "adapter" not in signature.parameters

