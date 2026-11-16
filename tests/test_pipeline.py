import pytest

from ssa.contracts import QueryResult, ValidatedQuery
from ssa.pipeline import (
    PipelineExecutionError,
    PipelineOrchestrator,
    PipelineStageError,
    TIMING_KEYS,
)
from tests.fakes import (
    FakeRelationalDatabaseAdapter,
    FakeSQLGenerator,
    TimingAwareFakeRelationalDatabaseAdapter,
)


def _result() -> QueryResult:
    return QueryResult(
        columns=["value"],
        rows=[{"value": 1}],
        row_count=1,
        max_rows=100,
        truncated=False,
    )


class TimingAdapter:
    def __init__(self, calls, result=None, error=None):
        self.calls = calls
        self.result = result or _result()
        self.error = error

    def execute_read(self, query, max_rows=100, timeout=None, *, timings=None):
        self.calls.append(("execute", query, max_rows, timings))
        if self.error:
            raise self.error
        timings["validation_ms"] = 2.0
        timings["execution_ms"] = 3.0
        return self.result


class BasicAdapter:
    def __init__(self, calls):
        self.calls = calls

    def execute_read(self, query, max_rows=100, timeout=None):
        self.calls.append((query, max_rows))
        return _result()


def test_success_preserves_sql_usage_result_and_call_order():
    calls = []

    def generator(question, *, return_usage, timings):
        calls.append(("generate", question))
        timings["retrieval_ms"] = 1.0
        timings["generation_ms"] = 4.0
        return "SELECT 1", {"model": "fake", "total_tokens": 7}

    result = PipelineOrchestrator(
        generator,
        TimingAdapter(calls),
        timing_aware=True,
    ).run("question")

    assert calls[0] == ("generate", "question")
    assert calls[1][0] == "execute"
    assert result.sql == "SELECT 1"
    assert result.usage == {"model": "fake", "total_tokens": 7}
    assert result.result == _result()
    assert result.diagnostics.failure_stage is None
    assert result.diagnostics.retrieval_ms == 1.0
    assert result.diagnostics.validation_ms == 2.0
    assert result.diagnostics.execution_ms == 3.0
    assert result.diagnostics.total_ms is not None


def test_timing_aware_adapter_receives_one_shared_timing_dictionary():
    calls = []

    def generator(question, *, return_usage, timings):
        calls.append(timings)
        return "SELECT 1"

    adapter = TimingAdapter(calls)
    PipelineOrchestrator(generator, adapter, timing_aware=True).run("question")

    assert calls[0] is calls[1][3]
    assert set(calls[0]) == set(TIMING_KEYS)


def test_each_run_gets_fresh_timings():
    timings_seen = []

    def generator(question, *, return_usage, timings):
        timings_seen.append(timings)
        return "SELECT 1"

    orchestrator = PipelineOrchestrator(generator, BasicAdapter([]))
    orchestrator.run("one")
    orchestrator.run("two")

    assert timings_seen[0] is not timings_seen[1]


def test_basic_protocol_adapter_does_not_receive_unsupported_timing_argument():
    calls = []

    def generator(question, *, return_usage, timings):
        return "SELECT 1"

    result = PipelineOrchestrator(generator, BasicAdapter(calls)).run("question")

    assert result.result == _result()
    assert calls == [(ValidatedQuery("SELECT 1"), 100)]


def test_custom_max_rows_is_forwarded():
    calls = []

    def generator(question, *, return_usage, timings):
        return "SELECT 1"

    PipelineOrchestrator(
        generator,
        BasicAdapter(calls),
        max_rows=25,
    ).run("question")

    assert calls[0][1] == 25


def test_generation_failure_preserves_identity_and_stage():
    error = RuntimeError("generation failed")

    def generator(question, *, return_usage, timings):
        timings["retrieval_ms"] = 1.0
        raise error

    with pytest.raises(PipelineExecutionError) as raised:
        PipelineOrchestrator(generator, BasicAdapter([])).run("question")

    assert raised.value.cause is error
    assert raised.value.__cause__ is error
    assert raised.value.stage == "generation"
    assert raised.value.diagnostics.failure_stage == "generation"
    assert raised.value.diagnostics.retrieval_ms == 1.0
    assert raised.value.diagnostics.total_ms is not None


def test_explicit_validation_failure_preserves_stage_and_identity():
    error = ValueError("validation failed")
    generator = FakeSQLGenerator(
        stage_error=PipelineStageError("validation", error),
    )

    with pytest.raises(PipelineExecutionError) as raised:
        PipelineOrchestrator(generator, BasicAdapter([])).run("question")

    assert raised.value.cause is error
    assert raised.value.__cause__ is error
    assert raised.value.stage == "validation"
    assert raised.value.diagnostics.failure_stage == "validation"


def test_execution_failure_preserves_identity_and_stage():
    error = RuntimeError("execution failed")

    def generator(question, *, return_usage, timings):
        return "SELECT 1"

    with pytest.raises(PipelineExecutionError) as raised:
        PipelineOrchestrator(
            generator,
            TimingAdapter([], error=error),
            timing_aware=True,
        ).run("question")

    assert raised.value.cause is error
    assert raised.value.__cause__ is error
    assert raised.value.stage == "execution"
    assert raised.value.diagnostics.failure_stage == "execution"


def test_no_redundant_validation_or_second_execution():
    calls = []

    def generator(question, *, return_usage, timings):
        return "SELECT 1"

    adapter = TimingAdapter(calls)
    PipelineOrchestrator(generator, adapter, timing_aware=True).run("question")

    assert [call[0] for call in calls] == ["execute"]
    assert len(calls) == 1


def test_pipeline_has_no_provider_specific_runtime_imports():
    import ssa.pipeline as pipeline

    assert not hasattr(pipeline, "ollama")
    assert not hasattr(pipeline, "chromadb")
    assert not hasattr(pipeline, "pyodbc")


def test_reusable_fake_generator_supports_sql_only_and_usage_results():
    sql_only = FakeSQLGenerator(sql="SELECT 1")
    with_usage = FakeSQLGenerator(
        sql="SELECT 2",
        usage={"model": "fake", "total_tokens": 3},
    )

    assert sql_only("question", return_usage=True) == "SELECT 1"
    assert with_usage("question", return_usage=True) == (
        "SELECT 2",
        {"model": "fake", "total_tokens": 3},
    )
    assert sql_only.calls[0]["question"] == "question"
    assert with_usage.calls[0]["return_usage"] is True


def test_reusable_database_fakes_support_standard_and_timing_aware_calls():
    standard = FakeRelationalDatabaseAdapter()
    timing_aware = TimingAwareFakeRelationalDatabaseAdapter()
    timings = {"execution_ms": None}

    standard_result = standard.execute_read(ValidatedQuery("SELECT 1"))
    timing_result = timing_aware.execute_read(
        ValidatedQuery("SELECT 1"),
        max_rows=10,
        timings=timings,
    )

    assert standard_result.max_rows == 100
    assert timing_result.max_rows == 10
    assert standard.execution_calls[0][1] == 100
    assert timing_aware.execution_calls[0][1] == 10
    assert timing_aware.timing_calls == [timings]


def test_reusable_fake_instances_have_independent_logs_and_state():
    first = FakeSQLGenerator()
    second = FakeSQLGenerator()
    first("one")

    assert first.calls != second.calls
    assert second.calls == []
