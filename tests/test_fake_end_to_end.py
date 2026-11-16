import pytest

from ssa import assistant
from ssa import ollama_provider, schema_retrieval, sql_executor
from ssa.contracts import GenerationResult, QueryResult, SchemaDocument, UsageMetadata
import ssa.sql_generator as generator
from tests.fakes import (
    FakeLLMProvider,
    FakeSchemaRetriever,
    TimingAwareFakeRelationalDatabaseAdapter,
)


def _install_fake_generator(monkeypatch, retriever, provider, factory=None):
    if factory is None:
        factory = lambda prompt_builder, timeout: provider

    def fake_generate_sql(question, *, return_usage=False, timings=None):
        return generator._generate_sql_with_dependencies(
            question,
            5,
            return_usage,
            timings,
            retriever=retriever,
            provider_factory=factory,
        )

    monkeypatch.setattr(assistant, "generate_sql", fake_generate_sql)
    return fake_generate_sql


def _success_provider(sql="SELECT 1 AS value"):
    return FakeLLMProvider(GenerationResult(
        query=sql,
        usage=UsageMetadata(
            provider="fake",
            model="fake-model",
            prompt_tokens=10,
            completion_tokens=4,
            total_tokens=999,
        ),
    ))


def test_full_fake_request_uses_real_generation_pipeline_orchestrator_and_assistant(
    monkeypatch,
):
    retriever = FakeSchemaRetriever([
        SchemaDocument("table:Customer", "CUSTOMER SCHEMA"),
        SchemaDocument("table:Order", "ORDER SCHEMA"),
    ])
    provider = _success_provider()
    database = TimingAwareFakeRelationalDatabaseAdapter(
        result=QueryResult(
            columns=["value"],
            rows=[{"value": 1}],
            row_count=1,
            max_rows=100,
            truncated=False,
        ),
        timing_updates={"validation_ms": 1.0, "execution_ms": 2.0},
    )
    _install_fake_generator(monkeypatch, retriever, provider)
    monkeypatch.setattr(assistant, "sql_server_adapter", database)

    response = assistant.ask("show one value")

    assert response["success"] is True
    assert response["sql"] == "SELECT 1 AS value"
    assert retriever.calls == [("show one value", 5)]
    assert provider.calls[0].question == "show one value"
    assert provider.calls[0].schema_context == "CUSTOMER SCHEMA\n\nORDER SCHEMA"
    assert provider.calls[0].dialect == "tsql"
    assert database.execution_calls == [(
        assistant.ValidatedQuery("SELECT 1 AS value"),
        100,
        None,
    )]
    assert database.validation_calls == []
    assert response["result"] == {
        "columns": ["value"],
        "rows": [{"value": 1}],
        "row_count": 1,
        "max_rows": 100,
        "truncated": False,
    }
    assert response["generation"]["usage"] == {
        "model": "fake-model",
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
    }
    assert set(response["diagnostics"]["timings_ms"]) == {
        "retrieval_ms",
        "generation_ms",
        "validation_ms",
        "execution_ms",
        "total_ms",
    }
    assert all(
        isinstance(value, (int, float))
        for value in response["diagnostics"]["timings_ms"].values()
    )
    assert response["diagnostics"]["failure_stage"] is None


def test_fake_llm_cleanup_and_real_sqlglot_validation(monkeypatch):
    provider = _success_provider("```sql\nSELECT TOP 5 1 AS value\n```")
    retriever = FakeSchemaRetriever()
    database = TimingAwareFakeRelationalDatabaseAdapter()
    _install_fake_generator(monkeypatch, retriever, provider)
    monkeypatch.setattr(assistant, "sql_server_adapter", database)

    response = assistant.ask("clean query")

    assert response["success"] is True
    assert response["sql"] == "SELECT TOP 5 1 AS value"
    assert len(database.execution_calls) == 1


def test_invalid_or_unsafe_generated_sql_never_reaches_database(monkeypatch):
    provider = _success_provider("DELETE FROM dbo.Users")
    retriever = FakeSchemaRetriever()
    database = TimingAwareFakeRelationalDatabaseAdapter()
    _install_fake_generator(monkeypatch, retriever, provider)
    monkeypatch.setattr(assistant, "sql_server_adapter", database)

    response = assistant.ask("unsafe query")

    assert response["success"] is False
    assert response["error_stage"] == "validation"
    assert response["diagnostics"]["failure_stage"] == "validation"
    assert response["validation"]["success"] is False
    assert database.execution_calls == []


def test_retrieval_failure_stops_before_provider_and_database(monkeypatch):
    error = RuntimeError("retrieval unavailable")
    retriever = FakeSchemaRetriever(error=error)
    provider = _success_provider()
    database = TimingAwareFakeRelationalDatabaseAdapter()
    _install_fake_generator(monkeypatch, retriever, provider)
    monkeypatch.setattr(assistant, "sql_server_adapter", database)

    response = assistant.ask("retrieval failure")

    assert response["success"] is False
    assert response["error_stage"] == "generation"
    assert response["diagnostics"]["failure_stage"] == "generation"
    assert response["diagnostics"]["timings_ms"]["retrieval_ms"] is not None
    assert provider.calls == []
    assert database.execution_calls == []
    assert "PipelineExecutionError" not in str(response)


@pytest.mark.parametrize(
    ("failure", "expected_type", "expected_message"),
    [
        (
            ollama_provider.OllamaConnectionError("raw connection details"),
            "generation",
            "Ollama is unavailable.",
        ),
        (
            ollama_provider.OllamaGenerationTimeoutError("raw timeout details"),
            "generation",
            "Ollama generation timed out.",
        ),
    ],
)
def test_provider_failures_translate_through_real_generation_and_assistant(
    monkeypatch,
    failure,
    expected_type,
    expected_message,
):
    retriever = FakeSchemaRetriever()
    provider = FakeLLMProvider(error=failure)
    database = TimingAwareFakeRelationalDatabaseAdapter()
    _install_fake_generator(monkeypatch, retriever, provider)
    monkeypatch.setattr(assistant, "sql_server_adapter", database)

    response = assistant.ask("provider failure")

    assert response["success"] is False
    assert response["error_stage"] == expected_type
    assert response["diagnostics"]["failure_stage"] == expected_type
    assert expected_message in response["error"]
    assert database.execution_calls == []
    assert "PipelineExecutionError" not in str(response)


def test_malformed_injected_provider_result_uses_existing_generation_error(
    monkeypatch,
):
    retriever = FakeSchemaRetriever()
    provider = FakeLLMProvider(GenerationResult(query=""))
    database = TimingAwareFakeRelationalDatabaseAdapter()
    _install_fake_generator(monkeypatch, retriever, provider)
    monkeypatch.setattr(assistant, "sql_server_adapter", database)

    response = assistant.ask("malformed provider result")

    assert response["success"] is False
    assert response["error_stage"] == "generation"
    assert response["diagnostics"]["failure_stage"] == "generation"
    assert "Ollama could not generate SQL." in response["error"]
    assert database.execution_calls == []


def test_provider_factory_failure_is_translated_without_database_execution(
    monkeypatch,
):
    retriever = FakeSchemaRetriever()
    provider = _success_provider()
    database = TimingAwareFakeRelationalDatabaseAdapter()

    def failing_factory(prompt_builder, timeout):
        raise ConnectionError("raw provider endpoint")

    _install_fake_generator(
        monkeypatch,
        retriever,
        provider,
        factory=failing_factory,
    )
    monkeypatch.setattr(assistant, "sql_server_adapter", database)

    response = assistant.ask("factory failure")

    assert response["success"] is False
    assert response["error_stage"] == "generation"
    assert response["diagnostics"]["failure_stage"] == "generation"
    assert "Ollama is unavailable." in response["error"]
    assert database.execution_calls == []


@pytest.mark.parametrize(
    ("failure", "error_type", "message"),
    [
        (
            assistant.DatabaseConnectionError("connection unavailable"),
            "connection",
            "connection unavailable",
        ),
        (
            assistant.SQLQueryTimeoutError("timeout"),
            "timeout",
            "SQL query timed out.",
        ),
        (
            assistant.SQLQueryCancelledError("cancelled"),
            "cancelled",
            "SQL query was cancelled.",
        ),
        (
            assistant.SQLExecutionError("execution failure"),
            "execution",
            "SQL execution failed.",
        ),
    ],
)
def test_downstream_database_failures_preserve_generated_sql_and_usage(
    monkeypatch,
    failure,
    error_type,
    message,
):
    retriever = FakeSchemaRetriever()
    provider = _success_provider()
    database = TimingAwareFakeRelationalDatabaseAdapter(execution_error=failure)
    _install_fake_generator(monkeypatch, retriever, provider)
    monkeypatch.setattr(assistant, "sql_server_adapter", database)

    response = assistant.ask("database failure")

    assert response["success"] is False
    assert response["error_stage"] == (
        "connection" if error_type == "connection" else "execution"
    )
    assert response["diagnostics"]["failure_stage"] == response["error_stage"]
    assert response["sql"] == "SELECT 1 AS value"
    assert response["generation"]["usage"] == {
        "model": "fake-model",
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "total_tokens": 14,
    }
    assert response["result"] is None
    assert message in response["error"]


def test_fake_end_to_end_timing_dictionaries_are_isolated(monkeypatch):
    retriever = FakeSchemaRetriever()
    provider = _success_provider()
    database = TimingAwareFakeRelationalDatabaseAdapter(
        timing_updates={"validation_ms": 1.0, "execution_ms": 2.0},
    )
    timing_refs = []
    original_generator = _install_fake_generator(
        monkeypatch,
        retriever,
        provider,
    )

    def recording_generator(question, *, return_usage=False, timings=None):
        timing_refs.append(timings)
        return original_generator(
            question,
            return_usage=return_usage,
            timings=timings,
        )

    monkeypatch.setattr(assistant, "generate_sql", recording_generator)
    monkeypatch.setattr(assistant, "sql_server_adapter", database)

    first = assistant.ask("first request")
    second = assistant.ask("second request")

    assert timing_refs[0] is database.timing_calls[0]
    assert timing_refs[1] is database.timing_calls[1]
    assert timing_refs[0] is not timing_refs[1]
    for response in (first, second):
        timings = response["diagnostics"]["timings_ms"]
        assert all(isinstance(value, (int, float)) for value in timings.values())
    assert database.execution_calls == [
        (assistant.ValidatedQuery("SELECT 1 AS value"), 100, None),
        (assistant.ValidatedQuery("SELECT 1 AS value"), 100, None),
    ]


def test_fake_end_to_end_guards_external_entry_points(monkeypatch):
    def external_call(*args, **kwargs):
        raise AssertionError("external dependency was called")

    monkeypatch.setattr(ollama_provider.ollama, "Client", external_call)
    monkeypatch.setattr(schema_retrieval, "search_schema", external_call)
    monkeypatch.setattr(sql_executor, "get_connection", external_call)

    retriever = FakeSchemaRetriever()
    provider = _success_provider()
    database = TimingAwareFakeRelationalDatabaseAdapter()
    _install_fake_generator(monkeypatch, retriever, provider)
    monkeypatch.setattr(assistant, "sql_server_adapter", database)

    response = assistant.ask("isolated fake request")

    assert response["success"] is True
    assert database.execution_calls
    assert provider.calls


def test_pipeline_execution_error_is_not_publicly_exposed(monkeypatch):
    retriever = FakeSchemaRetriever()
    provider = FakeLLMProvider(error=generator.ProviderGenerationError("raw"))
    database = TimingAwareFakeRelationalDatabaseAdapter()
    _install_fake_generator(monkeypatch, retriever, provider)
    monkeypatch.setattr(assistant, "sql_server_adapter", database)

    response = assistant.ask("wrapped failure")

    assert all(
        "PipelineExecutionError" not in str(value)
        for value in response.values()
        if isinstance(value, str)
    )
