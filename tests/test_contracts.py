from dataclasses import FrozenInstanceError

import pytest

from ssa.contracts import (
    GenerationRequest,
    GenerationResult,
    LLMProvider,
    PipelineDiagnostics,
    QueryResult,
    RelationalDatabaseAdapter,
    SchemaDocument,
    SchemaRetriever,
    UsageMetadata,
    ValidatedQuery,
)
from tests.fakes import (
    FakeLLMProvider,
    FakeRelationalDatabaseAdapter,
    FakeSchemaRetriever,
)


def test_usage_metadata_preserves_missing_and_zero_values():
    missing = UsageMetadata()
    reported_zero = UsageMetadata(
        provider="test",
        model="test-model",
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
    )

    assert missing.provider is None
    assert missing.model is None
    assert missing.prompt_tokens is None
    assert missing.completion_tokens is None
    assert missing.total_tokens is None
    assert reported_zero.prompt_tokens == 0
    assert reported_zero.completion_tokens == 0
    assert reported_zero.total_tokens == 0


def test_generation_request_required_and_default_fields():
    request = GenerationRequest("question", "schema")

    assert request.question == "question"
    assert request.schema_context == "schema"
    assert request.dialect == "tsql"
    assert request.model is None

    configured = GenerationRequest(
        question="question",
        schema_context="schema",
        dialect="postgres",
        model="model",
    )
    assert configured.dialect == "postgres"
    assert configured.model == "model"


def test_generation_result_contains_query_and_optional_usage():
    usage = UsageMetadata(model="model", total_tokens=4)
    result = GenerationResult(
        query="SELECT 1",
        usage=usage,
        provider="test",
        model="model",
    )

    assert result.query == "SELECT 1"
    assert result.usage is usage
    assert result.provider == "test"
    assert result.model == "model"
    assert GenerationResult("SELECT 1").usage is None


def test_query_result_matches_bounded_executor_shape():
    result = QueryResult(
        columns=["HealthCheck"],
        rows=[{"HealthCheck": 1}],
        row_count=1,
        max_rows=100,
        truncated=False,
    )

    assert result.columns == ["HealthCheck"]
    assert result.rows == [{"HealthCheck": 1}]
    assert result.row_count == 1
    assert result.max_rows == 100
    assert result.truncated is False


def test_query_result_mutable_defaults_are_independent():
    first = QueryResult()
    second = QueryResult()

    first.columns.append("id")
    first.rows.append({"id": 1})

    assert second.columns == []
    assert second.rows == []


def test_pipeline_diagnostics_support_partial_and_complete_timings():
    partial = PipelineDiagnostics(retrieval_ms=12.5, failure_stage="generation")
    complete = PipelineDiagnostics(
        retrieval_ms=1.0,
        generation_ms=2.0,
        validation_ms=3.0,
        execution_ms=4.0,
        total_ms=10.0,
    )

    assert partial.retrieval_ms == 12.5
    assert partial.generation_ms is None
    assert partial.validation_ms is None
    assert partial.execution_ms is None
    assert partial.total_ms is None
    assert partial.failure_stage == "generation"
    assert complete.total_ms == 10.0
    assert complete.failure_stage is None


def test_contract_dataclasses_are_constructible_and_immutable_where_metadata_is_value_like():
    assert SchemaDocument("table", "schema").distance is None
    assert ValidatedQuery("SELECT 1").sql == "SELECT 1"

    with pytest.raises(FrozenInstanceError):
        GenerationRequest("question", "schema").question = "changed"


def test_fake_implementations_are_compatible_with_protocols():
    provider: LLMProvider = FakeLLMProvider(
        GenerationResult(query="SELECT 'question'")
    )
    retriever: SchemaRetriever = FakeSchemaRetriever([
        SchemaDocument(id="table", document="schema")
    ])
    database: RelationalDatabaseAdapter = FakeRelationalDatabaseAdapter()

    generation = provider.generate(GenerationRequest("question", "schema"))
    documents = retriever.retrieve("schema", limit=1)
    validated = database.validate_read_query(generation.query)
    result = database.execute_read(validated, max_rows=1)

    assert generation.query == "SELECT 'question'"
    assert len(documents) == 1
    assert validated.sql == generation.query
    assert result.rows == [{"value": generation.query}]


def test_fake_provider_records_requests_and_injects_failures():
    error = RuntimeError("provider failure")
    provider = FakeLLMProvider(error=error)
    request = GenerationRequest("question", "schema")

    with pytest.raises(RuntimeError) as raised:
        provider.generate(request)

    assert raised.value is error
    assert provider.calls == [request]


def test_fake_retriever_records_arguments_and_returns_defensive_copy():
    retriever = FakeSchemaRetriever([
        SchemaDocument(id="table", document="schema")
    ])

    returned = retriever.retrieve("customer question", limit=1)
    returned.clear()

    assert retriever.calls == [("customer question", 1)]
    assert len(retriever.documents) == 1


def test_fake_database_records_validation_execution_and_limits():
    database = FakeRelationalDatabaseAdapter()

    validated = database.validate_read_query("SELECT 1", timeout=7)
    result = database.execute_read(validated, max_rows=25, timeout=8)

    assert database.validation_calls == [("SELECT 1", 7)]
    assert database.execution_calls == [(ValidatedQuery("SELECT 1"), 25, 8)]
    assert result.max_rows == 25
