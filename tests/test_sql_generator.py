import pytest
import ollama

from ssa.contracts import GenerationResult, SchemaDocument, UsageMetadata
from tests.fakes import FakeLLMProvider, FakeSchemaRetriever
import ssa.sql_generator as generator


def _stub_retrieval_and_validation(monkeypatch):
    monkeypatch.setattr(
        generator.schema_retriever,
        "retrieve",
        lambda question, limit: [],
    )
    monkeypatch.setattr(generator, "validate_sql", lambda sql: sql)


def _injected_generation(
    retriever,
    provider,
    *,
    return_usage=False,
    timings=None,
    provider_factory=None,
):
    if provider_factory is None:
        provider_factory = lambda prompt_builder, timeout: provider
    return generator._generate_sql_with_dependencies(
        "test question",
        2,
        return_usage,
        timings,
        retriever=retriever,
        provider_factory=provider_factory,
    )


def test_injected_retriever_and_provider_receive_contract_values(monkeypatch):
    retriever = FakeSchemaRetriever([
        SchemaDocument("table:First", "FIRST DOCUMENT"),
        SchemaDocument("table:Second", "SECOND DOCUMENT"),
    ])
    provider = FakeLLMProvider(GenerationResult("SELECT 1"))
    monkeypatch.setattr(generator, "validate_sql", lambda sql: sql)

    result = _injected_generation(retriever, provider)

    assert result == "SELECT 1"
    assert retriever.calls == [("test question", 2)]
    assert provider.calls[0].question == "test question"
    assert provider.calls[0].schema_context == (
        "FIRST DOCUMENT\n\nSECOND DOCUMENT"
    )
    assert provider.calls[0].dialect == "tsql"
    assert provider.calls[0].model == generator.OLLAMA_MODEL


def test_injected_provider_supports_usage_and_missing_counts(monkeypatch):
    monkeypatch.setattr(generator, "validate_sql", lambda sql: sql)
    provider = FakeLLMProvider(GenerationResult(
        "SELECT 1",
        UsageMetadata(model="fake-model", prompt_tokens=None, completion_tokens=None),
    ))

    result = _injected_generation(
        FakeSchemaRetriever(),
        provider,
        return_usage=True,
    )

    assert result == (
        "SELECT 1",
        {
            "model": "fake-model",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    )


def test_injected_provider_usage_recomputes_total(monkeypatch):
    monkeypatch.setattr(generator, "validate_sql", lambda sql: sql)
    provider = FakeLLMProvider(GenerationResult(
        "SELECT 1",
        UsageMetadata(
            model="fake-model",
            prompt_tokens=100,
            completion_tokens=25,
            total_tokens=999,
        ),
    ))

    _, usage = _injected_generation(
        FakeSchemaRetriever(),
        provider,
        return_usage=True,
    )

    assert usage == {
        "model": "fake-model",
        "prompt_tokens": 100,
        "completion_tokens": 25,
        "total_tokens": 125,
    }


def test_injected_prompt_matches_fixed_sql_server_prompt(monkeypatch):
    captured = {}
    provider = FakeLLMProvider(GenerationResult("SELECT 1"))

    def provider_factory(prompt_builder, timeout):
        captured["prompt_builder"] = prompt_builder
        captured["timeout"] = timeout
        return provider

    monkeypatch.setattr(generator, "validate_sql", lambda sql: sql)
    retriever = FakeSchemaRetriever([
        SchemaDocument("table:First", "FIRST DOCUMENT"),
        SchemaDocument("table:Second", "SECOND DOCUMENT"),
    ])
    _injected_generation(
        retriever,
        provider,
        provider_factory=provider_factory,
    )

    actual = captured["prompt_builder"](provider.calls[0])
    expected = """
You are SSA, a SQL Server AI Assistant.

Generate a SQL Server 2025 T-SQL query answering the user's question.

Rules:
- Use ONLY the tables, views, and columns provided in the schema context.
- Do not invent table names or column names.
- Prefer current-state tables over history tables when the question asks
  for current values.
- For total sales including tax and freight, use SalesOrderHeader.TotalDue.
- For sales before tax and freight, use SalesOrderHeader.SubTotal.
- Do not use SubTotal when the question explicitly asks for total sales including tax and freight.
- When aggregating sales by territory, use Sales.SalesTerritory joined directly to Sales.SalesOrderHeader on TerritoryID. Use SalesOrderHeader.TotalDue for total sales. Do not join SalesOrderDetail unless the question explicitly requires order-line information.
- When aggregating sales by customer, group by CustomerID.
- When the user explicitly requests customer names, resolve customer names as follows:
- Join Sales.Customer C ON C.CustomerID = SalesOrderHeader.CustomerID.
- CUSTOMER AGGREGATION RULES:
- For customer sales aggregation WITHOUT customer names:
  use Sales.SalesOrderHeader.CustomerID directly.
  Select CustomerID and aggregate TotalDue or SubTotal as requested.
  GROUP BY CustomerID.
  Do NOT join Sales.Customer, Person.Person, or Sales.Store.
- Only when customer names are explicitly requested:
  join Sales.Customer C ON C.CustomerID = SOH.CustomerID.
  LEFT JOIN Person.Person P ON C.PersonID = P.BusinessEntityID.
  LEFT JOIN Sales.Store S ON C.StoreID = S.BusinessEntityID.
  Use COALESCE(P.FirstName + N' ' + P.LastName, S.Name)
  as CustomerName.
  Include C.CustomerID and the CustomerName expression
  in GROUP BY when aggregating sales.
- Never generate customer-name joins unless the user explicitly
  requests customer names or other customer attributes.
- Do not include customer names unless explicitly requested.
- Use history tables only when the question explicitly asks about
  historical changes, previous values, or trends over time.
- Avoid unnecessary JOINs.
- Use the table that directly contains the requested columns whenever possible.
- Do not JOIN Sales.Customer when CustomerID is already available
  in Sales.SalesOrderHeader and no customer attributes are requested.
- When aggregating sales by customer, use SalesOrderHeader.CustomerID
  directly unless customer attributes are explicitly requested.
- Do not JOIN tables unless the requested information requires it.
- EXAMPLE: For "Show the top 10 customers by total sales",
  generate:
  SELECT TOP 10 SOH.CustomerID, SUM(SOH.TotalDue) AS TotalSales
  FROM Sales.SalesOrderHeader SOH
  GROUP BY SOH.CustomerID
  ORDER BY TotalSales DESC;
  Do not add customer-name columns or joins.
- Generate SELECT queries only.
- In SQL Server, TOP must appear immediately after SELECT, e.g. SELECT TOP 10 ...
- Never place TOP at the end of a query.
- Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or TRUNCATE.
- Use SQL Server T-SQL syntax.
- Use TOP instead of LIMIT.
- Return only the SQL query.
- Do not include Markdown code fences or explanations.
- Always use parentheses for SQL aggregate functions: SUM(column), COUNT(*), AVG(column), MIN(column), and MAX(column). Never write an aggregate function without parentheses.
- For product-level sales aggregation, use Sales.SalesOrderDetail.LineTotal grouped by ProductID and product name. Join Production.Product on ProductID when product names are requested. Do not use SalesOrderHeader.TotalDue for product-level sales.
- EXAMPLE: For "Show the top 5 products by total sales",
  generate:
  SELECT TOP 5 p.ProductID, p.Name, SUM(sod.LineTotal) AS TotalSales
  FROM Sales.SalesOrderDetail sod
  JOIN Production.Product p ON sod.ProductID = p.ProductID
  GROUP BY p.ProductID, p.Name
  ORDER BY TotalSales DESC;
- Use exactly the aliases defined in FROM and JOIN.
- Never reference an alias that has not been defined.


DATABASE SCHEMA:
FIRST DOCUMENT

SECOND DOCUMENT

USER QUESTION:
test question

T-SQL QUERY:
"""
    assert actual == expected
    assert captured["timeout"] == generator.DEFAULT_OLLAMA_REQUEST_TIMEOUT


def test_injected_generation_removes_fences_repairs_top_and_validates(monkeypatch):
    provider = FakeLLMProvider(GenerationResult(
        "```sql\nSELECT value FROM table_name TOP 5\n```"
    ))
    validated = []
    monkeypatch.setattr(generator, "validate_sql", lambda sql: validated.append(sql) or sql)

    result = _injected_generation(FakeSchemaRetriever(), provider)

    assert result == "SELECT TOP 5 value FROM table_name"
    assert validated == ["SELECT TOP 5 value FROM table_name"]


def test_injected_retrieval_failure_propagates_unchanged():
    error = RuntimeError("retrieval unavailable")
    provider = FakeLLMProvider(GenerationResult("SELECT 1"))

    with pytest.raises(RuntimeError) as raised:
        _injected_generation(FakeSchemaRetriever(error=error), provider)

    assert raised.value is error
    assert provider.calls == []


def test_injected_provider_failure_keeps_legacy_translation():
    error = generator.ProviderConnectionError("raw endpoint details")
    provider = FakeLLMProvider(error=error)

    with pytest.raises(generator.OllamaConnectionError) as raised:
        _injected_generation(FakeSchemaRetriever(), provider)

    assert "raw endpoint details" in str(raised.value)


def test_injected_generation_preserves_stage_timings(monkeypatch):
    clock = iter([10.0, 10.125, 20.0, 20.25, 30.0, 30.05])
    monkeypatch.setattr(generator.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(generator, "validate_sql", lambda sql: sql)
    timings = {}

    result = _injected_generation(
        FakeSchemaRetriever(),
        FakeLLMProvider(GenerationResult("SELECT 1")),
        timings=timings,
    )

    assert result == "SELECT 1"
    assert timings["retrieval_ms"] == pytest.approx(125.0)
    assert timings["generation_ms"] == pytest.approx(250.0)
    assert timings["validation_ms"] == pytest.approx(50.0)


def test_default_provider_factory_preserves_ollama_configuration(monkeypatch):
    captured = {}

    class FakeProvider:
        def __init__(self, prompt_builder, **kwargs):
            captured["prompt_builder"] = prompt_builder
            captured.update(kwargs)

    monkeypatch.setattr(generator, "OllamaProvider", FakeProvider)

    provider = generator._default_provider_factory(lambda request: "PROMPT", 17)

    assert isinstance(provider, FakeProvider)
    assert captured["prompt_builder"](None) == "PROMPT"
    assert captured["model"] == generator.OLLAMA_MODEL
    assert captured["base_url"] == generator.OLLAMA_BASE_URL
    assert captured["request_timeout"] == 17


def test_generate_sql_uses_retriever_and_preserves_schema_context(monkeypatch):
    calls = []
    captured = {}

    class FakeRetriever:
        def retrieve(self, question, limit):
            calls.append((question, limit))
            return [
                SchemaDocument("table:First", "FIRST DOCUMENT"),
                SchemaDocument("table:Second", "SECOND DOCUMENT"),
            ]

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            captured.update(kwargs)
            return {"response": "SELECT 1"}

    monkeypatch.setattr(generator, "schema_retriever", FakeRetriever())
    monkeypatch.setattr(generator, "validate_sql", lambda sql: sql)
    monkeypatch.setattr("ssa.ollama_provider.ollama.Client", FakeClient)

    assert generator.generate_sql("test question", n_schema_results=2) == "SELECT 1"

    assert calls == [("test question", 2)]
    prompt_context = captured["prompt"].split(
        "DATABASE SCHEMA:\n",
        1,
    )[1].split("\n\nUSER QUESTION:", 1)[0]
    assert prompt_context == "FIRST DOCUMENT\n\nSECOND DOCUMENT"


def test_generate_sql_preserves_empty_retrieval_context(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            captured.update(kwargs)
            return {"response": "SELECT 1"}

    _stub_retrieval_and_validation(monkeypatch)
    monkeypatch.setattr("ssa.ollama_provider.ollama.Client", FakeClient)

    assert generator.generate_sql("test") == "SELECT 1"
    prompt_context = captured["prompt"].split(
        "DATABASE SCHEMA:\n",
        1,
    )[1].split("\n\nUSER QUESTION:", 1)[0]
    assert prompt_context == ""


def test_generate_sql_propagates_retrieval_errors(monkeypatch):
    class FailingRetriever:
        def retrieve(self, question, limit):
            raise RuntimeError("retrieval unavailable")

    monkeypatch.setattr(generator, "schema_retriever", FailingRetriever())

    with pytest.raises(RuntimeError, match="retrieval unavailable"):
        generator.generate_sql("test")


def test_uses_configured_positive_request_timeout(monkeypatch):
    monkeypatch.setenv("OLLAMA_REQUEST_TIMEOUT", "17")
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def generate(self, **kwargs):
            return {"response": "SELECT 1"}

    _stub_retrieval_and_validation(monkeypatch)
    monkeypatch.setattr("ssa.ollama_provider.ollama.Client", FakeClient)

    assert generator.generate_sql("test") == "SELECT 1"
    assert captured["timeout"] == 17


def test_accepts_typed_ollama_response_and_reads_usage(monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            return ollama.GenerateResponse(
                response="SELECT 1",
                model="qwen2.5-coder:7b",
                prompt_eval_count=11,
                eval_count=7,
            )

    _stub_retrieval_and_validation(monkeypatch)
    monkeypatch.setattr("ssa.ollama_provider.ollama.Client", FakeClient)

    sql, usage = generator.generate_sql("test", return_usage=True)

    assert sql == "SELECT 1"
    assert usage == {
        "model": "qwen2.5-coder:7b",
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }


def test_legacy_usage_recomputes_total_and_preserves_zero_defaults(monkeypatch):
    responses = [
        {
            "response": "SELECT 1",
            "prompt_eval_count": 100,
            "eval_count": 25,
            "total_tokens": 999,
        },
        {"response": "SELECT 1"},
        {
            "response": "SELECT 1",
            "prompt_eval_count": 0,
            "eval_count": 0,
            "total_tokens": 0,
        },
    ]

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            return responses.pop(0)

    _stub_retrieval_and_validation(monkeypatch)
    monkeypatch.setattr("ssa.ollama_provider.ollama.Client", FakeClient)

    _, reported_usage = generator.generate_sql("test", return_usage=True)
    _, missing_usage = generator.generate_sql("test", return_usage=True)
    _, zero_usage = generator.generate_sql("test", return_usage=True)

    assert reported_usage == {
        "model": "qwen2.5-coder:7b",
        "prompt_tokens": 100,
        "completion_tokens": 25,
        "total_tokens": 125,
    }
    assert missing_usage == {
        "model": "qwen2.5-coder:7b",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    assert zero_usage == {
        "model": "qwen2.5-coder:7b",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def test_records_retrieval_generation_and_validation_milliseconds(monkeypatch):
    clock = iter([10.0, 10.125, 20.0, 20.25, 30.0, 30.05])

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            return {"response": "SELECT 1"}

    _stub_retrieval_and_validation(monkeypatch)
    monkeypatch.setattr("ssa.ollama_provider.ollama.Client", FakeClient)
    monkeypatch.setattr(generator.time, "perf_counter", lambda: next(clock))

    timings = {}
    assert generator.generate_sql("test", timings=timings) == "SELECT 1"

    assert timings["retrieval_ms"] == pytest.approx(125.0)
    assert timings["generation_ms"] == pytest.approx(250.0)
    assert timings["validation_ms"] == pytest.approx(50.0)


def test_rejects_invalid_request_timeout(monkeypatch):
    monkeypatch.setenv("OLLAMA_REQUEST_TIMEOUT", "0")

    with pytest.raises(generator.ConfigurationError, match="OLLAMA_REQUEST_TIMEOUT"):
        generator.generate_sql("test")


def test_uses_cpu_friendly_default_request_timeout(monkeypatch):
    monkeypatch.delenv("OLLAMA_REQUEST_TIMEOUT", raising=False)

    assert generator.get_ollama_request_timeout() == 300


@pytest.mark.parametrize(
    ("failure", "error_type"),
    [
        (ConnectionError("http://user:password@example.invalid"), generator.OllamaConnectionError),
        (TimeoutError("request timed out at http://example.invalid"), generator.OllamaGenerationTimeoutError),
    ],
)
def test_classifies_ollama_failures_without_leaking_details(
    monkeypatch, failure, error_type
):
    class FailingClient:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            raise failure

    _stub_retrieval_and_validation(monkeypatch)
    monkeypatch.setattr("ssa.ollama_provider.ollama.Client", FailingClient)

    with pytest.raises(error_type) as error:
        generator.generate_sql("test")

    assert "password" not in str(error.value).lower()
    assert "http://" not in str(error.value)


@pytest.mark.parametrize("response", [{}, {"response": None}, {"response": ""}])
def test_rejects_malformed_ollama_responses(monkeypatch, response):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            return response

    _stub_retrieval_and_validation(monkeypatch)
    monkeypatch.setattr("ssa.ollama_provider.ollama.Client", FakeClient)

    with pytest.raises(generator.OllamaGenerationError, match="response"):
        generator.generate_sql("test")
