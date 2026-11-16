import pytest
import ollama

from ssa.contracts import SchemaDocument
import ssa.sql_generator as generator


def _stub_retrieval_and_validation(monkeypatch):
    monkeypatch.setattr(
        generator.schema_retriever,
        "retrieve",
        lambda question, limit: [],
    )
    monkeypatch.setattr(generator, "validate_sql", lambda sql: sql)


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
