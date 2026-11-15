import pytest
import ollama

import ssa.sql_generator as generator


def _stub_retrieval_and_validation(monkeypatch):
    monkeypatch.setattr(generator, "search_schema", lambda question, n_results: [])
    monkeypatch.setattr(generator, "validate_sql", lambda sql: sql)


def test_uses_configured_positive_request_timeout(monkeypatch):
    monkeypatch.setenv("OLLAMA_REQUEST_TIMEOUT", "17")
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def generate(self, **kwargs):
            return {"response": "SELECT 1"}

    _stub_retrieval_and_validation(monkeypatch)
    monkeypatch.setattr(generator.ollama, "Client", FakeClient)

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
    monkeypatch.setattr(generator.ollama, "Client", FakeClient)

    sql, usage = generator.generate_sql("test", return_usage=True)

    assert sql == "SELECT 1"
    assert usage == {
        "model": "qwen2.5-coder:7b",
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }


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
    monkeypatch.setattr(generator.ollama, "Client", FailingClient)

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
    monkeypatch.setattr(generator.ollama, "Client", FakeClient)

    with pytest.raises(generator.OllamaGenerationError, match="response"):
        generator.generate_sql("test")
