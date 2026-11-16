from unittest.mock import Mock

import ollama
import pytest

from ssa.contracts import GenerationRequest, LLMProvider
from ssa.db import ConfigurationError
from ssa.ollama_provider import (
    OllamaConnectionError,
    OllamaGenerationError,
    OllamaGenerationTimeoutError,
    OllamaProvider,
    get_ollama_request_timeout,
)


def prompt_builder(request: GenerationRequest) -> str:
    return (
        f"Dialect: {request.dialect}\n"
        f"Schema: {request.schema_context}\n"
        f"Question: {request.question}"
    )


def test_successful_generation_normalizes_dictionary_response(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

        def generate(self, **kwargs):
            captured["generate"] = kwargs
            return {
                "response": "SELECT 1",
                "model": "qwen2.5-coder:7b",
                "prompt_eval_count": 11,
                "eval_count": 7,
            }

    monkeypatch.setattr("ssa.ollama_provider.ollama.Client", FakeClient)
    provider = OllamaProvider(
        prompt_builder,
        model="qwen2.5-coder:7b",
        base_url="http://ollama.test:11434",
        request_timeout=17,
    )

    result = provider.generate(GenerationRequest("question", "schema"))

    assert result.query == "SELECT 1"
    assert result.provider == "ollama"
    assert result.model == "qwen2.5-coder:7b"
    assert result.usage is not None
    assert result.usage.prompt_tokens == 11
    assert result.usage.completion_tokens == 7
    assert result.usage.total_tokens == 18
    assert captured["client"] == {
        "host": "http://ollama.test:11434",
        "timeout": 17,
    }
    assert captured["generate"]["model"] == "qwen2.5-coder:7b"
    assert "Question: question" in captured["generate"]["prompt"]


def test_successful_generation_normalizes_typed_ollama_response(monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            return ollama.GenerateResponse(
                response="SELECT 1",
                model="typed-model",
                prompt_eval_count=3,
                eval_count=4,
            )

    monkeypatch.setattr("ssa.ollama_provider.ollama.Client", FakeClient)
    result = OllamaProvider(prompt_builder).generate(
        GenerationRequest("question", "schema")
    )

    assert result.query == "SELECT 1"
    assert result.model == "typed-model"
    assert result.usage.prompt_tokens == 3
    assert result.usage.completion_tokens == 4


def test_missing_and_zero_token_counts_remain_distinct(monkeypatch):
    responses = [
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

    monkeypatch.setattr("ssa.ollama_provider.ollama.Client", FakeClient)
    provider = OllamaProvider(prompt_builder)

    missing = provider.generate(GenerationRequest("question", "schema"))
    zero = provider.generate(GenerationRequest("question", "schema"))

    assert missing.usage.prompt_tokens is None
    assert missing.usage.completion_tokens is None
    assert missing.usage.total_tokens is None
    assert zero.usage.prompt_tokens == 0
    assert zero.usage.completion_tokens == 0
    assert zero.usage.total_tokens == 0


def test_provider_uses_environment_configuration_and_default_timeout(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "configured-model")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://configured.test")
    monkeypatch.delenv("OLLAMA_REQUEST_TIMEOUT", raising=False)

    provider = OllamaProvider(prompt_builder)

    assert provider.model == "configured-model"
    assert provider.base_url == "http://configured.test"
    assert provider.request_timeout == 300


def test_provider_accepts_custom_timeout_and_rejects_invalid_timeout(monkeypatch):
    assert get_ollama_request_timeout(17) == 17
    monkeypatch.setenv("OLLAMA_REQUEST_TIMEOUT", "0")

    with pytest.raises(ConfigurationError, match="OLLAMA_REQUEST_TIMEOUT"):
        OllamaProvider(prompt_builder)


@pytest.mark.parametrize(
    ("failure", "error_type"),
    [
        (ConnectionError("raw connection details"), OllamaConnectionError),
        (TimeoutError("raw timeout details"), OllamaGenerationTimeoutError),
        (RuntimeError("raw provider details"), OllamaGenerationError),
    ],
)
def test_provider_classifies_client_failures_without_exposing_details(
    monkeypatch, failure, error_type
):
    class FailingClient:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            raise failure

    monkeypatch.setattr("ssa.ollama_provider.ollama.Client", FailingClient)

    with pytest.raises(error_type) as error:
        OllamaProvider(prompt_builder).generate(
            GenerationRequest("question", "schema")
        )

    assert "raw" not in str(error.value)
    assert "connection details" not in str(error.value)


@pytest.mark.parametrize(
    "response",
    [{}, {"response": None}, {"response": ""}, {"response": 1}],
)
def test_provider_rejects_malformed_responses(monkeypatch, response):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            return response

    monkeypatch.setattr("ssa.ollama_provider.ollama.Client", FakeClient)

    with pytest.raises(OllamaGenerationError, match="response"):
        OllamaProvider(prompt_builder).generate(
            GenerationRequest("question", "schema")
        )


def test_provider_implements_llm_provider_shape(monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def generate(self, **kwargs):
            return {"response": "SELECT 1"}

    monkeypatch.setattr("ssa.ollama_provider.ollama.Client", FakeClient)
    provider: LLMProvider = OllamaProvider(prompt_builder)

    result = provider.generate(GenerationRequest("question", "schema"))

    assert result.query == "SELECT 1"


def test_prompt_builder_failure_is_sanitized(monkeypatch):
    provider = OllamaProvider(Mock(side_effect=RuntimeError("raw prompt details")))

    with pytest.raises(OllamaGenerationError) as error:
        provider.generate(GenerationRequest("question", "schema"))

    assert "raw prompt details" not in str(error.value)
