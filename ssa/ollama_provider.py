"""Ollama implementation of the provider-neutral generation contract.

The adapter is intentionally not wired into ``ssa.sql_generator`` yet. The
active generation path remains unchanged during CP23.1.
"""

import os
import socket
from collections.abc import Callable, Mapping

import ollama
from dotenv import load_dotenv

from ssa.contracts import GenerationRequest, GenerationResult, UsageMetadata
from ssa.db import ConfigurationError


load_dotenv()

OLLAMA_PROVIDER_NAME = "ollama"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:7b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_REQUEST_TIMEOUT = 300

PromptBuilder = Callable[[GenerationRequest], str]


class OllamaConnectionError(RuntimeError):
    """Raised when Ollama cannot be reached."""


class OllamaGenerationTimeoutError(RuntimeError):
    """Raised when Ollama does not finish generation in time."""


class OllamaGenerationError(RuntimeError):
    """Raised when Ollama returns an unusable generation response."""


def get_ollama_request_timeout(request_timeout: int | None = None) -> int:
    """Return a positive Ollama request timeout in seconds."""

    value = request_timeout
    if value is None:
        value = os.getenv(
            "OLLAMA_REQUEST_TIMEOUT",
            str(DEFAULT_OLLAMA_REQUEST_TIMEOUT),
        )

    try:
        timeout = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            "OLLAMA_REQUEST_TIMEOUT must be a positive integer."
        ) from exc

    if timeout < 1:
        raise ConfigurationError("OLLAMA_REQUEST_TIMEOUT must be a positive integer.")

    return timeout


def _response_value(response: object, key: str, default: object = None) -> object:
    """Read fields from Ollama mappings and typed response models."""

    if isinstance(response, Mapping):
        return response.get(key, default)

    getter = getattr(response, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            return getter(key)

    try:
        return response[key]  # type: ignore[index]
    except (KeyError, TypeError, AttributeError):
        return getattr(response, key, default)


def _optional_int(value: object) -> int | None:
    """Preserve missing usage as None while retaining explicit zero values."""

    if value is None:
        return None
    return int(value)


def _usage_from_response(response: object, model: str) -> UsageMetadata:
    """Normalize Ollama token metadata into the CP22 usage contract."""

    try:
        response_model = _response_value(response, "model")
        prompt_tokens = _optional_int(
            _response_value(response, "prompt_eval_count")
        )
        completion_tokens = _optional_int(
            _response_value(response, "eval_count")
        )
        reported_total = _optional_int(
            _response_value(response, "total_tokens")
        )
        total_tokens = reported_total
        if total_tokens is None and (
            prompt_tokens is not None and completion_tokens is not None
        ):
            total_tokens = prompt_tokens + completion_tokens

        return UsageMetadata(
            provider=OLLAMA_PROVIDER_NAME,
            model=response_model if isinstance(response_model, str) else model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
    except (TypeError, ValueError) as exc:
        raise OllamaGenerationError(
            "Ollama returned invalid token usage metadata."
        ) from exc


def _ollama_error(exc: Exception) -> RuntimeError:
    """Translate client failures without exposing raw client details."""

    error_name = type(exc).__name__.lower()
    if isinstance(exc, (TimeoutError, socket.timeout)) or "timeout" in error_name:
        return OllamaGenerationTimeoutError(
            "Ollama generation timed out. Check Ollama or increase "
            "OLLAMA_REQUEST_TIMEOUT."
        )

    if isinstance(exc, ConnectionError) or any(
        name in error_name for name in ("connect", "network", "connection")
    ):
        return OllamaConnectionError(
            "Ollama is unavailable. Start Ollama and verify the local service."
        )

    return OllamaGenerationError(
        "Ollama could not generate SQL. Verify the configured model is available."
    )


class OllamaProvider:
    """Generate relational queries through the Ollama Python client."""

    def __init__(
        self,
        prompt_builder: PromptBuilder,
        *,
        model: str | None = None,
        base_url: str | None = None,
        request_timeout: int | None = None,
    ) -> None:
        self.prompt_builder = prompt_builder
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL",
            DEFAULT_OLLAMA_BASE_URL,
        )
        self.request_timeout = get_ollama_request_timeout(request_timeout)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate and normalize one Ollama response."""

        try:
            prompt = self.prompt_builder(request)
            if not isinstance(prompt, str) or not prompt.strip():
                raise OllamaGenerationError("Ollama prompt is empty.")

            client = ollama.Client(
                host=self.base_url,
                timeout=self.request_timeout,
            )
            response = client.generate(
                model=self.model,
                prompt=prompt,
            )

            response_text = _response_value(response, "response")
            if not isinstance(response_text, str) or not response_text.strip():
                raise OllamaGenerationError(
                    "Ollama returned an invalid generation response."
                )

            usage = _usage_from_response(response, self.model)
            return GenerationResult(
                query=response_text.strip(),
                usage=usage,
                provider=OLLAMA_PROVIDER_NAME,
                model=usage.model or self.model,
            )
        except (
            OllamaConnectionError,
            OllamaGenerationTimeoutError,
            OllamaGenerationError,
        ):
            raise
        except Exception as exc:
            raise _ollama_error(exc) from exc
