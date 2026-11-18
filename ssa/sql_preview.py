"""Execution-free SQL preview orchestration.

This module deliberately stops before any database adapter, SQL Server
``NOEXEC`` validation, approval state, or query execution. The injected
generator is responsible for retrieval, model generation, SQL cleanup, and
local SQLGlot validation.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
import time
from typing import Literal, Protocol

from ssa.sql_ordering_safeguard import (
    OrderingRisk,
    inspect_top_ordering,
)
from ssa.sql_validator import SQLValidationError


PreviewStatus = Literal[
    "preview_ready",
    "invalid_request",
    "generation_failed",
    "validation_failed",
    "preview_failed",
]


class PreviewGenerator(Protocol):
    """Existing generation calling convention used by the preview service."""

    def __call__(
        self,
        question: str,
        *,
        return_usage: bool = False,
        timings: dict | None = None,
    ) -> str | tuple[str, dict]:
        ...


@dataclass(frozen=True)
class SQLPreviewResult:
    """Immutable result of local SQL generation and inspection only."""

    question: str
    status: PreviewStatus
    sql: str | None = None
    validation_passed: bool | None = None
    error: str | None = None
    ordering_advisory: OrderingRisk | None = None
    ordering_error: str | None = None
    usage: Mapping[str, object] | None = None
    retrieval_ms: float | None = None
    generation_ms: float | None = None
    validation_ms: float | None = None
    total_ms: float | None = None
    execution_attempted: bool = False

    def __post_init__(self) -> None:
        if self.execution_attempted is not False:
            raise ValueError("SQL preview results cannot indicate execution.")


class SQLPreviewService:
    """Generate and inspect SQL without accepting an execution dependency."""

    def __init__(
        self,
        generator: PreviewGenerator,
        primary_keys: Mapping[str, Sequence[str]] | None = None,
        *,
        ordering_inspector: Callable[
            [str, Mapping[str, Sequence[str]]], OrderingRisk
        ] = inspect_top_ordering,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._generator = generator
        self._primary_keys = primary_keys
        self._ordering_inspector = ordering_inspector
        self._clock = clock

    def preview(self, question: str) -> SQLPreviewResult:
        """Return a locally validated SQL preview without executing SQL."""

        normalized_question = question.strip() if isinstance(question, str) else ""
        if not normalized_question:
            return SQLPreviewResult(
                question=normalized_question,
                status="invalid_request",
                error="Question must not be empty.",
            )

        started_at = self._clock()
        timings: dict[str, float | None] = {
            "retrieval_ms": None,
            "generation_ms": None,
            "validation_ms": None,
        }

        try:
            generated = self._generator(
                normalized_question,
                return_usage=True,
                timings=timings,
            )
        except SQLValidationError:
            return self._result(
                normalized_question,
                started_at,
                timings,
                status="validation_failed",
                validation_passed=False,
                error="Generated SQL failed local validation.",
            )
        except Exception as exc:
            status = (
                "generation_failed"
                if type(exc).__name__ in _GENERATION_ERROR_NAMES
                else "preview_failed"
            )
            return self._result(
                normalized_question,
                started_at,
                timings,
                status=status,
                error=_ERROR_MESSAGES[status],
            )

        sql, usage = _split_generation_result(generated)
        if not isinstance(sql, str) or not sql.strip():
            return self._result(
                normalized_question,
                started_at,
                timings,
                status="validation_failed",
                validation_passed=False,
                error="Generated SQL was empty.",
                usage=usage,
            )

        ordering_advisory = None
        ordering_error = None
        try:
            ordering_advisory = self._ordering_inspector(
                sql,
                self._primary_keys or {},
            )
        except Exception:
            # Ordering inspection is advisory and must not make valid local
            # generation unavailable.
            ordering_error = "ordering_inspection_unavailable"

        return self._result(
            normalized_question,
            started_at,
            timings,
            status="preview_ready",
            sql=sql,
            validation_passed=True,
            ordering_advisory=ordering_advisory,
            ordering_error=ordering_error,
            usage=usage,
        )

    def _result(
        self,
        question: str,
        started_at: float,
        timings: Mapping[str, float | None],
        *,
        status: PreviewStatus,
        sql: str | None = None,
        validation_passed: bool | None = None,
        error: str | None = None,
        ordering_advisory: OrderingRisk | None = None,
        ordering_error: str | None = None,
        usage: Mapping[str, object] | None = None,
    ) -> SQLPreviewResult:
        total_ms = (self._clock() - started_at) * 1000
        immutable_usage = (
            MappingProxyType(dict(usage)) if usage is not None else None
        )
        return SQLPreviewResult(
            question=question,
            status=status,
            sql=sql,
            validation_passed=validation_passed,
            error=error,
            ordering_advisory=ordering_advisory,
            ordering_error=ordering_error,
            usage=immutable_usage,
            retrieval_ms=timings.get("retrieval_ms"),
            generation_ms=timings.get("generation_ms"),
            validation_ms=timings.get("validation_ms"),
            total_ms=total_ms,
        )


def _split_generation_result(
    generated: object,
) -> tuple[object, Mapping[str, object] | None]:
    if isinstance(generated, tuple) and len(generated) == 2:
        sql, usage = generated
        if isinstance(usage, Mapping):
            return sql, usage
        return sql, None
    return generated, None


_GENERATION_ERROR_NAMES = frozenset({
    "OllamaConnectionError",
    "OllamaGenerationError",
    "OllamaGenerationTimeoutError",
})

_ERROR_MESSAGES = {
    "generation_failed": "SQL generation failed.",
    "preview_failed": "SQL preview failed.",
}


__all__ = [
    "PreviewGenerator",
    "PreviewStatus",
    "SQLPreviewResult",
    "SQLPreviewService",
]
