"""Inactive provider-neutral pipeline orchestration.

This module defines the internal orchestration seam without changing the
active ``assistant.ask`` entry point or any provider/database implementation.
"""

from dataclasses import dataclass
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from ssa.contracts import (
    PipelineDiagnostics,
    QueryResult,
    RelationalDatabaseAdapter,
    ValidatedQuery,
)
from ssa.sql_ordering_safeguard import OrderingRisk, inspect_top_ordering


TIMING_KEYS = (
    "retrieval_ms",
    "generation_ms",
    "validation_ms",
    "execution_ms",
    "total_ms",
)


class SQLGeneratorCallable(Protocol):
    """Existing SQL generator calling convention used by the seam."""

    def __call__(
        self,
        question: str,
        *,
        return_usage: bool = False,
        timings: dict | None = None,
    ) -> str | tuple[str, dict]:
        ...


class TimingAwareDatabaseAdapter(Protocol):
    """Internal opt-in extension for adapters that accept shared timings."""

    def execute_read(
        self,
        query: ValidatedQuery,
        max_rows: int = 100,
        timeout: int | None = None,
        *,
        timings: dict | None = None,
    ) -> QueryResult:
        ...


@dataclass(frozen=True)
class PipelineRunResult:
    """Typed successful output from one orchestration run."""

    sql: str
    usage: dict | None
    result: QueryResult
    diagnostics: PipelineDiagnostics
    ordering_advisory: OrderingRisk | None = None


class PipelineStageError(RuntimeError):
    """Optional explicit stage signal for injected components."""

    def __init__(self, stage: str, cause: Exception) -> None:
        self.stage = stage
        self.cause = cause
        super().__init__(str(cause))


class PipelineExecutionError(RuntimeError):
    """Wrap one pipeline failure while preserving its original cause."""

    def __init__(
        self,
        stage: str,
        cause: Exception,
        diagnostics: PipelineDiagnostics,
        sql: str = "",
        usage: dict | None = None,
        ordering_advisory: OrderingRisk | None = None,
    ) -> None:
        self.stage = stage
        self.cause = cause
        self.diagnostics = diagnostics
        self.sql = sql
        self.usage = usage
        self.ordering_advisory = ordering_advisory
        super().__init__(str(cause))


class PipelineOrchestrator:
    """Compose an injected SQL generator with a relational database adapter.

    ``timing_aware`` is an explicit internal compatibility seam. When true,
    the injected adapter must implement ``TimingAwareDatabaseAdapter`` and
    receives the shared timing dictionary. When false, only the CP22 protocol
    is required and validation/execution timings are not fabricated.
    """

    def __init__(
        self,
        sql_generator: SQLGeneratorCallable,
        database: RelationalDatabaseAdapter,
        *,
        max_rows: int = 100,
        timing_aware: bool = False,
        primary_keys: Mapping[str, Sequence[str]] | None = None,
        ordering_detector: Callable[
            [str, Mapping[str, Sequence[str]]], OrderingRisk
        ] = inspect_top_ordering,
    ) -> None:
        self.sql_generator = sql_generator
        self.database = database
        self.max_rows = max_rows
        self.timing_aware = timing_aware
        self.primary_keys = primary_keys
        self.ordering_detector = ordering_detector

    def run(self, question: str) -> PipelineRunResult:
        """Run generation and one bounded database read."""

        started_at = time.perf_counter()
        timings = {key: None for key in TIMING_KEYS}
        sql = ""
        usage = None
        ordering_advisory = None
        stage = "generation"

        try:
            generated = self.sql_generator(
                question,
                return_usage=True,
                timings=timings,
            )
            if isinstance(generated, tuple):
                sql, usage = generated
            else:
                sql = generated

            if self.primary_keys is not None:
                try:
                    ordering_advisory = self.ordering_detector(
                        sql,
                        self.primary_keys,
                    )
                except Exception as exc:
                    # Advisory analysis must never change request behavior.
                    logging.getLogger(__name__).warning(
                        "Ordering advisory failed: error_type=%s",
                        type(exc).__name__,
                    )

            stage = "execution"
            validated_query = ValidatedQuery(sql)
            if self.timing_aware:
                result = self.database.execute_read(
                    validated_query,
                    max_rows=self.max_rows,
                    timings=timings,
                )  # type: ignore[union-attr]
            else:
                result = self.database.execute_read(
                    validated_query,
                    max_rows=self.max_rows,
                )
        except PipelineStageError as exc:
            stage = exc.stage
            raise self._failure(
                stage,
                exc.cause,
                timings,
                started_at,
                sql,
                usage,
                ordering_advisory,
            ) from exc.cause
        except Exception as exc:
            raise self._failure(
                stage,
                exc,
                timings,
                started_at,
                sql,
                usage,
                ordering_advisory,
            ) from exc

        diagnostics = self._diagnostics(timings, None, started_at)
        return PipelineRunResult(
            sql=sql,
            usage=usage,
            result=result,
            diagnostics=diagnostics,
            ordering_advisory=ordering_advisory,
        )

    @staticmethod
    def _diagnostics(
        timings: dict,
        failure_stage: str | None,
        started_at: float,
    ) -> PipelineDiagnostics:
        timings["total_ms"] = (time.perf_counter() - started_at) * 1000
        return PipelineDiagnostics(
            retrieval_ms=timings.get("retrieval_ms"),
            generation_ms=timings.get("generation_ms"),
            validation_ms=timings.get("validation_ms"),
            execution_ms=timings.get("execution_ms"),
            total_ms=timings["total_ms"],
            failure_stage=failure_stage,
        )

    @classmethod
    def _failure(
        cls,
        stage: str,
        cause: Exception,
        timings: dict,
        started_at: float,
        sql: str,
        usage: dict | None,
        ordering_advisory: OrderingRisk | None = None,
    ) -> PipelineExecutionError:
        diagnostics = cls._diagnostics(timings, stage, started_at)
        return PipelineExecutionError(
            stage,
            cause,
            diagnostics,
            sql,
            usage,
            ordering_advisory,
        )
