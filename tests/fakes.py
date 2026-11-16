"""Reusable deterministic fakes for provider-neutral SSA tests."""

from ssa.contracts import (
    GenerationRequest,
    GenerationResult,
    QueryResult,
    SchemaDocument,
    ValidatedQuery,
)
from ssa.pipeline import PipelineStageError


class FakeLLMProvider:
    """Record requests and return or raise a configured deterministic result."""

    def __init__(
        self,
        result: GenerationResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result or GenerationResult("SELECT 1")
        self.error = error
        self.calls: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return self.result


class FakeSchemaRetriever:
    """Return defensive copies of configured schema documents."""

    def __init__(
        self,
        documents: list[SchemaDocument] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.documents = list(documents or [])
        self.error = error
        self.calls: list[tuple[str, int]] = []

    def retrieve(
        self,
        question: str,
        limit: int = 5,
    ) -> list[SchemaDocument]:
        self.calls.append((question, limit))
        if self.error is not None:
            raise self.error
        return list(self.documents[:limit])


class FakeRelationalDatabaseAdapter:
    """Standard CP22 database fake without the timing extension."""

    def __init__(
        self,
        result: QueryResult | None = None,
        validation_error: Exception | None = None,
        execution_error: Exception | None = None,
    ) -> None:
        self.result = result
        self.validation_error = validation_error
        self.execution_error = execution_error
        self.validation_calls: list[tuple[str, int | None]] = []
        self.execution_calls: list[tuple[ValidatedQuery, int, int | None]] = []

    def validate_read_query(
        self,
        query: str,
        timeout: int | None = None,
    ) -> ValidatedQuery:
        self.validation_calls.append((query, timeout))
        if self.validation_error is not None:
            raise self.validation_error
        return ValidatedQuery(query)

    def execute_read(
        self,
        query: ValidatedQuery,
        max_rows: int = 100,
        timeout: int | None = None,
    ) -> QueryResult:
        self.execution_calls.append((query, max_rows, timeout))
        if self.execution_error is not None:
            raise self.execution_error
        if self.result is not None:
            return self.result
        return QueryResult(
            columns=["value"],
            rows=[{"value": query.sql}],
            row_count=1,
            max_rows=max_rows,
            truncated=False,
        )


class TimingAwareFakeRelationalDatabaseAdapter(FakeRelationalDatabaseAdapter):
    """Explicit timing-aware extension of the standard CP22 database fake."""

    def __init__(self, *args, timing_updates=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.timing_calls: list[dict | None] = []
        self.timing_updates = dict(timing_updates or {})

    def execute_read(
        self,
        query: ValidatedQuery,
        max_rows: int = 100,
        timeout: int | None = None,
        *,
        timings: dict | None = None,
    ) -> QueryResult:
        self.timing_calls.append(timings)
        if timings is not None:
            timings.update(self.timing_updates)
        return super().execute_read(query, max_rows, timeout)


class FakeSQLGenerator:
    """Deterministic callable matching the active generator convention."""

    def __init__(
        self,
        sql: str = "SELECT 1",
        usage: dict | None = None,
        error: Exception | None = None,
        stage_error: PipelineStageError | None = None,
        timing_updates: dict | None = None,
    ) -> None:
        self.sql = sql
        self.usage = usage
        self.error = error
        self.stage_error = stage_error
        self.timing_updates = dict(timing_updates or {})
        self.calls: list[dict] = []

    def __call__(
        self,
        question: str,
        *,
        return_usage: bool = False,
        timings: dict | None = None,
    ) -> str | tuple[str, dict]:
        self.calls.append({
            "question": question,
            "return_usage": return_usage,
            "timings": timings,
        })
        if timings is not None:
            timings.update(self.timing_updates)
        if self.stage_error is not None:
            raise self.stage_error
        if self.error is not None:
            raise self.error
        if return_usage and self.usage is not None:
            return self.sql, self.usage
        return self.sql
