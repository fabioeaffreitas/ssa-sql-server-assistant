"""Internal provider-neutral contracts for the SSA pipeline.

These dataclasses and Protocols describe the intended internal boundaries for
future provider and database adapters. They are not yet used by the running
Ollama + SQL Server application; existing functions and dictionary response
shapes remain unchanged.
"""

from dataclasses import dataclass, field
from typing import Protocol, TypeAlias


QueryRow: TypeAlias = dict[str, object]
TimingValue: TypeAlias = float | None


@dataclass(frozen=True)
class UsageMetadata:
    """Optional usage information reported by an LLM provider.

    ``None`` means that a provider did not report a value. A value of ``0``
    remains distinct and means that the provider explicitly reported zero.
    """

    provider: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class GenerationRequest:
    """Provider-neutral input for relational query generation."""

    question: str
    schema_context: str
    dialect: str = "tsql"
    model: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    """Provider-neutral generated query and optional provider metadata."""

    query: str
    usage: UsageMetadata | None = None
    provider: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class ValidatedQuery:
    """A query that has passed the selected relational safety checks."""

    sql: str


@dataclass
class QueryResult:
    """A bounded successful relational query result."""

    columns: list[str] = field(default_factory=list)
    rows: list[QueryRow] = field(default_factory=list)
    row_count: int = 0
    max_rows: int = 100
    truncated: bool = False


@dataclass
class PipelineDiagnostics:
    """Provider-neutral CP21.6 pipeline timings and failure information."""

    retrieval_ms: TimingValue = None
    generation_ms: TimingValue = None
    validation_ms: TimingValue = None
    execution_ms: TimingValue = None
    total_ms: TimingValue = None
    failure_stage: str | None = None


class LLMProvider(Protocol):
    """Interface for a provider that generates a relational query."""

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate a query from a question and retrieved schema context."""


class SchemaRetriever(Protocol):
    """Interface for retrieving schema documents relevant to a question."""

    def retrieve(
        self,
        question: str,
        limit: int = 5,
    ) -> list["SchemaDocument"]:
        """Return up to ``limit`` relevant schema documents."""


@dataclass(frozen=True)
class SchemaDocument:
    """Minimal typed representation of a retrieved schema document."""

    id: str
    document: str
    distance: float | None = None


class RelationalDatabaseAdapter(Protocol):
    """Interface for validation and bounded read execution on SQL databases."""

    def validate_read_query(
        self,
        query: str,
        timeout: int | None = None,
    ) -> ValidatedQuery:
        """Validate raw relational query text before execution."""

    def execute_read(
        self,
        query: ValidatedQuery,
        max_rows: int = 100,
        timeout: int | None = None,
    ) -> QueryResult:
        """Execute a previously validated, bounded read query."""
