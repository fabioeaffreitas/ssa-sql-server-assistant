"""SQL Server implementation of the relational adapter contract."""

from ssa.contracts import QueryResult, RelationalDatabaseAdapter, ValidatedQuery
from ssa.sql_executor import execute_sql
from ssa.sql_schema_validator import validate_sql_schema


class SQLServerDatabaseAdapter:
    """Expose the existing SQL Server validation and execution functions."""

    def validate_read_query(
        self,
        query: str,
        timeout: int | None = None,
    ) -> ValidatedQuery:
        """Validate SQL through the existing SQL Server schema validator."""

        validated_sql = validate_sql_schema(query, query_timeout=timeout)
        return ValidatedQuery(validated_sql)

    def execute_read(
        self,
        query: ValidatedQuery,
        max_rows: int = 100,
        timeout: int | None = None,
        *,
        timings: dict | None = None,
    ) -> QueryResult:
        """Execute through the existing executor, including its validation."""

        execute_kwargs = {
            "max_rows": max_rows,
            "query_timeout": timeout,
        }
        if timings is not None:
            execute_kwargs["timings"] = timings

        result = execute_sql(query.sql, **execute_kwargs)
        return QueryResult(
            columns=result["columns"],
            rows=result["rows"],
            row_count=result["row_count"],
            max_rows=result["max_rows"],
            truncated=result["truncated"],
        )


def _protocol_check(adapter: SQLServerDatabaseAdapter) -> RelationalDatabaseAdapter:
    """Keep protocol compatibility visible to static type checkers."""

    return adapter
