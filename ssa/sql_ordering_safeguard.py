"""Advisory detection of potentially nondeterministic TOP ordering."""

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Literal

import sqlglot
from sqlglot import exp


OrderingStatus = Literal["risk", "sufficient", "unsupported"]


@dataclass(frozen=True)
class OrderingRisk:
    """Immutable advisory result for one generated SQL statement."""

    status: OrderingStatus
    reason: str
    table: str | None = None
    primary_key_columns: tuple[str, ...] = ()
    missing_key_columns: tuple[str, ...] = ()


def inspect_top_ordering(
    sql: str | None,
    primary_keys: Mapping[str, Sequence[str]],
) -> OrderingRisk:
    """Inspect a narrow, single-table TOP query without changing or executing it.

    ``primary_keys`` maps fully qualified table names to their ordered primary
    key columns, for example ``{"Catalog.Item": ("ItemKey",)}``. The
    mapping is caller-supplied and is never read from disk or external services.
    """

    if not isinstance(sql, str) or not sql.strip():
        return _unsupported("unsupported_empty_sql")

    metadata = _resolve_primary_keys(primary_keys)
    if metadata is None:
        return _unsupported("unsupported_primary_key_metadata")

    try:
        statements = sqlglot.parse(sql, read="tsql")
    except (sqlglot.errors.ParseError, ValueError, TypeError):
        return _unsupported("unsupported_parse_error")

    if len(statements) != 1:
        return _unsupported("unsupported_multiple_statements")

    statement = statements[0]
    if not isinstance(statement, exp.Select):
        return _unsupported("unsupported_non_select")

    if statement.args.get("with_") is not None:
        return _unsupported("unsupported_cte")
    if statement.find(exp.Subquery) is not None:
        return _unsupported("unsupported_subquery")
    if statement.find(exp.Window) is not None:
        return _unsupported("unsupported_window")
    if statement.args.get("joins"):
        return _unsupported("unsupported_join")
    if statement.args.get("group") is not None or statement.find(exp.AggFunc):
        return _unsupported("unsupported_grouping_or_aggregate")
    if statement.args.get("distinct") is not None:
        return _unsupported("unsupported_distinct")

    limit = statement.args.get("limit")
    if limit is None:
        return _unsupported("unsupported_top_missing")
    if limit.meta.get("top") is not True:
        return _unsupported("unsupported_non_top_limit")

    limit_options = limit.args.get("limit_options")
    if limit_options is not None and (
        limit_options.args.get("with_ties")
        or limit_options.args.get("percent")
    ):
        return _unsupported("unsupported_top_options")

    expression = limit.args.get("expression")
    if (
        not isinstance(expression, exp.Literal)
        or expression.is_string
        or not expression.is_int
        or int(expression.this) < 1
    ):
        return _unsupported("unsupported_top_expression")

    from_expression = statement.args.get("from_")
    table = getattr(from_expression, "this", None)
    if not isinstance(table, exp.Table):
        return _unsupported("unsupported_table_shape")
    if len(list(statement.find_all(exp.Table))) != 1:
        return _unsupported("unsupported_table_shape")

    table_name = _qualified_table_name(table)
    resolved = metadata.get(_normalize_identifier(table_name))
    if resolved is None:
        return _unsupported("unsupported_table_metadata", table=table_name)

    metadata_name, key_columns = resolved
    if len(key_columns) != 1:
        return _unsupported(
            "unsupported_composite_primary_key",
            table=metadata_name,
            primary_key_columns=key_columns,
        )

    order = statement.args.get("order")
    if order is None or not order.expressions:
        return _unsupported(
            "unsupported_order_by_missing",
            table=metadata_name,
            primary_key_columns=key_columns,
        )

    output_aliases = {
        _normalize_identifier(expression.alias)
        for expression in statement.expressions
        if isinstance(expression, exp.Alias) and expression.alias
    }
    table_alias = _normalize_identifier(table.alias) if table.alias else None
    table_leaf = _normalize_identifier(table.name)
    allowed_qualifiers = {table_alias} if table_alias else {table_leaf}
    key_name = _normalize_identifier(key_columns[0])

    for ordered in order.expressions:
        column = ordered.this
        if not isinstance(column, exp.Column):
            return _unsupported(
                "unsupported_order_expression",
                table=metadata_name,
                primary_key_columns=key_columns,
            )

        column_name = _normalize_identifier(column.name)
        qualifier = _normalize_identifier(column.table) if column.table else None
        if qualifier and qualifier not in allowed_qualifiers:
            return _unsupported(
                "unsupported_ambiguous_order_column",
                table=metadata_name,
                primary_key_columns=key_columns,
            )
        if column_name in output_aliases:
            return _unsupported(
                "unsupported_order_alias",
                table=metadata_name,
                primary_key_columns=key_columns,
            )
        if column_name == key_name:
            return OrderingRisk(
                status="sufficient",
                reason="primary_key_in_root_order",
                table=metadata_name,
                primary_key_columns=key_columns,
            )

    return OrderingRisk(
        status="risk",
        reason="primary_key_missing_from_root_order",
        table=metadata_name,
        primary_key_columns=key_columns,
        missing_key_columns=key_columns,
    )


def _resolve_primary_keys(
    primary_keys: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, tuple[str, ...]]] | None:
    if not isinstance(primary_keys, Mapping) or not primary_keys:
        return None

    resolved: dict[str, tuple[str, tuple[str, ...]]] = {}
    for table_name, columns in primary_keys.items():
        if not isinstance(table_name, str) or not table_name.strip():
            return None
        if (
            not isinstance(columns, Sequence)
            or isinstance(columns, (str, bytes))
            or not columns
            or not all(isinstance(column, str) and column.strip() for column in columns)
        ):
            return None
        normalized_name = _normalize_identifier(table_name)
        normalized_columns = tuple(column.strip() for column in columns)
        resolved[normalized_name] = (table_name, normalized_columns)

    return resolved


def _qualified_table_name(table: exp.Table) -> str:
    parts = [
        part
        for part in (table.catalog, table.db, table.name)
        if part
    ]
    return ".".join(parts)


def _normalize_identifier(value: str) -> str:
    return value.strip().strip("[]").replace("[", "").replace("]", "").lower()


def _unsupported(
    reason: str,
    *,
    table: str | None = None,
    primary_key_columns: Sequence[str] = (),
) -> OrderingRisk:
    return OrderingRisk(
        status="unsupported",
        reason=reason,
        table=table,
        primary_key_columns=tuple(primary_key_columns),
    )


__all__ = ["OrderingRisk", "OrderingStatus", "inspect_top_ordering"]
