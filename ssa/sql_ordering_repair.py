"""Disabled-by-default proposal generation for narrow TOP ordering repairs."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import sqlglot
from sqlglot import exp

from ssa.sql_ordering_safeguard import inspect_top_ordering
from ssa.sql_validator import SQLValidationError, validate_sql


RepairStatus = Literal["disabled", "proposed", "not_eligible"]


@dataclass(frozen=True)
class OrderingRepairProposal:
    """Original SQL and an optional, independently validated proposal."""

    status: RepairStatus
    reason: str
    original_sql: str | None
    proposed_sql: str | None
    table: str | None = None
    primary_key_columns: tuple[str, ...] = ()
    appended_columns: tuple[str, ...] = ()


def propose_top_ordering_repair(
    sql: str | None,
    primary_keys: Mapping[str, Sequence[str]],
    *,
    enabled: bool = False,
    tie_breaking_permitted: bool = False,
) -> OrderingRepairProposal:
    """Propose a key tie-breaker for one strictly supported TOP query.

    This function never executes SQL and never reads metadata from disk. The
    caller is responsible for supplying authoritative, sufficiently current
    metadata and for making the explicit policy decision that tie-breaking is
    permitted. Both policy switches must be the literal boolean ``True``.
    """

    if enabled is not True:
        return _disabled(sql)
    if tie_breaking_permitted is not True:
        return _not_eligible(sql, "tie_breaking_not_permitted")
    if not isinstance(sql, str) or not sql.strip():
        return _not_eligible(sql, "unsupported_empty_sql")

    try:
        statements = sqlglot.parse(sql, read="tsql")
    except (sqlglot.errors.ParseError, TypeError, ValueError):
        return _not_eligible(sql, "unsupported_parse_error")

    if len(statements) != 1:
        return _not_eligible(sql, "unsupported_multiple_statements")
    statement = statements[0]
    if not isinstance(statement, exp.Select):
        return _not_eligible(sql, "unsupported_non_select")

    eligibility = _eligible_select(statement, primary_keys)
    if eligibility is None:
        return _not_eligible(sql, "unsupported_query_shape")
    table_name, key_columns = eligibility

    advisory = inspect_top_ordering(sql, primary_keys)
    if advisory.status == "sufficient":
        return OrderingRepairProposal(
            status="not_eligible",
            reason="primary_key_already_ordered",
            original_sql=sql,
            proposed_sql=None,
            table=advisory.table or table_name,
            primary_key_columns=key_columns,
        )
    if advisory.status != "risk":
        return _not_eligible(sql, advisory.reason)

    order = statement.args.get("order")
    if order is None or not order.expressions:
        return _not_eligible(sql, "unsupported_order_by_missing")

    original_order_sql = tuple(
        expression.sql(dialect="tsql") for expression in order.expressions
    )
    original_projection_sql = tuple(
        expression.sql(dialect="tsql") for expression in statement.expressions
    )
    top_expression = statement.args["limit"].args.get("expression")
    top_value = top_expression.sql(dialect="tsql")

    key_name = key_columns[0]
    table = statement.args["from_"].this
    qualifier = table.alias or None
    appended = exp.Ordered(
        this=exp.column(key_name, table=qualifier),
        desc=False,
        # SQLGlot otherwise renders an implicit NULLS ordering CASE expression
        # for ascending T-SQL terms. Primary-key columns are non-null by
        # definition, so the explicit setting preserves the intended direct
        # column shape while keeping the proposal conservative.
        nulls_first=True,
    )
    order.append("expressions", appended)
    proposed_sql = statement.sql(dialect="tsql")

    if not _proposal_is_safe(
        proposed_sql,
        original_order_sql,
        original_projection_sql,
        top_value,
        key_name,
        qualifier,
    ):
        return _not_eligible(sql, "proposal_verification_failed")

    return OrderingRepairProposal(
        status="proposed",
        reason="primary_key_appended_to_root_order",
        original_sql=sql,
        proposed_sql=proposed_sql,
        table=advisory.table or table_name,
        primary_key_columns=key_columns,
        appended_columns=(key_name,),
    )


def _eligible_select(
    statement: exp.Select,
    primary_keys: Mapping[str, Sequence[str]],
) -> tuple[str, tuple[str, ...]] | None:
    if statement.args.get("with_") is not None:
        return None
    if statement.find(exp.Subquery) is not None:
        return None
    if statement.find(exp.Window) is not None:
        return None
    if statement.args.get("joins"):
        return None
    if statement.args.get("group") is not None:
        return None
    if statement.args.get("having") is not None:
        return None
    if statement.find(exp.AggFunc) is not None:
        return None
    if statement.args.get("distinct") is not None:
        return None
    if statement.find(exp.Rand) is not None:
        return None

    limit = statement.args.get("limit")
    if limit is None or limit.meta.get("top") is not True:
        return None
    options = limit.args.get("limit_options")
    if options is not None and (
        options.args.get("with_ties") or options.args.get("percent")
    ):
        return None
    expression = limit.args.get("expression")
    if (
        not isinstance(expression, exp.Literal)
        or expression.is_string
        or not expression.is_int
        or int(expression.this) < 1
    ):
        return None

    from_expression = statement.args.get("from_")
    table = getattr(from_expression, "this", None)
    if not isinstance(table, exp.Table):
        return None
    if len(list(statement.find_all(exp.Table))) != 1:
        return None
    if statement.args.get("order") is None:
        return None

    table_name = _qualified_table_name(table)
    metadata = _resolve_primary_keys(primary_keys)
    resolved = metadata.get(_normalize_identifier(table_name)) if metadata else None
    if resolved is None or len(resolved[1]) != 1:
        return None

    allowed_qualifier = _normalize_identifier(table.alias or table.name)
    for ordered in statement.args["order"].expressions:
        if not isinstance(ordered.this, exp.Column):
            return None
        qualifier = _normalize_identifier(ordered.this.table) if ordered.this.table else None
        if qualifier and qualifier != allowed_qualifier:
            return None
    return resolved


def _proposal_is_safe(
    proposed_sql: str,
    original_order_sql: tuple[str, ...],
    original_projection_sql: tuple[str, ...],
    top_value: str,
    key_name: str,
    qualifier: str | None,
) -> bool:
    try:
        validated_sql = validate_sql(proposed_sql)
        reparsed = sqlglot.parse(validated_sql, read="tsql")
    except (SQLValidationError, sqlglot.errors.ParseError, TypeError, ValueError):
        return False

    if len(reparsed) != 1 or not isinstance(reparsed[0], exp.Select):
        return False
    statement = reparsed[0]
    limit = statement.args.get("limit")
    order = statement.args.get("order")
    if limit is None or order is None:
        return False
    expression = limit.args.get("expression")
    if expression is None or expression.sql(dialect="tsql") != top_value:
        return False

    current_order = tuple(
        item.sql(dialect="tsql") for item in order.expressions
    )
    if current_order[: len(original_order_sql)] != original_order_sql:
        return False
    if len(current_order) != len(original_order_sql) + 1:
        return False
    appended = order.expressions[-1].this
    if not isinstance(appended, exp.Column):
        return False
    if _normalize_identifier(appended.name) != _normalize_identifier(key_name):
        return False
    if bool(order.expressions[-1].args.get("desc")):
        return False
    if qualifier and _normalize_identifier(appended.table) != _normalize_identifier(qualifier):
        return False
    if tuple(expression.sql(dialect="tsql") for expression in statement.expressions) != original_projection_sql:
        return False
    return True


def _resolve_primary_keys(
    primary_keys: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, tuple[str, ...]]] | None:
    if not isinstance(primary_keys, Mapping) or not primary_keys:
        return None
    resolved: dict[str, tuple[str, tuple[str, ...]]] = {}
    for table_name, columns in primary_keys.items():
        if not isinstance(table_name, str) or not _valid_table_name(table_name):
            return None
        if (
            not isinstance(columns, Sequence)
            or isinstance(columns, (str, bytes))
            or not columns
            or not all(isinstance(column, str) and column.strip() for column in columns)
        ):
            return None
        normalized_columns = tuple(column.strip() for column in columns)
        normalized = [_normalize_identifier(column) for column in normalized_columns]
        if len(set(normalized)) != len(normalized):
            return None
        normalized_name = _normalize_identifier(table_name)
        if normalized_name in resolved:
            return None
        resolved[normalized_name] = (table_name, normalized_columns)
    return resolved


def _qualified_table_name(table: exp.Table) -> str:
    return ".".join(part for part in (table.catalog, table.db, table.name) if part)


def _valid_table_name(value: str) -> bool:
    parts = [part.strip() for part in value.split(".")]
    return len(parts) == 2 and all(parts)


def _normalize_identifier(value: str | None) -> str:
    if not value:
        return ""
    return value.strip().strip("[]").replace("[", "").replace("]", "").casefold()


def _disabled(sql: str | None) -> OrderingRepairProposal:
    return OrderingRepairProposal(
        status="disabled",
        reason="repair_disabled",
        original_sql=sql,
        proposed_sql=None,
    )


def _not_eligible(
    sql: str | None,
    reason: str,
    *,
    table: str | None = None,
    primary_key_columns: Sequence[str] = (),
) -> OrderingRepairProposal:
    return OrderingRepairProposal(
        status="not_eligible",
        reason=reason,
        original_sql=sql,
        proposed_sql=None,
        table=table,
        primary_key_columns=tuple(primary_key_columns),
    )


__all__ = ["OrderingRepairProposal", "RepairStatus", "propose_top_ordering_repair"]
