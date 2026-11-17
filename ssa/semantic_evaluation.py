"""Deterministic semantic comparison for reviewed SQL evaluation cases.

This module is intentionally not connected to the live evaluation runner. It
accepts one CP31.2 case mapping, generated T-SQL, a legacy or typed query
result, and pipeline success. Unsupported or incomplete expectations are
reported as ``not_evaluated`` rather than treated as passing.

Supported structural invariant grammar:

* ``table:<name>``
* ``aggregate:<SUM|COUNT|AVG|MIN|MAX>``
* ``order_by:<column> <ASC|DESC>,<column> <ASC|DESC>``
* ``top:<positive integer>``

The grammar is deliberately narrow. Other non-empty invariant strings are
unsupported until a later checkpoint defines and tests them.

JSON ground-truth proposals may encode precise scalar values as tagged
objects: ``{"__type__": "decimal", "value": "12.3400"}``, or equivalent
``date`` and ``datetime`` tags. These are decoded before comparison so money
values are not coerced through binary floating point.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import math
from typing import Any

import sqlglot
from sqlglot import exp

from ssa.contracts import QueryResult


@dataclass(frozen=True)
class SemanticEvaluationResult:
    """Outcome of one deterministic semantic comparison."""

    status: str
    pipeline_success: bool
    sql_structure_pass: bool | None
    result_pass: bool | None
    semantic_pass: bool | None
    failure_reasons: tuple[str, ...] = ()


def compare_semantics(
    case: Mapping[str, Any],
    generated_sql: str | None,
    query_result: QueryResult | Mapping[str, Any] | None,
    pipeline_success: bool,
) -> SemanticEvaluationResult:
    """Compare one case without contacting a provider or database."""

    if not pipeline_success:
        return _not_evaluated(False, "pipeline_failed")

    review_status = case.get("review_status")
    if review_status in {"review_required", "excluded"}:
        return _not_evaluated(True, f"case_{review_status}")
    if review_status != "verified":
        return _not_evaluated(True, "unsupported_review_status")

    if not _has_ground_truth(case):
        return _not_evaluated(True, "insufficient_ground_truth")

    tolerance = case.get("numeric_tolerance")
    if tolerance is not None and (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or tolerance < 0
        or not math.isfinite(tolerance)
    ):
        return _not_evaluated(True, "invalid_numeric_tolerance")

    structure_pass: bool | None = None
    reasons: list[str] = []
    invariants = list(case.get("required_sql_invariants") or [])
    forbidden = list(case.get("forbidden_sql_invariants") or [])
    if invariants or forbidden:
        structure_pass, structure_reason = _compare_structure(
            generated_sql,
            invariants,
            forbidden,
        )
        if structure_reason:
            reasons.append(structure_reason)

    result_pass: bool | None = None
    expected_columns = case.get("expected_columns")
    expected_rows = case.get("expected_rows")
    expected_row_count = case.get("expected_row_count")
    needs_result = (
        expected_columns is not None
        or expected_rows is not None
        or expected_row_count is not None
    )

    if needs_result:
        if query_result is None:
            result_pass = None
            reasons.append("query_result_missing")
        else:
            actual = _result_parts(query_result)
            if actual is None:
                result_pass = None
                reasons.append("unsupported_query_result_shape")
            elif actual["truncated"] and (
                expected_rows is not None or expected_row_count is not None
            ):
                result_pass = None
                reasons.append("query_result_truncated")
            else:
                result_pass, result_reasons = _compare_result(
                    case,
                    actual,
                    tolerance,
                )
                reasons.extend(result_reasons)

    if "query_result_truncated" in reasons or "query_result_missing" in reasons:
        return _not_evaluated(True, *reasons, structure_pass=structure_pass, result_pass=result_pass)
    if "unsupported_query_result_shape" in reasons:
        return _not_evaluated(True, *reasons, structure_pass=structure_pass, result_pass=result_pass)
    if structure_pass is False or result_pass is False:
        return SemanticEvaluationResult(
            status="fail",
            pipeline_success=True,
            sql_structure_pass=structure_pass,
            result_pass=result_pass,
            semantic_pass=False,
            failure_reasons=tuple(reasons),
        )
    if structure_pass is None and result_pass is None:
        return _not_evaluated(True, "no_evaluable_expectations", failure_reasons=reasons)
    if reasons:
        return _not_evaluated(True, *reasons, structure_pass=structure_pass, result_pass=result_pass)

    return SemanticEvaluationResult(
        status="pass",
        pipeline_success=True,
        sql_structure_pass=structure_pass,
        result_pass=result_pass,
        semantic_pass=True,
    )


def _not_evaluated(
    pipeline_success: bool,
    *reasons: str,
    structure_pass: bool | None = None,
    result_pass: bool | None = None,
    failure_reasons: Sequence[str] | None = None,
) -> SemanticEvaluationResult:
    combined = tuple(failure_reasons or ()) + tuple(reasons)
    return SemanticEvaluationResult(
        status="not_evaluated",
        pipeline_success=pipeline_success,
        sql_structure_pass=structure_pass,
        result_pass=result_pass,
        semantic_pass=None,
        failure_reasons=combined,
    )


def _has_ground_truth(case: Mapping[str, Any]) -> bool:
    return any(
        case.get(field) is not None
        for field in (
            "expected_columns",
            "expected_rows",
            "expected_row_count",
        )
    ) or bool(
        case.get("required_sql_invariants")
        or case.get("forbidden_sql_invariants")
    )


def _compare_structure(
    generated_sql: str | None,
    required: list[str],
    forbidden: list[str],
) -> tuple[bool | None, str | None]:
    if not generated_sql or not generated_sql.strip():
        return False, "generated_sql_missing"
    try:
        statement = sqlglot.parse_one(generated_sql, read="tsql")
    except (sqlglot.errors.ParseError, ValueError):
        return False, "sql_structure_parse_failed"

    checks = [(item, True) for item in required] + [(item, False) for item in forbidden]
    known_failure = None
    unsupported = None
    for invariant, required_invariant in checks:
        kind, separator, value = invariant.partition(":")
        normalized_kind = kind.strip().lower()
        normalized_value = value.strip()
        if not separator or normalized_kind not in {"table", "aggregate", "order_by", "top"} or not normalized_value:
            unsupported = unsupported or f"unsupported_invariant:{invariant}"
            continue

        matched, reason = _matches_invariant(
            statement,
            normalized_kind,
            normalized_value,
        )
        if reason:
            if matched is None:
                unsupported = unsupported or reason
            else:
                known_failure = known_failure or reason
            continue
        if required_invariant and not matched:
            known_failure = known_failure or f"missing_invariant:{invariant}"
        if not required_invariant and matched:
            known_failure = known_failure or f"forbidden_invariant:{invariant}"

    if known_failure:
        return False, known_failure
    if unsupported:
        return None, unsupported
    return True, None


def _matches_invariant(
    statement: exp.Expression,
    kind: str,
    value: str,
) -> tuple[bool | None, str | None]:
    target = _normalize_identifier(value)
    if kind == "table":
        for table in statement.find_all(exp.Table):
            table_name = _normalize_identifier(table.name)
            qualified_name = _normalize_identifier(
                ".".join(part for part in (table.db, table.name) if part)
            )
            if target in {table_name, qualified_name}:
                return True, None
        return False, None

    if kind == "aggregate":
        aggregate_names = {"SUM", "COUNT", "AVG", "MIN", "MAX"}
        target_upper = target.upper()
        if target_upper not in aggregate_names:
            return None, f"unsupported_invariant:{kind}:{value}"
        for node in statement.walk():
            if isinstance(node, exp.AggFunc):
                try:
                    if node.sql_name().upper() == target_upper:
                        return True, None
                except AttributeError:
                    if type(node).__name__.upper() == target_upper:
                        return True, None
        return False, None

    if kind == "order_by":
        expected = _parse_order_by_invariant(value)
        if expected is None:
            return None, f"unsupported_invariant:order_by:{value}"
        order = statement.args.get("order")
        if order is None:
            return False, None
        actual = order.expressions
        if len(actual) != len(expected):
            return False, None
        for ordered, (expected_column, expected_direction) in zip(actual, expected):
            if not isinstance(ordered.this, exp.Column):
                return None, "unsupported_order_by_expression"
            actual_column = _normalize_identifier(ordered.this.name)
            if actual_column != expected_column:
                return False, None
            actual_direction = "DESC" if ordered.args.get("desc") else "ASC"
            if actual_direction != expected_direction:
                return False, None
        return True, None

    if kind == "top":
        if not value.isdigit() or int(value) < 1:
            return None, f"unsupported_invariant:top:{value}"
        limit = statement.args.get("limit")
        if limit is None:
            return False, None
        expression = limit.args.get("expression")
        if not isinstance(expression, exp.Literal) or expression.is_string:
            return None, "unsupported_top_expression"
        options = limit.args.get("limit_options")
        if options is not None and (
            options.args.get("with_ties") or options.args.get("percent")
        ):
            return False, None
        if not expression.is_int or int(expression.this) != int(value):
            return False, None
        return True, None

    return None, f"unsupported_invariant:{kind}:{value}"


def _parse_order_by_invariant(value: str) -> list[tuple[str, str]] | None:
    """Parse the deliberately narrow direct-column ordering grammar."""

    specifications = [item.strip() for item in value.split(",")]
    if not specifications or any(not item for item in specifications):
        return None
    parsed = []
    for specification in specifications:
        tokens = specification.split()
        if len(tokens) == 1:
            column, direction = tokens[0], "ASC"
        elif len(tokens) == 2 and tokens[1].upper() in {"ASC", "DESC"}:
            column, direction = tokens
            direction = direction.upper()
        else:
            return None
        normalized_column = _normalize_identifier(column.split(".")[-1])
        if not normalized_column or any(not part for part in column.split(".")):
            return None
        parsed.append((normalized_column, direction))
    return parsed


def _normalize_identifier(value: str) -> str:
    return value.strip().strip("[]").replace("[", "").replace("]", "").lower()


def _result_parts(result: QueryResult | Mapping[str, Any]) -> dict[str, Any] | None:
    if isinstance(result, QueryResult):
        return {
            "columns": result.columns,
            "rows": result.rows,
            "row_count": result.row_count,
            "truncated": result.truncated,
        }
    if not isinstance(result, Mapping):
        return None
    columns = result.get("columns")
    rows = result.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        return None
    return {
        "columns": columns,
        "rows": rows,
        "row_count": result.get("row_count", len(rows)),
        "truncated": bool(result.get("truncated", False)),
    }


def _compare_result(
    case: Mapping[str, Any],
    actual: dict[str, Any],
    tolerance: int | float | None,
) -> tuple[bool | None, list[str]]:
    reasons: list[str] = []
    expected_columns = case.get("expected_columns")
    expected_rows = case.get("expected_rows")
    expected_row_count = case.get("expected_row_count")

    if expected_columns is not None:
        if not isinstance(expected_columns, list) or not all(
            isinstance(column, str) for column in expected_columns
        ):
            return None, ["unsupported_expected_columns"]
        actual_columns = actual["columns"]
        if len(actual_columns) != len(expected_columns):
            reasons.append("column_count_mismatch")
        elif any(
            _normalize_column(actual_name) != _normalize_column(expected_name)
            for actual_name, expected_name in zip(actual_columns, expected_columns)
        ):
            reasons.append("column_order_or_name_mismatch")

    if expected_row_count is not None:
        if actual["row_count"] != expected_row_count:
            reasons.append("row_count_mismatch")

    if expected_rows is not None:
        if not isinstance(expected_rows, list):
            return None, ["unsupported_expected_rows"]
        invalid_number = _contains_invalid_number(expected_rows) or _contains_invalid_number(actual["rows"])
        if invalid_number:
            return None, ["non_finite_numeric_value"]
        rows_match = _rows_match(
            expected_rows,
            actual["rows"],
            actual["columns"],
            bool(case.get("order_sensitive")),
            tolerance,
        )
        if rows_match is None:
            return None, ["unsupported_row_value_type"]
        if not rows_match:
            reasons.append("row_values_mismatch")

    return not reasons, reasons


def _normalize_column(value: Any) -> str:
    return str(value).strip().strip("[]").lower()


def _rows_match(
    expected_rows: list[Any],
    actual_rows: list[Any],
    columns: list[str],
    order_sensitive: bool,
    tolerance: int | float | None,
) -> bool | None:
    expected_signatures = [_row_signature(row, columns) for row in expected_rows]
    actual_signatures = [_row_signature(row, columns) for row in actual_rows]
    if any(signature is None for signature in expected_signatures + actual_signatures):
        return None
    if order_sensitive:
        return all(
            _values_equal(expected, actual, tolerance)
            for expected, actual in zip(expected_signatures, actual_signatures)
        ) and len(expected_signatures) == len(actual_signatures)

    unmatched = list(actual_signatures)
    for expected in expected_signatures:
        for index, actual in enumerate(unmatched):
            if _values_equal(expected, actual, tolerance):
                unmatched.pop(index)
                break
        else:
            return False
    return not unmatched


def _row_signature(row: Any, columns: list[str]) -> tuple[Any, ...] | None:
    if isinstance(row, Mapping):
        normalized = {
            _normalize_column(key): _decode_ground_truth_value(value)
            for key, value in row.items()
        }
        if columns:
            return tuple(normalized.get(_normalize_column(column)) for column in columns)
        return tuple(sorted(normalized.items()))
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
        return tuple(_decode_ground_truth_value(value) for value in row)
    return None


def _decode_ground_truth_value(value: Any) -> Any:
    """Decode the explicit JSON scalar representation used by ground truth."""

    if not isinstance(value, Mapping) or set(value) != {"__type__", "value"}:
        return value

    value_type = value.get("__type__")
    raw_value = value.get("value")
    if not isinstance(raw_value, str):
        return value
    try:
        if value_type == "decimal":
            decoded = Decimal(raw_value)
            return decoded if decoded.is_finite() else value
        if value_type == "date":
            return date.fromisoformat(raw_value)
        if value_type == "datetime":
            return datetime.fromisoformat(raw_value)
    except (InvalidOperation, ValueError):
        return value
    return value


def _contains_invalid_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (float, Decimal)):
        try:
            return not math.isfinite(float(value))
        except (OverflowError, ValueError):
            return True
    if isinstance(value, Mapping):
        return any(_contains_invalid_number(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_invalid_number(item) for item in value)
    return False


def _values_equal(left: Any, right: Any, tolerance: int | float | None) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if _is_numeric(left) and _is_numeric(right):
        try:
            left_decimal = Decimal(str(left))
            right_decimal = Decimal(str(right))
            if tolerance is None:
                return left_decimal == right_decimal
            return abs(left_decimal - right_decimal) <= Decimal(str(tolerance))
        except (InvalidOperation, ValueError):
            return False
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _values_equal(item_left, item_right, tolerance)
            for item_left, item_right in zip(left, right)
        )
    if isinstance(left, (datetime, date)) or isinstance(right, (datetime, date)):
        return type(left) is type(right) and left == right
    return type(left) is type(right) and left == right


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float, Decimal)) and not isinstance(value, bool)
