from datetime import date, datetime
from decimal import Decimal
from math import inf, nan

import pytest

from ssa.contracts import QueryResult
from ssa.semantic_evaluation import SemanticEvaluationResult, compare_semantics


def case(**overrides):
    value = {
        "case_id": "TEST-001",
        "question": "test",
        "review_status": "verified",
        "review_notes": "reviewed",
        "expected_columns": ["id", "amount"],
        "required_sql_invariants": [],
        "forbidden_sql_invariants": [],
        "expected_rows": [[1, Decimal("10.00")]],
        "expected_row_count": 1,
        "order_sensitive": True,
        "numeric_tolerance": None,
    }
    value.update(overrides)
    if "expected_rows" in overrides and "expected_row_count" not in overrides:
        value["expected_row_count"] = len(value["expected_rows"])
    return value


def result(rows, columns=None, truncated=False, row_count=None):
    columns = ["id", "amount"] if columns is None else columns
    return QueryResult(
        columns=columns,
        rows=rows,
        row_count=len(rows) if row_count is None else row_count,
        max_rows=100,
        truncated=truncated,
    )


def test_review_required_case_is_not_evaluated():
    outcome = compare_semantics(case(review_status="review_required"), "SELECT 1", None, True)

    assert outcome.status == "not_evaluated"
    assert outcome.pipeline_success is True
    assert outcome.semantic_pass is None
    assert "case_review_required" in outcome.failure_reasons


def test_excluded_case_is_not_evaluated():
    outcome = compare_semantics(case(review_status="excluded"), "SELECT 1", None, True)

    assert outcome.status == "not_evaluated"
    assert outcome.failure_reasons == ("case_excluded",)


def test_verified_case_without_ground_truth_is_not_evaluated():
    outcome = compare_semantics(
        case(expected_columns=None, expected_rows=None, expected_row_count=None),
        "SELECT 1",
        None,
        True,
    )

    assert outcome.status == "not_evaluated"
    assert "insufficient_ground_truth" in outcome.failure_reasons


def test_correct_ordered_results_pass():
    outcome = compare_semantics(
        case(),
        "SELECT id, amount FROM dbo.Results",
        result([[1, Decimal("10.00")]]),
        True,
    )

    assert outcome == SemanticEvaluationResult("pass", True, None, True, True)


def test_incorrect_ordering_fails():
    outcome = compare_semantics(
        case(expected_rows=[[1, 10], [2, 20]]),
        "SELECT id, amount FROM dbo.Results",
        result([[2, 20], [1, 10]]),
        True,
    )

    assert outcome.status == "fail"
    assert "row_values_mismatch" in outcome.failure_reasons


def test_correct_unordered_results_pass():
    outcome = compare_semantics(
        case(expected_rows=[[1, 10], [2, 20]], order_sensitive=False),
        "SELECT id, amount FROM dbo.Results",
        result([[2, 20], [1, 10]]),
        True,
    )

    assert outcome.status == "pass"


def test_duplicate_row_mismatch_fails():
    outcome = compare_semantics(
        case(expected_rows=[[1, 10], [1, 10]], order_sensitive=False),
        "SELECT id, amount FROM dbo.Results",
        result([[1, 10], [2, 10]]),
        True,
    )

    assert outcome.status == "fail"


def test_missing_and_extra_rows_fail():
    for actual in ([[1, 10]], [[1, 10], [2, 20], [3, 30]]):
        outcome = compare_semantics(
            case(expected_rows=[[1, 10], [2, 20]], order_sensitive=False),
            "SELECT id, amount FROM dbo.Results",
            result(actual),
            True,
        )
        assert outcome.status == "fail"


def test_correct_empty_results_pass():
    outcome = compare_semantics(
        case(expected_columns=[], expected_rows=[], expected_row_count=0),
        "SELECT id, amount FROM dbo.Results",
        result([], columns=[], row_count=0),
        True,
    )

    assert outcome.status == "pass"


def test_null_differs_from_empty_string_and_zero():
    for actual_value in ["", 0]:
        outcome = compare_semantics(
            case(expected_rows=[[1, None]]),
            "SELECT id, amount FROM dbo.Results",
            result([[1, actual_value]]),
            True,
        )
        assert outcome.status == "fail"


def test_column_names_are_case_and_bracket_normalized():
    outcome = compare_semantics(
        case(expected_columns=["ID", "[AMOUNT]"]),
        "SELECT id, amount FROM dbo.Results",
        result([[1, 10]], columns=["id", "amount"]),
        True,
    )

    assert outcome.status == "pass"


def test_column_order_mismatch_fails():
    outcome = compare_semantics(
        case(expected_columns=["amount", "id"]),
        "SELECT id, amount FROM dbo.Results",
        result([[1, 10]], columns=["id", "amount"]),
        True,
    )

    assert outcome.status == "fail"
    assert "column_order_or_name_mismatch" in outcome.failure_reasons


def test_exact_numeric_comparison():
    outcome = compare_semantics(
        case(expected_rows=[[1, Decimal("10.00")]]),
        "SELECT id, amount FROM dbo.Results",
        result([[1, 10]]),
        True,
    )

    assert outcome.status == "pass"


def test_numeric_tolerance_passes_and_fails():
    passing = compare_semantics(
        case(expected_rows=[[1, 10.0]], numeric_tolerance=0.01),
        "SELECT id, amount FROM dbo.Results",
        result([[1, 10.005]]),
        True,
    )
    failing = compare_semantics(
        case(expected_rows=[[1, 10.0]], numeric_tolerance=0.01),
        "SELECT id, amount FROM dbo.Results",
        result([[1, 10.02]]),
        True,
    )

    assert passing.status == "pass"
    assert failing.status == "fail"


@pytest.mark.parametrize("value", [nan, inf, -inf])
def test_non_finite_numeric_values_are_not_evaluated(value):
    outcome = compare_semantics(
        case(expected_rows=[[1, value]]),
        "SELECT id, amount FROM dbo.Results",
        result([[1, 10]]),
        True,
    )

    assert outcome.status == "not_evaluated"
    assert "non_finite_numeric_value" in outcome.failure_reasons


def test_dates_and_datetimes_compare_by_exact_type_and_value():
    date_case = case(expected_columns=["value"], expected_rows=[[date(2024, 1, 1)]], expected_row_count=1)
    date_result = result([[date(2024, 1, 1)]], columns=["value"])
    datetime_result = result([[datetime(2024, 1, 1)]], columns=["value"])

    assert compare_semantics(date_case, "SELECT value FROM dbo.Results", date_result, True).status == "pass"
    assert compare_semantics(date_case, "SELECT value FROM dbo.Results", datetime_result, True).status == "fail"


def test_expected_row_count_is_checked_independently():
    outcome = compare_semantics(
        case(expected_rows=[[1, 10]], expected_row_count=2),
        "SELECT id, amount FROM dbo.Results",
        result([[1, 10]], row_count=1),
        True,
    )

    assert outcome.status == "fail"
    assert "row_count_mismatch" in outcome.failure_reasons


def test_truncated_result_is_not_a_full_semantic_pass():
    outcome = compare_semantics(
        case(),
        "SELECT id, amount FROM dbo.Results",
        result([[1, 10]], truncated=True),
        True,
    )

    assert outcome.status == "not_evaluated"
    assert outcome.semantic_pass is None
    assert "query_result_truncated" in outcome.failure_reasons


def test_pipeline_failure_is_preserved_and_not_semantically_failed():
    outcome = compare_semantics(case(), None, None, False)

    assert outcome.status == "not_evaluated"
    assert outcome.pipeline_success is False
    assert outcome.semantic_pass is None
    assert outcome.sql_structure_pass is None
    assert outcome.result_pass is None
    assert outcome.failure_reasons == ("pipeline_failed",)


def test_supported_table_and_aggregate_invariants_pass_and_fail():
    passing = compare_semantics(
        case(
            expected_columns=None,
            expected_rows=None,
            expected_row_count=None,
            required_sql_invariants=["table:Sales.OrderHeader", "aggregate:SUM"],
        ),
        "SELECT SUM(TotalDue) FROM Sales.OrderHeader",
        None,
        True,
    )
    failing = compare_semantics(
        case(
            expected_columns=None,
            expected_rows=None,
            expected_row_count=None,
            required_sql_invariants=["table:Sales.OrderHeader", "aggregate:COUNT"],
        ),
        "SELECT SUM(TotalDue) FROM Sales.OrderHeader",
        None,
        True,
    )

    assert passing.status == "pass"
    assert passing.sql_structure_pass is True
    assert failing.status == "fail"
    assert failing.sql_structure_pass is False


def test_malformed_sql_fails_when_structure_is_required():
    outcome = compare_semantics(
        case(
            expected_columns=None,
            expected_rows=None,
            expected_row_count=None,
            required_sql_invariants=["table:Sales.OrderHeader"],
        ),
        "SELECT FROM",
        None,
        True,
    )

    assert outcome.status == "fail"
    assert outcome.failure_reasons == ("sql_structure_parse_failed",)


def test_unsupported_structural_invariant_is_not_evaluated():
    outcome = compare_semantics(
        case(
            expected_columns=None,
            expected_rows=None,
            expected_row_count=None,
            required_sql_invariants=["group_by:CustomerID"],
        ),
        "SELECT CustomerID FROM Sales.Customer GROUP BY CustomerID",
        None,
        True,
    )

    assert outcome.status == "not_evaluated"
    assert outcome.semantic_pass is None
    assert "unsupported_invariant:group_by:CustomerID" in outcome.failure_reasons


def test_legacy_dictionary_result_is_supported():
    outcome = compare_semantics(
        case(),
        "SELECT id, amount FROM dbo.Results",
        {
            "columns": ["id", "amount"],
            "rows": [[1, 10]],
            "row_count": 1,
            "max_rows": 100,
            "truncated": False,
        },
        True,
    )

    assert outcome.status == "pass"


def structural_case(*invariants):
    return case(
        expected_columns=None,
        expected_rows=None,
        expected_row_count=None,
        required_sql_invariants=list(invariants),
    )


ORDER_INVARIANT = "order_by:ListPrice DESC,ProductID ASC"
TOP_INVARIANT = "top:10"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT TOP 10 p.ProductID, p.ListPrice FROM Production.Product p ORDER BY p.ListPrice DESC, p.ProductID ASC",
        "SELECT TOP (10) p.ProductID, p.ListPrice FROM Production.Product AS p ORDER BY p.ListPrice DESC, p.ProductID",
        "SELECT TOP (10) [p].[ProductID], [p].[ListPrice] FROM [Production].[Product] AS [p] ORDER BY [p].[ListPrice] DESC, [p].[ProductID] ASC",
        "SELECT TOP (10) P.PRODUCTID, P.LISTPRICE FROM PRODUCTION.PRODUCT P ORDER BY P.LISTPRICE DESC, P.PRODUCTID ASC",
    ],
)
def test_order_by_invariant_accepts_safe_direct_column_forms(sql):
    outcome = compare_semantics(structural_case(ORDER_INVARIANT), sql, None, True)

    assert outcome.status == "pass"
    assert outcome.sql_structure_pass is True


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT TOP 10 p.ProductID, p.ListPrice FROM Production.Product p ORDER BY p.ListPrice DESC",
        "SELECT TOP 10 p.ProductID, p.ListPrice FROM Production.Product p",
        "SELECT TOP 10 p.ProductID, p.ListPrice FROM Production.Product p ORDER BY p.ListPrice ASC, p.ProductID ASC",
        "SELECT TOP 10 p.ProductID, p.ListPrice FROM Production.Product p ORDER BY p.ListPrice DESC, p.ProductID DESC",
        "SELECT TOP 10 p.ProductID, p.ListPrice FROM Production.Product p ORDER BY p.ProductID ASC, p.ListPrice DESC",
        "SELECT TOP 10 p.ProductID, p.ListPrice FROM Production.Product p ORDER BY p.ListPrice DESC, p.ProductID ASC, p.Name ASC",
    ],
)
def test_order_by_invariant_rejects_known_ordering_violations(sql):
    outcome = compare_semantics(structural_case(ORDER_INVARIANT), sql, None, True)

    assert outcome.status == "fail"
    assert outcome.semantic_pass is False


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT TOP 10 p.ProductID, p.ListPrice FROM Production.Product p ORDER BY 3 DESC, 1 ASC",
        "SELECT TOP 10 p.ProductID, p.ListPrice AS Price FROM Production.Product p ORDER BY LOWER(p.ListPrice) DESC, p.ProductID ASC",
    ],
)
def test_order_by_invariant_fails_closed_for_unsupported_expressions(sql):
    outcome = compare_semantics(structural_case(ORDER_INVARIANT), sql, None, True)

    assert outcome.status == "not_evaluated"
    assert outcome.semantic_pass is None


def test_inner_order_by_does_not_satisfy_root_order_requirement():
    sql = """
    SELECT TOP 10 q.ProductID, q.ListPrice
    FROM (
        SELECT p.ProductID, p.ListPrice
        FROM Production.Product p
        ORDER BY p.ListPrice DESC, p.ProductID ASC
    ) AS q
    """
    outcome = compare_semantics(structural_case(ORDER_INVARIANT), sql, None, True)

    assert outcome.status == "fail"
    assert outcome.semantic_pass is False


def test_malformed_order_by_invariant_is_not_evaluated():
    outcome = compare_semantics(
        structural_case("order_by:ListPrice SIDEWAYS,ProductID ASC"),
        "SELECT ProductID FROM Production.Product",
        None,
        True,
    )

    assert outcome.status == "not_evaluated"
    assert "unsupported_invariant:order_by" in outcome.failure_reasons[0]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT TOP 10 ProductID FROM Production.Product",
        "SELECT TOP (10) ProductID FROM Production.Product",
    ],
)
def test_top_invariant_accepts_literal_top_forms(sql):
    outcome = compare_semantics(structural_case(TOP_INVARIANT), sql, None, True)

    assert outcome.status == "pass"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT ProductID FROM Production.Product",
        "SELECT TOP 9 ProductID FROM Production.Product",
        "SELECT TOP (10) WITH TIES ProductID FROM Production.Product ORDER BY ListPrice DESC",
        "SELECT TOP (10) PERCENT ProductID FROM Production.Product",
    ],
)
def test_top_invariant_rejects_missing_wrong_and_special_top_forms(sql):
    outcome = compare_semantics(structural_case(TOP_INVARIANT), sql, None, True)

    assert outcome.status == "fail"
    assert outcome.semantic_pass is False


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT TOP (@n) ProductID FROM Production.Product",
        "SELECT TOP (1 + 9) ProductID FROM Production.Product",
    ],
)
def test_top_invariant_does_not_infer_nonliteral_counts(sql):
    outcome = compare_semantics(structural_case(TOP_INVARIANT), sql, None, True)

    assert outcome.status == "not_evaluated"
    assert outcome.semantic_pass is None


def test_nested_top_does_not_satisfy_root_top_requirement():
    sql = """
    SELECT ProductID
    FROM (
        SELECT TOP 10 ProductID
        FROM Production.Product
    ) AS q
    """
    outcome = compare_semantics(structural_case(TOP_INVARIANT), sql, None, True)

    assert outcome.status == "fail"
    assert outcome.semantic_pass is False


def test_invalid_top_invariant_is_not_evaluated():
    outcome = compare_semantics(
        structural_case("top:0"),
        "SELECT TOP 10 ProductID FROM Production.Product",
        None,
        True,
    )

    assert outcome.status == "not_evaluated"
    assert "unsupported_invariant:top:0" in outcome.failure_reasons


def test_known_order_violation_takes_precedence_over_unsupported_invariant():
    outcome = compare_semantics(
        structural_case(ORDER_INVARIANT, "order_by:unsupported ???"),
        "SELECT TOP 10 ProductID FROM Production.Product ORDER BY ListPrice DESC",
        None,
        True,
    )

    assert outcome.status == "fail"
    assert outcome.semantic_pass is False
