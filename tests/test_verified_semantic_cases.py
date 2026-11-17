from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path

from ssa.contracts import QueryResult
from ssa.semantic_evaluation import compare_semantics


DATASET_PATH = Path("evaluation/semantic_cases.json")
EXPECTED_QUESTIONS = {
    "AW2025-004": "Show the 10 most expensive products by list price.",
    "AW2025-012": "Show customers with more than 10 sales orders.",
}
AW004_SQL = """SELECT TOP (10)
    p.ProductID,
    p.Name,
    p.ListPrice
FROM Production.Product AS p
ORDER BY
    p.ListPrice DESC,
    p.ProductID ASC;"""
AW012_SQL = """SELECT
    c.CustomerID,
    COUNT(soh.SalesOrderID) AS SalesOrderCount
FROM Sales.Customer AS c
INNER JOIN Sales.SalesOrderHeader AS soh
    ON soh.CustomerID = c.CustomerID
GROUP BY
    c.CustomerID
HAVING COUNT(soh.SalesOrderID) > 10
ORDER BY
    COUNT(soh.SalesOrderID) DESC,
    c.CustomerID ASC;"""


def load_cases():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return dataset, {case["case_id"]: case for case in dataset["cases"]}


def decode(value):
    if isinstance(value, dict) and value.get("__type__") == "decimal":
        return Decimal(value["value"])
    return value


def result_for(case, max_rows=200, truncated=False):
    return QueryResult(
        columns=case["expected_columns"],
        rows=[[decode(value) for value in row] for row in case["expected_rows"]],
        row_count=case["expected_row_count"],
        max_rows=max_rows,
        truncated=truncated,
    )


def test_exactly_two_cases_are_verified_and_other_cases_remain_review_required():
    dataset, cases = load_cases()

    assert dataset["schema_version"] == 1
    assert dataset["dataset_id"] == "adventureworks2025-semantic-v1"
    assert len(cases) == 14
    assert set(case_id for case_id in cases if cases[case_id]["review_status"] == "verified") == {
        "AW2025-004",
        "AW2025-012",
    }
    assert all(
        case["review_status"] == "review_required"
        for case_id, case in cases.items()
        if case_id not in {"AW2025-004", "AW2025-012"}
    )


def test_verified_case_ids_and_questions_are_stable():
    _, cases = load_cases()
    assert {case_id: cases[case_id]["question"] for case_id in EXPECTED_QUESTIONS} == EXPECTED_QUESTIONS


def test_aw004_reference_result_passes_with_exact_decimal_values():
    _, cases = load_cases()
    case = cases["AW2025-004"]

    outcome = compare_semantics(case, AW004_SQL, result_for(case), True)

    assert case["expected_columns"] == ["ProductID", "Name", "ListPrice"]
    assert case["expected_row_count"] == 10
    assert case["order_sensitive"] is True
    assert case["expected_rows"][0][2] == {
        "__type__": "decimal",
        "value": "3578.2700",
    }
    assert outcome.status == "pass"
    assert outcome.semantic_pass is True


def test_aw004_monetary_change_order_change_and_missing_row_fail():
    _, cases = load_cases()
    case = cases["AW2025-004"]

    changed_money = result_for(case)
    changed_money.rows[0][2] = Decimal("0.01")
    changed_order = result_for(case)
    changed_order.rows[0], changed_order.rows[1] = changed_order.rows[1], changed_order.rows[0]
    missing_row = result_for(case)
    missing_row.rows.pop()
    missing_row.row_count = 9

    assert compare_semantics(case, AW004_SQL, changed_money, True).status == "fail"
    assert compare_semantics(case, AW004_SQL, changed_order, True).status == "fail"
    assert compare_semantics(case, AW004_SQL, missing_row, True).status == "fail"


def test_aw012_complete_result_passes_in_any_order():
    _, cases = load_cases()
    case = cases["AW2025-012"]

    result = result_for(case)
    result.rows.reverse()
    outcome = compare_semantics(case, AW012_SQL, result, True)

    assert case["expected_columns"] == ["CustomerID", "SalesOrderCount"]
    assert case["expected_row_count"] == 109
    assert case["order_sensitive"] is False
    assert outcome.status == "pass"


def test_aw012_identifier_count_and_missing_row_fail():
    _, cases = load_cases()
    case = cases["AW2025-012"]

    changed_identifier = result_for(case)
    changed_identifier.rows[0][0] = -1
    changed_count = result_for(case)
    changed_count.rows[0][1] += 1
    missing_row = result_for(case)
    missing_row.rows.pop()
    missing_row.row_count = 108

    assert compare_semantics(case, AW012_SQL, changed_identifier, True).status == "fail"
    assert compare_semantics(case, AW012_SQL, changed_count, True).status == "fail"
    assert compare_semantics(case, AW012_SQL, missing_row, True).status == "fail"


def test_aw012_truncated_100_row_result_is_not_evaluated():
    _, cases = load_cases()
    case = cases["AW2025-012"]
    truncated = result_for(case, max_rows=100, truncated=True)
    truncated.rows = truncated.rows[:100]
    truncated.row_count = 100

    outcome = compare_semantics(case, AW012_SQL, truncated, True)

    assert outcome.status == "not_evaluated"
    assert outcome.semantic_pass is None
    assert "query_result_truncated" in outcome.failure_reasons


def test_pipeline_failure_is_not_a_semantic_failure():
    _, cases = load_cases()
    outcome = compare_semantics(cases["AW2025-004"], None, None, False)

    assert outcome.status == "not_evaluated"
    assert outcome.pipeline_success is False
    assert outcome.semantic_pass is None


def test_invalid_tagged_numeric_value_cannot_false_pass():
    _, cases = load_cases()
    case = deepcopy(cases["AW2025-004"])
    case["expected_rows"][0][2] = {"__type__": "decimal", "value": "not-a-number"}

    outcome = compare_semantics(case, AW004_SQL, result_for(cases["AW2025-004"]), True)

    assert outcome.status != "pass"
    assert outcome.semantic_pass is not True


def test_verified_cases_have_complete_rows_and_supported_invariants():
    _, cases = load_cases()
    assert len(cases["AW2025-004"]["expected_rows"]) == 10
    assert len(cases["AW2025-012"]["expected_rows"]) == 109
    assert cases["AW2025-004"]["required_sql_invariants"] == ["table:Production.Product"]
    assert cases["AW2025-012"]["required_sql_invariants"] == [
        "table:Sales.Customer",
        "table:Sales.SalesOrderHeader",
        "aggregate:COUNT",
    ]
    assert cases["AW2025-004"]["numeric_tolerance"] is None
    assert cases["AW2025-012"]["numeric_tolerance"] is None
