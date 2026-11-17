import csv
import json
from decimal import Decimal
from math import inf, nan

import pytest

import scripts.run_live_sql_eval as evaluator
from ssa.semantic_evaluation import compare_semantics


def make_case(**overrides):
    case = {
        "case_id": "TEST-001",
        "question": "Synthetic question",
        "review_status": "verified",
        "review_notes": "Reviewed synthetic case",
        "expected_columns": ["id", "amount"],
        "required_sql_invariants": [],
        "forbidden_sql_invariants": [],
        "expected_rows": [[1, 10]],
        "expected_row_count": 1,
        "order_sensitive": True,
        "numeric_tolerance": None,
    }
    case.update(overrides)
    return case


def make_result(rows, columns=None, row_count=None, truncated=False):
    return {
        "columns": columns if columns is not None else ["id", "amount"],
        "rows": rows,
        "row_count": len(rows) if row_count is None else row_count,
        "max_rows": 100,
        "truncated": truncated,
    }


def compare(case, result=None, sql="SELECT id, amount FROM dbo.Results", success=True):
    return compare_semantics(case, sql, result, success)


def test_verified_correct_result_passes():
    outcome = compare(make_case(), make_result([[1, 10]]))
    assert outcome.status == "pass"
    assert outcome.semantic_pass is True


def test_verified_incorrect_result_fails():
    outcome = compare(make_case(), make_result([[1, 11]]))
    assert outcome.status == "fail"
    assert outcome.semantic_pass is False


@pytest.mark.parametrize("status", ["review_required", "excluded"])
def test_nonverified_statuses_are_not_evaluated(status):
    outcome = compare(make_case(review_status=status), make_result([[1, 10]]))
    assert outcome.status == "not_evaluated"
    assert outcome.semantic_pass is None


def test_verified_case_without_ground_truth_is_not_evaluated():
    outcome = compare(
        make_case(
            expected_columns=None,
            expected_rows=None,
            expected_row_count=None,
        ),
        make_result([[1, 10]]),
    )
    assert outcome.status == "not_evaluated"
    assert "insufficient_ground_truth" in outcome.failure_reasons


def test_pipeline_failure_is_not_a_semantic_failure():
    outcome = compare(make_case(), None, sql=None, success=False)
    assert outcome.pipeline_success is False
    assert outcome.status == "not_evaluated"
    assert outcome.semantic_pass is None


def test_missing_or_incomplete_results_never_pass():
    missing = compare(make_case(), None)
    incomplete = compare(make_case(), {"columns": ["id"]})

    assert missing.status == "not_evaluated"
    assert incomplete.status == "not_evaluated"
    assert missing.semantic_pass is None
    assert incomplete.semantic_pass is None


def test_truncated_result_is_not_a_full_result_pass():
    outcome = compare(make_case(), make_result([[1, 10]], truncated=True))
    assert outcome.status == "not_evaluated"
    assert "query_result_truncated" in outcome.failure_reasons


def test_order_sensitive_results_require_original_order():
    outcome = compare(
        make_case(expected_rows=[[1, 10], [2, 20]], expected_row_count=2),
        make_result([[2, 20], [1, 10]], row_count=2),
    )
    assert outcome.status == "fail"


def test_order_insensitive_results_can_match_in_any_order():
    outcome = compare(
        make_case(
            expected_rows=[[1, 10], [2, 20]],
            expected_row_count=2,
            order_sensitive=False,
        ),
        make_result([[2, 20], [1, 10]], row_count=2),
    )
    assert outcome.status == "pass"


def test_duplicate_missing_and_extra_rows_fail():
    cases_and_results = (
        ([[1, 10], [1, 10]], [[1, 10], [2, 20]]),
        ([[1, 10], [2, 20]], [[1, 10]]),
        ([[1, 10]], [[1, 10], [2, 20]]),
    )
    for expected, actual in cases_and_results:
        outcome = compare(
            make_case(expected_rows=expected, expected_row_count=len(expected), order_sensitive=False),
            make_result(actual, row_count=len(actual)),
        )
        assert outcome.status == "fail"


def test_null_is_not_equal_to_empty_string():
    outcome = compare(
        make_case(expected_rows=[[1, None]]),
        make_result([[1, ""]]),
    )
    assert outcome.status == "fail"


def test_numeric_tolerance_boundary_is_inclusive_and_outside_fails():
    passing = compare(
        make_case(expected_rows=[[1, 10.0]], numeric_tolerance=0.5),
        make_result([[1, 10.5]]),
    )
    failing = compare(
        make_case(expected_rows=[[1, 10.0]], numeric_tolerance=0.5),
        make_result([[1, 10.5001]]),
    )
    assert passing.status == "pass"
    assert failing.status == "fail"


@pytest.mark.parametrize("value", [nan, inf, -inf])
def test_nonfinite_values_are_not_evaluated(value):
    outcome = compare(make_case(expected_rows=[[1, value]]), make_result([[1, 10]]))
    assert outcome.status == "not_evaluated"
    assert "non_finite_numeric_value" in outcome.failure_reasons


def test_decimal_values_compare_without_float_coercion():
    outcome = compare(
        make_case(expected_rows=[[1, Decimal("10.00")]]),
        make_result([[1, Decimal("10.00")]]),
    )
    assert outcome.status == "pass"


def test_supported_structural_invariants_pass_and_fail():
    passing = compare(
        make_case(
            expected_columns=None,
            expected_rows=None,
            expected_row_count=None,
            required_sql_invariants=["table:Sales.OrderHeader", "aggregate:SUM"],
        ),
        None,
        "SELECT SUM(TotalDue) FROM Sales.OrderHeader",
    )
    failing = compare(
        make_case(
            expected_columns=None,
            expected_rows=None,
            expected_row_count=None,
            required_sql_invariants=["aggregate:COUNT"],
        ),
        None,
        "SELECT SUM(TotalDue) FROM Sales.OrderHeader",
    )
    assert passing.status == "pass"
    assert failing.status == "fail"


def test_unsupported_structural_invariant_is_not_evaluated():
    outcome = compare(
        make_case(
            expected_columns=None,
            expected_rows=None,
            expected_row_count=None,
            required_sql_invariants=["group_by:CustomerID"],
        ),
        None,
        "SELECT CustomerID FROM Sales.Customer GROUP BY CustomerID",
    )
    assert outcome.status == "not_evaluated"
    assert outcome.semantic_pass is None


def test_malformed_sql_with_structural_expectation_fails():
    outcome = compare(
        make_case(
            expected_columns=None,
            expected_rows=None,
            expected_row_count=None,
            required_sql_invariants=["table:Sales.OrderHeader"],
        ),
        None,
        "SELECT FROM",
    )
    assert outcome.status == "fail"
    assert "sql_structure_parse_failed" in outcome.failure_reasons


def test_runner_uses_synthetic_responses_without_extra_ask_calls(tmp_path):
    calls = []
    dataset = {
        "schema_version": 1,
        "dataset_id": "adventureworks2025-semantic-v1",
        "cases": [
            {
                "case_id": f"TEST-{index:03d}",
                "question": question,
                "review_status": "review_required",
                "review_notes": "synthetic",
                "expected_columns": None,
                "required_sql_invariants": [],
                "forbidden_sql_invariants": [],
                "expected_rows": None,
                "expected_row_count": None,
                "order_sensitive": None,
                "numeric_tolerance": None,
            }
            for index, question in enumerate(evaluator.EVALUATION_CASES, start=1)
        ],
    }
    dataset_path = tmp_path / "semantic_cases.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    def fake_ask(question):
        calls.append(question)
        return {
            "sql": "SELECT 1",
            "generation": {"success": True},
            "validation": {"success": True},
            "execution": {"success": True},
            "result": make_result([[1]], columns=["value"]),
            "success": True,
            "error": "",
            "diagnostics": {},
        }

    rows, _ = evaluator.run_evaluation(fake_ask, dataset_path=dataset_path)

    assert calls == evaluator.EVALUATION_CASES
    assert len(rows) == 14
    assert all(row["semantic_status"] == "not_evaluated" for row in rows)


def test_runner_dataset_validation_precedes_any_ask_call(tmp_path):
    calls = []
    dataset_path = tmp_path / "invalid.json"
    dataset_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        evaluator.run_evaluation(
            lambda question: calls.append(question),
            dataset_path=dataset_path,
        )
    assert calls == []


def test_runner_csv_contract_and_failure_reason_serialization(tmp_path):
    case = make_case(expected_rows=[[1, 10], [2, 20]], expected_row_count=2)
    row = evaluator.evaluate_response(
        "Synthetic question",
        {
            "sql": "SELECT id, amount FROM dbo.Results",
            "generation": {"success": True},
            "validation": {"success": True},
            "execution": {"success": True},
            "result": make_result([[1, 11]], row_count=1),
            "success": True,
            "error": "",
            "diagnostics": {},
        },
        (case,),
    )
    path = tmp_path / "report.csv"
    evaluator.write_report([row], path)

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        record = next(reader)
        assert reader.fieldnames == [*evaluator.HISTORICAL_COLUMNS, *evaluator.SEMANTIC_COLUMNS]

    assert record["case_id"] == "TEST-001"
    assert record["review_status"] == "verified"
    assert record["semantic_status"] == "fail"
    assert record["semantic_failure_reason"] == "row_count_mismatch;row_values_mismatch"
    assert "credentials" not in record


def test_summary_uses_only_evaluated_pass_and_fail_denominator():
    rows = [
        {"generation": True, "validation": True, "execution": True, "success": True, "semantic_status": "pass"},
        {"generation": True, "validation": True, "execution": True, "success": True, "semantic_status": "fail"},
        {"generation": True, "validation": True, "execution": True, "success": True, "semantic_status": "not_evaluated"},
        {"generation": True, "validation": False, "execution": False, "success": False, "semantic_status": "not_evaluated"},
        {"generation": True, "validation": True, "execution": False, "success": False, "semantic_status": "not_evaluated"},
    ]
    metrics = evaluator.summary_metrics(rows)

    assert metrics["semantic_pass_count"] == 1
    assert metrics["semantic_fail_count"] == 1
    assert metrics["semantic_not_evaluated_count"] == 3
    assert metrics["semantic_accuracy"] == 0.5
    assert metrics["pipeline_success"] == 3


def test_all_not_evaluated_summary_has_no_accuracy():
    metrics = evaluator.summary_metrics(
        [
            {"generation": True, "validation": True, "execution": True, "success": True, "semantic_status": "not_evaluated"}
        ]
    )
    assert metrics["semantic_accuracy"] is None
