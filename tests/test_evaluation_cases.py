import json
import math
import re
from pathlib import Path

from scripts.run_live_sql_eval import EVALUATION_CASES


DATASET_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "semantic_cases.json"
REQUIRED_FIELDS = {
    "case_id",
    "question",
    "review_status",
    "review_notes",
    "expected_columns",
    "required_sql_invariants",
    "forbidden_sql_invariants",
    "expected_rows",
    "expected_row_count",
    "order_sensitive",
    "numeric_tolerance",
}
ALLOWED_STATUSES = {"review_required", "verified", "excluded"}


def load_dataset():
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def test_dataset_parses_and_has_supported_identity():
    dataset = load_dataset()

    assert dataset["schema_version"] == 1
    assert dataset["dataset_id"] == "adventureworks2025-semantic-v1"
    assert dataset["dataset_id"]


def test_dataset_has_exact_case_count_and_sequential_ids():
    cases = load_dataset()["cases"]

    assert len(cases) == 14
    assert [case["case_id"] for case in cases] == [
        f"AW2025-{index:03d}" for index in range(1, 15)
    ]
    assert len({case["case_id"] for case in cases}) == 14


def test_questions_match_live_evaluator_exactly_and_in_order():
    cases = load_dataset()["cases"]

    assert [case["question"] for case in cases] == EVALUATION_CASES


def test_each_case_has_valid_schema_fields():
    for case in load_dataset()["cases"]:
        assert set(case) == REQUIRED_FIELDS
        assert case["review_status"] in ALLOWED_STATUSES
        assert isinstance(case["question"], str) and case["question"]
        assert isinstance(case["review_notes"], str) and case["review_notes"]
        assert isinstance(case["required_sql_invariants"], list)
        assert all(isinstance(item, str) for item in case["required_sql_invariants"])
        assert isinstance(case["forbidden_sql_invariants"], list)
        assert all(isinstance(item, str) for item in case["forbidden_sql_invariants"])


def test_unknown_expected_values_are_null_and_review_status_is_honest():
    for case in load_dataset()["cases"]:
        if case["review_status"] == "review_required":
            assert case["expected_columns"] is None
            assert case["expected_rows"] is None
            assert case["expected_row_count"] is None
            assert case["numeric_tolerance"] is None
            assert case["required_sql_invariants"] == []
            assert case["forbidden_sql_invariants"] == []
            assert case["review_status"] != "verified"


def test_optional_numeric_and_order_fields_are_valid():
    for case in load_dataset()["cases"]:
        row_count = case["expected_row_count"]
        tolerance = case["numeric_tolerance"]
        order_sensitive = case["order_sensitive"]

        assert row_count is None or (
            isinstance(row_count, int) and not isinstance(row_count, bool) and row_count >= 0
        )
        assert tolerance is None or (
            isinstance(tolerance, (int, float))
            and not isinstance(tolerance, bool)
            and tolerance >= 0
            and math.isfinite(tolerance)
        )
        assert order_sensitive is None or isinstance(order_sensitive, bool)


def test_no_secrets_connection_strings_or_generated_sql_are_present():
    raw = DATASET_PATH.read_text(encoding="utf-8").lower()
    forbidden_patterns = (
        "sql_password",
        "uid=",
        "pwd=",
        "trusted_connection=",
        "select ",
        "insert ",
        "update ",
        "delete ",
        "server=",
        "password",
    )

    assert not any(pattern in raw for pattern in forbidden_patterns)


def test_dataset_contains_exactly_the_five_approved_verified_cases():
    cases = load_dataset()["cases"]

    verified_ids = [case["case_id"] for case in cases if case["review_status"] == "verified"]
    assert verified_ids == [
        "AW2025-004",
        "AW2025-007",
        "AW2025-008",
        "AW2025-010",
        "AW2025-012",
    ]
    assert len(cases) == 14
    assert len(set(case["case_id"] for case in cases)) == 14
    assert len(verified_ids) == 5
    assert sum(case["review_status"] == "review_required" for case in cases) == 9
    assert sum(case["review_status"] == "excluded" for case in cases) == 0
    assert all(
        case["review_status"] == "review_required"
        for case in cases
        if case["case_id"] not in set(verified_ids)
    )


def test_first_batch_expectations_are_exact_and_order_insensitive():
    cases = {case["case_id"]: case for case in load_dataset()["cases"]}
    assert cases["AW2025-007"]["expected_columns"] == ["TerritoryID", "TerritoryName", "TotalSales"]
    assert len(cases["AW2025-007"]["expected_rows"]) == 10
    assert cases["AW2025-007"]["required_sql_invariants"] == [
        "table:Sales.SalesOrderHeader", "table:Sales.SalesTerritory", "aggregate:SUM"
    ]
    assert cases["AW2025-007"]["expected_rows"][0][2] == {
        "__type__": "decimal", "value": "18061660.3710"
    }
    assert cases["AW2025-008"]["expected_columns"] == ["OrderYear", "SalesOrderCount"]
    assert cases["AW2025-008"]["expected_rows"] == [[2022, 1692], [2023, 3830], [2024, 14244], [2025, 11699]]
    assert cases["AW2025-008"]["required_sql_invariants"] == [
        "table:Sales.SalesOrderHeader", "aggregate:COUNT"
    ]
    assert cases["AW2025-010"]["expected_columns"] == ["TotalSales"]
    assert cases["AW2025-010"]["expected_rows"] == [[
        {"__type__": "decimal", "value": "49020486.5120"}
    ]]
    assert cases["AW2025-010"]["required_sql_invariants"] == [
        "table:Sales.SalesOrderHeader", "aggregate:SUM"
    ]
    for case_id in ("AW2025-007", "AW2025-008", "AW2025-010"):
        assert cases[case_id]["forbidden_sql_invariants"] == []
        assert cases[case_id]["order_sensitive"] is False
        assert cases[case_id]["numeric_tolerance"] is None


def test_verified_cases_do_not_contain_proposal_only_fields():
    for case in load_dataset()["cases"]:
        assert not {"reference_sql", "ordering_policy", "captured_result"}.intersection(case)
