from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path

import pytest

import scripts.prepare_semantic_ground_truth as prepare
from ssa.contracts import QueryResult
from ssa.semantic_evaluation import compare_semantics


DATASET_PATH = Path("evaluation/semantic_cases.json")


def synthetic_capture():
    product_rows = [
        [749, "Road-150 Red, 62", {"__type__": "decimal", "value": "3578.2700"}],
        [750, "Road-150 Red, 44", {"__type__": "decimal", "value": "3578.2700"}],
    ] + [
        [index, f"Product {index}", {"__type__": "decimal", "value": f"{1000 - index}.0000"}]
        for index in range(751, 759)
    ]
    order_rows = [[11000 + index, 11 + index % 4] for index in range(109)]
    return {
        "capture_schema_version": 1,
        "captured_at_utc": "2026-11-17T10:39:19+00:00",
        "database": {
            "database_name": "AdventureWorks2025",
            "sql_server_version": "17.0.1000.7",
        },
        "cases": [
            {
                "case_id": "AW2025-004",
                "question": prepare.PILOT_QUESTIONS["AW2025-004"],
                "reference_sql": prepare.PILOT_SQL["AW2025-004"],
                "ordering_policy": "ListPrice descending, ProductID ascending tie-breaker.",
                "captured_result": {
                    "columns": ["ProductID", "Name", "ListPrice"],
                    "rows": product_rows,
                    "row_count": 10,
                    "truncated": False,
                    "top_n_complete": True,
                },
            },
            {
                "case_id": "AW2025-012",
                "question": prepare.PILOT_QUESTIONS["AW2025-012"],
                "reference_sql": prepare.PILOT_SQL["AW2025-012"],
                "ordering_policy": "Count descending, CustomerID ascending.",
                "captured_result": {
                    "columns": ["CustomerID", "SalesOrderCount"],
                    "rows": order_rows,
                    "row_count": 109,
                    "truncated": False,
                    "top_n_complete": None,
                },
            },
        ],
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def write_dataset(path):
    write_json(path, json.loads(DATASET_PATH.read_text(encoding="utf-8")))


def write_pre_promotion_dataset(path):
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    prior_notes = {
        "AW2025-007": "The sales measure, territory inclusion rules, and expected values require business review.",
        "AW2025-008": "The date column, year extraction semantics, and expected values require review.",
        "AW2025-010": "The date column, date boundaries, and sales measure require confirmation before scoring.",
    }
    for case in data["cases"]:
        if case["case_id"] in prior_notes:
            case.update({
                "review_status": "review_required",
                "review_notes": prior_notes[case["case_id"]],
                "expected_columns": None,
                "required_sql_invariants": [],
                "forbidden_sql_invariants": [],
                "expected_rows": None,
                "expected_row_count": None,
                "order_sensitive": None,
                "numeric_tolerance": None,
            })
    write_json(path, data)


def make_proposal(tmp_path):
    capture_path = tmp_path / "capture.json"
    dataset_path = tmp_path / "semantic_cases.json"
    output_dir = tmp_path / "proposals"
    write_json(capture_path, synthetic_capture())
    write_pre_promotion_dataset(dataset_path)
    output_path = prepare.prepare_proposal(capture_path, dataset_path, output_dir)
    return json.loads(output_path.read_text(encoding="utf-8")), output_path


def additional_capture():
    cases = []
    rows_by_id = {
        "AW2025-007": [[1, "Northwest", {"__type__": "decimal", "value": "12.3400"}]],
        "AW2025-008": [[2024, 5]],
        "AW2025-010": [[None]],
    }
    for case_id, rows in rows_by_id.items():
        registered = prepare.REFERENCE_CASES_BY_ID[case_id]
        cases.append({
            "case_id": case_id,
            "question": registered["question"],
            "reference_sql": registered["sql"],
            "ordering_policy": registered["ordering_policy"],
            "captured_result": {
                "columns": registered["expected_columns"],
                "rows": rows,
                "row_count": len(rows),
                "truncated": False,
                "top_n_complete": None,
            },
        })
    return {
        "capture_schema_version": 1,
        "captured_at_utc": "2026-11-17T10:39:19+00:00",
        "database": {
            "database_name": "AdventureWorks2025",
            "sql_server_version": "17.0.1000.7",
        },
        "cases": cases,
    }


def second_batch_capture():
    rows_by_id = {
        "AW2025-001": [[1, {"__type__": "decimal", "value": "1.0000"}]] * 10,
        "AW2025-009": [[1, "Widget", 2147483648]] * 10,
        "AW2025-011": [[1, "Widget", {"__type__": "decimal", "value": "1.234567"}]] * 5,
        "AW2025-014": [[2024, 1, {"__type__": "decimal", "value": "1.0000"}]],
    }
    cases = []
    for case_id, rows in rows_by_id.items():
        registered = prepare.REFERENCE_CASES_BY_ID[case_id]
        cases.append({
            "case_id": case_id,
            "question": registered["question"],
            "reference_sql": registered["sql"],
            "ordering_policy": registered["ordering_policy"],
            "captured_result": {
                "columns": registered["expected_columns"],
                "rows": rows,
                "row_count": len(rows),
                "truncated": False,
                "top_n_complete": registered["top_n"] is not None,
            },
        })
    return {
        "capture_schema_version": 1,
        "captured_at_utc": "2026-11-17T10:39:19+00:00",
        "database": {"database_name": "AdventureWorks2025", "sql_server_version": "17.0.1000.7"},
        "cases": cases,
    }


def test_valid_capture_prepares_both_pilot_expectations(tmp_path):
    original_path = tmp_path / "expected_pre_promotion_dataset.json"
    write_pre_promotion_dataset(original_path)
    original = json.loads(original_path.read_text(encoding="utf-8"))
    proposal, output_path = make_proposal(tmp_path)
    cases = {case["case_id"]: case for case in proposal["dataset"]["cases"]}

    assert output_path.name.startswith("semantic_ground_truth_proposal_capture")
    assert cases["AW2025-004"]["expected_columns"] == ["ProductID", "Name", "ListPrice"]
    assert cases["AW2025-004"]["expected_row_count"] == 10
    assert cases["AW2025-004"]["order_sensitive"] is True
    assert cases["AW2025-004"]["required_sql_invariants"] == ["table:Production.Product"]
    assert cases["AW2025-012"]["expected_columns"] == ["CustomerID", "SalesOrderCount"]
    assert cases["AW2025-012"]["expected_row_count"] == 109
    assert cases["AW2025-012"]["order_sensitive"] is False
    assert cases["AW2025-004"]["review_status"] == "review_required"
    assert cases["AW2025-012"]["review_status"] == "review_required"
    for original_case in original["cases"]:
        if original_case["case_id"] not in prepare.PILOT_IDS:
            assert cases[original_case["case_id"]] == original_case


def test_round_trip_correct_query_results_pass_actual_comparator(tmp_path):
    proposal, _ = make_proposal(tmp_path)
    cases = {case["case_id"]: case for case in proposal["dataset"]["cases"]}
    product_case = deepcopy(cases["AW2025-004"])
    product_case["review_status"] = "verified"
    actual_product_rows = [
        [749, "Road-150 Red, 62", Decimal("3578.2700")],
        [750, "Road-150 Red, 44", Decimal("3578.2700")],
    ] + [[index, f"Product {index}", Decimal(f"{1000 - index}.0000")] for index in range(751, 759)]

    outcome = compare_semantics(
        product_case,
        prepare.PILOT_SQL["AW2025-004"],
        QueryResult(
            columns=product_case["expected_columns"],
            rows=actual_product_rows,
            row_count=10,
            max_rows=100,
            truncated=False,
        ),
        True,
    )
    assert outcome.status == "pass"
    assert outcome.semantic_pass is True

    order_case = deepcopy(cases["AW2025-012"])
    order_case["review_status"] = "verified"
    order_rows = [[11000 + index, 11 + index % 4] for index in range(109)]
    order_outcome = compare_semantics(
        order_case,
        prepare.PILOT_SQL["AW2025-012"],
        QueryResult(
            columns=order_case["expected_columns"],
            rows=order_rows,
            row_count=109,
            max_rows=200,
            truncated=False,
        ),
        True,
    )
    assert order_outcome.status == "pass"


def test_incorrect_missing_and_truncated_results_do_not_pass(tmp_path):
    proposal, _ = make_proposal(tmp_path)
    case = deepcopy(next(case for case in proposal["dataset"]["cases"] if case["case_id"] == "AW2025-004"))
    case["review_status"] = "verified"
    good_rows = [[749, "Road-150 Red, 62", Decimal("3578.2700")]] + [
        [index, f"Product {index}", Decimal(f"{1000 - index}.0000")] for index in range(750, 759)
    ]
    incorrect = deepcopy(good_rows)
    incorrect[0][2] = Decimal("0.01")

    failed = compare_semantics(case, case["reference_sql"], QueryResult(columns=case["expected_columns"], rows=incorrect, row_count=10), True)
    missing = compare_semantics(case, case["reference_sql"], None, True)
    truncated = compare_semantics(case, case["reference_sql"], QueryResult(columns=case["expected_columns"], rows=good_rows, row_count=10, truncated=True), True)

    assert failed.status == "fail"
    assert missing.status == "not_evaluated"
    assert truncated.status == "not_evaluated"


def test_decimal_precision_is_preserved_as_explicit_tag(tmp_path):
    proposal, _ = make_proposal(tmp_path)
    case = next(case for case in proposal["dataset"]["cases"] if case["case_id"] == "AW2025-004")
    assert case["expected_rows"][0][2] == {
        "__type__": "decimal",
        "value": "3578.2700",
    }


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda data: data["cases"][0]["captured_result"]["rows"][0].__setitem__(2, {"__type__": "decimal", "value": "NaN"}), "Decimal"),
        (lambda data: data["cases"][0]["captured_result"]["rows"][0].__setitem__(2, {"__type__": "money", "value": "1.00"}), "Unsupported"),
        (lambda data: data.update({"database": {"database_name": "Other", "sql_server_version": "17"}}), "AdventureWorks2025"),
        (lambda data: data["cases"][0].update({"question": "wrong"}), "Question"),
        (lambda data: data["cases"][0].update({"reference_sql": "DELETE FROM x"}), "SQL"),
        (lambda data: data["cases"][0]["captured_result"].update({"row_count": 9}), "Row count"),
        (lambda data: data["cases"][0]["captured_result"].update({"truncated": True}), "truncated"),
        (lambda data: data.pop("captured_at_utc", None), "timestamp"),
        (lambda data: data["database"].pop("sql_server_version", None), "version"),
    ],
)
def test_invalid_capture_evidence_fails_closed(tmp_path, mutator, message):
    capture_data = synthetic_capture()
    mutator(capture_data)
    capture_path = tmp_path / "capture.json"
    dataset_path = tmp_path / "semantic_cases.json"
    write_json(capture_path, capture_data)
    write_pre_promotion_dataset(dataset_path)

    with pytest.raises(prepare.GroundTruthPreparationError, match=message):
        prepare.prepare_proposal(capture_path, dataset_path, tmp_path / "output")

    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda cases: cases[1].update({"case_id": cases[0]["case_id"]}), "unique"),
        (lambda cases: cases.pop(), "missing"),
        (lambda cases: cases.append({"case_id": "unexpected"}), "unexpected"),
    ],
)
def test_case_identity_is_strictly_validated(tmp_path, mutation, message):
    capture_data = synthetic_capture()
    mutation(capture_data["cases"])
    capture_path = tmp_path / "capture.json"
    dataset_path = tmp_path / "semantic_cases.json"
    write_json(capture_path, capture_data)
    write_dataset(dataset_path)

    with pytest.raises(prepare.GroundTruthPreparationError, match=message):
        prepare.prepare_proposal(capture_path, dataset_path, tmp_path / "output")


def test_unselected_cases_remain_unchanged_and_statuses_are_not_promoted(tmp_path):
    original_path = tmp_path / "expected_pre_promotion_dataset.json"
    write_pre_promotion_dataset(original_path)
    original = json.loads(original_path.read_text(encoding="utf-8"))
    proposal, _ = make_proposal(tmp_path)
    original_cases = {case["case_id"]: case for case in original["cases"]}
    proposal_cases = {case["case_id"]: case for case in proposal["dataset"]["cases"]}

    for case_id, original_case in original_cases.items():
        if case_id not in prepare.PILOT_IDS:
            assert proposal_cases[case_id] == original_case
    assert proposal_cases["AW2025-004"]["review_status"] == "review_required"
    assert proposal_cases["AW2025-012"]["review_status"] == "review_required"
    assert "SSA_ReadOnly" not in json.dumps(proposal)
    assert "SQL_PASSWORD" not in json.dumps(proposal)


def test_provenance_is_recorded_without_sensitive_paths_or_credentials(tmp_path):
    proposal, _ = make_proposal(tmp_path)
    provenance = proposal["source_capture"]

    assert provenance["file_name"] == "capture.json"
    assert provenance["database_name"] == "AdventureWorks2025"
    assert provenance["sql_server_version"] == "17.0.1000.7"
    assert str(tmp_path) not in json.dumps(proposal)


def test_capture_timestamp_is_parseable_and_proposal_is_stable(tmp_path):
    proposal, output_path = make_proposal(tmp_path)
    datetime.fromisoformat(proposal["source_capture"]["captured_at_utc"])
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8").endswith("\n")


def test_selected_additional_cases_prepare_review_only_expectations(tmp_path):
    original_dataset_path = tmp_path / "pre_promotion_semantic_cases.json"
    write_pre_promotion_dataset(original_dataset_path)
    original_dataset = json.loads(original_dataset_path.read_text(encoding="utf-8"))
    capture_path = tmp_path / "additional.json"
    dataset_path = tmp_path / "semantic_cases.json"
    output_dir = tmp_path / "proposals"
    write_json(capture_path, additional_capture())
    write_pre_promotion_dataset(dataset_path)

    output_path = prepare.prepare_proposal(
        capture_path,
        dataset_path,
        output_dir,
        case_ids=["AW2025-007", "AW2025-008", "AW2025-010"],
    )
    proposal = json.loads(output_path.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in proposal["dataset"]["cases"]}
    assert cases["AW2025-007"]["expected_columns"] == [
        "TerritoryID", "TerritoryName", "TotalSales"
    ]
    assert cases["AW2025-007"]["required_sql_invariants"] == [
        "table:Sales.SalesOrderHeader",
        "table:Sales.SalesTerritory",
        "aggregate:SUM",
    ]
    assert cases["AW2025-008"]["expected_columns"] == [
        "OrderYear", "SalesOrderCount"
    ]
    assert cases["AW2025-010"]["expected_rows"] == [[None]]
    assert all(cases[case_id]["review_status"] == "review_required" for case_id in (
        "AW2025-007", "AW2025-008", "AW2025-010"
    ))
    assert "TotalDue includes tax and freight" in cases["AW2025-007"]["review_notes"]
    assert "AW2025-004" in cases
    assert cases["AW2025-004"]["review_status"] == "verified"
    for original_case in original_dataset["cases"]:
        if original_case["case_id"] not in {"AW2025-007", "AW2025-008", "AW2025-010"}:
            assert cases[original_case["case_id"]] == original_case
    assert json.loads(dataset_path.read_text(encoding="utf-8")) == original_dataset


def test_aw2025_010_multiple_rows_are_rejected_in_proposal_evidence(tmp_path):
    capture_data = additional_capture()
    target = next(case for case in capture_data["cases"] if case["case_id"] == "AW2025-010")
    target["captured_result"]["rows"] = [[None], [None]]
    target["captured_result"]["row_count"] = 2
    capture_path = tmp_path / "additional.json"
    dataset_path = tmp_path / "semantic_cases.json"
    write_json(capture_path, capture_data)
    write_pre_promotion_dataset(dataset_path)

    with pytest.raises(prepare.GroundTruthPreparationError, match="exactly one aggregate row"):
        prepare.prepare_proposal(
            capture_path,
            dataset_path,
            tmp_path / "output",
            case_ids=["AW2025-007", "AW2025-008", "AW2025-010"],
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "case_id",
    [
        "AW2025-001", "AW2025-007", "AW2025-008", "AW2025-009",
        "AW2025-010", "AW2025-011", "AW2025-014",
    ],
)
def test_selected_verified_nonpilot_case_is_rejected_without_proposal(tmp_path, case_id):
    original_dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    capture_path = tmp_path / "additional.json"
    dataset_path = tmp_path / "semantic_cases.json"
    capture_data = additional_capture()
    capture_data["cases"].extend(
        case for case in second_batch_capture()["cases"]
        if case["case_id"] not in {item["case_id"] for item in capture_data["cases"]}
    )
    capture_data["cases"] = [
        case for case in capture_data["cases"] if case["case_id"] == case_id
    ]
    verified_case = next(
        case for case in original_dataset["cases"] if case["case_id"] == case_id
    )
    verified_case["review_status"] = "verified"
    write_json(capture_path, capture_data)
    write_json(dataset_path, original_dataset)

    with pytest.raises(prepare.GroundTruthPreparationError, match=case_id):
        prepare.prepare_proposal(
            capture_path,
            dataset_path,
            tmp_path / "output",
            case_ids=[case_id],
        )
    assert not (tmp_path / "output").exists()
    assert json.loads(dataset_path.read_text(encoding="utf-8")) == original_dataset


@pytest.mark.parametrize(
    "case_id, row_count, expected_success",
    [
        ("AW2025-001", 9, False),
        ("AW2025-001", 10, True),
        ("AW2025-001", 11, False),
        ("AW2025-009", 9, False),
        ("AW2025-009", 10, True),
        ("AW2025-009", 11, False),
        ("AW2025-011", 4, False),
        ("AW2025-011", 5, True),
        ("AW2025-011", 6, False),
    ],
)
def test_ranked_second_batch_proposal_requires_exact_top_count(
    tmp_path, case_id, row_count, expected_success
):
    capture_data = second_batch_capture()
    selected = next(case for case in capture_data["cases"] if case["case_id"] == case_id)
    selected["captured_result"]["rows"] = selected["captured_result"]["rows"][:row_count]
    selected["captured_result"]["row_count"] = row_count
    capture_data["cases"] = [selected]
    capture_path = tmp_path / "second.json"
    dataset_path = tmp_path / "semantic_cases.json"
    output_dir = tmp_path / "output"
    write_json(capture_path, capture_data)
    write_pre_promotion_dataset(dataset_path)

    if expected_success:
        output_path = prepare.prepare_proposal(
            capture_path, dataset_path, output_dir, case_ids=[case_id]
        )
        assert output_path.exists()
    else:
        with pytest.raises(prepare.GroundTruthPreparationError, match=case_id):
            prepare.prepare_proposal(
                capture_path, dataset_path, output_dir, case_ids=[case_id]
            )
        assert not output_dir.exists()


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda data: data["cases"].pop(), "missing"),
        (lambda data: data["cases"].append(data["cases"][0].copy()), "unique"),
        (lambda data: data["cases"].__setitem__(0, {**data["cases"][0], "case_id": "AW2025-999"}), "unexpected"),
    ],
)
def test_selected_additional_capture_identity_is_strict(tmp_path, mutation, message):
    capture_data = additional_capture()
    mutation(capture_data)
    capture_path = tmp_path / "additional.json"
    dataset_path = tmp_path / "semantic_cases.json"
    write_json(capture_path, capture_data)
    write_dataset(dataset_path)

    with pytest.raises(prepare.GroundTruthPreparationError, match=message):
        prepare.prepare_proposal(
            capture_path,
            dataset_path,
            tmp_path / "output",
            case_ids=["AW2025-007", "AW2025-008", "AW2025-010"],
        )


def test_selected_second_batch_cases_prepare_review_only_expectations(tmp_path):
    capture_path = tmp_path / "second.json"
    dataset_path = tmp_path / "semantic_cases.json"
    write_json(capture_path, second_batch_capture())
    write_dataset(dataset_path)
    original = json.loads(dataset_path.read_text(encoding="utf-8"))

    output_path = prepare.prepare_proposal(
        capture_path,
        dataset_path,
        tmp_path / "output",
        case_ids=["AW2025-001", "AW2025-009", "AW2025-011", "AW2025-014"],
    )
    proposal = json.loads(output_path.read_text(encoding="utf-8"))
    cases = {case["case_id"]: case for case in proposal["dataset"]["cases"]}
    assert [cases[case_id]["review_status"] for case_id in (
        "AW2025-001", "AW2025-009", "AW2025-011", "AW2025-014"
    )] == ["review_required"] * 4
    assert cases["AW2025-009"]["expected_rows"][0][2] == 2147483648
    assert cases["AW2025-011"]["expected_rows"][0][2] == {
        "__type__": "decimal", "value": "1.234567"
    }
    assert cases["AW2025-014"]["order_sensitive"] is True
    assert json.loads(dataset_path.read_text(encoding="utf-8")) == original

