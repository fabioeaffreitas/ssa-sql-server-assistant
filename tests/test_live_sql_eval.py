import csv
import json
from pathlib import Path

import pytest

import scripts.run_live_sql_eval as evaluator
from scripts.run_live_sql_eval import timing_columns


def test_evaluation_timing_columns_are_generic_and_optional():
    response = {
        "diagnostics": {
            "timings_ms": {
                "retrieval_ms": 1.5,
                "generation_ms": 250.0,
                "validation_ms": None,
                "execution_ms": None,
                "total_ms": 251.5,
            }
        }
    }

    assert timing_columns(response) == {
        "retrieval_ms": 1.5,
        "generation_ms": 250.0,
        "validation_ms": None,
        "execution_ms": None,
        "total_ms": 251.5,
    }


def test_evaluation_timing_columns_default_to_none():
    assert timing_columns({}) == {
        "retrieval_ms": None,
        "generation_ms": None,
        "validation_ms": None,
        "execution_ms": None,
        "total_ms": None,
    }


def dataset_case(**overrides):
    case = {
        "case_id": "TEST-001",
        "question": "Question 1",
        "review_status": "verified",
        "review_notes": "reviewed",
        "expected_columns": ["value"],
        "required_sql_invariants": [],
        "forbidden_sql_invariants": [],
        "expected_rows": [[1]],
        "expected_row_count": 1,
        "order_sensitive": True,
        "numeric_tolerance": None,
    }
    case.update(overrides)
    return case


def response(value=1, success=True, timings=True):
    return {
        "sql": "SELECT value FROM dbo.Results" if success else None,
        "generation": {"success": success},
        "validation": {"success": success},
        "execution": {"success": success},
        "result": {
            "columns": ["value"],
            "rows": [[value]],
            "row_count": 1,
            "max_rows": 100,
            "truncated": False,
        } if success else None,
        "success": success,
        "error": "" if success else "failed",
        "diagnostics": {"timings_ms": {
            "retrieval_ms": 1.0,
            "generation_ms": 2.0,
            "validation_ms": 3.0,
            "execution_ms": 4.0,
            "total_ms": 10.0,
        }} if timings else {},
    }


def test_dataset_load_is_relative_to_script_not_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    cases = evaluator.load_semantic_dataset()

    assert len(cases) == 14


def test_missing_dataset_fails_before_requests(tmp_path):
    calls = []

    with pytest.raises(ValueError, match="missing"):
        evaluator.run_evaluation(
            ask_callable=lambda question: calls.append(question),
            dataset_path=tmp_path / "missing.json",
        )

    assert calls == []


def test_malformed_dataset_fails_before_requests(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text("{invalid", encoding="utf-8")
    calls = []

    with pytest.raises(ValueError, match="malformed"):
        evaluator.run_evaluation(
            ask_callable=lambda question: calls.append(question),
            dataset_path=path,
        )
    assert calls == []


def test_dataset_identity_version_duplicates_and_question_mismatch_are_rejected(tmp_path):
    original = json.loads(evaluator.DATASET_PATH.read_text(encoding="utf-8"))
    mutations = [
        ("schema_version", 2, "version"),
        ("dataset_id", "wrong", "identity"),
    ]
    for field, value, expected in mutations:
        data = json.loads(json.dumps(original))
        data[field] = value
        path = tmp_path / f"{field}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match=expected):
            evaluator.load_semantic_dataset(path)

    duplicate = json.loads(json.dumps(original))
    duplicate["cases"][1]["case_id"] = duplicate["cases"][0]["case_id"]
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        evaluator.load_semantic_dataset(duplicate_path)

    mismatch = json.loads(json.dumps(original))
    mismatch["cases"][0]["question"] = "different"
    mismatch_path = tmp_path / "mismatch.json"
    mismatch_path.write_text(json.dumps(mismatch), encoding="utf-8")
    with pytest.raises(ValueError, match="questions"):
        evaluator.load_semantic_dataset(mismatch_path)


def test_verified_cases_are_evaluated_and_remaining_cases_are_not_evaluated():
    cases = evaluator.load_semantic_dataset()
    rows = [evaluator.evaluate_response(case["question"], response(), cases) for case in cases]

    assert len(rows) == 14
    assert sum(row["semantic_status"] == "fail" for row in rows) == 2
    assert sum(row["semantic_status"] == "not_evaluated" for row in rows) == 12
    assert all(row["pipeline_success"] is True for row in rows)


def test_verified_matching_case_passes():
    case = dataset_case()
    row = evaluator.evaluate_response(
        "Question 1",
        response(),
        (case,),
    )

    assert row["semantic_status"] == "pass"
    assert row["semantic_pass"] is True
    assert row["pipeline_success"] is True


def test_verified_incorrect_case_fails():
    case = dataset_case()
    row = evaluator.evaluate_response("Question 1", response(value=2), (case,))

    assert row["semantic_status"] == "fail"
    assert row["semantic_pass"] is False


def test_pipeline_failure_remains_separate():
    case = dataset_case()
    failed_response = response(success=False)
    failed_response["validation"] = None
    row = evaluator.evaluate_response("Question 1", failed_response, (case,))

    assert row["pipeline_success"] is False
    assert row["semantic_status"] == "not_evaluated"
    assert row["semantic_pass"] is None


def test_csv_preserves_historical_columns_and_appends_semantic_columns(tmp_path):
    case = dataset_case()
    row = evaluator.evaluate_response("Question 1", response(), (case,))
    path = tmp_path / "report.csv"
    evaluator.write_report([row], path)

    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    assert header[:len(evaluator.HISTORICAL_COLUMNS)] == list(evaluator.HISTORICAL_COLUMNS)
    assert header[len(evaluator.HISTORICAL_COLUMNS):] == list(evaluator.SEMANTIC_COLUMNS)


def test_missing_timings_and_none_values_serialize_as_empty_fields(tmp_path):
    case = dataset_case(review_status="review_required")
    row = evaluator.evaluate_response("Question 1", response(timings=False), (case,))
    path = tmp_path / "report.csv"
    evaluator.write_report([row], path)

    with path.open(encoding="utf-8-sig", newline="") as handle:
        record = next(csv.DictReader(handle))
    assert record["retrieval_ms"] == ""
    assert record["sql_structure_pass"] == ""
    assert record["semantic_pass"] == ""


def test_multiple_failure_reasons_use_stable_delimiter():
    case = dataset_case(expected_rows=[[1], [2]], expected_row_count=2)
    row = evaluator.evaluate_response("Question 1", response(value=3), (case,))

    assert row["semantic_failure_reason"] == "row_count_mismatch;row_values_mismatch"


def test_summary_zero_eligible_cases_reports_na():
    rows = [
        evaluator.evaluate_response("Question 1", response(), (dataset_case(review_status="review_required"),))
    ]
    metrics = evaluator.summary_metrics(rows)

    assert metrics["semantic_pass_count"] == 0
    assert metrics["semantic_fail_count"] == 0
    assert metrics["semantic_not_evaluated_count"] == 1
    assert metrics["semantic_accuracy"] is None


def test_summary_mixed_semantic_statuses():
    cases = (
        dataset_case(),
        dataset_case(case_id="TEST-002", question="Question 2", review_status="review_required"),
        dataset_case(case_id="TEST-003", question="Question 3", expected_rows=[[2]]),
    )
    rows = [
        evaluator.evaluate_response("Question 1", response(), cases),
        evaluator.evaluate_response("Question 2", response(), cases),
        evaluator.evaluate_response("Question 3", response(), cases),
    ]
    metrics = evaluator.summary_metrics(rows)

    assert metrics["semantic_pass_count"] == 1
    assert metrics["semantic_fail_count"] == 1
    assert metrics["semantic_not_evaluated_count"] == 1
    assert metrics["semantic_accuracy"] == 0.5


def test_semantic_scoring_uses_synthetic_response_without_external_calls():
    calls = []
    case = dataset_case()

    row = evaluator.evaluate_response(
        "Question 1",
        response(),
        (case,),
    )
    calls.append(row["semantic_status"])

    assert calls == ["pass"]
