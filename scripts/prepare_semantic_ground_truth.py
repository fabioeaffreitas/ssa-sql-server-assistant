"""Prepare a review-only semantic dataset proposal from captured evidence.

The proposal preserves precise JSON scalar tags used by capture artifacts,
including ``{"__type__": "decimal", "value": "3578.2700"}``. The comparator
documents and decodes these tags without converting monetary values to floats.
"""

import argparse
from copy import deepcopy
from datetime import datetime
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from typing import Any

from scripts.capture_semantic_ground_truth import REFERENCE_CASES


CAPTURE_SCHEMA_VERSION = 1
PROPOSAL_SCHEMA_VERSION = 1
EXPECTED_DATABASE = "AdventureWorks2025"
PILOT_IDS = tuple(case["case_id"] for case in REFERENCE_CASES)
PILOT_QUESTIONS = {case["case_id"]: case["question"] for case in REFERENCE_CASES}
PILOT_SQL = {case["case_id"]: case["sql"].strip() for case in REFERENCE_CASES}
PILOT_COLUMNS = {
    "AW2025-004": ["ProductID", "Name", "ListPrice"],
    "AW2025-012": ["CustomerID", "SalesOrderCount"],
}
PILOT_ROW_COUNTS = {"AW2025-004": 10, "AW2025-012": 109}


class GroundTruthPreparationError(ValueError):
    """Raised when captured evidence is incomplete or inconsistent."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise GroundTruthPreparationError(
            "Capture or semantic dataset JSON is missing or malformed."
        ) from exc
    if not isinstance(value, dict):
        raise GroundTruthPreparationError("JSON root must be an object.")
    return value


def _validate_tagged_value(value: Any) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, list):
        for item in value:
            _validate_tagged_value(item)
        return
    if isinstance(value, dict):
        if set(value) != {"__type__", "value"}:
            raise GroundTruthPreparationError("Unsupported captured value object.")
        value_type = value["__type__"]
        raw_value = value["value"]
        if value_type == "decimal":
            if not isinstance(raw_value, str):
                raise GroundTruthPreparationError("Decimal tag value must be text.")
            try:
                decimal_value = Decimal(raw_value)
            except (InvalidOperation, ValueError) as exc:
                raise GroundTruthPreparationError("Invalid Decimal tag.") from exc
            if not decimal_value.is_finite():
                raise GroundTruthPreparationError("Decimal tag must be finite.")
            return
        if value_type == "date":
            if not isinstance(raw_value, str):
                raise GroundTruthPreparationError("Date tag value must be text.")
            try:
                datetime.strptime(raw_value, "%Y-%m-%d")
            except ValueError as exc:
                raise GroundTruthPreparationError("Invalid date tag.") from exc
            return
        if value_type == "datetime":
            if not isinstance(raw_value, str):
                raise GroundTruthPreparationError("Datetime tag value must be text.")
            try:
                datetime.fromisoformat(raw_value)
            except ValueError as exc:
                raise GroundTruthPreparationError("Invalid datetime tag.") from exc
            return
        raise GroundTruthPreparationError("Unsupported captured value type.")
    raise GroundTruthPreparationError("Unsupported captured value type.")


def _validate_capture(capture: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if capture.get("capture_schema_version") != CAPTURE_SCHEMA_VERSION:
        raise GroundTruthPreparationError("Unsupported capture schema version.")
    if not isinstance(capture.get("captured_at_utc"), str):
        raise GroundTruthPreparationError("Capture timestamp is missing.")
    try:
        datetime.fromisoformat(capture["captured_at_utc"])
    except ValueError as exc:
        raise GroundTruthPreparationError("Capture timestamp is invalid.") from exc

    database = capture.get("database")
    if not isinstance(database, dict):
        raise GroundTruthPreparationError("Capture database provenance is missing.")
    if database.get("database_name") != EXPECTED_DATABASE:
        raise GroundTruthPreparationError("Capture database is not AdventureWorks2025.")
    if not isinstance(database.get("sql_server_version"), str) or not database["sql_server_version"]:
        raise GroundTruthPreparationError("SQL Server version is missing.")

    cases = capture.get("cases")
    if not isinstance(cases, list):
        raise GroundTruthPreparationError("Capture cases are missing.")
    ids = [case.get("case_id") for case in cases if isinstance(case, dict)]
    if len(ids) != len(cases) or len(set(ids)) != len(ids):
        raise GroundTruthPreparationError("Capture case IDs must be unique.")
    if set(ids) != set(PILOT_IDS):
        raise GroundTruthPreparationError("Capture contains unexpected or missing cases.")

    validated = {}
    for case in cases:
        case_id = case["case_id"]
        if case.get("question") != PILOT_QUESTIONS[case_id]:
            raise GroundTruthPreparationError(f"Question mismatch for {case_id}.")
        if case.get("reference_sql", "").strip() != PILOT_SQL[case_id]:
            raise GroundTruthPreparationError(f"Reference SQL mismatch for {case_id}.")
        result = case.get("captured_result")
        if not isinstance(result, dict):
            raise GroundTruthPreparationError(f"Result is missing for {case_id}.")
        if result.get("truncated") is not False:
            raise GroundTruthPreparationError(f"Result is truncated for {case_id}.")
        columns = result.get("columns")
        rows = result.get("rows")
        if columns != PILOT_COLUMNS[case_id] or not isinstance(rows, list):
            raise GroundTruthPreparationError(f"Output columns or rows are invalid for {case_id}.")
        if result.get("row_count") != len(rows):
            raise GroundTruthPreparationError(f"Row count mismatch for {case_id}.")
        if len(rows) != PILOT_ROW_COUNTS[case_id]:
            raise GroundTruthPreparationError(f"Unexpected row count for {case_id}.")
        if case_id == "AW2025-004" and result.get("top_n_complete") is not True:
            raise GroundTruthPreparationError("AW2025-004 top-10 result is incomplete.")
        for row in rows:
            if not isinstance(row, list) or len(row) != len(columns):
                raise GroundTruthPreparationError(f"Malformed row for {case_id}.")
            for value in row:
                _validate_tagged_value(value)
        validated[case_id] = case
    return validated


def _pilot_expectation(case: dict[str, Any]) -> dict[str, Any]:
    case_id = case["case_id"]
    result = case["captured_result"]
    required = (
        ["table:Production.Product"]
        if case_id == "AW2025-004"
        else [
            "table:Sales.Customer",
            "table:Sales.SalesOrderHeader",
            "aggregate:COUNT",
        ]
    )
    updated = deepcopy(case)
    updated.pop("captured_result", None)
    updated["reference_sql"] = case["reference_sql"]
    updated["expected_columns"] = result["columns"]
    updated["required_sql_invariants"] = required
    updated["forbidden_sql_invariants"] = []
    updated["expected_rows"] = result["rows"]
    updated["expected_row_count"] = result["row_count"]
    updated["order_sensitive"] = case_id == "AW2025-004"
    updated["numeric_tolerance"] = None
    updated["review_status"] = "review_required"
    updated["review_notes"] = (
        "Proposed from captured reference evidence; remains review_required until "
        "business and database review authorizes promotion. " + case.get("ordering_policy", "")
    )
    return updated


def prepare_proposal(
    capture_path: Path,
    dataset_path: Path,
    output_dir: Path,
) -> Path:
    """Create a review-only proposal without modifying the authoritative dataset."""

    capture = _load_json(capture_path)
    validated = _validate_capture(capture)
    dataset = _load_json(dataset_path)
    if dataset.get("schema_version") != 1 or dataset.get("dataset_id") != "adventureworks2025-semantic-v1":
        raise GroundTruthPreparationError("Unsupported semantic dataset identity or version.")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or len(cases) != 14:
        raise GroundTruthPreparationError("Authoritative dataset must contain 14 cases.")

    proposal_cases = []
    for case in cases:
        if case["case_id"] in validated:
            proposal_cases.append(_pilot_expectation(validated[case["case_id"]]))
        else:
            proposal_cases.append(deepcopy(case))

    proposal = {
        "proposal_schema_version": PROPOSAL_SCHEMA_VERSION,
        "source_capture": {
            "file_name": capture_path.name,
            "capture_schema_version": capture["capture_schema_version"],
            "captured_at_utc": capture["captured_at_utc"],
            "database_name": capture["database"]["database_name"],
            "sql_server_version": capture["database"]["sql_server_version"],
        },
        "dataset": {
            "schema_version": dataset["schema_version"],
            "dataset_id": dataset["dataset_id"],
            "cases": proposal_cases,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"semantic_ground_truth_proposal_{capture_path.stem}.json"
    output_path.write_text(json.dumps(proposal, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_path", type=Path)
    parser.add_argument("--dataset-path", type=Path, default=Path("evaluation/semantic_cases.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/semantic_ground_truth/proposals"))
    args = parser.parse_args()
    try:
        path = prepare_proposal(args.capture_path, args.dataset_path, args.output_dir)
    except GroundTruthPreparationError as exc:
        print(f"Ground-truth proposal rejected: {exc}")
        return 1
    print(f"Ground-truth proposal written to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
