"""Run the live Text-to-SQL evaluation against local Ollama and SQL Server."""

import csv
from datetime import datetime
import json
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Callable
import argparse

from ssa.assistant import ask
from ssa.semantic_evaluation import compare_semantics


EVALUATION_CASES = [
    "Show the top 10 customers by total sales.",
    "Show total sales before tax and freight by customer.",
    "Show total sales including tax and freight by customer.",
    "Show the 10 most expensive products by list price.",
    "Show the number of sales orders per customer.",
    "Show the top 5 customers by total sales, including their names.",
    "Show total sales by sales territory.",
    "Show the number of sales orders per year.",
    "Show the top 10 products by total quantity sold.",
    "Show total sales for orders placed in 2024.",
    "Show the top 5 products by total sales.",
    "Show customers with more than 10 sales orders.",
    "Show products that have no product subcategory.",
    "Show total sales by month.",
]

TIMING_COLUMNS = (
    "retrieval_ms",
    "generation_ms",
    "validation_ms",
    "execution_ms",
    "total_ms",
)

DATASET_PATH = Path(__file__).resolve().parent.parent / "evaluation" / "semantic_cases.json"
SUPPORTED_SCHEMA_VERSION = 1
EXPECTED_DATASET_ID = "adventureworks2025-semantic-v1"
REVIEW_STATUSES = {"review_required", "verified", "excluded"}

HISTORICAL_COLUMNS = (
    "question",
    "sql",
    "generation",
    "validation",
    "execution",
    "success",
    "error",
    *TIMING_COLUMNS,
)

SEMANTIC_COLUMNS = (
    "case_id",
    "review_status",
    "pipeline_success",
    "sql_structure_pass",
    "result_pass",
    "semantic_status",
    "semantic_pass",
    "semantic_failure_reason",
)


def timing_columns(response: dict) -> dict:
    """Return generic timing fields for one evaluation response."""

    diagnostics = response.get("diagnostics") or {}
    timings = diagnostics.get("timings_ms") or {}
    return {key: timings.get(key) for key in TIMING_COLUMNS}


def load_semantic_dataset(path: Path | None = None) -> tuple[dict[str, Any], ...]:
    """Load and validate the semantic dataset before any live request."""

    dataset_path = path or DATASET_PATH
    try:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("Semantic evaluation dataset is missing.") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("Semantic evaluation dataset is malformed JSON.") from exc

    if not isinstance(dataset, dict):
        raise ValueError("Semantic evaluation dataset must be a JSON object.")
    if dataset.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ValueError("Unsupported semantic evaluation dataset schema version.")
    if dataset.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("Unexpected semantic evaluation dataset identity.")

    cases = dataset.get("cases")
    if not isinstance(cases, list) or len(cases) != len(EVALUATION_CASES):
        raise ValueError("Semantic evaluation dataset must contain exactly 14 cases.")

    if not all(isinstance(case, dict) for case in cases):
        raise ValueError("Semantic evaluation cases must be objects.")

    case_ids = [case["case_id"] for case in cases]
    if len(case_ids) != len(cases) or len(set(case_ids)) != len(case_ids):
        raise ValueError("Semantic evaluation case IDs must be unique.")

    questions = [case.get("question") for case in cases]
    if questions != EVALUATION_CASES:
        raise ValueError("Semantic evaluation questions do not match the live evaluator.")

    for case in cases:
        if case.get("review_status") not in REVIEW_STATUSES:
            raise ValueError("Semantic evaluation case has an invalid review status.")

    return tuple(cases)


def select_cases(
    cases: tuple[dict[str, Any], ...],
    requested_case_ids: Sequence[str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Select validated cases in canonical order, or return the full set."""

    if requested_case_ids is None:
        return cases

    requested = list(requested_case_ids)
    if any(not isinstance(case_id, str) or not case_id.strip() for case_id in requested):
        raise ValueError("Each --case-id value must be non-empty.")
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate --case-id values are not allowed.")

    by_id = {case["case_id"]: case for case in cases}
    unknown = [case_id for case_id in requested if case_id not in by_id]
    if unknown:
        raise ValueError(f"Unknown semantic case ID: {unknown[0]}")

    requested_set = set(requested)
    return tuple(case for case in cases if case["case_id"] in requested_set)


def scope_summary(
    cases: tuple[dict[str, Any], ...],
    requested_case_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Describe full versus targeted execution without changing CSV output."""

    selected = select_cases(cases, requested_case_ids)
    return {
        "scope": "full" if requested_case_ids is None else "targeted",
        "selected_count": len(selected),
        "canonical_count": len(cases),
        "selected_case_ids": tuple(case["case_id"] for case in selected),
    }


def evaluate_response(
    question: str,
    response: dict,
    cases: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Convert one assistant response into a backward-compatible evaluation row."""

    case = next(case for case in cases if case["question"] == question)
    pipeline_success = bool(response.get("success", False))
    generation = response.get("generation") or {}
    validation = response.get("validation") or {}
    execution = response.get("execution") or {}
    semantic = compare_semantics(
        case,
        response.get("sql"),
        response.get("result"),
        pipeline_success,
    )
    return {
        "question": question,
        "sql": response.get("sql", ""),
        "generation": generation.get("success", False),
        "validation": validation.get("success", False),
        "execution": execution.get("success", False),
        "success": pipeline_success,
        "error": response.get("error", ""),
        **timing_columns(response),
        "case_id": case["case_id"],
        "review_status": case["review_status"],
        "pipeline_success": semantic.pipeline_success,
        "sql_structure_pass": semantic.sql_structure_pass,
        "result_pass": semantic.result_pass,
        "semantic_status": semantic.status,
        "semantic_pass": semantic.semantic_pass,
        "semantic_failure_reason": ";".join(semantic.failure_reasons),
    }


def summary_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Return historical pipeline and semantic summary metrics."""

    semantic_counts = {
        status: sum(row["semantic_status"] == status for row in results)
        for status in ("pass", "fail", "not_evaluated")
    }
    evaluated = semantic_counts["pass"] + semantic_counts["fail"]
    return {
        "total": len(results),
        "generation_success": sum(bool(row["generation"]) for row in results),
        "validation_success": sum(bool(row["validation"]) for row in results),
        "execution_success": sum(bool(row["execution"]) for row in results),
        "pipeline_success": sum(bool(row["success"]) for row in results),
        "semantic_pass_count": semantic_counts["pass"],
        "semantic_fail_count": semantic_counts["fail"],
        "semantic_not_evaluated_count": semantic_counts["not_evaluated"],
        "semantic_accuracy": (
            semantic_counts["pass"] / evaluated if evaluated else None
        ),
    }


def write_report(results: list[dict[str, Any]], report_path: Path) -> None:
    """Write the historical CSV fields followed by semantic fields."""

    with report_path.open("w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[*HISTORICAL_COLUMNS, *SEMANTIC_COLUMNS],
        )
        writer.writeheader()
        writer.writerows(results)


def run_evaluation(
    ask_callable: Callable[[str], dict] = ask,
    dataset_path: Path | None = None,
    report_path: Path | None = None,
    verbose: bool = False,
    requested_case_ids: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], Path | None]:
    """Run selected cases after validating the complete authoritative dataset."""

    cases = load_semantic_dataset(dataset_path)
    selected_cases = select_cases(cases, requested_case_ids)
    results = []
    output_path = report_path

    if output_path is not None:
        output_path.parent.mkdir(exist_ok=True)

    for index, case in enumerate(selected_cases, start=1):
        question = case["question"]
        if verbose:
            print(f"\n[{index}/{len(selected_cases)}] {question}")
            print("-" * 70)
        response = ask_callable(question)
        row = evaluate_response(question, response, cases)
        results.append(row)
        if verbose:
            print(
                f"Generation: {'PASS' if row['generation'] else 'FAIL'} | "
                f"Validation: {'PASS' if row['validation'] else 'FAIL'} | "
                f"Execution: {'PASS' if row['execution'] else 'FAIL'}"
            )
            if not row["success"]:
                print(f"Error: {row['error']}")

    if output_path is not None:
        write_report(results, output_path)
    return results, output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-id",
        action="append",
        dest="case_ids",
        help="Run one selected semantic case; repeat for multiple case IDs.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / f"sql_live_eval_{datetime.now():%Y%m%d_%H%M%S}.csv"

    print("\nSSA LIVE TEXT-TO-SQL EVALUATION")
    print("=" * 70)
    try:
        cases = load_semantic_dataset()
        scope = scope_summary(cases, args.case_ids)
        results, report_path = run_evaluation(
            report_path=report_path,
            verbose=True,
            requested_case_ids=args.case_ids,
        )
    except ValueError as exc:
        parser.error(str(exc))

    metrics = summary_metrics(results)
    total = metrics["total"]
    print("\n\nEVALUATION SUMMARY")
    print("=" * 70)
    print(f"Run scope: {scope['scope']}")
    print(f"Selected cases: {', '.join(scope['selected_case_ids'])}")
    print(f"Total canonical cases: {scope['canonical_count']}")
    print(f"Questions evaluated: {total}")
    print(f"Generation success: {metrics['generation_success']}/{total}")
    print(f"Validation success: {metrics['validation_success']}/{total}")
    print(f"Execution success: {metrics['execution_success']}/{total}")
    print(f"End-to-end success: {metrics['pipeline_success']}/{total}")
    print(f"End-to-end reliability: {metrics['pipeline_success'] / total:.1%}")
    print(f"Semantic pass: {metrics['semantic_pass_count']}/{total}")
    print(f"Semantic fail: {metrics['semantic_fail_count']}/{total}")
    print(f"Semantic not evaluated: {metrics['semantic_not_evaluated_count']}/{total}")
    accuracy = metrics["semantic_accuracy"]
    print(f"Semantic accuracy: {accuracy:.1%}" if accuracy is not None else "Semantic accuracy: N/A")
    print(f"\nCSV report saved to:\n{report_path}")


if __name__ == "__main__":
    main()
