"""Run the live Text-to-SQL evaluation against local Ollama and SQL Server."""

import csv
from datetime import datetime
from pathlib import Path

from ssa.assistant import ask


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


def main():
    results = []
    project_root = Path(__file__).resolve().parent.parent
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / f"sql_live_eval_{datetime.now():%Y%m%d_%H%M%S}.csv"

    print("\nSSA LIVE TEXT-TO-SQL EVALUATION")
    print("=" * 70)

    for index, question in enumerate(EVALUATION_CASES, start=1):
        print(f"\n[{index}/{len(EVALUATION_CASES)}] {question}")
        print("-" * 70)
        response = ask(question)
        generation_ok = response.get("generation", {}).get("success", False)
        validation_ok = response.get("validation", {}).get("success", False)
        execution_ok = response.get("execution", {}).get("success", False)
        success = response.get("success", False)
        results.append({
            "question": question,
            "sql": response.get("sql", ""),
            "generation": generation_ok,
            "validation": validation_ok,
            "execution": execution_ok,
            "success": success,
            "error": response.get("error", ""),
        })
        print(
            f"Generation: {'PASS' if generation_ok else 'FAIL'} | "
            f"Validation: {'PASS' if validation_ok else 'FAIL'} | "
            f"Execution: {'PASS' if execution_ok else 'FAIL'}"
        )
        if not success:
            print(f"Error: {response.get('error')}")

    total = len(results)
    print("\n\nEVALUATION SUMMARY")
    print("=" * 70)
    print(f"Questions evaluated: {total}")
    print(f"Generation success: {sum(r['generation'] for r in results)}/{total}")
    print(f"Validation success: {sum(r['validation'] for r in results)}/{total}")
    print(f"Execution success: {sum(r['execution'] for r in results)}/{total}")
    print(f"End-to-end success: {sum(r['success'] for r in results)}/{total}")
    print(f"End-to-end reliability: {sum(r['success'] for r in results) / total:.1%}")

    fieldnames = ["question", "sql", "generation", "validation", "execution", "success", "error"]
    with report_path.open("w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV report saved to:\n{report_path}")


if __name__ == "__main__":
    main()
