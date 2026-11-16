# Semantic evaluation dataset

`semantic_cases.json` is a versioned review register for the existing 14-question AdventureWorks2025 live evaluation set.

The current schema version is `1`, and the dataset identifier is `adventureworks2025-semantic-v1`. Case IDs are stable from `AW2025-001` through `AW2025-014`, and the question wording and order are copied from `scripts/run_live_sql_eval.py`.

Review statuses are:

- `review_required`: ground truth is not complete enough for semantic scoring.
- `verified`: reviewed expected structure and/or results are sufficient for meaningful deterministic scoring.
- `excluded`: intentionally omitted from semantic scoring with a documented reason.

All current cases are `review_required`. No case is currently verified.

`null` means the expected value is unknown and must not be interpreted as an expected empty result. An empty array is reserved for a reviewed criterion that is intentionally empty, such as no required invariants; it is not a substitute for unknown expected columns or rows.

Execution success is not semantic correctness. A query can generate, validate, and execute successfully while answering the question incorrectly. A case becomes `verified` only after the business definition, expected columns, SQL invariants, result values, row-count rules, ordering policy, and numeric comparison policy have been independently reviewed.

Ambiguous questions must be clarified before semantic scoring. Examples include the sales measure, customer identity, date boundaries, tie ordering, null semantics, and whether “month” means calendar month or month-year.

This dataset is structure only at this checkpoint. It is not an accuracy measurement and is not integrated into the live evaluation runner.
