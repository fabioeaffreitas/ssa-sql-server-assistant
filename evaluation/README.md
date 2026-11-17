# Semantic evaluation dataset

`semantic_cases.json` is a versioned review register for the existing 14-question AdventureWorks2025 live evaluation set.

The current schema version is `1`, and the dataset identifier is `adventureworks2025-semantic-v1`. Case IDs are stable from `AW2025-001` through `AW2025-014`, and the question wording and order are copied from `scripts/run_live_sql_eval.py`.

Review statuses are:

- `review_required`: ground truth is not complete enough for semantic scoring.
- `verified`: reviewed expected structure and/or results are sufficient for meaningful deterministic scoring.
- `excluded`: intentionally omitted from semantic scoring with a documented reason.

The dataset now contains two reviewed cases: `AW2025-004` and `AW2025-012`. The remaining twelve cases are still `review_required`.

`null` means the expected value is unknown and must not be interpreted as an expected empty result. An empty array is reserved for a reviewed criterion that is intentionally empty, such as no required invariants; it is not a substitute for unknown expected columns or rows.

Execution success is not semantic correctness. A query can generate, validate, and execute successfully while answering the question incorrectly. A case becomes `verified` only after the business definition, expected columns, SQL invariants, result values, row-count rules, ordering policy, and numeric comparison policy have been independently reviewed.

Ambiguous questions must be clarified before semantic scoring. Examples include the sales measure, customer identity, date boundaries, tie ordering, null semantics, and whether “month” means calendar month or month-year.

The live evaluation CSV appends `case_id`, `review_status`, `pipeline_success`, `sql_structure_pass`, `result_pass`, `semantic_status`, `semantic_pass`, and `semantic_failure_reason` after the historical pipeline and timing columns. Multiple failure reasons are joined with `;`, and unavailable values are written as empty CSV fields.

`review_required` and `excluded` cases produce `not_evaluated`; they are not counted as semantic failures. Semantic accuracy uses only evaluated `pass` and `fail` cases as its denominator. If no verified cases are eligible, semantic accuracy is reported as `N/A`.

The verified pilot expectations come from the reviewed capture taken at `2026-11-17T10:39:19.497643+00:00` against `AdventureWorks2025` on SQL Server `17.0.1000.7`. `AW2025-004` expects ten products ordered by descending `ListPrice` with ascending `ProductID` tie-breaking, including zero-priced and discontinued products. Its money values use explicit tagged Decimal values for exact comparison.

`AW2025-012` expects 109 customers identified by `Sales.Customer.CustomerID`, using `SalesOrderHeader.SalesOrderID` counts strictly greater than 10. Its comparison is order-insensitive and uses exact integer counts. The assistant currently limits results to 100 rows, so a truncated live response for this case is `not_evaluated`, never a semantic pass.

Verified ground truth means reviewed reference expectations exist; it does not mean the model has passed a live semantic evaluation. The existing 14/14 pipeline-success result remains a reliability metric and is not semantic accuracy. Expected results must be reverified if the AdventureWorks database contents, schema, or business definitions change.

## Running the evaluator

Run the complete fixed 14-case evaluation with:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_live_sql_eval
```

For a lower-cost targeted run, select one or more stable case IDs:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_live_sql_eval --case-id AW2025-004
.\.venv\Scripts\python.exe -m scripts.run_live_sql_eval --case-id AW2025-004 --case-id AW2025-012
```

The runner validates the complete 14-case dataset before making any assistant request. Selected cases are executed in canonical dataset order, regardless of command-line order. Invalid, duplicate, empty, or unknown IDs fail before evaluation begins.

Targeted summaries report only the selected cases and identify the run as targeted; they must not be interpreted as full-suite accuracy. `AW2025-004` is the first eligible live semantic pilot. `AW2025-012` has 109 reviewed expected rows, while the active assistant result limit is 100, so truncated responses remain `not_evaluated` until an evaluation-only row-limit mechanism is available.

## SQL structural invariants

The comparator supports these restricted structural forms in addition to table and aggregate checks:

- `order_by:ListPrice DESC,ProductID ASC`
- `top:10`

`order_by` checks the outermost query only, requires the exact expression count and priority, accepts direct columns with normalized case, brackets, and safe qualifiers, and treats omitted direction as ascending. Positional references, output aliases, functions, arbitrary expressions, and unsupported invariant syntax are not treated as equivalent; they produce `not_evaluated` where equivalence cannot be established.

`top` requires a positive integer literal on the outermost query. `TOP 10` and `TOP (10)` are supported. `WITH TIES`, `PERCENT`, missing TOP, wrong counts, and nested-only TOP expressions do not satisfy `top:10`. SQLGlot normalizes some `LIMIT` and `TOP` forms to the same AST; the existing SQL Server validation boundary remains responsible for dialect enforcement.

Result equality alone does not guarantee the generated SQL has the approved ordering or row-limit semantics. The historical AW2025-004 CP31.6f result passed under its former table-only invariant; the same generated SQL fails under the strengthened `order_by` requirement because it omits the `ProductID ASC` tie-breaker. No historical report was rewritten and no new live result is claimed.
