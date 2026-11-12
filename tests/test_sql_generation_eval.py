
"""
SSA - SQL Generation Evaluation
Evaluates generated SQL against expected business logic.
"""

import pytest

from ssa.sql_generator import generate_sql


EVALUATION_CASES = [
    {
        "question": "Show the top 10 customers by total sales.",
        "expected": ["CustomerID", "TotalDue", "GROUP BY"],
        "forbidden": ["Sales.Customer"],
    },
    {
        "question": "Show total sales before tax and freight by customer.",
        "expected": ["CustomerID", "SubTotal", "GROUP BY"],
        "forbidden": ["TotalDue"],
    },
    {
        "question": "Show total sales including tax and freight by customer.",
        "expected": ["CustomerID", "TotalDue", "GROUP BY"],
        "forbidden": ["SubTotal"],
    },
]


@pytest.mark.parametrize("case", EVALUATION_CASES)
def test_sql_generation_business_logic(case):
    sql = generate_sql(case["question"])
    sql_upper = sql.upper()

    for expected in case["expected"]:
        assert expected.upper() in sql_upper, (
            f"Missing expected SQL element: {expected}\n"
            f"Generated SQL:\n{sql}"
        )

    for forbidden in case["forbidden"]:
        assert forbidden.upper() not in sql_upper, (
            f"Unexpected SQL element: {forbidden}\n"
            f"Generated SQL:\n{sql}"
        )
