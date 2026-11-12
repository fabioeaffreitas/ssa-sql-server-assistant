
"""
SSA - Deterministic SQL Pattern Tests
Tests SQL business logic without calling an LLM.
"""

import pytest


SQL_CASES = [
    {
        "sql": """
            SELECT CustomerID, SUM(TotalDue) AS TotalSales
            FROM Sales.SalesOrderHeader
            GROUP BY CustomerID
        """,
        "required": ["SUM(TotalDue)", "CustomerID", "GROUP BY"],
        "forbidden": ["SUM(SubTotal)", "Sales.Customer"],
    },
    {
        "sql": """
            SELECT CustomerID, SUM(SubTotal) AS SalesBeforeTax
            FROM Sales.SalesOrderHeader
            GROUP BY CustomerID
        """,
        "required": ["SUM(SubTotal)", "CustomerID", "GROUP BY"],
        "forbidden": ["SUM(TotalDue)"],
    },
]


@pytest.mark.parametrize("case", SQL_CASES)
def test_sql_business_patterns(case):
    sql_upper = case["sql"].upper()

    for pattern in case["required"]:
        assert pattern.upper() in sql_upper

    for pattern in case["forbidden"]:
        assert pattern.upper() not in sql_upper
