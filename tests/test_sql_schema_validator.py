import pytest

from ssa.sql_schema_validator import validate_sql_schema
from ssa.sql_validator import SQLValidationError


pytestmark = pytest.mark.integration


def test_accepts_valid_column():
    sql = "SELECT TOP 10 ProductID FROM Production.Product"

    assert validate_sql_schema(sql) == sql


def test_rejects_invalid_column():
    sql = "SELECT CustomerName FROM Sales.Customer"

    with pytest.raises(SQLValidationError):
        validate_sql_schema(sql)


def test_accepts_valid_join():
    sql = """
    SELECT TOP 10 soh.SalesOrderID
    FROM Sales.SalesOrderHeader soh
    JOIN Sales.Customer c
        ON soh.CustomerID = c.CustomerID
    """

    validated_sql = validate_sql_schema(sql)

    assert "Sales.SalesOrderHeader" in validated_sql
    assert "Sales.Customer" in validated_sql
    assert "JOIN" in validated_sql.upper()
