import pytest

from ssa.sql_validator import validate_sql, SQLValidationError


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM Sales.Customer",
        "UPDATE Sales.Customer SET TerritoryID=1",
        "INSERT INTO Sales.Customer DEFAULT VALUES",
        "DROP TABLE Sales.Customer",
        "SELECT * INTO dbo.TestTable FROM Production.Product",
        "SELECT 1; DELETE FROM Sales.Customer",
        "EXEC sp_executesql N'SELECT 1'",
    ],
)
def test_rejects_unsafe_sql(sql):
    with pytest.raises(SQLValidationError):
        validate_sql(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "SELECT TOP 10 ProductID FROM Production.Product",
    ],
)
def test_accepts_valid_select(sql):
    assert validate_sql(sql) == sql