"""Run a manual customer-name query against the configured SQL Server."""

from pprint import pprint

from ssa.sql_executor import execute_sql


SQL = """
SELECT TOP 5
    C.CustomerID,
    COALESCE(P.FirstName + N' ' + P.LastName, S.Name) AS CustomerName,
    SUM(SOH.TotalDue) AS TotalSales
FROM Sales.SalesOrderHeader SOH
LEFT JOIN Sales.Customer C
    ON C.CustomerID = SOH.CustomerID
LEFT JOIN Person.Person P
    ON C.PersonID = P.BusinessEntityID
LEFT JOIN Sales.Store S
    ON C.StoreID = S.BusinessEntityID
GROUP BY
    C.CustomerID,
    COALESCE(P.FirstName + N' ' + P.LastName, S.Name)
ORDER BY TotalSales DESC;
"""


if __name__ == "__main__":
    pprint(execute_sql(SQL))
