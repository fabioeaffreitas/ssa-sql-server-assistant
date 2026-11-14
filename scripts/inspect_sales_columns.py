"""Print AdventureWorks Sales.SalesOrderHeader column metadata."""

from ssa.db import get_connection


QUERY = """
SELECT COLUMN_NAME, DATA_TYPE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = 'Sales'
AND TABLE_NAME = 'SalesOrderHeader'
ORDER BY ORDINAL_POSITION
"""


if __name__ == "__main__":
    with get_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(QUERY)
        for row in cursor.fetchall():
            print(row)
