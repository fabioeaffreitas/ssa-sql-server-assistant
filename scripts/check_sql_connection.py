"""Check the configured SQL Server connection and print server details."""

from ssa.db import get_connection


if __name__ == "__main__":
    try:
        with get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT @@VERSION AS SQLServerVersion, DB_NAME() AS DatabaseName")
            row = cursor.fetchone()
            print("Connection successful!")
            print("Database:", row.DatabaseName)
            print("SQL Server:", row.SQLServerVersion)
    except Exception as exc:
        print("Connection failed:")
        print(exc)
