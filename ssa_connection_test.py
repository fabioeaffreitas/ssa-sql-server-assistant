
import pyodbc

server = "localhost"  # Replace with your SSMS server name
database = "AdventureWorks2025"  # Replace with actual database name

connection_string = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    f"SERVER={server};"
    f"DATABASE={database};"
    "Trusted_Connection=yes;"
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
)

try:
    with pyodbc.connect(connection_string) as conn:
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                @@VERSION AS SQLServerVersion,
                DB_NAME() AS DatabaseName
        """)

        row = cursor.fetchone()

        print("Connection successful!")
        print("Database:", row.DatabaseName)
        print("SQL Server:", row.SQLServerVersion)

except Exception as e:
    print("Connection failed:")
    print(e)
