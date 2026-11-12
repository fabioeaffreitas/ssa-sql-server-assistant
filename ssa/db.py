
"""
SSA - Database Connection
Supports Windows Authentication and SQL Authentication.
"""

import os

import pyodbc
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    server = os.getenv("SQL_SERVER", "localhost")
    database = os.getenv("SQL_DATABASE", "AdventureWorks2025")
    driver = os.getenv(
        "SQL_DRIVER",
        "ODBC Driver 18 for SQL Server"
    )

    trusted_connection = os.getenv(
        "SQL_TRUSTED_CONNECTION", "yes"
    ).lower()

    encrypt = os.getenv("SQL_ENCRYPT", "yes")
    trust_server_certificate = os.getenv(
        "SQL_TRUST_SERVER_CERTIFICATE", "yes"
    )

    connection_string = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"Encrypt={encrypt};"
        f"TrustServerCertificate={trust_server_certificate};"
    )

    if trusted_connection in ("yes", "true", "1"):
        connection_string += "Trusted_Connection=yes;"
    else:
        username = os.getenv("SQL_USERNAME")
        password = os.getenv("SQL_PASSWORD")

        if not username or not password:
            raise ValueError(
                "SQL_USERNAME and SQL_PASSWORD are required "
                "when SQL_TRUSTED_CONNECTION=no."
            )

        connection_string += (
            f"UID={username};"
            f"PWD={password};"
        )

    return pyodbc.connect(connection_string)
