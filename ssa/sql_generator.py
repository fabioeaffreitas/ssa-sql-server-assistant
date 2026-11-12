
"""
SSA - SQL Generator
Natural language to SQL Server T-SQL using Ollama and retrieved schema.
"""

import os

import ollama
from dotenv import load_dotenv

from ssa.schema_retrieval import search_schema
from ssa.sql_validator import validate_sql

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def generate_sql(question: str, n_schema_results: int = 5) -> str:
    """Generate SQL Server T-SQL using retrieved database schema."""

    # Retrieve relevant schema documents
    schema_results = search_schema(question, n_results=n_schema_results)

    schema_context = "\n\n".join(
        candidate["document"] for candidate in schema_results
    )

    prompt = f"""
You are SSA, a SQL Server AI Assistant.

Generate a SQL Server 2025 T-SQL query answering the user's question.

Rules:
Rules:
- Use ONLY the tables, views, and columns provided in the schema context.
- Do not invent table names or column names.
- Prefer current-state tables over history tables when the question asks
  for current values.
- For total sales including tax and freight, use SalesOrderHeader.TotalDue.
- For sales before tax and freight, use SalesOrderHeader.SubTotal.
- Do not use SubTotal when the question explicitly asks for total sales including tax and freight.
- When aggregating sales by customer, group by CustomerID.
- Use history tables only when the question explicitly asks about
  historical changes, previous values, or trends over time.
- Avoid unnecessary JOINs.
- Do not JOIN tables unless the requested information requires it.
- Generate SELECT queries only.
- In SQL Server, TOP must appear immediately after SELECT, e.g. SELECT TOP 10 ...
- Never place TOP at the end of a query.
- Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or TRUNCATE.
- Use SQL Server T-SQL syntax.
- Use TOP instead of LIMIT.
- Return only the SQL query.
- Do not include Markdown code fences or explanations.

DATABASE SCHEMA:
{schema_context}

USER QUESTION:
{question}

T-SQL QUERY:
"""

    client = ollama.Client(host=OLLAMA_BASE_URL)

    response = client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
    )

    sql = response["response"].strip()

    # Remove Markdown code fences if the model adds them
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[1]
        sql = sql.rsplit("```", 1)[0].strip()

    return validate_sql(sql)


if __name__ == "__main__":
    question = "Show the top 10 products with the highest list price."

    print("\nUSER QUESTION:")
    print(question)

    print("\nGENERATED SQL:")
    print(generate_sql(question))
