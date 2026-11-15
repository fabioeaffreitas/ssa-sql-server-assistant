
"""
SSA - SQL Generator
Natural language to SQL Server T-SQL using Ollama and retrieved schema.
"""

import os
import re
import socket
from collections.abc import Mapping

import ollama
from dotenv import load_dotenv

from ssa.db import ConfigurationError
from ssa.schema_retrieval import search_schema
from ssa.sql_validator import validate_sql

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_REQUEST_TIMEOUT = 300


class OllamaConnectionError(RuntimeError):
    """Raised when Ollama cannot be reached."""


class OllamaGenerationTimeoutError(RuntimeError):
    """Raised when Ollama does not finish generation in time."""


class OllamaGenerationError(RuntimeError):
    """Raised when Ollama returns an unusable generation response."""


def get_ollama_request_timeout(request_timeout: int | None = None) -> int:
    """Return a positive Ollama request timeout in seconds."""

    value = request_timeout
    if value is None:
        value = os.getenv(
            "OLLAMA_REQUEST_TIMEOUT",
            str(DEFAULT_OLLAMA_REQUEST_TIMEOUT),
        )
    try:
        timeout = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            "OLLAMA_REQUEST_TIMEOUT must be a positive integer."
        ) from exc
    if timeout < 1:
        raise ConfigurationError("OLLAMA_REQUEST_TIMEOUT must be a positive integer.")
    return timeout


def _ollama_error(exc: Exception) -> RuntimeError:
    """Translate client failures without returning raw client details."""

    error_name = type(exc).__name__.lower()
    if isinstance(exc, (TimeoutError, socket.timeout)) or "timeout" in error_name:
        return OllamaGenerationTimeoutError(
            "Ollama generation timed out. Check Ollama or increase "
            "OLLAMA_REQUEST_TIMEOUT."
        )
    if isinstance(exc, ConnectionError) or any(
        name in error_name for name in ("connect", "network", "connection")
    ):
        return OllamaConnectionError(
            "Ollama is unavailable. Start Ollama and verify the local service."
        )
    return OllamaGenerationError(
        "Ollama could not generate SQL. Verify the configured model is available."
    )


def _response_value(response: object, key: str, default: object = None) -> object:
    """Read fields from Ollama mappings and typed response models."""

    if isinstance(response, Mapping):
        return response.get(key, default)

    getter = getattr(response, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            return getter(key)

    try:
        return response[key]  # type: ignore[index]
    except (KeyError, TypeError, AttributeError):
        return getattr(response, key, default)


def token_usage_from_response(response: object) -> dict:
    """Return the token counts reported by Ollama for one generation."""

    try:
        prompt_tokens = int(_response_value(response, "prompt_eval_count") or 0)
        completion_tokens = int(_response_value(response, "eval_count") or 0)
    except (TypeError, ValueError) as exc:
        raise OllamaGenerationError(
            "Ollama returned invalid token usage metadata."
        ) from exc

    return {
        "model": _response_value(response, "model") or OLLAMA_MODEL,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def repair_terminal_top(sql: str) -> str:
    """Move a misplaced terminal TOP N clause into the SELECT statement."""

    match = re.search(
        r"\s+TOP\s+(\d+)\s*(;?)\s*$",
        sql,
        re.IGNORECASE,
    )

    if not match:
        return sql

    # Do not modify queries that already contain TOP.
    if re.search(r"\bTOP\s+\d+\b", sql[:match.start()], re.IGNORECASE):
        return sql

    top_clause = f"TOP {match.group(1)}"
    trailing_semicolon = match.group(2)

    sql_without_terminal_top = sql[:match.start()].rstrip()

    select_match = re.match(
        r"(\s*SELECT\s+(?:DISTINCT\s+)?)(.*)",
        sql_without_terminal_top,
        re.IGNORECASE | re.DOTALL,
    )

    if not select_match:
        return sql

    return (
        select_match.group(1)
        + top_clause
        + " "
        + select_match.group(2)
        + trailing_semicolon
    )


def generate_sql(
    question: str,
    n_schema_results: int = 5,
    return_usage: bool = False,
) -> str | tuple[str, dict]:
    """Generate SQL Server T-SQL using retrieved database schema."""

    request_timeout = get_ollama_request_timeout()

    # Retrieve relevant schema documents
    schema_results = search_schema(question, n_results=n_schema_results)

    schema_context = "\n\n".join(
        candidate["document"] for candidate in schema_results
    )

    prompt = f"""
You are SSA, a SQL Server AI Assistant.

Generate a SQL Server 2025 T-SQL query answering the user's question.

Rules:
- Use ONLY the tables, views, and columns provided in the schema context.
- Do not invent table names or column names.
- Prefer current-state tables over history tables when the question asks
  for current values.
- For total sales including tax and freight, use SalesOrderHeader.TotalDue.
- For sales before tax and freight, use SalesOrderHeader.SubTotal.
- Do not use SubTotal when the question explicitly asks for total sales including tax and freight.
- When aggregating sales by territory, use Sales.SalesTerritory joined directly to Sales.SalesOrderHeader on TerritoryID. Use SalesOrderHeader.TotalDue for total sales. Do not join SalesOrderDetail unless the question explicitly requires order-line information.
- When aggregating sales by customer, group by CustomerID.
- When the user explicitly requests customer names, resolve customer names as follows:
- Join Sales.Customer C ON C.CustomerID = SalesOrderHeader.CustomerID.
- CUSTOMER AGGREGATION RULES:
- For customer sales aggregation WITHOUT customer names:
  use Sales.SalesOrderHeader.CustomerID directly.
  Select CustomerID and aggregate TotalDue or SubTotal as requested.
  GROUP BY CustomerID.
  Do NOT join Sales.Customer, Person.Person, or Sales.Store.
- Only when customer names are explicitly requested:
  join Sales.Customer C ON C.CustomerID = SOH.CustomerID.
  LEFT JOIN Person.Person P ON C.PersonID = P.BusinessEntityID.
  LEFT JOIN Sales.Store S ON C.StoreID = S.BusinessEntityID.
  Use COALESCE(P.FirstName + N' ' + P.LastName, S.Name)
  as CustomerName.
  Include C.CustomerID and the CustomerName expression
  in GROUP BY when aggregating sales.
- Never generate customer-name joins unless the user explicitly
  requests customer names or other customer attributes.
- Do not include customer names unless explicitly requested.
- Use history tables only when the question explicitly asks about
  historical changes, previous values, or trends over time.
- Avoid unnecessary JOINs.
- Use the table that directly contains the requested columns whenever possible.
- Do not JOIN Sales.Customer when CustomerID is already available
  in Sales.SalesOrderHeader and no customer attributes are requested.
- When aggregating sales by customer, use SalesOrderHeader.CustomerID
  directly unless customer attributes are explicitly requested.
- Do not JOIN tables unless the requested information requires it.
- EXAMPLE: For "Show the top 10 customers by total sales",
  generate:
  SELECT TOP 10 SOH.CustomerID, SUM(SOH.TotalDue) AS TotalSales
  FROM Sales.SalesOrderHeader SOH
  GROUP BY SOH.CustomerID
  ORDER BY TotalSales DESC;
  Do not add customer-name columns or joins.
- Generate SELECT queries only.
- In SQL Server, TOP must appear immediately after SELECT, e.g. SELECT TOP 10 ...
- Never place TOP at the end of a query.
- Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or TRUNCATE.
- Use SQL Server T-SQL syntax.
- Use TOP instead of LIMIT.
- Return only the SQL query.
- Do not include Markdown code fences or explanations.
- Always use parentheses for SQL aggregate functions: SUM(column), COUNT(*), AVG(column), MIN(column), and MAX(column). Never write an aggregate function without parentheses.
- For product-level sales aggregation, use Sales.SalesOrderDetail.LineTotal grouped by ProductID and product name. Join Production.Product on ProductID when product names are requested. Do not use SalesOrderHeader.TotalDue for product-level sales.
- EXAMPLE: For "Show the top 5 products by total sales",
  generate:
  SELECT TOP 5 p.ProductID, p.Name, SUM(sod.LineTotal) AS TotalSales
  FROM Sales.SalesOrderDetail sod
  JOIN Production.Product p ON sod.ProductID = p.ProductID
  GROUP BY p.ProductID, p.Name
  ORDER BY TotalSales DESC;
- Use exactly the aliases defined in FROM and JOIN.
- Never reference an alias that has not been defined.


DATABASE SCHEMA:
{schema_context}

USER QUESTION:
{question}

T-SQL QUERY:
"""

    try:
        client = ollama.Client(host=OLLAMA_BASE_URL, timeout=request_timeout)
        response = client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
        )
    except (OllamaConnectionError, OllamaGenerationTimeoutError, OllamaGenerationError):
        raise
    except Exception as exc:
        raise _ollama_error(exc) from exc

    response_text = _response_value(response, "response")
    if not isinstance(response_text, str):
        raise OllamaGenerationError("Ollama returned an invalid generation response.")

    sql = response_text.strip()
    if not sql:
        raise OllamaGenerationError("Ollama returned an empty SQL response.")

    # Remove Markdown code fences if the model adds them
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[1]
        sql = sql.rsplit("```", 1)[0].strip()

    # Repair misplaced terminal TOP N before SQL validation
    sql = repair_terminal_top(sql)

    validated_sql = validate_sql(sql)
    if return_usage:
        return validated_sql, token_usage_from_response(response)
    return validated_sql


def generate_sql_with_usage(question: str, n_schema_results: int = 5) -> tuple[str, dict]:
    """Generate SQL and return Ollama's token accounting for the request."""
    return generate_sql(question, n_schema_results, return_usage=True)


if __name__ == "__main__":
    question = "Show the top 10 products with the highest list price."

    print("\nUSER QUESTION:")
    print(question)

    print("\nGENERATED SQL:")
    print(generate_sql(question))
