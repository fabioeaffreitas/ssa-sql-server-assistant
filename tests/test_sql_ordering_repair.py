import pytest

from ssa.contracts import QueryResult
from ssa.semantic_evaluation import compare_semantics
from ssa.sql_ordering_repair import (
    OrderingRepairProposal,
    propose_top_ordering_repair,
)


KEYS = {"Catalog.Item": ("ItemKey",)}
AW_KEYS = {"Production.Product": ("ProductID",)}
BASE = "SELECT TOP 10 ItemKey, ItemName, Price FROM Catalog.Item ORDER BY Price DESC"


def propose(sql=BASE, keys=KEYS, **kwargs):
    return propose_top_ordering_repair(
        sql,
        keys,
        enabled=True,
        tie_breaking_permitted=True,
        **kwargs,
    )


def test_default_is_disabled_and_original_is_preserved():
    sql = BASE + ";"
    result = propose_top_ordering_repair(sql, KEYS)

    assert result == OrderingRepairProposal(
        status="disabled",
        reason="repair_disabled",
        original_sql=sql,
        proposed_sql=None,
    )


@pytest.mark.parametrize(
    ("enabled", "permitted", "reason"),
    [
        (True, False, "tie_breaking_not_permitted"),
        (False, True, "repair_disabled"),
        ("yes", True, "repair_disabled"),
        (True, 1, "tie_breaking_not_permitted"),
    ],
)
def test_both_explicit_boolean_permissions_are_required(enabled, permitted, reason):
    result = propose_top_ordering_repair(
        BASE,
        KEYS,
        enabled=enabled,
        tie_breaking_permitted=permitted,
    )
    assert result.status in {"disabled", "not_eligible"}
    assert result.reason == reason
    assert result.proposed_sql is None


def test_historical_aw2025_sql_proposes_product_id_tie_breaker():
    sql = (
        "SELECT TOP 10 ProductID, Name, ListPrice "
        "FROM Production.Product ORDER BY ListPrice DESC;"
    )
    result = propose(sql, AW_KEYS)

    assert result.status == "proposed"
    assert result.reason == "primary_key_appended_to_root_order"
    assert result.original_sql == sql
    assert result.proposed_sql is not None
    assert result.proposed_sql.endswith(
        "ORDER BY ListPrice DESC, ProductID ASC"
    )
    assert result.table == "Production.Product"
    assert result.primary_key_columns == ("ProductID",)
    assert result.appended_columns == ("ProductID",)


def test_qualified_alias_is_preserved_on_appended_key():
    result = propose(
        "SELECT TOP (5) i.ItemKey, i.Price FROM Catalog.Item AS i "
        "WHERE i.Price > 0 ORDER BY i.Price ASC",
    )

    assert result.status == "proposed"
    assert result.proposed_sql.endswith("ORDER BY i.Price ASC, i.ItemKey ASC")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT TOP 10 ItemKey, ItemName, Price FROM Catalog.Item ORDER BY Price DESC, ItemKey ASC",
        "SELECT TOP 10 ItemKey, ItemName, Price FROM Catalog.Item ORDER BY ItemKey DESC",
        "SELECT TOP 10 [ItemKey], [Price] FROM [Catalog].[Item] ORDER BY [Price] DESC, [ItemKey] ASC",
    ],
)
def test_existing_key_is_idempotent(sql):
    result = propose(sql)

    assert result.status == "not_eligible"
    assert result.reason == "primary_key_already_ordered"
    assert result.proposed_sql is None
    assert result.original_sql == sql


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT ItemKey FROM Catalog.Item ORDER BY Price DESC",
        "SELECT TOP 10 ItemKey FROM Catalog.Item",
        "SELECT TOP (@n) ItemKey FROM Catalog.Item ORDER BY Price DESC",
        "SELECT TOP 10 PERCENT ItemKey FROM Catalog.Item ORDER BY Price DESC",
        "SELECT TOP 10 WITH TIES ItemKey FROM Catalog.Item ORDER BY Price DESC",
        "SELECT TOP 10 ItemKey FROM Catalog.Item JOIN Catalog.Other ON 1=1 ORDER BY Price DESC",
        "SELECT TOP 10 ItemKey, COUNT(*) FROM Catalog.Item GROUP BY ItemKey ORDER BY COUNT(*) DESC",
        "SELECT DISTINCT TOP 10 ItemKey FROM Catalog.Item ORDER BY Price DESC",
        "WITH items AS (SELECT TOP 10 ItemKey FROM Catalog.Item ORDER BY Price DESC) SELECT * FROM items",
        "SELECT TOP 10 ItemKey FROM Catalog.Item WHERE ItemKey IN (SELECT ItemKey FROM Catalog.Other) ORDER BY Price DESC",
        "SELECT TOP 10 ItemKey FROM Catalog.Item ORDER BY ROW_NUMBER() OVER (ORDER BY Price)",
        "SELECT TOP 10 ItemKey FROM Catalog.Item ORDER BY RAND()",
        "SELECT TOP 10 ItemKey FROM Catalog.Item ORDER BY Price + 1 DESC",
        "SELECT TOP 10 ItemKey FROM Catalog.Item UNION SELECT TOP 10 ItemKey FROM Catalog.Other",
        "DELETE FROM Catalog.Item",
        "SELECT TOP 10 ItemKey FROM Catalog.Item ORDER BY Price DESC; SELECT 1",
    ],
)
def test_unsupported_shapes_are_never_repaired(sql):
    result = propose(sql)

    assert result.status == "not_eligible"
    assert result.proposed_sql is None
    assert result.original_sql == sql


@pytest.mark.parametrize(
    "keys",
    [
        {},
        {"Catalog.Item": ("PartA", "PartB")},
        {"Catalog.Item": []},
        {"Catalog.Item": "ItemKey"},
        {"Catalog.Item": ("ItemKey", "itemkey")},
        {"Catalog.Item": ("ItemKey",), "catalog.item": ("OtherKey",)},
    ],
)
def test_invalid_or_ambiguous_key_metadata_is_not_repaired(keys):
    result = propose(BASE, keys)

    assert result.status == "not_eligible"
    assert result.proposed_sql is None


@pytest.mark.parametrize(
    "sql",
    [
        None,
        "",
        "not valid sql",
        "SELECT TOP 0 ItemKey FROM Catalog.Item ORDER BY Price DESC",
    ],
)
def test_missing_or_malformed_sql_is_not_repaired(sql):
    result = propose(sql)
    assert result.status == "not_eligible"
    assert result.proposed_sql is None
    assert result.original_sql == sql


def test_approved_reference_is_not_modified():
    sql = (
        "SELECT TOP (10) p.ProductID, p.Name, p.ListPrice "
        "FROM Production.Product AS p "
        "ORDER BY p.ListPrice DESC, p.ProductID ASC"
    )
    result = propose(sql, AW_KEYS)

    assert result.status == "not_eligible"
    assert result.reason == "primary_key_already_ordered"
    assert result.original_sql == sql
    assert result.proposed_sql is None


def test_proposal_does_not_mutate_metadata_or_original_sql():
    sql = BASE
    keys = {"Catalog.Item": ["ItemKey"]}
    original_keys = {name: list(columns) for name, columns in keys.items()}

    result = propose(sql, keys)

    assert result.status == "proposed"
    assert sql == BASE
    assert keys == original_keys


def _case(required_invariants, expected_rows=None):
    return {
        "review_status": "verified",
        "expected_columns": ["ItemKey", "ItemName", "Price"],
        "expected_rows": expected_rows if expected_rows is not None else [[1, "One", 10]],
        "expected_row_count": 1,
        "order_sensitive": True,
        "numeric_tolerance": None,
        "required_sql_invariants": required_invariants,
        "forbidden_sql_invariants": [],
    }


def test_comparator_distinguishes_original_and_repaired_sql():
    result = propose(BASE)
    actual = QueryResult(
        columns=["ItemKey", "ItemName", "Price"],
        rows=[{"ItemKey": 1, "ItemName": "One", "Price": 10}],
        row_count=1,
        max_rows=100,
        truncated=False,
    )
    case = _case([
        "table:Catalog.Item",
        "order_by:Price DESC,ItemKey ASC",
        "top:10",
    ])

    original = compare_semantics(case, result.original_sql, actual, True)
    repaired = compare_semantics(case, result.proposed_sql, actual, True)

    assert original.status == "fail"
    assert repaired.status == "pass"
    assert result.original_sql != result.proposed_sql


def test_repaired_sql_still_fails_incorrect_rows_or_other_invariants():
    result = propose(BASE)
    wrong_rows = QueryResult(
        columns=["ItemKey", "ItemName", "Price"],
        rows=[{"ItemKey": 2, "ItemName": "Two", "Price": 10}],
        row_count=1,
        max_rows=100,
        truncated=False,
    )
    case = _case([
        "table:Catalog.Item",
        "order_by:Price DESC,ItemKey ASC",
        "top:10",
    ])
    assert compare_semantics(case, result.proposed_sql, wrong_rows, True).status == "fail"

    other_case = _case([
        "table:Other.Item",
        "order_by:Price DESC,ItemKey ASC",
        "top:10",
    ])
    assert compare_semantics(case=other_case, generated_sql=result.proposed_sql, query_result=wrong_rows, pipeline_success=True).status == "fail"


def test_result_type_is_immutable():
    result = propose(BASE)
    with pytest.raises(AttributeError):
        result.status = "proposed"
