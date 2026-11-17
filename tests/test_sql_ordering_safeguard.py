import pytest

from ssa.sql_ordering_safeguard import OrderingRisk, inspect_top_ordering


KEYS = {"Catalog.Item": ("ItemKey",)}
BASE = "SELECT TOP 10 ItemKey, ItemName, Price FROM Catalog.Item"


def assert_status(sql, status, reason=None, metadata=KEYS):
    result = inspect_top_ordering(sql, metadata)
    assert isinstance(result, OrderingRisk)
    assert result.status == status
    if reason is not None:
        assert result.reason == reason
    return result


def test_missing_declared_key_is_a_risk():
    result = assert_status(
        f"{BASE} ORDER BY Price DESC",
        "risk",
        "primary_key_missing_from_root_order",
    )
    assert result.table == "Catalog.Item"
    assert result.primary_key_columns == ("ItemKey",)
    assert result.missing_key_columns == ("ItemKey",)


@pytest.mark.parametrize("ordering", [
    "ItemKey ASC",
    "ItemKey DESC",
    "i.ItemKey ASC",
    "[ItemKey] ASC",
    "[i].[ItemKey] DESC",
])
def test_direct_primary_key_ordering_is_sufficient(ordering):
    result = assert_status(
        f"SELECT TOP 10 i.ItemKey, i.ItemName, i.Price "
        f"FROM [Catalog].[Item] AS i ORDER BY {ordering}",
        "sufficient",
        "primary_key_in_root_order",
    )
    assert result.missing_key_columns == ()


def test_identifier_case_is_normalized():
    result = inspect_top_ordering(
        "SELECT TOP 10 ITEMKEY FROM catalog.item ORDER BY ITEMKEY DESC",
        {"[CATALOG].[ITEM]": ("[ItemKey]",)},
    )
    assert result.status == "sufficient"


def test_composite_primary_key_is_unsupported():
    result = assert_status(
        "SELECT TOP 10 A, B, Price FROM Catalog.Composite ORDER BY Price DESC",
        "unsupported",
        "unsupported_composite_primary_key",
        {"Catalog.Composite": ("A", "B")},
    )
    assert result.primary_key_columns == ("A", "B")


@pytest.mark.parametrize("metadata", [
    {},
    {"Catalog.Item": ()},
    {"Catalog.Item": ("ItemKey",), "Catalog.Other": "ItemKey"},
    {"Catalog.Item": ("",)},
])
def test_missing_or_malformed_metadata_is_unsupported(metadata):
    assert_status(
        f"{BASE} ORDER BY Price DESC",
        "unsupported",
        "unsupported_primary_key_metadata",
        metadata,
    )


def test_foreign_key_or_identity_like_names_do_not_establish_uniqueness():
    assert_status(
        "SELECT TOP 10 CustomerID, Price FROM Sales.OrderLine ORDER BY Price DESC",
        "unsupported",
        "unsupported_primary_key_metadata",
        {"Sales.OrderLine": ()},
    )


@pytest.mark.parametrize("sql, reason", [
    ("SELECT ItemKey FROM Catalog.Item ORDER BY Price DESC", "unsupported_top_missing"),
    ("SELECT TOP (10) PERCENT ItemKey FROM Catalog.Item ORDER BY Price DESC", "unsupported_top_options"),
    ("SELECT TOP (10) WITH TIES ItemKey FROM Catalog.Item ORDER BY Price DESC", "unsupported_top_options"),
    ("SELECT TOP (@n) ItemKey FROM Catalog.Item ORDER BY Price DESC", "unsupported_top_expression"),
    ("SELECT TOP (1 + 9) ItemKey FROM Catalog.Item ORDER BY Price DESC", "unsupported_top_expression"),
    ("SELECT TOP 10 ItemKey FROM Catalog.Item", "unsupported_order_by_missing"),
])
def test_top_and_ordering_requirements_are_conservative(sql, reason):
    assert_status(sql, "unsupported", reason)


@pytest.mark.parametrize("sql, reason", [
    ("SELECT TOP 10 i.ItemKey FROM Catalog.Item i JOIN Catalog.Other o ON i.ItemKey=o.ItemKey ORDER BY i.Price DESC", "unsupported_join"),
    ("SELECT TOP 10 ItemKey, COUNT(*) FROM Catalog.Item GROUP BY ItemKey ORDER BY COUNT(*) DESC", "unsupported_grouping_or_aggregate"),
    ("SELECT DISTINCT TOP 10 ItemKey FROM Catalog.Item ORDER BY ItemKey", "unsupported_distinct"),
    ("WITH x AS (SELECT TOP 10 ItemKey FROM Catalog.Item ORDER BY Price DESC) SELECT * FROM x", "unsupported_cte"),
    ("SELECT TOP 10 ItemKey FROM Catalog.Item WHERE ItemKey IN (SELECT ItemKey FROM Catalog.Other) ORDER BY Price DESC", "unsupported_subquery"),
    ("SELECT TOP 10 ItemKey FROM Catalog.Item ORDER BY ROW_NUMBER() OVER (ORDER BY Price)", "unsupported_window"),
    ("SELECT TOP 10 ItemKey FROM Catalog.Item UNION SELECT TOP 10 ItemKey FROM Catalog.Other", "unsupported_non_select"),
    ("SELECT TOP 10 ItemKey FROM Catalog.Item, Catalog.Other ORDER BY Price DESC", "unsupported_join"),
    ("DELETE FROM Catalog.Item", "unsupported_non_select"),
])
def test_unsupported_query_shapes_fail_closed(sql, reason):
    assert_status(sql, "unsupported", reason)


@pytest.mark.parametrize("ordering", [
    "1 DESC",
    "ItemKey + 1 ASC",
    "LOWER(ItemKey) ASC",
])
def test_unsupported_order_expressions_fail_closed(ordering):
    sql = f"{BASE} ORDER BY {ordering}"
    result = inspect_top_ordering(sql, KEYS)
    assert result.status == "unsupported"
    assert result.reason in {
        "unsupported_order_expression",
        "unsupported_order_alias",
    }


def test_output_alias_ordering_is_unsupported():
    result = inspect_top_ordering(
        "SELECT TOP 10 ItemKey AS ItemKeyAlias, Price "
        "FROM Catalog.Item ORDER BY ItemKeyAlias ASC",
        KEYS,
    )
    assert result.status == "unsupported"
    assert result.reason == "unsupported_order_alias"


def test_unknown_qualified_order_column_is_unsupported():
    result = inspect_top_ordering(
        "SELECT TOP 10 i.ItemKey FROM Catalog.Item i ORDER BY x.ItemKey",
        KEYS,
    )
    assert result.status == "unsupported"
    assert result.reason == "unsupported_ambiguous_order_column"


def test_base_table_qualifier_is_not_used_after_aliasing():
    result = inspect_top_ordering(
        "SELECT TOP 10 i.ItemKey FROM Catalog.Item AS i ORDER BY Item.ItemKey",
        KEYS,
    )
    assert result.status == "unsupported"
    assert result.reason == "unsupported_ambiguous_order_column"


def test_view_without_authoritative_key_is_unsupported():
    result = inspect_top_ordering(
        "SELECT TOP 10 ItemKey, Price FROM Catalog.ItemView ORDER BY Price DESC",
        {"Catalog.Item": ("ItemKey",)},
    )
    assert result.status == "unsupported"
    assert result.reason == "unsupported_table_metadata"


def test_multiple_and_malformed_sql_are_unsupported():
    assert_status(
        f"{BASE} ORDER BY Price DESC; SELECT 1",
        "unsupported",
        "unsupported_multiple_statements",
    )
    assert_status(
        "SELECT TOP 10 FROM",
        "unsupported",
        "unsupported_parse_error",
    )
    assert_status(None, "unsupported", "unsupported_empty_sql")


def test_original_sql_and_metadata_are_not_modified():
    sql = f"{BASE} ORDER BY Price DESC"
    metadata = {"Catalog.Item": ["ItemKey"]}
    original_sql = sql
    original_metadata = {key: list(value) for key, value in metadata.items()}

    result = inspect_top_ordering(sql, metadata)

    assert sql == original_sql
    assert metadata == original_metadata
    assert result.status == "risk"


def test_detector_is_pure_and_does_not_require_external_services():
    result = inspect_top_ordering(
        "SELECT TOP 10 ItemKey FROM Catalog.Item ORDER BY ItemKey",
        KEYS,
    )
    assert result.status == "sufficient"
