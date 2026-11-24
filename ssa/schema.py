
"""
SSA - SQL Server Schema Introspection

Extracts SQL Server tables, views, columns, descriptions,
primary keys, and foreign keys.
"""

from collections import defaultdict


class SchemaColumnResolutionError(RuntimeError):
    """Raised when visible column metadata cannot be resolved safely."""


def _resolve_column_type(row, object_key):
    """Resolve one column type without allowing silent row loss."""

    declared_count = _valid_type_match_count(
        row.declared_type_match_count,
        object_key,
        row.column_name,
        "declared",
    )
    system_count = _valid_type_match_count(
        row.system_type_match_count,
        object_key,
        row.column_name,
        "system",
    )

    if declared_count > 1:
        raise SchemaColumnResolutionError(
            f"ambiguous declared type for {object_key}.{row.column_name} "
            f"(matches={declared_count})"
        )

    if declared_count == 1:
        return _valid_type_name(
            row.declared_type_name,
            object_key,
            row.column_name,
            "declared",
        )

    if system_count > 1:
        raise SchemaColumnResolutionError(
            f"ambiguous system type for {object_key}.{row.column_name} "
            f"(matches={system_count})"
        )

    if system_count == 1:
        return _valid_type_name(
            row.system_type_name,
            object_key,
            row.column_name,
            "system fallback",
        )

    raise SchemaColumnResolutionError(
        f"unresolved type for {object_key}.{row.column_name} "
        f"(declared_matches={declared_count}, system_matches={system_count})"
    )


def _valid_type_match_count(value, object_key, column_name, lookup_name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaColumnResolutionError(
            f"invalid {lookup_name} type match count for "
            f"{object_key}.{column_name}"
        )
    return value


def _valid_type_name(value, object_key, column_name, lookup_name):
    if not isinstance(value, str) or not value.strip():
        raise SchemaColumnResolutionError(
            f"invalid {lookup_name} type name for "
            f"{object_key}.{column_name}"
        )
    return value


def get_schema_descriptions(connection):
    """Retrieve table and view descriptions from MS_Description."""

    query = """
    SELECT
        s.name AS schema_name,
        o.name AS object_name,
        c.name AS column_name,
        CAST(object_ep.value AS NVARCHAR(MAX)) AS object_description,
        CAST(column_ep.value AS NVARCHAR(MAX)) AS column_description
    FROM sys.objects AS o
    INNER JOIN sys.schemas AS s
        ON o.schema_id = s.schema_id
    INNER JOIN sys.columns AS c
        ON o.object_id = c.object_id
    LEFT JOIN sys.extended_properties AS object_ep
        ON object_ep.major_id = o.object_id
        AND object_ep.minor_id = 0
        AND object_ep.name = 'MS_Description'
    LEFT JOIN sys.extended_properties AS column_ep
        ON column_ep.major_id = c.object_id
        AND column_ep.minor_id = c.column_id
        AND column_ep.name = 'MS_Description'
    WHERE o.type IN ('U', 'V')
    ORDER BY s.name, o.name, c.column_id;
    """

    cursor = connection.cursor()
    cursor.execute(query)

    descriptions = {}

    for row in cursor.fetchall():
        object_key = f"{row.schema_name}.{row.object_name}"

        if object_key not in descriptions:
            descriptions[object_key] = {
                "description": row.object_description,
                "columns": {},
            }

        descriptions[object_key]["columns"][row.column_name] = (
            row.column_description
        )

    return descriptions


def get_schema_metadata(connection):
    """Extract tables and views with their schema metadata."""

    cursor = connection.cursor()
    descriptions = get_schema_descriptions(connection)

    # ---------------------------------------------------------
    # 1. Discover tables and views
    # ---------------------------------------------------------

    objects_query = """
    SELECT
        s.name AS schema_name,
        o.name AS object_name,
        RTRIM(o.type) AS object_type,
        c.name AS column_name,
        COALESCE(declared_type.match_count, 0) AS declared_type_match_count,
        declared_type.type_name AS declared_type_name,
        COALESCE(system_type.match_count, 0) AS system_type_match_count,
        system_type.type_name AS system_type_name,
        c.max_length,
        c.precision,
        c.scale,
        c.is_nullable,
        c.is_identity,
        c.column_id
    FROM sys.objects AS o
    INNER JOIN sys.schemas AS s
        ON o.schema_id = s.schema_id
    INNER JOIN sys.columns AS c
        ON o.object_id = c.object_id
    LEFT JOIN (
        SELECT
            user_type_id,
            COUNT(*) AS match_count,
            MIN(name) AS type_name
        FROM sys.types
        GROUP BY user_type_id
    ) AS declared_type
        ON declared_type.user_type_id = c.user_type_id
    LEFT JOIN (
        SELECT
            system_type_id,
            COUNT(*) AS match_count,
            MIN(name) AS type_name
        FROM sys.types
        WHERE user_type_id = system_type_id
          AND is_user_defined = 0
          AND is_assembly_type = 0
        GROUP BY system_type_id
    ) AS system_type
        ON system_type.system_type_id = c.system_type_id
    WHERE o.type IN ('U', 'V')
    ORDER BY s.name, o.name, c.column_id;
    """

    cursor.execute(objects_query)

    tables = {}
    views = {}
    seen_column_names = {}
    seen_column_ids = {}

    for row in cursor.fetchall():
        object_key = f"{row.schema_name}.{row.object_name}"

        object_type = row.object_type.strip()
        if object_type not in {"U", "V"}:
            raise SchemaColumnResolutionError(
                f"unsupported object type for {object_key}: {object_type!r}"
            )
        target = tables if object_type == "U" else views

        if object_key not in target:
            target[object_key] = {
                "schema": row.schema_name,
                "table" if object_type == "U" else "view":
                    row.object_name,
                "description": descriptions.get(
                    object_key, {}
                ).get("description"),
                "columns": [],
            }

            seen_column_names[object_key] = set()
            seen_column_ids[object_key] = set()

            if object_type == "U":
                target[object_key]["primary_keys"] = []
                target[object_key]["foreign_keys"] = []

        if row.column_name in seen_column_names[object_key]:
            raise SchemaColumnResolutionError(
                f"duplicate column name for {object_key}.{row.column_name}"
            )
        if row.column_id in seen_column_ids[object_key]:
            raise SchemaColumnResolutionError(
                f"duplicate column id for {object_key}.{row.column_name}"
            )
        seen_column_names[object_key].add(row.column_name)
        seen_column_ids[object_key].add(row.column_id)

        column_description = descriptions.get(
            object_key, {}
        ).get("columns", {}).get(row.column_name)

        target[object_key]["columns"].append({
            "name": row.column_name,
            "data_type": _resolve_column_type(row, object_key),
            "max_length": row.max_length,
            "precision": row.precision,
            "scale": row.scale,
            "nullable": bool(row.is_nullable),
            "identity": bool(row.is_identity),
            "description": column_description,
        })

    # ---------------------------------------------------------
    # 2. Discover primary keys (tables only)
    # ---------------------------------------------------------

    primary_keys_query = """
    SELECT
        s.name AS schema_name,
        t.name AS table_name,
        c.name AS column_name,
        ic.key_ordinal
    FROM sys.key_constraints AS kc
    INNER JOIN sys.tables AS t
        ON kc.parent_object_id = t.object_id
    INNER JOIN sys.schemas AS s
        ON t.schema_id = s.schema_id
    INNER JOIN sys.index_columns AS ic
        ON kc.parent_object_id = ic.object_id
        AND kc.unique_index_id = ic.index_id
    INNER JOIN sys.columns AS c
        ON ic.object_id = c.object_id
        AND ic.column_id = c.column_id
    WHERE kc.type = 'PK'
    ORDER BY s.name, t.name, ic.key_ordinal;
    """

    cursor.execute(primary_keys_query)

    primary_keys = defaultdict(list)

    for row in cursor.fetchall():
        table_key = f"{row.schema_name}.{row.table_name}"
        primary_keys[table_key].append(row.column_name)

    for table_key, columns in primary_keys.items():
        if table_key in tables:
            tables[table_key]["primary_keys"] = columns

    # ---------------------------------------------------------
    # 3. Discover foreign keys (tables only)
    # ---------------------------------------------------------

    foreign_keys_query = """
    SELECT
        fk.name AS foreign_key_name,
        ps.name AS parent_schema,
        pt.name AS parent_table,
        pc.name AS parent_column,
        rs.name AS referenced_schema,
        rt.name AS referenced_table,
        rc.name AS referenced_column,
        fkc.constraint_column_id
    FROM sys.foreign_keys AS fk
    INNER JOIN sys.foreign_key_columns AS fkc
        ON fk.object_id = fkc.constraint_object_id
    INNER JOIN sys.tables AS pt
        ON fkc.parent_object_id = pt.object_id
    INNER JOIN sys.schemas AS ps
        ON pt.schema_id = ps.schema_id
    INNER JOIN sys.columns AS pc
        ON fkc.parent_object_id = pc.object_id
        AND fkc.parent_column_id = pc.column_id
    INNER JOIN sys.tables AS rt
        ON fkc.referenced_object_id = rt.object_id
    INNER JOIN sys.schemas AS rs
        ON rt.schema_id = rs.schema_id
    INNER JOIN sys.columns AS rc
        ON fkc.referenced_object_id = rc.object_id
        AND fkc.referenced_column_id = rc.column_id
    ORDER BY
        ps.name,
        pt.name,
        fk.name,
        fkc.constraint_column_id;
    """

    cursor.execute(foreign_keys_query)

    foreign_keys = defaultdict(dict)

    for row in cursor.fetchall():
        table_key = f"{row.parent_schema}.{row.parent_table}"
        fk_name = row.foreign_key_name

        if fk_name not in foreign_keys[table_key]:
            foreign_keys[table_key][fk_name] = {
                "name": fk_name,
                "referenced_schema": row.referenced_schema,
                "referenced_table": row.referenced_table,
                "columns": [],
                "referenced_columns": [],
            }

        foreign_keys[table_key][fk_name]["columns"].append(
            row.parent_column
        )

        foreign_keys[table_key][fk_name]["referenced_columns"].append(
            row.referenced_column
        )

    for table_key, fk_dict in foreign_keys.items():
        if table_key in tables:
            tables[table_key]["foreign_keys"] = list(
                fk_dict.values()
            )

    # ---------------------------------------------------------
    # 4. Build final metadata
    # ---------------------------------------------------------

    foreign_key_count = sum(
        len(table["foreign_keys"])
        for table in tables.values()
    )

    cursor.execute("SELECT DB_NAME();")
    database_name = cursor.fetchone()[0]

    return {
        "database": database_name,
        "tables": tables,
        "views": views,
        "table_count": len(tables),
        "view_count": len(views),
        "foreign_key_count": foreign_key_count,
    }


def print_schema_summary(metadata):
    """Print a readable summary of schema metadata."""

    print(f"\nDatabase: {metadata['database']}")
    print(f"Tables: {metadata['table_count']}")
    print(f"Views: {metadata['view_count']}")
    print(f"Foreign keys: {metadata['foreign_key_count']}")

    for object_type in ("tables", "views"):
        print(f"\n{object_type.upper()}:")

        for object_key, obj in metadata[object_type].items():
            print(f"\n{object_key}")

            if obj.get("description"):
                print(f"  Description: {obj['description']}")

            for column in obj["columns"]:
                nullable = "NULL" if column["nullable"] else "NOT NULL"
                identity = " IDENTITY" if column["identity"] else ""

                print(
                    f"  {column['name']} "
                    f"({column['data_type']}) "
                    f"{nullable}{identity}"
                )

                if column.get("description"):
                    print(
                        f"    Description: {column['description']}"
                    )

            if object_type == "tables":
                if obj["primary_keys"]:
                    print(
                        "  PK: "
                        + ", ".join(obj["primary_keys"])
                    )

                for fk in obj["foreign_keys"]:
                    parent_columns = ", ".join(fk["columns"])
                    referenced_columns = ", ".join(
                        fk["referenced_columns"]
                    )

                    print(
                        f"  FK: {fk['name']} "
                        f"({parent_columns}) -> "
                        f"{fk['referenced_schema']}."
                        f"{fk['referenced_table']} "
                        f"({referenced_columns})"
                    )
