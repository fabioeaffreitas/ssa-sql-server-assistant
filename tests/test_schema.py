
"""
SSA - Schema Introspection Test
"""

from ssa.db import get_connection
from ssa.schema import get_schema_metadata, print_schema_summary


def main():
    with get_connection() as connection:
        metadata = get_schema_metadata(connection)

    print_schema_summary(metadata)


if __name__ == "__main__":
    main()
