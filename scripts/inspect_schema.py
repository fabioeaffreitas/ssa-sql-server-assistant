"""Print the configured database schema summary."""

from ssa.db import get_connection
from ssa.schema import get_schema_metadata, print_schema_summary


if __name__ == "__main__":
    with get_connection() as connection:
        metadata = get_schema_metadata(connection)
    print_schema_summary(metadata)
