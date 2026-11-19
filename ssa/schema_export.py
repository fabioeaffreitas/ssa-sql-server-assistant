
"""SSA - Versioned SQL Server schema metadata export."""

from datetime import datetime, timezone
import uuid

from ssa.db import get_connection
from ssa.schema import get_schema_metadata
from ssa.schema_capture_assembly import assemble_versioned_schema_metadata
from ssa.schema_identity import read_sql_server_identity
from ssa.schema_metadata_writer import write_versioned_schema_metadata


class SchemaIdentityMismatchError(RuntimeError):
    """Raised when source identity changes during schema extraction."""


def export_schema(
    output_path="data/schema_metadata.json",
    *,
    connection_factory=None,
    identity_reader=None,
    schema_reader=None,
    clock=None,
    snapshot_id_factory=None,
    assembler=None,
    writer=None,
):
    """Export versioned SQL Server schema metadata to JSON."""

    connection_factory = get_connection if connection_factory is None else connection_factory
    identity_reader = read_sql_server_identity if identity_reader is None else identity_reader
    schema_reader = get_schema_metadata if schema_reader is None else schema_reader
    clock = (lambda: datetime.now(timezone.utc)) if clock is None else clock
    snapshot_id_factory = uuid.uuid4 if snapshot_id_factory is None else snapshot_id_factory
    assembler = (
        assemble_versioned_schema_metadata if assembler is None else assembler
    )
    writer = (
        write_versioned_schema_metadata if writer is None else writer
    )

    connection = None
    primary_error = None
    try:
        connection = connection_factory()
        before = identity_reader(connection)
        metadata = schema_reader(connection)
        after = identity_reader(connection)
        _require_stable_identity(before, after)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                if primary_error is None:
                    raise

    captured_at_utc = clock()
    snapshot_id = str(snapshot_id_factory())
    assembled = assembler(
        metadata,
        snapshot_id=snapshot_id,
        captured_at_utc=captured_at_utc,
        server_identity=before.identity,
        server_version=before.server_version,
    )
    writer(assembled, output_path)

    print(f"Schema exported to: {output_path}")
    print(f"Tables: {metadata['table_count']}")
    print(f"Foreign keys: {metadata['foreign_key_count']}")


def _require_stable_identity(before, after):
    if before.identity != after.identity or before.server_version != after.server_version:
        raise SchemaIdentityMismatchError(
            "SQL Server identity changed during schema extraction"
        )


if __name__ == "__main__":
    export_schema()
