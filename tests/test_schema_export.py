from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import pytest

import ssa.schema_export as schema_export
from ssa.schema_capture_assembly import assemble_versioned_schema_metadata
from ssa.schema_capture_provenance import SQLServerIdentityEvidence, parse_capture_provenance
from ssa.schema_identity import SQLServerIdentityCapture
from ssa.schema_key_metadata import fingerprint_schema_metadata
from ssa.schema_metadata_writer import write_versioned_schema_metadata


CAPTURED_AT = datetime(2026, 11, 19, 12, 34, 56, 123456, tzinfo=timezone.utc)
SNAPSHOT_ID = UUID("11111111-2222-3333-4444-555555555555")


def semantic_metadata():
    return {
        "database": "AdventureWorks2025",
        "tables": {
            "dbo.Item": {
                "schema": "dbo",
                "table": "Item",
                "description": "Items",
                "columns": [
                    {
                        "name": "ItemID",
                        "data_type": "int",
                        "max_length": 4,
                        "precision": 10,
                        "scale": 0,
                        "nullable": False,
                        "identity": True,
                        "description": "Identifier",
                    }
                ],
                "primary_keys": ["ItemID"],
                "foreign_keys": [],
            }
        },
        "views": {},
        "table_count": 1,
        "view_count": 0,
        "foreign_key_count": 0,
    }


def identity(server_name="SQLHOST\\SQLEXPRESS", *, database_id=7):
    return SQLServerIdentityCapture(
        identity=SQLServerIdentityEvidence(
            server_name=server_name,
            instance_name="SQLEXPRESS",
            database_id=database_id,
        ),
        server_version="17.0.1000.7",
    )


class FakeConnection:
    def __init__(self, events, *, close_error=None):
        self.events = events
        self.close_error = close_error
        self.closed = False

    def close(self):
        self.events.append("close")
        self.closed = True
        if self.close_error:
            raise self.close_error


def configured_fakes(
    *,
    output_path="ignored.json",
    first=None,
    second=None,
    metadata=None,
    connection=None,
    events=None,
    writer=None,
    assembler=None,
    clock=None,
    snapshot_id_factory=None,
):
    events = [] if events is None else events
    connection = FakeConnection(events) if connection is None else connection
    first = identity() if first is None else first
    second = first if second is None else second
    metadata = semantic_metadata() if metadata is None else metadata
    identities = iter((first, second))

    def connection_factory():
        events.append("connection")
        return connection

    def identity_reader(received_connection):
        assert received_connection is connection
        events.append("identity")
        return next(identities)

    def schema_reader(received_connection):
        assert received_connection is connection
        events.append("schema")
        return metadata

    return {
        "output_path": output_path,
        "connection_factory": connection_factory,
        "identity_reader": identity_reader,
        "schema_reader": schema_reader,
        "clock": (lambda: CAPTURED_AT) if clock is None else clock,
        "snapshot_id_factory": (
            (lambda: SNAPSHOT_ID)
            if snapshot_id_factory is None
            else snapshot_id_factory
        ),
        "assembler": assembler or assemble_versioned_schema_metadata,
        "writer": writer or writer_spy,
        "connection": connection,
        "events": events,
    }


def writer_spy(metadata, destination):
    return Path(destination)


def run_export(**overrides):
    dependencies = configured_fakes(**overrides)
    schema_export.export_schema(**{
        key: value
        for key, value in dependencies.items()
        if key in {
            "output_path",
            "connection_factory",
            "identity_reader",
            "schema_reader",
            "clock",
            "snapshot_id_factory",
            "assembler",
            "writer",
        }
    })
    return dependencies


def test_default_and_explicit_output_paths_and_positional_call(monkeypatch, capsys, tmp_path):
    captured = []

    def writer(metadata, destination):
        captured.append(destination)

    monkeypatch.setattr(schema_export, "write_versioned_schema_metadata", writer)
    dependencies = configured_fakes()
    schema_export.export_schema(
        "data/schema_metadata.json",
        connection_factory=dependencies["connection_factory"],
        identity_reader=dependencies["identity_reader"],
        schema_reader=dependencies["schema_reader"],
        clock=dependencies["clock"],
        snapshot_id_factory=dependencies["snapshot_id_factory"],
        assembler=dependencies["assembler"],
    )
    assert captured == ["data/schema_metadata.json"]
    assert "Schema exported to: data/schema_metadata.json" in capsys.readouterr().out

    dependencies = configured_fakes(output_path=tmp_path / "explicit.json", writer=writer)
    schema_export.export_schema(**{
        key: dependencies[key]
        for key in (
            "output_path", "connection_factory", "identity_reader", "schema_reader",
            "clock", "snapshot_id_factory", "assembler", "writer",
        )
    })
    assert captured[-1] == tmp_path / "explicit.json"


def test_exact_order_single_connection_and_success_cleanup():
    dependencies = run_export()

    assert dependencies["events"] == [
        "connection", "identity", "schema", "identity", "close"
    ]
    assert dependencies["connection"].closed is True


@pytest.mark.parametrize("field", ["server_name", "instance_name", "database_id", "database_guid"])
def test_identity_mismatch_is_rejected(field):
    first = identity()
    values = {
        "server_name": first.identity.server_name,
        "instance_name": first.identity.instance_name,
        "database_id": first.identity.database_id,
        "database_guid": first.identity.database_guid,
    }
    values[field] = "OTHER" if field != "database_id" else 8
    changed = SQLServerIdentityCapture(
        identity=SQLServerIdentityEvidence(**values),
        server_version=first.server_version,
    )

    dependencies = configured_fakes(first=first, second=changed)
    with pytest.raises(schema_export.SchemaIdentityMismatchError):
        schema_export.export_schema(**{
            key: dependencies[key]
            for key in (
                "output_path", "connection_factory", "identity_reader", "schema_reader",
                "clock", "snapshot_id_factory", "assembler", "writer",
            )
        })
    assert dependencies["connection"].closed is True


def test_server_version_mismatch_is_rejected():
    first = identity()
    second = SQLServerIdentityCapture(first.identity, "17.0.2000.1")
    dependencies = configured_fakes(first=first, second=second)

    with pytest.raises(schema_export.SchemaIdentityMismatchError):
        schema_export.export_schema(**{
            key: dependencies[key]
            for key in (
                "output_path", "connection_factory", "identity_reader", "schema_reader",
                "clock", "snapshot_id_factory", "assembler", "writer",
            )
        })


@pytest.mark.parametrize("stage", ["first_identity", "schema", "second_identity"])
def test_source_failures_close_connection_and_do_not_call_post_capture_dependencies(stage):
    events = []
    connection = FakeConnection(events)
    calls = {"clock": 0, "uuid": 0, "assembler": 0, "writer": 0}
    first = identity()
    identity_calls = iter((RuntimeError("identity failure"), first))

    def connection_factory():
        events.append("connection")
        return connection

    def identity_reader(_):
        events.append("identity")
        value = next(identity_calls)
        if isinstance(value, Exception):
            raise value
        return value

    def schema_reader(_):
        events.append("schema")
        raise RuntimeError("schema failure")

    if stage == "schema":
        identity_reader = lambda _: first
    elif stage == "second_identity":
        identity_values = iter((first, RuntimeError("identity failure")))
        identity_reader = lambda _: next(identity_values)

    def counted_clock():
        calls["clock"] += 1
        return CAPTURED_AT

    def counted_uuid():
        calls["uuid"] += 1
        return SNAPSHOT_ID

    def counted_assembler(*args, **kwargs):
        calls["assembler"] += 1

    def counted_writer(*args, **kwargs):
        calls["writer"] += 1

    kwargs = dict(
        output_path="ignored.json",
        connection_factory=connection_factory,
        identity_reader=identity_reader,
        schema_reader=schema_reader,
        clock=counted_clock,
        snapshot_id_factory=counted_uuid,
        assembler=counted_assembler,
        writer=counted_writer,
    )
    with pytest.raises(RuntimeError):
        schema_export.export_schema(**kwargs)
    assert connection.closed is True
    assert calls == {"clock": 0, "uuid": 0, "assembler": 0, "writer": 0}


def test_cleanup_failure_blocks_write_and_preserves_primary_error():
    events = []
    connection = FakeConnection(events, close_error=OSError("close failure"))
    writer_calls = []
    dependencies = configured_fakes(connection=connection, events=events, writer=lambda *args: writer_calls.append(args))

    with pytest.raises(OSError, match="close failure"):
        schema_export.export_schema(**{
            key: dependencies[key]
            for key in (
                "output_path", "connection_factory", "identity_reader", "schema_reader",
                "clock", "snapshot_id_factory", "assembler", "writer",
            )
        })
    assert writer_calls == []

    events = []
    connection = FakeConnection(events)
    dependencies = configured_fakes(connection=connection, events=events)
    dependencies["schema_reader"] = lambda _: (_ for _ in ()).throw(RuntimeError("primary"))
    connection.close_error = OSError("cleanup")
    with pytest.raises(RuntimeError, match="primary"):
        schema_export.export_schema(**{
            key: dependencies[key]
            for key in (
                "output_path", "connection_factory", "identity_reader", "schema_reader",
                "clock", "snapshot_id_factory", "assembler", "writer",
            )
        })


def test_capture_dependencies_called_once_after_connection_close_and_writer_after_close():
    events = []
    calls = {"clock": 0, "uuid": 0, "assembler": 0, "writer": 0}
    dependencies = configured_fakes(events=events)

    def clock():
        calls["clock"] += 1
        assert dependencies["connection"].closed is True
        return CAPTURED_AT

    def snapshot_id_factory():
        calls["uuid"] += 1
        assert dependencies["connection"].closed is True
        return SNAPSHOT_ID

    def assembler(metadata, **kwargs):
        calls["assembler"] += 1
        assert dependencies["connection"].closed is True
        assert metadata == semantic_metadata()
        assert kwargs == {
            "snapshot_id": str(SNAPSHOT_ID),
            "captured_at_utc": CAPTURED_AT,
            "server_identity": identity().identity,
            "server_version": "17.0.1000.7",
        }
        return assemble_versioned_schema_metadata(metadata, **kwargs)

    def writer(metadata, destination):
        calls["writer"] += 1
        assert dependencies["connection"].closed is True
        assert destination == "ignored.json"
        parse_capture_provenance(metadata)

    schema_export.export_schema(
        "ignored.json",
        connection_factory=dependencies["connection_factory"],
        identity_reader=dependencies["identity_reader"],
        schema_reader=dependencies["schema_reader"],
        clock=clock,
        snapshot_id_factory=snapshot_id_factory,
        assembler=assembler,
        writer=writer,
    )
    assert calls == {"clock": 1, "uuid": 1, "assembler": 1, "writer": 1}


def test_invalid_timestamp_and_snapshot_id_reach_contract_and_do_not_write():
    for clock, snapshot_factory in [
        (lambda: datetime(2026, 1, 1), lambda: SNAPSHOT_ID),
        (lambda: CAPTURED_AT, lambda: ""),
    ]:
        writer_calls = []
        dependencies = configured_fakes(
            clock=clock,
            snapshot_id_factory=snapshot_factory,
            writer=lambda *args: writer_calls.append(args),
        )
        with pytest.raises(ValueError):
            schema_export.export_schema(**{
                key: dependencies[key]
                for key in (
                    "output_path", "connection_factory", "identity_reader", "schema_reader",
                    "clock", "snapshot_id_factory", "assembler", "writer",
                )
            })
        assert writer_calls == []


def test_real_assembler_and_writer_produce_verified_version_two_file(tmp_path, capsys):
    destination = tmp_path / "nested" / "schema_metadata.json"
    dependencies = configured_fakes(output_path=destination, writer=write_versioned_schema_metadata)

    schema_export.export_schema(**{
        key: dependencies[key]
        for key in (
            "output_path", "connection_factory", "identity_reader", "schema_reader",
            "clock", "snapshot_id_factory", "assembler", "writer",
        )
    })

    import json

    document = json.loads(destination.read_text(encoding="utf-8"))
    assert document["metadata_format_version"] == 2
    assert parse_capture_provenance(document).content_fingerprint == fingerprint_schema_metadata(document)
    output = capsys.readouterr().out
    assert f"Schema exported to: {destination}" in output
    assert "Tables: 1" in output
    assert "Foreign keys: 0" in output


def test_assembly_and_writer_failures_do_not_print_success_or_replace_existing_file(tmp_path, capsys):
    destination = tmp_path / "schema.json"
    destination.write_text("old", encoding="utf-8")

    for failing_dependency in ("assembler", "writer"):
        dependencies = configured_fakes(output_path=destination)
        if failing_dependency == "assembler":
            dependencies["assembler"] = lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("assembly failure")
            )
        else:
            dependencies["writer"] = lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("writer failure")
            )
        with pytest.raises(RuntimeError):
            schema_export.export_schema(**{
                key: dependencies[key]
                for key in (
                    "output_path", "connection_factory", "identity_reader", "schema_reader",
                    "clock", "snapshot_id_factory", "assembler", "writer",
                )
            })
        assert destination.read_text(encoding="utf-8") == "old"

    assert "Schema exported to:" not in capsys.readouterr().out


def test_module_import_does_not_export_or_contact_other_services(monkeypatch):
    monkeypatch.setattr(schema_export, "get_connection", lambda: (_ for _ in ()).throw(
        AssertionError("connection must not occur at import")
    ))
    assert hasattr(schema_export, "export_schema")
