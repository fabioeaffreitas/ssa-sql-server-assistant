from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from ssa.schema_capture_assembly import assemble_versioned_schema_metadata
from ssa.schema_capture_provenance import SQLServerIdentityEvidence
import ssa.schema_metadata_writer as writer


def semantic_metadata():
    return {
        "database": "AdventureWorks2025",
        "tables": {
            "Catalog.Item": {
                "schema": "Catalog",
                "table": "Item",
                "description": "Items—é",
                "columns": [
                    {
                        "name": "ItemKey",
                        "data_type": "int",
                        "max_length": 4,
                        "precision": 10,
                        "scale": 0,
                        "nullable": False,
                        "identity": True,
                        "description": "Identifier",
                    },
                    {
                        "name": "ItemName",
                        "data_type": "nvarchar",
                        "max_length": 200,
                        "precision": 0,
                        "scale": 0,
                        "nullable": False,
                        "identity": False,
                        "description": "Name",
                    },
                ],
                "primary_keys": ["ItemKey"],
                "foreign_keys": [],
            }
        },
        "views": {},
        "table_count": 1,
        "view_count": 0,
        "foreign_key_count": 0,
    }


def valid_document(metadata=None):
    return assemble_versioned_schema_metadata(
        semantic_metadata() if metadata is None else metadata,
        snapshot_id="capture-1",
        captured_at_utc=datetime(
            2026,
            11,
            19,
            12,
            34,
            56,
            123456,
            tzinfo=timezone.utc,
        ),
        server_identity=SQLServerIdentityEvidence(
            server_name="SQLHOST\\SQLEXPRESS",
            instance_name="SQLEXPRESS",
            database_id=7,
            database_guid="11111111-2222-3333-4444-555555555555",
        ),
        server_version="17.0.1000.7",
        capture_tool_version="ssa-schema-export/1",
    )


def write_document(tmp_path, metadata=None, name="schema_metadata.json"):
    return writer.write_versioned_schema_metadata(
        valid_document() if metadata is None else metadata,
        tmp_path / name,
    )


def test_writes_valid_document_and_returns_destination(tmp_path):
    destination = write_document(tmp_path)
    assert destination == tmp_path / "schema_metadata.json"
    assert destination.exists()
    assert json.loads(destination.read_text(encoding="utf-8"))["metadata_format_version"] == 2


def test_creates_missing_parent_and_replaces_existing_destination(tmp_path):
    destination = tmp_path / "nested" / "schema.json"
    assert write_document(destination.parent, name=destination.name) == destination

    old = destination.read_text(encoding="utf-8")
    changed = valid_document()
    changed["provenance"]["snapshot_id"] = "capture-2"
    writer.write_versioned_schema_metadata(changed, destination)
    assert destination.read_text(encoding="utf-8") != old


def test_preserves_unicode_arrays_and_nested_content(tmp_path):
    metadata = semantic_metadata()
    metadata["tables"]["Catalog.Item"]["columns"].append(
        {
            "name": "順位",
            "data_type": "nvarchar",
            "description": "追加",
        }
    )
    write_document(tmp_path, valid_document(metadata))
    readback = json.loads((tmp_path / "schema_metadata.json").read_text(encoding="utf-8"))
    assert readback["tables"]["Catalog.Item"]["columns"][-1]["name"] == "順位"


def test_serialized_output_is_deterministic(tmp_path):
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    metadata = valid_document()
    writer.write_versioned_schema_metadata(metadata, first)
    writer.write_versioned_schema_metadata(metadata, second)
    assert first.read_bytes() == second.read_bytes()


def test_input_is_not_mutated_and_unrelated_files_are_untouched(tmp_path):
    metadata = valid_document()
    before = deepcopy(metadata)
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("keep", encoding="utf-8")
    write_document(tmp_path, metadata)
    assert metadata == before
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_temporary_file_is_same_directory_and_removed_after_success(tmp_path, monkeypatch):
    captured = {}
    real_mkstemp = writer.tempfile.mkstemp

    def capture_mkstemp(*args, **kwargs):
        captured["directory"] = Path(kwargs["dir"])
        result = real_mkstemp(*args, **kwargs)
        captured["temporary"] = Path(result[1])
        return result

    monkeypatch.setattr(writer.tempfile, "mkstemp", capture_mkstemp)
    destination = write_document(tmp_path)
    assert captured["directory"] == destination.parent
    assert not captured["temporary"].exists()


def test_replace_observes_closed_temporary_file(tmp_path, monkeypatch):
    observed = {}
    real_replace = writer.os.replace

    def observe_replace(source, destination):
        source_path = Path(source)
        with source_path.open("r+", encoding="utf-8") as handle:
            observed["readable"] = handle.read().startswith("{")
        real_replace(source, destination)

    monkeypatch.setattr(writer.os, "replace", observe_replace)
    write_document(tmp_path)
    assert observed["readable"] is True


@pytest.mark.parametrize(
    "mutator",
    [
        lambda document: document.pop("provenance"),
        lambda document: document.__setitem__("metadata_format_version", 3),
        lambda document: document["provenance"].__setitem__("captured_at_utc", "bad"),
        lambda document: document["provenance"].__setitem__("content_fingerprint", "A" * 64),
        lambda document: document["tables"]["Catalog.Item"].__setitem__("columns", "bad"),
    ],
)
def test_invalid_input_is_rejected_before_destination_change(tmp_path, mutator):
    destination = tmp_path / "schema.json"
    destination.write_text("old", encoding="utf-8")
    invalid = valid_document()
    mutator(invalid)
    with pytest.raises(Exception):
        writer.write_versioned_schema_metadata(invalid, destination)
    assert destination.read_text(encoding="utf-8") == "old"


def test_legacy_input_is_rejected_and_missing_destination_stays_absent(tmp_path):
    destination = tmp_path / "schema.json"
    with pytest.raises(Exception):
        writer.write_versioned_schema_metadata(semantic_metadata(), destination)
    assert not destination.exists()


def test_stored_fingerprint_mismatch_is_rejected(tmp_path):
    destination = tmp_path / "schema.json"
    invalid = valid_document()
    invalid["provenance"]["content_fingerprint"] = "b" * 64
    with pytest.raises(Exception):
        writer.write_versioned_schema_metadata(invalid, destination)
    assert not destination.exists()


def test_serialization_failure_preserves_destination(tmp_path, monkeypatch):
    destination = tmp_path / "schema.json"
    destination.write_text("old", encoding="utf-8")
    monkeypatch.setattr(writer, "_serialize_document", lambda *args, **kwargs: (_ for _ in ()).throw(
        ValueError("serialize failure")
    ))
    with pytest.raises(writer.SchemaMetadataWriteError):
        writer.write_versioned_schema_metadata(valid_document(), destination)
    assert destination.read_text(encoding="utf-8") == "old"


def test_parent_directory_failure_is_reported(tmp_path, monkeypatch):
    destination = tmp_path / "nested" / "schema.json"
    monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: (_ for _ in ()).throw(
        OSError("mkdir failure")
    ))
    with pytest.raises(writer.SchemaMetadataWriteError):
        writer.write_versioned_schema_metadata(valid_document(), destination)


def test_temporary_file_creation_failure_preserves_destination(tmp_path, monkeypatch):
    destination = tmp_path / "schema.json"
    destination.write_text("old", encoding="utf-8")
    monkeypatch.setattr(writer.tempfile, "mkstemp", lambda **kwargs: (_ for _ in ()).throw(
        OSError("mkstemp failure")
    ))
    with pytest.raises(writer.SchemaMetadataWriteError):
        writer.write_versioned_schema_metadata(valid_document(), destination)
    assert destination.read_text(encoding="utf-8") == "old"


@pytest.mark.parametrize("failure", ["write", "flush", "fsync", "close"])
def test_temporary_write_boundary_failures_preserve_destination(
    tmp_path,
    monkeypatch,
    failure,
):
    destination = tmp_path / "schema.json"
    destination.write_text("old", encoding="utf-8")

    def fail_write(*args):
        raise OSError(f"{failure} failure")

    if failure == "fsync":
        monkeypatch.setattr(writer.os, "fsync", fail_write)
    elif failure == "write":
        monkeypatch.setattr(writer, "_write_temporary_file", fail_write)
    else:
        real_fdopen = writer.os.fdopen

        def wrapped_fdopen(*args, **kwargs):
            handle = real_fdopen(*args, **kwargs)
            if failure == "flush":
                handle.flush = fail_write
            else:
                handle.close = fail_write
            return handle

        monkeypatch.setattr(writer.os, "fdopen", wrapped_fdopen)

    with pytest.raises(writer.SchemaMetadataWriteError):
        writer.write_versioned_schema_metadata(valid_document(), destination)
    assert destination.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".*.tmp"))


def test_readback_parse_failure_preserves_destination(tmp_path, monkeypatch):
    destination = tmp_path / "schema.json"
    destination.write_text("old", encoding="utf-8")
    monkeypatch.setattr(writer, "_readback_document", lambda path: (_ for _ in ()).throw(
        ValueError("duplicate JSON object key in readback")
    ))
    with pytest.raises(writer.SchemaMetadataWriteError):
        writer.write_versioned_schema_metadata(valid_document(), destination)
    assert destination.read_text(encoding="utf-8") == "old"


def test_readback_fingerprint_mismatch_preserves_destination(tmp_path, monkeypatch):
    destination = tmp_path / "schema.json"
    destination.write_text("old", encoding="utf-8")
    real_fingerprint = writer.fingerprint_schema_metadata
    calls = {"count": 0}

    def mismatch_after_prevalidation(value):
        calls["count"] += 1
        return real_fingerprint(value) if calls["count"] == 1 else "b" * 64

    monkeypatch.setattr(writer, "fingerprint_schema_metadata", mismatch_after_prevalidation)
    with pytest.raises(writer.SchemaMetadataWriteError):
        writer.write_versioned_schema_metadata(valid_document(), destination)
    assert destination.read_text(encoding="utf-8") == "old"


def test_readback_structural_mismatch_preserves_destination(tmp_path, monkeypatch):
    destination = tmp_path / "schema.json"
    destination.write_text("old", encoding="utf-8")
    altered = valid_document()
    altered["tables"]["Catalog.Item"]["description"] = "altered"
    monkeypatch.setattr(writer, "_readback_document", lambda path: altered)
    with pytest.raises(writer.SchemaMetadataWriteError):
        writer.write_versioned_schema_metadata(valid_document(), destination)
    assert destination.read_text(encoding="utf-8") == "old"


def test_replace_failure_preserves_destination_and_cleans_temp(tmp_path, monkeypatch):
    destination = tmp_path / "schema.json"
    destination.write_text("old", encoding="utf-8")
    monkeypatch.setattr(writer.os, "replace", lambda *args: (_ for _ in ()).throw(
        OSError("replace failure")
    ))
    with pytest.raises(writer.SchemaMetadataWriteError):
        writer.write_versioned_schema_metadata(valid_document(), destination)
    assert destination.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".*.tmp"))


def test_cleanup_failure_does_not_mask_primary_failure(tmp_path, monkeypatch):
    destination = tmp_path / "schema.json"
    destination.write_text("old", encoding="utf-8")
    monkeypatch.setattr(writer.os, "replace", lambda *args: (_ for _ in ()).throw(
        OSError("primary replace failure")
    ))
    monkeypatch.setattr(Path, "unlink", lambda *args, **kwargs: (_ for _ in ()).throw(
        OSError("cleanup failure")
    ))
    with pytest.raises(writer.SchemaMetadataWriteError) as error:
        writer.write_versioned_schema_metadata(valid_document(), destination)
    assert isinstance(error.value.__cause__, OSError)
    assert "primary replace failure" in str(error.value.__cause__)
