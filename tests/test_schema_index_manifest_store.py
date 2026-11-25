from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

import ssa.schema_index_manifest_store as store
from ssa.schema_index_manifest import (
    SchemaIndexDocumentEvidence,
    SchemaIndexManifest,
    SchemaIndexManifestError,
    SchemaIndexSource,
)


STARTED = datetime(2026, 11, 25, 12, 0, 0, tzinfo=timezone.utc)
COMPLETED = datetime(2026, 11, 25, 12, 1, 0, tzinfo=timezone.utc)


def source():
    return SchemaIndexSource(
        database="AdventureWorks2025",
        metadata_format_version=2,
        content_fingerprint="a" * 64,
        snapshot_id="capture-1",
        captured_at_utc=STARTED,
    )


def documents(count=2):
    return SchemaIndexDocumentEvidence(
        generation_format_version=1,
        expected_count=count,
        id_digest="b" * 64,
        content_digest="c" * 64,
    )


def manifest(status="complete", count=2):
    return SchemaIndexManifest(
        status=status,
        source=source(),
        documents=documents(count),
        build_id="build-1",
        started_at_utc=STARTED,
        completed_at_utc=COMPLETED if status == "complete" else None,
        embedding_identity="embedding-v1",
    )


def test_write_and_read_incomplete_manifest(tmp_path):
    destination = tmp_path / "nested" / "manifest.json"
    value = manifest("incomplete")
    assert store.write_index_manifest(value, destination) == destination
    assert store.read_index_manifest(destination) == value


def test_write_and_read_complete_manifest_with_string_path(tmp_path):
    destination = str(tmp_path / "manifest.json")
    value = manifest()
    returned = store.write_index_manifest(value, destination)
    assert returned == Path(destination)
    assert store.read_index_manifest(destination) == value


def test_serialized_bytes_are_deterministic_and_unicode_is_preserved(tmp_path):
    first = tmp_path / "one.json"
    second = tmp_path / "two.json"
    value = manifest()
    store.write_index_manifest(value, first)
    store.write_index_manifest(value, second)
    assert first.read_bytes() == second.read_bytes()
    assert b"AdventureWorks2025" in first.read_bytes()
    assert first.read_bytes().endswith(b"}\n") is False


@pytest.mark.parametrize("first, second", [("complete", "incomplete"), ("incomplete", "complete")])
def test_existing_manifest_can_be_replaced_between_states(tmp_path, first, second):
    destination = tmp_path / "manifest.json"
    store.write_index_manifest(manifest(first), destination)
    store.write_index_manifest(manifest(second), destination)
    assert store.read_index_manifest(destination).status == second


@pytest.mark.parametrize(
    "payload, exception",
    [
        ("{", store.SchemaIndexManifestReadError),
        ('{"status": "complete", "status": "incomplete"}', store.SchemaIndexManifestReadError),
        ('{"value": NaN}', store.SchemaIndexManifestReadError),
        ("not utf8", store.SchemaIndexManifestReadError),
    ],
)
def test_strict_loading_rejects_invalid_files(tmp_path, payload, exception):
    path = tmp_path / "manifest.json"
    if payload == "not utf8":
        path.write_bytes(b"\xff\xfe")
    else:
        path.write_text(payload, encoding="utf-8")
    with pytest.raises(exception):
        store.read_index_manifest(path)


def test_missing_file_has_distinct_exception(tmp_path):
    with pytest.raises(store.SchemaIndexManifestNotFoundError):
        store.read_index_manifest(tmp_path / "missing.json")


def test_invalid_manifest_evidence_is_rejected(tmp_path):
    path = tmp_path / "manifest.json"
    value = store.serialize_index_manifest(manifest())
    value["status"] = "unknown"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(store.SchemaIndexManifestReadError):
        store.read_index_manifest(path)


def test_invalid_manifest_is_rejected_before_destination_change(tmp_path):
    destination = tmp_path / "manifest.json"
    destination.write_text("old", encoding="utf-8")
    invalid = object.__new__(SchemaIndexManifest)
    with pytest.raises(store.SchemaIndexManifestStoreError):
        store.write_index_manifest(invalid, destination)
    assert destination.read_text(encoding="utf-8") == "old"


def test_temporary_file_is_same_directory_and_removed_after_success(tmp_path, monkeypatch):
    observed = {}
    real_mkstemp = store.tempfile.mkstemp

    def capture(*args, **kwargs):
        observed["directory"] = Path(kwargs["dir"])
        result = real_mkstemp(*args, **kwargs)
        observed["temporary"] = Path(result[1])
        return result

    monkeypatch.setattr(store.tempfile, "mkstemp", capture)
    destination = tmp_path / "manifest.json"
    store.write_index_manifest(manifest(), destination)
    assert observed["directory"] == tmp_path
    assert not observed["temporary"].exists()
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("failure", ["mkdir", "mkstemp", "write", "flush", "fsync", "readback", "replace"])
def test_failures_preserve_existing_destination(tmp_path, monkeypatch, failure):
    destination = tmp_path / "manifest.json"
    destination.write_text("old", encoding="utf-8")

    if failure == "mkdir":
        monkeypatch.setattr(Path, "mkdir", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("mkdir")))
    elif failure == "mkstemp":
        monkeypatch.setattr(store.tempfile, "mkstemp", lambda **kwargs: (_ for _ in ()).throw(OSError("mkstemp")))
    elif failure == "write":
        monkeypatch.setattr(store.os, "fdopen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("write")))
    elif failure == "flush":
        real_fdopen = store.os.fdopen
        def fdopen_flush(*args, **kwargs):
            handle = real_fdopen(*args, **kwargs)
            handle.flush = lambda: (_ for _ in ()).throw(OSError("flush"))
            return handle
        monkeypatch.setattr(store.os, "fdopen", fdopen_flush)
    elif failure == "fsync":
        monkeypatch.setattr(store.os, "fsync", lambda *args: (_ for _ in ()).throw(OSError("fsync")))
    elif failure == "readback":
        monkeypatch.setattr(store, "read_index_manifest", lambda *args: (_ for _ in ()).throw(store.SchemaIndexManifestReadError("readback")))
    else:
        monkeypatch.setattr(store.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("replace")))

    with pytest.raises(store.SchemaIndexManifestStoreError):
        store.write_index_manifest(manifest(), destination)
    assert destination.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(".*.tmp"))


def test_round_trip_mismatch_preserves_destination(tmp_path, monkeypatch):
    destination = tmp_path / "manifest.json"
    destination.write_text("old", encoding="utf-8")
    different = manifest()
    monkeypatch.setattr(store, "read_index_manifest", lambda path: different.__class__(
        status="incomplete",
        source=different.source,
        documents=different.documents,
        build_id=different.build_id,
        started_at_utc=different.started_at_utc,
    ))
    with pytest.raises(store.SchemaIndexManifestStoreError):
        store.write_index_manifest(manifest(), destination)
    assert destination.read_text(encoding="utf-8") == "old"


def test_cleanup_failure_does_not_mask_primary_failure(tmp_path, monkeypatch):
    destination = tmp_path / "manifest.json"
    destination.write_text("old", encoding="utf-8")
    monkeypatch.setattr(store.os, "replace", lambda *args: (_ for _ in ()).throw(OSError("primary replace")))
    monkeypatch.setattr(Path, "unlink", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cleanup")))
    with pytest.raises(store.SchemaIndexManifestStoreError) as error:
        store.write_index_manifest(manifest(), destination)
    assert isinstance(error.value.__cause__, OSError)
    assert "primary replace" in str(error.value.__cause__)


def test_write_does_not_generate_or_change_manifest_evidence(tmp_path):
    value = manifest()
    before = value
    store.write_index_manifest(value, tmp_path / "manifest.json")
    assert value == before
