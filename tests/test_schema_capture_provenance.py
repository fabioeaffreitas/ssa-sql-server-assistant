from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
import copy

import pytest

from ssa.schema_capture_provenance import (
    DatabaseIdentityMismatchError,
    InvalidFingerprintError,
    InvalidServerIdentityError,
    InvalidTimestampError,
    METADATA_FORMAT_VERSION,
    MissingProvenanceError,
    SQLServerIdentityEvidence,
    SchemaCaptureProvenance,
    UnsupportedMetadataVersionError,
    parse_capture_provenance,
    serialize_capture_provenance,
)


FINGERPRINT = "a" * 64
CAPTURED_AT = datetime(2026, 11, 19, 12, 34, 56, 123456, tzinfo=timezone.utc)


def valid_provenance(**overrides):
    values = {
        "snapshot_id": "capture-20261119-001",
        "captured_at_utc": CAPTURED_AT,
        "database": "AdventureWorks2025",
        "server_identity": SQLServerIdentityEvidence(
            server_name="SQLHOST\\SQLEXPRESS",
            instance_name="SQLEXPRESS",
            database_id=7,
            database_guid="11111111-2222-3333-4444-555555555555",
        ),
        "content_fingerprint": FINGERPRINT,
        "server_version": "17.0.1000.7",
        "capture_tool_version": "ssa-schema-export/1",
    }
    values.update(overrides)
    return SchemaCaptureProvenance(**values)


def valid_root(provenance=None):
    fragment = serialize_capture_provenance(provenance or valid_provenance())
    return {
        **fragment,
        "tables": {},
        "views": {},
        "table_count": 0,
        "view_count": 0,
        "foreign_key_count": 0,
    }


def test_valid_versioned_provenance_round_trips_with_canonical_timestamp():
    provenance = valid_provenance()
    serialized = serialize_capture_provenance(provenance)

    assert serialized["metadata_format_version"] == METADATA_FORMAT_VERSION
    assert serialized["database"] == "AdventureWorks2025"
    assert serialized["provenance"]["captured_at_utc"] == (
        "2026-11-19T12:34:56.123456Z"
    )
    assert parse_capture_provenance(serialized | {"tables": {}, "views": {}}) == provenance


def test_provenance_and_server_identity_are_immutable():
    provenance = valid_provenance()
    identity = provenance.server_identity

    with pytest.raises(FrozenInstanceError):
        provenance.database = "OtherDatabase"
    with pytest.raises(FrozenInstanceError):
        identity.server_name = "OtherServer"


def test_naive_timestamp_is_rejected():
    with pytest.raises(InvalidTimestampError):
        valid_provenance(
            captured_at_utc=datetime(2026, 11, 19, 12, 34, 56, 123456)
        )


def test_non_utc_timestamp_is_rejected():
    # A fixed non-zero offset is intentionally used instead of converting it.
    with pytest.raises(InvalidTimestampError):
        valid_provenance(
            captured_at_utc=datetime(
                2026,
                11,
                19,
                12,
                34,
                tzinfo=timezone(timedelta(hours=1)),
            )
        )


@pytest.mark.parametrize(
    "timestamp_text",
    [
        None,
        "2026-11-19T12:34:56Z",
        "2026-11-19T12:34:56.123456+00:00",
        "not-a-timestamp",
        "2026-02-30T12:34:56.123456Z",
    ],
)
def test_malformed_or_missing_timestamps_are_rejected(timestamp_text):
    root = valid_root()
    if timestamp_text is None:
        del root["provenance"]["captured_at_utc"]
    else:
        root["provenance"]["captured_at_utc"] = timestamp_text

    with pytest.raises(InvalidTimestampError):
        parse_capture_provenance(root)


def test_snapshot_id_is_required_and_explicit():
    with pytest.raises(ValueError):
        valid_provenance(snapshot_id="")
    with pytest.raises(ValueError):
        valid_provenance(snapshot_id=123)
    with pytest.raises(ValueError):
        valid_provenance(snapshot_id=" capture-1")


@pytest.mark.parametrize("fingerprint", ["", "a" * 63, "A" * 64, "g" * 64, 123])
def test_invalid_fingerprints_are_rejected(fingerprint):
    with pytest.raises(InvalidFingerprintError):
        valid_provenance(content_fingerprint=fingerprint)


def test_missing_or_malformed_server_identity_is_rejected():
    with pytest.raises(InvalidServerIdentityError):
        valid_provenance(server_identity=None)
    with pytest.raises(InvalidServerIdentityError):
        valid_provenance(server_identity=SQLServerIdentityEvidence(server_name=""))
    with pytest.raises(InvalidServerIdentityError):
        valid_provenance(
            server_identity=SQLServerIdentityEvidence(
                server_name="server",
                database_id=True,
            )
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("server_version", 17),
        ("server_version", " "),
        ("capture_tool_version", 1),
        ("capture_tool_version", " "),
    ],
)
def test_invalid_optional_diagnostics_are_rejected(field, value):
    with pytest.raises(ValueError):
        valid_provenance(**{field: value})


def test_root_and_provenance_database_must_match_case_insensitively():
    root = valid_root()
    root["database"] = "adventureworks2025"
    assert parse_capture_provenance(root).database == "AdventureWorks2025"

    root = valid_root()
    root["provenance"]["database"] = "OtherDatabase"
    with pytest.raises(DatabaseIdentityMismatchError):
        parse_capture_provenance(root)


def test_missing_root_database_is_rejected():
    root = valid_root()
    del root["database"]
    with pytest.raises(ValueError):
        parse_capture_provenance(root)


def test_unsupported_version_is_rejected():
    root = valid_root()
    root["metadata_format_version"] = 3
    with pytest.raises(UnsupportedMetadataVersionError):
        parse_capture_provenance(root)


def test_legacy_unversioned_metadata_has_no_invented_provenance():
    with pytest.raises(MissingProvenanceError):
        parse_capture_provenance(
            {
                "database": "AdventureWorks2025",
                "tables": {},
                "views": {},
            }
        )


def test_missing_provenance_is_classified():
    root = valid_root()
    del root["provenance"]
    with pytest.raises(MissingProvenanceError):
        parse_capture_provenance(root)


def test_unknown_optional_fields_are_ignored_without_being_serialized():
    root = valid_root()
    root["provenance"]["future_optional_field"] = {"enabled": True}
    parsed = parse_capture_provenance(root)
    serialized = serialize_capture_provenance(parsed)
    assert "future_optional_field" not in serialized["provenance"]


def test_serialization_and_parsing_do_not_mutate_input():
    provenance = valid_provenance()
    root = valid_root(provenance)
    before = copy.deepcopy(root)

    parse_capture_provenance(root)
    assert root == before

    serialized = serialize_capture_provenance(provenance)
    assert serialized["provenance"]["snapshot_id"] == provenance.snapshot_id


def test_identical_content_can_have_distinct_capture_event_identity():
    first = valid_provenance(snapshot_id="capture-1")
    second = valid_provenance(snapshot_id="capture-2")

    assert first.content_fingerprint == second.content_fingerprint
    assert first.snapshot_id != second.snapshot_id
    assert serialize_capture_provenance(first)["provenance"]["captured_at_utc"] == (
        serialize_capture_provenance(second)["provenance"]["captured_at_utc"]
    )


def test_validation_has_no_current_clock_dependency():
    first = valid_provenance()
    second = valid_provenance()
    assert first == second
    assert serialize_capture_provenance(first) == serialize_capture_provenance(second)


def test_server_identity_is_evidence_not_a_database_name_substitute():
    provenance = valid_provenance(
        server_identity=SQLServerIdentityEvidence(server_name="server-only")
    )
    assert provenance.server_identity.server_name == "server-only"
    assert provenance.database == "AdventureWorks2025"
