from datetime import datetime, timedelta, timezone

import pytest

from ssa.sql_candidate_approval import (
    CandidateApprovalCoordinator,
    CandidateExpiredError,
    CandidateMismatchError,
    CandidateNotFoundError,
    InvalidCandidateError,
    InvalidTransitionError,
    SQLCandidate,
    create_candidate,
)


UTC = timezone.utc
CREATED = datetime(2026, 11, 17, 10, 0, tzinfo=UTC)
EXPIRES = CREATED + timedelta(minutes=5)
BEFORE_EXPIRY = CREATED + timedelta(minutes=1)
AT_EXPIRY = EXPIRES


def candidate(**changes):
    values = {
        "question": "show top items",
        "original_sql": "SELECT TOP 10 ItemKey FROM Catalog.Item ORDER BY Price DESC",
        "proposed_sql": "SELECT TOP 10 ItemKey FROM Catalog.Item ORDER BY Price DESC, ItemKey ASC",
        "database_identity": "SyntheticDB",
        "metadata_fingerprint": "metadata-v1",
        "generation_id": "generation-1",
        "session_id": "session-1",
        "created_at": CREATED,
        "expires_at": EXPIRES,
    }
    values.update(changes)
    return create_candidate(**values)


def register(candidate_value=None):
    candidate_value = candidate_value or candidate()
    coordinator = CandidateApprovalCoordinator()
    record = coordinator.register(candidate_value, now=CREATED)
    return coordinator, candidate_value, record


def test_candidate_starts_pending_and_is_immutable():
    value = candidate()
    coordinator, _, record = register(value)

    assert record.state == "pending"
    assert record.candidate == value
    with pytest.raises((AttributeError, TypeError)):
        value.question = "changed"
    assert coordinator.get(value).state == "pending"


def test_identical_inputs_have_identical_fingerprints():
    assert candidate().candidate_id == candidate().candidate_id


@pytest.mark.parametrize(
    "change",
    [
        {"question": "different"},
        {"original_sql": "SELECT 1"},
        {"proposed_sql": "SELECT 2"},
        {"database_identity": "OtherDB"},
        {"metadata_fingerprint": "metadata-v2"},
        {"generation_id": "generation-2"},
        {"session_id": "session-2"},
        {"original_sql": "SELECT  TOP 10 ItemKey FROM Catalog.Item ORDER BY Price DESC"},
    ],
)
def test_identity_changes_for_each_bound_field(change):
    assert candidate().candidate_id != candidate(**change).candidate_id


def test_json_identity_encoding_distinguishes_empty_and_none_at_construction_boundary():
    with pytest.raises(InvalidCandidateError):
        candidate(metadata_fingerprint="")
    with pytest.raises(InvalidCandidateError):
        candidate(metadata_fingerprint=None)


def test_candidate_id_cannot_be_supplied_for_different_fields():
    with pytest.raises(InvalidCandidateError):
        SQLCandidate(
            question="show top items",
            original_sql="SELECT 1",
            proposed_sql="SELECT 2",
            database_identity="SyntheticDB",
            metadata_fingerprint="metadata-v1",
            generation_id="generation-1",
            session_id="session-1",
            created_at=CREATED,
            expires_at=EXPIRES,
            candidate_id="not-the-fingerprint",
        )


@pytest.mark.parametrize(
    "field",
    ["question", "original_sql", "proposed_sql", "database_identity", "metadata_fingerprint", "generation_id", "session_id"],
)
def test_required_identity_fields_must_be_nonempty(field):
    with pytest.raises(InvalidCandidateError):
        candidate(**{field: ""})


def test_approval_and_one_time_consumption():
    coordinator, value, record = register()
    approved = coordinator.approve(value, now=BEFORE_EXPIRY)
    assert approved.state == "approved"

    selection = coordinator.consume(value, now=BEFORE_EXPIRY)
    assert selection.selected_sql == value.proposed_sql
    assert selection.original_sql == value.original_sql
    assert selection.proposed_sql == value.proposed_sql
    assert selection.candidate_id == value.candidate_id
    assert coordinator.get(value).state == "consumed"

    with pytest.raises(InvalidTransitionError):
        coordinator.consume(value, now=BEFORE_EXPIRY)


def test_repeated_approval_is_idempotent_and_grants_no_extra_use():
    coordinator, value, _ = register()
    first = coordinator.approve(value, now=BEFORE_EXPIRY)
    second = coordinator.approve(value, now=BEFORE_EXPIRY)
    assert first == second
    coordinator.consume(value, now=BEFORE_EXPIRY)
    with pytest.raises(InvalidTransitionError):
        coordinator.approve(value, now=BEFORE_EXPIRY)


def test_decline_is_terminal_and_cannot_be_consumed_or_reapproved():
    coordinator, value, _ = register()
    assert coordinator.decline(value, now=BEFORE_EXPIRY).state == "declined"
    with pytest.raises(InvalidTransitionError):
        coordinator.approve(value, now=BEFORE_EXPIRY)
    with pytest.raises(InvalidTransitionError):
        coordinator.consume(value, now=BEFORE_EXPIRY)


def test_invalidation_is_terminal_for_pending_and_approved_candidates():
    coordinator, value, _ = register()
    assert coordinator.invalidate(value, now=BEFORE_EXPIRY).state == "invalidated"
    with pytest.raises(InvalidTransitionError):
        coordinator.consume(value, now=BEFORE_EXPIRY)

    coordinator, value, _ = register()
    coordinator.approve(value, now=BEFORE_EXPIRY)
    assert coordinator.invalidate(value, now=BEFORE_EXPIRY).state == "invalidated"
    with pytest.raises(InvalidTransitionError):
        coordinator.consume(value, now=BEFORE_EXPIRY)


@pytest.mark.parametrize("operation", ["approve", "decline", "invalidate", "consume"])
def test_pending_only_allows_its_legal_transition(operation):
    coordinator, value, _ = register()
    if operation == "approve":
        assert coordinator.approve(value, now=BEFORE_EXPIRY).state == "approved"
    elif operation == "decline":
        assert coordinator.decline(value, now=BEFORE_EXPIRY).state == "declined"
    elif operation == "invalidate":
        assert coordinator.invalidate(value, now=BEFORE_EXPIRY).state == "invalidated"
    else:
        with pytest.raises(InvalidTransitionError):
            coordinator.consume(value, now=BEFORE_EXPIRY)


def test_expiry_boundary_is_now_greater_than_or_equal_to_expires_at():
    coordinator, value, _ = register()
    with pytest.raises(CandidateExpiredError):
        coordinator.approve(value, now=AT_EXPIRY)
    assert coordinator.get(value).state == "expired"

    coordinator, value, _ = register()
    coordinator.approve(value, now=BEFORE_EXPIRY)
    with pytest.raises(CandidateExpiredError):
        coordinator.consume(value, now=AT_EXPIRY)
    assert coordinator.get(value).state == "expired"


def test_expiry_after_creation_is_rejected_at_registration():
    expired = candidate(expires_at=CREATED)
    coordinator = CandidateApprovalCoordinator()
    with pytest.raises(CandidateExpiredError):
        coordinator.register(expired, now=CREATED)


@pytest.mark.parametrize("timestamp", [CREATED.replace(tzinfo=None), EXPIRES.replace(tzinfo=None)])
def test_naive_timestamps_are_rejected(timestamp):
    with pytest.raises(InvalidCandidateError):
        candidate(created_at=timestamp)


def test_non_utc_aware_timestamps_are_normalized_to_utc():
    plus_two = timezone(timedelta(hours=2))
    value = candidate(
        created_at=CREATED.astimezone(plus_two),
        expires_at=EXPIRES.astimezone(plus_two),
    )
    assert value.created_at.tzinfo == UTC
    assert value.expires_at.tzinfo == UTC


@pytest.mark.parametrize(
    "change",
    [
        {"question": "other"},
        {"original_sql": "SELECT 9"},
        {"proposed_sql": "SELECT 9"},
        {"database_identity": "OtherDB"},
        {"metadata_fingerprint": "other-metadata"},
        {"generation_id": "other-generation"},
        {"session_id": "other-session"},
    ],
)
def test_full_candidate_matching_rejects_mismatches(change):
    coordinator, value, _ = register()
    with pytest.raises(CandidateMismatchError):
        coordinator.approve(
            candidate(**change),
            now=BEFORE_EXPIRY,
            candidate_id=value.candidate_id,
        )


def test_explicit_candidate_id_mismatch_is_rejected():
    coordinator, value, _ = register()
    with pytest.raises(CandidateMismatchError):
        coordinator.approve(value, now=BEFORE_EXPIRY, candidate_id="wrong")


def test_unknown_candidate_is_rejected():
    coordinator = CandidateApprovalCoordinator()
    with pytest.raises(CandidateNotFoundError):
        coordinator.get(candidate())


def test_duplicate_registration_requires_same_full_identity():
    coordinator, value, _ = register()
    assert coordinator.register(value, now=BEFORE_EXPIRY).state == "pending"
    changed = candidate(question="changed")
    assert coordinator.register(changed, now=BEFORE_EXPIRY).state == "pending"
    assert changed.candidate_id != value.candidate_id


def test_selection_does_not_execute_sql_or_write_files():
    coordinator, value, _ = register()
    coordinator.approve(value, now=BEFORE_EXPIRY)
    selection = coordinator.consume(value, now=BEFORE_EXPIRY)
    assert selection.selected_sql == value.proposed_sql
    assert selection.state == "consumed"
