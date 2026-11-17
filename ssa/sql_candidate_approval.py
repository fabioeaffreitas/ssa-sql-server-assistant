"""Offline identity and state coordination for proposed SQL candidates.

This module is deliberately not an authorization system. It provides a
deterministic policy-testing model for binding one approval to one exact SQL
candidate. It has no persistence, process coordination, or SQL execution.
"""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from typing import Literal


CandidateState = Literal[
    "pending",
    "approved",
    "declined",
    "invalidated",
    "expired",
    "consumed",
]

IDENTITY_VERSION = 1


class CandidateApprovalError(RuntimeError):
    """Base error for invalid candidate or state operations."""


class CandidateMismatchError(CandidateApprovalError):
    """The supplied candidate does not match the registered candidate."""


class CandidateNotFoundError(CandidateApprovalError):
    """No candidate is registered for the supplied identifier."""


class InvalidTransitionError(CandidateApprovalError):
    """The requested state transition is not permitted."""


class CandidateExpiredError(CandidateApprovalError):
    """The candidate is expired at the supplied clock value."""


class InvalidCandidateError(ValueError):
    """Candidate fields or timestamps are invalid."""


@dataclass(frozen=True)
class SQLCandidate:
    """Immutable exact identity for one generated/proposed SQL pair."""

    question: str
    original_sql: str
    proposed_sql: str
    database_identity: str
    metadata_fingerprint: str
    generation_id: str
    session_id: str
    created_at: datetime
    expires_at: datetime
    candidate_id: str = ""

    def __post_init__(self) -> None:
        _validate_text_fields(self)
        created_at = _utc_timestamp(self.created_at)
        expires_at = _utc_timestamp(self.expires_at)
        if expires_at < created_at:
            raise InvalidCandidateError("expires_at must not precede created_at.")
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "expires_at", expires_at)

        expected_id = _candidate_fingerprint(self)
        if self.candidate_id and self.candidate_id != expected_id:
            raise InvalidCandidateError("candidate_id does not match candidate fields.")
        object.__setattr__(self, "candidate_id", expected_id)


@dataclass(frozen=True)
class CandidateApproval:
    """Immutable snapshot of one candidate's current state."""

    candidate: SQLCandidate
    state: CandidateState
    changed_at: datetime


@dataclass(frozen=True)
class SQLSelection:
    """Attribution returned only after successful one-time consumption."""

    candidate_id: str
    original_sql: str
    proposed_sql: str
    selected_sql: str
    state: Literal["consumed"] = "consumed"


def create_candidate(
    *,
    question: str,
    original_sql: str,
    proposed_sql: str,
    database_identity: str,
    metadata_fingerprint: str,
    generation_id: str,
    session_id: str,
    created_at: datetime,
    expires_at: datetime,
) -> SQLCandidate:
    """Create one immutable candidate and compute its versioned fingerprint."""

    return SQLCandidate(
        question=question,
        original_sql=original_sql,
        proposed_sql=proposed_sql,
        database_identity=database_identity,
        metadata_fingerprint=metadata_fingerprint,
        generation_id=generation_id,
        session_id=session_id,
        created_at=created_at,
        expires_at=expires_at,
    )


class CandidateApprovalCoordinator:
    """In-memory state machine for deterministic approval-policy testing."""

    def __init__(self) -> None:
        self._records: dict[str, CandidateApproval] = {}

    def register(
        self,
        candidate: SQLCandidate,
        *,
        now: datetime,
    ) -> CandidateApproval:
        """Register a pending candidate, rejecting duplicate identities."""

        current_time = _utc_timestamp(now)
        _ensure_not_expired(candidate, current_time)
        existing = self._records.get(candidate.candidate_id)
        if existing is not None:
            if existing.candidate != candidate:
                raise CandidateMismatchError("candidate identity mismatch.")
            return existing
        record = CandidateApproval(candidate, "pending", current_time)
        self._records[candidate.candidate_id] = record
        return record

    def get(
        self,
        candidate: SQLCandidate,
        *,
        candidate_id: str | None = None,
    ) -> CandidateApproval:
        """Return a record only when the full candidate identity matches."""

        return self._lookup(candidate, candidate_id)

    def approve(
        self,
        candidate: SQLCandidate,
        *,
        now: datetime,
        candidate_id: str | None = None,
    ) -> CandidateApproval:
        """Approve one pending candidate; repeated approval is idempotent."""

        record = self._active_record(candidate, now, candidate_id)
        if record.state == "approved":
            return record
        if record.state != "pending":
            raise InvalidTransitionError(
                f"cannot approve candidate in state {record.state}."
            )
        return self._set_state(record, "approved", now)

    def decline(
        self,
        candidate: SQLCandidate,
        *,
        now: datetime,
        candidate_id: str | None = None,
    ) -> CandidateApproval:
        """Decline one pending candidate."""

        record = self._active_record(candidate, now, candidate_id)
        if record.state != "pending":
            raise InvalidTransitionError(
                f"cannot decline candidate in state {record.state}."
            )
        return self._set_state(record, "declined", now)

    def invalidate(
        self,
        candidate: SQLCandidate,
        *,
        now: datetime,
        candidate_id: str | None = None,
    ) -> CandidateApproval:
        """Invalidate a pending or approved candidate."""

        record = self._active_record(candidate, now, candidate_id)
        if record.state not in {"pending", "approved"}:
            raise InvalidTransitionError(
                f"cannot invalidate candidate in state {record.state}."
            )
        return self._set_state(record, "invalidated", now)

    def consume(
        self,
        candidate: SQLCandidate,
        *,
        now: datetime,
        candidate_id: str | None = None,
    ) -> SQLSelection:
        """Consume one approved candidate and return the exact proposed SQL."""

        record = self._active_record(candidate, now, candidate_id)
        if record.state != "approved":
            raise InvalidTransitionError(
                f"cannot consume candidate in state {record.state}."
            )
        self._set_state(record, "consumed", now)
        return SQLSelection(
            candidate_id=candidate.candidate_id,
            original_sql=candidate.original_sql,
            proposed_sql=candidate.proposed_sql,
            selected_sql=candidate.proposed_sql,
        )

    def _lookup(
        self,
        candidate: SQLCandidate,
        candidate_id: str | None,
    ) -> CandidateApproval:
        requested_id = candidate.candidate_id if candidate_id is None else candidate_id
        if requested_id != candidate.candidate_id:
            raise CandidateMismatchError("candidate identifier mismatch.")
        record = self._records.get(requested_id)
        if record is None:
            raise CandidateNotFoundError("candidate is not registered.")
        if record.candidate != candidate:
            raise CandidateMismatchError("candidate identity mismatch.")
        return record

    def _active_record(
        self,
        candidate: SQLCandidate,
        now: datetime,
        candidate_id: str | None,
    ) -> CandidateApproval:
        current_time = _utc_timestamp(now)
        record = self._lookup(candidate, candidate_id)
        if current_time >= candidate.expires_at and record.state not in {
            "expired",
            "consumed",
        }:
            record = self._set_state(record, "expired", current_time)
            raise CandidateExpiredError("candidate has expired.")
        return record

    def _set_state(
        self,
        record: CandidateApproval,
        state: CandidateState,
        now: datetime,
    ) -> CandidateApproval:
        changed_at = _utc_timestamp(now)
        updated = replace(record, state=state, changed_at=changed_at)
        self._records[record.candidate.candidate_id] = updated
        return updated


def _validate_text_fields(candidate: SQLCandidate) -> None:
    for name in (
        "question",
        "original_sql",
        "proposed_sql",
        "database_identity",
        "metadata_fingerprint",
        "generation_id",
        "session_id",
    ):
        value = getattr(candidate, name)
        if not isinstance(value, str) or not value:
            raise InvalidCandidateError(f"{name} must be a non-empty string.")


def _utc_timestamp(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise InvalidCandidateError("timestamps must be timezone-aware datetimes.")
    return value.astimezone(timezone.utc)


def _ensure_not_expired(candidate: SQLCandidate, now: datetime) -> None:
    if now >= candidate.expires_at:
        raise CandidateExpiredError("candidate has expired.")


def _candidate_fingerprint(candidate: SQLCandidate) -> str:
    payload = {
        "identity_version": IDENTITY_VERSION,
        "question": candidate.question,
        "original_sql": candidate.original_sql,
        "proposed_sql": candidate.proposed_sql,
        "database_identity": candidate.database_identity,
        "metadata_fingerprint": candidate.metadata_fingerprint,
        "generation_id": candidate.generation_id,
        "session_id": candidate.session_id,
        "created_at": candidate.created_at.isoformat(),
        "expires_at": candidate.expires_at.isoformat(),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


__all__ = [
    "CandidateApproval",
    "CandidateApprovalCoordinator",
    "CandidateApprovalError",
    "CandidateExpiredError",
    "CandidateMismatchError",
    "CandidateNotFoundError",
    "CandidateState",
    "IDENTITY_VERSION",
    "InvalidCandidateError",
    "InvalidTransitionError",
    "SQLCandidate",
    "SQLSelection",
    "create_candidate",
]
