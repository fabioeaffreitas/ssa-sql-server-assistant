"""Presentation policy for optional SQL ordering advisories."""


ORDERING_RISK_MESSAGE = (
    "Ordering note: This TOP query may select different rows when sort values "
    "are tied. This is an advisory about result stability, not a SQL safety "
    "or execution error."
)


def ordering_advisory_message(diagnostics: object) -> str | None:
    """Return the neutral UI message for a valid ordering-risk advisory."""

    if not isinstance(diagnostics, dict):
        return None

    ordering = diagnostics.get("ordering")
    if not isinstance(ordering, dict):
        return None

    if ordering.get("status") != "risk":
        return None

    return ORDERING_RISK_MESSAGE


__all__ = ["ORDERING_RISK_MESSAGE", "ordering_advisory_message"]
