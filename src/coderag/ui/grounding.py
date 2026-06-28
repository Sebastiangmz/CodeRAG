"""Grounding status formatting for UI surfaces."""


def format_grounding_status(
    *,
    grounded: bool,
    has_citations: bool,
    has_citation_verifications: bool,
) -> str:
    """Return user-facing grounding status from verified citation state."""
    if grounded:
        return "Grounded (verified citations)"
    if has_citations and has_citation_verifications:
        return "Not grounded (unverified citations)"
    return "Not grounded (no verified citations)"
