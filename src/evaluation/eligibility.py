"""Shared execution filters for live and shadow value-bet selection."""

from src.config import DEFAULT_MAX_EDGE, DEFAULT_MAX_ODDS, DEFAULT_MAX_OVERROUND


def is_execution_eligible(
    *,
    edge: float,
    threshold: float,
    b365_odds: float,
    fixture_overround: float,
    max_edge: float = DEFAULT_MAX_EDGE,
    max_odds: float = DEFAULT_MAX_ODDS,
    max_overround: float = DEFAULT_MAX_OVERROUND,
) -> bool:
    """Return whether a model edge passes every production execution filter."""
    return (
        edge > threshold
        and edge <= max_edge
        and b365_odds <= max_odds
        and fixture_overround <= max_overround
    )
