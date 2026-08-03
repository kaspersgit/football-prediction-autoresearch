"""Verified betting configuration shared by evaluation and prediction."""

DEFAULT_MAX_ODDS = 5.0
DEFAULT_MAX_EDGE = 0.20
DEFAULT_MAX_OVERROUND = 0.07

# These leagues remain available for modelling and reports but are excluded from
# the default backtest betting portfolio.
EXCLUDED_BETTING_LEAGUES = frozenset({"F1", "SP1", "D1", "I1", "SC0", "B1", "T1"})
