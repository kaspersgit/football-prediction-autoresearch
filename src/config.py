"""Verified betting configuration shared by evaluation and prediction."""

LEAGUE_NAMES = {
    "E0": "England",
    "D1": "Germany",
    "SP1": "Spain",
    "I1": "Italy",
    "F1": "France",
    "N1": "Netherlands",
    "P1": "Portugal",
    "G1": "Greece",
    "SC0": "Scotland",
    "B1": "Belgium",
    "T1": "Turkey",
}

SUPPORTED_LEAGUES = tuple(LEAGUE_NAMES)
PRODUCTION_LEAGUES = frozenset({"E0", "N1", "P1", "G1"})

DEFAULT_MAX_ODDS = 5.0
DEFAULT_MAX_EDGE = 0.20
DEFAULT_MAX_OVERROUND = 0.07

# Compatibility helper for APIs expressed as a skip set. Evaluation must not use
# this value; it is derived solely from the production allowlist.
EXCLUDED_BETTING_LEAGUES = frozenset(set(SUPPORTED_LEAGUES) - PRODUCTION_LEAGUES)


def filter_production_fixtures(fixtures):
    """Return only fixtures explicitly enabled for production inference."""
    return fixtures[fixtures["league"].isin(PRODUCTION_LEAGUES)].reset_index(drop=True)
