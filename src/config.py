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
# Re-chosen 2026-08-10 (EXP-20260810-002) from a production-methodology screen (real
# per-league calibrated thresholds + max-edge/overround caps + the Pinnacle-confirmation
# filter) across all 11 leagues: Portugal dropped (flat/slightly negative under the
# filter), France added on the expectation that live Predict runs close to kickoff
# will trend closer to Pinnacle's closing line than the opening-odds worst case tested.
# Portugal re-added 2026-08-28 (EXP-20260828-001) per explicit user direction: the
# all-leagues-production re-screen showed it flat-to-slightly-positive (+3.01% ROI,
# 135 bets) rather than clearly negative like the other seven non-production leagues —
# the weakest of the five kept leagues, but not the same "don't add" case as the rest.
PRODUCTION_LEAGUES = frozenset({"E0", "N1", "G1", "F1", "P1"})

DEFAULT_MAX_ODDS = 5.0
DEFAULT_MAX_EDGE = 0.20
DEFAULT_MAX_OVERROUND = 0.07
DEFAULT_PINNACLE_CONFIRMATION_MARGIN = 0.015

# Compatibility helper for APIs expressed as a skip set. Evaluation must not use
# this value; it is derived solely from the production allowlist.
EXCLUDED_BETTING_LEAGUES = frozenset(set(SUPPORTED_LEAGUES) - PRODUCTION_LEAGUES)


def filter_production_fixtures(fixtures):
    """Return only fixtures explicitly enabled for production inference."""
    return fixtures[fixtures["league"].isin(PRODUCTION_LEAGUES)].reset_index(drop=True)
