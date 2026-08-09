"""Team-name aliases: The Odds API's team names → football-data.co.uk's team names.

Covers every league in ``src.config.SUPPORTED_LEAGUES``, not just the current
production allowlist — the production leagues may change, and this table
should not need restructuring when they do.

Built by diffing each league's Odds API team list against its football-data.co.uk
team list. Production leagues (E0/N1/P1/G1) were diffed against a live Odds API
call on 2026-08-09; the rest await a live call once they're actually fetched (see
Task 2's ``fetch_pinnacle_odds`` unmatched-team log lines). Teams whose names
already match exactly need no entry. Ambiguous or diacritic-heavy names
(Portuguese, Greek, Turkish clubs) are resolved by hand here, never
fuzzy-matched — a wrong match would silently misprice a bet.
"""

ODDS_API_TEAM_ALIASES: dict[str, dict[str, str]] = {
    "E0": {
        "Manchester United": "Man United",
        "Manchester City": "Man City",
        "Newcastle United": "Newcastle",
        "Tottenham Hotspur": "Tottenham",
        "Brighton and Hove Albion": "Brighton",
        "Leeds United": "Leeds",
        "Nottingham Forest": "Nott'm Forest",
    },
    "D1": {},
    "SP1": {},
    "I1": {},
    "F1": {},
    "N1": {
        "FC Utrecht": "Utrecht",
        "FC Twente Enschede": "Twente",
        "FC Zwolle": "Zwolle",
    },
    "P1": {
        "FC Porto": "Porto",
        "Moreirense FC": "Moreirense",
        "Braga": "Sp Braga",
        "Rio Ave FC": "Rio Ave",
        "Sporting Lisbon": "Sp Lisbon",
        "Vitória SC": "Guimaraes",
    },
    "G1": {
        "AEK Athens": "AEK",
        "Aris Thessaloniki": "Aris",
        "Olympiakos Piraeus": "Olympiakos",
        "Atromitos Athens": "Atromitos",
        "Volos FC": "Volos NFC",
        "AE Kifisia FC": "Kifisia",
        "PAOK Thessaloniki": "PAOK",
        "Levadiakos": "Levadeiakos",
        "Panetolikos Agrinio": "Panetolikos",
    },
    "SC0": {},
    "B1": {},
    "T1": {},
}
