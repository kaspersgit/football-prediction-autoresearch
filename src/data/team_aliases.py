"""Team-name aliases: The Odds API's team names → football-data.co.uk's team names.

Covers every league in ``src.config.SUPPORTED_LEAGUES``, not just the current
production allowlist — the production leagues may change, and this table
should not need restructuring when they do.

Built by diffing each league's Odds API team list against its football-data.co.uk
team list. Production leagues were diffed against a live Odds API call:
E0/N1/P1/G1 on 2026-08-09, F1 on 2026-08-10 (when it entered the allowlist).
The rest await a live call once they're actually fetched (see Task 2's
``fetch_pinnacle_odds`` unmatched-team log lines). Teams whose names already
match exactly need no entry. Ambiguous or diacritic-heavy names (Portuguese,
Greek, Turkish clubs) are resolved by hand here, never fuzzy-matched — a
wrong match would silently misprice a bet.
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
    "F1": {
        "RC Lens": "Lens",
        "AS Monaco": "Monaco",
        "Paris Saint Germain": "Paris SG",
    },
    "N1": {
        "FC Utrecht": "Utrecht",
        "FC Twente Enschede": "Twente",
        "FC Zwolle": "Zwolle",
        "SC Telstar": "Telstar",
        "NEC Nijmegen": "Nijmegen",
        "Fortuna Sittard": "For Sittard",
        "SC Cambuur": "Cambuur",
        "ADO Den Haag": "Den Haag",
    },
    "P1": {
        "FC Porto": "Porto",
        "Moreirense FC": "Moreirense",
        "Braga": "Sp Braga",
        "Rio Ave FC": "Rio Ave",
        "Sporting Lisbon": "Sp Lisbon",
        "Vitória SC": "Guimaraes",
        "CF Estrela": "Estrela",
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
