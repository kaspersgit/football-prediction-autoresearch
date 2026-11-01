"""Team-name aliases: OddsPapi's participant names → football-data.co.uk's team names.

Separate table from ``src.data.team_aliases.ODDS_API_TEAM_ALIASES`` — OddsPapi and The
Odds API pull from different name databases, so even though some entries happen to
coincide (e.g. both call Utrecht's club "FC Utrecht"), the two tables diverge often
enough (Brighton's "&" vs "and", Oporto's "Olympiacos" vs "Olympiakos" spelling, full
legal names like "Paris Saint-Germain" vs "Paris Saint Germain") that merging them
would be a trap, not a shortcut.

Built 2026-11-01 by diffing a live ``GET /v4/odds-by-tournaments`` pull (both pinnacle
and bet365, all 11 SUPPORTED_LEAGUES, current matchweek) against each league's
football-data.co.uk team list (data/historical/<league>/<latest season>.csv). Teams
whose OddsPapi name already matches football-data.co.uk exactly need no entry here.

Resolved by hand only where the identification was unambiguous (I know these clubs);
never fuzzy-matched — a wrong match would silently misprice a bet. A handful of
OddsPapi names seen in the 2026-11-01 pull had no confident mapping (mostly clubs not
in this season's top flight, or spellings I wasn't sure of) and are deliberately left
out rather than guessed — see the list in each league's trailing comment. Re-diff
against a live pull before trusting a league whose comment flags open gaps.
"""

ODDSPAPI_TEAM_ALIASES: dict[str, dict[str, str]] = {
    "E0": {
        "AFC Bournemouth": "Bournemouth",
        "Arsenal FC": "Arsenal",
        "Brentford FC": "Brentford",
        "Brighton & Hove Albion": "Brighton",
        "Chelsea FC": "Chelsea",
        "Everton FC": "Everton",
        "Fulham FC": "Fulham",
        "Ipswich Town": "Ipswich",
        "Leeds United": "Leeds",
        "Liverpool FC": "Liverpool",
        "Manchester City": "Man City",
        "Manchester United": "Man United",
        "Newcastle United": "Newcastle",
        "Nottingham Forest": "Nott'm Forest",
        "Sunderland AFC": "Sunderland",
        "Tottenham Hotspur": "Tottenham",
    },
    # Coventry City, Hull City not in football-data's current E0 season (Championship) —
    # no alias needed unless/until promoted.
    "D1": {
        "Bayer Leverkusen": "Leverkusen",
        "Borussia Dortmund": "Dortmund",
        "Borussia Monchengladbach": "M'gladbach",
        "Eintracht Frankfurt": "Ein Frankfurt",
        "FC Augsburg": "Augsburg",
        "FSV Mainz": "Mainz",
        "Hamburger SV": "Hamburg",
        "SC Freiburg": "Freiburg",
        "TSG Hoffenheim": "Hoffenheim",
        "VfB Stuttgart": "Stuttgart",
    },
    # 1. FC Cologne, SC Paderborn 07, SV 07 Elversberg, Schalke 04 not in football-data's
    # current D1 (top-flight) season — no alias, re-check if promoted.
    "SP1": {
        "CA Osasuna": "Osasuna",
        "Deportivo Alaves": "Alaves",
        "Espanyol Barcelona": "Espanol",
        "FC Barcelona": "Barcelona",
        "Getafe CF": "Getafe",
        "RC Celta de Vigo": "Celta",
        "Rayo Vallecano": "Vallecano",
        "Real Betis Seville": "Betis",
        "Real Sociedad San Sebastian": "Sociedad",
        "Sevilla FC": "Sevilla",
        "Valencia CF": "Valencia",
        "Villarreal CF": "Villarreal",
    },
    # Athletic Bilbao, Atletico Madrid, Elche CF, Levante UD, Malaga CF, RC Deportivo de
    # A Coruna, Racing Santander not matched with confidence against this season's
    # football-data SP1 list — re-diff before relying on these.
    "I1": {
        "AC Milan": "Milan",
        "AC Monza": "Monza",
        "ACF Fiorentina": "Fiorentina",
        "AS Roma": "Roma",
        "Atalanta BC": "Atalanta",
        "Bologna FC": "Bologna",
        "Cagliari Calcio": "Cagliari",
        "Como 1907": "Como",
        "Genoa CFC": "Genoa",
        "Inter Milano": "Inter",
        "Juventus Turin": "Juventus",
        "Lazio Rome": "Lazio",
        "Parma Calcio": "Parma",
        "SSC Napoli": "Napoli",
        "Sassuolo Calcio": "Sassuolo",
        "Torino FC": "Torino",
        "US Lecce": "Lecce",
        "Udinese Calcio": "Udinese",
        "Venezia FC": "Venezia",
    },
    # Frosinone Calcio not in football-data's current I1 season.
    "F1": {
        "AJ Auxerre": "Auxerre",
        "AS Monaco": "Monaco",
        "Angers SCO": "Angers",
        "FC Lorient": "Lorient",
        "Le Havre AC": "Le Havre",
        "Lille OSC": "Lille",
        "OGC Nice": "Nice",
        "Olympique Lyon": "Lyon",
        "Olympique Marseille": "Marseille",
        "Paris Saint-Germain": "Paris SG",
        "Racing Club De Lens": "Lens",
        "Stade Brest 29": "Brest",
        "Stade Rennais FC": "Rennes",
        "Strasbourg Alsace": "Strasbourg",
        "Toulouse FC": "Toulouse",
    },
    # ESTAC Troyes, Le Mans FC, Paris FC not in football-data's current F1 season.
    "N1": {
        "ADO Den Haag": "Den Haag",
        "Ajax Amsterdam": "Ajax",
        "Excelsior Rotterdam": "Excelsior",
        "FC Groningen": "Groningen",
        "FC Twente Enschede": "Twente",
        "FC Utrecht": "Utrecht",
        "Feyenoord Rotterdam": "Feyenoord",
        "Fortuna Sittard": "For Sittard",
        "NEC Nijmegen": "Nijmegen",
        "PEC Zwolle": "Zwolle",
        "SC Cambuur": "Cambuur",
        "SC Heerenveen": "Heerenveen",
        "SC Telstar": "Telstar",
        "Willem II Tilburg": "Willem II",
    },
    "P1": {
        "Estoril Praia": "Estoril",
        "FC Arouca": "Arouca",
        "FC Famalicao": "Famalicao",
        "FC Porto": "Porto",
        "Gil Vicente Barcelos": "Gil Vicente",
        "Moreirense FC": "Moreirense",
        "Nacional da Madeira": "Nacional",
        "Rio Ave FC": "Rio Ave",
        "SC Braga": "Sp Braga",
        "SL Benfica": "Benfica",
        "Sporting CP": "Sp Lisbon",
        "Vitoria SC Guimaraes": "Guimaraes",
    },
    # Academico de Viseu FC, FC Alverca SAD, Maritimo Madeira not matched with
    # confidence — re-diff before relying on these.
    "G1": {
        "AE Kifisia FC": "Kifisia",
        "AEK Athens": "AEK",
        "APO Levadiakos FC": "Levadeiakos",
        "Aris Thessaloniki": "Aris",
        "Atromitos Athinon": "Atromitos",
        "Olympiacos Piraeus": "Olympiakos",
        "PAOK Thessaloniki": "PAOK",
        "Panaitolikos Agrinio": "Panetolikos",
        "Panathinaikos Athens": "Panathinaikos",
        "Volos NPS": "Volos NFC",
    },
    # PAE PS Kalamata, POT Iraklis not in football-data's current G1 season.
    "SC0": {
        "Aberdeen FC": "Aberdeen",
        "Celtic Glasgow": "Celtic",
        "Dundee FC": "Dundee",
        "Glasgow Rangers": "Rangers",
        "Heart of Midlothian FC": "Hearts",
        "Hibernian FC": "Hibernian",
        "Kilmarnock FC": "Kilmarnock",
        "Motherwell FC": "Motherwell",
        "St Mirren FC": "St Mirren",
        "St. Johnstone FC": "St Johnstone",
    },
    # Falkirk FC not in football-data's current SC0 (top-flight) season.
    "B1": {
        "KRC Genk": "Genk",
        "KV Kortrijk": "Kortrijk",
        "KVC Westerlo": "Westerlo",
        "RSC Anderlecht": "Anderlecht",
        "Royal Antwerp FC": "Antwerp",
        "Royal Charleroi SC": "Charleroi",
        "St. Truidense VV": "St Truiden",
        "Standard Liege": "Standard",
        "Yellow-Red KV Mechelen": "Mechelen",
    },
    # Lommel SK, RAAL La Louviere, SK Beveren, SV Zulte Waregem, Union Saint-Gilloise not
    # matched with confidence against this season's football-data B1 list.
    "T1": {
        "Caykur Rizespor": "Rizespor",
        "Fenerbahce Istanbul": "Fenerbahce",
        "Galatasaray Istanbul": "Galatasaray",
        "Gaziantep FK": "Gaziantep",
        "Goztepe Izmir": "Goztep",
        "Kasimpasa Istanbul": "Kasimpasa",
    },
    # Amed Sportif Faaliyetler, Corum FK, Erzurumspor FK, Genclerbirligi SK, Istanbul
    # Basaksehir, Kocaelispor not matched with confidence against this season's
    # football-data T1 list.
}
