from src.config import SUPPORTED_LEAGUES
from src.data.team_aliases import ODDS_API_TEAM_ALIASES


def test_every_supported_league_has_an_alias_table_entry():
    assert set(ODDS_API_TEAM_ALIASES) == set(SUPPORTED_LEAGUES)


def test_alias_values_are_string_to_string_maps():
    for league, aliases in ODDS_API_TEAM_ALIASES.items():
        assert isinstance(aliases, dict)
        for odds_api_name, fd_name in aliases.items():
            assert isinstance(odds_api_name, str) and odds_api_name
            assert isinstance(fd_name, str) and fd_name


def test_known_eredivisie_aliases_are_present():
    assert ODDS_API_TEAM_ALIASES["N1"]["FC Utrecht"] == "Utrecht"
    assert ODDS_API_TEAM_ALIASES["N1"]["FC Twente Enschede"] == "Twente"
    assert ODDS_API_TEAM_ALIASES["N1"]["FC Zwolle"] == "Zwolle"
    assert ODDS_API_TEAM_ALIASES["N1"]["SC Telstar"] == "Telstar"
    assert ODDS_API_TEAM_ALIASES["N1"]["NEC Nijmegen"] == "Nijmegen"
    assert ODDS_API_TEAM_ALIASES["N1"]["Fortuna Sittard"] == "For Sittard"
    assert ODDS_API_TEAM_ALIASES["N1"]["SC Cambuur"] == "Cambuur"
    assert ODDS_API_TEAM_ALIASES["N1"]["ADO Den Haag"] == "Den Haag"


def test_known_premier_league_aliases_are_present():
    assert ODDS_API_TEAM_ALIASES["E0"]["Manchester United"] == "Man United"
    assert ODDS_API_TEAM_ALIASES["E0"]["Nottingham Forest"] == "Nott'm Forest"


def test_known_primeira_liga_aliases_are_present():
    assert ODDS_API_TEAM_ALIASES["P1"]["FC Porto"] == "Porto"
    assert ODDS_API_TEAM_ALIASES["P1"]["Sporting Lisbon"] == "Sp Lisbon"
    assert ODDS_API_TEAM_ALIASES["P1"]["CF Estrela"] == "Estrela"


def test_known_super_league_greece_aliases_are_present():
    assert ODDS_API_TEAM_ALIASES["G1"]["PAOK Thessaloniki"] == "PAOK"
    assert ODDS_API_TEAM_ALIASES["G1"]["Levadiakos"] == "Levadeiakos"


def test_known_ligue_one_aliases_are_present():
    assert ODDS_API_TEAM_ALIASES["F1"]["RC Lens"] == "Lens"
    assert ODDS_API_TEAM_ALIASES["F1"]["AS Monaco"] == "Monaco"
    assert ODDS_API_TEAM_ALIASES["F1"]["Paris Saint Germain"] == "Paris SG"
