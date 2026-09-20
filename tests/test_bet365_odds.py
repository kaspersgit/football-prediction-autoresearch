import pandas as pd
import pytest
import requests

from src.data.bet365_odds import attach_bet365_odds, fetch_bet365_odds

_ODDS_COLUMNS = ["league", "HomeTeam", "AwayTeam", "Date", "B365H", "B365D", "B365A"]


class _FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_missing_api_key_returns_empty_dataframe(monkeypatch):
    monkeypatch.delenv("THEODDS_API", raising=False)

    result = fetch_bet365_odds({"E0"})

    assert result.empty
    assert list(result.columns) == _ODDS_COLUMNS


def test_per_league_request_failure_is_skipped(monkeypatch):
    monkeypatch.setenv("THEODDS_API", "test-key")

    def fake_get(url, params=None, timeout=None):
        raise requests.exceptions.Timeout("boom")

    monkeypatch.setattr("src.data._live_odds.requests.get", fake_get)

    result = fetch_bet365_odds({"E0"})

    assert result.empty


def test_successful_parse_uses_uk_region_and_bet365_key(monkeypatch):
    monkeypatch.setenv("THEODDS_API", "test-key")
    payload = [
        {
            "home_team": "FC Utrecht",
            "away_team": "FC Twente Enschede",
            "commence_time": "2026-08-16T14:00:00Z",
            "bookmakers": [
                {
                    "key": "bet365",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "FC Utrecht", "price": 2.0},
                                {"name": "FC Twente Enschede", "price": 3.5},
                                {"name": "Draw", "price": 3.4},
                            ],
                        }
                    ],
                }
            ],
        }
    ]
    captured_params = {}

    def fake_get(url, params=None, timeout=None):
        captured_params.update(params)
        return _FakeResponse(payload)

    monkeypatch.setattr("src.data._live_odds.requests.get", fake_get)

    result = fetch_bet365_odds({"N1"})

    assert captured_params["bookmakers"] == "bet365"
    assert captured_params["regions"] == "uk"
    assert len(result) == 1
    row = result.iloc[0]
    assert row["league"] == "N1"
    assert row["HomeTeam"] == "Utrecht"
    assert row["AwayTeam"] == "Twente"
    assert row["Date"] == pd.Timestamp("2026-08-16")
    assert row["B365H"] == pytest.approx(2.0)
    assert row["B365D"] == pytest.approx(3.4)
    assert row["B365A"] == pytest.approx(3.5)


def test_event_without_bet365_bookmaker_is_dropped(monkeypatch, capsys):
    monkeypatch.setenv("THEODDS_API", "test-key")
    payload = [
        {
            "home_team": "Ajax",
            "away_team": "Feyenoord",
            "bookmakers": [{"key": "unibet", "markets": []}],
        }
    ]

    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(payload)

    monkeypatch.setattr("src.data._live_odds.requests.get", fake_get)

    result = fetch_bet365_odds({"N1"})

    assert result.empty
    out = capsys.readouterr().out
    assert "not yet priced by Bet365" in out


def test_attach_bet365_odds_overwrites_stale_fixtures_csv_values(monkeypatch):
    """B365 columns from fixtures.csv are already populated (unlike PSH/PSD/PSA's NaN
    placeholders) but only as fresh as football-data.co.uk's last publish — live odds
    must still take priority over them, not just fill gaps."""
    fixtures_df = pd.DataFrame({
        "Date": pd.to_datetime(["2026-08-10"]),
        "HomeTeam": ["Utrecht"],
        "AwayTeam": ["Twente"],
        "league": ["N1"],
        "B365H": [2.2], "B365D": [3.6], "B365A": [3.9],
    })

    def fake_fetch(leagues):
        return pd.DataFrame([{
            "league": "N1", "HomeTeam": "Utrecht", "AwayTeam": "Twente",
            "Date": pd.Timestamp("2026-08-10"),
            "B365H": 2.0, "B365D": 3.4, "B365A": 4.1,
        }])

    monkeypatch.setattr("src.data.bet365_odds.fetch_bet365_odds", fake_fetch)

    result = attach_bet365_odds(fixtures_df)

    assert result.loc[0, "B365H"] == pytest.approx(2.0)
    assert result.loc[0, "B365D"] == pytest.approx(3.4)
    assert result.loc[0, "B365A"] == pytest.approx(4.1)


def test_attach_bet365_odds_rejects_a_different_matchweek(monkeypatch):
    fixtures_df = pd.DataFrame({
        "Date": pd.to_datetime(["2026-08-10"]),
        "HomeTeam": ["Utrecht"],
        "AwayTeam": ["Twente"],
        "league": ["N1"],
        "B365H": [2.2], "B365D": [3.6], "B365A": [3.9],
    })

    def fake_fetch(leagues):
        return pd.DataFrame([{
            "league": "N1", "HomeTeam": "Utrecht", "AwayTeam": "Twente",
            "Date": pd.Timestamp("2026-08-24"),  # two weeks later — a different round
            "B365H": 2.0, "B365D": 3.4, "B365A": 4.1,
        }])

    monkeypatch.setattr("src.data.bet365_odds.fetch_bet365_odds", fake_fetch)

    result = attach_bet365_odds(fixtures_df)

    # Falls back to fixtures.csv's own value rather than accepting a different round's.
    assert result.loc[0, "B365H"] == pytest.approx(2.2)
    assert result.loc[0, "B365D"] == pytest.approx(3.6)
    assert result.loc[0, "B365A"] == pytest.approx(3.9)


def test_attach_bet365_odds_leaves_fixtures_unchanged_when_fetch_is_empty(monkeypatch):
    fixtures_df = pd.DataFrame({
        "HomeTeam": ["Utrecht"], "AwayTeam": ["Twente"], "league": ["N1"],
        "B365H": [2.2], "B365D": [3.6], "B365A": [3.9],
    })

    def fake_fetch(leagues):
        return pd.DataFrame(columns=_ODDS_COLUMNS)

    monkeypatch.setattr("src.data.bet365_odds.fetch_bet365_odds", fake_fetch)

    result = attach_bet365_odds(fixtures_df)

    assert result.loc[0, "B365H"] == pytest.approx(2.2)
