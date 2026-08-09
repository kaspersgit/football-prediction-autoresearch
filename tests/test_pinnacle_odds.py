import pandas as pd
import pytest
import requests

from src.config import SUPPORTED_LEAGUES
from src.data.pinnacle_odds import _LEAGUE_TO_SPORT_KEY, attach_pinnacle_odds, fetch_pinnacle_odds

_ODDS_COLUMNS = ["league", "HomeTeam", "AwayTeam", "PSH", "PSD", "PSA"]


def test_sport_key_map_covers_every_supported_league():
    assert set(_LEAGUE_TO_SPORT_KEY) == set(SUPPORTED_LEAGUES)


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_missing_api_key_returns_empty_dataframe(monkeypatch):
    monkeypatch.delenv("THEODDS_API", raising=False)

    result = fetch_pinnacle_odds({"E0"})

    assert result.empty
    assert list(result.columns) == _ODDS_COLUMNS


def test_per_league_request_failure_is_skipped(monkeypatch):
    monkeypatch.setenv("THEODDS_API", "test-key")

    def fake_get(url, params=None, timeout=None):
        raise requests.exceptions.Timeout("boom")

    monkeypatch.setattr("src.data.pinnacle_odds.requests.get", fake_get)

    result = fetch_pinnacle_odds({"E0"})

    assert result.empty


def test_successful_parse_and_alias_resolution(monkeypatch):
    monkeypatch.setenv("THEODDS_API", "test-key")
    payload = [
        {
            "home_team": "FC Utrecht",
            "away_team": "FC Twente Enschede",
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "FC Utrecht", "price": 2.1},
                                {"name": "FC Twente Enschede", "price": 3.4},
                                {"name": "Draw", "price": 3.3},
                            ],
                        }
                    ],
                }
            ],
        }
    ]

    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(payload)

    monkeypatch.setattr("src.data.pinnacle_odds.requests.get", fake_get)

    result = fetch_pinnacle_odds({"N1"})

    assert len(result) == 1
    row = result.iloc[0]
    assert row["league"] == "N1"
    assert row["HomeTeam"] == "Utrecht"
    assert row["AwayTeam"] == "Twente"
    assert row["PSH"] == pytest.approx(2.1)
    assert row["PSD"] == pytest.approx(3.3)
    assert row["PSA"] == pytest.approx(3.4)


def test_event_without_pinnacle_bookmaker_is_dropped(monkeypatch):
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

    monkeypatch.setattr("src.data.pinnacle_odds.requests.get", fake_get)

    result = fetch_pinnacle_odds({"N1"})

    assert result.empty


def test_attach_pinnacle_odds_overwrites_nan_placeholders(monkeypatch):
    fixtures_df = pd.DataFrame({
        "Date": pd.to_datetime(["2026-08-10"]),
        "HomeTeam": ["Utrecht"],
        "AwayTeam": ["Twente"],
        "league": ["N1"],
        "B365H": [2.0], "B365D": [3.5], "B365A": [4.0],
        "PSH": [float("nan")], "PSD": [float("nan")], "PSA": [float("nan")],
    })

    def fake_fetch(leagues):
        return pd.DataFrame([{
            "league": "N1", "HomeTeam": "Utrecht", "AwayTeam": "Twente",
            "PSH": 2.1, "PSD": 3.3, "PSA": 3.4,
        }])

    monkeypatch.setattr("src.data.pinnacle_odds.fetch_pinnacle_odds", fake_fetch)

    result = attach_pinnacle_odds(fixtures_df)

    assert result.loc[0, "PSH"] == pytest.approx(2.1)
    assert result.loc[0, "PSD"] == pytest.approx(3.3)
    assert result.loc[0, "PSA"] == pytest.approx(3.4)
    assert result.loc[0, "B365H"] == pytest.approx(2.0)  # untouched


def test_attach_pinnacle_odds_leaves_fixtures_unchanged_when_fetch_is_empty(monkeypatch):
    fixtures_df = pd.DataFrame({
        "HomeTeam": ["Utrecht"], "AwayTeam": ["Twente"], "league": ["N1"],
        "PSH": [float("nan")], "PSD": [float("nan")], "PSA": [float("nan")],
    })

    def fake_fetch(leagues):
        return pd.DataFrame(columns=_ODDS_COLUMNS)

    monkeypatch.setattr("src.data.pinnacle_odds.fetch_pinnacle_odds", fake_fetch)

    result = attach_pinnacle_odds(fixtures_df)

    assert result["PSH"].isna().all()
