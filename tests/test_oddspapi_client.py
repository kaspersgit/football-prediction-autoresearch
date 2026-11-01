import pandas as pd
import pytest
import requests

from src.data._oddspapi_client import OddsPapiUnavailable, fetch_oddspapi_bookmaker_odds
from src.data.oddspapi_team_aliases import ODDSPAPI_TEAM_ALIASES

_ODDS_COLUMNS = ["league", "HomeTeam", "AwayTeam", "Date", "PSH", "PSD", "PSA"]
_PARTICIPANTS = {"39": "Manchester City", "48": "Everton FC"}


@pytest.fixture(autouse=True)
def _no_real_pacing_delay(monkeypatch):
    """The pacing cooldown is global (not per-fetch), so without this every test here
    would actually sleep — stub it out and reset the shared timer between tests."""
    monkeypatch.setattr("src.data._oddspapi_client._last_request_at", None)
    monkeypatch.setattr("src.data._oddspapi_client.time.sleep", lambda s: None)


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _reset_participants_cache(monkeypatch):
    monkeypatch.setattr("src.data._oddspapi_client._participants_cache", None)


def _fixture(status_id=0, p1=39, p2=48, home=2.32, draw=3.4, away=3.1, bm="pinnacle"):
    return {
        "tournamentId": 17,
        "participant1Id": p1,
        "participant2Id": p2,
        "statusId": status_id,
        "startTime": "2026-11-02T20:00:00.000Z",
        "bookmakerOdds": {
            bm: {
                "markets": {
                    "101": {
                        "outcomes": {
                            "101": {"players": {"0": {"price": home}}},
                            "102": {"players": {"0": {"price": draw}}},
                            "103": {"players": {"0": {"price": away}}},
                        }
                    }
                }
            }
        },
    }


def test_missing_api_key_raises_unavailable(monkeypatch):
    monkeypatch.delenv("ODDSPAPI_API_KEY", raising=False)

    with pytest.raises(OddsPapiUnavailable):
        fetch_oddspapi_bookmaker_odds({"E0"}, "pinnacle", "PS", "Pinnacle")


def test_request_failure_raises_unavailable(monkeypatch):
    monkeypatch.setenv("ODDSPAPI_API_KEY", "test-key")
    _reset_participants_cache(monkeypatch)

    def fake_get(url, params=None, timeout=None):
        raise requests.exceptions.Timeout("boom")

    monkeypatch.setattr("src.data._oddspapi_client.requests.get", fake_get)

    with pytest.raises(OddsPapiUnavailable):
        fetch_oddspapi_bookmaker_odds({"E0"}, "pinnacle", "PS", "Pinnacle")


def test_error_payload_with_200_status_raises_unavailable(monkeypatch):
    """OddsPapi returns some errors (e.g. rate limiting) as a 200 with an error body."""
    monkeypatch.setenv("ODDSPAPI_API_KEY", "test-key")
    _reset_participants_cache(monkeypatch)
    calls = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        calls["n"] += 1
        if "participants" in url:
            return _FakeResponse(_PARTICIPANTS)
        return _FakeResponse({"error": {"message": "rate limited"}})

    monkeypatch.setattr("src.data._oddspapi_client.requests.get", fake_get)

    with pytest.raises(OddsPapiUnavailable):
        fetch_oddspapi_bookmaker_odds({"E0"}, "pinnacle", "PS", "Pinnacle")


def test_successful_parse_and_alias_resolution(monkeypatch):
    monkeypatch.setenv("ODDSPAPI_API_KEY", "test-key")
    _reset_participants_cache(monkeypatch)

    def fake_get(url, params=None, timeout=None):
        if "participants" in url:
            return _FakeResponse(_PARTICIPANTS)
        return _FakeResponse([_fixture()])

    monkeypatch.setattr("src.data._oddspapi_client.requests.get", fake_get)

    result = fetch_oddspapi_bookmaker_odds({"E0"}, "pinnacle", "PS", "Pinnacle")

    assert list(result.columns) == _ODDS_COLUMNS
    assert len(result) == 1
    row = result.iloc[0]
    assert row["league"] == "E0"
    assert row["HomeTeam"] == "Man City"  # alias resolved
    assert row["AwayTeam"] == "Everton"  # alias resolved
    assert row["Date"] == pd.Timestamp("2026-11-02")
    assert row["PSH"] == pytest.approx(2.32)
    assert row["PSD"] == pytest.approx(3.4)
    assert row["PSA"] == pytest.approx(3.1)


def test_fixture_already_started_is_excluded(monkeypatch):
    monkeypatch.setenv("ODDSPAPI_API_KEY", "test-key")
    _reset_participants_cache(monkeypatch)

    def fake_get(url, params=None, timeout=None):
        if "participants" in url:
            return _FakeResponse(_PARTICIPANTS)
        return _FakeResponse([_fixture(status_id=1)])  # in-play / finished

    monkeypatch.setattr("src.data._oddspapi_client.requests.get", fake_get)

    result = fetch_oddspapi_bookmaker_odds({"E0"}, "pinnacle", "PS", "Pinnacle")

    assert result.empty


def test_fixture_missing_requested_bookmaker_is_excluded(monkeypatch):
    monkeypatch.setenv("ODDSPAPI_API_KEY", "test-key")
    _reset_participants_cache(monkeypatch)

    def fake_get(url, params=None, timeout=None):
        if "participants" in url:
            return _FakeResponse(_PARTICIPANTS)
        return _FakeResponse([_fixture(bm="bet365")])  # no pinnacle odds on this fixture

    monkeypatch.setattr("src.data._oddspapi_client.requests.get", fake_get)

    result = fetch_oddspapi_bookmaker_odds({"E0"}, "pinnacle", "PS", "Pinnacle")

    assert result.empty


def test_batches_tournament_ids_at_three_per_call(monkeypatch):
    monkeypatch.setenv("ODDSPAPI_API_KEY", "test-key")
    _reset_participants_cache(monkeypatch)
    monkeypatch.setattr("src.data._oddspapi_client.time.sleep", lambda s: None)
    captured_params = []

    def fake_get(url, params=None, timeout=None):
        if "participants" in url:
            return _FakeResponse(_PARTICIPANTS)
        captured_params.append(params["tournamentIds"])
        return _FakeResponse([])

    monkeypatch.setattr("src.data._oddspapi_client.requests.get", fake_get)

    # E0, D1, SP1, I1 -> 4 leagues, batch size 3 -> 2 calls, not 4
    fetch_oddspapi_bookmaker_odds({"E0", "D1", "SP1", "I1"}, "pinnacle", "PS", "Pinnacle")

    assert len(captured_params) == 2
    all_ids = {int(i) for batch in captured_params for i in batch.split(",")}
    assert all_ids == {17, 35, 8, 23}


def test_participants_fetched_only_once_across_two_fetches(monkeypatch):
    monkeypatch.setenv("ODDSPAPI_API_KEY", "test-key")
    _reset_participants_cache(monkeypatch)
    calls = {"participants": 0}

    def fake_get(url, params=None, timeout=None):
        if "participants" in url:
            calls["participants"] += 1
            return _FakeResponse(_PARTICIPANTS)
        return _FakeResponse([])

    monkeypatch.setattr("src.data._oddspapi_client.requests.get", fake_get)

    fetch_oddspapi_bookmaker_odds({"E0"}, "pinnacle", "PS", "Pinnacle")
    fetch_oddspapi_bookmaker_odds({"E0"}, "bet365", "B365", "Bet365")

    assert calls["participants"] == 1


def test_unknown_league_codes_are_ignored():
    assert set(ODDSPAPI_TEAM_ALIASES) <= {
        "E0", "D1", "SP1", "I1", "F1", "N1", "P1", "G1", "SC0", "B1", "T1",
    }
