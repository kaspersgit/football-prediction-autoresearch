import pytest
import requests

from src.data import download


class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


_VALID_E0_CSV = b"Div,Date,HomeTeam,AwayTeam,FTR\nE0,08/08/2026,Arsenal,Chelsea,H\n"
_WRONG_LEAGUE_CSV = b"Div,Date,HomeTeam,AwayTeam,FTR\nP1,08/08/2026,Benfica,Porto,H\n"
_HTML_300_PAGE = b"<!DOCTYPE HTML PUBLIC><html><head><title>300 Multiple Choices</title>"


def test_download_season_accepts_csv_whose_div_matches_requested_league(monkeypatch, tmp_path):
    monkeypatch.setattr(download, "RAW_DIR", tmp_path)
    monkeypatch.setattr(download.requests, "get", lambda *a, **k: _FakeResponse(_VALID_E0_CSV))

    dest = download.download_season("E0", "2627", force=True)

    assert dest.read_bytes() == _VALID_E0_CSV


def test_download_season_rejects_response_from_a_different_league(monkeypatch, tmp_path):
    """A redirect to another league's file (observed: SP1 2627 -> P1's data) must
    never be silently written to disk under the requested league's filename."""
    monkeypatch.setattr(download, "RAW_DIR", tmp_path)
    monkeypatch.setattr(download.requests, "get", lambda *a, **k: _FakeResponse(_WRONG_LEAGUE_CSV))

    with pytest.raises(requests.HTTPError):
        download.download_season("SP1", "2627", force=True)

    assert not (tmp_path / "SP1_2627.csv").exists()


def test_download_season_rejects_html_error_page_served_as_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(download, "RAW_DIR", tmp_path)
    monkeypatch.setattr(download.requests, "get", lambda *a, **k: _FakeResponse(_HTML_300_PAGE))

    with pytest.raises(requests.HTTPError):
        download.download_season("D1", "2627", force=True)

    assert not (tmp_path / "D1_2627.csv").exists()


def test_download_season_uses_cached_file_without_revalidation_when_not_forced(monkeypatch, tmp_path):
    monkeypatch.setattr(download, "RAW_DIR", tmp_path)
    cached = tmp_path / "E0_2627.csv"
    cached.write_bytes(_VALID_E0_CSV)

    def _fail_get(*a, **k):
        raise AssertionError("must not hit the network when the cached file already exists")

    monkeypatch.setattr(download.requests, "get", _fail_get)

    dest = download.download_season("E0", "2627", force=False)

    assert dest == cached
