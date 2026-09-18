import pandas as pd

from src.data.loader import REQUIRED_COLS, load_all_data


def _isolate(monkeypatch, tmp_path):
    """Point both season-file locations at empty, isolated directories.

    Without this, a real data/historical/ (once populated) would leak extra
    rows into these tests' row-count assertions.
    """
    monkeypatch.setattr("src.data.loader.RAW_DIR", tmp_path)
    monkeypatch.setattr("src.data.loader.HISTORICAL_DIR", tmp_path / "historical")


def test_load_all_data_returns_dataframe(tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    csv_content = (
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A\n"
        "12/08/2023,Arsenal,Forest,2,1,H,1.8,3.5,5.0\n"
    )
    (tmp_path / "E0_2324.csv").write_text(csv_content)
    df = load_all_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    for col in REQUIRED_COLS:
        assert col in df.columns

def test_load_adds_league_and_season_columns(tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    csv_content = (
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A\n"
        "12/08/2023,Arsenal,Forest,2,1,H,1.8,3.5,5.0\n"
    )
    (tmp_path / "E0_2324.csv").write_text(csv_content)
    df = load_all_data()
    assert "league" in df.columns
    assert "season" in df.columns
    assert df["league"].iloc[0] == "E0"
    assert df["season"].iloc[0] == "2324"

def test_rows_with_missing_result_dropped(tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    csv_content = (
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A\n"
        "12/08/2023,Arsenal,Forest,2,1,H,1.8,3.5,5.0\n"
        "13/08/2023,City,United,,,,,,\n"
    )
    (tmp_path / "E0_2324.csv").write_text(csv_content)
    df = load_all_data()
    assert len(df) == 1

def test_load_all_data_passes_through_pinnacle_opening_and_closing_odds(tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    csv_content = (
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,PSH,PSD,PSA,PSCH,PSCD,PSCA\n"
        "12/08/2023,Arsenal,Forest,2,1,H,1.8,3.5,5.0,1.75,3.6,5.2,1.7,3.65,5.4\n"
    )
    (tmp_path / "E0_2324.csv").write_text(csv_content)
    df = load_all_data()
    assert df.loc[0, "PSH"] == 1.75
    assert df.loc[0, "PSD"] == 3.6
    assert df.loc[0, "PSA"] == 5.2
    assert df.loc[0, "PSCH"] == 1.7
    assert df.loc[0, "PSCD"] == 3.65
    assert df.loc[0, "PSCA"] == 5.4


def test_load_all_data_reads_committed_historical_archive(tmp_path, monkeypatch):
    """data/historical/<league>/<season>.csv derives league/season from the path,
    not the filename, since archived files are just named <season>.csv."""
    _isolate(monkeypatch, tmp_path)
    historical_dir = tmp_path / "historical" / "E0"
    historical_dir.mkdir(parents=True)
    csv_content = (
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A\n"
        "12/08/2013,Arsenal,Forest,2,1,H,1.8,3.5,5.0\n"
    )
    (historical_dir / "1314.csv").write_text(csv_content)

    df = load_all_data()

    assert len(df) == 1
    assert df["league"].iloc[0] == "E0"
    assert df["season"].iloc[0] == "1314"


def test_load_all_data_combines_historical_archive_and_raw_cache(tmp_path, monkeypatch):
    _isolate(monkeypatch, tmp_path)
    historical_dir = tmp_path / "historical" / "E0"
    historical_dir.mkdir(parents=True)
    (historical_dir / "1314.csv").write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A\n"
        "12/08/2013,Arsenal,Forest,2,1,H,1.8,3.5,5.0\n"
    )
    (tmp_path / "E0_2627.csv").write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A\n"
        "12/08/2026,Arsenal,Chelsea,1,1,D,2.0,3.3,3.6\n"
    )

    df = load_all_data()

    assert sorted(df["season"]) == ["1314", "2627"]
