from datetime import datetime, timezone

from src.data.kickoff import earliest_kickoff_utc, upcoming_kickoff_groups_utc


def _write_fixtures(tmp_path, rows: str):
    path = tmp_path / "fixtures.csv"
    path.write_text("Div,Date,Time,HomeTeam,AwayTeam\n" + rows)
    return path


def test_groups_are_deduped_and_sorted_ascending_in_utc(tmp_path):
    # BST is in effect in August, so 14:30 and 12:30 UK local time are 13:30/11:30 UTC.
    path = _write_fixtures(
        tmp_path,
        "E0,15/08/2026,14:30,Arsenal,Chelsea\n"
        "E0,15/08/2026,12:30,Everton,Fulham\n"
        "E0,15/08/2026,14:30,Liverpool,Wolves\n"  # same wave as Arsenal v Chelsea
        "E0,16/08/2026,14:00,Leeds,Burnley\n",
    )
    now = datetime(2026, 8, 15, 0, 0, tzinfo=timezone.utc)

    groups = upcoming_kickoff_groups_utc(path, now)

    assert groups == [
        datetime(2026, 8, 15, 11, 30, tzinfo=timezone.utc),
        datetime(2026, 8, 15, 13, 30, tzinfo=timezone.utc),
        datetime(2026, 8, 16, 13, 0, tzinfo=timezone.utc),
    ]


def test_ignores_fixtures_that_have_already_kicked_off(tmp_path):
    path = _write_fixtures(tmp_path, "E0,15/08/2026,12:30,Everton,Fulham\n")
    now = datetime(2026, 8, 15, 12, 0, tzinfo=timezone.utc)  # already past 12:30 BST (11:30 UTC)

    assert upcoming_kickoff_groups_utc(path, now) == []


def test_returns_empty_when_time_column_missing(tmp_path):
    path = tmp_path / "fixtures.csv"
    path.write_text("Div,Date,HomeTeam,AwayTeam\nE0,15/08/2026,Arsenal,Chelsea\n")

    assert upcoming_kickoff_groups_utc(path, datetime.now(timezone.utc)) == []


def test_returns_empty_when_file_missing(tmp_path):
    assert upcoming_kickoff_groups_utc(tmp_path / "fixtures.csv", datetime.now(timezone.utc)) == []


def test_earliest_kickoff_utc_returns_first_group(tmp_path):
    path = _write_fixtures(
        tmp_path,
        "E0,15/08/2026,14:30,Arsenal,Chelsea\n"
        "E0,15/08/2026,12:30,Everton,Fulham\n",
    )
    now = datetime(2026, 8, 15, 0, 0, tzinfo=timezone.utc)

    assert earliest_kickoff_utc(path, now) == datetime(2026, 8, 15, 11, 30, tzinfo=timezone.utc)


def test_earliest_kickoff_utc_returns_none_when_no_groups(tmp_path):
    assert earliest_kickoff_utc(tmp_path / "fixtures.csv", datetime.now(timezone.utc)) is None
