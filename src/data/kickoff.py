"""Upcoming kickoff waves, used to align live Predict runs with kickoff."""

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

RAW_DIR = Path("data/raw")

# football-data.co.uk lists fixture kickoff times in UK local time (GMT/BST).
_FIXTURES_TZ = ZoneInfo("Europe/London")


def upcoming_kickoff_groups_utc(path: Path | None = None, now: datetime | None = None) -> list[datetime]:
    """Return the distinct upcoming kickoff instants in fixtures.csv, ascending.

    Matches kicking off at the same instant (a "wave") collapse to one entry.
    Returns an empty list if the file is missing, has no usable Date/Time
    columns, or every listed fixture's kickoff has already passed.
    """
    path = path or RAW_DIR / "fixtures.csv"
    now = now or datetime.now(timezone.utc)
    if not path.exists():
        return []

    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if "Date" not in df.columns or "Time" not in df.columns:
        return []

    dates = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    raw_time = df["Time"].astype(str).str.strip()
    padded_time = raw_time.where(raw_time.str.count(":") != 1, raw_time + ":00")
    deltas = pd.to_timedelta(padded_time, errors="coerce")

    valid = dates.notna() & deltas.notna()
    if not valid.any():
        return []

    kickoffs_local = (dates[valid] + deltas[valid]).dt.tz_localize(
        _FIXTURES_TZ, nonexistent="shift_forward", ambiguous="NaT"
    )
    kickoffs_utc = kickoffs_local.dt.tz_convert(timezone.utc).dropna()
    upcoming = kickoffs_utc[kickoffs_utc > pd.Timestamp(now)]
    return sorted({ts.to_pydatetime() for ts in upcoming})


def earliest_kickoff_utc(path: Path | None = None, now: datetime | None = None) -> datetime | None:
    """Return the soonest upcoming kickoff, or None if there isn't one."""
    groups = upcoming_kickoff_groups_utc(path, now)
    return groups[0] if groups else None


if __name__ == "__main__":
    for kickoff in upcoming_kickoff_groups_utc():
        print(kickoff.isoformat())
