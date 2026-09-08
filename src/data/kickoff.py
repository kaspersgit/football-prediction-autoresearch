"""Earliest upcoming kickoff time, used to align live Predict runs with kickoff."""

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

RAW_DIR = Path("data/raw")

# football-data.co.uk lists fixture kickoff times in UK local time (GMT/BST).
_FIXTURES_TZ = ZoneInfo("Europe/London")


def earliest_kickoff_utc(path: Path | None = None, now: datetime | None = None) -> datetime | None:
    """Return the UTC kickoff time of the soonest fixture in fixtures.csv still ahead of `now`.

    Returns None if the file is missing, has no usable Date/Time columns, or every
    listed fixture's kickoff has already passed.
    """
    path = path or RAW_DIR / "fixtures.csv"
    now = now or datetime.now(timezone.utc)
    if not path.exists():
        return None

    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if "Date" not in df.columns or "Time" not in df.columns:
        return None

    dates = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    raw_time = df["Time"].astype(str).str.strip()
    padded_time = raw_time.where(raw_time.str.count(":") != 1, raw_time + ":00")
    deltas = pd.to_timedelta(padded_time, errors="coerce")

    valid = dates.notna() & deltas.notna()
    if not valid.any():
        return None

    kickoffs_local = (dates[valid] + deltas[valid]).dt.tz_localize(
        _FIXTURES_TZ, nonexistent="shift_forward", ambiguous="NaT"
    )
    kickoffs_utc = kickoffs_local.dt.tz_convert(timezone.utc).dropna()
    upcoming = kickoffs_utc[kickoffs_utc > pd.Timestamp(now)]
    if upcoming.empty:
        return None
    return upcoming.min().to_pydatetime()


if __name__ == "__main__":
    kickoff = earliest_kickoff_utc()
    print(kickoff.isoformat() if kickoff is not None else "NONE")
