# Live Pinnacle Odds via The Odds API — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fetch live pre-match Pinnacle odds for the four production leagues from The Odds API, attach them to live fixtures, and land (but keep default-off) a restored Pinnacle-confirmation betting filter so it can be evaluated in a later autoresearch iteration.

**Architecture:** A new `src/data/pinnacle_odds.py` module fetches and normalizes odds via `requests`, using a hand-maintained alias table (`src/data/team_aliases.py`) to reconcile Odds API team names with football-data.co.uk names. `main.py:_run_predict()` attaches these odds to fixtures right after `load_fixtures()`. The Pinnacle-confirmation filter is restored as an opt-in, default-off parameter in both `compute_value_betting_results` (historical `PSCH/PSCD/PSCA`, for backtesting) and `_build_prediction_rows` (live `PSH/PSD/PSA`), so no betting behavior changes until a future autoresearch iteration explicitly turns it on after backtest re-verification.

**Tech Stack:** Python 3.10, pandas, `requests` (already a dependency), pytest with `monkeypatch` (this repo's established mocking idiom — no `responses`/`requests-mock` library is used anywhere in the codebase).

## Global Constraints

- Only production leagues are *fetched* live today: `E0`, `N1`, `P1`, `G1` (`src.config.PRODUCTION_LEAGUES`). But the sport-key map and the team-alias table cover all `src.config.SUPPORTED_LEAGUES` (11 leagues), so widening `PRODUCTION_LEAGUES` later needs no lookup-table changes — only `attach_pinnacle_odds`'s call to `fetch_pinnacle_odds` would need to pass a bigger set.
- `fetch_pinnacle_odds` must never raise — any failure (missing key, request error, malformed response) degrades to fewer/zero rows, never breaks the Predict workflow. Mirrors `main.py`'s `_save_empty_predictions_report` "always produce an artifact" philosophy.
- Team-name matching is table-driven only. A name absent from the alias table and not already identical to the football-data.co.uk name is never guessed — it silently fails to merge (stays `NaN`), it is never fuzzy-matched.
- The restored Pinnacle-confirmation filter is opt-in and defaults to off in both call sites. This plan does **not** flip it on in production — that is a separate autoresearch iteration per the design spec's "Evaluation plan" section.
- Margin constant starts at `0.015` (last-confirmed value from `EXP-20260513-S062`), defined once in `src/config.py`.
- Style: match existing conventions — `requests.get(..., timeout=30)` + `raise_for_status()` (see `src/data/download.py`), `UPPER_SNAKE_CASE` constants in `src/config.py`, `pytest` + `monkeypatch` for mocking (see `tests/test_loader.py`), ruff line-length 100.

---

## File Structure

- **Create** `src/data/team_aliases.py` — `ODDS_API_TEAM_ALIASES: dict[str, dict[str, str]]`, one entry per *supported* league (all 11), not just production, so the table doesn't need restructuring if the production allowlist changes.
- **Create** `src/data/pinnacle_odds.py` — `fetch_pinnacle_odds(leagues) -> pd.DataFrame` and `attach_pinnacle_odds(fixtures_df) -> pd.DataFrame`.
- **Modify** `main.py` — call `attach_pinnacle_odds` in `_run_predict()`; add opt-in `pinnacle_confirmation_margin` parameter to `_build_prediction_rows`.
- **Modify** `src/config.py` — add `DEFAULT_PINNACLE_CONFIRMATION_MARGIN`.
- **Modify** `src/evaluation/metrics.py` — add opt-in `pinnacle_confirmation_margin` parameter to `compute_value_betting_results`.
- **Create** `tests/test_team_aliases.py`, `tests/test_pinnacle_odds.py`.
- **Modify** `tests/test_metrics.py`, `tests/test_shadow_cli.py` — add cases for the new opt-in parameters.

---

### Task 1: Team alias table

**Files:**
- Create: `src/data/team_aliases.py`
- Test: `tests/test_team_aliases.py`

**Interfaces:**
- Produces: `ODDS_API_TEAM_ALIASES: dict[str, dict[str, str]]`, keyed by **every** `src.config.SUPPORTED_LEAGUES` code (`E0`, `D1`, `SP1`, `I1`, `F1`, `N1`, `P1`, `G1`, `SC0`, `B1`, `T1` — 11 leagues), each value mapping an Odds API team name to the matching football-data.co.uk team name. Leagues with no known mismatches yet have an empty `{}` (see the note in Step 3).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_team_aliases.py
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


def test_known_premier_league_aliases_are_present():
    assert ODDS_API_TEAM_ALIASES["E0"]["Manchester United"] == "Man United"
    assert ODDS_API_TEAM_ALIASES["E0"]["Nottingham Forest"] == "Nott'm Forest"


def test_known_primeira_liga_aliases_are_present():
    assert ODDS_API_TEAM_ALIASES["P1"]["FC Porto"] == "Porto"
    assert ODDS_API_TEAM_ALIASES["P1"]["Sporting Lisbon"] == "Sp Lisbon"


def test_known_super_league_greece_aliases_are_present():
    assert ODDS_API_TEAM_ALIASES["G1"]["PAOK Thessaloniki"] == "PAOK"
    assert ODDS_API_TEAM_ALIASES["G1"]["Levadiakos"] == "Levadeiakos"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_team_aliases.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.team_aliases'`

- [ ] **Step 2b: One-time lookup — verify production-league team aliases against the live Odds API**

Before writing the table, fetch each production league's current live event list and diff it against football-data.co.uk's team names for that league:

```bash
set -a; source .env; set +a
for pair in "E0:soccer_epl" "N1:soccer_netherlands_eredivisie" \
            "P1:soccer_portugal_primeira_liga" "G1:soccer_greece_super_league"; do
  code="${pair%%:*}"; key="${pair##*:}"
  echo "=== $code ($key) ==="
  curl -s "https://api.the-odds-api.com/v4/sports/${key}/odds/?apiKey=${THEODDS_API}&regions=eu&markets=h2h&bookmakers=pinnacle&oddsFormat=decimal" \
    | python3 -c "
import json, sys
data = json.load(sys.stdin)
for ev in data:
    print(ev.get('home_team'), '|', ev.get('away_team'))
"
done
```

Compare each name against the corresponding league's most recent `data/raw/{code}_*.csv` team list (`sorted(set(df['HomeTeam']) | set(df['AwayTeam']))`). This was run live on 2026-08-09; results are baked into the table below. A handful of observed Odds API names had no football-data counterpart at all (English League Cup fixtures pulling in Championship sides like `Coventry City`/`Hull City`/`Ipswich Town`; newly promoted/lower-division sides like `Académico de Viseu`, `Kalamata FC`, `Iraklis FC` not yet in this season's CSV) — those are deliberately left **unmapped**, per the "drop, don't guess" rule; they'll simply fail to merge rather than risk a wrong price.

- [ ] **Step 3: Write the alias table**

```python
# src/data/team_aliases.py
"""Team-name aliases: The Odds API's team names → football-data.co.uk's team names.

Covers every league in ``src.config.SUPPORTED_LEAGUES``, not just the current
production allowlist — the production leagues may change, and this table
should not need restructuring when they do.

Built by diffing each league's Odds API team list against its football-data.co.uk
team list. Production leagues (E0/N1/P1/G1) were diffed against a live Odds API
call on 2026-08-09; the rest await a live call once they're actually fetched (see
Task 2's ``fetch_pinnacle_odds`` unmatched-team log lines). Teams whose names
already match exactly need no entry. Ambiguous or diacritic-heavy names
(Portuguese, Greek, Turkish clubs) are resolved by hand here, never
fuzzy-matched — a wrong match would silently misprice a bet.
"""

ODDS_API_TEAM_ALIASES: dict[str, dict[str, str]] = {
    "E0": {
        "Manchester United": "Man United",
        "Manchester City": "Man City",
        "Newcastle United": "Newcastle",
        "Tottenham Hotspur": "Tottenham",
        "Brighton and Hove Albion": "Brighton",
        "Leeds United": "Leeds",
        "Nottingham Forest": "Nott'm Forest",
    },
    "D1": {},
    "SP1": {},
    "I1": {},
    "F1": {},
    "N1": {
        "FC Utrecht": "Utrecht",
        "FC Twente Enschede": "Twente",
        "FC Zwolle": "Zwolle",
    },
    "P1": {
        "FC Porto": "Porto",
        "Moreirense FC": "Moreirense",
        "Braga": "Sp Braga",
        "Rio Ave FC": "Rio Ave",
        "Sporting Lisbon": "Sp Lisbon",
        "Vitória SC": "Guimaraes",
    },
    "G1": {
        "AEK Athens": "AEK",
        "Aris Thessaloniki": "Aris",
        "Olympiakos Piraeus": "Olympiakos",
        "Atromitos Athens": "Atromitos",
        "Volos FC": "Volos NFC",
        "AE Kifisia FC": "Kifisia",
        "PAOK Thessaloniki": "PAOK",
        "Levadiakos": "Levadeiakos",
        "Panetolikos Agrinio": "Panetolikos",
    },
    "SC0": {},
    "B1": {},
    "T1": {},
}
```

`D1`, `SP1`, `I1`, `F1`, `SC0`, `B1`, `T1` start empty: they're not fetched live by this plan (Task 3 only calls `fetch_pinnacle_odds` for `PRODUCTION_LEAGUES`), so there's no live Odds API team list to diff yet. If `PRODUCTION_LEAGUES` widens later, repeat the Step 2b lookup for the newly added league and fill in its entry the same way. `"Vitória SC": "Guimaraes"` is the one lower-confidence mapping here — Portuguese club "Vitória SC" is commonly known as Vitória de Guimarães, matching football-data's `Guimaraes` abbreviation, but double-check it against an actual Guimarães fixture the first time it's live before trusting it for staking.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_team_aliases.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/data/team_aliases.py tests/test_team_aliases.py
git commit -m "feat: add Odds API team-name alias table for production leagues"
```

---

### Task 2: Pinnacle odds fetcher

**Files:**
- Create: `src/data/pinnacle_odds.py`
- Test: `tests/test_pinnacle_odds.py`

**Interfaces:**
- Consumes: `ODDS_API_TEAM_ALIASES` from `src.data.team_aliases` (Task 1); `PRODUCTION_LEAGUES` from `src.config`.
- Produces: `fetch_pinnacle_odds(leagues: set[str]) -> pd.DataFrame` with columns `league, HomeTeam, AwayTeam, PSH, PSD, PSA`. It accepts *any* subset of `src.config.SUPPORTED_LEAGUES` — `_LEAGUE_TO_SPORT_KEY` maps all 11, not just the 4 production leagues — so a future change to `PRODUCTION_LEAGUES` only needs its callers updated, not this module. `attach_pinnacle_odds(fixtures_df: pd.DataFrame) -> pd.DataFrame`, same shape as `fixtures_df` with `PSH`/`PSD`/`PSA` overwritten where live odds matched.

- [ ] **Step 0: One-time lookup — verify every sport key against the live Odds API**

`_LEAGUE_TO_SPORT_KEY` (Step 3 below) maps all 11 `SUPPORTED_LEAGUES` codes to Odds API sport-key slugs. Before hard-coding them, confirm each one against the live `/v4/sports` list rather than guessing from naming convention:

```bash
set -a; source .env; set +a
curl -s "https://api.the-odds-api.com/v4/sports/?apiKey=${THEODDS_API}" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for d in data:
    if d.get('group') == 'Soccer':
        print(d['key'], '|', d['title'])
"
```

Match each returned `key` to a `LEAGUE_NAMES` entry in `src/config.py` by its `title`/country, and use those exact keys in Step 3. This was already run once on 2026-08-09 against the live API and confirmed all 11 keys below are correct as written — re-run only if The Odds API renames or retires a sport (their `/v4/sports` response is otherwise stable). No `THEODDS_API` credits are consumed by `/v4/sports` — it's a free metadata endpoint, unlike `/v4/sports/{key}/odds/`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pinnacle_odds.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pinnacle_odds.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.pinnacle_odds'`

- [ ] **Step 3: Implement the module**

```python
# src/data/pinnacle_odds.py
"""Live pre-match Pinnacle 1X2 odds for production leagues, via The Odds API."""

import os

import pandas as pd
import requests

from src.data.team_aliases import ODDS_API_TEAM_ALIASES

_ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
# Covers every src.config.SUPPORTED_LEAGUES code, not just the current production
# allowlist, so widening PRODUCTION_LEAGUES later needs no change here. All 11 keys
# were verified against a live GET /v4/sports?apiKey=... call on 2026-08-09 (see
# Task 2, Step 0) — re-verify if The Odds API renames/retires a sport key.
_LEAGUE_TO_SPORT_KEY = {
    "E0": "soccer_epl",
    "D1": "soccer_germany_bundesliga",
    "SP1": "soccer_spain_la_liga",
    "I1": "soccer_italy_serie_a",
    "F1": "soccer_france_ligue_one",
    "N1": "soccer_netherlands_eredivisie",
    "P1": "soccer_portugal_primeira_liga",
    "G1": "soccer_greece_super_league",
    "SC0": "soccer_spl",
    "B1": "soccer_belgium_first_div",
    "T1": "soccer_turkey_super_league",
}
_ODDS_COLUMNS = ["league", "HomeTeam", "AwayTeam", "PSH", "PSD", "PSA"]


def _empty_odds_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_ODDS_COLUMNS)


def _resolve_team_name(league: str, odds_api_name: str) -> str:
    """Map an Odds API team name to its football-data.co.uk name (identity if unmapped)."""
    return ODDS_API_TEAM_ALIASES.get(league, {}).get(odds_api_name, odds_api_name)


def _parse_event(league: str, event: dict) -> dict | None:
    """Return a normalized odds row for one Odds API event, or None if unparseable."""
    pinnacle = next(
        (bk for bk in event.get("bookmakers", []) if bk.get("key") == "pinnacle"), None
    )
    if pinnacle is None:
        return None
    h2h = next((m for m in pinnacle.get("markets", []) if m.get("key") == "h2h"), None)
    if h2h is None:
        return None

    prices = {o.get("name"): o.get("price") for o in h2h.get("outcomes", [])}
    home_name = event.get("home_team")
    away_name = event.get("away_team")
    if home_name not in prices or away_name not in prices or "Draw" not in prices:
        return None

    try:
        psh, psd, psa = float(prices[home_name]), float(prices["Draw"]), float(prices[away_name])
    except (TypeError, ValueError):
        return None

    return {
        "league": league,
        "HomeTeam": _resolve_team_name(league, home_name),
        "AwayTeam": _resolve_team_name(league, away_name),
        "PSH": psh,
        "PSD": psd,
        "PSA": psa,
    }


def fetch_pinnacle_odds(leagues: set[str]) -> pd.DataFrame:
    """Fetch live pre-match Pinnacle 1X2 odds for the given production league codes.

    Never raises: a missing API key, a per-league request failure, or an
    unparseable response degrades to fewer (or zero) rows, never breaks the caller.
    """
    api_key = os.environ.get("THEODDS_API")
    if not api_key:
        print("THEODDS_API not set — skipping live Pinnacle odds")
        return _empty_odds_df()

    rows = []
    for league in leagues:
        sport_key = _LEAGUE_TO_SPORT_KEY.get(league)
        if sport_key is None:
            continue
        try:
            response = requests.get(
                f"{_ODDS_API_BASE}/{sport_key}/odds/",
                params={
                    "apiKey": api_key,
                    "regions": "eu",
                    "markets": "h2h",
                    "bookmakers": "pinnacle",
                    "oddsFormat": "decimal",
                },
                timeout=30,
            )
            response.raise_for_status()
            events = response.json()
        except (requests.RequestException, ValueError) as e:
            print(f"Skipping live Pinnacle odds for {league}: {e}")
            continue

        for event in events:
            row = _parse_event(league, event)
            if row is None:
                print(
                    f"Unmatched or unparseable {league} event: "
                    f"{event.get('home_team')} v {event.get('away_team')}"
                )
                continue
            rows.append(row)

    if not rows:
        return _empty_odds_df()
    return pd.DataFrame(rows, columns=_ODDS_COLUMNS)


def attach_pinnacle_odds(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Left-merge live Pinnacle odds onto fixtures_df, overwriting NaN PSH/PSD/PSA placeholders."""
    from src.config import PRODUCTION_LEAGUES

    odds = fetch_pinnacle_odds(set(PRODUCTION_LEAGUES))
    if odds.empty:
        return fixtures_df

    merged = fixtures_df.merge(
        odds, on=["league", "HomeTeam", "AwayTeam"], how="left", suffixes=("", "_live")
    )
    for col in ["PSH", "PSD", "PSA"]:
        merged[col] = merged[f"{col}_live"].combine_first(merged[col])
        merged = merged.drop(columns=[f"{col}_live"])
    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pinnacle_odds.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/data/pinnacle_odds.py tests/test_pinnacle_odds.py
git commit -m "feat: fetch and attach live Pinnacle odds from The Odds API"
```

---

### Task 3: Wire live odds into the Predict workflow

**Files:**
- Modify: `main.py:247-280` (`_run_predict()`)

**Interfaces:**
- Consumes: `attach_pinnacle_odds` from `src.data.pinnacle_odds` (Task 2).

- [ ] **Step 1: Add the import and call**

In `main.py`, inside `_run_predict()` (around line 248-249), add the import next to the other lazy imports in the function:

```python
def _run_predict():
    from src.data.download import download_fixtures
    from src.data.loader import load_fixtures
    from src.data.pinnacle_odds import attach_pinnacle_odds
```

Then update the fixture-loading block (`main.py:277-279`):

```python
    print("Loading fixtures...")
    fixtures_df = load_fixtures()
    fixtures_df = attach_pinnacle_odds(fixtures_df)
    print(f"Found {len(fixtures_df)} upcoming fixtures in tracked leagues")
```

- [ ] **Step 2: Run the full test suite to confirm no regression**

Run: `uv run pytest tests/ -v`
Expected: all tests pass (this call site has no dedicated unit test — `_run_predict` is exercised only end-to-end via `./predict.sh`/CI, matching the existing pattern for the rest of that function).

- [ ] **Step 3: Manual smoke check (requires `THEODDS_API` set locally)**

Run: `uv run python main.py --predict` (or `./predict.sh`) and confirm the console shows fixture counts and no unhandled exception. If `THEODDS_API` is unset, confirm the one-line "THEODDS_API not set — skipping live Pinnacle odds" log appears and the run completes exactly as before this change.

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat: attach live Pinnacle odds to fixtures in the Predict workflow"
```

---

### Task 4: Restore the Pinnacle-confirmation filter in backtest evaluation (opt-in, default off)

**Files:**
- Modify: `src/config.py`
- Modify: `src/evaluation/metrics.py:86-228` (`compute_value_betting_results`)
- Modify: `tests/test_metrics.py`

**Interfaces:**
- Produces: `src.config.DEFAULT_PINNACLE_CONFIRMATION_MARGIN: float = 0.015`. `compute_value_betting_results(..., pinnacle_confirmation_margin: float | None = None)` — when `None` (default), behavior is unchanged; when set, vetoes a bet unless `pinnacle_fair[outcome] > b365_fair[outcome] + pinnacle_confirmation_margin`, using historical `PSCH`/`PSCD`/`PSCA`, skipping the check (never vetoing) when those columns are null for a row.

- [ ] **Step 1: Add the config constant**

In `src/config.py`, after `DEFAULT_MAX_OVERROUND`:

```python
DEFAULT_MAX_OVERROUND = 0.07
DEFAULT_PINNACLE_CONFIRMATION_MARGIN = 0.015
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_metrics.py`:

```python
def _make_pinnacle_row(psch, pscd, psca):
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=1),
        "HomeTeam": ["Home"],
        "AwayTeam": ["Away"],
        "league": ["E0"],
        "y_true": ["H"],
        "B365H": [2.0],
        "B365D": [4.0],
        "B365A": [4.0],
        "PSCH": [psch],
        "PSCD": [pscd],
        "PSCA": [psca],
    })


def test_pinnacle_confirmation_filter_keeps_bet_when_pinnacle_agrees():
    matches = _make_pinnacle_row(1.8, 6.0, 6.0)  # pinnacle_fair[H] = 0.625 > 0.5 + 0.015
    probabilities = np.array([[0.1, 0.2, 0.7]])
    classes = np.array(["A", "D", "H"])

    result = compute_value_betting_results(
        matches, probabilities, classes, pinnacle_confirmation_margin=0.015,
    )

    assert result["y_pred"].tolist() == ["H"]


def test_pinnacle_confirmation_filter_vetoes_bet_when_pinnacle_disagrees():
    matches = _make_pinnacle_row(3.0, 3.0, 3.0)  # pinnacle_fair[H] = 0.333 <= 0.5 + 0.015
    probabilities = np.array([[0.1, 0.2, 0.7]])
    classes = np.array(["A", "D", "H"])

    result = compute_value_betting_results(
        matches, probabilities, classes, pinnacle_confirmation_margin=0.015,
    )

    assert result.empty


def test_pinnacle_confirmation_filter_skipped_when_pinnacle_odds_are_null():
    matches = _make_pinnacle_row(float("nan"), float("nan"), float("nan"))
    probabilities = np.array([[0.1, 0.2, 0.7]])
    classes = np.array(["A", "D", "H"])

    result = compute_value_betting_results(
        matches, probabilities, classes, pinnacle_confirmation_margin=0.015,
    )

    assert result["y_pred"].tolist() == ["H"]


def test_pinnacle_confirmation_filter_is_off_by_default():
    matches = _make_pinnacle_row(3.0, 3.0, 3.0)  # would veto if the filter were active
    probabilities = np.array([[0.1, 0.2, 0.7]])
    classes = np.array(["A", "D", "H"])

    result = compute_value_betting_results(matches, probabilities, classes)

    assert result["y_pred"].tolist() == ["H"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL — `compute_value_betting_results() got an unexpected keyword argument 'pinnacle_confirmation_margin'`

- [ ] **Step 4: Implement the filter**

In `src/evaluation/metrics.py`, add the parameter to the signature (`metrics.py:86-101`):

```python
def compute_value_betting_results(
    df: pd.DataFrame,
    y_proba,
    classes,
    threshold: float = 0.0,
    kelly_fraction: float = 0.0,
    edge_baseline: str = "fair",
    inv_odds_factor: float = 0.0,
    min_stake: float = 1.0,
    max_odds: float = float("inf"),
    skip_leagues: set | None = None,
    skip_outcomes: set | None = None,
    max_edge: float = float("inf"),
    min_season_games: int = 0,
    max_overround: float = float("inf"),
    pinnacle_confirmation_margin: float | None = None,
) -> pd.DataFrame:
```

Add one line to the docstring, after the `max_edge` line:

```
    pinnacle_confirmation_margin: if set, only place a bet when the historical Pinnacle
                      fair probability (PSCH/PSCD/PSCA) exceeds the B365 fair probability
                      by more than this margin. None (default) disables the check.
                      Rows with null Pinnacle columns are never vetoed.
```

After the `fair` dict is computed (`metrics.py:171-172`), compute a null-safe `pinnacle_fair` once per row:

```python
        # Vig-corrected fair probabilities (sum to 1.0)
        fair = {outcome: raw[outcome] / total_implied for outcome in raw}

        pinnacle_fair = None
        if pinnacle_confirmation_margin is not None:
            psch, pscd, psca = row.get("PSCH"), row.get("PSCD"), row.get("PSCA")
            if (
                psch is not None and pscd is not None and psca is not None
                and not (pd.isna(psch) or pd.isna(pscd) or pd.isna(psca))
            ):
                pinnacle_raw = {
                    "H": 1.0 / float(psch),
                    "D": 1.0 / float(pscd),
                    "A": 1.0 / float(psca),
                }
                pinnacle_total = sum(pinnacle_raw.values())
                pinnacle_fair = {o: pinnacle_raw[o] / pinnacle_total for o in pinnacle_raw}
```

In the per-outcome loop, right after the edge threshold check (`metrics.py:185-187`), add the veto:

```python
            edge = model_prob - baseline_prob
            if edge <= threshold or edge > max_edge:
                continue

            if (
                pinnacle_fair is not None
                and pinnacle_fair[outcome] <= fair[outcome] + pinnacle_confirmation_margin
            ):
                continue
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS (all tests, including the 4 new ones)

- [ ] **Step 6: Commit**

```bash
git add src/config.py src/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: restore opt-in Pinnacle-confirmation filter in backtest evaluation"
```

---

### Task 5: Land the same filter for live predictions (opt-in, stays disabled)

**Files:**
- Modify: `main.py:358-423` (`_build_prediction_rows`)
- Modify: `tests/test_shadow_cli.py`

**Interfaces:**
- Produces: `_build_prediction_rows(fixture_features, y_proba, classes, threshold, league_thresholds=None, pinnacle_confirmation_margin=None)`. Same veto semantics as Task 4, using live `PSH`/`PSD`/`PSA`. The `_run_predict()` call site (`main.py:340-341`) is **not** changed to pass a non-`None` value — the design spec gates enabling this on a separate backtest re-verification, tracked outside this plan.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_shadow_cli.py`:

```python
def _make_pinnacle_fixture(psh, psd, psa):
    return pd.DataFrame({
        "Date": [pd.Timestamp("2026-08-09")],
        "league": ["E0"],
        "HomeTeam": ["Arsenal"],
        "AwayTeam": ["Chelsea"],
        "B365H": [2.0],
        "B365D": [4.0],
        "B365A": [4.0],
        "PSH": [psh],
        "PSD": [psd],
        "PSA": [psa],
    })


def test_build_prediction_rows_vetoes_bet_when_pinnacle_disagrees():
    fixture_features = _make_pinnacle_fixture(3.0, 3.0, 3.0)  # pinnacle_fair[H] = 0.333

    rows = main._build_prediction_rows(
        fixture_features,
        np.array([[0.1, 0.2, 0.7]]),
        ["A", "D", "H"],
        threshold=0.0,
        pinnacle_confirmation_margin=0.015,
    )

    assert rows[0]["ValueBets"] == []


def test_build_prediction_rows_keeps_bet_when_pinnacle_agrees():
    fixture_features = _make_pinnacle_fixture(1.8, 6.0, 6.0)  # pinnacle_fair[H] = 0.625

    rows = main._build_prediction_rows(
        fixture_features,
        np.array([[0.1, 0.2, 0.7]]),
        ["A", "D", "H"],
        threshold=0.0,
        pinnacle_confirmation_margin=0.015,
    )

    assert [o for o, _ in rows[0]["ValueBets"]] == ["H"]


def test_build_prediction_rows_skips_pinnacle_check_when_odds_are_null():
    fixture_features = _make_pinnacle_fixture(float("nan"), float("nan"), float("nan"))

    rows = main._build_prediction_rows(
        fixture_features,
        np.array([[0.1, 0.2, 0.7]]),
        ["A", "D", "H"],
        threshold=0.0,
        pinnacle_confirmation_margin=0.015,
    )

    assert [o for o, _ in rows[0]["ValueBets"]] == ["H"]


def test_build_prediction_rows_pinnacle_check_is_off_by_default():
    fixture_features = _make_pinnacle_fixture(3.0, 3.0, 3.0)  # would veto if active

    rows = main._build_prediction_rows(
        fixture_features,
        np.array([[0.1, 0.2, 0.7]]),
        ["A", "D", "H"],
        threshold=0.0,
    )

    assert [o for o, _ in rows[0]["ValueBets"]] == ["H"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_shadow_cli.py -v`
Expected: FAIL — `_build_prediction_rows() got an unexpected keyword argument 'pinnacle_confirmation_margin'`

- [ ] **Step 3: Implement the filter**

In `main.py`, update the `_build_prediction_rows` signature (`main.py:358-361`):

```python
def _build_prediction_rows(
    fixture_features, y_proba, classes, threshold: float,
    league_thresholds: dict | None = None,
    pinnacle_confirmation_margin: float | None = None,
) -> list[dict]:
```

After the overround check and before computing `league`/`t` (`main.py:373-378`), compute a null-safe `pinnacle_fair`:

```python
        # Skip high-vig markets (same filter as backtest)
        overround = total_implied - 1.0
        if overround > _PREDICT_MAX_OVERROUND:
            continue

        pinnacle_fair = None
        if pinnacle_confirmation_margin is not None:
            psh, psd, psa = row.get("PSH"), row.get("PSD"), row.get("PSA")
            if (
                psh is not None and psd is not None and psa is not None
                and not (pd.isna(psh) or pd.isna(psd) or pd.isna(psa))
            ):
                pinnacle_raw = {
                    "H": 1.0 / float(psh),
                    "D": 1.0 / float(psd),
                    "A": 1.0 / float(psa),
                }
                pinnacle_total = sum(pinnacle_raw.values())
                pinnacle_fair = {o: pinnacle_raw[o] / pinnacle_total for o in pinnacle_raw}

        league = row.get("league", "")
        t = (league_thresholds or {}).get(league, threshold)
```

In the per-outcome loop, add the veto right before the `is_execution_eligible` call (`main.py:380-393`):

```python
        value_bets = []
        for o in ["H", "D", "A"]:
            edge = probs[o] - fair[o]
            b365_odds = {"H": b365h, "D": b365d, "A": b365a}[o]
            if (
                pinnacle_fair is not None
                and pinnacle_fair[o] <= fair[o] + pinnacle_confirmation_margin
            ):
                continue
            if is_execution_eligible(
                edge=edge,
                threshold=t,
                b365_odds=b365_odds,
                fixture_overround=overround,
                max_edge=_PREDICT_MAX_EDGE,
                max_odds=_PREDICT_MAX_ODDS,
                max_overround=_PREDICT_MAX_OVERROUND,
            ):
                value_bets.append((o, edge))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_shadow_cli.py -v`
Expected: PASS (all tests, including the 4 new ones)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest tests/ -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_shadow_cli.py
git commit -m "feat: land opt-in Pinnacle-confirmation filter for live predictions (disabled)"
```

---

## Explicitly not in this plan

- **Enabling** the filter in production (`_run_predict()` still calls `_build_prediction_rows` without `pinnacle_confirmation_margin`). Per the design spec's "Evaluation plan," that requires running `uv run python main.py --per-league --threshold 0.0` with the filter on vs. off against the current model, applying the `EVALUATION.md` keep/revert rules, and recording the outcome in `autoresearch/experiments.md` / `autoresearch/current.md`. That is a separate autoresearch iteration, not a coding task — use the `autoresearch` skill for it once this plan lands.
- **Populating** the `ODDS_API_TEAM_ALIASES` tables for the 7 non-production leagues (`D1`, `SP1`, `I1`, `F1`, `SC0`, `B1`, `T1`) — their sport keys are verified (Task 2, Step 0), but their team-name tables stay empty until `PRODUCTION_LEAGUES` widens or someone runs a manual smoke call against them (repeat Task 1's Step 2b lookup for the new league). The 4 production leagues (`E0`/`N1`/`P1`/`G1`) are already populated from a live diff run on 2026-08-09 (Task 1, Step 3) — though note the "Vitória SC" → "Guimaraes" mapping there is a lower-confidence guess flagged for a first-live-fixture sanity check, not a hard-verified diacritic match.
- **Re-diffing** the production alias tables against a fresh live pull before this goes live for real staking — the 2026-08-09 lookup covered whatever fixtures the API had listed that day; a club not in that day's fixture window (end of season, international break) wouldn't have been observed and would silently fail to merge until it's seen once and added by hand.
- Adding `THEODDS_API` as a GitHub Actions repository secret — explicitly a manual, non-scripted step per the design spec.
