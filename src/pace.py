"""Current-season win projection for weekly scoring.

The model is trained on finished seasons, so the active-season token uses a
full-season win total: the median posted total on the VegasInsider board.
That number already includes wins banked. Teams missing from the board fall
back to wins so far plus the preseason rate on the games left.

nflverse supplies season-to-date wins and losses either way. When the season
has no completed regular-season games, callers should skip export.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from io import StringIO

import pandas as pd
from bs4 import BeautifulSoup as bs

from .reference import canonical_team

VEGASINSIDER_URL = "https://www.vegasinsider.com/nfl/odds/win-totals/"
NFLVERSE_GAMES_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
)
_TOTAL_RE = re.compile(r"[ou]?\s*(\d+(?:\.\d+)?)\s*$", re.IGNORECASE)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


class PaceError(RuntimeError):
    """Season results are missing or unusable. Do not publish."""


@dataclass(frozen=True)
class PaceLoad:
    """Team table indexed by canonical abbrev, plus fetch bookkeeping."""

    table: pd.DataFrame
    completed_games: int
    market_teams: int
    blend_teams: int


def median_total(points: list[float]) -> float:
    """Median of posted season win totals. Even counts average the middle pair."""
    if not points:
        raise ValueError("median_total requires at least one number")
    return float(statistics.median(points))


def blend_proj_wins(
    wins: int,
    losses: int,
    ties: int,
    preseason_wins_exp: float,
    games: int,
) -> float:
    """Wins so far, plus the preseason rate applied to the games left."""
    if games <= 0:
        raise ValueError(f"games must be positive, got {games}")
    played = wins + losses + ties
    games_left = max(0, games - played)
    return float(wins) + 0.5 * float(ties) + (float(preseason_wins_exp) / games) * games_left


def parse_posted_total(text: str) -> float | None:
    """Pull a win-total point from a cell. Prices like -175 are ignored."""
    match = _TOTAL_RE.fullmatch(text.strip())
    if match is None:
        return None
    value = float(match.group(1))
    if value <= 0 or value > 20:
        return None
    return value


def parse_vegasinsider_totals(html: str) -> dict[str, list[float]]:
    """Map source abbrev → posted totals from each book that has a number."""
    soup = bs(html, "lxml")
    found: dict[str, list[float]] = {}
    for tr in soup.find_all("tr"):
        link = tr.find("a", attrs={"data-abbr": True})
        if link is None:
            continue
        abbr = str(link["data-abbr"]).strip().upper()
        if not abbr:
            continue
        totals: list[float] = []
        for td in tr.find_all("td", class_="game-odds"):
            for span in td.find_all("span", class_="data-value"):
                value = parse_posted_total(span.get_text(strip=True))
                if value is not None:
                    totals.append(value)
                    break
        found[abbr] = totals
    return found


def records_from_schedule(
    games: pd.DataFrame,
    *,
    season: int,
    games_per_season: int,
    aliases: dict[str, str],
    valid: set[str],
) -> tuple[pd.DataFrame, int]:
    """Aggregate completed regular-season W-L-T.

    Returns a frame indexed by canonical abbrev and the count of completed
    games. Raises ``PaceError`` when a started season is missing a team or
    a full schedule. Zero completed games is an offseason / pre-kickoff skip.
    """
    needed = {"season", "game_type", "home_team", "away_team", "home_score", "away_score"}
    missing = needed - set(games.columns)
    if missing:
        raise PaceError(f"nflverse schedule missing columns: {sorted(missing)}")

    slate = games[
        (games["season"] == season) & (games["game_type"] == "REG")
    ].copy()
    if slate.empty:
        return pd.DataFrame(), 0

    completed = slate.dropna(subset=["home_score", "away_score"])
    if completed.empty:
        return pd.DataFrame(), 0

    counts = pd.concat([slate["home_team"], slate["away_team"]], ignore_index=True)
    by_team = counts.value_counts()
    if len(by_team) != 32 or not bool((by_team == games_per_season).all()):
        bad = by_team[by_team != games_per_season]
        raise PaceError(
            f"{season} schedule is not {32} teams × {games_per_season} games "
            f"({len(by_team)} teams; mismatches={bad.to_dict()})"
        )

    wins: dict[str, int] = {}
    losses: dict[str, int] = {}
    ties: dict[str, int] = {}

    def bump(team: str, column: dict[str, int]) -> None:
        column[team] = column.get(team, 0) + 1

    for row in completed.itertuples(index=False):
        home = str(row.home_team)
        away = str(row.away_team)
        if row.home_score > row.away_score:
            bump(home, wins)
            bump(away, losses)
        elif row.away_score > row.home_score:
            bump(away, wins)
            bump(home, losses)
        else:
            bump(home, ties)
            bump(away, ties)

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in by_team.index.astype(str):
        canon = canonical_team(raw, aliases=aliases, valid=valid)
        if canon is None:
            raise PaceError(f"Unmapped nflverse abbrev {raw}")
        if canon in seen:
            raise PaceError(f"Multiple schedule abbrevs map to {canon}")
        seen.add(canon)
        rows.append(
            {
                "abbrev": canon,
                "wins": wins.get(raw, 0),
                "losses": losses.get(raw, 0),
                "ties": ties.get(raw, 0),
            }
        )
    table = pd.DataFrame(rows).set_index("abbrev").sort_index()
    return table, int(len(completed))


def project_wins(
    records: pd.DataFrame,
    market: dict[str, list[float]],
    *,
    games: int,
    preseason_wins: pd.Series,
    aliases: dict[str, str],
    valid: set[str],
) -> pd.DataFrame:
    """Attach ``proj_wins``, ``proj_win_pct``, and ``source`` to season records.

    ``market`` keys are source abbrevs (nflverse / VegasInsider), not necessarily
    the canonical codes on ``records``.
    """
    by_canon: dict[str, list[float]] = {}
    for raw, points in market.items():
        if not points:
            continue
        canon = canonical_team(str(raw).upper(), aliases=aliases, valid=valid)
        if canon is None:
            print(f"pace: ignoring unmapped win-total abbrev {raw}")
            continue
        by_canon.setdefault(canon, []).extend(points)

    out = records.copy()
    proj_wins: list[float] = []
    sources: list[str] = []
    for abbrev, row in out.iterrows():
        points = by_canon.get(str(abbrev), [])
        if points:
            proj_wins.append(median_total(points))
            sources.append("market")
            continue
        if abbrev not in preseason_wins.index:
            raise PaceError(f"No market total and no preseason wins_exp for {abbrev}")
        proj_wins.append(
            blend_proj_wins(
                int(row["wins"]),
                int(row["losses"]),
                int(row["ties"]),
                float(preseason_wins.loc[abbrev]),
                games,
            )
        )
        sources.append("blend")
    out["proj_wins"] = proj_wins
    out["proj_win_pct"] = [round(w / games, 2) for w in proj_wins]
    out["source"] = sources
    return out


def load_pace(
    *,
    season: int,
    games: int,
    preseason_wins: pd.Series,
    aliases: dict[str, str],
    valid: set[str],
) -> PaceLoad:
    """Fetch the win-total board and nflverse results, then project each team."""
    schedule = _read_schedule()
    records, completed = records_from_schedule(
        schedule,
        season=season,
        games_per_season=games,
        aliases=aliases,
        valid=valid,
    )
    if completed == 0:
        print(f"pace: no completed regular-season games for {season}")
        return PaceLoad(
            table=pd.DataFrame(),
            completed_games=0,
            market_teams=0,
            blend_teams=0,
        )

    market = _load_market_totals()
    table = project_wins(
        records,
        market,
        games=games,
        preseason_wins=preseason_wins,
        aliases=aliases,
        valid=valid,
    )
    market_teams = int((table["source"] == "market").sum())
    blend_teams = int((table["source"] == "blend").sum())
    blended = table.index[table["source"] == "blend"].astype(str).tolist()
    print(
        f"pace: market={market_teams} blend={blend_teams} "
        f"({', '.join(blended) if blended else 'none'}) "
        f"completed_games={completed}"
    )
    return PaceLoad(
        table=table,
        completed_games=completed,
        market_teams=market_teams,
        blend_teams=blend_teams,
    )


def _load_market_totals() -> dict[str, list[float]]:
    try:
        html = _fetch_text(VEGASINSIDER_URL)
        parsed = parse_vegasinsider_totals(html)
    except Exception as exc:
        print(f"pace: win-total board unavailable ({exc}); using blend for every team")
        return {}
    if len(parsed) < 32:
        print(
            f"pace: win-total board parsed {len(parsed)} teams; "
            "missing teams use the blend"
        )
    return parsed


def _read_schedule() -> pd.DataFrame:
    try:
        text = _fetch_text(NFLVERSE_GAMES_URL)
        return pd.read_csv(StringIO(text))
    except Exception as exc:
        raise PaceError(f"could not read nflverse schedule: {exc}") from exc


def _fetch_text(url: str, *, timeout: float = 60.0) -> str:
    try:
        from curl_cffi import requests as crequests

        resp = crequests.get(url, headers=_HEADERS, timeout=timeout, impersonate="chrome")
        status = resp.status_code
        text = resp.text
    except ImportError:
        import requests

        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        status = resp.status_code
        text = resp.text
    if status >= 400 or not text:
        raise RuntimeError(f"HTTP {status} for {url}")
    return text
