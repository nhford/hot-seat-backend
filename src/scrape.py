"""Scrape Pro-Football-Reference / odds caches into data/scraped/.

Usage (from repo root):
  python -m src.scrape index
  python -m src.scrape index --html coaches.htm   # if Cloudflare blocks live fetch
  python -m src.scrape coaches
  python -m src.scrape coaches --id ReidAn0 --html ReidAn0.htm
  python -m src.scrape odds
  python -m src.scrape odds --year 2024 --fetch
  python -m src.scrape standings --year 2025 --html years_2025.htm
  python -m src.scrape coy --year 2025 --html awards_2025.htm
  python -m src.scrape ingest-coaches --year 2025
  python -m src.scrape teams --year 2025 --fetch
  python -m src.scrape all
"""

from __future__ import annotations

import argparse
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup as bs
from tqdm import tqdm

from .cache import load_or_build
from .fetch import get_html, load_html, read_html
from .reference import load_team_aliases
from .season import ROOT, Settings, load_settings

COACHES_INDEX_URL = "https://www.pro-football-reference.com/coaches/"
COACHES_PATH = ROOT / "data" / "scraped" / "coaches.csv"
COACHES_DIR = ROOT / "data" / "scraped" / "coaches"
COACHES_RAW_DIR = ROOT / "data" / "scraped" / "coaches_raw"
ODDS_DIR = ROOT / "data" / "scraped" / "odds"
STANDINGS_DIR = ROOT / "data" / "scraped" / "standings"
AWARDS_DIR = ROOT / "data" / "scraped" / "awards"
TEAMS_DIR = ROOT / "data" / "scraped" / "teams"
TEAMS_CSV = ROOT / "config" / "teams.csv"
DERIVED_DIR = ROOT / "data" / "derived"

COACH_SEASON_COLUMNS = [
    "Year",
    "Age",
    "Tm",
    "Lg",
    "G",
    "W",
    "L",
    "T",
    "W-L%",
    "SRS",
    "OSRS",
    "DSRS",
    "G plyf",
    "W plyf",
    "L plyf",
    "W-L% plyf",
    "Rank",
    "Num",
    "Won",
    "Notes",
]

ODDS_START_YEAR = 1989
HISTORY_START_YEAR = 1970

GM_ROLE_NAMES = [
    "General Manager",
    "of Player Personnel:",
    "Exec. VP of Football Ops",
]
OWNER_ROLE_NAMES = [
    "Owner",
    "CEO",
    "Chairman",
    "Chair:",
    "President:",
    "Secretary of the Board of Directors",
]


def clean_hof(name: str) -> str:
    return name[:-1] if name.endswith("+") else name


def clean_team_name(team: str, team_aliases: dict[str, str] | None = None) -> str:
    """Strip playoff markers and map historical names → current."""
    aliases = team_aliases if team_aliases is not None else load_team_aliases()
    if team and team[-1] in "+*":
        team = team[:-1]
    return aliases.get(team, team)


def load_teams_table() -> pd.DataFrame:
    return pd.read_csv(TEAMS_CSV, index_col=0)


def _resolve_years(
    *,
    start: int,
    end: int | None = None,
    year: int | None = None,
) -> list[int]:
    settings = load_settings()
    if year is not None:
        return [year]
    end = settings.history_end if end is None else end
    return list(range(start, end + 1))


def _parse_coaches_index_html(html: str) -> pd.DataFrame:
    """Build fresh id/name/to frame from a PFR coaches index HTML document."""
    soup = bs(html, "lxml")
    table = soup.find("table", id="coaches")
    if table is None:
        raise RuntimeError(
            "Could not find table#coaches in HTML; "
            "save the full PFR coaches index page and retry with --html."
        )

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    body = table.find("tbody") or table
    for tr in body.find_all("tr"):
        link = tr.find("a", href=True)
        href = (link.get("href") if link else "") or ""
        if "/coaches/" not in href or not href.endswith(".htm"):
            continue
        coach_id = href.split("/")[-1].split(".")[0]
        if not coach_id or coach_id in seen:
            continue
        seen.add(coach_id)

        name_cell = tr.find(attrs={"data-stat": "coach"})
        to_cell = tr.find(attrs={"data-stat": "year_max"})
        raw_name = (
            name_cell.get_text(strip=True)
            if name_cell is not None
            else (link.get_text(strip=True) if link else "")
        )
        to_raw = to_cell.get_text(strip=True) if to_cell is not None else ""
        rows.append(
            {
                "id": coach_id,
                "name": clean_hof(raw_name),
                "to": int(to_raw) if to_raw.isdigit() else to_raw,
            }
        )

    if not rows:
        raise RuntimeError(
            "No coach rows parsed from table#coaches; "
            "PFR page structure may have changed (or the HTML is incomplete)."
        )
    return pd.DataFrame(rows)


def _upsert_coaches_frame(fresh: pd.DataFrame) -> pd.DataFrame:
    if COACHES_PATH.exists():
        existing = pd.read_csv(COACHES_PATH)
        before_ids = set(existing["id"])
        key = existing.merge(fresh, on="id", how="outer", suffixes=("_old", ""))
        for col in ["name", "to"]:
            key[col] = key[col].combine_first(key[f"{col}_old"])
            key = key.drop(columns=[f"{col}_old"])
        key = key[["id", "name", "to"]]
        n_new = int((~key["id"].isin(before_ids)).sum())
        n_updated = int(key["id"].isin(before_ids).sum())
        print(
            f"Upserted {COACHES_PATH.relative_to(ROOT)}: "
            f"{n_updated} existing, {n_new} new ({len(key)} total)"
        )
    else:
        key = fresh
        print(f"Created {COACHES_PATH.relative_to(ROOT)} with {len(key)} coaches")

    COACHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    key.to_csv(COACHES_PATH, index=False)
    return key


def upsert_coaches_index(
    *,
    sleep: float = 3.0,
    html_path: str | Path | None = None,
    cookie: str | None = None,
) -> pd.DataFrame:
    """Update data/scraped/coaches.csv from PFR (live fetch or a saved HTML file)."""
    if html_path is not None:
        print(f"Parsing coaches index from {html_path}")
    html = load_html(
        COACHES_INDEX_URL, html_path=html_path, sleep=sleep, cookie=cookie
    )
    return _upsert_coaches_frame(_parse_coaches_index_html(html))


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten PFR multi-index coach table columns."""
    out = df.copy()
    out.columns = out.columns.get_level_values(1)
    cols = list(out.columns)
    # Second W-L% is playoff win%.
    second = cols.index("W-L%", cols.index("W-L%") + 1)
    out.columns.values[second] = "W-L% plyf"
    return out


def coach_csv_path(coach_id: str) -> Path:
    return COACHES_DIR / f"{coach_id}.csv"


def get_coach(
    coach_id: str,
    *,
    fetch: bool = False,
    sleep: float = 3.0,
    cookie: str | None = None,
    html_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load or scrape one coach's season table into data/scraped/coaches/{id}.csv."""
    path = coach_csv_path(coach_id)

    def build() -> pd.DataFrame:
        url = f"https://www.pro-football-reference.com/coaches/{coach_id}.htm"
        table = read_html(
            url, html_path=html_path, sleep=sleep, cookie=cookie
        )[0]
        return clean_columns(table)

    return load_or_build(
        path, build, fetch=fetch or html_path is not None, index_col=0
    )


def scrape_coaches(
    coach_ids: list[str],
    *,
    fetch: bool = False,
    sleep: float = 3.0,
) -> None:
    COACHES_DIR.mkdir(parents=True, exist_ok=True)
    for coach_id in tqdm(coach_ids, desc="coaches"):
        get_coach(coach_id, fetch=fetch, sleep=sleep)


def normalize_raw_coach_csv(path: str | Path) -> pd.DataFrame:
    """Normalize a manual PFR coaching-results CSV into scrape-cache columns."""
    path = Path(path)
    raw = pd.read_csv(path, header=1)
    rename = {
        "G.1": "G plyf",
        "W.1": "W plyf",
        "L.1": "L plyf",
        "W-L%.1": "W-L% plyf",
    }
    df = raw.rename(columns=rename)
    missing = [c for c in COACH_SEASON_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    out = df[COACH_SEASON_COLUMNS].copy()
    # Match existing caches: blank Notes → NaN, numeric cols stay numeric.
    out["Notes"] = out["Notes"].replace("", pd.NA)
    return out.reset_index(drop=True)


def ingest_raw_coaches(
    raw_dir: str | Path | None = None,
    *,
    year: int | None = None,
) -> list[str]:
    """Write normalized CSVs from coaches_raw/{year}/ into data/scraped/coaches/.

    Returns the list of coach ids ingested.
    """
    settings = load_settings()
    y = settings.history_end if year is None else year
    src = Path(raw_dir) if raw_dir is not None else (COACHES_RAW_DIR / str(y))
    if not src.is_dir():
        raise FileNotFoundError(
            f"Missing raw coaches dir {src.relative_to(ROOT)}; "
            "expected manual PFR table dumps there."
        )

    COACHES_DIR.mkdir(parents=True, exist_ok=True)
    ingested: list[str] = []
    for path in sorted(src.glob("*.csv")):
        if path.stat().st_size == 0:
            print(f"skip empty {path.name}")
            continue
        coach_id = path.stem
        df = normalize_raw_coach_csv(path)
        out = coach_csv_path(coach_id)
        df.to_csv(out)
        years = [str(v) for v in df["Year"] if str(v).isdigit()]
        has_year = str(y) in years
        print(
            f"ingested {coach_id}: {len(df)} rows "
            f"(has {y}={has_year}) → {out.relative_to(ROOT)}"
        )
        ingested.append(coach_id)
    print(f"Ingested {len(ingested)} coaches from {src.relative_to(ROOT)}")
    return ingested


def active_coach_ids(settings: Settings | None = None) -> list[str]:
    settings = settings or load_settings()
    if not COACHES_PATH.exists():
        raise FileNotFoundError(
            f"Missing {COACHES_PATH.relative_to(ROOT)}; run: python -m src.scrape index"
        )
    coaches = pd.read_csv(COACHES_PATH)
    active = coaches[coaches["to"] == settings.history_end]
    return active["id"].tolist()


def get_odds_table(
    year: int,
    *,
    fetch: bool = False,
    sleep: float = 3.0,
) -> pd.DataFrame:
    path = ODDS_DIR / f"ou_{year}.csv"

    def build() -> pd.DataFrame:
        url = (
            f"https://www.sportsoddshistory.com/nfl-win/"
            f"?y={year}&sa=nfl&t=win&o=t"
        )
        odds = read_html(url, sleep=sleep)[0].sort_values(
            by=["Win Total", "Over Odds"], ascending=[False, True]
        )
        clean = odds[["Team", "Win Total", "Actual Wins"]].reset_index(drop=True)
        clean.columns = ["Team", "OU", "Wins"]
        return clean

    return load_or_build(path, build, fetch=fetch, index_col=0)


def scrape_odds(
    *,
    start: int = ODDS_START_YEAR,
    end: int | None = None,
    year: int | None = None,
    fetch: bool = False,
    sleep: float = 3.0,
) -> None:
    years = _resolve_years(start=start, end=end, year=year)
    for y in tqdm(years, desc="odds"):
        get_odds_table(y, fetch=fetch, sleep=sleep)


# --- Standings -----------------------------------------------------------------


def _standings_tables_from_html(html: str) -> list[pd.DataFrame]:
    """Prefer table#AFC / table#NFC; fall back to first two page tables."""
    soup = bs(html, "lxml")
    frames: list[pd.DataFrame] = []
    for table_id in ("AFC", "NFC"):
        table = soup.find("table", id=table_id)
        if table is not None:
            frames.append(pd.read_html(StringIO(str(table)), flavor="lxml")[0])
    if len(frames) == 2:
        return frames
    tables = pd.read_html(StringIO(html), flavor="lxml")
    if len(tables) < 2:
        raise RuntimeError(
            "Could not find AFC/NFC standings tables in HTML; "
            "save the full PFR year page and retry with --html."
        )
    return [tables[0], tables[1]]


def get_standings(
    year: int,
    *,
    fetch: bool = False,
    sleep: float = 3.0,
    cookie: str | None = None,
    html_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load or scrape one season's standings into data/scraped/standings/."""
    path = STANDINGS_DIR / f"standings_{year}.csv"
    aliases = load_team_aliases()
    teams = load_teams_table()

    def build() -> pd.DataFrame:
        url = f"https://www.pro-football-reference.com/years/{year}/"
        if html_path is not None:
            print(f"Parsing standings {year} from {html_path}")
        html = load_html(
            url, html_path=html_path, sleep=sleep, cookie=cookie
        )
        afc, nfc = _standings_tables_from_html(html)
        nfl = pd.concat([afc, nfc])[["Tm", "W-L%"]].rename(columns={"Tm": "Team"})
        nfl["Team"] = nfl["Team"].map(lambda t: clean_team_name(str(t), aliases))
        nfl = nfl.merge(teams, on="Team")
        return nfl[["Abbrev", "Team", "W-L%"]].reset_index(drop=True)

    return load_or_build(
        path, build, fetch=fetch or html_path is not None, index_col=0
    )


def scrape_standings(
    *,
    start: int = HISTORY_START_YEAR,
    end: int | None = None,
    year: int | None = None,
    fetch: bool = False,
    sleep: float = 3.0,
    cookie: str | None = None,
    html_path: str | Path | None = None,
) -> None:
    STANDINGS_DIR.mkdir(parents=True, exist_ok=True)
    if html_path is not None:
        settings = load_settings()
        y = year if year is not None else settings.history_end
        get_standings(y, html_path=html_path, sleep=sleep, cookie=cookie)
        return
    years = _resolve_years(start=start, end=end, year=year)
    for y in tqdm(years, desc="standings"):
        get_standings(y, fetch=fetch, sleep=sleep, cookie=cookie)


def rebuild_standings_derived(
    *,
    start: int = HISTORY_START_YEAR,
    end: int | None = None,
) -> pd.DataFrame:
    """Pivot year caches → data/derived/standings.csv."""
    settings = load_settings()
    end = settings.history_end if end is None else end
    merged = pd.DataFrame()
    for yr in range(start, end + 1):
        path = STANDINGS_DIR / f"standings_{yr}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0).rename(columns={"W-L%": str(yr)})
        merged = df if merged.empty else merged.merge(
            df, how="outer", on=["Abbrev", "Team"]
        )
    if merged.empty:
        raise FileNotFoundError(
            f"No standings caches in {STANDINGS_DIR.relative_to(ROOT)}; "
            "run: python -m src.scrape standings"
        )
    merged = merged.fillna(0).set_index("Abbrev")
    out = DERIVED_DIR / "standings.csv"
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out)
    print(f"Wrote {out.relative_to(ROOT)} ({start}–{end})")
    return merged


# --- Coach of the Year ---------------------------------------------------------


def approx_coy_share(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing down-ballot shares for pre-2022 COY tables."""
    out = df.copy()
    remainder = 1 - out["coy_share"].sum()
    zeroes = out[out["coy_share"] == 0.0].index
    distributed: list[float] = []
    for _ in range(len(zeroes)):
        share_to_add = max(round(remainder / 2, 3), 0.004)
        distributed.append(share_to_add)
        remainder -= share_to_add
    out.loc[zeroes, "coy_share"] = distributed
    return out


def _parse_coy_tables(tables: list[pd.DataFrame], year: int) -> pd.DataFrame:
    """Normalize PFR awards-page tables into rank/name/coy_share."""
    df = tables[-1].droplevel([0], axis=1)
    if year < 2022:
        df = df[["Rk", "Coach", "share"]].rename(columns={"share": "coy_share"})
        df["coy_share"] = (
            df["coy_share"]
            .fillna("0%")
            .map(lambda x: round(float(str(x)[:-1]) / 100, 3))
        )
        df = approx_coy_share(df)
    else:
        df = df[["Rk", "Coach", "Vote Pts"]].copy()
        df["coy_share"] = round(df["Vote Pts"] / df["Vote Pts"].sum(), 3)
        df = df.drop(columns=["Vote Pts"])
    df.columns = ["rank", "name", "coy_share"]
    return df.reset_index(drop=True)


def get_coy_voting(
    year: int,
    *,
    fetch: bool = False,
    sleep: float = 3.0,
    cookie: str | None = None,
    html_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load or scrape AP COY voting into data/scraped/awards/coy_{year}.csv.

    Pass ``html_path`` to parse a locally saved awards page (Cloudflare workaround).
    """
    path = AWARDS_DIR / f"coy_{year}.csv"

    def build() -> pd.DataFrame:
        url = (
            f"https://www.pro-football-reference.com/awards/"
            f"awards_{year}.htm#voting_apcoy"
        )
        if html_path is not None:
            print(f"Parsing COY {year} from {html_path}")
        tables = read_html(
            url, html_path=html_path, sleep=sleep, cookie=cookie
        )
        return _parse_coy_tables(tables, year)

    return load_or_build(
        path, build, fetch=fetch or html_path is not None, index_col=0
    )


def scrape_coy(
    *,
    start: int = HISTORY_START_YEAR,
    end: int | None = None,
    year: int | None = None,
    fetch: bool = False,
    sleep: float = 3.0,
    cookie: str | None = None,
    html_path: str | Path | None = None,
) -> None:
    AWARDS_DIR.mkdir(parents=True, exist_ok=True)
    if html_path is not None:
        settings = load_settings()
        y = year if year is not None else settings.history_end
        get_coy_voting(y, html_path=html_path)
        return
    years = _resolve_years(start=start, end=end, year=year)
    for y in tqdm(years, desc="coy"):
        get_coy_voting(y, fetch=fetch, sleep=sleep, cookie=cookie)


def rebuild_coy_derived(
    *,
    start: int = HISTORY_START_YEAR,
    end: int | None = None,
) -> pd.DataFrame:
    """Join year COY caches to coach ids → data/derived/coy.csv."""
    settings = load_settings()
    end = settings.history_end if end is None else end
    frames: list[pd.DataFrame] = []
    for year in range(start, end + 1):
        path = AWARDS_DIR / f"coy_{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0)
        df["year"] = year
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No COY caches in {AWARDS_DIR.relative_to(ROOT)}; "
            "run: python -m src.scrape coy"
        )
    full = pd.concat(frames, ignore_index=True)
    full = full.rename(columns={"rank": "coy_rank"})
    coaches = pd.read_csv(COACHES_PATH)[["id", "name"]]
    full = full.merge(coaches, on="name", how="inner")
    out = DERIVED_DIR / "coy.csv"
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(out)
    print(f"Wrote {out.relative_to(ROOT)} ({len(full)} rows, {start}–{end})")
    return full


# --- Team pages (GM / owner) ---------------------------------------------------


def team_htm_path(abbrev: str, year: int) -> Path:
    return TEAMS_DIR / f"{abbrev.lower()}{year}.htm"


def get_team_page(
    abbrev: str,
    year: int,
    *,
    fetch: bool = False,
    sleep: float = 3.0,
    cookie: str | None = None,
    html_path: str | Path | None = None,
) -> Path | None:
    """Fetch/cache one team-year HTML page. Returns None if PFR has no page.

    Pass ``html_path`` to copy a locally saved team page into the cache
    (Cloudflare workaround).
    """
    path = team_htm_path(abbrev, year)
    if html_path is not None:
        html = load_html(html_path=html_path)
        TEAMS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        print(f"Cached team page {abbrev} {year} from {html_path}")
        return path
    if not fetch and path.exists():
        return path
    url = (
        f"https://www.pro-football-reference.com/teams/"
        f"{abbrev.lower()}/{year}.htm"
    )
    try:
        html = get_html(url, sleep=sleep, cookie=cookie)
    except RuntimeError as exc:
        # Franchise may not exist that year (404) or Cloudflare block.
        if "HTTP 404" in str(exc):
            return None
        raise
    TEAMS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def scrape_teams(
    *,
    start: int | None = None,
    end: int | None = None,
    year: int | None = None,
    fetch: bool = False,
    sleep: float = 3.0,
    cookie: str | None = None,
) -> None:
    """Scrape team-year HTML pages used for GM/owner extraction."""
    settings = load_settings()
    if year is not None:
        years = [year]
    elif start is None and end is None:
        # Season rollover default: only the newly completed year.
        years = [settings.history_end]
    else:
        years = _resolve_years(
            start=HISTORY_START_YEAR if start is None else start,
            end=end,
            year=None,
        )
    abbrevs = load_teams_table()["Abbrev"].tolist()
    for y in years:
        for abbrev in tqdm(abbrevs, desc=f"teams {y}"):
            get_team_page(
                abbrev, y, fetch=fetch, sleep=sleep, cookie=cookie
            )


def get_role(
    abbrev: str,
    year: int,
    roles: list[str] | None = None,
) -> str:
    """Parse a cached team page for the first matching role name.

    Returns ``\"null\"`` if the HTML cache is missing (matches historical
    derived gm/owner tables).
    """
    roles = roles or GM_ROLE_NAMES
    path = team_htm_path(abbrev, year)
    if not path.exists():
        return "null"
    text = path.read_text(encoding="utf-8", errors="replace")
    names = [
        tag.a.text
        for tag in bs(text, "lxml").find_all("p")
        if tag.a is not None and any(role in tag.text for role in roles)
    ]
    return names[0] if names else ""


def rebuild_roles_derived(
    role: str = "gm",
    *,
    names: list[str] | None = None,
    start: int = HISTORY_START_YEAR,
    end: int | None = None,
) -> pd.DataFrame:
    """Build data/derived/{gm,owner}.csv from cached team HTML pages."""
    settings = load_settings()
    end = settings.history_end if end is None else end
    if names is None:
        names = GM_ROLE_NAMES if role == "gm" else OWNER_ROLE_NAMES
    teams = load_teams_table()
    year_frames: list[pd.DataFrame] = []
    for year in range(start, end + 1):
        column = teams["Abbrev"].map(
            lambda abbrev, y=year: get_role(abbrev, y, roles=names)
        )
        year_frames.append(pd.DataFrame({year: column}))
    combined = pd.concat(year_frames, axis=1)
    final = pd.concat([teams[["Abbrev", "Team"]], combined], axis=1).set_index(
        "Abbrev"
    )
    out = DERIVED_DIR / f"{role}.csv"
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    final.to_csv(out)
    print(f"Wrote {out.relative_to(ROOT)} ({start}–{end})")
    return final


def rebuild_gm_owner_derived(
    *,
    start: int = HISTORY_START_YEAR,
    end: int | None = None,
) -> None:
    rebuild_roles_derived("gm", names=GM_ROLE_NAMES, start=start, end=end)
    rebuild_roles_derived("owner", names=OWNER_ROLE_NAMES, start=start, end=end)


# --- CLI -----------------------------------------------------------------------


def cmd_index(args: argparse.Namespace) -> None:
    upsert_coaches_index(
        sleep=args.sleep,
        html_path=getattr(args, "html", None),
        cookie=getattr(args, "cookie", None),
    )


def cmd_coaches(args: argparse.Namespace) -> None:
    settings = load_settings()
    html_path = getattr(args, "html", None)
    if html_path is not None:
        if not args.id or len(args.id) != 1:
            raise SystemExit("--html requires exactly one --id")
        get_coach(
            args.id[0],
            html_path=html_path,
            sleep=args.sleep,
            cookie=args.cookie,
        )
        return
    if args.id:
        ids = list(args.id)
    elif args.all:
        coaches = pd.read_csv(COACHES_PATH)
        ids = coaches["id"].tolist()
    else:
        ids = active_coach_ids(settings)
        print(
            f"Refreshing {len(ids)} active coaches "
            f"(to == {settings.history_end})"
        )
    scrape_coaches(ids, fetch=args.fetch, sleep=args.sleep)


def cmd_odds(args: argparse.Namespace) -> None:
    scrape_odds(
        start=args.start,
        end=args.end,
        year=args.year,
        fetch=args.fetch,
        sleep=args.sleep,
    )


def cmd_standings(args: argparse.Namespace) -> None:
    scrape_standings(
        start=args.start,
        end=args.end,
        year=args.year,
        fetch=args.fetch,
        sleep=args.sleep,
        cookie=args.cookie,
        html_path=getattr(args, "html", None),
    )
    if not args.no_derive:
        rebuild_standings_derived()


def cmd_coy(args: argparse.Namespace) -> None:
    scrape_coy(
        start=args.start,
        end=args.end,
        year=args.year,
        fetch=args.fetch,
        sleep=args.sleep,
        cookie=args.cookie,
        html_path=getattr(args, "html", None),
    )
    if not args.no_derive:
        rebuild_coy_derived()


def cmd_teams(args: argparse.Namespace) -> None:
    scrape_teams(
        start=args.start,
        end=args.end,
        year=args.year,
        fetch=args.fetch,
        sleep=args.sleep,
        cookie=args.cookie,
    )
    if not args.no_derive:
        rebuild_gm_owner_derived()


def cmd_ingest_coaches(args: argparse.Namespace) -> None:
    ingest_raw_coaches(raw_dir=args.dir, year=args.year)


def cmd_all(args: argparse.Namespace) -> None:
    settings = load_settings()
    print(f"Season {settings.season} | history through {settings.history_end}")
    upsert_coaches_index(
        sleep=args.sleep,
        html_path=getattr(args, "html", None),
        cookie=getattr(args, "cookie", None),
    )
    ids = active_coach_ids(settings)
    print(f"Refreshing {len(ids)} active coach season tables")
    scrape_coaches(ids, fetch=args.fetch, sleep=args.sleep)
    scrape_odds(fetch=args.fetch, sleep=args.sleep)
    scrape_standings(
        fetch=args.fetch, sleep=args.sleep, cookie=args.cookie
    )
    scrape_coy(fetch=args.fetch, sleep=args.sleep, cookie=args.cookie)
    print(f"Refreshing team pages for {settings.history_end}")
    scrape_teams(
        year=settings.history_end,
        fetch=args.fetch,
        sleep=args.sleep,
        cookie=args.cookie,
    )
    rebuild_standings_derived()
    rebuild_coy_derived()
    rebuild_gm_owner_derived()
    print("Done.")


def _add_html_arg(parser: argparse.ArgumentParser, *, help_text: str) -> None:
    parser.add_argument(
        "--html",
        default=None,
        help=help_text,
    )


def _add_year_range_args(
    parser: argparse.ArgumentParser,
    *,
    start_default: int,
) -> None:
    parser.add_argument("--year", type=int, help="Single year only")
    parser.add_argument("--start", type=int, default=start_default)
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Inclusive (default: history_end)",
    )
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument(
        "--no-derive",
        action="store_true",
        help="Skip rebuilding data/derived pivots after scrape",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.scrape",
        description="Scrape NFL coach / odds / standings / COY / team data",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=3.0,
        help="Seconds between HTTP requests (default: 3)",
    )
    parser.add_argument(
        "--cookie",
        default=None,
        help="Browser Cookie header for PFR (or set PFR_COOKIE). Helps when Cloudflare blocks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Upsert data/scraped/coaches.csv from PFR")
    _add_html_arg(
        p_index,
        help_text="Parse a locally saved PFR coaches index page instead of fetching",
    )
    p_index.set_defaults(func=cmd_index)

    p_coaches = sub.add_parser(
        "coaches",
        help="Scrape per-coach season CSVs (default: active = to == history_end)",
    )
    p_coaches.add_argument(
        "--id",
        action="append",
        dest="id",
        help="Coach id (repeatable). Default: active coaches",
    )
    p_coaches.add_argument(
        "--all",
        action="store_true",
        help="Scrape every coach in coaches.csv (slow)",
    )
    p_coaches.add_argument(
        "--fetch",
        action="store_true",
        help="Re-download even if cached CSV exists",
    )
    _add_html_arg(
        p_coaches,
        help_text="Parse a locally saved coach page (requires exactly one --id)",
    )
    p_coaches.set_defaults(func=cmd_coaches)

    p_odds = sub.add_parser("odds", help="Scrape Vegas O/U tables into data/scraped/odds/")
    p_odds.add_argument("--year", type=int, help="Single year only")
    p_odds.add_argument("--start", type=int, default=ODDS_START_YEAR)
    p_odds.add_argument(
        "--end", type=int, default=None, help="Inclusive (default: history_end)"
    )
    p_odds.add_argument("--fetch", action="store_true")
    p_odds.set_defaults(func=cmd_odds)

    p_standings = sub.add_parser(
        "standings",
        help="Scrape year standings + rebuild data/derived/standings.csv",
    )
    _add_year_range_args(p_standings, start_default=HISTORY_START_YEAR)
    _add_html_arg(
        p_standings,
        help_text=(
            "Parse a locally saved PFR year page instead of fetching "
            "(use with --year, e.g. years_2025.htm)"
        ),
    )
    p_standings.set_defaults(func=cmd_standings)

    p_coy = sub.add_parser(
        "coy",
        help="Scrape AP COY voting + rebuild data/derived/coy.csv",
    )
    _add_year_range_args(p_coy, start_default=HISTORY_START_YEAR)
    _add_html_arg(
        p_coy,
        help_text=(
            "Parse a locally saved PFR awards page instead of fetching "
            "(use with --year, e.g. awards_2025.htm)"
        ),
    )
    p_coy.set_defaults(func=cmd_coy)

    p_ingest = sub.add_parser(
        "ingest-coaches",
        help=(
            "Normalize data/scraped/coaches_raw/{year}/*.csv into "
            "data/scraped/coaches/{id}.csv"
        ),
    )
    p_ingest.add_argument(
        "--year",
        type=int,
        default=None,
        help="Raw subfolder year (default: history_end)",
    )
    p_ingest.add_argument(
        "--dir",
        default=None,
        help="Override raw directory (default: data/scraped/coaches_raw/{year})",
    )
    p_ingest.set_defaults(func=cmd_ingest_coaches)

    p_teams = sub.add_parser(
        "teams",
        help="Scrape team-year HTML + rebuild data/derived/gm.csv and owner.csv",
    )
    p_teams.add_argument(
        "--year",
        type=int,
        help="Single year (default for this command when start/end omitted: history_end)",
    )
    p_teams.add_argument(
        "--start",
        type=int,
        default=None,
        help="Start year for backfill (use with --end)",
    )
    p_teams.add_argument(
        "--end",
        type=int,
        default=None,
        help="Inclusive end year",
    )
    p_teams.add_argument("--fetch", action="store_true")
    p_teams.add_argument(
        "--no-derive",
        action="store_true",
        help="Skip rebuilding gm/owner derived tables",
    )
    p_teams.set_defaults(func=cmd_teams)

    p_all = sub.add_parser(
        "all",
        help="index + coaches + odds + standings + coy + teams(history_end) + derive",
    )
    p_all.add_argument("--fetch", action="store_true")
    p_all.add_argument(
        "--html",
        default=None,
        help="Optional saved coaches index HTML for the index step",
    )
    p_all.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Ensure imports work when run as a script from any cwd.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
