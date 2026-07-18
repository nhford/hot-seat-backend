"""Build display-feature training table (`model/training.csv`).

Used by `src.score` for export columns (prob comes from LightGBM separately).
Omits the `gm` feature at build time (added in score). Rebuilds coach seasons,
odds, and playoff pivots through `history_end` before writing the table.

Usage (from repo root):

    python -m src.training
    python -m src.training --skip-derived   # reuse existing odds/playoffs pivots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .examples import check_fired, load_firings, load_poc, season_length
from .reference import (
    canonical_team,
    franchise_exists,
    load_abbrev_aliases,
    load_franchise_map,
    load_retired,
    load_team_aliases,
    normalize_team_typo,
)
from .scrape import (
    COACHES_PATH,
    DERIVED_DIR,
    HISTORY_START_YEAR,
    ODDS_START_YEAR,
    clean_team_name,
    get_coach,
    get_odds_table,
    load_teams_table,
)
from .season import ROOT, load_settings

TRAINING_PATH = ROOT / "model" / "training.csv"
COACH_SEASONS_PATH = DERIVED_DIR / "coach_seasons.csv"
ODDS_PATH = DERIVED_DIR / "odds.csv"
PLAYOFFS_PATH = DERIVED_DIR / "playoffs.csv"
STANDINGS_PATH = DERIVED_DIR / "standings.csv"
OWNER_PATH = DERIVED_DIR / "owner.csv"
COY_PATH = DERIVED_DIR / "coy.csv"
PLAYOFFS_LONG_PATH = ROOT / "data" / "scraped" / "playoffs.csv"

TRAINING_COLUMNS = [
    "Fired",
    "id",
    "Year",
    "Tm",
    "Age",
    "Round",
    "W-L%",
    "W plyf",
    "Exp",
    "Tenure",
    "Tenure (W-L)",
    "Tenure W plyf",
    "Tenure coy_share",
    "Exp coy_share",
    "SRS",
    "ou",
    "owner",
    "coy_share",
    "coy_rank",
    "poc",
    "Delta 1yr W-L%",
    "Delta 2yr W-L%",
    "Delta 3yr W-L%",
    "Delta 1yr plyf",
    "Delta 2yr plyf",
    "Delta 3yr plyf",
]


def convert_team(
    abbrev: str,
    *,
    year: int | None = None,
    aliases: dict[str, str],
    valid: set[str],
    franchise_map: dict | None = None,
) -> str:
    key = canonical_team(
        abbrev,
        year,
        aliases=aliases,
        valid=valid,
        franchise_map=franchise_map,
    )
    return key if key is not None else "N/A"


def clean_rows(coach_id: str) -> pd.DataFrame:
    df = get_coach(coach_id)
    rows = df[df["Year"].astype(str).str.isdigit()].copy()
    if "Num" not in rows.columns:
        rows["Num"] = 0
    if "Won" not in rows.columns:
        rows["Won"] = 0
    rows = rows.fillna(0)
    rows["Age"] = rows["Age"].astype(int)
    rows["Year"] = rows["Year"].astype(int)
    rows["Notes"] = rows["Notes"].astype(str)
    return rows


def rebuild_coach_seasons(*, coach_ids: list[str] | None = None) -> pd.DataFrame:
    if coach_ids is None:
        coach_ids = pd.read_csv(COACHES_PATH)["id"].astype(str).tolist()
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    for coach_id in coach_ids:
        path = ROOT / "data" / "scraped" / "coaches" / f"{coach_id}.csv"
        if not path.exists():
            skipped.append(coach_id)
            continue
        frames.append(clean_rows(coach_id).assign(id=coach_id))
    if not frames:
        raise RuntimeError("No coach season caches found under data/scraped/coaches/")
    agg = pd.concat(frames, ignore_index=True)
    agg = agg[(agg["Lg"] == "NFL") & (agg["Year"] >= HISTORY_START_YEAR)].reset_index(
        drop=True
    )
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(COACH_SEASONS_PATH)
    if skipped:
        print(f"Skipped {len(skipped)} coaches without cache (e.g. {skipped[0]})")
    print(
        f"Wrote {COACH_SEASONS_PATH.relative_to(ROOT)} "
        f"({len(agg)} rows, Year max {int(agg['Year'].max())})"
    )
    return agg


def rebuild_odds_derived(*, end: int | None = None) -> pd.DataFrame:
    settings = load_settings()
    end = settings.season if end is None else end
    teams = load_teams_table()
    abbrev_by_team = teams.set_index("Team")["Abbrev"].to_dict()
    aliases = load_team_aliases()

    merged = pd.DataFrame()
    for year in range(ODDS_START_YEAR, end):
        df = get_odds_table(year).copy()
        df["Abbrev"] = (
            df["Team"].map(lambda t: clean_team_name(str(t), aliases)).map(abbrev_by_team)
        )
        if df["Abbrev"].isna().any():
            bad = df.loc[df["Abbrev"].isna(), "Team"].tolist()
            raise ValueError(f"Unmapped odds teams for {year}: {bad}")
        df["Diff"] = df["Wins"] - df["OU"]
        year_df = df[["Abbrev", "Diff"]].rename(columns={"Diff": str(year)})
        merged = year_df if merged.empty else merged.merge(
            year_df, how="outer", on="Abbrev"
        )
    out = merged.fillna(0).set_index("Abbrev")
    out.to_csv(ODDS_PATH)
    print(
        f"Wrote {ODDS_PATH.relative_to(ROOT)} "
        f"(years {ODDS_START_YEAR}-{end - 1})"
    )
    return out


def rebuild_playoffs_derived(*, end: int | None = None) -> pd.DataFrame:
    settings = load_settings()
    end = settings.season if end is None else end
    fmap = load_franchise_map()
    playoffs = pd.read_csv(PLAYOFFS_LONG_PATH)[["Year", "Team", "Tm", "Round"]].copy()
    playoffs["Tm"] = playoffs["Tm"].map(
        lambda a: normalize_team_typo(str(a), fmap["typos"])
    )
    playoffs = playoffs.drop_duplicates(["Tm", "Year"]).set_index(["Tm", "Year"])
    base = load_teams_table()

    def get_round(abbrev: str, year: int) -> int:
        if not franchise_exists(abbrev, year, franchise_map=fmap):
            return 0
        try:
            val = playoffs.loc[(abbrev, year), "Round"]
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            return int(val)
        except KeyError:
            return 0

    year_frames = [
        pd.DataFrame({year: base["Abbrev"].map(lambda a, y=year: get_round(a, y))})
        for year in range(HISTORY_START_YEAR, end)
    ]
    combined = pd.concat(year_frames, axis=1)
    final = pd.concat([base[["Abbrev", "Team"]], combined], axis=1).set_index("Abbrev")
    final.to_csv(PLAYOFFS_PATH)
    print(
        f"Wrote {PLAYOFFS_PATH.relative_to(ROOT)} "
        f"(years {HISTORY_START_YEAR}-{end - 1})"
    )
    return final


def _label_interim(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def flag(row: pd.Series) -> bool:
        notes = str(row["Notes"]).lower() if pd.notna(row["Notes"]) else ""
        flagged = ("starting" in notes) or ("interim" in notes)
        short = (row["G"] < (season_length(int(row["Year"])) - 2)) and (
            "fired" not in notes
        )
        return bool(flagged or short)

    out["interim"] = out.apply(flag, axis=1)
    return out


def _add_tenure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True).reset_index(names=["Exp"])
    df["Exp"] = df["Exp"] + 1
    df["Tenure"] = df.groupby("Tm", group_keys=False)["Year"].apply(
        lambda x: (x.diff() == 1).cumsum() + 1
    )
    df["(W-L)"] = df["W"] - df["L"]
    df["Tenure (W-L)"] = df.groupby("Tm")["W"].cumsum() - df.groupby("Tm")["L"].cumsum()
    df["Tenure W plyf"] = df.groupby("Tm")["W plyf"].cumsum()
    return df


def _compare_role(
    role_table: pd.DataFrame,
    team: str,
    year: int,
    year_one: int,
    *,
    aliases: dict[str, str],
    valid: set[str],
    franchise_map: dict | None = None,
) -> int:
    team = convert_team(
        team, year=year, aliases=aliases, valid=valid, franchise_map=franchise_map
    )
    if team == "N/A":
        return 0
    curr = role_table.at[team, str(year)] if str(year) in role_table.columns else pd.NA
    y1 = (
        role_table.at[team, str(year_one)]
        if str(year_one) in role_table.columns
        else pd.NA
    )
    if curr == y1 and not pd.isna(y1):
        return 1
    if pd.isna(curr) and pd.isna(y1):
        return 0
    return -1


def _add_owner(
    df: pd.DataFrame,
    owner_table: pd.DataFrame,
    *,
    aliases: dict[str, str],
    valid: set[str],
    franchise_map: dict | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out["Y1"] = out.groupby("Tm")["Year"].transform("min")
    out["owner"] = out.apply(
        lambda row: _compare_role(
            owner_table,
            row["Tm"],
            int(row["Year"]),
            int(row["Y1"]),
            aliases=aliases,
            valid=valid,
            franchise_map=franchise_map,
        ),
        axis=1,
    )
    return out.drop(columns=["Y1"])


def _delta(
    team: str,
    year: int,
    curr: float,
    delta: int,
    table: pd.DataFrame,
    *,
    aliases: dict[str, str],
    valid: set[str],
    franchise_map: dict | None = None,
) -> float:
    team = convert_team(
        team, year=year, aliases=aliases, valid=valid, franchise_map=franchise_map
    )
    if (year + delta) < HISTORY_START_YEAR:
        return curr
    if team == "N/A" or team not in table.index:
        return curr
    prev_year = year + delta
    if not franchise_exists(team, prev_year, franchise_map=franchise_map):
        return curr
    col = str(prev_year)
    if col not in table.columns:
        return curr
    prev = table.at[team, col]
    if pd.isna(prev):
        return curr
    return round(float(curr) - float(prev), 3)


def build_training() -> pd.DataFrame:
    """Assemble the notebook `with_race` table without the `gm` column."""
    settings = load_settings()
    aliases = load_abbrev_aliases()
    franchise_map = load_franchise_map()
    retired = load_retired()
    fired = load_firings(settings.season)
    poc = load_poc()
    teams = load_teams_table()
    valid = set(teams["Abbrev"].astype(str))

    seasons = pd.read_csv(COACH_SEASONS_PATH, index_col=0)
    seasons = seasons[(seasons["Lg"] == "NFL") & (seasons["Year"] >= HISTORY_START_YEAR)]
    seasons["Tm"] = seasons["Tm"].map(
        lambda a: normalize_team_typo(str(a), franchise_map["typos"])
    )

    playoffs_long = pd.read_csv(PLAYOFFS_LONG_PATH)[["Tm", "Year", "Round"]].copy()
    playoffs_long["Tm"] = playoffs_long["Tm"].map(
        lambda a: normalize_team_typo(str(a), franchise_map["typos"])
    )
    playoffs_long = playoffs_long.rename(columns={"Tm": "Team_temp"})

    parts: list[pd.DataFrame] = []
    for coach_id, g in seasons.groupby("id", sort=False):
        g = g.copy()
        g["Fired"] = 0
        last_years = g.groupby("Tm")["Year"].idxmax()
        for idx in last_years:
            g.loc[idx, "Fired"] = check_fired(
                int(g.loc[idx, "Year"]),
                str(coach_id),
                retired=retired,
                history_end=settings.history_end,
                fired=fired,
            )
        parts.append(g)
    df = pd.concat(parts, ignore_index=True)
    df["Team_temp"] = [
        convert_team(
            str(tm),
            year=int(year),
            aliases=aliases,
            valid=valid,
            franchise_map=franchise_map,
        )
        for tm, year in zip(df["Tm"], df["Year"])
    ]
    df = df.merge(playoffs_long, how="left", on=["Team_temp", "Year"])
    df = df.drop(columns=["Team_temp"])
    df["Round"] = df["Round"].fillna(0).astype(int)

    df = pd.concat(
        [_add_tenure(g) for _, g in df.groupby("id", sort=False)],
        ignore_index=True,
    )
    df = _label_interim(df)
    df = df[~df["interim"]].reset_index(drop=True)

    owner_table = pd.read_csv(OWNER_PATH, index_col=0)
    df = pd.concat(
        [
            _add_owner(
                g,
                owner_table,
                aliases=aliases,
                valid=valid,
                franchise_map=franchise_map,
            )
            for _, g in df.groupby("id", sort=False)
        ],
        ignore_index=True,
    )

    coy = pd.read_csv(COY_PATH, index_col=0)
    coy = coy[["id", "year", "coy_share", "coy_rank"]].rename(columns={"year": "Year"})
    df = df.merge(coy, on=["Year", "id"], how="left")
    df["coy_share"] = df["coy_share"].fillna(0).astype(float)
    df["coy_rank"] = df["coy_rank"].fillna(0).astype(int)
    df["Tenure coy_share"] = (
        df.groupby(["id", "Tm"], sort=False)["coy_share"].cumsum().round(3)
    )
    df["Exp coy_share"] = df.groupby("id", sort=False)["coy_share"].cumsum().round(3)

    odds = pd.read_csv(ODDS_PATH, index_col=0)

    def lookup_ou(row: pd.Series) -> float:
        if int(row["Year"]) < ODDS_START_YEAR:
            return 0.0
        team = convert_team(
            str(row["Tm"]),
            year=int(row["Year"]),
            aliases=aliases,
            valid=valid,
            franchise_map=franchise_map,
        )
        if team == "N/A" or team not in odds.index:
            return 0.0
        col = str(int(row["Year"]))
        if col not in odds.columns:
            return 0.0
        val = odds.at[team, col]
        return 0.0 if pd.isna(val) else float(val)

    df["ou"] = df.apply(lookup_ou, axis=1)

    standings = pd.read_csv(STANDINGS_PATH, index_col=0)
    playoffs = pd.read_csv(PLAYOFFS_PATH, index_col=0)
    for frame in (standings, playoffs):
        if "Team" in frame.columns:
            frame.drop(columns=["Team"], inplace=True)

    for lag in range(1, 4):
        df[f"Delta {lag}yr W-L%"] = df.apply(
            lambda row, d=lag: _delta(
                str(row["Tm"]),
                int(row["Year"]),
                float(row["W-L%"]),
                -d,
                standings,
                aliases=aliases,
                valid=valid,
                franchise_map=franchise_map,
            ),
            axis=1,
        )
        df[f"Delta {lag}yr plyf"] = df.apply(
            lambda row, d=lag: _delta(
                str(row["Tm"]),
                int(row["Year"]),
                float(row["Round"]),
                -d,
                playoffs,
                aliases=aliases,
                valid=valid,
                franchise_map=franchise_map,
            ),
            axis=1,
        )

    df["poc"] = df["id"].map(lambda x: int(x in poc))
    out = df[TRAINING_COLUMNS].copy()
    return out.reset_index(drop=True)


def build_and_write(
    *,
    out_path: Path | None = None,
    skip_derived: bool = False,
) -> pd.DataFrame:
    settings = load_settings()
    rebuild_coach_seasons()
    if not skip_derived:
        rebuild_odds_derived(end=settings.season)
        rebuild_playoffs_derived(end=settings.season)

    training = build_training()
    out_path = out_path or TRAINING_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(out_path)
    n_2025 = int((training["Year"] == settings.history_end).sum())
    n_fired_end = int(
        ((training["Year"] == settings.history_end) & (training["Fired"] == 1)).sum()
    )
    print(
        f"Wrote {out_path.relative_to(ROOT)}: {len(training)} rows "
        f"(Year max {int(training['Year'].max())}, "
        f"{n_2025} @ {settings.history_end}, "
        f"{n_fired_end} fired labels @ {settings.history_end})"
    )
    assert "gm" not in training.columns
    return training


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=TRAINING_PATH,
        help="Output CSV path (default model/training.csv)",
    )
    p.add_argument(
        "--skip-derived",
        action="store_true",
        help="Skip rebuilding odds/playoffs derived pivots",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_and_write(out_path=args.out, skip_derived=args.skip_derived)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
