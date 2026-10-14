"""Build `data/export.csv` for the active prediction season.

Scores historical coach-seasons with LightGBM (GroupKFold OOF from `src.fit`),
rolls non-fired coaches from `history_end` into `season`, synthesizes new-hire
rows from `config/<season>/hires.yaml` + futures, and writes the display table.

``--pace`` re-scores every active-season coach from the weekly win projection
(market season total, or a results blend when the board is missing a team).
Exit code 2 means the season has no completed games and the export was not written.

Usage (from repo root):

    python -m src.score
    python -m src.score --skip-train   # reuse training.csv + lightgbm artifacts
    python -m src.score --skip-train --pace
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .examples import (
    build_and_write as build_examples,
    build_incumbent_flats,
    build_new_hire_flats,
    load_base_panel,
)
from .predict import FLAT_PATH, OOF_PATH, load_artifact, predict_proba
from .predict import DEFAULT_MODEL_PATH as MODEL_PATH
from .reference import canonical_team, load_abbrev_aliases
from .scrape import COACHES_PATH, DERIVED_DIR, load_teams_table
from .season import ROOT, load_settings
from .training import TRAINING_PATH, build_and_write, convert_team

EXPORT_PATH = ROOT / "data" / "export.csv"
FUTURES_PATH = DERIVED_DIR / "futures.csv"
COLORS_PATH = ROOT / "config" / "team_colors.csv"
COACH_SEASONS_PATH = DERIVED_DIR / "coach_seasons.csv"
GM_PATH = DERIVED_DIR / "gm.csv"

LABEL_COLS = ["fired", "year", "tm", "id"]
# Publish column order: predicted prob immediately after fired.
EXPORT_COLUMNS = [
    "fired",
    "prob",
    "year",
    "tm",
    "id",
    "age",
    "round",
    "win_pct",
    "w_plyf",
    "exp",
    "tenure",
    "tenure_over_500",
    "tenure_w_plyf",
    "tenure_coy_share",
    "exp_coy_share",
    "srs",
    "ou",
    "gm",
    "owner",
    "coy_share",
    "coy_rank",
    "poc",
    "delta_1yr_win_pct",
    "delta_2yr_win_pct",
    "delta_3yr_win_pct",
    "delta_1yr_plyf",
    "delta_2yr_plyf",
    "delta_3yr_plyf",
    "pred",
    "name",
    "team",
    "win_pct_proj",
    "color1",
    "color2",
    "wins",
    "losses",
    "l_plyf",
    "ou_line",
]
DELTA_STATS = [
    "win_pct",
    "delta_1yr_plyf",
    "delta_2yr_plyf",
    "delta_3yr_plyf",
    "delta_1yr_win_pct",
    "delta_2yr_win_pct",
    "delta_3yr_win_pct",
]


def load_hires(season: int | None = None) -> tuple[dict[str, str], dict[str, int]]:
    settings = load_settings()
    season = settings.season if season is None else season
    path = ROOT / "config" / str(season) / "hires.yaml"
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    teams = {str(k): str(v) for k, v in (data.get("teams") or {}).items()}
    ages = {str(k): int(v) for k, v in (data.get("ages") or {}).items()}
    return teams, ages


def sb_odds(plus: float) -> float:
    """Map American SB moneyline to an approximate playoff-round expectation."""
    return round(25 * 100 / (100 + float(plus)), 2)


def build_futures(*, season: int | None = None, games: int | None = None) -> pd.DataFrame:
    """Rank-align sb_futures + wins_exp (sorted by odds) → derived/futures.csv."""
    settings = load_settings()
    season = settings.season if season is None else season
    games = settings.games if games is None else games
    season_dir = ROOT / "config" / str(season)

    teams = load_teams_table()
    abbrev_by_team = teams.set_index("Team")["Abbrev"].to_dict()

    sb = pd.read_csv(season_dir / "sb_futures.csv", index_col=0)
    sb = sb.sort_values(by=["Odds"]).reset_index(drop=True).reset_index(names=["Rank"])
    sb["Abbrev"] = sb["Team"].map(abbrev_by_team)
    if sb["Abbrev"].isna().any():
        bad = sb.loc[sb["Abbrev"].isna(), "Team"].tolist()
        raise ValueError(f"Unmapped futures teams: {bad}")
    sb["Round"] = sb["Odds"].map(sb_odds)

    wins = pd.read_csv(season_dir / "wins_exp.csv", index_col=0).reset_index(names=["Rank"])
    futures = sb.merge(wins, on="Rank")
    futures["win_pct"] = futures["wins_exp"].map(lambda x: round(float(x) / games, 2))
    out = futures[["Abbrev", "win_pct", "wins_exp", "Round"]].set_index("Abbrev")
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(FUTURES_PATH)
    print(f"Wrote {FUTURES_PATH.relative_to(ROOT)} ({len(out)} teams)")
    return out


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={"Tenure (W-L)": "tenure_over_500"})
    out.columns = [
        str(c).lower().replace(" ", "_").replace("w-l%", "win_pct") for c in out.columns
    ]
    return out


def _add_gm_column(training: pd.DataFrame) -> pd.DataFrame:
    """Attach gm (hire-era match) for export schema; forward-fill null year cols."""
    if "gm" in training.columns:
        return training
    if not GM_PATH.exists():
        out = training.copy()
        out["gm"] = 0
        return out

    gm = pd.read_csv(GM_PATH, index_col=0)
    # Forward-fill empty trailing season columns from the prior year.
    year_cols = [c for c in gm.columns if str(c).isdigit()]
    gm_years = gm[year_cols].copy()
    gm_years = gm_years.replace({None: np.nan, "null": np.nan, "": np.nan})
    gm_years = gm_years.ffill(axis=1)

    aliases = load_abbrev_aliases()
    teams = load_teams_table()
    valid = set(teams["Abbrev"].astype(str))

    def lookup(row: pd.Series) -> int:
        team = convert_team(
            str(row["Tm"]), year=int(row["Year"]), aliases=aliases, valid=valid
        )
        if team == "N/A" or team not in gm_years.index:
            return 0
        year = str(int(row["Year"]))
        # Y1 within this coach-team stint approximated by first year in training
        stint = training[
            (training["id"] == row["id"]) & (training["Tm"] == row["Tm"])
        ]
        y1 = str(int(stint["Year"].min())) if len(stint) else year
        curr = gm_years.at[team, year] if year in gm_years.columns else pd.NA
        first = gm_years.at[team, y1] if y1 in gm_years.columns else pd.NA
        if curr == first and not pd.isna(first):
            return 1
        if pd.isna(curr) and pd.isna(first):
            return 0
        return -1

    out = training.copy()
    out["gm"] = out.apply(lookup, axis=1)
    # Keep column order close to notebook with_race (gm before owner).
    cols = [c for c in out.columns if c != "gm"]
    if "owner" in cols:
        i = cols.index("owner")
        cols = cols[:i] + ["gm"] + cols[i:]
    else:
        cols.append("gm")
    return out[cols]


def attach_oof_probs(
    display: pd.DataFrame,
    oof_path: Path = OOF_PATH,
) -> pd.DataFrame:
    """Merge LightGBM OOF probabilities onto display rows by coach id + year."""
    oof = pd.read_csv(oof_path)
    merged = display.merge(
        oof[["id", "year", "proba"]],
        on=["id", "year"],
        how="left",
    )
    if merged["proba"].isna().any():
        missing = int(merged["proba"].isna().sum())
        raise ValueError(
            f"{missing} display rows missing OOF scores; rebuild examples/model "
            f"with `python -m src.score` (no --skip-train)."
        )
    merged["prob"] = merged.pop("proba")
    merged["pred"] = (merged["prob"] >= 0.5).astype(int)
    return merged


def new_coach_row(
    coach_id: str,
    team: str,
    *,
    season: int,
    ages: dict[str, int],
    dataset: pd.DataFrame,
    futures: pd.DataFrame,
    proj_win_pct: float | None = None,
) -> pd.Series:
    """Synthesize a tenure-1 prediction-season feature row (notebook new_coach)."""
    custom: dict[str, object] = {
        "exp": 1,
        "tenure": 1,
        "year": season,
        "owner": 1,
        "gm": 1,
        "coy_rank": 0,
        "coy_share": 0,
        "fired": 0,
    }
    hist = dataset[dataset["id"] == coach_id].sort_values("year")
    if len(hist):
        old = hist.iloc[-1]
        custom = {
            "exp": int(old["exp"]) + 1,
            "tenure": 1,
            "exp_coy_share": old.get("exp_coy_share", 0),
            "poc": old.get("poc", 0),
            "year": season,
            "owner": 1,
            "gm": 1,
            "coy_rank": 0,
            "coy_share": 0,
            "fired": 0,
        }

    row = {
        col: (0 if pd.api.types.is_numeric_dtype(dataset[col]) else "")
        for col in dataset.columns
    }
    row.update(custom)
    row["id"] = coach_id
    row["tm"] = team
    row["year"] = season
    if coach_id not in ages and not len(hist):
        raise KeyError(f"Missing age for new coach {coach_id} in hires.yaml")
    row["age"] = ages.get(coach_id, int(hist.iloc[-1]["age"]) + 1 if len(hist) else 0)
    team_key = _canonical_abbrev(
        team,
        aliases=load_abbrev_aliases(),
        valid=set(load_teams_table()["Abbrev"].astype(str)),
    )
    if team_key not in futures.index:
        raise KeyError(f"Team {team} missing from futures")
    row["round"] = float(futures.loc[team_key, "Round"])
    row["w_plyf"] = round(float(row["round"]) * (13 / 30), 2)
    row["win_pct"] = (
        float(futures.loc[team_key, "win_pct"])
        if proj_win_pct is None
        else float(proj_win_pct)
    )
    return pd.Series(row)


def build_new_coaches(
    *,
    season: int,
    teams: dict[str, str],
    ages: dict[str, int],
    dataset: pd.DataFrame,
    futures: pd.DataFrame,
    artifact: dict,
    panel: pd.DataFrame,
    k: int,
    proj_win_pct: dict[str, float] | None = None,
) -> pd.DataFrame:
    display_rows = [
        new_coach_row(
            coach_id,
            team,
            season=season,
            ages=ages,
            dataset=dataset,
            futures=futures,
            proj_win_pct=None if proj_win_pct is None else proj_win_pct[team],
        )
        for team, coach_id in teams.items()
    ]
    out = pd.DataFrame(display_rows)
    flats = build_new_hire_flats(
        season=season,
        teams=teams,
        ages=ages,
        futures=futures,
        panel=panel,
        k=k,
        proj_win_pct=proj_win_pct,
    )
    out["prob"] = predict_proba(flats, artifact).values
    out["pred"] = (out["prob"] >= 0.5).astype(int)
    return out


def _names_dict() -> dict[str, str]:
    coaches = pd.read_csv(COACHES_PATH)
    return dict(zip(coaches["id"].astype(str), coaches["name"].astype(str)))


def _team_key(abbrev: str, *, aliases: dict[str, str], valid: set[str]) -> str:
    return convert_team(abbrev, aliases=aliases, valid=valid)


def _canonical_abbrev(
    abbrev: str,
    *,
    aliases: dict[str, str],
    valid: set[str],
) -> str:
    key = canonical_team(str(abbrev), aliases=aliases, valid=valid)
    if key is None:
        raise KeyError(f"Unmapped team {abbrev}")
    return str(key)


def _project_season_display(
    rows: pd.DataFrame,
    futures: pd.DataFrame,
    proj_win_pct: dict[str, float],
    *,
    aliases: dict[str, str],
    valid: set[str],
) -> pd.DataFrame:
    """Point the active-season display row at the weekly projection.

    Playoff round stays on the preseason futures mapping. COY stays 0.
    ``proj_win_pct`` is keyed by canonical abbrev. Row ``tm`` may be the
    coach-page code (for example LAC) and is left unchanged.
    """
    out = rows.copy()
    for col in ("win_pct", "round", "w_plyf", "coy_share"):
        out[col] = out[col].astype(float)
    for idx, row in out.iterrows():
        team = _canonical_abbrev(str(row["tm"]), aliases=aliases, valid=valid)
        if team not in proj_win_pct:
            raise KeyError(f"{row['tm']} ({team}) missing from pace projection")
        if team not in futures.index:
            raise KeyError(f"Team {team} missing from futures")
        rnd = float(futures.at[team, "Round"])
        out.at[idx, "win_pct"] = float(proj_win_pct[team])
        out.at[idx, "round"] = rnd
        out.at[idx, "w_plyf"] = round(rnd * (13 / 30), 2)
        out.at[idx, "coy_share"] = 0.0
        out.at[idx, "coy_rank"] = 0
    return out


def assemble_export(
    historical: pd.DataFrame,
    incumbents: pd.DataFrame,
    new_coaches: pd.DataFrame,
    *,
    season: int,
    history_end: int,
    futures: pd.DataFrame,
    records: pd.DataFrame | None = None,
) -> pd.DataFrame:
    aliases = load_abbrev_aliases()
    teams_tbl = load_teams_table()
    valid = set(teams_tbl["Abbrev"].astype(str))
    names = _names_dict()

    output = pd.concat(
        [historical, incumbents, new_coaches], ignore_index=True
    ).reset_index(drop=True)
    output = output.sort_values(
        by=["year", "prob"], ascending=[False, False]
    ).reset_index(drop=True)
    output["name"] = output["id"].map(names)
    output["team"] = output["tm"].map(
        lambda a: _team_key(str(a), aliases=aliases, valid=valid)
    )
    output["win_pct_proj"] = output["win_pct"]

    # Display: tenure-1 rows inherit predecessor win_pct / deltas from history_end.
    fired_prev = historical[
        (historical["year"] == history_end) & (historical["fired"] == 1)
    ][["tm", *DELTA_STATS]].copy()
    fired_prev["team"] = fired_prev["tm"].map(
        lambda a: _team_key(str(a), aliases=aliases, valid=valid)
    )
    fired_map = {
        row["team"]: {stat: row[stat] for stat in DELTA_STATS}
        for _, row in fired_prev.iterrows()
    }
    new_mask = output["tenure"] == 1
    for stat in DELTA_STATS:
        output.loc[new_mask, stat] = output.loc[new_mask, "team"].map(
            lambda t, s=stat: fired_map.get(t, {}).get(s, 0)
        )

    colors = pd.read_csv(COLORS_PATH, index_col=0)[
        ["team", "primary", "secondary"]
    ].rename(columns={"primary": "color1", "secondary": "color2"})
    output = output.merge(colors, on="team", how="left")

    seasons = pd.read_csv(COACH_SEASONS_PATH, index_col=0)
    counting = seasons[["id", "Year", "W", "L", "L plyf"]].rename(
        columns={"Year": "year", "W": "wins", "L": "losses", "L plyf": "l_plyf"}
    )
    for col in ["wins", "losses", "l_plyf"]:
        counting[col] = counting[col].fillna(0).astype(int)
    merged = output.merge(counting, how="left", on=["id", "year"])
    for col in ["wins", "losses", "l_plyf"]:
        merged[col] = merged[col].fillna(0).astype(int)
    if records is not None:
        season_mask = merged["year"] == season
        for idx in merged.index[season_mask]:
            team = str(merged.at[idx, "team"])
            if team not in records.index:
                raise KeyError(f"No season-to-date record for {team}")
            merged.at[idx, "wins"] = int(records.at[team, "wins"])
            merged.at[idx, "losses"] = int(records.at[team, "losses"])

    def ou_line(row: pd.Series) -> float:
        if int(row["year"]) < season:
            return float(row["wins"]) - float(row["ou"])
        team = row["team"]
        if team not in futures.index:
            return 0.0
        return float(futures.loc[team, "wins_exp"])

    merged["ou_line"] = merged.apply(ou_line, axis=1)
    export = merged.sort_values(by=["year"], ascending=False).reset_index(drop=True)
    export.index = export.index + 1
    missing = [c for c in EXPORT_COLUMNS if c not in export.columns]
    if missing:
        raise KeyError(f"Export missing columns: {missing}")
    extra = [c for c in export.columns if c not in EXPORT_COLUMNS]
    return export[EXPORT_COLUMNS + extra]


def _keep_published_history(export: pd.DataFrame, *, season: int) -> pd.DataFrame:
    """Swap rebuilt past seasons for the rows already in ``data/export.csv``."""
    if not EXPORT_PATH.exists():
        print("pace: no existing export.csv; writing rebuilt history")
        return export
    prior = pd.read_csv(EXPORT_PATH, index_col=0)
    hist = prior[prior["year"] < season].copy()
    current = export[export["year"] == season].copy()
    for col in current.columns:
        if col not in hist.columns:
            hist[col] = pd.NA
    combined = pd.concat([current, hist[current.columns]], ignore_index=True)
    combined = combined.sort_values(by=["year"], ascending=False).reset_index(drop=True)
    combined.index = combined.index + 1
    print(
        f"pace: kept {len(hist)} published historical rows; "
        f"replaced {len(current)} rows @ {season}"
    )
    return combined


def build_export(
    *,
    skip_train: bool = False,
    pace: bool = False,
    out_path: Path | None = None,
) -> pd.DataFrame | None:
    settings = load_settings()
    season = settings.season
    history_end = settings.history_end

    futures = build_futures(season=season, games=settings.games)
    pace_table: pd.DataFrame | None = None
    proj_win_pct: dict[str, float] | None = None
    if pace:
        from .pace import PaceError, load_pace
        from .reference import load_abbrev_aliases

        teams_tbl = load_teams_table()
        try:
            loaded = load_pace(
                season=season,
                games=settings.games,
                preseason_wins=futures["wins_exp"],
                aliases=load_abbrev_aliases(),
                valid=set(teams_tbl["Abbrev"].astype(str)),
            )
        except PaceError as exc:
            print(f"pace: {exc}", file=sys.stderr)
            raise
        if loaded.completed_games == 0:
            print(
                f"pace: no completed regular-season games for {season}; "
                "not writing export"
            )
            return None
        pace_table = loaded.table
        proj_win_pct = {
            str(team): float(pace_table.at[team, "proj_win_pct"])
            for team in pace_table.index
        }

    if skip_train and all(
        p.exists()
        for p in (TRAINING_PATH, FLAT_PATH, MODEL_PATH, OOF_PATH)
    ):
        print(
            f"Loaded artifacts: {TRAINING_PATH.name}, {FLAT_PATH.name}, "
            f"{MODEL_PATH.name}, {OOF_PATH.name}"
        )
    else:
        build_examples()
        from .fit import train as train_lgb

        train_lgb()

    if skip_train and TRAINING_PATH.exists():
        training_raw = pd.read_csv(TRAINING_PATH, index_col=0)
        print(f"Loaded {TRAINING_PATH.relative_to(ROOT)} ({len(training_raw)} rows)")
    else:
        training_raw = build_and_write(skip_derived=True)

    training_raw = _add_gm_column(training_raw)
    data = clean_column_names(training_raw)
    historical = attach_oof_probs(data)
    artifact = load_artifact(MODEL_PATH)
    panel = load_base_panel()

    # Incumbents: non-fired history_end rows → prediction season.
    # Without --pace they keep the prior season's OOF probability.
    incumbents = historical[
        (historical["year"] == history_end) & (historical["fired"] != 1)
    ].copy()
    incumbents = incumbents.assign(year=season)
    incumbents["exp"] = incumbents["exp"] + 1
    incumbents["tenure"] = incumbents["tenure"] + 1
    incumbents["fired"] = 0
    k = int(artifact.get("k", 5))
    if proj_win_pct is not None:
        aliases = load_abbrev_aliases()
        valid = set(load_teams_table()["Abbrev"].astype(str))
        incumbents = _project_season_display(
            incumbents,
            futures,
            proj_win_pct,
            aliases=aliases,
            valid=valid,
        )
        incumbent_flats = build_incumbent_flats(
            incumbents,
            season=season,
            history_end=history_end,
            futures=futures,
            panel=panel,
            proj_win_pct=proj_win_pct,
            k=k,
        )
        incumbents["prob"] = predict_proba(incumbent_flats, artifact).values
        incumbents["pred"] = (incumbents["prob"] >= 0.5).astype(int)

    teams, ages = load_hires(season)
    new_coaches = build_new_coaches(
        season=season,
        teams=teams,
        ages=ages,
        dataset=data,
        futures=futures,
        artifact=artifact,
        panel=panel,
        k=k,
        proj_win_pct=proj_win_pct,
    )

    export = assemble_export(
        historical,
        incumbents,
        new_coaches,
        season=season,
        history_end=history_end,
        futures=futures,
        records=None if pace_table is None else pace_table[["wins", "losses"]],
    )
    if pace:
        # Keep the published historical snapshot. Re-assembling from OOF would
        # rewrite every past season on a weekly upsert.
        export = _keep_published_history(export, season=season)

    out_path = out_path or EXPORT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(out_path)
    n_season = int((export["year"] == season).sum())
    n_new = int(((export["year"] == season) & (export["tenure"] == 1)).sum())
    try:
        shown = out_path.resolve().relative_to(ROOT)
    except ValueError:
        shown = out_path
    print(
        f"Wrote {shown}: {len(export)} rows "
        f"({n_season} @ {season}, {n_new} new hires)"
    )
    return export


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=EXPORT_PATH,
        help="Output CSV path (default data/export.csv)",
    )
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Reuse training.csv and LightGBM artifacts instead of rebuilding",
    )
    p.add_argument(
        "--pace",
        action="store_true",
        help=(
            "Re-score the active season from the weekly win projection. "
            "Exits 2 without writing when no regular-season games are complete."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    export = build_export(
        skip_train=args.skip_train,
        pace=args.pace,
        out_path=args.out,
    )
    if export is None:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
