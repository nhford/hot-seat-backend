"""Build v1 coach-firing examples: current-stop sequences + fixed context.

Primary artifact is a last-k flattened table for LightGBM
(`model/examples_flat.csv`). Optional JSONL keeps the variable-length
sequence form for a future sequence model.

Usage (from repo root):

    python -m src.examples
    python -m src.examples --k 12 --jsonl --out-flat model/examples_flat.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .reference import (
    canonical_team,
    franchise_exists,
    load_abbrev_aliases,
    load_franchise_map,
    load_retired,
    normalize_team_typo,
)
from .season import ROOT, load_settings

DEFAULT_K = 5
SEASON_TOKEN_FIELDS = (
    "age",
    "playoff_round",
    "win_pct",
    "w_plyf",
    "coy_share",
    "coy_rank",
    "tenure_idx",
)
CONTEXT_FIELDS = (
    "exp",
    "tenure",
    "poc",
    "prior_hc_stops",
    "prior_hc_seasons",
    "prior_win_pct",
    "prior_w_plyf",
    "prior_sb_wins",
    "prior_sb_apps",
    "career_sb_wins",
    "career_sb_apps",
    "prior_coy_share",
    "years_since_last_hc",
    "prehire_win_pct_1",
    "prehire_round_1",
    "prehire_win_pct_2",
    "prehire_round_2",
    "prehire_win_pct_3",
    "prehire_round_3",
)


def load_poc(path: Path | None = None) -> set[str]:
    path = path or (ROOT / "config" / "poc.yaml")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return set(data.get("poc") or [])


def load_firings(season: int | None = None) -> set[str]:
    settings = load_settings()
    season = season if season is not None else settings.season
    path = ROOT / "config" / str(season) / "firings.yaml"
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return set(data.get("fired") or [])


def convert_team(
    abbrev: str,
    aliases: dict[str, str],
    valid: set[str],
    year: int | None = None,
    franchise_map: dict | None = None,
) -> str | None:
    """Year-aware when possible; otherwise static abbrev aliases."""
    return canonical_team(
        abbrev,
        year,
        aliases=aliases,
        valid=valid,
        franchise_map=franchise_map,
    )


def season_length(year: int) -> int:
    if year == 1982:
        return 9
    if year == 1987:
        return 15
    if year > 2020:
        return 17
    if year < 1978:
        return 14
    return 16


def is_interim(row: pd.Series) -> bool:
    notes = row["Notes"] if pd.notna(row.get("Notes")) else ""
    notes_l = str(notes).lower()
    flagged = ("starting" in notes_l) or ("interim" in notes_l)
    short = (row["G"] < (season_length(int(row["Year"])) - 2)) and (
        "fired" not in notes_l
    )
    return bool(flagged or short)


def check_fired(
    year: int,
    coach_id: str,
    *,
    retired: set[str],
    history_end: int,
    fired: set[str],
) -> int:
    label = f"{coach_id}-{str(year)[2:]}"
    if label in retired:
        return 0
    if int(year) == int(history_end) and label not in fired:
        return 0
    return 1


def _win_pct(w: float, l: float, fallback: float | None = None) -> float:
    denom = w + l
    if denom > 0:
        return float(w / denom)
    if fallback is not None and pd.notna(fallback):
        return float(fallback)
    return float("nan")


def _team_year_lookup(
    table: pd.DataFrame,
    abbrev: str | None,
    year: int,
    *,
    franchise_map: dict | None = None,
) -> float:
    """Read team×year value from standings/playoffs pivot; NaN if missing."""
    if abbrev is None or abbrev not in table.index:
        return float("nan")
    if not franchise_exists(abbrev, year, franchise_map=franchise_map):
        return float("nan")
    col = str(year)
    if col not in table.columns:
        return float("nan")
    val = table.at[abbrev, col]
    if pd.isna(val):
        return float("nan")
    return float(val)


def load_base_panel() -> pd.DataFrame:
    """Non-interim NFL coach-seasons with stop ids, playoff round, COY, labels."""
    settings = load_settings()
    aliases = load_abbrev_aliases()
    franchise_map = load_franchise_map()
    retired = load_retired()
    fired = load_firings(settings.season)
    poc = load_poc()

    teams = pd.read_csv(ROOT / "config" / "teams.csv")
    valid = set(teams["Abbrev"].astype(str))

    seasons = pd.read_csv(ROOT / "data" / "derived" / "coach_seasons.csv", index_col=0)
    seasons = seasons[(seasons["Lg"] == "NFL") & (seasons["Year"] >= 1970)].copy()
    seasons["Tm"] = seasons["Tm"].map(
        lambda a: normalize_team_typo(str(a), franchise_map["typos"])
    )
    seasons = seasons[~seasons.apply(is_interim, axis=1)].reset_index(drop=True)

    playoffs_long = pd.read_csv(ROOT / "data" / "scraped" / "playoffs.csv")
    playoffs_long["Tm"] = playoffs_long["Tm"].map(
        lambda a: normalize_team_typo(str(a), franchise_map["typos"])
    )
    playoffs_long = playoffs_long[["Tm", "Year", "Round"]].drop_duplicates(
        ["Tm", "Year"]
    )

    seasons["tm_key"] = [
        convert_team(str(tm), aliases, valid, year=int(year), franchise_map=franchise_map)
        for tm, year in zip(seasons["Tm"], seasons["Year"])
    ]
    seasons = seasons.merge(
        playoffs_long.rename(columns={"Tm": "tm_key", "Round": "playoff_round"}),
        on=["tm_key", "Year"],
        how="left",
    )
    seasons["playoff_round"] = seasons["playoff_round"].fillna(0).astype(int)

    coy = pd.read_csv(ROOT / "data" / "derived" / "coy.csv", index_col=0)
    coy = coy.rename(columns={"year": "Year"})[["Year", "id", "coy_share", "coy_rank"]]
    seasons = seasons.merge(coy, on=["Year", "id"], how="left")
    seasons["coy_share"] = seasons["coy_share"].fillna(0.0).astype(float)
    # Rank is ordinal 1=best … n=worst among vote-getters. No votes → NaN
    # (not 0), so 0 doesn't sit next to the winner in numeric space.
    seasons["coy_rank"] = seasons["coy_rank"].astype("float64")

    seasons["win_pct"] = [
        _win_pct(w, l, fb)
        for w, l, fb in zip(seasons["W"], seasons["L"], seasons["W-L%"])
    ]
    seasons["w_plyf"] = seasons["W plyf"].fillna(0.0).astype(float)
    seasons["age"] = seasons["Age"].astype(float)
    seasons["poc"] = seasons["id"].map(lambda x: int(x in poc))

    standings = pd.read_csv(
        ROOT / "data" / "derived" / "standings.csv", index_col=0
    )
    playoffs_pivot = pd.read_csv(
        ROOT / "data" / "derived" / "playoffs.csv", index_col=0
    )
    # Drop non-year cols if present as first columns named Team
    for frame in (standings, playoffs_pivot):
        if "Team" in frame.columns:
            frame.drop(columns=["Team"], inplace=True)

    parts: list[pd.DataFrame] = []
    for coach_id, g in seasons.groupby("id", sort=False):
        g = g.sort_values("Year").reset_index(drop=True)
        new_stop = (g["Tm"] != g["Tm"].shift()) | (
            g["Year"] != g["Year"].shift() + 1
        )
        g = g.copy()
        g["stop_id"] = new_stop.cumsum()
        g["exp"] = np.arange(1, len(g) + 1)
        g["tenure"] = g.groupby("stop_id").cumcount() + 1
        g["Fired"] = 0
        last_idx = g.groupby("stop_id")["Year"].idxmax()
        for idx in last_idx:
            g.loc[idx, "Fired"] = check_fired(
                int(g.loc[idx, "Year"]),
                str(coach_id),
                retired=retired,
                history_end=settings.history_end,
                fired=fired,
            )
        parts.append(g)

    panel = pd.concat(parts, ignore_index=True)
    panel.attrs["standings"] = standings
    panel.attrs["playoffs_pivot"] = playoffs_pivot
    panel.attrs["aliases"] = aliases
    panel.attrs["valid"] = valid
    panel.attrs["franchise_map"] = franchise_map
    return panel


def _prior_context(prior: pd.DataFrame) -> dict[str, float]:
    if prior.empty:
        return {
            "prior_hc_stops": 0,
            "prior_hc_seasons": 0,
            "prior_win_pct": float("nan"),
            "prior_w_plyf": 0.0,
            "prior_sb_wins": 0,
            "prior_sb_apps": 0,
            "prior_coy_share": 0.0,
        }
    w = float(prior["W"].sum())
    l = float(prior["L"].sum())
    return {
        "prior_hc_stops": int(prior["stop_id"].nunique()),
        "prior_hc_seasons": int(len(prior)),
        "prior_win_pct": _win_pct(w, l),
        "prior_w_plyf": float(prior["w_plyf"].sum()),
        "prior_sb_wins": int((prior["playoff_round"] == 5).sum()),
        "prior_sb_apps": int((prior["playoff_round"] >= 4).sum()),
        "prior_coy_share": float(prior["coy_share"].sum()),
    }


def _career_sb_context(career_through: pd.DataFrame) -> dict[str, int]:
    """SB wins/apps through the season being scored (all stops)."""
    if career_through.empty:
        return {"career_sb_wins": 0, "career_sb_apps": 0}
    return {
        "career_sb_wins": int((career_through["playoff_round"] == 5).sum()),
        "career_sb_apps": int((career_through["playoff_round"] >= 4).sum()),
    }


def _prehire_context(
    tm_key: str | None,
    hire_year: int,
    standings: pd.DataFrame,
    playoffs_pivot: pd.DataFrame,
    *,
    franchise_map: dict | None = None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for lag in (1, 2, 3):
        y = hire_year - lag
        out[f"prehire_win_pct_{lag}"] = _team_year_lookup(
            standings, tm_key, y, franchise_map=franchise_map
        )
        out[f"prehire_round_{lag}"] = _team_year_lookup(
            playoffs_pivot, tm_key, y, franchise_map=franchise_map
        )
    return out


def feature_columns(flat: pd.DataFrame) -> list[str]:
    """Numeric/categorical model inputs (drop label + keys)."""
    skip = {"fired", "id", "year", "tm"}
    return [c for c in flat.columns if c not in skip]


def build_examples(panel: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    """One example per coach-team-season with sequence + context + Fired."""
    if panel is None:
        panel = load_base_panel()
    standings = panel.attrs["standings"]
    playoffs_pivot = panel.attrs["playoffs_pivot"]
    franchise_map = panel.attrs.get("franchise_map") or load_franchise_map()

    examples: list[dict[str, Any]] = []
    for (coach_id, stop_id), stop in panel.groupby(["id", "stop_id"], sort=False):
        stop = stop.sort_values("Year").reset_index(drop=True)
        hire_year = int(stop.iloc[0]["Year"])
        tm = str(stop.iloc[0]["Tm"])
        tm_key = stop.iloc[0]["tm_key"]
        if pd.isna(tm_key):
            tm_key = None
        else:
            tm_key = str(tm_key)

        prior = panel[
            (panel["id"] == coach_id) & (panel["stop_id"] < stop_id)
        ]
        prior_ctx = _prior_context(prior)
        prehire = _prehire_context(
            tm_key,
            hire_year,
            standings,
            playoffs_pivot,
            franchise_map=franchise_map,
        )

        if prior.empty:
            years_since = float("nan")
        else:
            prior_end = int(prior["Year"].max())
            years_since = float(hire_year - prior_end - 1)

        coach_career = panel[panel["id"] == coach_id]

        for t in range(len(stop)):
            row = stop.iloc[t]
            seq_rows = stop.iloc[: t + 1]
            sequence = [
                {
                    "age": float(r["age"]),
                    "playoff_round": int(r["playoff_round"]),
                    "win_pct": float(r["win_pct"]),
                    "w_plyf": float(r["w_plyf"]),
                    "coy_share": float(r["coy_share"]),
                    "coy_rank": (
                        float(r["coy_rank"]) if pd.notna(r["coy_rank"]) else None
                    ),
                    "tenure_idx": int(r["tenure"]),
                }
                for _, r in seq_rows.iterrows()
            ]
            career_through = coach_career[coach_career["Year"] <= int(row["Year"])]
            context = {
                "exp": int(row["exp"]),
                "tenure": int(row["tenure"]),
                "poc": int(row["poc"]),
                **prior_ctx,
                **_career_sb_context(career_through),
                "years_since_last_hc": years_since,
                **prehire,
            }
            examples.append(
                {
                    "id": str(coach_id),
                    "year": int(row["Year"]),
                    "tm": tm,
                    "fired": int(row["Fired"]),
                    "sequence": sequence,
                    "context": context,
                }
            )
    return examples


def flatten_examples(
    examples: list[dict[str, Any]], *, k: int = DEFAULT_K
) -> pd.DataFrame:
    """Pad/truncate each sequence to last-k seasons for LightGBM."""
    rows: list[dict[str, Any]] = []
    for ex in examples:
        seq = ex["sequence"]
        # Most recent season is t0; older seasons t1..t{k-1}
        recent = list(reversed(seq[-k:]))
        flat: dict[str, Any] = {
            "fired": ex["fired"],
            "id": ex["id"],
            "year": ex["year"],
            "tm": ex["tm"],
        }
        flat.update(ex["context"])
        for lag in range(k):
            suffix = f"_t{lag}"
            if lag < len(recent):
                tok = recent[lag]
                for field in SEASON_TOKEN_FIELDS:
                    flat[f"{field}{suffix}"] = tok[field]
            else:
                for field in SEASON_TOKEN_FIELDS:
                    flat[f"{field}{suffix}"] = np.nan
        rows.append(flat)
    return pd.DataFrame(rows)


def categorical_feature_names(k: int = DEFAULT_K) -> list[str]:
    names = [f"playoff_round_t{lag}" for lag in range(k)]
    names += [f"prehire_round_{lag}" for lag in (1, 2, 3)]
    return names


def _flatten_sequence(
    context: dict[str, Any],
    sequence: list[dict[str, Any]],
    *,
    coach_id: str,
    year: int,
    team: str,
    k: int = DEFAULT_K,
    fired: int = 0,
) -> dict[str, Any]:
    recent = list(reversed(sequence[-k:]))
    flat: dict[str, Any] = {
        "fired": fired,
        "id": coach_id,
        "year": year,
        "tm": team,
    }
    flat.update(context)
    for lag in range(k):
        suffix = f"_t{lag}"
        if lag < len(recent):
            tok = recent[lag]
            for field in SEASON_TOKEN_FIELDS:
                flat[f"{field}{suffix}"] = tok[field]
        else:
            for field in SEASON_TOKEN_FIELDS:
                flat[f"{field}{suffix}"] = np.nan
    return flat


def new_hire_flat_row(
    coach_id: str,
    team: str,
    *,
    season: int,
    ages: dict[str, int],
    futures: pd.DataFrame,
    panel: pd.DataFrame,
    k: int = DEFAULT_K,
) -> dict[str, Any]:
    """Build one flattened LightGBM row for a prediction-season new hire."""
    standings = panel.attrs["standings"]
    playoffs_pivot = panel.attrs["playoffs_pivot"]
    franchise_map = panel.attrs.get("franchise_map") or load_franchise_map()
    aliases = panel.attrs.get("aliases") or load_abbrev_aliases()
    valid = panel.attrs.get("valid") or set(
        pd.read_csv(ROOT / "config" / "teams.csv")["Abbrev"].astype(str)
    )

    coach_hist = panel[panel["id"] == coach_id].sort_values("Year")
    tm_key = convert_team(
        team, aliases, valid, year=season, franchise_map=franchise_map
    )
    prehire = _prehire_context(
        tm_key,
        season,
        standings,
        playoffs_pivot,
        franchise_map=franchise_map,
    )

    if coach_hist.empty:
        if coach_id not in ages:
            raise KeyError(f"Missing age for new coach {coach_id} in hires.yaml")
        age = ages[coach_id]
        exp = 1
        poc = int(coach_id in load_poc())
        prior_ctx = _prior_context(pd.DataFrame())
        years_since = float("nan")
        career_through = pd.DataFrame()
    else:
        last = coach_hist.iloc[-1]
        age = ages.get(coach_id, int(last["age"]) + 1)
        exp = int(last["exp"]) + 1
        poc = int(last["poc"])
        prior_ctx = _prior_context(coach_hist)
        years_since = float(season - int(coach_hist["Year"].max()) - 1)
        career_through = coach_hist

    if team not in futures.index:
        raise KeyError(f"Team {team} missing from futures")
    round_val = float(futures.loc[team, "Round"])
    t0 = {
        "age": float(age),
        "playoff_round": int(round(round_val)),
        "win_pct": float(futures.loc[team, "win_pct"]),
        "w_plyf": round(round_val * (13 / 30), 2),
        "coy_share": 0.0,
        "coy_rank": None,
        "tenure_idx": 1,
    }
    context = {
        "exp": exp,
        "tenure": 1,
        "poc": poc,
        **prior_ctx,
        **_career_sb_context(career_through),
        "years_since_last_hc": years_since,
        **prehire,
    }
    return _flatten_sequence(
        context,
        [t0],
        coach_id=coach_id,
        year=season,
        team=team,
        k=k,
        fired=0,
    )


def build_new_hire_flats(
    *,
    season: int,
    teams: dict[str, str],
    ages: dict[str, int],
    futures: pd.DataFrame,
    panel: pd.DataFrame | None = None,
    k: int = DEFAULT_K,
) -> pd.DataFrame:
    if panel is None:
        panel = load_base_panel()
    rows = [
        new_hire_flat_row(
            coach_id,
            team,
            season=season,
            ages=ages,
            futures=futures,
            panel=panel,
            k=k,
        )
        for team, coach_id in teams.items()
    ]
    return pd.DataFrame(rows)


def write_jsonl(examples: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def build_and_write(
    *,
    k: int = DEFAULT_K,
    out_flat: Path | None = None,
    out_jsonl: Path | None = None,
) -> pd.DataFrame:
    out_flat = out_flat or (ROOT / "model" / "examples_flat.csv")
    examples = build_examples()
    flat = flatten_examples(examples, k=k)
    out_flat.parent.mkdir(parents=True, exist_ok=True)
    flat.to_csv(out_flat, index=False)

    if out_jsonl is not None:
        write_jsonl(examples, out_jsonl)

    meta = {
        "n_examples": len(flat),
        "n_fired": int(flat["fired"].sum()),
        "k": k,
        "season_token_fields": list(SEASON_TOKEN_FIELDS),
        "context_fields": list(CONTEXT_FIELDS),
        "categorical_features": categorical_feature_names(k),
        "flat_path": str(out_flat),
        "jsonl_path": str(out_jsonl) if out_jsonl else None,
    }
    meta_path = out_flat.with_name("examples_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return flat


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Last-k seasons to keep when flattening (default {DEFAULT_K})",
    )
    p.add_argument(
        "--out-flat",
        type=Path,
        default=ROOT / "model" / "examples_flat.csv",
        help="LightGBM-ready CSV path",
    )
    p.add_argument(
        "--out-jsonl",
        type=Path,
        default=ROOT / "model" / "examples.jsonl",
        help="Variable-length sequence JSONL path (only written with --jsonl)",
    )
    p.add_argument(
        "--jsonl",
        action="store_true",
        help="Also write variable-length sequence JSONL",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out_jsonl = args.out_jsonl if args.jsonl else None
    flat = build_and_write(k=args.k, out_flat=args.out_flat, out_jsonl=out_jsonl)
    n = len(flat)
    n_pos = int(flat["fired"].sum())
    print(
        f"Wrote {n} examples ({n_pos} fired) → {args.out_flat}"
        + (f" and {out_jsonl}" if out_jsonl else "")
    )
    # Sanity: tenure-1 should have real prehire when team history exists
    t1 = flat[flat["tenure"] == 1]
    prehire_ok = t1["prehire_win_pct_1"].notna().mean()
    print(
        f"tenure=1 rows: {len(t1)}; "
        f"prehire_win_pct_1 non-null: {prehire_ok:.1%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
