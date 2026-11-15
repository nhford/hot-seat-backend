"""Score the What If grid and write `data/what_if.csv`.

One row per prediction-season coach × record (wins 0–17) × playoff round (0–5).
Panel context comes from `incumbent_flat_row` / `new_hire_flat_row`. The
hypothetical season matches the What If select, not the batch scorer's
`round * 13/30` playoff-wins heuristic:

- ``win_pct = wins / 17``
- ``playoff_round`` is the dropdown value 0–5
- ``w_plyf = max(round - 1, 0)``
- Coach of the Year is empty
- career Super Bowl wins and appearances increase when the scenario reaches
  the Super Bowl

Incumbents are rolled forward from ``history_end``. Tenure-1 hires are scored
in place. ``python -m src.score`` calls this after a successful export so a
retrain cannot drift from the tab.

Usage (from repo root):

    python -m src.whatif
    python -m src.whatif --out data/what_if.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .examples import (
    DEFAULT_K,
    incumbent_flat_row,
    load_base_panel,
    new_hire_flat_row,
)
from .predict import DEFAULT_MODEL_PATH, load_artifact, predict_proba
from .season import ROOT, load_settings

WHAT_IF_PATH = ROOT / "data" / "what_if.csv"
# `scenario.ts` `Games` / `Records`: the select is wins out of 17.
WHAT_IF_GAMES = 17
N_RECORDS = WHAT_IF_GAMES + 1  # wins 0–17
N_ROUNDS = 6  # playoff round 0–5


def apply_scenario(
    base: dict[str, Any],
    wins: int,
    playoff_round: int,
) -> dict[str, Any]:
    """Overwrite the projected season token. Career Super Bowls start from base."""
    row = dict(base)
    row["win_pct_t0"] = wins / WHAT_IF_GAMES
    row["playoff_round_t0"] = int(playoff_round)
    # Same recipe as buildWhatIfInputs. Not round * 13/30.
    row["w_plyf_t0"] = float(max(playoff_round - 1, 0))
    row["coy_share_t0"] = 0.0
    row["coy_rank_t0"] = None
    row["career_sb_wins"] = int(base["career_sb_wins"]) + (1 if playoff_round == 5 else 0)
    row["career_sb_apps"] = int(base["career_sb_apps"]) + (1 if playoff_round >= 4 else 0)
    return row


def _futures_frame() -> pd.DataFrame:
    """Read the futures table the flat-row builders use to resolve teams.

    The projected win% and playoff round on that table are discarded; each
    grid cell replaces the hypothetical season token.
    """
    from .score import FUTURES_PATH, build_futures

    if FUTURES_PATH.exists():
        futures = pd.read_csv(FUTURES_PATH, index_col=0)
        if len(futures) and "Round" in futures.columns:
            return futures
    return build_futures()


def _require_features(flat: dict[str, Any], feature_names: list[str], coach_id: str) -> None:
    missing = [name for name in feature_names if name not in flat]
    if missing:
        raise ValueError(f"{coach_id} flat row missing features: {missing}")


def _base_rows(
    panel: pd.DataFrame,
    futures: pd.DataFrame,
    *,
    feature_names: list[str],
    k: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """Incumbents rolled forward, then tenure-1 hires scored in place."""
    from .score import load_hires

    settings = load_settings()
    season = settings.season
    history_end = settings.history_end
    teams, ages = load_hires(season)

    end = panel[panel["Year"] == history_end]
    incumbents = (
        end[end["Fired"] != 1]
        .sort_values(["id", "stop_id", "tenure"])
        .drop_duplicates("id", keep="last")
    )

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, rec in incumbents.iterrows():
        coach_id = str(rec["id"])
        flat = incumbent_flat_row(
            coach_id,
            str(rec["Tm"]),
            season=season,
            history_end=history_end,
            futures=futures,
            panel=panel,
            proj_win_pct=0.0,
            k=k,
        )
        _require_features(flat, feature_names, coach_id)
        rows.append(flat)
        seen.add(coach_id)

    n_incumbent = len(rows)
    for team, coach_id in teams.items():
        if coach_id in seen:
            raise ValueError(f"{coach_id} is both an incumbent and a new hire")
        flat = new_hire_flat_row(
            coach_id,
            team,
            season=season,
            ages=ages,
            futures=futures,
            panel=panel,
            k=k,
            proj_win_pct=0.0,
        )
        _require_features(flat, feature_names, coach_id)
        rows.append(flat)
        seen.add(coach_id)

    if not rows:
        raise ValueError(
            f"No prediction-season coaches for {season} "
            f"(history_end={history_end})"
        )
    return rows, n_incumbent, len(rows) - n_incumbent


def _check_recipe(base: dict[str, Any]) -> None:
    """The hypothetical token must match buildWhatIfInputs, not the 13/30 heuristic."""
    season = apply_scenario(base, 8, 5)
    if season["win_pct_t0"] != 8 / WHAT_IF_GAMES:
        raise ValueError("win_pct_t0 is not wins/17")
    if season["playoff_round_t0"] != 5 or season["w_plyf_t0"] != 4:
        raise ValueError("Super Bowl token is not round 5 with w_plyf 4")
    if season["coy_share_t0"] != 0.0 or season["coy_rank_t0"] is not None:
        raise ValueError("Coach of the Year token is not empty")
    if season["career_sb_wins"] != int(base["career_sb_wins"]) + 1:
        raise ValueError("Super Bowl win was not added to career_sb_wins")
    if season["career_sb_apps"] != int(base["career_sb_apps"]) + 1:
        raise ValueError("Super Bowl appearance was not added to career_sb_apps")
    # Losing the wild-card round is still zero playoff wins in the UI recipe.
    if apply_scenario(base, 10, 1)["w_plyf_t0"] != 0:
        raise ValueError("wild-card loss should score w_plyf 0")
    title = apply_scenario(base, 12, 4)
    if title["w_plyf_t0"] != 3:
        raise ValueError("conference-title loss should score w_plyf 3")
    if title["career_sb_wins"] != int(base["career_sb_wins"]):
        raise ValueError("a Super Bowl loss must not add a career win")
    if title["career_sb_apps"] != int(base["career_sb_apps"]) + 1:
        raise ValueError("a Super Bowl loss must add a career appearance")


def _validate(grid: pd.DataFrame, *, n_coaches: int) -> None:
    expected = n_coaches * N_RECORDS * N_ROUNDS
    if len(grid) != expected:
        raise ValueError(f"expected {expected} what-if rows, got {len(grid)}")
    if list(grid.columns) != ["id", "record", "round", "prob"]:
        raise ValueError(f"unexpected columns: {list(grid.columns)}")
    if grid.duplicated(["id", "record", "round"]).any():
        raise ValueError("duplicate (id, record, round) rows")
    if set(grid["record"].unique()) != set(range(N_RECORDS)):
        raise ValueError("record values are not wins 0–17")
    if set(grid["round"].unique()) != set(range(N_ROUNDS)):
        raise ValueError("round values are not 0–5")
    counts = grid.groupby("id").size()
    if not (counts == N_RECORDS * N_ROUNDS).all():
        raise ValueError("a coach is missing record × round cells")
    probs = grid["prob"]
    if probs.isna().any() or not ((probs >= 0) & (probs <= 1)).all():
        raise ValueError("prob outside [0, 1]")


def build_what_if(
    *,
    model_path: Path | None = None,
) -> tuple[pd.DataFrame, int, int]:
    """Return the grid plus incumbent and new-hire counts."""
    artifact = load_artifact(model_path or DEFAULT_MODEL_PATH)
    feature_names = list(artifact["feature_names"])
    k = int(artifact.get("k", DEFAULT_K))
    panel = load_base_panel()
    futures = _futures_frame()
    bases, n_incumbent, n_new = _base_rows(
        panel,
        futures,
        feature_names=feature_names,
        k=k,
    )
    _check_recipe(bases[0])

    feature_rows: list[dict[str, Any]] = []
    keys: list[tuple[str, int, int]] = []
    for base in bases:
        coach_id = str(base["id"])
        for wins in range(N_RECORDS):
            for playoff_round in range(N_ROUNDS):
                feature_rows.append(apply_scenario(base, wins, playoff_round))
                keys.append((coach_id, wins, playoff_round))

    scored = pd.DataFrame(feature_rows)
    probs = predict_proba(scored, artifact)
    grid = pd.DataFrame(
        {
            "id": [coach_id for coach_id, _, _ in keys],
            "record": [wins for _, wins, _ in keys],
            "round": [playoff_round for _, _, playoff_round in keys],
            "prob": probs.to_numpy(),
        }
    )
    grid = grid.sort_values(["id", "record", "round"]).reset_index(drop=True)
    _validate(grid, n_coaches=len(bases))
    return grid, n_incumbent, n_new


def write_what_if(
    path: Path | None = None,
    *,
    model_path: Path | None = None,
) -> pd.DataFrame:
    path = path or WHAT_IF_PATH
    grid, n_incumbent, n_new = build_what_if(model_path=model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(path, index=False)
    try:
        shown = path.resolve().relative_to(ROOT)
    except ValueError:
        shown = path
    n_coaches = n_incumbent + n_new
    print(
        f"Wrote {shown}: {len(grid)} rows "
        f"({n_incumbent} incumbents, {n_new} new hires, "
        f"{n_coaches} coaches × {N_RECORDS} records × {N_ROUNDS} rounds)"
    )
    return grid


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=WHAT_IF_PATH,
        help=f"Output CSV (default {WHAT_IF_PATH.relative_to(ROOT)})",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"LightGBM artifact (default {DEFAULT_MODEL_PATH.relative_to(ROOT)})",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    write_what_if(args.out, model_path=args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
