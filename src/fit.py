"""Train LightGBM on last-k flattened coach-firing examples.

Uses `model/examples_flat.csv` (rebuild with `python -m src.examples`).
Evaluates with GroupKFold by coach id, then fits a final model on all rows.

Usage (from repo root):

    python -m src.fit
    python -m src.fit --rebuild-examples
    python -m src.fit --k 12 --folds 5
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from .examples import (
    DEFAULT_K,
    build_and_write,
    categorical_feature_names,
    feature_columns,
)
from .season import ROOT, load_settings

FLAT_PATH = ROOT / "model" / "examples_flat.csv"
MODEL_PATH = ROOT / "model" / "lightgbm.pkl"
METRICS_PATH = ROOT / "model" / "lightgbm_metrics.json"
OOF_PATH = ROOT / "model" / "lightgbm_oof.csv"

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


def load_flat(path: Path | None = None) -> pd.DataFrame:
    path = path or FLAT_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; run `python -m src.examples` first "
            "or pass --rebuild-examples."
        )
    return pd.read_csv(path)


def prepare_xy(
    flat: pd.DataFrame, *, k: int = DEFAULT_K
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Return X, y, groups, and categorical feature names present in X."""
    feats = feature_columns(flat)
    X = flat[feats].copy()
    y = flat["fired"].astype(int)
    groups = flat["id"].astype(str)

    cat_names = [c for c in categorical_feature_names(k) if c in X.columns]
    for col in cat_names:
        # Keep NaN pads as missing; LightGBM categorical needs int codes otherwise.
        X[col] = X[col].astype("Int64")
    return X, y, groups, cat_names


def make_model(
    *,
    scale_pos_weight: float,
    categorical_feature: list[str],
    params: dict[str, Any] | None = None,
) -> lgb.LGBMClassifier:
    cfg = {**DEFAULT_PARAMS, **(params or {})}
    cfg["scale_pos_weight"] = scale_pos_weight
    model = lgb.LGBMClassifier(**cfg)
    # Stash for fit() — sklearn API accepts categorical_feature in fit
    model._categorical_feature = categorical_feature  # type: ignore[attr-defined]
    return model


def _fit(
    model: lgb.LGBMClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    eval_set: list[tuple[pd.DataFrame, pd.Series]] | None = None,
) -> lgb.LGBMClassifier:
    cat = getattr(model, "_categorical_feature", "auto")
    fit_kwargs: dict[str, Any] = {"categorical_feature": cat}
    if eval_set is not None:
        fit_kwargs["eval_set"] = eval_set
        fit_kwargs["callbacks"] = [lgb.early_stopping(50, verbose=False)]
    model.fit(X, y, **fit_kwargs)
    return model


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    *,
    categorical_features: list[str],
    folds: int = 5,
    params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    gkf = GroupKFold(n_splits=folds)
    oof = np.zeros(len(y), dtype=float)
    fold_metrics: list[dict[str, float]] = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        pos = max(int(y_tr.sum()), 1)
        neg = max(int((y_tr == 0).sum()), 1)
        spw = neg / pos

        model = make_model(
            scale_pos_weight=spw,
            categorical_feature=categorical_features,
            params=params,
        )
        _fit(model, X_tr, y_tr, eval_set=[(X_te, y_te)])

        proba = model.predict_proba(X_te)[:, 1]
        oof[te] = proba
        auc = roc_auc_score(y_te, proba)
        ap = average_precision_score(y_te, proba)
        best_iter = getattr(model, "best_iteration_", None)
        if not best_iter:
            best_iter = model.n_estimators
        fold_metrics.append(
            {
                "fold": fold,
                "n": int(len(te)),
                "n_pos": int(y_te.sum()),
                "roc_auc": float(auc),
                "avg_precision": float(ap),
                "best_iteration": int(best_iter),
            }
        )
        print(
            f"fold {fold}: ROC-AUC={auc:.3f}  AP={ap:.3f}  "
            f"n={len(te)} pos={int(y_te.sum())}"
        )

    overall = {
        "roc_auc": float(roc_auc_score(y, oof)),
        "avg_precision": float(average_precision_score(y, oof)),
        "folds": fold_metrics,
        "classification_report": classification_report(
            y, (oof >= 0.5).astype(int), digits=3, output_dict=True
        ),
    }
    print(
        f"OOF ROC-AUC={overall['roc_auc']:.3f}  "
        f"AP={overall['avg_precision']:.3f}"
    )
    print(classification_report(y, (oof >= 0.5).astype(int), digits=3))

    oof_df = pd.DataFrame(
        {
            "id": groups.values,
            "fired": y.values,
            "proba": oof,
        }
    )
    return oof_df, overall


def fit_final(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    categorical_features: list[str],
    n_estimators: int | None = None,
    params: dict[str, Any] | None = None,
) -> lgb.LGBMClassifier:
    pos = max(int(y.sum()), 1)
    neg = max(int((y == 0).sum()), 1)
    cfg = dict(params or {})
    if n_estimators is not None:
        cfg["n_estimators"] = n_estimators
    model = make_model(
        scale_pos_weight=neg / pos,
        categorical_feature=categorical_features,
        params=cfg,
    )
    _fit(model, X, y)
    return model


def train(
    *,
    flat_path: Path = FLAT_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    oof_path: Path = OOF_PATH,
    k: int = DEFAULT_K,
    folds: int = 5,
    rebuild_examples: bool = False,
) -> dict[str, Any]:
    settings = load_settings()
    if rebuild_examples:
        build_and_write(k=k, out_flat=flat_path, out_jsonl=None)

    flat = load_flat(flat_path)
    X, y, groups, cat_names = prepare_xy(flat, k=k)

    oof_df, cv_metrics = cross_validate(
        X, y, groups, categorical_features=cat_names, folds=folds
    )
    # Use median best_iteration from CV for the final fit when available
    best_iters = [
        f["best_iteration"]
        for f in cv_metrics["folds"]
        if f.get("best_iteration")
    ]
    n_est = int(np.median(best_iters)) if best_iters else DEFAULT_PARAMS["n_estimators"]
    n_est = max(n_est, 50)

    model = fit_final(
        X, y, categorical_features=cat_names, n_estimators=n_est
    )
    artifact = {
        "model": model,
        "feature_names": list(X.columns),
        "categorical_features": cat_names,
        "k": k,
        "n_estimators": n_est,
        "scale_pos_weight": float(
            max(int((y == 0).sum()), 1) / max(int(y.sum()), 1)
        ),
        "history_end": settings.history_end,
        "season": settings.season,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f)

    # Attach keys for OOF file
    oof_out = flat[["id", "year", "tm", "fired"]].copy()
    oof_out["proba"] = oof_df["proba"].values
    oof_out.to_csv(oof_path, index=False)

    meta = {
        "n_examples": int(len(flat)),
        "n_fired": int(y.sum()),
        "n_features": int(X.shape[1]),
        "k": k,
        "folds": folds,
        "final_n_estimators": n_est,
        "categorical_features": cat_names,
        "cv": cv_metrics,
        "model_path": str(model_path.relative_to(ROOT)),
        "oof_path": str(oof_path.relative_to(ROOT)),
        "flat_path": str(flat_path.relative_to(ROOT)),
        "history_end": settings.history_end,
        "year_max": int(flat["year"].max()),
        "year_min": int(flat["year"].min()),
    }
    with open(metrics_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(
        f"Saved {model_path.relative_to(ROOT)} "
        f"(n_estimators={n_est}, features={X.shape[1]})"
    )
    print(f"Saved {metrics_path.relative_to(ROOT)} and {oof_path.relative_to(ROOT)}")
    return meta


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=DEFAULT_K)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument(
        "--rebuild-examples",
        action="store_true",
        help="Rebuild model/examples_flat.csv before fitting",
    )
    p.add_argument(
        "--flat",
        type=Path,
        default=FLAT_PATH,
        help="Flattened examples CSV",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=MODEL_PATH,
        help="Pickle path for the fitted model artifact",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train(
        flat_path=args.flat,
        model_path=args.out,
        k=args.k,
        folds=args.folds,
        rebuild_examples=args.rebuild_examples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
