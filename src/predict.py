"""Load LightGBM artifact and score feature rows."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = ROOT / "model" / "lightgbm.pkl"
FLAT_PATH = ROOT / "model" / "examples_flat.csv"
OOF_PATH = ROOT / "model" / "lightgbm_oof.csv"


def load_artifact(path: Path | None = None) -> dict[str, Any]:
    path = path or DEFAULT_MODEL_PATH
    with open(path, "rb") as f:
        return pickle.load(f)


def prepare_features(
    rows: pd.DataFrame,
    artifact: dict[str, Any],
) -> pd.DataFrame:
    feature_names = artifact["feature_names"]
    cat_names = artifact.get("categorical_features") or []
    X = rows.reindex(columns=feature_names).copy()
    # JSON null → None becomes object dtype; coerce so LightGBM sees float NaN.
    for col in feature_names:
        if col in cat_names:
            continue
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in cat_names:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype("Int64")
    return X


def predict_proba(
    rows: pd.DataFrame,
    artifact: dict[str, Any] | None = None,
    *,
    model_path: Path | None = None,
) -> pd.Series:
    if artifact is None:
        artifact = load_artifact(model_path)
    model = artifact["model"]
    X = prepare_features(rows, artifact)
    return pd.Series(model.predict_proba(X)[:, 1], index=rows.index)
