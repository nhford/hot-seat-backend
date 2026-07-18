"""FastAPI predict service for Hot Seat What-If scoring.

Loads `model/lightgbm.pkl` (written by `python -m src.fit` / `python -m src.score`)
and exposes the same `/predict` contract the Netlify frontend already calls.

Usage (from repo root):

    uvicorn src.serve:app --reload --port 8000
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .predict import DEFAULT_MODEL_PATH, load_artifact, prepare_features

_artifact = load_artifact(DEFAULT_MODEL_PATH)
model = _artifact["model"]
features: list[str] = list(_artifact["feature_names"])
categorical_features: list[str] = list(_artifact.get("categorical_features") or [])

app = FastAPI(title="Hot Seat Predict API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8888",
        "https://localhost:8888",
        "https://hot-seat.netlify.app",
        "https://hot-seat.netlify.app/",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelInput(BaseModel):
    """Prefer `named_features` so column order matches the trained model."""

    features: list[float] | None = None
    named_features: dict[str, Any] | None = Field(default=None)


def _row_from_request(data: ModelInput) -> pd.DataFrame:
    if data.named_features is not None:
        missing = [name for name in features if name not in data.named_features]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing features: {missing}",
            )
        row = {name: data.named_features[name] for name in features}
        return prepare_features(pd.DataFrame([row]), _artifact)

    if data.features is None:
        raise HTTPException(
            status_code=422,
            detail="Provide named_features or features",
        )
    if len(data.features) != len(features):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Expected {len(features)} features in model order "
                f"{features}, got {len(data.features)}"
            ),
        )
    row = dict(zip(features, data.features))
    return prepare_features(pd.DataFrame([row]), _artifact)


@app.get("/")
def home() -> dict[str, str]:
    return {"message": "Backend is running!"}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": DEFAULT_MODEL_PATH.name,
        "n_features": len(features),
        "features": features,
        "categorical_features": categorical_features,
    }


@app.post("/predict")
def predict(data: ModelInput) -> dict[str, Any]:
    input_df = _row_from_request(data)
    prediction = model.predict_proba(input_df)
    return {"prediction": prediction.tolist()}
