"""FastAPI predict service for Hot Seat What-If scoring.

Loads `model/random_forest.pkl` (written by `python -m src.score`) and exposes
the same `/predict` contract the Netlify frontend already calls.

Usage (from repo root):

    uvicorn src.serve:app --reload --port 8000
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "random_forest.pkl"

with open(MODEL_PATH, "rb") as file:
    _model_data = pickle.load(file)
    model = _model_data["model"]
    features: list[str] = list(_model_data["feature_names"])

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


def _row_from_request(data: ModelInput) -> list[float]:
    if data.named_features is not None:
        missing = [name for name in features if name not in data.named_features]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing features: {missing}",
            )
        return [float(data.named_features[name]) for name in features]

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
    return [float(v) for v in data.features]


@app.get("/")
def home() -> dict[str, str]:
    return {"message": "Backend is running!"}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "n_features": len(features), "features": features}


@app.post("/predict")
def predict(data: ModelInput) -> dict[str, Any]:
    row = _row_from_request(data)
    input_df = pd.DataFrame([row], columns=features)
    prediction = model.predict_proba(input_df)
    return {"prediction": prediction.tolist()}
