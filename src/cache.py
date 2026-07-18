"""Shared CSV cache helper: load from disk or build + save."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd


def load_or_build(
    path: str | Path,
    builder: Callable[[], pd.DataFrame],
    *,
    fetch: bool = False,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Return CSV at path, or call builder(), write it, and return the result."""
    path = Path(path)
    if not fetch and path.exists():
        return pd.read_csv(path, **read_csv_kwargs)
    df = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df
