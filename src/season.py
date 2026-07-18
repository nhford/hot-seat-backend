"""Season settings and repo path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    season: int
    history_end: int
    games: int

    @property
    def season_dir(self) -> Path:
        return ROOT / "config" / str(self.season)


def load_settings(path: Path | None = None) -> Settings:
    path = path or (ROOT / "config" / "settings.yaml")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Settings(
        season=int(raw["season"]),
        history_end=int(raw["history_end"]),
        games=int(raw["games"]),
    )
