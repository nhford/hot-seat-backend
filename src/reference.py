"""Load static reference config (abbrev/team aliases, retired labels)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .season import ROOT


def _load_yaml(path: Path) -> Any:
    with open(path) as f:
        return yaml.safe_load(f)


def load_abbrev_aliases(path: Path | None = None) -> dict[str, str]:
    """Alternate abbrevs → canonical PFR abbrevs (config/abbrev_aliases.yaml)."""
    path = path or (ROOT / "config" / "abbrev_aliases.yaml")
    data = _load_yaml(path) or {}
    return {str(k): str(v) for k, v in data.items()}


def load_team_aliases(path: Path | None = None) -> dict[str, str]:
    """Historical team names → current names (config/team_aliases.yaml)."""
    path = path or (ROOT / "config" / "team_aliases.yaml")
    data = _load_yaml(path) or {}
    return {str(k): str(v) for k, v in data.items()}


def load_retired(path: Path | None = None) -> set[str]:
    """Retirement labels excluded from firing (config/retired.yaml)."""
    path = path or (ROOT / "config" / "retired.yaml")
    data = _load_yaml(path) or {}
    return set(data.get("retired") or [])


def load_franchise_map(path: Path | None = None) -> dict[str, Any]:
    """Year-aware franchise codes + typos (config/franchise_map.yaml)."""
    path = path or (ROOT / "config" / "franchise_map.yaml")
    data = _load_yaml(path) or {}
    typos = {str(k): str(v) for k, v in (data.get("typos") or {}).items()}
    era_splits: dict[str, dict[str, Any]] = {}
    for code, spec in (data.get("era_splits") or {}).items():
        era_splits[str(code)] = {
            "since": int(spec["since"]),
            "legacy": str(spec["legacy"]),
            "modern": str(spec["modern"]),
        }
    franchise_start = {
        str(k): int(v) for k, v in (data.get("franchise_start") or {}).items()
    }
    return {
        "typos": typos,
        "era_splits": era_splits,
        "franchise_start": franchise_start,
    }


def normalize_team_typo(abbrev: str, typos: dict[str, str] | None = None) -> str:
    typos = typos if typos is not None else load_franchise_map()["typos"]
    a = str(abbrev).strip()
    return typos.get(a, a)


def canonical_team(
    abbrev: str,
    year: int | None = None,
    *,
    aliases: dict[str, str] | None = None,
    valid: set[str] | None = None,
    franchise_map: dict[str, Any] | None = None,
) -> str | None:
    """Map a coach/source abbrev to the PFR code used in pivots/playoffs.

    Year-aware for codes that switched franchise meaning (BAL, HOU).
    Without ``year``, falls back to static ``abbrev_aliases`` (modern sense).
    """
    fmap = franchise_map if franchise_map is not None else load_franchise_map()
    aliases = aliases if aliases is not None else load_abbrev_aliases()
    a = normalize_team_typo(abbrev, fmap["typos"])

    if year is not None and a in fmap["era_splits"]:
        spec = fmap["era_splits"][a]
        return spec["legacy"] if int(year) < spec["since"] else spec["modern"]

    if a in aliases:
        return aliases[a]
    if valid is not None and a in valid:
        return a
    if valid is None and a in aliases.values():
        return a
    # Already a canonical code present in teams.csv / pivots
    if valid is not None:
        return None
    return a


def franchise_exists(
    abbrev: str,
    year: int,
    *,
    franchise_map: dict[str, Any] | None = None,
) -> bool:
    """False when a pivot row is a filler year before the franchise existed."""
    fmap = franchise_map if franchise_map is not None else load_franchise_map()
    start = fmap["franchise_start"].get(abbrev)
    if start is None:
        return True
    return int(year) >= int(start)
