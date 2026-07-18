"""Upsert `data/export.csv` into Supabase.

Default target is `coach_year_v2` (new schema, avoids conflicts with legacy
`coach_year`). Create the table first — see `supabase/coach_year_v2.sql`.

Requires `.env` with `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE`.

Usage (from repo root):

    python -m src.export
    python -m src.export --table coach_year_v2
    python -m src.export --path data/export.csv
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from .season import ROOT

DEFAULT_EXPORT = ROOT / "data" / "export.csv"
DEFAULT_ENV = ROOT / ".env"
TABLE = "coach_year_v2"
# Matches PRIMARY KEY in supabase/coach_year_v2.sql
ON_CONFLICT = "id,year,tm"
# PostgREST / gateway payloads blow up if the whole CSV goes in one request.
DEFAULT_CHUNK_SIZE = 200
SCHEMA_SQL = ROOT / "supabase" / "coach_year_v2.sql"

# DB integer/smallint columns. Pandas often serializes these as 0.0 → PostgREST 22P02.
INT_COLUMNS = frozenset(
    {
        "fired",
        "year",
        "age",
        "exp",
        "tenure",
        "tenure_over_500",
        "gm",
        "owner",
        "coy_rank",
        "poc",
        "pred",
        "wins",
        "losses",
        "l_plyf",
        "delta_1yr_plyf",
        "delta_2yr_plyf",
        "delta_3yr_plyf",
    }
)


def _json_safe(value: Any, *, column: str | None = None) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if pd.isna(value):
        return None
    # numpy / pandas scalars → plain Python
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if column in INT_COLUMNS and isinstance(value, (int, float)) and not isinstance(
        value, bool
    ):
        return int(value)
    return value


def records_from_export(path: Path) -> list[dict[str, Any]]:
    data = pd.read_csv(path, index_col=0)
    rows = data.to_dict(orient="records")
    return [
        {k: _json_safe(v, column=k) for k, v in row.items()} for row in rows
    ]


def _format_api_error(exc: BaseException) -> str:
    parts: list[str] = []
    for attr in ("message", "code", "details", "hint"):
        val = getattr(exc, attr, None)
        if val:
            parts.append(f"{attr}={val}")
    if parts:
        return "; ".join(parts)
    return str(exc) or repr(exc)


def _ensure_table_exists(supabase: Any, table: str) -> None:
    try:
        supabase.table(table).select("id,year,team").limit(1).execute()
    except Exception as exc:
        msg = _format_api_error(exc)
        if "does not exist" in msg or "42P01" in msg:
            raise RuntimeError(
                f'Table "{table}" does not exist. '
                f"Run {SCHEMA_SQL.relative_to(ROOT)} in the Supabase SQL editor, "
                f"then retry."
            ) from exc
        raise RuntimeError(f"Could not reach table {table}: {msg}") from exc


def upsert_export(
    *,
    path: Path = DEFAULT_EXPORT,
    env_path: Path = DEFAULT_ENV,
    table: str = TABLE,
    on_conflict: str = ON_CONFLICT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> int:
    try:
        from postgrest.exceptions import APIError
        from supabase import create_client
    except ImportError:
        print(
            "Missing dependency: pip install supabase",
            file=sys.stderr,
        )
        return 1

    load_dotenv(dotenv_path=env_path)

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE")
    if not url or not key:
        print(
            "Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE in environment / .env",
            file=sys.stderr,
        )
        return 1

    if not path.exists():
        print(f"Missing export file: {path}", file=sys.stderr)
        return 1

    rows = records_from_export(path)
    if not rows:
        print("Export CSV has no rows", file=sys.stderr)
        return 1

    supabase = create_client(url, key)
    try:
        _ensure_table_exists(supabase, table)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    total = 0
    try:
        for start in range(0, len(rows), chunk_size):
            chunk = rows[start : start + chunk_size]
            response = (
                supabase.table(table)
                .upsert(chunk, on_conflict=on_conflict)
                .execute()
            )
            n = len(response.data or [])
            total += n
            print(
                f"Upserted rows {start + 1}-{start + len(chunk)} "
                f"({n} returned) into {table}"
            )
    except APIError as exc:
        print(f"Upsert failed: {_format_api_error(exc)}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Older/buggy clients sometimes raise ValidationError on empty error bodies.
        print(f"Upsert failed: {_format_api_error(exc)}", file=sys.stderr)
        print(
            "If this is opaque, confirm the table exists and try a smaller "
            f"--chunk-size (current {chunk_size}).",
            file=sys.stderr,
        )
        return 1

    print(f"Done: {total} rows upserted into {table}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_EXPORT,
        help=f"CSV to upsert (default {DEFAULT_EXPORT.relative_to(ROOT)})",
    )
    p.add_argument(
        "--env",
        type=Path,
        default=DEFAULT_ENV,
        help=f".env path (default {DEFAULT_ENV.relative_to(ROOT)})",
    )
    p.add_argument(
        "--table",
        default=TABLE,
        help=f"Supabase table name (default {TABLE})",
    )
    p.add_argument(
        "--on-conflict",
        default=ON_CONFLICT,
        help=f"Upsert conflict columns (default {ON_CONFLICT})",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Rows per upsert request (default {DEFAULT_CHUNK_SIZE})",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return upsert_export(
        path=args.path,
        env_path=args.env,
        table=args.table,
        on_conflict=args.on_conflict,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
