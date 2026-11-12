"""Upsert `data/export.csv` into Supabase, then `data/what_if.csv`.

Default season table is `coach_year_v2` (avoids conflicts with legacy
`coach_year`). The What If grid goes to `what_if`. Create both tables first —
see `supabase/coach_year_v2.sql` and `supabase/what_if.sql`.

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
DEFAULT_WHAT_IF = ROOT / "data" / "what_if.csv"
DEFAULT_ENV = ROOT / ".env"
TABLE = "coach_year_v2"
WHAT_IF_TABLE = "what_if"
# Live primary key is (id, year, tm). `team` is a separate display column.
ON_CONFLICT = "id,year,tm"
WHAT_IF_ON_CONFLICT = "id,record,round"
# PostgREST / gateway payloads blow up if the whole CSV goes in one request.
DEFAULT_CHUNK_SIZE = 200
SCHEMA_SQL = ROOT / "supabase" / "coach_year_v2.sql"
WHAT_IF_SCHEMA_SQL = ROOT / "supabase" / "what_if.sql"

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
        "pred",
        "wins",
        "losses",
        "l_plyf",
        "delta_1yr_plyf",
        "delta_2yr_plyf",
        "delta_3yr_plyf",
    }
)
# Live `poc` is boolean. CSV stores 0/1.
BOOL_COLUMNS = frozenset({"poc"})


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
    if column in BOOL_COLUMNS and isinstance(value, (int, float)) and not isinstance(
        value, bool
    ):
        return bool(int(value))
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


# what_if.record / what_if.round are integers. The season table's `round`
# column is a float expectation, so it must not share this cast.
WHAT_IF_INT_COLUMNS = frozenset({"record", "round"})
WHAT_IF_COLUMNS = ("id", "record", "round", "prob")


def _what_if_value(value: Any, column: str) -> Any:
    value = _json_safe(value, column=None)
    if (
        column in WHAT_IF_INT_COLUMNS
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        return int(value)
    return value


def records_from_what_if(path: Path) -> list[dict[str, Any]]:
    data = pd.read_csv(path)
    missing = [col for col in WHAT_IF_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"what_if CSV missing columns: {missing}")
    rows = data.loc[:, WHAT_IF_COLUMNS].to_dict(orient="records")
    return [
        {k: _what_if_value(v, k) for k, v in row.items()} for row in rows
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


def _ensure_table_exists(
    supabase: Any,
    table: str,
    *,
    probe: str = "id,year,team",
    schema_sql: Path = SCHEMA_SQL,
) -> None:
    try:
        supabase.table(table).select(probe).limit(1).execute()
    except Exception as exc:
        msg = _format_api_error(exc)
        if "does not exist" in msg or "42P01" in msg or "PGRST205" in msg:
            raise RuntimeError(
                f'Table "{table}" does not exist. '
                f"Run {schema_sql.relative_to(ROOT)} in the Supabase SQL editor, "
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


def upsert_what_if(
    *,
    path: Path = DEFAULT_WHAT_IF,
    env_path: Path = DEFAULT_ENV,
    table: str = WHAT_IF_TABLE,
    on_conflict: str = WHAT_IF_ON_CONFLICT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> int:
    """Upsert the precomputed What If grid after the season table."""
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
        print(
            f"Missing What If grid: {path}. Run `python -m src.whatif` first.",
            file=sys.stderr,
        )
        return 1

    try:
        rows = records_from_what_if(path)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    if not rows:
        print("What If CSV has no rows", file=sys.stderr)
        return 1
    if any(value is None for row in rows for value in row.values()):
        print("What If CSV has null id, record, round, or prob", file=sys.stderr)
        return 1

    supabase = create_client(url, key)
    try:
        _ensure_table_exists(
            supabase,
            table,
            probe="id,record,round",
            schema_sql=WHAT_IF_SCHEMA_SQL,
        )
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
    p.add_argument(
        "--what-if",
        type=Path,
        default=DEFAULT_WHAT_IF,
        help=f"What If grid CSV (default {DEFAULT_WHAT_IF.relative_to(ROOT)})",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code = upsert_export(
        path=args.path,
        env_path=args.env,
        table=args.table,
        on_conflict=args.on_conflict,
        chunk_size=args.chunk_size,
    )
    if code != 0:
        return code
    return upsert_what_if(
        path=args.what_if,
        env_path=args.env,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    raise SystemExit(main())
