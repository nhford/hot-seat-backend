# NFL Head Coach Hot Seat

Predicts NFL head-coach firing probability from historical Pro-Football-Reference data, Vegas odds, and tenure/context features. LightGBM scores each coach-season; results land in `data/export.csv` and are upserted to Supabase for the frontend.

## Layout

```
config/                 # Human-edited sources of truth
  settings.yaml         # season, history_end, games
  teams.csv             # PFR abbrev ↔ team name
  team_colors.csv       # NFL hex colors
  abbrev_aliases.yaml   # modern/common abbrev → PFR code
  team_aliases.yaml     # historical team name → current name
  retired.yaml          # coach-season labels that are retirements, not firings
  <season>/             # Yearly overrides (e.g. 2025/)
    firings.yaml        # Who was fired after history_end
    hires.yaml          # New HCs + ages
    sb_futures.csv      # SB futures odds
    wins_exp.csv        # Expected win totals

data/
  scraped/              # Cached web pulls (safe to re-fetch)
    coaches.csv         # Master coach index
    coaches/*.csv       # Per-coach season stats
    teams/*.htm         # Team-year pages (GM / owner)
    playoffs.csv        # Historical playoff rounds
    awards/ odds/ standings/
  derived/              # Regenerable pivots / features
  export.csv            # Final publish artifact

model/                  # training.csv, examples_flat.csv, lightgbm.pkl
src/                    # Pipeline Python modules
  fetch.py              # HTTP + read_html helpers
  cache.py              # load_or_build CSV cache
  season.py             # settings.yaml loader
  scrape.py             # scrape CLI (index / coaches / odds)
  score.py              # score coaches with LightGBM, write data/export.csv
  export.py             # upsert data/export.csv → Supabase coach_year_v2
  serve.py              # FastAPI /predict for Hot Seat What-If
hot_seat.ipynb          # Features + model (still notebook)
supabase/
  coach_year_v2.sql     # CREATE TABLE for the new publish target
render.yaml             # Render web service for src.serve
requirements-serve.txt  # Lean deps for the predict API only
```

| Layer | What belongs here | Edited by |
|-------|-------------------|-----------|
| `config/` | Season knobs, firings, hires, futures, teams, colors | You |
| `data/scraped/` | Raw caches from PFR / odds sites | `python -m src.scrape` |
| `data/derived/` | Joined / pivoted tables used by the model | Pipeline |
| `data/export.csv` | Display-ready rows for Supabase | Pipeline |

## Setup

```bash
pip install -r requirements.txt
```

Run commands from the **repo root**. Put Supabase credentials in `.env`:

```
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE=...
```

## Pipeline

1. **Scrape** (Python module — replaces `scrape.ipynb`):

   ```bash
   python -m src.scrape index                 # upsert data/scraped/coaches.csv
   python -m src.scrape coaches               # active coaches (to == history_end)
   python -m src.scrape coaches --id ReidAn0 --fetch
   python -m src.scrape odds                  # O/U tables through history_end
   python -m src.scrape odds --year 2025 --fetch
   python -m src.scrape standings --year 2025 --fetch
   python -m src.scrape coy --year 2025 --fetch
   python -m src.scrape coy --year 2025 --html awards_2025.htm  # Cloudflare workaround
   python -m src.scrape teams --year 2025 --fetch   # 32 team pages + gm/owner derive
   python -m src.scrape all                   # index + coaches + odds + standings + coy + teams
   python -m src.scrape all --fetch           # force re-download
   ```

   Cached files are reused unless you pass `--fetch`. HTTP goes through `src/fetch.py` (Chrome impersonation via `curl_cffi` when installed, else `requests` + browser UA; parse with `StringIO` + `lxml`).

   `standings` / `coy` / `teams` also rebuild `data/derived/` pivots (`standings.csv`, `coy.csv`, `gm.csv`, `owner.csv`) unless you pass `--no-derive`. `all` scrapes team pages for `history_end` only (not the full 1970+ backfill).

   **If PFR returns 403 / Cloudflare:** save the page in your browser and pass `--html`, or set `PFR_COOKIE`:

   ```bash
   python -m src.scrape index --html coaches.htm
   python -m src.scrape coy --year 2025 --html awards_2025.htm
   python -m src.scrape standings --year 2025 --html years_2025.htm
   python -m src.scrape coaches --id ReidAn0 --html ReidAn0.htm
   export PFR_COOKIE='...'   # alternative: live fetch with browser cookie
   ```

   `load_html` / `read_html` in `src/fetch.py` accept `html_path` for all scrapers.
2. **Edit `config/`** — firings, hires, `sb_futures.csv`, `wins_exp.csv` for the active season.
3. **Score** (replaces the scoring half of `hot_seat.ipynb`):

   ```bash
   python -m src.score              # rebuild training if needed, score, write data/export.csv
   python -m src.score --skip-train # reuse training.csv + lightgbm artifacts
   ```

4. **Publish** to Supabase (`coach_year_v2`):

   ```bash
   # once: run supabase/coach_year_v2.sql in the Supabase SQL editor
   python -m src.export
   ```

`hot_seat.ipynb` remains useful for exploratory feature work; production scoring goes through `src.score`.

## Predict API (What-If)

The Netlify Hot Seat app calls `POST /predict` for interactive next-season scoring. That endpoint lives here so it always uses the same `model/lightgbm.pkl` written by `python -m src.fit` / `python -m src.score`.

Local:

```bash
pip install -r requirements-serve.txt
uvicorn src.serve:app --reload --port 8000
```

- `GET /` — liveness
- `GET /health` — feature list
- `POST /predict` — body `{ "named_features": { ... } }` (preferred) or `{ "features": [ ... ] }` in model column order

Prefer `named_features` (the frontend already sends this) so column order cannot drift from training.

### Deploy to Render

1. Commit and push `src/serve.py`, `requirements-serve.txt`, `render.yaml`, and an up-to-date `model/lightgbm.pkl` (from `python -m src.fit` or `python -m src.score`).
2. In [Render](https://dashboard.render.com): **New → Blueprint**, connect `NFordUMass/coaches`, apply `render.yaml`.  
   Or **New → Web Service**, connect the repo, then set:
   - **Runtime:** Python
   - **Build command:** `pip install -r requirements-serve.txt`
   - **Start command:** `uvicorn src.serve:app --host 0.0.0.0 --port $PORT`
   - **Health check path:** `/health`
3. Deploy. Note the service URL (e.g. `https://hot-seat-backend.onrender.com`).
4. Point the frontend (`Hot-Seat` → `WhatIf.tsx`) at that URL’s `/predict` if it differs from the current one.
5. After each model retrain, commit the new `model/lightgbm.pkl` and let Render redeploy (or trigger a manual deploy) so What-If stays in sync with batch scores.

Free-tier notes: the service spins down when idle; the first request after sleep is slow. Keep scrape/train deps out of this service — use `requirements-serve.txt`, not full `requirements.txt`.

## Updating for a new season (e.g. 2026)

1. Fill `config/2026/` (firings, hires, futures, wins_exp).
2. Bump `config/settings.yaml`:

   ```yaml
   season: 2026
   history_end: 2025
   games: 17
   ```

3. Refresh caches:

   ```bash
   python -m src.scrape all --fetch
   ```

4. Build the publish table, then upsert:

   ```bash
   python -m src.score
   # confirm data/export.csv has 32 rows for the new season
   # create supabase/coach_year_v2.sql in the Supabase SQL editor (once)
   python -m src.export
   ```

## Pruned / consolidated

- `scrape.ipynb` → `python -m src.scrape`
- Unused coach HTML cache, headshots, multi-league colors, decision-tree artifacts
- Yearly scrape caches live under `data/scraped/`, not mixed into derived features
