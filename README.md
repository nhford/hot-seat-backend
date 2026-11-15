# Hot Seat Backend

Backend for [Hot Seat](https://hot-seat.netlify.app) — a model and data pipeline that estimates NFL head-coach firing probability from Pro-Football-Reference history, Vegas odds, and tenure/context features.

LightGBM scores each coach-season. Season rows, and a What If grid over every coach, record, and playoff result, are published to Supabase. The site looks up a probability when a dropdown changes. `src.serve` remains a local scorer for a single feature vector.

**Related repo:** [hot-seat](https://github.com/nhford/hot-seat) (Netlify frontend) · [hot-seat-backend](https://github.com/nhford/hot-seat-backend) (this repo)

## Architecture

```mermaid
flowchart LR
  subgraph config [config/]
    settings[settings.yaml]
    season[season overrides]
  end
  subgraph scrape [Scrape]
    S[src.scrape]
  end
  subgraph data [data/]
    raw[scraped/]
    derived[derived/]
    export[export.csv]
    grid[what_if.csv]
  end
  subgraph model [model/]
    flat[examples_flat.csv]
    lgb[lightgbm.pkl]
  end
  subgraph publish [Publish]
    SB[src.export → Supabase]
  end
  config --> S
  S --> raw --> derived
  derived --> flat --> lgb
  lgb --> export
  lgb --> grid
  export --> SB
  grid --> SB
  SB --> FE[hot-seat frontend]
```

`src.serve` scores one feature vector locally. The publish path above is the precomputed grid.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add Supabase credentials for export
```

Run from the **repo root**.

### Full pipeline

```bash
python -m src.scrape all          # refresh scraped caches + derived pivots
python -m src.score               # train LightGBM; write data/export.csv and data/what_if.csv
python -m src.export              # upsert coach_year_v2, then what_if
```

`python -m src.score` writes the What If grid at the end, from the same LightGBM artifact as the season rows. To rebuild only that grid:

```bash
python -m src.whatif              # data/what_if.csv
```

Run `supabase/what_if.sql` once in the Supabase SQL editor before the first export that includes the grid (table `what_if`, anon `SELECT`). Later exports only upsert.

Fast iteration when caches and model artifacts are already built:

```bash
python -m src.score --skip-train  # reuse the model; still rewrites both CSVs
python -m src.export
```

### Local scorer

`src.serve` loads `model/lightgbm.pkl` and scores one 54-feature vector on your machine. What If reads the precomputed grid.

```bash
pip install -r requirements-serve.txt
uvicorn src.serve:app --reload --port 8000
```

- `GET /health` — model metadata and feature list
- `POST /predict` — `{ "named_features": { ... } }` (preferred) or `{ "features": [ ... ] }`

`render.yaml` can still host that service. A retrain reaches the site through `python -m src.score` and `python -m src.export`, which rewrite the season rows and the What If grid from the same artifact.

## Repository layout

```
config/                 Human-edited sources of truth
  settings.yaml         Active season, history_end, games per season
  teams.csv, team_colors.csv, abbrev_aliases.yaml, …
  <season>/             Per-year firings, hires, futures, win totals

data/
  scraped/              Cached PFR / odds pulls (regenerable)
  derived/                Pivots and joins used by the model
  export.csv              Display table for Supabase + frontend
  what_if.csv             Precomputed What If probabilities

model/
  examples_flat.csv       Last-k flattened training examples
  lightgbm.pkl            Production classifier (score + local scorer)
  lightgbm_oof.csv        Out-of-fold probabilities for historical rows
  training.csv            Display-feature table for export assembly

src/
  scrape.py               Scrape CLI (index, coaches, odds, standings, coy, teams)
  training.py             Build model/training.csv display features
  examples.py             Build model/examples_flat.csv for LightGBM
  fit.py                  Train LightGBM (GroupKFold OOF + final model)
  score.py                Score coaches; write data/export.csv and data/what_if.csv
  whatif.py               What If grid: coach × record × playoff round
  export.py               Upsert export and what_if → Supabase
  serve.py                Local FastAPI scorer
  fetch.py, cache.py, season.py, reference.py, predict.py

supabase/coach_year_v2.sql   Season table DDL (run once in the Supabase SQL editor)
supabase/what_if.sql         What If grid DDL (run once in the Supabase SQL editor)
render.yaml                  Optional host for the local scorer
```

| Layer | Contents | Updated by |
|-------|----------|------------|
| `config/` | Season knobs, firings, hires, futures | Manual edits |
| `data/scraped/` | Raw web caches | `python -m src.scrape` |
| `data/derived/` | Feature pivots | Scrape derive step / pipeline |
| `data/export.csv` | Published coach-season rows | `python -m src.score` |
| `data/what_if.csv` | What If probabilities (coach × record × round) | `python -m src.score` or `python -m src.whatif` |
| `model/` | Training artifacts + fitted model | `src.examples`, `src.fit`, `src.score` |

## Scrape CLI

```bash
python -m src.scrape index                 # data/scraped/coaches.csv
python -m src.scrape coaches               # active coaches through history_end
python -m src.scrape coaches --id ReidAn0 --fetch
python -m src.scrape odds                  # O/U tables through history_end
python -m src.scrape standings --year 2025 --fetch
python -m src.scrape coy --year 2025 --fetch
python -m src.scrape teams --year 2025 --fetch
python -m src.scrape ingest-coaches        # bulk import coaches_raw → coaches/
python -m src.scrape all                   # index + coaches + odds + standings + coy + teams
python -m src.scrape all --fetch           # force re-download
```

Cached files are reused unless you pass `--fetch`. HTTP uses `curl_cffi` (Chrome impersonation) when available, else `requests` with a browser user-agent.

**Cloudflare / 403 workaround:** save the page in your browser and pass `--html`, or set `PFR_COOKIE` in `.env`:

```bash
python -m src.scrape index --html coaches.htm
export PFR_COOKIE='...'
```

## Model training

The production model is a **LightGBM classifier** on last-`k` flattened coach-stop sequences (default `k=5`).

```bash
python -m src.examples              # model/examples_flat.csv
python -m src.fit                   # model/lightgbm.pkl + OOF metrics
python -m src.fit --rebuild-examples --k 5 --folds 5
```

`python -m src.score` runs examples + fit automatically unless `--skip-train` is passed, then writes the What If grid so those probabilities match the artifact.

Metrics from the latest CV run live in `model/lightgbm_metrics.json`.

## Supabase export

1. Run `supabase/coach_year_v2.sql` once in the Supabase SQL editor.
2. Run `supabase/what_if.sql` once (table `what_if`, primary key `(id, record, round)`, anon `SELECT`).
3. Set `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE` in `.env` (see `.env.example`).
4. `python -m src.export` upserts `data/export.csv` into `coach_year_v2`, then `data/what_if.csv` into `what_if`.

## Rolling to a new season

Example: predicting **2027** after **2026** completes.

1. Create `config/2027/` with `firings.yaml`, `hires.yaml`, `sb_futures.csv`, `wins_exp.csv`.
2. Update `config/settings.yaml`:

   ```yaml
   season: 2027
   history_end: 2026
   games: 17
   ```

3. Refresh data and republish:

   ```bash
   python -m src.scrape all --fetch
   python -m src.score      # data/export.csv and data/what_if.csv
   python -m src.export     # coach_year_v2, then what_if
   ```

   `python -m src.score` writes the What If grid from the new artifact (`python -m src.whatif` rewrites only that CSV). `what_if` is created once with `supabase/what_if.sql`; later seasons only need the upsert.

See `config/2026/README.md` for a season-folder template.

## Data in git

This repo commits scraped caches (`data/scraped/`, ~2k files) so the pipeline runs without a full re-scrape. You can regenerate everything from scratch with `python -m src.scrape all --fetch` if needed.

## Development

```bash
pip install -r requirements-dev.txt   # adds matplotlib, SHAP, Jupyter, etc.
```

## License

MIT — see [LICENSE](LICENSE).
