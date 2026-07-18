# Season config template

Copy this folder when rolling to a new prediction season (e.g. `config/2027/`).

Required files:

| File | Purpose |
|------|---------|
| `firings.yaml` | Coaches fired after the completed season (`history_end`) |
| `hires.yaml` | New head coaches (team abbrev → PFR id) + ages for first-time HCs |
| `sb_futures.csv` | Super Bowl futures odds by team |
| `wins_exp.csv` | Expected win totals (rank-aligned with futures) |

Then update `config/settings.yaml`:

```yaml
season: 2027        # upcoming / in-progress NFL year
history_end: 2026   # last completed season with full stats
games: 17
```

Run `python -m src.scrape all --fetch`, then `python -m src.score` and `python -m src.export`.
