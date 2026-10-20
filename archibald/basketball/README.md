# Basketball (2025–26 NBA season)

Season data pulled from stats.nba.com through `nba_api`. The 2025–26 season is complete: regular season 21 Oct 2025–12 Apr 2026 (1,230 games), play-in (6), playoffs (85), preseason (71).

## Layout

| folder | contents |
|---|---|
| `code/pull_2025_26.py` | resumable pull into Postgres. Skips tables already loaded |
| `code/db.py` | writes frames into the `nba` database |
| `code/shot_viewer/` | local court viewer for downloaded games |
| `logs/pull.log` | progress. `logs/errors.jsonl` is anything the API refused |

The data lives in the local Postgres database `nba`, schemas `league` and `games`. The column-level breakdown is `data/README.md`. Postico:

```
postgresql://nathanclark@127.0.0.1:5432/nba
```

Leave the password blank. Host `127.0.0.1`, port `5432`, user `nathanclark`, database `nba`.

Shot locations for each attempt are columns on `games.pbp` (`x_legacy`, `y_legacy`, `shot_distance`). The league shot-chart endpoint only returns zone totals, in `league.shot_zones`.

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/pull_2025_26.py                 # league tables, then every game
.venv/bin/python code/pull_2025_26.py --league-only
.venv/bin/python code/pull_2025_26.py --games-only    # resume per-game pull
python3 code/shot_viewer/server.py                     # http://127.0.0.1:8766
```

The per-game pull is about 1,400 games × 11 endpoints. It waits 2.5s between requests, pauses 45s every 40 requests, and waits 90s then 3 minutes if the API returns an empty body. It can be stopped and started again. The win-probability endpoint now returns an empty body, so it is not in the pull. 5-man lineup tables are the API's top 2,000 lineups, not every lineup. Team usage is not a team endpoint. The all-star game log came back empty.
