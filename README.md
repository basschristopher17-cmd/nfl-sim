# NFL Sim

Play-by-play Monte Carlo that projects NFL game outcomes (scores, totals, margins, win probability)
and player stat lines. Data: [nflverse](https://nflverse.com) via `nflreadpy`. No betting lines are used anywhere.

## Run it
```
pip install -r requirements.txt
python src/ingest.py                 # download + cache nflverse data
python src/run_slate.py              # simulate the next slate  -> output/slate.json
python src/build_html.py             # build the dashboard      -> docs/index.html
python src/backtest.py --season 2025 --start 6 --end 18 --sims 300   # accuracy report
```
Useful flags: `run_slate.py --week 10 --sims 5000`.

## Files
| File | Job |
|---|---|
| `src/ingest.py` | pulls play-by-play, player stats, schedule, injuries; caches to `data/` |
| `src/ratings.py` | recency-weighted, shrunk team ratings + situation tables + yardage pools |
| `src/players.py` | starting QB, target / carry shares, catch rate, yards per catch/carry, injuries |
| `src/engine.py` | the game simulator (drives, plays, clock, 4th downs, FGs, penalties, OT) |
| `src/report.py` | turns simulation arrays into means + 10/25/50/75/90 percentile bands |
| `src/run_slate.py` | simulates a week of games -> `output/slate.json` |
| `src/backtest.py` | scores the model on finished games using only pre-game data |
| `overrides.json` | hand-edit: force players out/in, force a starting QB |
| `config.py` knobs | `MODEL` dict in `src/config.py` (decay, sims, home field, variance, calibration) |

## Rosters (automatic)
`ingest.py` pulls nflverse weekly rosters and the newest depth chart every run, so trades, cuts, IR moves and new signings are picked up automatically (the GitHub Action runs every 6 hours). Players on IR / cut are excluded, traded players move with their stats, and rookies with a depth-chart role get a small placeholder share (tagged NEW). Each team tab lists the roster changes the model applied.

## Overrides
Edit `overrides.json`, commit, and the next run uses it:
```json
{ "out": ["Player Name"], "in": ["Player Name"], "qb": {"BAL": "Lamar Jackson"}, "team": {"Player Name": "KC"} }
```
