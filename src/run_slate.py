"""Simulate a week of games and write output/slate.json.
   python run_slate.py                -> next upcoming slate (auto)
   python run_slate.py --week 9       -> a specific week of the current season
   python run_slate.py --sims 5000    -> more sims per game"""
import argparse, json, sys
from datetime import date, datetime, timedelta, timezone
import pandas as pd
from config import DATA, OUT, MODEL, current_season
from ratings import build_ratings
from players import build_players
from engine import Game, team_params
from report import game_summary


def pick_week(sched, season, today):
    s = sched[(sched.season == season) & sched.home_score.isna()]
    if s.empty:
        return None
    s = s.copy()
    s["gd"] = pd.to_datetime(s.gameday).dt.date
    s = s[s.gd >= today - timedelta(days=1)]
    return int(s.sort_values("gd").iloc[0].week) if len(s) else None


def simulate_week(season, week, pbp, ps, inj, sched, n_sims, rosters, depth=None, seed0=11):
    as_of = (season, week)
    R = build_ratings(pbp, as_of)
    P = build_players(pbp, ps, inj, rosters, depth, as_of)
    games = sched[(sched.season == season) & (sched.week == week)]
    out = []
    for k, r in enumerate(games.itertuples()):
        a, h = r.away_team, r.home_team
        if a not in P or h not in P:
            print("skip (no data):", a, h)
            continue
        outdoors = str(r.roof) in ("outdoors", "open")
        away, home = dict(P[a], abbr=a), dict(P[h], abbr=h)
        params = [team_params(R, a, h, False, r.wind, outdoors), team_params(R, h, a, True, r.wind, outdoors)]
        res = Game(R, [away, home], params, seed=seed0 + k).run(n_sims)
        meta = {"game_id": r.game_id, "week": int(week), "gameday": str(r.gameday), "gametime": str(r.gametime),
                "roof": str(r.roof), "wind": None if pd.isna(r.wind) else float(r.wind),
                "notes": [n for n in {P[a]["note"], P[h]["note"]} if n],
                "flags": {"away": P[a]["flags"], "home": P[h]["flags"]}}
        out.append((meta, away, home, res))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", type=int)
    ap.add_argument("--season", type=int)
    ap.add_argument("--sims", type=int, default=MODEL["n_sims"])
    a = ap.parse_args()
    season = a.season or current_season()
    pbp = pd.read_parquet(DATA / "pbp.parquet"); ps = pd.read_parquet(DATA / "player_stats.parquet")
    inj = pd.read_parquet(DATA / "injuries.parquet"); sched = pd.read_parquet(DATA / "schedule.parquet")
    rosters = pd.read_parquet(DATA / "rosters.parquet")
    depth = pd.read_parquet(DATA / "depth.parquet") if (DATA / "depth.parquet").exists() else None
    week = a.week or pick_week(sched, season, date.today())
    if week is None:
        print("No upcoming games found; nothing to do.")
        sys.exit(0)
    print(f"Simulating {season} week {week}, {a.sims} sims/game")
    sims = simulate_week(season, week, pbp, ps, inj, sched, a.sims, rosters, depth)
    games = [game_summary(res, away, home, meta) for meta, away, home, res in sims]
    games.sort(key=lambda g: (g["gameday"], g["gametime"]))
    OUT.mkdir(exist_ok=True)
    payload = {"season": season, "week": week, "n_sims": a.sims,
               "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "games": games}
    (OUT / "slate.json").write_text(json.dumps(payload, separators=(",", ":")))
    print("wrote", OUT / "slate.json", f"({len(games)} games)")


if __name__ == "__main__":
    main()
