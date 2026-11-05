"""Score the model on finished games, using only data available before each week.
   python backtest.py --season 2025 --start 6 --end 18 --sims 300"""
import argparse
import numpy as np
import pandas as pd
from config import DATA
from run_slate import simulate_week
from report import S

ACT = {"pass_yds": "passing_yards", "rush_yds": "rushing_yards", "rec_yds": "receiving_yards", "rec": "receptions"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--start", type=int, default=6)
    ap.add_argument("--end", type=int, default=18)
    ap.add_argument("--sims", type=int, default=300)
    a = ap.parse_args()
    pbp = pd.read_parquet(DATA / "pbp.parquet"); ps = pd.read_parquet(DATA / "player_stats.parquet")
    inj = pd.read_parquet(DATA / "injuries.parquet"); sched = pd.read_parquet(DATA / "schedule.parquet")
    rosters = pd.read_parquet(DATA / "rosters.parquet")
    psr = ps[(ps.season == a.season) & (ps.season_type == "REG")]
    rows, prow = [], []
    for wk in range(a.start, a.end + 1):
        for meta, away, home, res in simulate_week(a.season, wk, pbp, ps, inj, sched, a.sims, rosters, None):
            g = sched[sched.game_id == meta["game_id"]].iloc[0]
            if pd.isna(g.home_score):
                continue
            sc = res["score"]; tot = sc.sum(1); mg = sc[:, 1] - sc[:, 0]
            act_tot, act_mg = g.home_score + g.away_score, g.home_score - g.away_score
            hw = (mg > 0).mean() + 0.5 * (mg == 0).mean()
            rows.append(dict(week=wk, pred_tot=tot.mean(), act_tot=act_tot, pred_mg=mg.mean(), act_mg=act_mg,
                             p_home=hw, home_won=float(act_mg > 0) if act_mg != 0 else 0.5,
                             tot_in_80=float(np.percentile(tot, 10) <= act_tot <= np.percentile(tot, 90)),
                             pred_plays=res["plays"].mean()))
            for t, team in enumerate((away, home)):
                abbr = team["abbr"]
                got = psr[(psr.week == wk) & (psr.team == abbr)].set_index("player_id")
                for i, p in enumerate(team["players"]):
                    if p["id"] not in got.index:
                        continue
                    m = res["active"][t][:, i]
                    if m.sum() < 30:
                        continue
                    for stat, col in ACT.items():
                        x = res["stats"][t][m, i, S[stat]]
                        if x.mean() < (20 if "yds" in stat else 2.5):
                            continue
                        act = float(got.loc[p["id"], col])
                        prow.append(dict(stat=stat, pred=x.mean(), act=act,
                                         inside=float(np.percentile(x, 10) <= act <= np.percentile(x, 90)),
                                         below_p25=float(act < np.percentile(x, 25)), above_p75=float(act > np.percentile(x, 75))))
    g = pd.DataFrame(rows); p = pd.DataFrame(prow)
    print(f"\n=== GAME LEVEL: {len(g)} games, {a.season} weeks {a.start}-{a.end} ===")
    print(f"Total points  MAE {np.abs(g.pred_tot-g.act_tot).mean():.2f} | bias (pred-actual) {np.mean(g.pred_tot-g.act_tot):+.2f} | 10-90 band coverage {g.tot_in_80.mean():.0%} (target 80%)")
    print(f"Margin (home) MAE {np.abs(g.pred_mg-g.act_mg).mean():.2f} | bias {np.mean(g.pred_mg-g.act_mg):+.2f}")
    print(f"Win prob      Brier {np.mean((g.p_home-g.home_won)**2):.4f} (coin flip = 0.2500) | accuracy {((g.p_home>0.5)==(g.home_won>0.5)).mean():.1%}")
    print(f"Plays/team    pred {g.pred_plays.mean():.1f}")
    bins = pd.cut(g.p_home, [0, .3, .4, .5, .6, .7, 1.0])
    print("Calibration (predicted bucket -> actual home win rate):")
    print(g.groupby(bins, observed=True).agg(n=("home_won", "size"), pred=("p_home", "mean"), actual=("home_won", "mean")).round(2).to_string())
    print(f"\n=== PLAYER LEVEL ({len(p)} player-games) ===")
    print(p.groupby("stat").agg(n=("act", "size"), pred_mean=("pred", "mean"), act_mean=("act", "mean"),
                                inside_10_90=("inside", "mean"), below_p25=("below_p25", "mean"), above_p75=("above_p75", "mean")).round(2).to_string())
    print("(inside_10_90 should be ~0.80; below_p25 and above_p75 should be ~0.25)")


if __name__ == "__main__":
    main()
