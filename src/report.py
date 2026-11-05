"""Turn raw simulation arrays into compact summaries (means + percentile bands)."""
import numpy as np
from engine import STATS

S = {n: i for i, n in enumerate(STATS)}
QB_STATS = ["pass_att", "pass_cmp", "pass_yds", "pass_td", "pass_int", "sacked", "rush_att", "rush_yds"]
REC_STATS = ["tgt", "rec", "rec_yds", "rec_td"]
RUSH_STATS = ["rush_att", "rush_yds", "rush_td"]


def band(a, nd=1):
    a = np.asarray(a, dtype=float)
    p = np.percentile(a, [10, 25, 50, 75, 90])
    return [round(float(a.mean()), nd)] + [round(float(x), nd) for x in p]  # mean, p10, p25, p50, p75, p90


def player_lines(res, t, team):
    out = []
    st, act = res["stats"][t], res["active"][t]
    for i, p in enumerate(team["players"]):
        m = act[:, i]
        if m.sum() < 20:
            continue
        x = st[m, i, :]
        names = []
        if i == 0:
            names = QB_STATS
        else:
            if p["rec"]:
                names += REC_STATS
            if p["rush"]:
                names += RUSH_STATS
        stats = {n: band(x[:, S[n]]) for n in names}
        td = (x[:, S["rush_td"]] + x[:, S["rec_td"]] + x[:, S["pass_td"]] * (i == 0)) >= 1 if i else (x[:, S["rush_td"]] >= 1)
        out.append({"id": p["id"], "name": p["name"], "pos": "QB" if i == 0 else p["pos"],
                    "p_active": round(float(m.mean()), 2), "new": bool(p.get("new")), "p_td": round(float(td.mean()), 3), "stats": stats})
    return out


def game_summary(res, away, home, meta):
    sc = res["score"]
    tot = sc.sum(1)
    margin = sc[:, 1] - sc[:, 0]  # home - away
    hw = (margin > 0).mean() + 0.5 * (margin == 0).mean()
    d = dict(meta)
    d.update({
        "away": away["abbr"], "home": home["abbr"],
        "win_prob_home": round(float(hw), 3),
        "score_away": band(sc[:, 0]), "score_home": band(sc[:, 1]),
        "total": band(tot), "margin_home": band(margin),
        "ot_rate": round(float(res["ot"].mean()), 3),
        "team_stats": {
            "away": {"plays": band(res["plays"][:, 0]), "pass_yds": band(res["pyd"][:, 0]), "rush_yds": band(res["ryd"][:, 0]), "turnovers": band(res["tov"][:, 0])},
            "home": {"plays": band(res["plays"][:, 1]), "pass_yds": band(res["pyd"][:, 1]), "rush_yds": band(res["ryd"][:, 1]), "turnovers": band(res["tov"][:, 1])},
        },
        "players": {"away": player_lines(res, 0, away), "home": player_lines(res, 1, home)},
    })
    return d
