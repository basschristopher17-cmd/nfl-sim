"""Team ratings: opponent-neutral-ish event rates for every offense and defense,
recency-weighted and shrunk toward the league average. Also builds the league
yardage pools the engine samples from."""
import numpy as np
import pandas as pd
from config import MODEL


def sit_of(down, togo):
    """Down-and-distance bucket 0..9: 1st down; then 2nd/3rd/4th x (<=3, 4-7, 8+ yards to go)."""
    if down <= 1:
        return 0
    b = 0 if togo <= 3 else (1 if togo <= 7 else 2)
    return (1 if down == 2 else 4 if down == 3 else 7) + b


def prep(pbp, as_of):
    """Regular-season pass/run plays strictly before (season, week) = as_of."""
    s, w = as_of
    d = pbp[(pbp.season_type == "REG") & pbp.posteam.notna() & pbp.play_type.isin(["pass", "run"])].copy()
    d["order"] = d.season * 100 + d.week
    d = d[d.order < s * 100 + w].copy()
    d["scr"] = ((d.play_type == "run") & (d.qb_scramble == 1)).astype(int)
    d["dropback"] = (d.play_type == "pass").astype(int)
    d["pass_call"] = ((d.play_type == "pass") | (d.scr == 1)).astype(int)
    for c in ["sack", "interception", "complete_pass", "fumble_lost", "return_touchdown"]:
        d[c] = d[c].fillna(0)
    d = d[d.down.notna() & d.ydstogo.notna()].copy()
    d["sit"] = [sit_of(a, b) for a, b in zip(d.down.astype(int), d.ydstogo.astype(int))]
    return d


def add_weights(d):
    """w_o = recency weight of the offense's game, w_d = same for the defense."""
    decay = MODEL["decay"]
    for side, col in (("o", "posteam"), ("d", "defteam")):
        g = d.groupby([col, "game_id"], as_index=False)["order"].max()
        g["ago"] = g.groupby(col)["order"].rank(method="first", ascending=False) - 1
        g["w_" + side] = decay ** g["ago"]
        d = d.merge(g[[col, "game_id", "w_" + side]], on=[col, "game_id"], how="left")
    return d


def _shrunk(d, side, mask, val, k):
    col, w = ("posteam", "w_o") if side == "o" else ("defteam", "w_d")
    x = d[mask]
    lg = x[val].mean()
    num = (x[val] * x[w]).groupby(x[col]).sum()
    den = x[w].groupby(x[col]).sum()
    return (num + k * lg) / (den + k), lg


def build_ratings(pbp, as_of):
    d = add_weights(prep(pbp, as_of))
    pass_m = d.dropback == 1
    nosack = pass_m & (d.sack == 0)
    comp_m = nosack & (d.complete_pass == 1)
    designed = (d.play_type == "run") & (d.scr == 0)
    scr_m = pass_m | (d.scr == 1)
    neutral = d.wp.between(0.2, 0.8) & (d.qtr <= 3)

    # name: (mask, value column, shrink strength in weighted plays, sides)
    spec = {
        "sack": (pass_m, "sack", 120, "od"),
        "comp": (nosack, "complete_pass", 150, "od"),
        "int": (nosack, "interception", 500, "od"),
        "ypc": (comp_m, "yards_gained", 120, "od"),   # yards per completion
        "ypr": (designed, "yards_gained", 120, "od"),  # yards per designed rush
        "scr": (scr_m, "scr", 300, "o"),               # QB scramble rate per dropback
        "passrate": (neutral, "pass_call", 250, "o"),  # neutral-situation pass rate
    }
    R = {"league": {}, "off": {}, "def": {}}
    teams = sorted(set(d.posteam) | set(d.defteam))
    for t in teams:
        R["off"][t], R["def"][t] = {}, {}
    for name, (mask, val, k, sides) in spec.items():
        for side in sides:
            ser, lg = _shrunk(d, side, mask, val, k)
            R["league"][name] = float(lg)
            for t in teams:
                R["off" if side == "o" else "def"][t][name] = float(ser.get(t, lg))
    for t in teams:  # defense has no scramble/pass-rate rating -> neutral
        for name in ("scr", "passrate"):
            R["def"][t][name] = R["league"][name]

    # pace = offensive plays per game
    tg = d.groupby(["posteam", "game_id"]).agg(n=("play_type", "size"), w=("w_o", "first")).reset_index()
    lg_ppg = tg.n.mean()
    num = (tg.n * tg.w).groupby(tg.posteam).sum()
    den = tg.w.groupby(tg.posteam).sum()
    ppg = (num + 3 * lg_ppg) / (den + 3)
    R["league"]["ppg"] = float(lg_ppg)
    for t in teams:
        R["off"][t]["pace"] = float(ppg.get(t, lg_ppg) / lg_ppg)

    # empirical yardage pools + small constants
    def zones(mask):
        # yardage pools by [situation][distance-to-goal zone]; thin pools fall back to a wider pool
        x = d[mask & d.yardline_100.notna()].copy()
        x["z"] = ((x.yardline_100 - 1) // 5).clip(0, 8).astype(int)
        allp = x.yards_gained.round().astype(int).tolist()
        out = []
        for s in range(10):
            xs = x[x.sit == s]
            sp = xs.yards_gained.round().astype(int).tolist() if len(xs) >= 60 else allp
            row = []
            for z in range(9):
                p = xs[xs.z == z].yards_gained.round().astype(int).tolist()
                row.append(p if len(p) >= 60 else sp)
            out.append(row)
        return out

    def mult(mask, val, base_mask=None):
        x = d[mask]
        lg_ = x[val].mean()
        g = x.groupby("sit")[val].agg(["sum", "count"])
        return [float(((g["sum"].get(s, 0) + 100 * lg_) / (g["count"].get(s, 0) + 100)) / lg_) for s in range(10)]

    R["sit"] = {
        "pass": mult(neutral, "pass_call"),
        "comp": mult(nosack, "complete_pass"),
        "sack": mult(pass_m, "sack"),
        "int": mult(nosack, "interception"),
    }

    R["pools"] = {
        "comp": zones(comp_m),
        "run": zones(designed),
        "scr": d[d.scr == 1].yards_gained.round().astype(int).tolist(),
        "sack": d[d.sack == 1].yards_gained.round().astype(int).tolist(),
    }
    R["const"] = {
        "fum_run": float(d[designed | (d.scr == 1)].fumble_lost.mean()),
        "fum_comp": float(d[comp_m].fumble_lost.mean()),
        "pick6": float(d[d.interception == 1].return_touchdown.mean()),
    }
    R["fg"] = fg_curve(pbp, as_of)
    return R


def fg_curve(pbp, as_of):
    s, w = as_of
    f = pbp[(pbp.field_goal_attempt == 1) & pbp.kick_distance.notna()].copy()
    f = f[(f.season * 100 + f.week) < s * 100 + w]
    f["made"] = (f.field_goal_result == "made").astype(int)
    xs = np.array([20, 30, 40, 50, 56, 62], dtype=float)
    ys = []
    for lo, hi in [(0, 25), (25, 35), (35, 45), (45, 53), (53, 59), (59, 80)]:
        sub = f[(f.kick_distance >= lo) & (f.kick_distance < hi)]
        ys.append(sub.made.mean() if len(sub) > 15 else np.nan)
    ys = np.array(ys, dtype=float)
    fallback = np.array([0.99, 0.95, 0.87, 0.77, 0.62, 0.45])
    ys = np.where(np.isnan(ys), fallback, ys)
    return {int(dist): float(np.interp(dist, xs, ys)) for dist in range(18, 76)}
