"""Play-by-play Monte Carlo engine.

One simulated game = drives -> plays, with down/distance/field position, a game clock,
4th-down decisions, field goals, punts, turnovers, overtime. Every yard is credited to
a player, so team scores, totals, and player stat lines all come from the same plays."""
import random
import numpy as np
from config import MODEL

STATS = ["pass_att", "pass_cmp", "pass_yds", "pass_td", "pass_int", "sacked",
         "rush_att", "rush_yds", "rush_td", "tgt", "rec", "rec_yds", "rec_td"]
NS = len(STATS)
(PA, PC, PY, PT, PI, SK, RA, RY, RT, TG, RC, RYD, RTD) = range(NS)

RUN_T, COMP_T, INC_T, SACK_T, SCR_T = 36.0, 30.0, 7.0, 29.0, 22.0
RET_TD = 0.010  # chance a kickoff/punt is returned for a touchdown
KICK_YTG = 66  # avg yards-to-goal after a kickoff (2025 rules: touchbacks come out to the 35)
PEN_D, PEN_O = 0.040, 0.032  # per-snap chance of an accepted defensive / offensive penalty


def team_params(R, off, dfn, is_home, wind=None, outdoors=False):
    """Matchup-adjusted event rates for `off` offense vs `dfn` defense."""
    lg, O, D = R["league"], R["off"][off], R["def"][dfn]
    mu = lambda k: lg[k] + (O[k] - lg[k]) + (D[k] - lg[k])
    hfa = MODEL["hfa"] * (1 if is_home else -1)
    p = dict(
        pass_rate=min(0.72, max(0.45, O["passrate"] + (0.0))),
        scr_p=max(0.01, O["scr"]),
        sack_p=min(0.14, max(0.02, mu("sack"))),
        comp_p=min(0.80, max(0.50, mu("comp"))),
        int_p=min(0.05, max(0.008, mu("int"))),
        ypc_scale=max(0.7, mu("ypc") / lg["ypc"]) * (1 + hfa) * MODEL["eff"],
        run_scale=max(0.7, mu("ypr") / lg["ypr"]) * (1 + hfa) * MODEL["eff"],
    )
    if outdoors and wind is not None and wind == wind and wind >= 15:
        p["ypc_scale"] *= 0.95 if wind < 20 else 0.90
        p["comp_p"] -= 0.02 if wind < 20 else 0.04
        p["pass_rate"] -= 0.02
    return p


class Game:
    def __init__(self, R, teams, params, seed=0):
        """teams: [teamA, teamB] player dicts from players.build_players; params: [pA, pB]."""
        self.R, self.T, self.P = R, teams, params
        self.rnd = random.Random(seed)
        self.nrng = np.random.default_rng(seed)
        pace = (R["off"][teams[0]["abbr"]]["pace"] * R["off"][teams[1]["abbr"]]["pace"]) ** 0.5
        self.spp = MODEL["time_scale"] / pace
        pools = R["pools"]
        self.pc, self.pr, self.ps, self.pk = pools["comp"], pools["run"], pools["scr"], pools["sack"]
        self.fg = R["fg"]
        self.const = R["const"]

    # ---------- per-sim player shares ----------
    def _draw(self, t):
        T, rnd = self.T[t], self.rnd
        pl = T["players"]
        avail = [rnd.random() < p["p_play"] for p in pl]
        T["qb_i"] = 0
        avail[0] = True

        def build(key, conc):
            idx = [i for i, p in enumerate(pl) if p[key] and avail[i]]
            if not idx:
                idx = [i for i, p in enumerate(pl) if p[key]]
            if not idx:
                return [], [1.0]
            w = np.array([pl[i][key]["share"] for i in idx])
            w = w / w.sum()
            s = self.nrng.dirichlet(np.maximum(w * conc, 1e-3))
            return idx, np.cumsum(s).tolist()

        ridx, rcum = build("rec", MODEL["conc_tgt"])
        cidx, ccum = build("rush", MODEL["conc_car"])
        return avail, ridx, rcum, cidx, ccum

    # ---------- one game ----------
    def sim_one(self):
        rnd = self.rnd.random
        gauss = self.rnd.gauss
        from bisect import bisect
        T = self.T
        draws = [self._draw(0), self._draw(1)]
        fsd = MODEL["form_sd"]
        PP = []
        for base in self.P:
            f = gauss(1.0, fsd)
            q = dict(base)
            q["ypc_scale"] = base["ypc_scale"] * f
            q["run_scale"] = base["run_scale"] * (1 + (f - 1) * 0.6)
            q["comp_p"] = min(0.8, max(0.45, base["comp_p"] + (f - 1) * 0.5))
            PP.append(q)
        acc = [[[0] * NS for _ in T[t]["players"]] for t in (0, 1)]
        score = [0, 0]
        plays = [0, 0]
        tov = [0, 0]
        pyd = [0, 0]
        ryd = [0, 0]
        pcz, prz, ps_, pk = self.pc, self.pr, self.ps, self.pk
        npcz = [[len(x) for x in row] for row in pcz]
        nprz = [[len(x) for x in row] for row in prz]
        SIT = self.R["sit"]
        nps, npk = len(ps_), len(pk)
        fgp, const, spp = self.fg, self.const, self.spp

        def kick():
            return min(80, max(45, int(gauss(KICK_YTG, 8))))

        clock = 3600.0
        pos = 0 if rnd() < 0.5 else 1
        second = 1 - pos
        half = 1
        ot = False
        ot_poss = 0
        ytg = kick()

        while True:
            o = PP[pos]
            dfn_i = 1 - pos
            avail, ridx, rcum, cidx, ccum = draws[pos]
            pl = T[pos]["players"]
            A = acc[pos]
            y = ytg
            down = 1
            togo = 10 if y > 10 else y
            res = None
            npos = dfn_i
            nytg = None

            while True:
                lead = score[pos] - score[dfn_i]
                # ----- 4th down -----
                if down == 4:
                    act = "punt"
                    if lead < 0 and clock < 300 and half == 2:
                        act = "fg" if (y <= 37 and lead >= -3) else "go"
                    else:
                        pgo = (0.70 if togo <= 1 else 0.55 if togo == 2 else 0.30 if togo <= 4 else 0.10 if togo <= 7 else 0.0)
                        if y > 55 or (y > 50 and togo > 1) or y < 1:
                            pgo = 0.0
                        if pgo and rnd() < pgo:
                            act = "go"
                        elif y <= 37:
                            act = "fg"
                    if act == "fg":
                        ok = rnd() < fgp[min(75, y + 17)]
                        clock -= 6
                        if ok:
                            score[pos] += 3
                            res, nytg = "fg", kick()
                        else:
                            res, nytg = "miss", min(80, max(1, 100 - (y + 7)))
                        break
                    if act == "punt":
                        clock -= 10
                        net = gauss(41, 7)
                        res, nytg = "punt", min(80, max(1, int(100 - y + net)))
                        break
                # ----- penalties (replay the down) -----
                u = rnd()
                if u < PEN_D + PEN_O:
                    if u < PEN_D:  # defense: +5 / +10 / spot foul, sometimes automatic first down
                        v = rnd()
                        if v < 0.5:
                            g, auto = 5, rnd() < 0.35
                        elif v < 0.8:
                            g, auto = 10, rnd() < 0.20
                        else:
                            g, auto = 15 + int(rnd() * 15), True
                        g = min(g, max(1, y // 2))
                        y -= g
                        if auto or g >= togo:
                            down = 1
                            togo = 10 if y > 10 else y
                        else:
                            togo -= g
                    else:  # offense: holding / false start
                        v = rnd()
                        L = 5 if v < 0.6 else (10 if v < 0.95 else 15)
                        y = min(99, y + L)
                        togo += L
                    clock -= 9 * spp
                    if (half == 1 and clock <= 1800) or (half == 2 and clock <= 0) or (ot and clock <= 0):
                        res = "clock"
                        break
                    continue
                # ----- choose play -----
                z = min(8, (y - 1) // 5)
                st = 0 if down <= 1 else ((1 if down == 2 else 4 if down == 3 else 7) + (0 if togo <= 3 else 1 if togo <= 7 else 2))
                pc, npc, pr, npr = pcz[st][z], npcz[st][z], prz[st][z], nprz[st][z]
                rzm = MODEL["rz_boost"] if y <= 12 else 1.0
                pr_ = o["pass_rate"] * SIT["pass"][st]
                if half == 1 and clock <= 1920:
                    pr_ += 0.15
                elif half == 2 and clock < 900:
                    pr_ += max(-0.15, min(0.25, -0.012 * lead))
                else:
                    pr_ += max(-0.04, min(0.06, -0.004 * lead))
                pr_ = min(0.88, max(0.25, pr_))
                tm = 0.0
                turnover = None
                late_lead = half == 2 and clock < 480 and lead > 0
                hurry = (half == 2 and clock < 480 and lead < 0) or (half == 1 and clock <= 1920) or (half == 2 and clock <= 120 and lead <= 0)
                if rnd() < pr_:
                    if rnd() < o["scr_p"]:
                        yds = int(round(ps_[int(rnd() * nps)] * o["run_scale"] * rzm))
                        yds = min(yds, y)
                        A[0][RA] += 1; A[0][RY] += yds
                        ryd[pos] += yds
                        td = yds >= y
                        if td:
                            A[0][RT] += 1
                        tm = SCR_T
                        kind = "scr"
                    elif rnd() < o["sack_p"] * SIT["sack"][st]:
                        yds = pk[int(rnd() * npk)]
                        A[0][SK] += 1
                        pyd[pos] += yds
                        td = False
                        tm = SACK_T
                        kind = "sack"
                    else:
                        j = bisect(rcum, rnd())
                        j = min(j, len(ridx) - 1)
                        pi = ridx[j]
                        rf = pl[pi]["rec"]
                        A[0][PA] += 1
                        A[pi][TG] += 1
                        td = False
                        if rnd() < o["int_p"] * SIT["int"][st]:
                            A[0][PI] += 1
                            yds = 0
                            kind = "int"
                            tm = INC_T + 4
                        else:
                            pcatch = min(0.92, max(0.15, o["comp_p"] * SIT["comp"][st] * rf["catch_f"]))
                            if rnd() < pcatch:
                                yds = int(round(pc[int(rnd() * npc)] * o["ypc_scale"] * rzm * rf["ypr_f"]))
                                yds = min(yds, y)
                                A[0][PC] += 1; A[0][PY] += yds
                                A[pi][RC] += 1; A[pi][RYD] += yds
                                pyd[pos] += yds
                                td = yds >= y
                                if td:
                                    A[0][PT] += 1; A[pi][RTD] += 1
                                kind = "comp"
                                tm = COMP_T
                            else:
                                yds = 0
                                kind = "inc"
                                tm = INC_T
                else:
                    if cidx:
                        j = bisect(ccum, rnd())
                        j = min(j, len(cidx) - 1)
                        ci = cidx[j]
                        cf = pl[ci]["rush"]["ypc_f"]
                    else:
                        ci, cf = 0, 1.0
                    yds = int(round(pr[int(rnd() * npr)] * o["run_scale"] * rzm * cf))
                    yds = min(yds, y)
                    A[ci][RA] += 1; A[ci][RY] += yds
                    ryd[pos] += yds
                    td = yds >= y
                    if td:
                        A[ci][RT] += 1
                    kind = "run"
                    tm = RUN_T
                # fumbles
                if kind in ("run", "scr", "comp") and not td:
                    if rnd() < (const["fum_run"] if kind != "comp" else const["fum_comp"]):
                        turnover = "fum"
                plays[pos] += 1
                # clock
                if late_lead:
                    tm = max(tm, RUN_T + 4) if kind in ("run", "scr") else tm + 6
                elif hurry:
                    tm = tm * 0.55
                clock -= (tm + rnd() * 6 - 3) * spp
                # ----- resolve -----
                if kind == "int":
                    tov[pos] += 1
                    if rnd() < const["pick6"]:
                        score[dfn_i] += 6 + (1 if rnd() < 0.94 else 0)
                        res, npos, nytg = "pick6", pos, kick()
                    else:
                        res, nytg = "int", min(80, max(1, 100 - y))
                    break
                if turnover:
                    tov[pos] += 1
                    ynew = y - yds
                    res, nytg = "fum", min(99, max(1, 100 - ynew))
                    break
                if td:
                    score[pos] += 6 + (1 if rnd() < 0.94 else 0)
                    res, nytg = "td", kick()
                    break
                y = min(99, y - yds)
                if yds >= togo:
                    down = 1
                    togo = 10 if y > 10 else y
                else:
                    down += 1
                    togo -= yds
                    if down > 4:
                        res, nytg = "downs", min(99, max(1, 100 - y))
                        break
                if (half == 1 and clock <= 1800) or (half == 2 and clock <= 0) or (ot and clock <= 0):
                    res = "clock"
                    break

            # ----- after the drive -----
            if half == 1 and clock <= 1800:
                half = 2
                clock = 1800.0
                pos, ytg = second, kick()
                continue
            if half == 2 and clock <= 0 and not ot:
                if score[0] != score[1]:
                    break
                ot, clock, ot_poss = True, 600.0, 0
                pos, ytg = (0 if rnd() < 0.5 else 1), kick()
                continue
            if ot:
                ot_poss += 1
                if clock <= 0:
                    break
                if res == "td" and ot_poss == 1:
                    break
                if ot_poss >= 2 and score[0] != score[1]:
                    break
            if res == "clock":
                continue
            if res in ("td", "fg", "punt") and rnd() < RET_TD:  # kick / punt return touchdown
                score[npos] += 6 + (1 if rnd() < 0.94 else 0)
                npos, nytg = 1 - npos, kick()
            pos, ytg = npos, nytg

        return score, plays, tov, pyd, ryd, acc, draws, ot

    def run(self, n):
        res = dict(score=np.zeros((n, 2), dtype=np.int16), plays=np.zeros((n, 2), dtype=np.int16),
                   tov=np.zeros((n, 2), dtype=np.int16), pyd=np.zeros((n, 2), dtype=np.int16),
                   ryd=np.zeros((n, 2), dtype=np.int16), ot=np.zeros(n, dtype=bool))
        stats = [np.zeros((n, len(self.T[t]["players"]), NS), dtype=np.int16) for t in (0, 1)]
        active = [np.zeros((n, len(self.T[t]["players"])), dtype=bool) for t in (0, 1)]
        for s in range(n):
            sc, pl, tv, py, ry, acc, draws, ot = self.sim_one()
            res["score"][s] = sc; res["plays"][s] = pl; res["tov"][s] = tv
            res["pyd"][s] = py; res["ryd"][s] = ry; res["ot"][s] = ot
            for t in (0, 1):
                stats[t][s] = acc[t]
                active[t][s] = draws[t][0]
        res["stats"], res["active"] = stats, active
        return res
