"""Who is on the field and how the ball is shared.

ROSTER TRUTH: the current team for every player comes from nflverse rosters + the newest
depth chart, NOT from where he used to play. Players on IR / cut / retired are dropped,
traded players move with their stats, and rookies with a depth-chart role get a small
placeholder share. Usage (targets, carries, catch rate, yards) is recency-weighted and
shrunk toward the position average, then normalised so team totals are set by team ratings."""
import json
import pandas as pd
from config import ROOT, MODEL
from ratings import prep, add_weights

OUT_STATUS = {"Out", "Doubtful"}
BAD_ROSTER = {"RES", "CUT", "RET", "EXE", "TRD", "TRC", "TRT", "DEV"}  # not available to play


def load_overrides():
    p = ROOT / "overrides.json"
    j = json.loads(p.read_text()) if p.exists() else {}
    return {"out": j.get("out", []), "in": j.get("in", []), "qb": j.get("qb", {}), "team": j.get("team", {})}


def out_persistence(inj, rosters, as_of):
    """Measured from history: of players listed Out/Doubtful who are still on the active roster next report,
    what share are Out again? (~0.4 in 2024-26 data). Only uses weeks before as_of."""
    season, week = as_of
    cut = season * 100 + week
    i = inj[(inj.game_type == "REG") & (inj.season * 100 + inj.week < cut)].copy()
    ro = rosters[rosters.season * 100 + rosters.week < cut]
    st = ro.drop_duplicates(["season", "week", "gsis_id"]).set_index(["season", "week", "gsis_id"]).status.to_dict()
    weeks_by = i.groupby(["season", "team"]).week.apply(lambda s: sorted(set(s))).to_dict()
    out = i[i.report_status.isin(OUT_STATUS)].groupby(["season", "team", "gsis_id"]).week.apply(set).to_dict()
    hit = tot = 0
    for (s, t, pid), wks in out.items():
        ws = weeks_by[(s, t)]
        for k in range(1, len(ws)):
            if ws[k - 1] in wks and st.get((s, ws[k], pid)) in ("ACT", "INA"):
                tot += 1
                hit += ws[k] in wks
    return hit / tot if tot >= 200 else 0.41


def availability(inj, season, week, persist=0.41):
    """-> ({gsis_id: chance he plays}, notes).  Works team by team because the weekly report arrives in pieces.
    Game designation wins (Out/Doubtful 0, Questionable ~0.8). Before one is issued, practice participation is used
    (did not practice 0.35, limited 0.9). A team with no report yet inherits last week's Out/Doubtful list."""
    i = inj[(inj.season == season) & (inj.game_type == "REG") & (inj.week <= week)]
    pplay, carried, stale = {}, [], set()
    if i.empty:
        return pplay, ["no injury data"]
    q = MODEL["q_play_prob"]
    for team, t in i.groupby("team"):
        wk = t.week.max()
        cur = t[t.week == wk]
        if wk == week:
            for r in cur.itertuples():
                rs, ps_ = r.report_status, r.practice_status
                if rs in OUT_STATUS:
                    pr = 0.0
                elif rs == "Questionable":
                    pr = 0.6 if ps_ == "Did Not Participate In Practice" else q
                elif isinstance(ps_, str) and ps_.startswith("Did Not"):
                    pr = 0.35
                elif isinstance(ps_, str) and ps_.startswith("Limited"):
                    pr = 0.9
                else:
                    pr = 1.0
                if pr < 1.0:
                    pplay[r.gsis_id] = pr
        else:
            carried.append(team)
            for r in cur[cur.report_status.isin(OUT_STATUS)].itertuples():
                pplay[r.gsis_id] = round(1 - persist, 2)   # stale Out: measured chance he is back
                stale.add(r.gsis_id)
    note = (f"Week {week} injury report not out yet for {', '.join(sorted(carried))}; players last listed Out get a "
            f"{round((1 - persist) * 100)}% chance to play (measured from history)." if carried else "")
    return pplay, [note] if note else [], stale


def roster_context(rosters, depth, as_of):
    """Latest roster row per player at or before this week (+ depth chart if given)."""
    season, week = as_of
    r = rosters[(rosters.season == season) & (rosters.week <= week)]
    if "game_type" in r.columns:
        r = r[r.game_type == "REG"]
    r = r.sort_values("week").drop_duplicates("gsis_id", keep="last").set_index("gsis_id")
    ctx = {"team": r.team.to_dict(), "status": r.status.to_dict(), "pos": r.position.to_dict(),
           "name": r.full_name.to_dict(), "depth": {}, "depth_team": {}}
    if depth is not None and len(depth):
        for row in depth.itertuples():
            ctx["depth_team"][row.gsis_id] = row.team
            ctx["depth"].setdefault(row.gsis_id, {})[row.pos_abb] = int(row.pos_rank)
        ctx["depth_date"] = str(depth.dt.iloc[0])[:10]
    return ctx


def build_players(pbp, ps, inj, rosters, depth, as_of, overrides=None):
    """Returns {team: {'qb', 'players', 'note', 'flags'}} for every team with data."""
    ov = overrides or load_overrides()
    season, week = as_of
    ctx = roster_context(rosters, depth, as_of)
    last = ps.sort_values(["season", "week"]).drop_duplicates("player_id", keep="last")
    names = {**dict(zip(last.player_id, last.player_display_name)), **ctx["name"]}
    pos_ps = dict(zip(last.player_id, last.position))
    posf = lambda i: ctx["pos"].get(i) or pos_ps.get(i) or "UNK"
    n2id = {v: k for k, v in names.items()}
    ov_out = {n2id[n] for n in ov["out"] if n in n2id}
    ov_in = {n2id[n] for n in ov["in"] if n in n2id}
    ov_team = {n2id[n]: t for n, t in ov["team"].items() if n in n2id}
    pplay, notes, stale = availability(inj, season, week, out_persistence(inj, rosters, as_of))
    note = notes[0] if notes else ""
    for i in ov_out:
        pplay[i] = 0.0
    for i in ov_in:
        pplay.pop(i, None)
    out_ids = {i for i, v in pplay.items() if v <= 0}
    shaky = {i for i, v in pplay.items() if v < 0.5} | stale  # won't be picked as the starting QB

    d = add_weights(prep(pbp, as_of))
    # ---------- current team + eligibility, per player ----------
    def team_now(i):
        return ov_team.get(i) or ctx["depth_team"].get(i) or ctx["team"].get(i)

    def eligible(i):
        if i in ov_in or i in ov_team:
            return True
        st = ctx["status"].get(i)
        if i in ctx["depth_team"]:           # on the newest depth chart
            return st not in {"RES", "RET", "EXE"}
        return st not in BAD_ROSTER and st is not None

    # ---------- usage aggregated per PLAYER (across any team he has played for) ----------
    tg = d[(d.dropback == 1) & (d.sack == 0) & d.receiver_player_id.notna()].copy()
    tg["yds"] = tg.yards_gained * tg.complete_pass
    rg = pd.DataFrame({"tgt": tg.groupby("receiver_player_id").w_o.sum(),
                       "rec": (tg.complete_pass * tg.w_o).groupby(tg.receiver_player_id).sum(),
                       "yds": (tg.yds * tg.w_o).groupby(tg.receiver_player_id).sum()})
    rs = d[(d.play_type == "run") & (d.scr == 0) & d.rusher_player_id.notna()]
    cg = pd.DataFrame({"car": rs.groupby("rusher_player_id").w_o.sum(),
                       "yds": (rs.yards_gained * rs.w_o).groupby(rs.rusher_player_id).sum()})
    qbw = d[d.dropback == 1].groupby("passer_player_id").w_o.sum()
    qbw_recent = d[(d.dropback == 1) & (d.w_o >= MODEL["decay"] ** 3)].groupby("passer_player_id").w_o.sum()
    last_team = pd.concat([d[["receiver_player_id", "posteam"]].rename(columns={"receiver_player_id": "i"}),
                           d[["rusher_player_id", "posteam"]].rename(columns={"rusher_player_id": "i"})]).dropna().groupby("i").posteam.last().to_dict()
    for i, t in d[d.dropback == 1].groupby("passer_player_id").posteam.last().items():
        last_team.setdefault(i, t)

    # league baselines by position (all teams)
    rg["pos"] = [posf(i) for i in rg.index]
    cg["pos"] = [posf(i) for i in cg.index]
    pos_catch = (rg.groupby("pos").rec.sum() / rg.groupby("pos").tgt.sum()).to_dict()
    pos_ypr = (rg.groupby("pos").yds.sum() / rg.groupby("pos").rec.sum().clip(lower=1e-9)).to_dict()
    pos_ypc = (cg.groupby("pos").yds.sum() / cg.groupby("pos").car.sum()).to_dict()
    lg_catch, lg_ypr, lg_ypc = rg.rec.sum() / rg.tgt.sum(), rg.yds.sum() / rg.rec.sum(), cg.yds.sum() / cg.car.sum()
    team_tgt = tg.groupby("posteam").w_o.sum().to_dict()

    roster_by_team = {}
    for i in set(ctx["team"]) | set(ctx["depth_team"]) | set(ov_team):
        t = team_now(i)
        if t:
            roster_by_team.setdefault(t, []).append(i)

    result = {}
    for team in sorted(set(d.posteam)):
        ids = roster_by_team.get(team, [])
        ok = [i for i in ids if eligible(i) and i not in out_ids]
        flags = []

        # ---------- quarterback ----------
        qb_c = [i for i in ok if posf(i) == "QB" and (qbw.get(i, 0) > 0 or (ctx["depth"].get(i, {}).get("QB", 99) <= 3))]
        healthy = [i for i in qb_c if i not in shaky]
        qb_c = healthy or qb_c
        forced = ov["qb"].get(team)
        dep = lambda i: ctx["depth"].get(i, {}).get("QB", 99)
        qb_c.sort(key=lambda i: (dep(i), -qbw_recent.get(i, 0), -qbw.get(i, 0)) if ctx["depth_team"] else (-qbw_recent.get(i, 0), -qbw.get(i, 0)))
        if forced and forced in n2id:
            qb_c = [n2id[forced]] + [i for i in qb_c if i != n2id[forced]]
        if not qb_c:
            continue
        qb = qb_c[0]
        for i in ids:  # notable QBs not playing
            if posf(i) == "QB" and i != qb and qbw.get(i, 0) > 3:
                why = "injury report: out" if i in out_ids else ("not practicing" if i in shaky else ("IR" if ctx["status"].get(i) == "RES" else "backup"))
                if why != "backup":
                    flags.append(f"{names.get(i, i)} (QB) excluded - {why}")

        players = {}

        def get(pid):
            if pid not in players:
                players[pid] = {"id": pid, "name": names.get(pid, pid), "pos": posf(pid),
                                "p_play": pplay.get(pid, 1.0),
                                "rec": None, "rush": None, "new": False}
            return players[pid]
        get(qb)["pos"] = "QB"

        cand = [i for i in ok if i != qb]
        # ---------- receivers ----------
        rc = rg[rg.index.isin(cand)]
        tot = rc.tgt.sum()
        rc = rc[rc.tgt / max(tot, 1e-9) >= 0.015] if tot > 0 else rc
        rec_share = {}
        if len(rc):
            share = rc.tgt / rc.tgt.sum()
            catch_s, ypr_s = {}, {}
            for pid, r in rc.iterrows():
                pc, py = pos_catch.get(r.pos, lg_catch), pos_ypr.get(r.pos, lg_ypr)
                catch_s[pid] = (r.rec + 30 * pc) / (r.tgt + 30)
                ypr_s[pid] = (r.yds + 15 * py) / (r.rec + 15)
            C = sum(share[p] * catch_s[p] for p in share.index)
            Y = sum(share[p] * catch_s[p] * ypr_s[p] for p in share.index) / C
            for pid in share.index:
                rec_share[pid] = {"share": float(share[pid]), "catch_f": catch_s[pid] / C, "ypr_f": ypr_s[pid] / Y}
        # ---------- rushers ----------
        cc = cg[cg.index.isin(cand)]
        tot = cc.car.sum()
        cc = cc[cc.car / max(tot, 1e-9) >= 0.02] if tot > 0 else cc
        rush_share = {}
        if len(cc):
            share = cc.car / cc.car.sum()
            ypc_s = {pid: (r.yds + 40 * pos_ypc.get(r.pos, lg_ypc)) / (r.car + 40) for pid, r in cc.iterrows()}
            Z = sum(share[p] * ypc_s[p] for p in share.index)
            for pid in share.index:
                rush_share[pid] = {"share": float(share[pid]), "ypc_f": ypc_s[pid] / Z}

        # ---------- rookies / new faces with a depth-chart role but no history ----------
        for i in cand:
            if rg.tgt.get(i, 0) > 0.5 or cg.car.get(i, 0) > 0.5 or i in rec_share or i in rush_share:
                continue
            dpos = ctx["depth"].get(i, {})
            pos = posf(i)
            if pos in ("WR", "TE") and min([dpos.get(pos, 99)] + [99]) <= 3:
                rec_share[i] = {"share": 0.04, "catch_f": 1.0, "ypr_f": 1.0, "_new": True}
            elif pos == "RB" and dpos.get("RB", 99) <= 2:
                rush_share[i] = {"share": 0.10, "ypc_f": 1.0, "_new": True}
                rec_share[i] = {"share": 0.03, "catch_f": 1.0, "ypr_f": 1.0, "_new": True}
        for table in (rec_share, rush_share):  # renormalise after placeholders
            z = sum(v["share"] for v in table.values())
            for v in table.values():
                v["share"] = v["share"] / z if z else v["share"]
        for pid, v in rec_share.items():
            get(pid)["rec"] = {k: x for k, x in v.items() if k != "_new"}
            get(pid)["new"] |= bool(v.get("_new"))
        for pid, v in rush_share.items():
            get(pid)["rush"] = {k: x for k, x in v.items() if k != "_new"}
            get(pid)["new"] |= bool(v.get("_new"))

        # ---------- roster flags: who left / arrived / is out ----------
        used = set(players)
        for i, lt in last_team.items():
            if i in used or lt != team:
                continue
            share = max(rg.tgt.get(i, 0) / max(team_tgt.get(team, 1), 1e-9), 0)
            car_big = cg.car.get(i, 0) > 25
            if share < 0.07 and not car_big:
                continue
            nt = team_now(i)
            if i in out_ids:
                flags.append(f"{names.get(i, i)} ({posf(i)}) excluded - injury report: out")
            elif ctx["status"].get(i) == "RES" and nt == team:
                flags.append(f"{names.get(i, i)} ({posf(i)}) excluded - injured reserve")
            elif nt and nt != team:
                flags.append(f"{names.get(i, i)} ({posf(i)}) now on {nt}")
            elif ctx["status"].get(i) in BAD_ROSTER or nt is None:
                flags.append(f"{names.get(i, i)} ({posf(i)}) excluded - not on active roster")
        for i in used:
            big = (players[i]["rec"] or {"share": 0})["share"] >= 0.08 or (players[i]["rush"] or {"share": 0})["share"] >= 0.12
            if i != qb and big and last_team.get(i) and last_team[i] != team:
                flags.append(f"{names.get(i, i)} ({posf(i)}) joined from {last_team[i]}")
        for pid, pp in players.items():
            sh = max((pp["rec"] or {"share": 0})["share"], (pp["rush"] or {"share": 0})["share"])
            if pid != qb and pp["p_play"] < 1 and sh >= 0.08:
                flags.append(f"{pp['name']} ({pp['pos']}) {round(pp['p_play'] * 100)}% to play")
        pl = [p for p in players.values() if p["id"] == qb or p["rec"] or p["rush"]]
        pl.sort(key=lambda p: (p["id"] != qb, -(p["rec"] or {"share": 0})["share"] - (p["rush"] or {"share": 0})["share"]))
        result[team] = {"qb": qb, "players": pl, "note": note, "flags": flags[:10]}
    return result
