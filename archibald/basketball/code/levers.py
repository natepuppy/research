"""New levers a coach controls, measured from the rebuilt possessions and the
play-by-play. Every comparison is also run on even- and odd-numbered games.

  crash          the rebounding result again on every loaded game
  two_for_one    shoot quickly with 28-36 s left in a quarter, or hold
  bonus          what being in the penalty is worth, and how early teams get there
  after_oreb     putback now, kick-out three, or reset
  take_fouls     fouling the break
  push_vs_crash  is the push worth more against teams that crash
  push_by_period does the push gain fade late in the game
  run_timeouts   does a timeout stop a run
  heaves         end-of-quarter possessions with no shot
  challenges     when challenges are used and how often they win
  lead_pace      slowing down with a fourth-quarter lead

Writes results/levers/summary.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoreboard import clock_seconds, period_len  # noqa: E402

DSN = "postgresql://nathanclark@127.0.0.1:5432/nba"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "levers"
EARLY = 8.0
TEAM_FOULS = {"Shooting", "Personal", "Loose Ball", "Personal Take", "Transition Take", "Away From Play",
              "Clear Path", "Flagrant Type 1", "Flagrant Type 2", "Double Personal"}


def read(sql: str) -> pd.DataFrame:
    with psycopg.connect(DSN) as conn:
        return pd.read_sql(sql, conn)


def ols(X, y):
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    y = np.asarray(y, float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    s2 = float(r @ r) / max(len(y) - X.shape[1], 1)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return b, se


def demean(df, cols, by):
    out = df.copy()
    for c in cols:
        out[c] = df[c] - df.groupby(by)[c].transform("mean")
    return out


def half(game_id: pd.Series) -> pd.Series:
    return np.where(game_id.str[-1].astype(int) % 2 == 0, "even", "odd")


def mean_se(x) -> dict:
    x = pd.Series(x).dropna()
    return {"mean": float(x.mean()) if len(x) else None, "se": float(x.sem()) if len(x) > 1 else None, "n": int(len(x))}


def compare(df, flag, y, label_true, label_false) -> dict:
    """Mean of y when flag is true vs false, overall and per half."""
    out = {}
    for h, d in [("all", df)] + [(k, g) for k, g in df.groupby("half")]:
        a, b = d.loc[d[flag], y], d.loc[~d[flag], y]
        out[h] = {label_true: mean_se(a), label_false: mean_se(b), "gap": float(a.mean() - b.mean()),
                  "noise": float(np.sqrt(a.sem() ** 2 + b.sem() ** 2)) if len(a) > 1 and len(b) > 1 else None}
    return out


def load_possessions() -> pd.DataFrame:
    p = pd.read_csv(ROOT / "results" / "scoreboard" / "possessions.csv", dtype={"game_id": str})
    p["half"] = half(p["game_id"])
    p["pushed"] = p["seconds_to_first_action"] <= EARLY
    p["opp"] = p["side"].map({"h": "v", "v": "h"})
    p = p.sort_values(["game_id", "period", "start_time"]).reset_index(drop=True)
    return p


def clean(p: pd.DataFrame) -> pd.DataFrame:
    live = p[p["ended_by"] != "period_end"]
    junk = ((live["period"] >= 4) & (live["start_remaining"] < 180)) | (live["margin_at_start"].abs() > 20)
    return live[~junk].copy()


def load_pbp(schema: str = "games") -> pd.DataFrame:
    where = "WHERE game_id IN (SELECT game_id FROM games.loaded)" if schema == "games" else ""
    pbp = read(
        f"""
        SELECT game_id, action_number, action_id, period, clock, team_id, person_id, location,
               action_type, sub_type, shot_value, shot_distance, shot_result, description,
               score_home, score_away
        FROM {schema}.pbp {where}
        """
    )
    pbp = pbp.sort_values(["game_id", "action_number", "action_id"], kind="stable")
    for c in ("score_home", "score_away"):
        pbp[c] = pd.to_numeric(pbp[c], errors="coerce")
        pbp[c] = pbp.groupby("game_id")[c].ffill().fillna(0)
    pbp["remaining"] = pbp["clock"].map(clock_seconds)
    pbp["elapsed"] = [period_len(p) - r for p, r in zip(pbp["period"], pbp["remaining"])]
    pbp["half"] = half(pbp["game_id"])
    return pbp.sort_values(["game_id", "action_number", "action_id"], kind="stable").reset_index(drop=True)


def team_games() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results" / "scoreboard" / "team_games.csv", dtype={"game_id": str})


# ---------------------------------------------------------------- crash, again

def pos_class(p: str) -> str:
    p = str(p)
    if p in ("C", "C-F", "F-C"):
        return "big"
    if p in ("G", "G-F"):
        return "guard"
    if p in ("F", "F-G"):
        return "forward"
    return "unknown"


def crash(tg: pd.DataFrame) -> dict:
    roster = read("SELECT player_id, position FROM league.rosters").drop_duplicates("player_id")
    roster["cls"] = roster["position"].map(pos_class)
    tp = read("SELECT game_id, team_tricode, person_id, rebound_chances_offensive AS och FROM games.box_tracking_player WHERE minutes LIKE '%:%'")
    tp = tp.merge(roster[["player_id", "cls"]], left_on="person_id", right_on="player_id", how="left")
    tp["cls"] = tp["cls"].fillna("unknown")
    pv = tp.pivot_table(index=["game_id", "team_tricode"], columns="cls", values="och", aggfunc="sum", fill_value=0).reset_index()
    pv.columns = ["game_id", "team_tricode"] + [f"och_{c}" for c in pv.columns[2:]]
    trad = read("SELECT game_id, team_tricode, field_goals_attempted - field_goals_made AS misses FROM games.box_traditional_team_1")
    misc = read("SELECT game_id, team_tricode, opp_points_fast_break AS opp_fb FROM games.box_misc_team")
    f = tg.merge(pv, on=["game_id", "team_tricode"]).merge(trad, on=["game_id", "team_tricode"]).merge(misc, on=["game_id", "team_tricode"], how="left")
    for c in ("och_big", "och_forward", "och_guard"):
        if c not in f.columns:
            f[c] = 0.0
    f = f[f["misses"] > 0].copy()
    f["crash_per_miss"] = (f["och_big"] + f["och_forward"] + f["och_guard"]) / f["misses"]
    for c in ("big", "forward", "guard"):
        f[f"{c}_per_miss"] = f[f"och_{c}"] / f["misses"]
    f["opp_fb_per_100"] = 100 * f["opp_fb"] / f["possessions"]
    f["half"] = half(f["game_id"])
    cols = ["crash_per_miss", "big_per_miss", "forward_per_miss", "guard_per_miss", "oreb_pct", "opp_fb_per_100",
            "offensive_rating", "defensive_rating", "net_rating", "d_ppp_after_dreb"]
    f = f.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
    out = {"team_games": int(len(f)), "mean_crash_per_miss": float(f["crash_per_miss"].mean()),
           "sd_within_team": float(demean(f, ["crash_per_miss"], "team_tricode")["crash_per_miss"].std())}
    for h, d in [("all", f)] + [(k, g) for k, g in f.groupby("half")]:
        dm = demean(d, cols, "team_tricode")
        blk = {}
        for y in ("oreb_pct", "offensive_rating", "defensive_rating", "net_rating", "opp_fb_per_100", "d_ppp_after_dreb"):
            b, se = ols(dm[["crash_per_miss"]].to_numpy(), dm[y].to_numpy())
            blk[y] = {"per_crasher": round(float(b[1]), 3), "se": round(float(se[1]), 3)}
        X = dm[["big_per_miss", "forward_per_miss", "guard_per_miss"]].to_numpy()
        pos = {}
        for y in ("oreb_pct", "defensive_rating", "net_rating"):
            b, se = ols(X, dm[y].to_numpy())
            pos[y] = {k: {"coef": round(float(b[i + 1]), 3), "se": round(float(se[i + 1]), 3)} for i, k in enumerate(("big", "forward", "guard"))}
        out[h] = {"total": blk, "by_position": pos, "team_games": int(len(d))}
    return out


# ---------------------------------------------------------------- two for one

def two_for_one(p: pd.DataFrame, teams: bool = True) -> dict:
    d = p[(p["period"] <= 3) & (p["start_remaining"].between(28, 36)) & (p["margin_at_start"].abs() <= 20)].copy()
    # points for the offense and the opponent over the rest of the period, from this possession on
    rest = []
    grp = p.groupby(["game_id", "period"])
    for idx, row in d.iterrows():
        g = grp.get_group((row["game_id"], row["period"]))
        later = g[g["start_time"] >= row["start_time"]]
        own = later.loc[later["side"] == row["side"], "points"].sum()
        opp = later.loc[later["side"] != row["side"], "points"].sum()
        rest.append((own, opp, len(later)))
    d["rest_own"], d["rest_opp"], d["rest_n"] = zip(*rest)
    d["rest_net"] = d["rest_own"] - d["rest_opp"]
    d["quick"] = d["seconds_to_first_action"] <= 8
    d["held"] = d["seconds_to_first_action"] >= 12
    both = d[d["quick"] | d["held"]].copy()
    out = {"candidates": int(len(d)), "quick_share": float(d["quick"].mean())}
    out["rest_of_period_net"] = compare(both, "quick", "rest_net", "quick", "held")
    out["this_possession_ppp"] = compare(both, "quick", "points", "quick", "held")
    out["opponent_points_rest"] = compare(both, "quick", "rest_opp", "quick", "held")
    out["own_points_rest"] = compare(both, "quick", "rest_own", "quick", "held")
    # per team: how often they go quick in this window
    if teams:
        tg = team_games()[["game_id", "side", "team_tricode"]]
        d = d.merge(tg, on=["game_id", "side"], how="left")
        out["team_quick_rate"] = d.groupby("team_tricode")["quick"].agg(["mean", "count"]).round(3).sort_values("mean").to_dict("index")
    return out


# ---------------------------------------------------------------- bonus

def bonus(p: pd.DataFrame, pbp: pd.DataFrame, teams: bool = True) -> dict:
    fouls = pbp[(pbp["action_type"] == "Foul") & (pbp["sub_type"].isin(TEAM_FOULS)) & (pbp["location"].isin(["h", "v"]))].copy()
    fouls = fouls.sort_values(["game_id", "period", "elapsed"])
    fouls["k"] = fouls.groupby(["game_id", "period", "location"]).cumcount() + 1
    fifth = fouls[fouls["k"] == 5][["game_id", "period", "location", "elapsed"]].rename(columns={"location": "opp", "elapsed": "penalty_at"})
    d = p.merge(fifth, on=["game_id", "period", "opp"], how="left")
    d["in_bonus"] = d["penalty_at"].notna() & (d["start_time"] >= d["penalty_at"])
    live = clean(d)
    live = live[live["period"] <= 4]
    # control for when in the quarter: compare inside bins of time remaining
    live["bin"] = pd.cut(live["start_remaining"], [0, 120, 240, 360, 480, 720], labels=["0-2", "2-4", "4-6", "6-8", "8-12"])
    out = {"possessions": int(len(live)), "share_in_bonus": float(live["in_bonus"].mean())}
    rows = []
    for (b, h), g in live.groupby(["bin", "half"], observed=True):
        a, c = g.loc[g["in_bonus"], "points"], g.loc[~g["in_bonus"], "points"]
        if len(a) > 50 and len(c) > 50:
            rows.append({"min_remaining": str(b), "half": h, "ppp_bonus": float(a.mean()), "ppp_not": float(c.mean()), "gap": float(a.mean() - c.mean()), "n_bonus": int(len(a)), "n_not": int(len(c))})
    out["by_time_remaining"] = rows
    w = pd.DataFrame(rows)
    out["lift_weighted"] = float(np.average(w["gap"], weights=w["n_bonus"]))
    for h, g in w.groupby("half"):
        out[f"lift_{h}"] = float(np.average(g["gap"], weights=g["n_bonus"]))
    live["ft_trip"] = live["fta"] > 0
    out["ft_trip_rate"] = compare(live, "in_bonus", "ft_trip", "bonus", "not")["all"]
    # how early: penalty time per team-quarter, and minutes spent in bonus per game on offense and defense
    pen = fifth.copy()
    pen["minutes_left"] = pen.apply(lambda r: (period_len(r["period"]) - r["penalty_at"]) / 60, axis=1)
    out["quarters_reaching_penalty"] = float(len(pen) / p.groupby(["game_id", "period"]).ngroups / 2)
    out["minutes_left_when_penalty_starts"] = mean_se(pen["minutes_left"])
    if not teams:
        return out
    tg = team_games()[["game_id", "side", "team_tricode", "net_rating", "offensive_rating", "defensive_rating", "ftr"]]
    # offense side = the team whose opponent fouled: fouling side is pen.opp, so the beneficiary is the other side
    pen["side"] = pen["opp"].map({"h": "v", "v": "h"})
    off_min = pen.groupby(["game_id", "side"])["minutes_left"].sum().rename("bonus_min_off").reset_index()
    def_min = pen.rename(columns={"opp": "side2"}).groupby(["game_id", "side2"])["minutes_left"].sum().rename("bonus_min_def").reset_index().rename(columns={"side2": "side"})
    t = tg.merge(off_min, on=["game_id", "side"], how="left").merge(def_min, on=["game_id", "side"], how="left").fillna({"bonus_min_off": 0, "bonus_min_def": 0})
    t["half"] = half(t["game_id"])
    out["minutes_in_bonus_per_game"] = {"offense": mean_se(t["bonus_min_off"]), "defense": mean_se(t["bonus_min_def"])}
    res = {}
    for h, g in [("all", t)] + [(k, v) for k, v in t.groupby("half")]:
        dm = demean(g, ["bonus_min_off", "bonus_min_def", "net_rating", "offensive_rating", "defensive_rating"], "team_tricode")
        b, se = ols(dm[["bonus_min_off", "bonus_min_def"]].to_numpy(), dm["net_rating"].to_numpy())
        bo, seo = ols(dm[["bonus_min_off"]].to_numpy(), dm["offensive_rating"].to_numpy())
        res[h] = {"net_per_offense_bonus_minute": round(float(b[1]), 3), "se": round(float(se[1]), 3),
                  "net_per_defense_bonus_minute": round(float(b[2]), 3), "se_def": round(float(se[2]), 3),
                  "off_rating_per_offense_bonus_minute": round(float(bo[1]), 3), "se_off": round(float(seo[1]), 3)}
    out["within_team"] = res
    out["team_offense_bonus_minutes"] = t.groupby("team_tricode")["bonus_min_off"].mean().round(2).sort_values().to_dict()
    out["team_defense_bonus_minutes"] = t.groupby("team_tricode")["bonus_min_def"].mean().round(2).sort_values().to_dict()
    return out


# ---------------------------------------------------------------- after an offensive rebound

def after_oreb(pbp: pd.DataFrame) -> dict:
    rows = []
    offensive = {"Made Shot", "Missed Shot", "Heave", "Turnover", "Free Throw"}
    for gid, g in pbp.groupby("game_id", sort=False):
        a = g["action_type"].to_numpy(); sub = g["sub_type"].fillna("").to_numpy(); loc = g["location"].fillna("").to_numpy()
        t = g["elapsed"].to_numpy(); per = g["period"].to_numpy(); val = g["shot_value"].fillna(0).to_numpy()
        dist = g["shot_distance"].to_numpy(); desc = g["description"].fillna("").to_numpy(); pid = g["person_id"].to_numpy(); tid = g["team_id"].to_numpy()
        remaining = g["remaining"].to_numpy()
        cur_side, pts, orebs, last_miss_side, cur_period = None, 0, [], None, None
        def flush():
            for o in orebs:
                o["points_after"] = pts - o["pts_at"]
                rows.append(o)
        for i in range(len(g)):
            if per[i] != cur_period:
                flush(); orebs = []; cur_side = None; pts = 0; cur_period = per[i]; last_miss_side = None
            k = a[i]
            if k == "Rebound":
                if not loc[i]:
                    continue
                if last_miss_side is not None and loc[i] == last_miss_side and cur_side == loc[i]:
                    orebs.append({"game_id": gid, "period": int(per[i]), "t": float(t[i]), "remaining": float(remaining[i]), "side": loc[i], "pts_at": pts,
                                  "team_rebound": bool(pid[i] == tid[i]), "next": None, "delta": np.nan, "zone": None, "made": np.nan,
                                  "half": "even" if int(gid[-1]) % 2 == 0 else "odd", "n_prev_oreb": len(orebs)})
                else:
                    flush(); orebs = []; cur_side = loc[i]; pts = 0
                last_miss_side = None
                continue
            if k not in offensive or not loc[i]:
                continue
            if k == "Free Throw" and "Technical" in sub[i]:
                continue
            if cur_side != loc[i]:
                flush(); orebs = []; cur_side = loc[i]; pts = 0
            # record the first action after each pending oreb that has none yet
            for o in orebs:
                if o["next"] is None:
                    o["next"] = k
                    o["delta"] = float(t[i] - o["t"])
                    if k in ("Made Shot", "Missed Shot", "Heave"):
                        v = int(val[i]) if val[i] else 2
                        o["zone"] = "three" if v == 3 else ("rim" if dist[i] < 5 else "other2")
                        o["made"] = int(k == "Made Shot")
            if k in ("Made Shot",):
                pts += int(val[i]) if val[i] else 2
                last_miss_side = None
            elif k == "Missed Shot" or k == "Heave":
                last_miss_side = loc[i]
            elif k == "Free Throw":
                if desc[i].startswith("MISS"):
                    last_miss_side = loc[i]
                else:
                    pts += 1
                    last_miss_side = None
            elif k == "Turnover":
                flush(); orebs = []; cur_side = None; pts = 0; last_miss_side = None
        flush()
    ev = pd.DataFrame(rows)
    ev = ev[ev["next"].notna() & (ev["remaining"] > 24)]  # ignore end-of-period scrambles
    ev["choice"] = np.select(
        [ev["next"] == "Turnover", (ev["next"] == "Free Throw"),
         (ev["delta"] <= 3) & (ev["zone"] == "rim"), (ev["delta"] <= 8) & (ev["zone"] == "three"),
         (ev["delta"] <= 8) & (ev["zone"] == "other2"), (ev["delta"] <= 8) & (ev["zone"] == "rim"), ev["delta"] > 8],
        ["turnover", "fouled", "putback_now", "kickout_three", "quick_other_two", "rim_within_8", "reset"], default="other")
    out = {"offensive_rebounds": int(len(ev)), "choice_mix": ev["choice"].value_counts(normalize=True).round(3).to_dict()}
    tab = []
    for (c, h), g in ev.groupby(["choice", "half"]):
        tab.append({"choice": c, "half": h, "points_after": float(g["points_after"].mean()), "se": float(g["points_after"].sem()), "n": int(len(g)), "fg_pct": float(g["made"].mean()) if g["made"].notna().any() else None})
    out["points_after_by_choice"] = tab
    allc = ev.groupby("choice")["points_after"].agg(["mean", "sem", "count"]).round(3)
    out["points_after_all"] = allc.to_dict("index")
    # reset: does the later shot's timing matter
    rs = ev[ev["delta"] > 8].copy()
    rs["bin"] = pd.cut(rs["delta"], [8, 12, 16, 24])
    out["reset_by_seconds"] = rs.groupby("bin", observed=True)["points_after"].agg(["mean", "count"]).round(3).reset_index().astype({"bin": str}).to_dict("records")
    # by whether the rebounder was near the rim (team rebounds excluded)
    return out


# ---------------------------------------------------------------- take fouls

def take_fouls(p: pd.DataFrame, pbp: pd.DataFrame) -> dict:
    tf = pbp[(pbp["action_type"] == "Foul") & (pbp["sub_type"].isin(["Personal Take", "Transition Take"]))].copy()
    tf["off"] = tf["location"].map({"h": "v", "v": "h"})
    live = clean(p).copy()
    hits = []
    grp = live.groupby(["game_id", "period", "side"])
    for _, r in tf.iterrows():
        key = (r["game_id"], r["period"], r["off"])
        if key not in grp.groups:
            continue
        g = live.loc[grp.groups[key]]
        m = g[(g["start_time"] <= r["elapsed"]) & (g["end_time"] >= r["elapsed"]) & (r["elapsed"] - g["start_time"] <= 10)]
        if len(m):
            hits.append(m.index[0])
    live["take_foul"] = False
    live.loc[hits, "take_foul"] = True
    live["fts"] = live["fta"] > 0
    out = {"take_fouls_matched": int(live["take_foul"].sum()), "take_fouls_total": int(len(tf)), "per_game": float(len(tf) / p["game_id"].nunique())}
    tfp = live[live["take_foul"]]
    out["start_type_mix"] = tfp["start_type"].value_counts().to_dict()
    out["share_followed_by_free_throws"] = float(tfp["fts"].mean())
    transition = live["start_type"].isin(["def_reb", "def_reb_ft", "steal"])
    out["ppp"] = {
        "after_take_foul_all": mean_se(tfp["points"]),
        "after_take_foul_no_free_throws": mean_se(tfp.loc[~tfp["fts"], "points"]),
        "after_take_foul_with_free_throws": mean_se(tfp.loc[tfp["fts"], "points"]),
        "transition_pushed_no_foul": mean_se(live.loc[~live["take_foul"] & transition & live["pushed"], "points"]),
        "transition_walked_no_foul": mean_se(live.loc[~live["take_foul"] & transition & ~live["pushed"], "points"]),
        "after_make_pushed_no_foul": mean_se(live.loc[~live["take_foul"] & live["start_type"].isin(["made_fg", "made_ft"]) & live["pushed"], "points"]),
        "after_make_walked_no_foul": mean_se(live.loc[~live["take_foul"] & live["start_type"].isin(["made_fg", "made_ft"]) & ~live["pushed"], "points"]),
    }
    out["by_half_no_free_throws"] = {h: mean_se(g.loc[g["take_foul"] & ~g["fts"], "points"]) for h, g in live.groupby("half")}
    return out


# ---------------------------------------------------------------- push vs opponent crash, push by period

def push_vs_crash(p: pd.DataFrame, tg: pd.DataFrame) -> dict:
    trad = read("SELECT game_id, team_tricode, field_goals_attempted - field_goals_made AS misses FROM games.box_traditional_team_1")
    c = tg[["game_id", "side", "team_tricode", "rebound_chances_offensive"]].merge(trad, on=["game_id", "team_tricode"])
    c["crash_per_miss"] = c["rebound_chances_offensive"] / c["misses"]
    c = c.rename(columns={"side": "opp"})[["game_id", "opp", "crash_per_miss"]]
    live = clean(p)
    d = live[live["start_type"] == "def_reb"].merge(c, on=["game_id", "opp"], how="inner")
    d["tercile"] = pd.qcut(d["crash_per_miss"], 3, labels=["opp crashes little", "middle", "opp crashes a lot"])
    out = {}
    for (t, h), g in d.groupby(["tercile", "half"], observed=True):
        a, b = g.loc[g["pushed"], "points"], g.loc[~g["pushed"], "points"]
        out.setdefault(str(t), {})[h] = {"pushed": float(a.mean()), "walked": float(b.mean()), "gap": float(a.mean() - b.mean()), "push_rate": float(g["pushed"].mean()), "n": int(len(g))}
    return out


def push_by_period(p: pd.DataFrame) -> dict:
    live = clean(p)
    d = live[live["start_type"].isin(["def_reb", "steal"])]
    out = {}
    for (s, per), g in d.groupby(["start_type", "period"]):
        if per > 4:
            continue
        a, b = g.loc[g["pushed"], "points"], g.loc[~g["pushed"], "points"]
        out[f"{s} Q{per}"] = {"pushed": round(float(a.mean()), 3), "walked": round(float(b.mean()), 3), "gap": round(float(a.mean() - b.mean()), 3), "push_rate": round(float(g["pushed"].mean()), 3), "n": int(len(g))}
    return out


# ---------------------------------------------------------------- timeouts and runs

def run_timeouts(p: pd.DataFrame) -> dict:
    rows = []
    for (gid, per), g in p.groupby(["game_id", "period"], sort=False):
        g = g.reset_index(drop=True)
        pts = g["points"].to_numpy(); side = g["side"].to_numpy(); tob = g["timeout_before_by"].fillna("").to_numpy()
        rem = g["start_remaining"].to_numpy(); mar = g["margin_at_start"].to_numpy()
        for i in range(6, len(g) - 6):
            s = side[i]
            prev = slice(i - 6, i)
            opp_pts = pts[prev][side[prev] != s].sum(); own_pts = pts[prev][side[prev] == s].sum()
            if opp_pts - own_pts < 8:
                continue
            if per >= 4 and rem[i] < 180:
                continue
            nxt = slice(i, i + 6)
            net_next = pts[nxt][side[nxt] == s].sum() - pts[nxt][side[nxt] != s].sum()
            rows.append({"game_id": gid, "timeout": tob[i] == "own", "run": int(opp_pts - own_pts), "net_next6": int(net_next), "margin": int(mar[i])})
    r = pd.DataFrame(rows)
    r["half"] = half(r["game_id"])
    out = {"runs": int(len(r)), "timeout_share": float(r["timeout"].mean())}
    out["net_next_6_possessions"] = compare(r, "timeout", "net_next6", "timeout", "no_timeout")
    # the run size is bigger when a timeout is called; match on run size
    bins = {}
    for k, g in r.groupby(pd.cut(r["run"], [7, 8, 10, 13, 40]), observed=True):
        if g["timeout"].sum() > 30 and (~g["timeout"]).sum() > 30:
            bins[str(k)] = {"timeout": mean_se(g.loc[g["timeout"], "net_next6"]), "no_timeout": mean_se(g.loc[~g["timeout"], "net_next6"])}
    out["by_run_size"] = bins
    return out


# ---------------------------------------------------------------- heaves

def heaves(p: pd.DataFrame, pbp: pd.DataFrame) -> dict:
    d = p[(p["period"] <= 3) & (p["start_remaining"].between(1.5, 5)) & (p["start_type"].isin(["def_reb", "made_fg", "made_ft", "steal", "dead_tov", "def_reb_ft"]))].copy()
    d["shot"] = (d["fga"] > 0) | (d["first_action"].isin(["Made Shot", "Missed Shot", "Heave"]))
    out = {"end_of_quarter_possessions": int(len(d)), "share_with_a_shot": float(d["shot"].mean()), "per_game": float(len(d) / p["game_id"].nunique())}
    out["points_when_shot"] = mean_se(d.loc[d["shot"], "points"])
    long_makes = pbp[(pbp["action_type"] == "Made Shot") & (pbp["shot_distance"] >= 40)]
    long_miss = pbp[((pbp["action_type"] == "Missed Shot") & (pbp["shot_distance"] >= 40)) | (pbp["action_type"] == "Heave")]
    out["shots_from_40_plus"] = {"made": int(len(long_makes)), "missed": int(len(long_miss)), "pct": float(len(long_makes) / max(len(long_makes) + len(long_miss), 1))}
    out["by_seconds"] = d.groupby(pd.cut(d["start_remaining"], [1.5, 2.5, 3.5, 5]), observed=True)["shot"].agg(["mean", "count"]).round(3).reset_index().astype({"start_remaining": str}).to_dict("records")
    tg = team_games()[["game_id", "side", "team_tricode"]]
    d = d.merge(tg, on=["game_id", "side"], how="left")
    out["team_shot_rate"] = d.groupby("team_tricode")["shot"].agg(["mean", "count"]).round(2).sort_values("mean").to_dict("index")
    return out


# ---------------------------------------------------------------- challenges

def challenges(pbp: pd.DataFrame) -> dict:
    ch = read(
        """
        SELECT r.game_id, r.period, r.clock, b.team_tricode AS tricode, r.sub_type
        FROM games.pbp r
        LEFT JOIN (SELECT DISTINCT team_id, team_tricode FROM games.box_traditional_team_1) b ON b.team_id = r.team_id
        WHERE r.action_type = 'Instant Replay' AND r.sub_type LIKE 'Coach Challenge%%'
          AND r.game_id IN (SELECT game_id FROM games.loaded)
        """
    )
    ch["remaining"] = ch["clock"].map(clock_seconds)
    ch["elapsed"] = [period_len(p) - r for p, r in zip(ch["period"], ch["remaining"])]
    ch["won"] = ch["sub_type"].str.contains("Overturn")
    out = {"challenges": int(len(ch)), "per_team_game": float(len(ch) / (2 * pbp["game_id"].nunique())), "win_rate": float(ch["won"].mean())}
    out["by_period"] = ch.groupby("period")["won"].agg(["mean", "count"]).round(3).to_dict("index")
    ch["min_left"] = ch["remaining"] / 60
    out["q4_last_3_min_share"] = float(((ch["period"] >= 4) & (ch["remaining"] <= 180)).mean())
    # The replay rows carry no team (team_id is 0), and only 174 of 965 sit next to a
    # "Coach Challenge" timeout row, so per-team challenge use cannot be measured here.
    out["team_identifiable"] = False
    out["period_mix"] = ch["period"].value_counts(normalize=True).round(3).sort_index().to_dict()
    return out


# ---------------------------------------------------------------- pace with a lead

def lead_pace(p: pd.DataFrame, tg: pd.DataFrame) -> dict:
    q4 = p[(p["period"] == 4) & (p["ended_by"] != "period_end")].copy()
    # margin_at_start is from the possession team's point of view
    first = q4.sort_values("start_time").groupby(["game_id", "side"]).first()[["margin_at_start"]].rename(columns={"margin_at_start": "lead"}).reset_index()
    q4 = q4.drop(columns=["lead"], errors="ignore").merge(first, on=["game_id", "side"])
    lead = q4[(q4["lead"] >= 6) & (q4["start_remaining"] >= 180)].copy()  # the leader's possessions, first 9 minutes
    # pace measured only on possessions that start with the leader's own defensive rebound, so the
    # opponent's scoring (which forces slow inbound starts) cannot masquerade as the leader's choice
    dreb = lead[lead["start_type"] == "def_reb"]
    pace = dreb.groupby(["game_id", "side"]).agg(sec=("seconds_to_first_action", "median"), push_rate=("pushed", "mean"), n_dreb=("points", "size")).reset_index()
    per = lead.groupby(["game_id", "side"]).agg(lead=("lead", "first"), ppp=("points", "mean"), n=("points", "size"),
                                                  opp_make_share=("start_type", lambda s: s.isin(["made_fg", "made_ft"]).mean())).reset_index()
    per = per.merge(pace, on=["game_id", "side"]).query("n_dreb >= 4")
    fin = tg[["game_id", "side", "team_tricode", "margin", "half"]] if "half" in tg.columns else tg[["game_id", "side", "team_tricode", "margin"]].assign(half=half(tg["game_id"]))
    per = per.merge(fin, on=["game_id", "side"])
    per["q4_change"] = per["margin"] - per["lead"]
    per["won"] = per["margin"] > 0
    per["slow"] = per["sec"] > per["sec"].median()
    out = {"leader_team_games": int(len(per)), "median_sec_to_first_action": float(per["sec"].median()), "mean_lead": float(per["lead"].mean())}
    out["q4_margin_change"] = compare(per, "slow", "q4_change", "slow", "fast")
    out["win_rate"] = compare(per, "slow", "won", "slow", "fast")
    out["q4_ppp"] = compare(per, "slow", "ppp", "slow", "fast")
    b, se = ols(per[["sec", "lead"]].to_numpy(), per["q4_change"].to_numpy())
    out["q4_change_per_second_slower_holding_lead"] = {"coef": float(b[1]), "se": float(se[1])}
    b, se = ols(per[["sec", "lead", "opp_make_share"]].to_numpy(), per["q4_change"].to_numpy())
    out["same_holding_opponent_make_share"] = {"coef": float(b[1]), "se": float(se[1])}
    b, se = ols(per[["push_rate", "lead"]].to_numpy(), per["q4_change"].to_numpy())
    out["q4_change_per_unit_push_rate_after_dreb"] = {"coef": float(b[1]), "se": float(se[1])}
    out["opp_make_share_slow_vs_fast"] = {"slow": float(per.loc[per["slow"], "opp_make_share"].mean()), "fast": float(per.loc[~per["slow"], "opp_make_share"].mean())}
    # by lead size
    per["lead_bin"] = pd.cut(per["lead"], [5, 9, 14, 60], labels=["6-9", "10-14", "15+"])
    out["by_lead"] = {str(k): {"slow_change": float(g.loc[g["slow"], "q4_change"].mean()), "fast_change": float(g.loc[~g["slow"], "q4_change"].mean()),
                               "slow_win": float(g.loc[g["slow"], "won"].mean()), "fast_win": float(g.loc[~g["slow"], "won"].mean()), "n": int(len(g))}
                      for k, g in per.groupby("lead_bin", observed=True)}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = load_possessions()
    pbp = load_pbp()
    tg = team_games()
    tg["half"] = half(tg["game_id"])
    res = {"games": int(p["game_id"].nunique())}
    for name, fn in [
        ("crash", lambda: crash(tg)),
        ("two_for_one", lambda: two_for_one(p)),
        ("bonus", lambda: bonus(p, pbp)),
        ("after_oreb", lambda: after_oreb(pbp)),
        ("take_fouls", lambda: take_fouls(p, pbp)),
        ("push_vs_crash", lambda: push_vs_crash(p, tg)),
        ("push_by_period", lambda: push_by_period(p)),
        ("run_timeouts", lambda: run_timeouts(p)),
        ("heaves", lambda: heaves(p, pbp)),
        ("challenges", lambda: challenges(pbp)),
        ("lead_pace", lambda: lead_pace(p, tg)),
    ]:
        print(f"== {name}", flush=True)
        try:
            res[name] = fn()
            print(json.dumps(res[name], indent=1, default=str)[:6000])
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            res[name] = {"error": repr(exc)}
    (OUT / "summary.json").write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
