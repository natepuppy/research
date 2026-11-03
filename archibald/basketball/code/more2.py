"""Third batch of levers: who rebounds and the push, blocks, missed-FT rebounds,
rest days, clutch free throws, home whistle.

Run from the basketball folder:  .venv/bin/python code/more2.py
Writes results/levers/more2.json (2025-26) and results/levers/more2_2425.json (2024-25 check).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import levers as L  # noqa: E402
from act import pos_class  # noqa: E402
from scoreboard import build_possessions  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "levers"


def clock_seconds(clock: pd.Series) -> pd.Series:
    m = clock.str.extract(r"PT(\d+)M([\d.]+)S")
    return m[0].astype(float) * 60 + m[1].astype(float)


# ------------------------------------------------------------ who rebounds -> push

def push_by_rebounder(p: pd.DataFrame, schema: str = "league") -> dict:
    roster = L.read(f"SELECT player_id, position FROM {schema}.rosters").drop_duplicates("player_id")
    cls = roster.set_index("player_id")["position"].map(pos_class)
    d = L.clean(p)
    d = d[d["start_type"] == "def_reb"].copy()
    d["reb_cls"] = d["prev_rebounder"].map(cls).fillna("team")
    d["long_miss"] = d["prev_shot_dist"] >= 10
    out = {"n": int(len(d)), "rebounder_mix": d["reb_cls"].value_counts(normalize=True).round(3).to_dict()}
    rows = {}
    for c, g in d.groupby("reb_cls"):
        if c in ("team", "unknown"):
            continue
        rows[c] = {
            "push_rate": float(g["pushed"].mean()),
            "ppp": L.mean_se(g["points"]),
            "push_vs_walk": L.compare(g, "pushed", "points", "push", "walk"),
            "ppp_by_miss": {("long" if k else "short"): L.mean_se(gg["points"]) for k, gg in g.groupby("long_miss")},
            "push_rate_by_miss": {("long" if k else "short"): float(gg["pushed"].mean()) for k, gg in g.groupby("long_miss")},
        }
    out["by_rebounder"] = rows
    # same miss type, guard vs big rebound: PPP of the whole possession
    d["guard_reb"] = (d["reb_cls"] == "guard").astype(int)
    d["big_reb"] = (d["reb_cls"] == "big").astype(int)
    X = d[["guard_reb", "big_reb", "long_miss", "margin_at_start"]].astype(float).to_numpy()
    b, se = L.ols(X, d["points"].to_numpy())
    out["ppp_vs_forward_rebound"] = {"guard": {"coef": float(b[1]), "se": float(se[1])}, "big": {"coef": float(b[2]), "se": float(se[2])}}
    # per team: share of defensive rebounds by guards, push rate after guard boards
    tg = L.team_games()[["game_id", "side", "team_tricode"]]
    t = d.merge(tg, on=["game_id", "side"])
    team = t.groupby("team_tricode").agg(guard_share=("guard_reb", "mean"), push_rate=("pushed", "mean"), n=("points", "size"))
    team["push_after_guard"] = t[t["guard_reb"] == 1].groupby("team_tricode")["pushed"].mean()
    team["push_after_big"] = t[t["big_reb"] == 1].groupby("team_tricode")["pushed"].mean()
    out["teams"] = team.round(3).sort_values("guard_share", ascending=False).to_dict("index")
    return out


# ------------------------------------------------------------ after a block

def after_block(p: pd.DataFrame, pbp: pd.DataFrame) -> dict:
    blk = pbp[pbp["description"].str.contains(r"\bBLOCK\b", na=False)].copy()
    blk["t"] = clock_seconds(blk["clock"])
    blk = blk[["game_id", "period", "t", "location"]].rename(columns={"location": "side"})
    d = p[p["start_type"] == "def_reb"].copy()
    d = d.sort_values("start_remaining")
    blk = blk.sort_values("t")
    m = pd.merge_asof(d, blk, left_on="start_remaining", right_on="t", by=["game_id", "period", "side"], direction="forward", tolerance=4.0)
    m["after_block"] = m["t"].notna()
    m = L.clean(m)
    out = {"def_reb_starts": int(len(m)), "share_after_block": float(m["after_block"].mean()),
           "blocked_shot_was_short": float((m.loc[m["after_block"], "prev_shot_dist"] < 10).mean())}
    out["push_rate"] = {"after_block": float(m.loc[m["after_block"], "pushed"].mean()), "other_def_reb": float(m.loc[~m["after_block"], "pushed"].mean())}
    out["ppp"] = L.compare(m, "after_block", "points", "after_block", "other_def_reb")
    ab = m[m["after_block"]]
    out["after_block_push_vs_walk"] = L.compare(ab, "pushed", "points", "push", "walk")
    short = m[m["prev_shot_dist"] < 10]
    out["short_miss_only_block_vs_not"] = L.compare(short, "after_block", "points", "after_block", "other_short_miss")
    out["short_miss_only_push_vs_walk"] = {"after_block": L.compare(short[short["after_block"]], "pushed", "points", "push", "walk"),
                                           "not_block": L.compare(short[~short["after_block"]], "pushed", "points", "push", "walk")}
    # where does the ball go after a block: who rebounds
    return out


# ------------------------------------------------------------ missed free throw rebound

def after_ft_miss(p: pd.DataFrame) -> dict:
    d = L.clean(p)
    out = {}
    for st in ("def_reb_ft", "def_reb"):
        g = d[d["start_type"] == st]
        out[st] = {"n": int(len(g)), "push_rate": float(g["pushed"].mean()), "ppp": L.mean_se(g["points"]),
                   "push_vs_walk": L.compare(g, "pushed", "points", "push", "walk")}
    return out


# ------------------------------------------------------------ rest

def rest(table: str = "league.game_log_team_regular_season", season_games: pd.DataFrame | None = None) -> dict:
    g = L.read(f"SELECT game_id, game_date, team_abbreviation AS team, matchup, pts, plus_minus, oreb, fga, fgm, fg3_a, fg3_m, fta, tov FROM {table}")
    g["game_date"] = pd.to_datetime(g["game_date"])
    g = g.sort_values(["team", "game_date"])
    g["days_rest"] = g.groupby("team")["game_date"].diff().dt.days
    g["b2b"] = (g["days_rest"] == 1).astype(float)
    g["home"] = g["matchup"].str.contains("vs.").astype(float)
    # 3 games in 4 nights
    g["g3in4"] = g.groupby("team")["game_date"].transform(lambda s: s.map(lambda d: ((s > d - pd.Timedelta(days=4)) & (s <= d)).sum())) >= 3
    g["g3in4"] = g["g3in4"].astype(float)
    opp = g[["game_id", "team", "b2b", "g3in4", "days_rest", "pts"]].rename(columns={"team": "opp", "b2b": "opp_b2b", "g3in4": "opp_3in4", "days_rest": "opp_rest", "pts": "opp_pts"})
    g["opp"] = g["matchup"].str[-3:]
    g = g.merge(opp, on=["game_id", "opp"], how="inner")
    g = g[g["days_rest"].notna() & g["opp_rest"].notna()].copy()
    g["rest_edge"] = np.clip(g["days_rest"], 1, 4) - np.clip(g["opp_rest"], 1, 4)
    out = {"team_games": int(len(g)), "b2b_share": float(g["b2b"].mean())}
    cols = ["plus_minus", "pts", "opp_pts", "b2b", "opp_b2b", "home", "g3in4", "opp_3in4", "rest_edge", "oreb", "tov", "fta"]
    dm = L.demean(g, cols, "team")
    X = dm[["b2b", "opp_b2b", "home"]].to_numpy()
    for y in ("plus_minus", "pts", "opp_pts", "oreb", "tov", "fta"):
        b, se = L.ols(X, dm[y].to_numpy())
        out[y] = {"own_b2b": {"coef": float(b[1]), "se": float(se[1])}, "opp_b2b": {"coef": float(b[2]), "se": float(se[2])}, "home": {"coef": float(b[3]), "se": float(se[3])}}
    b, se = L.ols(dm[["rest_edge", "home"]].to_numpy(), dm["plus_minus"].to_numpy())
    out["margin_per_day_of_rest_edge"] = {"coef": float(b[1]), "se": float(se[1])}
    b, se = L.ols(dm[["g3in4", "opp_3in4", "b2b", "opp_b2b", "home"]].to_numpy(), dm["plus_minus"].to_numpy())
    out["margin_3in4_on_top_of_b2b"] = {"own": {"coef": float(b[1]), "se": float(se[1])}, "opp": {"coef": float(b[2]), "se": float(se[2])}}
    out["raw_margin_by_rest"] = g.groupby(np.clip(g["days_rest"], 1, 4))["plus_minus"].agg(["mean", "count"]).round(2).to_dict("index")
    out["raw_margin_b2b_vs_rested_opp"] = g[(g["b2b"] == 1) & (g["opp_b2b"] == 0)]["plus_minus"].agg(["mean", "count", "sem"]).round(2).to_dict()
    # halves
    for h in ("even", "odd"):
        part = dm[L.half(g["game_id"]) == h]
        b, se = L.ols(part[["b2b", "opp_b2b", "home"]].to_numpy(), part["plus_minus"].to_numpy())
        out[f"plus_minus_{h}"] = {"own_b2b": {"coef": float(b[1]), "se": float(se[1])}, "opp_b2b": {"coef": float(b[2]), "se": float(se[2])}}
    # per team: how much of the schedule is b2b
    out["b2b_games_by_team"] = g.groupby("team")["b2b"].sum().astype(int).sort_values().to_dict()
    # link to per-game tracking: crash and pace on b2b
    if season_games is not None:
        tg = season_games.copy()
        tg["crash_per_miss"] = tg["rebound_chances_offensive"] / (tg["fga"] - tg["fgm"]) if "fgm" in tg else np.nan
        tg = tg.drop(columns=["home", "away"], errors="ignore").merge(g[["game_id", "team", "b2b", "opp_b2b", "home"]].rename(columns={"team": "team_tricode"}), on=["game_id", "team_tricode"], how="inner")
        ys = ["pace", "oreb_pct", "distance", "speed", "o_early_share", "d_early_share", "offensive_rating", "defensive_rating", "net_rating", "contested_shots", "deflections", "o_ppp_late", "o_late_share"]
        ys = [y for y in ys if y in tg]
        dm2 = L.demean(tg, ys + ["b2b", "opp_b2b", "home"], "team_tricode")
        X2 = dm2[["b2b", "opp_b2b", "home"]].to_numpy()
        out["tracking_on_b2b"] = {}
        for y in ys:
            ok = dm2[y].notna().to_numpy()
            if ok.sum() < 100:
                continue
            b, se = L.ols(X2[ok], dm2.loc[ok, y].to_numpy())
            out["tracking_on_b2b"][y] = {"own_b2b": {"coef": float(b[1]), "se": float(se[1])}, "opp_b2b": {"coef": float(b[2]), "se": float(se[2])}}
        out["tracking_games"] = int(len(tg))
    return out


# ------------------------------------------------------------ clutch free throws

def clutch_ft(pbp: pd.DataFrame) -> dict:
    ft = pbp[pbp["action_type"] == "Free Throw"].copy()
    ft = ft[~ft["description"].str.contains("Technical", na=False)]
    ft["made"] = (~ft["description"].str.startswith("MISS")).astype(float)
    ft["t"] = clock_seconds(ft["clock"])
    ft["margin"] = (ft["score_home"].astype(float) - ft["score_away"].astype(float))
    ft["margin"] = np.where(ft["location"] == "h", ft["margin"], -ft["margin"])
    ft["rate"] = ft.groupby("person_id")["made"].transform("mean")
    n_p = ft.groupby("person_id")["made"].transform("size")
    ft = ft[n_p >= 20].copy()
    ft["vs_own"] = ft["made"] - ft["rate"]
    ft["clutch"] = (ft["period"] >= 4) & (ft["t"] <= 120) & (ft["margin"].abs() <= 3)
    ft["late_close5"] = (ft["period"] >= 4) & (ft["t"] <= 300) & (ft["margin"].abs() <= 5)
    out = {"free_throws": int(len(ft)), "clutch_fts": int(ft["clutch"].sum())}
    out["clutch_vs_own_rate"] = L.compare(ft, "clutch", "vs_own", "clutch", "other")
    out["late_close5_vs_own_rate"] = L.compare(ft, "late_close5", "vs_own", "late_close", "other")
    c = ft[ft["clutch"]]
    out["clutch_by_state"] = {k: L.mean_se(g["vs_own"]) for k, g in c.groupby(np.where(c["margin"] < 0, "trailing", np.where(c["margin"] > 0, "leading", "tied")))}
    out["clutch_home_vs_away"] = {k: L.mean_se(g["vs_own"]) for k, g in c.groupby("location")}
    out["clutch_raw_pct"] = float(c["made"].mean())
    out["other_raw_pct"] = float(ft.loc[~ft["clutch"], "made"].mean())
    # who shoots them: do teams get the ball to good shooters
    out["clutch_shooter_season_rate"] = float(ft.loc[ft["clutch"], "rate"].mean())
    out["other_shooter_season_rate"] = float(ft.loc[~ft["clutch"], "rate"].mean())
    return out


# ------------------------------------------------------------ home whistle

def home_whistle(p: pd.DataFrame, pbp: pd.DataFrame) -> dict:
    f = pbp[(pbp["action_type"] == "Foul") & pbp["sub_type"].isin(["Shooting", "Personal", "Personal Take", "Transition Take", "Away From Play", "Clear Path"])].copy()
    f["t"] = clock_seconds(f["clock"])
    f["remaining"] = f["t"]
    f["shooting"] = f["sub_type"] == "Shooting"
    f["margin"] = f["score_home"].astype(float) - f["score_away"].astype(float)
    f["close_late"] = (f["period"] >= 4) & (f["t"] <= 300) & (f["margin"].abs() <= 5)
    games = pbp["game_id"].nunique()
    out = {"games": int(games)}
    out["fouls_per_game"] = {"on_home": float((f["location"] == "h").sum() / games), "on_away": float((f["location"] == "v").sum() / games)}
    out["shooting_fouls_per_game"] = {"on_home": float(((f["location"] == "h") & f["shooting"]).sum() / games), "on_away": float(((f["location"] == "v") & f["shooting"]).sum() / games)}
    cl = f[f["close_late"]]
    g_cl = cl["game_id"].nunique()
    out["close_late_games"] = int(g_cl)
    out["close_late_fouls_per_game"] = {"on_home": float((cl["location"] == "h").sum() / g_cl), "on_away": float((cl["location"] == "v").sum() / g_cl)}
    # possession level: FT trip share and PPP by side, clean, and late close
    d = L.clean(p)
    out["ft_trip_share"] = {s: float((g["fta"] > 0).mean()) for s, g in d.groupby("side")}
    out["ppp"] = {s: L.mean_se(g["points"]) for s, g in d.groupby("side")}
    late = p[(p["period"] >= 4) & (p["start_remaining"] <= 300) & (p["margin_at_start"].abs() <= 5) & (p["ended_by"] != "period_end")]
    out["late_close_ft_trip_share"] = {s: float((g["fta"] > 0).mean()) for s, g in late.groupby("side")}
    out["late_close_ppp"] = {s: L.mean_se(g["points"]) for s, g in late.groupby("side")}
    # home margin per game, split halves
    tg = L.team_games()
    h = tg[tg["side"] == "h"]
    out["home_margin"] = L.mean_se(h["margin"])
    out["home_margin_by_half"] = {k: L.mean_se(g["margin"]) for k, g in h.groupby(L.half(h["game_id"]))}
    # by period: foul differential (away minus home) per game
    out["foul_diff_away_minus_home_by_period"] = {int(k): float(((g["location"] == "v").sum() - (g["location"] == "h").sum()) / games) for k, g in f[f["period"] <= 4].groupby("period")}
    # per-team home whistle
    f2 = f.merge(tg[["game_id", "side", "team_tricode"]].rename(columns={"side": "location"}), on=["game_id", "location"])
    tf = f2.groupby(["team_tricode", "game_id"]).size().rename("fouls").reset_index().merge(tg[["game_id", "team_tricode", "side", "opponent"]], on=["game_id", "team_tricode"])
    team = tf.groupby(["team_tricode", "side"])["fouls"].mean().unstack()
    team["home_minus_away"] = team["h"] - team["v"]
    out["team_fouls_committed_home_minus_away"] = team["home_minus_away"].round(2).sort_values().to_dict()
    return out


# ------------------------------------------------------------ referees

def referees(pbp: pd.DataFrame, tg: pd.DataFrame) -> dict:
    f = pbp[pbp["action_type"] == "Foul"].copy()
    f["ref"] = f["description"].str.extract(r"\(([A-Z]\.[^()]+)\)\s*$")[0]
    f = f[f["ref"].notna()]
    f["personal"] = f["sub_type"].isin(["Shooting", "Personal", "Personal Take", "Transition Take", "Away From Play", "Clear Path"])
    f["shooting"] = f["sub_type"] == "Shooting"
    f["offensive"] = f["sub_type"].isin(["Offensive", "Offensive Charge"])
    f["tech"] = f["sub_type"].str.contains("Technical", na=False)
    f["on_home"] = f["location"] == "h"
    crew = f.groupby("game_id")["ref"].agg(lambda s: tuple(sorted(s.unique())))
    out = {"games": int(f["game_id"].nunique()), "refs": int(f["ref"].nunique()), "games_with_3_refs": float((crew.map(len) == 3).mean())}
    # game-level per-ref frame
    g = f.groupby(["game_id", "ref"]).agg(calls=("personal", "sum"), shooting=("shooting", "sum"), offensive=("offensive", "sum"), tech=("tech", "sum"), on_home=("on_home", lambda s: s[f.loc[s.index, "personal"]].sum())).reset_index()
    per_game = f[f["personal"]].groupby("game_id").size().rename("game_calls")
    g = g.merge(per_game, on="game_id")
    r = g.groupby("ref").agg(games=("game_id", "nunique"), calls_per_game=("calls", "mean"), shooting_per_game=("shooting", "mean"), offensive_per_game=("offensive", "mean"), techs_per_game=("tech", "mean"), home_share=("on_home", "sum"))
    r["home_share"] = r["home_share"] / g.groupby("ref")["calls"].sum()
    r = r[r["games"] >= 10]
    out["refs_with_10_games"] = int(len(r))
    out["calls_per_game_spread"] = {"min": float(r["calls_per_game"].min()), "median": float(r["calls_per_game"].median()), "max": float(r["calls_per_game"].max()), "sd": float(r["calls_per_game"].std())}
    # how much is a ref's rate real: split halves of his games
    g["half"] = L.half(g["game_id"])
    hh = g[g["ref"].isin(r.index)].groupby(["ref", "half"])["calls"].mean().unstack()
    hh = hh.dropna()
    out["ref_rate_split_half_corr"] = float(np.corrcoef(hh["even"], hh["odd"])[0, 1])
    hs = g[g["ref"].isin(r.index)].assign(hs=lambda d: d["on_home"] / d["calls"].clip(lower=1)).groupby(["ref", "half"])["hs"].mean().unstack().dropna()
    out["ref_home_share_split_half_corr"] = float(np.corrcoef(hs["even"], hs["odd"])[0, 1])
    # crew total fouls -> game FT trips and points: does the crew assignment predict the game
    crew_sum = g.groupby("game_id").agg(game_calls=("game_calls", "first"))
    ref_mean = r["calls_per_game"]
    # leave-one-game-out: expected calls = sum over crew refs of their mean in other games
    tot = g.groupby("ref")["calls"].sum()
    cnt = g.groupby("ref")["calls"].size()
    g["loo"] = (g["ref"].map(tot) - g["calls"]) / (g["ref"].map(cnt) - 1).clip(lower=1)
    exp = g.groupby("game_id")["loo"].sum().rename("expected_calls")
    gm = crew_sum.join(exp)
    t2 = tg.groupby("game_id").agg(fta=("fta", "sum"), points=("points", "sum"), pace=("pace", "mean"), fouls=("fouls", "sum"))
    gm = gm.join(t2)
    b, se = L.ols(gm[["expected_calls"]].to_numpy(), gm["game_calls"].to_numpy())
    out["actual_calls_per_expected_call_from_crew"] = {"coef": float(b[1]), "se": float(se[1]), "n": int(len(gm))}
    out["crew_expected_calls_sd"] = float(gm["expected_calls"].std())
    for y in ("fta", "points", "pace"):
        ok = gm[["expected_calls", y]].dropna()
        b, se = L.ols(ok[["expected_calls"]].to_numpy(), ok[y].to_numpy())
        out[f"game_{y}_per_expected_call"] = {"coef": float(b[1]), "se": float(se[1])}
    out["tightest"] = r.sort_values("calls_per_game").head(5).round(2).to_dict("index")
    out["loosest"] = r.sort_values("calls_per_game").tail(5).round(2).to_dict("index")
    out["most_home_friendly"] = r.sort_values("home_share").tail(5)[["games", "home_share"]].round(3).to_dict("index")
    out["least_home_friendly"] = r.sort_values("home_share").head(5)[["games", "home_share"]].round(3).to_dict("index")
    out["all_refs"] = r.round(3).sort_values("calls_per_game").to_dict("index")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = L.load_possessions()
    pbp = L.load_pbp()
    tg = L.team_games()
    res = {}
    for name, fn in [
        ("push_by_rebounder", lambda: push_by_rebounder(p)),
        ("after_block", lambda: after_block(p, pbp)),
        ("after_ft_miss", lambda: after_ft_miss(p)),
        ("rest", lambda: rest(season_games=tg)),
        ("clutch_ft", lambda: clutch_ft(pbp)),
        ("home_whistle", lambda: home_whistle(p, pbp)),
        ("referees", lambda: referees(pbp, tg)),
    ]:
        print("==", name, flush=True)
        res[name] = fn()
        print(json.dumps(res[name], indent=1, default=str))
    json.dump(res, open(OUT / "more2.json", "w"), indent=2, default=str)

    # 2024-25 check
    print("== 2024-25", flush=True)
    raw = L.read("SELECT * FROM s2425.pbp")
    p2 = build_possessions(raw)
    p2["half"] = L.half(p2["game_id"])
    p2["pushed"] = p2["seconds_to_first_action"] <= 8
    p2["opp"] = p2["side"].map({"h": "v", "v": "h"})
    pbp2 = L.load_pbp("s2425")
    res2 = {"games": int(p2["game_id"].nunique())}
    res2["push_by_rebounder"] = push_by_rebounder(p2, schema="s2425") if "teams" else None
    res2["push_by_rebounder"].pop("teams", None)
    res2["after_block"] = after_block(p2, pbp2)
    res2["after_ft_miss"] = after_ft_miss(p2)
    res2["rest"] = rest("s2425.game_log_team")
    res2["clutch_ft"] = clutch_ft(pbp2)
    print(json.dumps(res2, indent=1, default=str))
    json.dump(res2, open(OUT / "more2_2425.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
