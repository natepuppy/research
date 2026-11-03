"""Fourth batch: hustle stats that move the score, missed-free-throw rebounds,
momentum after runs, letdown and bounce-back games, fouling up three.

Run from the basketball folder:  .venv/bin/python code/more3.py
Writes results/levers/more3.json and results/levers/more3_2425.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import levers as L  # noqa: E402
from more2 import clock_seconds  # noqa: E402
from scoreboard import build_possessions  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "levers"


# ------------------------------------------------------------ hustle

def hustle(tg: pd.DataFrame) -> dict:
    cols = ["deflections", "loose_balls", "charges_drawn", "offensive_box_outs", "defensive_box_outs", "screen_assists",
            "contested_shots", "passes", "touches"]
    cols = [c for c in cols if c in tg and tg[c].notna().sum() > 500]
    d = tg.dropna(subset=cols + ["net_rating", "offensive_rating", "defensive_rating", "possessions"]).copy()
    # per 100 possessions so pace does not drive it
    for c in cols:
        if c not in ("speed", "distance"):
            d[c] = d[c] / d["possessions"] * 100
    out = {"team_games": int(len(d)), "means": d[cols].mean().round(2).to_dict(), "sd_within_team": (d[cols] - d.groupby("team_tricode")[cols].transform("mean")).std().round(2).to_dict()}
    ys = ["net_rating", "offensive_rating", "defensive_rating", "oreb_pct", "opp_oreb_pct"]
    ys = [y for y in ys if y in d]
    res = {}
    for c in cols:
        res[c] = {}
        for y in ys:
            dm = L.demean(d, [c, y, "home_flag"] if "home_flag" in d else [c, y], "team_tricode")
            b, se = L.ols(dm[[c]].to_numpy(), dm[y].to_numpy())
            row = {"per_unit": {"coef": float(b[1]), "se": float(se[1])}}
            row["per_sd"] = float(b[1] * out["sd_within_team"][c])
            for h in ("even", "odd"):
                part = dm[L.half(d["game_id"]) == h]
                bb, ss = L.ols(part[[c]].to_numpy(), part[y].to_numpy())
                row[h] = float(bb[1] * out["sd_within_team"][c])
            res[c][y] = row
    out["within_team"] = res
    # all hustle at once on net rating
    dm = L.demean(d, cols + ["net_rating"], "team_tricode")
    b, se = L.ols(dm[cols].to_numpy(), dm["net_rating"].to_numpy())
    out["joint_net_rating_per_sd"] = {c: {"coef": float(b[i + 1] * out["sd_within_team"][c]), "se": float(se[i + 1] * out["sd_within_team"][c])} for i, c in enumerate(cols)}
    # box outs and the opponent's crash
    if "defensive_box_outs" in d and "opp_oreb_pct" in d:
        dm = L.demean(d, ["defensive_box_outs", "opp_oreb_pct", "defensive_rating"], "team_tricode")
        b, se = L.ols(dm[["defensive_box_outs"]].to_numpy(), dm["opp_oreb_pct"].to_numpy())
        out["opp_oreb_pct_per_def_box_out_per_100"] = {"coef": float(b[1]), "se": float(se[1])}
    return out


# ------------------------------------------------------------ missed free throw rebounds

def ft_miss_rebounds(pbp: pd.DataFrame, tg: pd.DataFrame) -> dict:
    pb = pbp.sort_values(["game_id", "action_number", "action_id"]).reset_index(drop=True)
    ft = pb[(pb["action_type"] == "Free Throw") & pb["description"].str.startswith("MISS", na=False)
            & pb["description"].str.contains(r"(\d) of \1", na=False) & ~pb["description"].str.contains("Technical", na=False)]
    nxt = pb.shift(-1)
    nxt2 = pb.shift(-2)
    rows = []
    for i in ft.index:
        r = pb.loc[i]
        # the rebound is the next rebound row within the same game and period
        j = i + 1
        found = None
        while j < len(pb) and j <= i + 3 and pb.at[j, "game_id"] == r["game_id"]:
            if pb.at[j, "action_type"] == "Rebound":
                found = pb.loc[j]
                break
            j += 1
        if found is None:
            continue
        rows.append({"game_id": r["game_id"], "period": r["period"], "shooter_side": r["location"], "reb_side": found["location"],
                     "team_reb": found["person_id"] == found["team_id"], "remaining": clock_seconds(pd.Series([r["clock"]]))[0]})
    f = pd.DataFrame(rows)
    f["oreb"] = (f["reb_side"] == f["shooter_side"]).astype(float)
    f = f[~((f["period"] >= 4) & (f["remaining"] <= 180))]  # drop end-game intentional misses
    f["half"] = L.half(f["game_id"])
    out = {"missed_last_free_throws": int(len(f)), "per_game": float(len(f) / f["game_id"].nunique()), "oreb_rate": L.mean_se(f["oreb"]),
           "oreb_rate_by_half": {h: float(g["oreb"].mean()) for h, g in f.groupby("half")},
           "team_rebound_share": float(f["team_reb"].mean())}
    t = f.merge(tg[["game_id", "side", "team_tricode"]].rename(columns={"side": "shooter_side"}), on=["game_id", "shooter_side"])
    team = t.groupby("team_tricode")["oreb"].agg(["mean", "size"])
    team["even"] = t[t["half"] == "even"].groupby("team_tricode")["oreb"].mean()
    team["odd"] = t[t["half"] == "odd"].groupby("team_tricode")["oreb"].mean()
    out["team_oreb_rate_on_missed_ft"] = team.round(3).sort_values("mean", ascending=False).to_dict("index")
    out["team_split_half_corr"] = float(np.corrcoef(team["even"].fillna(team["mean"]), team["odd"].fillna(team["mean"]))[0, 1])
    # defensive side
    td = f.merge(tg[["game_id", "side", "team_tricode"]].rename(columns={"side": "def_side"}), left_on=["game_id", f["shooter_side"].map({"h": "v", "v": "h"})], right_on=["game_id", "def_side"])
    dteam = td.groupby("team_tricode")["oreb"].agg(["mean", "size"])
    out["team_allowed_oreb_rate_on_missed_ft"] = dteam.round(3).sort_values("mean").to_dict("index")
    return out


# ------------------------------------------------------------ momentum

def momentum(p: pd.DataFrame, run_size: int = 8, window: int = 6) -> dict:
    d = p[p["ended_by"] != "period_end"].sort_values(["game_id", "period", "start_time"], ascending=[True, True, False]).copy()
    d["signed"] = np.where(d["side"] == "h", d["points"], -d["points"])
    out = {}
    rows = []
    for gid, g in d.groupby("game_id", sort=False):
        s = g["signed"].to_numpy()
        per = g["period"].to_numpy()
        rem = g["start_remaining"].to_numpy()
        marg = np.where(g["side"].to_numpy() == "h", g["margin_at_start"].to_numpy(), -g["margin_at_start"].to_numpy())
        n = len(s)
        cs = np.concatenate([[0], np.cumsum(s)])
        for i in range(window, n - window):
            if per[i] >= 4 and rem[i] < 360:
                continue
            if abs(marg[i]) > 15:
                continue
            prev = cs[i] - cs[i - window]
            nxt = cs[i + window] - cs[i]
            rows.append((gid, prev, nxt, marg[i]))
    r = pd.DataFrame(rows, columns=["game_id", "prev", "next", "margin"])
    r["half"] = L.half(r["game_id"])
    # control for which team is better using the game's overall margin outside this window is messy; use the pregame-free margin control
    r["state"] = np.where(r["prev"] >= run_size, "home_run", np.where(r["prev"] <= -run_size, "away_run", "none"))
    out["windows"] = int(len(r))
    out["share_with_run"] = float((r["state"] != "none").mean())
    # next-6 net, signed in the direction of the run
    r["next_in_run_dir"] = np.where(r["state"] == "home_run", r["next"], np.where(r["state"] == "away_run", -r["next"], np.nan))
    r["margin_in_run_dir"] = np.where(r["state"] == "home_run", r["margin"], np.where(r["state"] == "away_run", -r["margin"], np.nan))
    runs = r[r["state"] != "none"].copy()
    out["after_run_next6_net_for_running_team"] = L.mean_se(runs["next_in_run_dir"])
    out["after_run_by_half"] = {h: L.mean_se(g["next_in_run_dir"]) for h, g in runs.groupby("half")}
    # regress on the margin at that point to remove "better team" effect
    b, se = L.ols(runs[["margin_in_run_dir"]].to_numpy(), runs["next_in_run_dir"].to_numpy())
    out["after_run_next6_controlling_for_margin"] = {"intercept": float(b[0]), "se": float(se[0]), "per_point_of_margin": float(b[1])}
    # baseline: next-6 net from the view of the team that is ahead, no run
    none = r[r["state"] == "none"].copy()
    none = none[none["margin"] != 0]
    none["next_for_leader"] = np.where(none["margin"] > 0, none["next"], -none["next"])
    out["no_run_next6_for_leader"] = L.mean_se(none["next_for_leader"])
    # does a run predict the rest of the game beyond the margin
    return out


# ------------------------------------------------------------ letdown and bounce-back

def letdown(table: str = "league.game_log_team_regular_season") -> dict:
    g = L.read(f"SELECT game_id, game_date, team_abbreviation AS team, matchup, plus_minus FROM {table}")
    g["game_date"] = pd.to_datetime(g["game_date"])
    g = g.sort_values(["team", "game_date"])
    g["prev_margin"] = g.groupby("team")["plus_minus"].shift(1)
    g["prev2"] = g.groupby("team")["plus_minus"].shift(2)
    g["home"] = g["matchup"].str.contains("vs.").astype(float)
    g["days_rest"] = g.groupby("team")["game_date"].diff().dt.days
    g["b2b"] = (g["days_rest"] == 1).astype(float)
    g = g.dropna(subset=["prev_margin"]).copy()
    g["big_win"] = (g["prev_margin"] >= 20).astype(float)
    g["big_loss"] = (g["prev_margin"] <= -20).astype(float)
    g["prev_win"] = (g["prev_margin"] > 0).astype(float)
    dm = L.demean(g, ["plus_minus", "big_win", "big_loss", "prev_win", "home", "b2b", "prev_margin"], "team")
    out = {"team_games": int(len(g))}
    b, se = L.ols(dm[["big_win", "big_loss", "home", "b2b"]].to_numpy(), dm["plus_minus"].to_numpy())
    out["margin_after_big_win"] = {"coef": float(b[1]), "se": float(se[1]), "n": int(g["big_win"].sum())}
    out["margin_after_big_loss"] = {"coef": float(b[2]), "se": float(se[2]), "n": int(g["big_loss"].sum())}
    b, se = L.ols(dm[["prev_margin", "home", "b2b"]].to_numpy(), dm["plus_minus"].to_numpy())
    out["margin_per_point_of_previous_margin_within_team"] = {"coef": float(b[1]), "se": float(se[1])}
    b, se = L.ols(dm[["prev_win", "home", "b2b"]].to_numpy(), dm["plus_minus"].to_numpy())
    out["margin_after_any_win_within_team"] = {"coef": float(b[1]), "se": float(se[1])}
    for h in ("even", "odd"):
        part = dm[L.half(g["game_id"]) == h]
        b, se = L.ols(part[["big_win", "big_loss", "home", "b2b"]].to_numpy(), part["plus_minus"].to_numpy())
        out[f"{h}"] = {"after_big_win": float(b[1]), "after_big_loss": float(b[2])}
    # streaks: 3+ losses in a row
    g["l3"] = (g.groupby("team")["plus_minus"].transform(lambda s: (s.shift(1) < 0) & (s.shift(2) < 0) & (s.shift(3) < 0))).astype(float)
    g["w3"] = (g.groupby("team")["plus_minus"].transform(lambda s: (s.shift(1) > 0) & (s.shift(2) > 0) & (s.shift(3) > 0))).astype(float)
    dm = L.demean(g, ["plus_minus", "l3", "w3", "home", "b2b"], "team")
    b, se = L.ols(dm[["l3", "w3", "home", "b2b"]].to_numpy(), dm["plus_minus"].to_numpy())
    out["margin_after_3_losses"] = {"coef": float(b[1]), "se": float(se[1]), "n": int(g["l3"].sum())}
    out["margin_after_3_wins"] = {"coef": float(b[2]), "se": float(se[2]), "n": int(g["w3"].sum())}
    return out


# ------------------------------------------------------------ fouling up three

def foul_up_three(pbp: pd.DataFrame, tg: pd.DataFrame) -> dict:
    pb = pbp.copy()
    pb["t"] = pb["remaining"]
    pb["margin_h"] = pb["score_home"] - pb["score_away"]
    # situations: period >= 4, 24 s or less, home or away up exactly 3, trailing team has the ball.
    # We look for the first event of the trailing team's possession: a foul by the leader (before a shot) or a shot by the trailer.
    sit = pb[(pb["period"] >= 4) & (pb["t"] <= 24) & (pb["t"] >= 2) & (pb["margin_h"].abs() == 3)].copy()
    rows = []
    for gid, g in sit.groupby("game_id"):
        g = g.sort_values(["period", "action_number"])
        for per, gg in g.groupby("period"):
            lead = "h" if gg["margin_h"].iloc[0] > 0 else "v"
            trail = "v" if lead == "h" else "h"
            # first trailing-team shot or leader foul in the window
            shot = gg[(gg["location"] == trail) & gg["action_type"].isin(["Made Shot", "Missed Shot"])]
            foul = gg[(gg["location"] == lead) & (gg["action_type"] == "Foul") & gg["sub_type"].isin(["Personal", "Personal Take", "Shooting", "Transition Take"])]
            first_shot_t = shot["t"].max() if len(shot) else -1
            first_foul_t = foul["t"].max() if len(foul) else -1
            if first_shot_t < 0 and first_foul_t < 0:
                continue
            fouled = first_foul_t > first_shot_t
            was_three = bool(len(shot) and shot.sort_values("t", ascending=False).iloc[0]["shot_value"] == 3) if not fouled else False
            rows.append({"game_id": gid, "period": per, "lead": lead, "fouled_first": fouled, "trailer_took_three": was_three})
    r = pd.DataFrame(rows)
    if r.empty:
        return {"situations": 0}
    # did the period end tied (went to OT or lost)?
    final = tg.groupby("game_id").agg(margin_h=("margin", lambda s: s.iloc[0]), side=("side", "first"))
    fm = tg[tg["side"] == "h"].set_index("game_id")["margin"]
    r["leader_won"] = [(fm.get(g, np.nan) > 0) if ld == "h" else (fm.get(g, np.nan) < 0) for g, ld in zip(r["game_id"], r["lead"])]
    # overtime happened if the game has a period 5
    ot = pbp.groupby("game_id")["period"].max() >= 5
    r["went_to_ot_after"] = [bool(ot.get(g, False)) and per == 4 for g, per in zip(r["game_id"], r["period"])]
    out = {"situations": int(len(r)), "fouled_first_share": float(r["fouled_first"].mean())}
    for k, g in r.groupby("fouled_first"):
        out["fouled" if k else "defended"] = {"n": int(len(g)), "leader_won": float(g["leader_won"].mean()), "went_to_ot": float(g["went_to_ot_after"].mean()),
                                              "trailer_took_three": float(g["trailer_took_three"].mean())}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = L.load_possessions()
    pbp = L.load_pbp()
    tg = L.team_games()
    res = {}
    for name, fn in [
        ("hustle", lambda: hustle(tg)),
        ("ft_miss_rebounds", lambda: ft_miss_rebounds(pbp, tg)),
        ("momentum", lambda: momentum(p)),
        ("letdown", lambda: letdown()),
        ("foul_up_three", lambda: foul_up_three(pbp, tg)),
    ]:
        print("==", name, flush=True)
        res[name] = fn()
        print(json.dumps(res[name], indent=1, default=str))
    json.dump(res, open(OUT / "more3.json", "w"), indent=2, default=str)

    print("== 2024-25", flush=True)
    raw = L.read("SELECT * FROM s2425.pbp")
    p2 = build_possessions(raw)
    p2["half"] = L.half(p2["game_id"])
    res2 = {"games": int(p2["game_id"].nunique())}
    res2["momentum"] = momentum(p2)
    res2["letdown"] = letdown("s2425.game_log_team")
    print(json.dumps(res2, indent=1, default=str))
    json.dump(res2, open(OUT / "more3_2425.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
