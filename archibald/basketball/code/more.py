"""Second batch of levers.

  set_plays      possessions after a stoppage: own timeout, opponent timeout, quarter start
  discipline     free points given away: technicals, fouls on three-point shooters,
                 shot-clock and backcourt violations, defensive three seconds
  foul_trouble   a starter picks up his second foul in the first quarter: sit him or not
  bench_windows  the first three minutes of the 2nd and 4th quarters
  hot_hand       a shooter's next three after two makes versus two misses
  ft_icing       free throws right after a timeout
  timeouts_left  timeouts a team still had at the end of a close game
  clutch_clock   last three minutes of a close game: when the shot comes

Writes results/levers/more.json. Each comparison is also run on even and odd games.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import levers as L  # noqa: E402
from scoreboard import clock_seconds, period_len  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "levers"


# ---------------------------------------------------------------- set plays

def set_plays(p: pd.DataFrame, teams: bool = True) -> dict:
    live = L.clean(p).copy()
    live["tob"] = live["timeout_before_by"].fillna("none")
    out = {}
    # after a stoppage the clock is dead, so compare with other dead-ball starts only
    dead = live[live["start_type"].isin(["made_fg", "made_ft", "dead_tov"])]
    for h, d in [("all", dead)] + [(k, g) for k, g in dead.groupby("half")]:
        out[h] = {k: L.mean_se(g["points"]) for k, g in d.groupby("tob")}
        out[h]["tov_rate"] = {k: float((g["tov"] > 0).mean()) for k, g in d.groupby("tob")}
        out[h]["ft_trip_rate"] = {k: float((g["fta"] > 0).mean()) for k, g in d.groupby("tob")}
        out[h]["early_share"] = {k: float(g["pushed"].mean()) for k, g in d.groupby("tob")}
    # quarter starts
    qs = live[live["start_type"] == "period_start"]
    out["quarter_start"] = {f"Q{k}": L.mean_se(g["points"]) for k, g in qs.groupby("period") if k <= 4}
    out["quarter_start_tov_rate"] = float((qs["tov"] > 0).mean())
    out["quarter_start_first_shot_zone"] = (np.where(qs["first_shot_value"] == 3, "three", np.where(qs["first_shot_dist"] < 5, "rim", "other")).tolist())
    z = pd.Series(out.pop("quarter_start_first_shot_zone"))
    out["quarter_start_first_shot_zone"] = z.value_counts(normalize=True).round(3).to_dict()
    out["dead_ball_baseline"] = L.mean_se(dead.loc[dead["tob"] == "none", "points"])
    out["quarter_start_by_half"] = {h: L.mean_se(g["points"]) for h, g in qs[qs["period"] <= 4].groupby("half")}
    if not teams:
        return out
    # per team: ATO offense
    tg = L.team_games()[["game_id", "side", "team_tricode"]]
    d = dead.merge(tg, on=["game_id", "side"], how="left")
    t = d[d["tob"] == "own"].groupby("team_tricode")["points"].agg(["mean", "count"])
    base = d[d["tob"] == "none"].groupby("team_tricode")["points"].mean()
    t["vs_own_flow"] = t["mean"] - base
    out["team_after_own_timeout"] = t.round(3).sort_values("vs_own_flow").to_dict("index")
    return out


# ---------------------------------------------------------------- discipline

def discipline(pbp: pd.DataFrame, p: pd.DataFrame) -> dict:
    games = pbp["game_id"].nunique()
    tg = L.team_games()[["game_id", "side", "team_tricode", "net_rating", "margin", "possessions"]]
    tg["half"] = L.half(tg["game_id"])
    f = pbp[pbp["action_type"].isin(["Foul", "Turnover", "Violation", "Free Throw"])].copy()
    rows = []
    # technical free throws: the shooter's side is the beneficiary; the fouling side is the other
    tech_ft = f[(f["action_type"] == "Free Throw") & (f["sub_type"].str.contains("Technical", na=False))]
    tech_made = tech_ft[~tech_ft["description"].str.startswith("MISS", na=False)]
    three_shot = f[(f["action_type"] == "Free Throw") & (f["sub_type"] == "Free Throw 1 of 3")]
    shot_clock = f[(f["action_type"] == "Turnover") & (f["sub_type"] == "Shot Clock Turnover")]
    backcourt = f[(f["action_type"] == "Turnover") & (f["sub_type"].isin(["8 Second Violation", "Backcourt Turnover", "5 Second Violation", "Inbound Turnover"]))]
    d3 = f[(f["action_type"] == "Foul") & (f["sub_type"] == "Defense 3 Second")]
    flagrant = f[(f["action_type"] == "Foul") & (f["sub_type"].str.startswith("Flagrant", na=False))]
    out = {
        "per_team_game": {
            "technical_fts_conceded": float(len(tech_ft) / (2 * games)),
            "technical_ft_pct": float(len(tech_made) / max(len(tech_ft), 1)),
            "fouls_on_three_point_shooters": float(len(three_shot) / (2 * games)),
            "shot_clock_violations": float(len(shot_clock) / (2 * games)),
            "backcourt_8s_5s_inbound": float(len(backcourt) / (2 * games)),
            "defensive_three_seconds": float(len(d3) / (2 * games)),
            "flagrants": float(len(flagrant) / (2 * games)),
        }
    }
    # cost per event: three-shot foul vs the possession it replaced (a three-point attempt, about 1.1 points)
    three_ft = f[(f["action_type"] == "Free Throw") & (f["sub_type"].isin(["Free Throw 1 of 3", "Free Throw 2 of 3", "Free Throw 3 of 3"]))]
    pts_per_trip = 3 * float((~three_ft["description"].str.startswith("MISS", na=False)).mean())
    out["three_shot_foul_points_per_trip"] = pts_per_trip
    out["three_shot_foul_cost_vs_a_three"] = pts_per_trip - 1.10
    # live possession value lost on a shot-clock violation: about the average possession
    live = L.clean(p)
    out["average_possession"] = float(live["points"].mean())
    # per team-game counts, cost in points, within-team relation with margin
    def count(df, who):
        # who = 'location' is the team the row belongs to
        c = df.groupby(["game_id", who]).size().rename("n").reset_index().rename(columns={who: "side"})
        return c
    costs = {}
    # fouling side for technical FTs is the opposite of the shooter
    t = count(tech_ft, "location"); t["side"] = t["side"].map({"h": "v", "v": "h"}); costs["technical_fts"] = (t, 0.78)
    s3 = count(three_shot, "location"); s3["side"] = s3["side"].map({"h": "v", "v": "h"}); costs["three_shot_fouls"] = (s3, out["three_shot_foul_cost_vs_a_three"])
    costs["shot_clock"] = (count(shot_clock, "location"), out["average_possession"])
    costs["backcourt"] = (count(backcourt, "location"), out["average_possession"])
    costs["def_three_seconds"] = (count(d3, "location"), 0.78)
    frame = tg.copy()
    frame["discipline_cost"] = 0.0
    for name, (c, per) in costs.items():
        frame = frame.merge(c.rename(columns={"n": name}), on=["game_id", "side"], how="left").fillna({name: 0})
        frame["discipline_cost"] += frame[name] * per
    out["points_given_away_per_team_game"] = L.mean_se(frame["discipline_cost"])
    team = frame.groupby("team_tricode").agg(cost=("discipline_cost", "mean"), technical_fts=("technical_fts", "mean"), three_shot_fouls=("three_shot_fouls", "mean"), shot_clock=("shot_clock", "mean"), backcourt=("backcourt", "mean"), def3=("def_three_seconds", "mean"))
    out["team_points_given_away_per_game"] = team.round(2).sort_values("cost").to_dict("index")
    res = {}
    for h, g in [("all", frame)] + [(k, v) for k, v in frame.groupby("half")]:
        dm = L.demean(g, ["discipline_cost", "net_rating"], "team_tricode")
        b, se = L.ols(dm[["discipline_cost"]].to_numpy(), dm["net_rating"].to_numpy())
        res[h] = {"net_rating_per_point_given_away": round(float(b[1]), 3), "se": round(float(se[1]), 3)}
    out["within_team"] = res
    return out


# ---------------------------------------------------------------- foul trouble

def foul_trouble(pbp: pd.DataFrame, schema: str = "games") -> dict:
    starters = L.read(
        f"""
        SELECT game_id, team_tricode, person_id, minutes
        FROM {schema}.box_traditional_player WHERE position IS NOT NULL AND position <> ''
        """
    )
    starters["min"] = starters["minutes"].map(lambda m: (int(str(m).split(":")[0]) + int(str(m).split(":")[1]) / 60) if isinstance(m, str) and ":" in m else np.nan)
    fouls = pbp[(pbp["action_type"] == "Foul") & pbp["description"].str.contains(r"\(P\d", na=False)].copy()
    fouls["pf"] = fouls["description"].str.extract(r"\(P(\d)")[0].astype(int)
    fouls = fouls[fouls["person_id"].notna()]
    fouls["gt"] = (fouls["period"] - 1) * 720 + fouls["elapsed"]  # game seconds for regulation
    subs = pbp[pbp["action_type"] == "Substitution"].copy()
    subs["gt"] = (subs["period"] - 1) * 720 + subs["elapsed"]
    second_q1 = fouls[(fouls["pf"] == 2) & (fouls["period"] == 1)]
    second_q1 = second_q1.merge(starters[["game_id", "person_id", "min"]], on=["game_id", "person_id"], how="inner")
    final = fouls.groupby(["game_id", "person_id"])["pf"].max().rename("final_pf").reset_index()
    second_q1 = second_q1.merge(final, on=["game_id", "person_id"])
    rows = []
    sub_grp = subs.groupby(["game_id", "person_id"])
    for _, r in second_q1.iterrows():
        key = (r["game_id"], r["person_id"])
        later = sub_grp.get_group(key) if key in sub_grp.groups else pd.DataFrame()
        later = later[later["gt"] >= r["gt"] - 1] if len(later) else later
        until = float(later["gt"].min() - r["gt"]) if len(later) else np.nan  # seconds until he sat, NaN if never subbed
        # period end counts as leaving the floor too, if no sub came first
        to_period_end = 720 - r["elapsed"]
        sat_within_90 = (not np.isnan(until)) and until <= 90
        rows.append({"game_id": r["game_id"], "sat_within_90s": bool(sat_within_90), "secs_until_sub": until if not np.isnan(until) else to_period_end,
                     "final_pf": int(r["final_pf"]), "minutes": r["min"], "q1_remaining": 720 - r["elapsed"]})
    d = pd.DataFrame(rows)
    d["half"] = L.half(d["game_id"])
    d["fouled_out"] = d["final_pf"] >= 6
    d["reached_5"] = d["final_pf"] >= 5
    out = {"starters_with_2_fouls_in_q1": int(len(d)), "per_game": float(len(d) / pbp["game_id"].nunique()),
           "share_sat_within_90s": float(d["sat_within_90s"].mean())}
    out["fouled_out"] = L.compare(d, "sat_within_90s", "fouled_out", "sat", "stayed")
    out["reached_5_fouls"] = L.compare(d, "sat_within_90s", "reached_5", "sat", "stayed")
    out["minutes_played"] = L.compare(d, "sat_within_90s", "minutes", "sat", "stayed")
    out["final_fouls"] = L.compare(d, "sat_within_90s", "final_pf", "sat", "stayed")
    # baseline: all starters' foul-out rate, and starters with 1 foul in Q1
    all_final = starters.merge(final, on=["game_id", "person_id"], how="left").fillna({"final_pf": 0})
    out["all_starters_foul_out_rate"] = float((all_final["final_pf"] >= 6).mean())
    out["all_starters_minutes"] = L.mean_se(all_final["min"])
    return out


# ---------------------------------------------------------------- bench windows

def bench_windows(p: pd.DataFrame) -> dict:
    live = p[p["ended_by"] != "period_end"].copy()
    live = live[live["margin_at_start"].abs() <= 20]
    live["window"] = np.select(
        [(live["period"] == 2) & (live["start_remaining"] > 540), (live["period"] == 4) & (live["start_remaining"] > 540),
         (live["period"] == 1) & (live["start_remaining"] > 360), (live["period"] == 3) & (live["start_remaining"] > 360)],
        ["Q2 first 3 min", "Q4 first 3 min", "Q1 first 6 min", "Q3 first 6 min"], default="other")
    out = {"ppp_by_window": {k: L.mean_se(g["points"]) for k, g in live.groupby("window")}}
    tg = L.team_games()[["game_id", "side", "team_tricode", "margin"]]
    d = live.merge(tg, on=["game_id", "side"], how="left")
    # team net points in each window per game: own points minus opponent points in the same window
    own = d.groupby(["game_id", "side", "window"])["points"].sum().rename("own").reset_index()
    opp = own.rename(columns={"side": "opp_side", "own": "opp"})
    opp["side"] = opp["opp_side"].map({"h": "v", "v": "h"})
    w = own.merge(opp[["game_id", "side", "window", "opp"]], on=["game_id", "side", "window"], how="left").fillna({"opp": 0})
    w["net"] = w["own"] - w["opp"]
    w = w.merge(tg, on=["game_id", "side"])
    out["sd_of_team_net_by_window"] = w.groupby("window")["net"].std().round(2).to_dict()
    out["team_net_per_game"] = {}
    for win in ("Q2 first 3 min", "Q4 first 3 min", "Q1 first 6 min"):
        t = w[w["window"] == win].groupby("team_tricode")["net"].mean().round(2)
        out["team_net_per_game"][win] = t.sort_values().to_dict()
        out["team_net_per_game"][win + " spread"] = {"best": float(t.max()), "worst": float(t.min()), "sd_across_teams": float(t.std())}
    # how much of a team's season margin is the Q2/Q4 opening windows
    season = w.groupby(["team_tricode", "window"])["net"].sum().unstack().fillna(0)
    total = tg.groupby("team_tricode")["margin"].sum()
    out["share_of_season_margin_from_bench_windows"] = ((season["Q2 first 3 min"] + season["Q4 first 3 min"]) / total.replace(0, np.nan)).round(2).to_dict()
    return out


# ---------------------------------------------------------------- hot hand

def hot_hand(pbp: pd.DataFrame) -> dict:
    s = pbp[pbp["action_type"].isin(["Made Shot", "Missed Shot"]) & (pbp["shot_value"] == 3) & pbp["person_id"].notna()].copy()
    s["made"] = (s["action_type"] == "Made Shot").astype(int)
    s = s.sort_values(["game_id", "person_id", "action_number"])
    g = s.groupby(["game_id", "person_id"])["made"]
    s["prev1"] = g.shift(1)
    s["prev2"] = g.shift(2)
    s = s[s["prev2"].notna()]
    s["state"] = np.select([(s["prev1"] == 1) & (s["prev2"] == 1), (s["prev1"] == 0) & (s["prev2"] == 0)], ["two_makes", "two_misses"], default="mixed")
    # within player: compare to the player's season three-point rate in these games
    rate = s.groupby("person_id")["made"].transform("mean")
    s["excess"] = s["made"] - rate
    out = {"threes_with_two_prior": int(len(s))}
    for h, d in [("all", s)] + [(k, v) for k, v in s.groupby("half")]:
        out[h] = {k: {"fg_pct": float(v["made"].mean()), "vs_own_rate": float(v["excess"].mean()), "se": float(v["excess"].sem()), "n": int(len(v))} for k, v in d.groupby("state")}
    # shot difficulty: distance and whether the next shot comes sooner
    out["distance_by_state"] = s.groupby("state")["shot_distance"].mean().round(2).to_dict()
    return out


# ---------------------------------------------------------------- free-throw icing

def ft_icing(pbp: pd.DataFrame) -> dict:
    ft = pbp[(pbp["action_type"] == "Free Throw") & pbp["sub_type"].isin(["Free Throw 1 of 2", "Free Throw 1 of 1", "Free Throw 1 of 3"])].copy()
    ft["made"] = (~ft["description"].str.startswith("MISS", na=False)).astype(int)
    # a timeout between the foul and the first free throw: the previous non-foul row is a Timeout
    prev_type = pbp["action_type"].shift(1)
    prev_type2 = pbp["action_type"].shift(2)
    same_game = pbp["game_id"].shift(1) == pbp["game_id"]
    ft["iced"] = ((prev_type == "Timeout") | (prev_type2 == "Timeout")).reindex(ft.index) & same_game.reindex(ft.index)
    rate = ft.groupby("person_id")["made"].transform("mean")
    ft["excess"] = ft["made"] - rate
    ft["half"] = L.half(ft["game_id"])
    out = {"first_free_throws": int(len(ft)), "iced_share": float(ft["iced"].mean())}
    out["made_vs_own_rate"] = L.compare(ft, "iced", "excess", "after_timeout", "normal")
    out["raw_pct"] = L.compare(ft, "iced", "made", "after_timeout", "normal")["all"]
    late = ft[(ft["period"] >= 4) & (ft["remaining"] <= 120)]
    out["late_game_only"] = L.compare(late, "iced", "excess", "after_timeout", "normal")["all"]
    return out


# ---------------------------------------------------------------- timeouts left

def timeouts_left(pbp: pd.DataFrame) -> dict:
    to = pbp[(pbp["action_type"] == "Timeout") & pbp["description"].str.contains(r"\((Full|Reg\.)\s*\d", na=False)].copy()
    # the number in the description counts up as timeouts are used, so it is timeouts used so far
    to["left"] = 7 - to["description"].str.extract(r"(?:Full|Reg\.)\s*(\d)")[0].astype(int)
    last = to.sort_values(["game_id", "period", "elapsed"]).groupby(["game_id", "location"]).last().reset_index()
    tg = L.team_games()[["game_id", "side", "team_tricode", "margin"]]
    t = tg.merge(last[["game_id", "location", "left", "period", "remaining"]].rename(columns={"location": "side"}), on=["game_id", "side"], how="left")
    close = t[t["margin"].abs() <= 5].copy()
    out = {"team_games_close": int(len(close)), "timeouts_left_after_last_call_close_games": L.mean_se(close["left"]),
           "share_close_games_ending_with_2_plus_unused": float((close["left"] >= 2).mean()),
           "close_losses_with_2_plus_unused": float((close.loc[close["margin"] < 0, "left"] >= 2).mean())}
    out["by_team_close_games"] = close.groupby("team_tricode")["left"].mean().round(2).sort_values().to_dict()
    out["note"] = "left = 7 minus the used-count on the team's last timeout row (the count includes challenges); a team that never called one is missing."
    return out


# ---------------------------------------------------------------- clutch clock

def clutch_clock(p: pd.DataFrame) -> dict:
    c = p[(p["period"] >= 4) & (p["start_remaining"] <= 180) & (p["margin_at_start"].abs() <= 5) & (p["ended_by"] != "period_end")].copy()
    c = c[c["start_remaining"] > 24]  # leave out the final-possession games
    c["sec_bin"] = pd.cut(c["seconds_to_first_shot"], [0, 6, 10, 14, 18, 24], labels=["0-6", "6-10", "10-14", "14-18", "18-24"])
    normal = L.clean(p)
    normal["sec_bin"] = pd.cut(normal["seconds_to_first_shot"], [0, 6, 10, 14, 18, 24], labels=["0-6", "6-10", "10-14", "14-18", "18-24"])
    out = {"clutch_possessions": int(len(c)), "clutch_ppp": L.mean_se(c["points"]), "normal_ppp": L.mean_se(normal["points"])}
    out["clutch_by_seconds_to_shot"] = c.groupby("sec_bin", observed=True)["points"].agg(["mean", "count"]).round(3).reset_index().astype({"sec_bin": str}).to_dict("records")
    out["normal_by_seconds_to_shot"] = normal.groupby("sec_bin", observed=True)["points"].agg(["mean", "count"]).round(3).reset_index().astype({"sec_bin": str}).to_dict("records")
    out["clutch_median_seconds_to_shot"] = float(c["seconds_to_first_shot"].median())
    out["normal_median_seconds_to_shot"] = float(normal["seconds_to_first_shot"].median())
    out["clutch_three_share"] = float((c["first_shot_value"] == 3).mean())
    out["normal_three_share"] = float((normal["first_shot_value"] == 3).mean())
    out["clutch_tov_rate"] = float((c["tov"] > 0).mean())
    out["normal_tov_rate"] = float((normal["tov"] > 0).mean())
    # trailing vs leading
    c["state"] = np.where(c["margin_at_start"] < 0, "trailing", np.where(c["margin_at_start"] > 0, "leading", "tied"))
    out["by_state"] = {k: {"ppp": float(g["points"].mean()), "median_sec": float(g["seconds_to_first_shot"].median()), "three_share": float((g["first_shot_value"] == 3).mean()), "n": int(len(g))} for k, g in c.groupby("state")}
    return out


def main() -> None:
    p = L.load_possessions()
    pbp = L.load_pbp()
    res = {"games": int(p["game_id"].nunique())}
    for name, fn in [
        ("set_plays", lambda: set_plays(p)),
        ("discipline", lambda: discipline(pbp, p)),
        ("foul_trouble", lambda: foul_trouble(pbp)),
        ("bench_windows", lambda: bench_windows(p)),
        ("hot_hand", lambda: hot_hand(pbp)),
        ("ft_icing", lambda: ft_icing(pbp)),
        ("timeouts_left", lambda: timeouts_left(pbp)),
        ("clutch_clock", lambda: clutch_clock(p)),
    ]:
        print(f"== {name}", flush=True)
        try:
            res[name] = fn()
            print(json.dumps(res[name], indent=1, default=str)[:7000])
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            res[name] = {"error": repr(exc)}
    (OUT / "more.json").write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
