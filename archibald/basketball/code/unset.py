"""Two follow-ups.

1. Offensive rebounding structure. No player positions exist, so the proxy for
   bodies sent to the glass is offensive rebound chances (a player within 3.5 ft
   of a missed shot) per own miss, split by the crasher's position. Tests what
   each kind of crasher buys in rebounds and costs in opponent transition.

2. What leaves a defense unset. Uses the possessions rebuilt from play-by-play
   with the detail of how the previous possession ended.

Writes results/unset/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

DSN = "postgresql://nathanclark@127.0.0.1:5432/nba"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "unset"
EARLY = 8.0


def read(sql: str) -> pd.DataFrame:
    with psycopg.connect(DSN) as conn:
        return pd.read_sql(sql, conn)


def ols(X, y, w=None):
    X = np.column_stack([np.ones(len(y)), X])
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    sw = np.sqrt(w)
    b, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    r = y - X @ b
    s2 = np.sum(w * r * r) / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv((X * sw[:, None]).T @ (X * sw[:, None]))))
    return b, se


def demean(df, cols, by="team_tricode"):
    out = df.copy()
    for c in cols:
        out[c] = df[c] - df.groupby(by)[c].transform("mean")
    return out


def pos_class(p: str) -> str:
    p = str(p)
    if p in ("C", "C-F", "F-C"):
        return "big"
    if p in ("G", "G-F"):
        return "guard"
    if p in ("F", "F-G"):
        return "forward"
    return "unknown"


def crash_structure(team_games: pd.DataFrame) -> dict:
    roster = read("SELECT player_id, position FROM league.rosters").drop_duplicates("player_id")
    roster["cls"] = roster["position"].map(pos_class)
    tp = read(
        """
        SELECT game_id, team_tricode, person_id, rebound_chances_offensive AS och, rebound_chances_defensive AS dch, minutes
        FROM games.box_tracking_player WHERE minutes LIKE '%:%'
        """
    )
    tp = tp.merge(roster[["player_id", "cls"]], left_on="person_id", right_on="player_id", how="left")
    tp["cls"] = tp["cls"].fillna("unknown")
    pivot = tp.pivot_table(index=["game_id", "team_tricode"], columns="cls", values="och", aggfunc="sum", fill_value=0).reset_index()
    pivot.columns = ["game_id", "team_tricode"] + [f"och_{c}" for c in pivot.columns[2:]]
    trad = read(
        """
        SELECT game_id, team_tricode, field_goals_attempted - field_goals_made AS misses,
               free_throws_attempted - free_throws_made AS ft_misses
        FROM games.box_traditional_team_1
        """
    )
    misc = read("SELECT game_id, team_tricode, points_fast_break AS fb_pts, points_second_chance AS second_chance_pts FROM games.box_misc_team")
    f = team_games.merge(pivot, on=["game_id", "team_tricode"]).merge(trad, on=["game_id", "team_tricode"]).merge(misc, on=["game_id", "team_tricode"], how="left")
    opp_fb = misc.rename(columns={"team_tricode": "opponent", "fb_pts": "opp_fb_pts", "second_chance_pts": "opp_second_chance"})
    f = f.merge(opp_fb[["game_id", "opponent", "opp_fb_pts"]], on=["game_id", "opponent"], how="left")
    for c in ("och_big", "och_forward", "och_guard"):
        if c not in f.columns:
            f[c] = 0.0
    f["och_total"] = f[["och_big", "och_forward", "och_guard"]].sum(axis=1)
    f["crash_per_miss"] = f["och_total"] / f["misses"]
    for c in ("big", "forward", "guard"):
        f[f"{c}_per_miss"] = f[f"och_{c}"] / f["misses"]
    f["opp_fb_per_100"] = 100 * f["opp_fb_pts"] / f["d_poss"]
    f["second_chance_per_100"] = 100 * f["second_chance_pts"] / f["o_poss"]
    f = f.replace([np.inf, -np.inf], np.nan)
    cols = ["crash_per_miss", "big_per_miss", "forward_per_miss", "guard_per_miss", "oreb_pct", "d_ppp_after_dreb",
            "d_early_after_dreb_share", "opp_fb_per_100", "second_chance_per_100", "net_rating", "offensive_rating", "defensive_rating", "d_ppp"]
    f = f.dropna(subset=cols)
    dm = demean(f, cols)
    out = {
        "team_games": int(len(f)),
        "league_mean_crashers_per_miss": float(f["crash_per_miss"].mean()),
        "league_mean_by_position_per_miss": {c: float(f[f"{c}_per_miss"].mean()) for c in ("big", "forward", "guard")},
        "sd_within_team_crash_per_miss": float(dm["crash_per_miss"].std()),
        "team_spread": f.groupby("team_tricode")[["crash_per_miss", "big_per_miss", "forward_per_miss", "guard_per_miss", "oreb_pct", "opp_fb_per_100", "d_ppp_after_dreb", "net_rating"]].mean().round(3).sort_values("crash_per_miss").reset_index().to_dict("records"),
    }
    # within team: what one more crasher per miss buys and costs
    y_cols = ["oreb_pct", "second_chance_per_100", "d_early_after_dreb_share", "d_ppp_after_dreb", "opp_fb_per_100", "d_ppp", "offensive_rating", "defensive_rating", "net_rating"]
    res = {}
    for y in y_cols:
        b, se = ols(dm[["crash_per_miss"]].to_numpy(), dm[y].to_numpy())
        res[y] = {"per_extra_crasher_per_miss": float(b[1]), "se": float(se[1])}
    out["within_team_total_crashers"] = res
    res = {}
    X = dm[["big_per_miss", "forward_per_miss", "guard_per_miss"]].to_numpy()
    for y in y_cols:
        b, se = ols(X, dm[y].to_numpy())
        res[y] = {k: {"coef": float(b[i + 1]), "se": float(se[i + 1])} for i, k in enumerate(("big", "forward", "guard"))}
    out["within_team_by_position"] = res
    # diminishing returns: oreb per chance as a function of chances per miss
    f["oreb_per_chance"] = f["oreb"] / f["och_total"]
    dm2 = demean(f.dropna(subset=["oreb_per_chance"]), ["oreb_per_chance", "crash_per_miss"])
    b, se = ols(dm2[["crash_per_miss"]].to_numpy(), dm2["oreb_per_chance"].to_numpy())
    out["conversion_vs_crashers_within_team"] = {"slope": float(b[1]), "se": float(se[1])}
    # across teams, season
    season = read(
        """
        SELECT r.team_abbreviation AS team, r.oreb_chances, r.oreb, r.oreb_chance_pct, r.avg_oreb_dist,
               a.oreb_pct, a.off_rating, a.def_rating, a.net_rating, m.opp_pts_fb, m.pts_2_nd_chance
        FROM league.tracking_rebounding_team_regular_season r
        JOIN league.team_advanced_regular_season a ON a.team_id = r.team_id
        JOIN league.team_misc_regular_season m ON m.team_id = r.team_id
        """
    )
    out["across_teams_season"] = {
        "chances_vs_oreb_pct": float(season["oreb_chances"].corr(season["oreb_pct"])),
        "chances_vs_conversion": float(season["oreb_chances"].corr(season["oreb_chance_pct"])),
        "chances_vs_opp_fast_break_pts": float(season["oreb_chances"].corr(season["opp_pts_fb"])),
        "chances_vs_def_rating": float(season["oreb_chances"].corr(season["def_rating"])),
        "chances_vs_net_rating": float(season["oreb_chances"].corr(season["net_rating"])),
        "chances_vs_second_chance_pts": float(season["oreb_chances"].corr(season["pts_2_nd_chance"])),
        "teams": season.sort_values("oreb_chances")[["team", "oreb_chances", "oreb_chance_pct", "oreb_pct", "pts_2_nd_chance", "opp_pts_fb", "def_rating", "net_rating"]].round(3).to_dict("records"),
    }
    return out


def unset_defense(poss: pd.DataFrame, team_games: pd.DataFrame) -> dict:
    roster = read("SELECT player_id, position FROM league.rosters").drop_duplicates("player_id")
    roster["cls"] = roster["position"].map(pos_class)
    cls = roster.set_index("player_id")["cls"]
    live = poss[poss["ended_by"] != "period_end"].copy()
    live = live[~(((live["period"] >= 4) & (live["start_remaining"] < 180)) | (live["margin_at_start"].abs() > 20))]
    live["early"] = (live["seconds_to_first_action"] <= EARLY).astype(int)
    live["sec"] = live["seconds_to_first_action"]
    live["reb_cls"] = live["prev_rebounder"].map(cls).fillna("unknown")
    live["actor_cls"] = live["prev_actor"].map(cls).fillna("unknown")

    def summarize(d):
        return {"poss": int(len(d)), "early_share": float(d["early"].mean()), "ppp": float(d["points"].mean()),
                "ppp_early": float(d.loc[d["early"] == 1, "points"].mean()) if d["early"].sum() else None,
                "ppp_not_early": float(d.loc[d["early"] == 0, "points"].mean()),
                "median_sec": float(d["sec"].median())}

    out = {"possessions": int(len(live)), "early_definition_seconds": EARLY}
    out["by_start_type"] = {k: summarize(d) for k, d in live.groupby("start_type")}
    # overall share of early possessions contributed by each start type
    e = live[live["early"] == 1].groupby("start_type").size()
    out["where_early_possessions_come_from"] = (e / e.sum()).round(3).to_dict()
    out["start_type_mix"] = live["start_type"].value_counts(normalize=True).round(3).to_dict()

    dreb = live[live["start_type"] == "def_reb"].copy()
    dreb["miss_type"] = np.select(
        [dreb["prev_shot_value"] == 3, dreb["prev_shot_dist"] >= 15, dreb["prev_shot_dist"] >= 5],
        ["missed 3", "missed long 2 (15+ ft)", "missed mid 2 (5-15 ft)"], "missed shot at rim (<5 ft)")
    out["after_def_rebound_by_miss_type"] = {k: summarize(d) for k, d in dreb.groupby("miss_type")}
    out["after_def_rebound_by_rebounder"] = {k: summarize(d) for k, d in dreb.groupby("reb_cls") if k != "unknown" or len(d) > 500}
    out["after_def_rebound_by_rebounder_and_miss"] = {
        f"{r} | {m}": summarize(d) for (r, m), d in dreb.groupby(["reb_cls", "miss_type"]) if r != "unknown" and len(d) >= 300
    }
    ft = live[live["start_type"] == "def_reb_ft"]
    out["after_missed_free_throw_rebound"] = summarize(ft)
    steal = live[live["start_type"] == "steal"]
    out["after_steal_by_stealer"] = {k: summarize(d) for k, d in steal.groupby("reb_cls") if k != "unknown"}
    made = live[live["start_type"] == "made_fg"].copy()
    made["make_type"] = np.where(made["prev_shot_value"] == 3, "after made 3", "after made 2")
    out["after_made_shot"] = {k: summarize(d) for k, d in made.groupby("make_type")}
    out["after_made_shot_timeout_or_foul"] = {
        "clean": summarize(made[(made["timeout_before_by"] == "") & (~made["dead_foul_before"].astype(bool))]),
        "timeout_or_dead_foul_between": summarize(made[(made["timeout_before_by"] != "") | (made["dead_foul_before"].astype(bool))]),
    }
    # is the early advantage after a rebound the rebounder pushing himself? first action by the rebounder
    dreb["rebounder_first"] = (dreb["prev_rebounder"] == dreb["prev_actor"])
    # (prev_actor for def_reb is the shooter who missed, so skip; keep for steals)

    # team level: what predicts a team's own early share and the opponent's early share, within team
    tg = team_games.copy()
    tg["steal_starts"] = tg["o_steal_start_share"]
    tg["dreb_starts"] = tg["o_poss"] * 0  # placeholder to keep columns
    # per team-game from possessions: guard-rebound share, 3-miss share of rebounds
    pg = []
    for (gid, side), d in live.groupby(["game_id", "side"]):
        dr = d[d["start_type"] == "def_reb"]
        pg.append({
            "game_id": gid, "side": side,
            "guard_reb_share": float((dr["reb_cls"] == "guard").mean()) if len(dr) else np.nan,
            "big_reb_share": float((dr["reb_cls"] == "big").mean()) if len(dr) else np.nan,
            "three_miss_share": float((dr["prev_shot_value"] == 3).mean()) if len(dr) else np.nan,
            "dreb_start_share": float((d["start_type"].isin(["def_reb", "def_reb_ft"])).mean()),
            "steal_share": float((d["start_type"] == "steal").mean()),
            "dead_start_share": float((d["start_type"].isin(["made_fg", "made_ft", "dead_tov"])).mean()),
            "early_share": float(d["early"].mean()),
            "early_after_dreb": float(dr["early"].mean()) if len(dr) else np.nan,
            "ppp": float(d["points"].mean()),
        })
    pg = pd.DataFrame(pg)
    tg = tg.merge(pg, on=["game_id", "side"])
    cols = ["guard_reb_share", "big_reb_share", "three_miss_share", "dreb_start_share", "steal_share", "dead_start_share", "early_share", "early_after_dreb", "ppp", "deflections", "steals", "offensive_rating", "net_rating", "pace"]
    tg = tg.dropna(subset=cols)
    dm = demean(tg, cols)
    res = {}
    for x in ["guard_reb_share", "big_reb_share", "three_miss_share", "dreb_start_share", "steal_share", "dead_start_share", "deflections", "steals"]:
        row = {}
        for y in ["early_share", "early_after_dreb", "ppp", "offensive_rating"]:
            b, se = ols(dm[[x]].to_numpy(), dm[y].to_numpy())
            row[y] = {"coef": float(b[1]), "se": float(se[1])}
        row["sd_within_team"] = float(dm[x].std())
        res[x] = row
    out["team_level_within_team"] = res
    # how much of a team's offensive rating is explained by early share alone
    b, se = ols(dm[["early_share"]].to_numpy(), dm["offensive_rating"].to_numpy())
    out["offensive_rating_per_pp_early_share"] = {"coef": float(b[1]) / 100, "se": float(se[1]) / 100}
    out["team_spread_early_share"] = tg.groupby("team_tricode")[["early_share", "early_after_dreb", "steal_share", "dreb_start_share", "offensive_rating"]].mean().round(3).sort_values("early_share").reset_index().to_dict("records")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    poss = pd.read_csv(ROOT / "results" / "scoreboard" / "possessions.csv", dtype={"game_id": str})
    tg = pd.read_csv(ROOT / "results" / "scoreboard" / "team_games.csv", dtype={"game_id": str})
    summary = {"crash_structure": crash_structure(tg), "unset_defense": unset_defense(poss, tg)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
