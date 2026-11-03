"""Check the two findings on 2024-25.

League tables cover all 30 teams. The possession and within-game crash checks
use the spread of games in schema s2425 (about one game in eight).

    python code/validate_2425.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scoreboard import build_possessions  # noqa: E402

DSN = "postgresql://nathanclark@127.0.0.1:5432/nba"
OUT = Path(__file__).resolve().parents[1] / "results" / "act"
EARLY = 8.0


def read(sql: str) -> pd.DataFrame:
    with psycopg.connect(DSN) as conn:
        return pd.read_sql(sql, conn)


def corr_p(x, y):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    r = float(d["x"].corr(d["y"]))
    n = len(d)
    if n < 4 or abs(r) >= 1:
        return {"r": r, "n": n, "p": None}
    t = r * np.sqrt((n - 2) / (1 - r * r))
    # two-sided p from a t with n-2 df, normal approximation is fine at n=30 but use a crude survival via scipy if present
    try:
        from math import erf, sqrt
        # Wilson-Hilferty-ish: use regularized beta is overkill. Student-t via scipy.
        from scipy.stats import t as student
        p = float(2 * student.sf(abs(t), n - 2))
    except Exception:
        p = None
    return {"r": round(r, 3), "n": n, "t": round(float(t), 2), "p": None if p is None else round(p, 3)}


def league() -> dict:
    specs = {
        "2024-25": ("s2425.tracking_rebounding", "s2425.team_advanced", "s2425.team_misc"),
        "2025-26": (
            "league.tracking_rebounding_team_regular_season",
            "league.team_advanced_regular_season",
            "league.team_misc_regular_season",
        ),
    }
    out = {}
    for season, (trk, adv, misc) in specs.items():
        df = read(
            f"""
            SELECT t.oreb_chances, a.oreb_pct, a.off_rating, a.def_rating, a.net_rating,
                   m.opp_pts_fb, m.pts_2_nd_chance
            FROM {trk} t
            JOIN {adv} a ON a.team_id = t.team_id
            JOIN {misc} m ON m.team_id = t.team_id
            """
        )
        out[season] = {col: corr_p(df["oreb_chances"], df[col]) for col in
                       ("oreb_pct", "off_rating", "def_rating", "net_rating", "opp_pts_fb", "pts_2_nd_chance")}
        out[season]["oreb_pct_vs_opp_pts_fb"] = corr_p(df["oreb_pct"], df["opp_pts_fb"])
        out[season]["oreb_pct_vs_def_rating"] = corr_p(df["oreb_pct"], df["def_rating"])
        out[season]["oreb_pct_vs_net_rating"] = corr_p(df["oreb_pct"], df["net_rating"])
    return out


def ols(X, y):
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    y = np.asarray(y, float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    dof = max(len(y) - X.shape[1], 1)
    s2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return b, se


def pos_class(p: str) -> str:
    p = str(p)
    if p in ("C", "C-F", "F-C"):
        return "big"
    if p in ("G", "G-F"):
        return "guard"
    if p in ("F", "F-G"):
        return "forward"
    return "unknown"


def games_ready() -> int:
    df = read("SELECT name FROM s2425.loaded")
    return int(df["name"].astype(str).str.fullmatch(r"\d{10}").sum()) if len(df) else 0


def push_check(poss: pd.DataFrame) -> dict:
    live = poss[poss["ended_by"] != "period_end"].copy()
    junk = ((live["period"] >= 4) & (live["start_remaining"] < 180)) | (live["margin_at_start"].abs() > 20)
    live = live[~junk]
    live["pushed"] = live["seconds_to_first_action"] <= EARLY
    short = (live["start_type"] == "def_reb") & (live["prev_was_ft"] != 1) & (live["prev_shot_dist"] < 10)
    long = (live["start_type"] == "def_reb") & (live["prev_was_ft"] != 1) & (live["prev_shot_dist"] >= 10)
    live["situation"] = np.select(
        [live["start_type"] == "steal", short, long, live["start_type"].isin(["made_fg", "made_ft"])],
        ["after_steal", "after_short_miss", "after_long_miss", "after_opponent_score"],
        default="other",
    )
    out = {}
    for sit, d in live.groupby("situation"):
        if sit == "other":
            continue
        a, b = d[d["pushed"]], d[~d["pushed"]]
        out[sit] = {
            "pushed_ppp": float(a["points"].mean()),
            "walked_ppp": float(b["points"].mean()),
            "gap": float(a["points"].mean() - b["points"].mean()),
            "n_pushed": int(len(a)),
            "n_walked": int(len(b)),
        }
    shot = live[live["seconds_to_first_shot"] <= EARLY].copy()
    shot["zone"] = np.where(shot["first_shot_value"] == 3, "three", np.where(shot["first_shot_dist"] < 5, "rim", "other"))
    zones = {}
    for sit in ("after_steal", "after_short_miss", "after_long_miss", "after_opponent_score"):
        d = shot[shot["situation"] == sit]
        zones[sit] = {z: {"ppp": float(dz["points"].mean()), "n": int(len(dz))} for z, dz in d.groupby("zone")}
    out["early_shot_zones"] = zones
    return out


def crash_check() -> dict:
    roster = read("SELECT player_id, position FROM s2425.rosters").drop_duplicates("player_id")
    roster["cls"] = roster["position"].map(pos_class)
    tp = read(
        """
        SELECT game_id, team_tricode, person_id, rebound_chances_offensive AS och
        FROM s2425.box_tracking_player WHERE minutes LIKE '%:%'
        """
    )
    tp = tp.merge(roster[["player_id", "cls"]], left_on="person_id", right_on="player_id", how="left")
    tp["cls"] = tp["cls"].fillna("unknown")
    pivot = tp.pivot_table(index=["game_id", "team_tricode"], columns="cls", values="och", aggfunc="sum", fill_value=0).reset_index()
    pivot.columns = ["game_id", "team_tricode"] + [f"och_{c}" for c in pivot.columns[2:]]
    trad = read(
        """
        SELECT game_id, team_tricode, field_goals_attempted - field_goals_made AS misses
        FROM s2425.box_traditional_team_1
        """
    )
    adv = read(
        """
        SELECT game_id, team_tricode, offensive_rebound_percentage AS oreb_pct,
               offensive_rating, defensive_rating, net_rating, possessions
        FROM s2425.box_advanced_team
        """
    )
    misc = read(
        """
        SELECT game_id, team_tricode, opp_points_fast_break AS opp_fb
        FROM s2425.box_misc_team
        """
    )
    f = pivot.merge(trad, on=["game_id", "team_tricode"]).merge(adv, on=["game_id", "team_tricode"]).merge(misc, on=["game_id", "team_tricode"])
    for c in ("och_big", "och_forward", "och_guard"):
        if c not in f.columns:
            f[c] = 0.0
    f = f[f["misses"] > 0].copy()
    f["crash_per_miss"] = (f["och_big"] + f["och_forward"] + f["och_guard"]) / f["misses"]
    for c in ("big", "forward", "guard"):
        f[f"{c}_per_miss"] = f[f"och_{c}"] / f["misses"]
    f["opp_fb_per_100"] = 100 * f["opp_fb"] / f["possessions"]
    cols = ["crash_per_miss", "big_per_miss", "forward_per_miss", "guard_per_miss", "oreb_pct", "opp_fb_per_100", "defensive_rating", "net_rating", "offensive_rating"]
    f = f.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
    dm = f.copy()
    for c in cols:
        dm[c] = f[c] - f.groupby("team_tricode")[c].transform("mean")
    # teams with a single game contribute nothing to a within-team estimate; drop all-zero rows
    dm = dm[(dm[cols].abs().sum(axis=1) > 0)]
    out = {"team_games": int(len(f)), "teams": int(f["team_tricode"].nunique()), "mean_crash_per_miss": float(f["crash_per_miss"].mean())}
    res = {}
    for y in ("oreb_pct", "opp_fb_per_100", "defensive_rating", "offensive_rating", "net_rating"):
        b, se = ols(dm[["crash_per_miss"]].to_numpy(), dm[y].to_numpy())
        res[y] = {"per_extra_crasher": float(b[1]), "se": float(se[1])}
    out["within_team_total"] = res
    res = {}
    X = dm[["big_per_miss", "forward_per_miss", "guard_per_miss"]].to_numpy()
    for y in ("oreb_pct", "opp_fb_per_100", "defensive_rating", "net_rating"):
        b, se = ols(X, dm[y].to_numpy())
        res[y] = {k: {"coef": float(b[i + 1]), "se": float(se[i + 1])} for i, k in enumerate(("big", "forward", "guard"))}
    out["within_team_by_position"] = res
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"league_across_teams": league(), "games_marked_loaded": games_ready()}
    print("LEAGUE")
    for season, block in result["league_across_teams"].items():
        print(season)
        for k, v in block.items():
            print(f"  {k}: {v}")
    n = result["games_marked_loaded"]
    if n < 80:
        result["games"] = f"only {n} games loaded; possession check skipped"
        print(result["games"])
    else:
        print(f"building possessions from {n} games")
        pbp = read("SELECT * FROM s2425.pbp")
        poss = build_possessions(pbp)
        result["push"] = push_check(poss)
        result["crash"] = crash_check()
        print("PUSH")
        for k, v in result["push"].items():
            if k == "early_shot_zones":
                continue
            print(f"  {k}: pushed {v['pushed_ppp']:.3f} walked {v['walked_ppp']:.3f} gap {v['gap']:+.3f} n {v['n_pushed']}/{v['n_walked']}")
        print("CRASH", json.dumps(result["crash"]["within_team_total"], indent=2))
        print("BY POSITION def", json.dumps(result["crash"]["within_team_by_position"]["defensive_rating"], indent=2))
        print("BY POSITION oreb", json.dumps(result["crash"]["within_team_by_position"]["oreb_pct"], indent=2))
    (OUT / "validate_2425.json").write_text(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
