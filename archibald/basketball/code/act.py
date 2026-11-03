"""Per-team report, the rebounding split, and the situation decision rule.

The decision rule is a table, not a black box: for each way a possession
starts, the points per possession of each thing the team actually did,
estimated on one half of the games and checked on the other half.

Writes results/act/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

DSN = "postgresql://nathanclark@127.0.0.1:5432/nba"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "act"
EARLY = 8.0


def read(sql: str) -> pd.DataFrame:
    with psycopg.connect(DSN) as conn:
        return pd.read_sql(sql, conn)


def ols(X, y):
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    y = np.asarray(y, float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    dof = max(len(y) - X.shape[1], 1)
    s2 = float(resid @ resid) / dof
    try:
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    except np.linalg.LinAlgError:
        se = np.full(X.shape[1], np.nan)
    return b, se


def demean(df, cols, by):
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


def zone(dist, value):
    if value == 3:
        return "three"
    if dist < 5:
        return "rim"
    return "other"


def live_possessions() -> pd.DataFrame:
    poss = pd.read_csv(ROOT / "results" / "scoreboard" / "possessions.csv", dtype={"game_id": str})
    teams = pd.read_csv(
        ROOT / "results" / "scoreboard" / "team_games.csv",
        dtype={"game_id": str},
        usecols=["game_id", "side", "team_tricode", "possessions", "oreb_pct", "offensive_rating", "defensive_rating", "net_rating"],
    )
    poss = poss.merge(teams, on=["game_id", "side"], how="left")
    poss = poss[(poss["ended_by"] != "period_end")].copy()
    junk = ((poss["period"] >= 4) & (poss["start_remaining"] < 180)) | (poss["margin_at_start"].abs() > 20)
    poss = poss[~junk].copy()
    poss["shot_early"] = poss["seconds_to_first_shot"] <= EARLY
    poss["pushed"] = poss["seconds_to_first_action"] <= EARLY
    poss["zone"] = [
        zone(d, v) if shot else "no_shot"
        for d, v, shot in zip(poss["first_shot_dist"], poss["first_shot_value"], poss["shot_early"])
    ]
    # A push is any first action within 8 seconds. The shot, if there is one, is a separate choice.
    poss["choice"] = np.select(
        [
            poss["shot_early"] & (poss["zone"] == "rim"),
            poss["shot_early"] & (poss["zone"] == "three"),
            poss["shot_early"] & (poss["zone"] == "other"),
            poss["pushed"] & (poss["first_action"] == "Turnover"),
            poss["pushed"],
        ],
        ["early_rim", "early_three", "early_other", "early_turnover", "pushed_then_settled"],
        default="walked",
    )
    short = (poss["start_type"] == "def_reb") & (poss["prev_was_ft"] != 1) & (poss["prev_shot_dist"] < 10)
    long = (poss["start_type"] == "def_reb") & (poss["prev_was_ft"] != 1) & (poss["prev_shot_dist"] >= 10)
    poss["situation"] = np.select(
        [poss["start_type"] == "steal", short, long, poss["start_type"].isin(["made_fg", "made_ft"])],
        ["after_steal", "after_short_miss", "after_long_miss", "after_opponent_score"],
        default="other",
    )
    poss["half"] = np.where(poss["game_id"].str[-1].astype(int) % 2 == 0, "even", "odd")
    return poss


def cell_means(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["situation", "choice"])["points"]
    out = g.agg(ppp="mean", n="count", sd="std").reset_index()
    out["se"] = out["sd"] / np.sqrt(out["n"])
    return out


def _half_mean(d: pd.DataFrame) -> dict:
    return {"ppp": float(d["points"].mean()), "n": int(len(d)), "se": float(d["points"].sem()) if len(d) else np.nan}


def decision_rule(poss: pd.DataFrame) -> dict:
    """Two separate decisions, each checked on even-numbered games and on odd-numbered games.

    Push: do something within 8 seconds, versus let the defense get set.
    Shot: given an early shot, rim versus three versus floater, against walking it up.
    A turnover is the cost of a push that never became a shot, not a choice of its own.
    """
    situations = ["after_steal", "after_short_miss", "after_long_miss", "after_opponent_score"]
    shot_choices = ["early_rim", "early_three", "early_other"]
    cells = []
    push = []
    for sit in situations:
        d = poss[poss["situation"] == sit]
        for half, dh in d.groupby("half"):
            pushed = dh[dh["pushed"]]
            walked = dh[~dh["pushed"]]
            push.append({"situation": sit, "half": half, "which": "pushed", **_half_mean(pushed)})
            push.append({"situation": sit, "half": half, "which": "walked", **_half_mean(walked)})
            for ch in shot_choices + ["early_turnover", "pushed_then_settled", "walked"]:
                cells.append({"situation": sit, "half": half, "choice": ch, **_half_mean(dh[dh["choice"] == ch])})
    cell = pd.DataFrame(cells)
    pushed = pd.DataFrame(push)

    def both(frame, sit, key, col):
        out = {}
        for half in ("even", "odd"):
            row = frame[(frame["situation"] == sit) & (frame[col] == key) & (frame["half"] == half)]
            out[half] = None if row.empty else row.iloc[0]
        return out

    shot_rules = []
    worse = []
    for sit in situations:
        winners = {}
        for half in ("even", "odd"):
            sub = cell[(cell.situation == sit) & (cell.half == half) & (cell.choice.isin(shot_choices)) & (cell.n >= 80)]
            winners[half] = None if sub.empty else sub.loc[sub["ppp"].idxmax(), "choice"]
        shot_rules.append({
            "situation": sit,
            "best_early_shot_even": winners["even"],
            "best_early_shot_odd": winners["odd"],
            "halves_agree": winners["even"] is not None and winners["even"] == winners["odd"],
        })
        walked = both(cell, sit, "walked", "choice")
        for ch in shot_choices:
            shot = both(cell, sit, ch, "choice")
            gaps = []
            ok = True
            for half in ("even", "odd"):
                if shot[half] is None or walked[half] is None or shot[half]["n"] < 80 or walked[half]["n"] < 80:
                    ok = False
                    break
                gap = float(walked[half]["ppp"] - shot[half]["ppp"])
                noise = float(np.sqrt(shot[half]["se"] ** 2 + walked[half]["se"] ** 2))
                if gap <= noise:
                    ok = False
                    break
                gaps.append(gap)
            if ok:
                worse.append({"situation": sit, "choice": ch, "worse_than_walking_by": float(np.mean(gaps))})
    push_rules = []
    for sit in situations:
        gaps = []
        detail = {}
        for half in ("even", "odd"):
            a = pushed[(pushed.situation == sit) & (pushed.half == half) & (pushed.which == "pushed")].iloc[0]
            b = pushed[(pushed.situation == sit) & (pushed.half == half) & (pushed.which == "walked")].iloc[0]
            gap = float(a["ppp"] - b["ppp"])
            noise = float(np.sqrt(a["se"] ** 2 + b["se"] ** 2))
            detail[half] = {"pushed": float(a["ppp"]), "walked": float(b["ppp"]), "gap": gap, "noise": noise, "n_pushed": int(a["n"]), "n_walked": int(b["n"])}
            gaps.append(gap)
        push_rules.append({
            "situation": sit,
            "gap_pushed_minus_walked": float(np.mean(gaps)),
            "same_sign_both_halves": bool(gaps[0] * gaps[1] > 0),
            "clears_the_noise_both_halves": bool(abs(detail["even"]["gap"]) > detail["even"]["noise"] and abs(detail["odd"]["gap"]) > detail["odd"]["noise"] and gaps[0] * gaps[1] > 0),
            "halves": detail,
        })
    wide = cell.pivot_table(index=["situation", "choice"], columns="half", values=["ppp", "n", "se"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    return {
        "cells": wide.reset_index().round(3).to_dict("records"),
        "best_early_shot": shot_rules,
        "worse_than_walking_it_up_on_both_halves": worse,
        "push_versus_walk": push_rules,
    }


def crash_frame() -> pd.DataFrame:
    roster = read("SELECT player_id, position FROM league.rosters").drop_duplicates("player_id")
    roster["cls"] = roster["position"].map(pos_class)
    tp = read(
        """
        SELECT game_id, team_tricode, person_id, rebound_chances_offensive AS och, minutes
        FROM games.box_tracking_player
        WHERE minutes LIKE '%:%'
        """
    )
    tp = tp.merge(roster[["player_id", "cls"]], left_on="person_id", right_on="player_id", how="left")
    tp["cls"] = tp["cls"].fillna("unknown")
    # rank players who were actually near a miss. 1 is the player nearest most often.
    active = tp[tp["och"] > 0].copy()
    active["rank"] = active.groupby(["game_id", "team_tricode"])["och"].rank(ascending=False, method="first")
    pivot = tp.pivot_table(index=["game_id", "team_tricode"], columns="cls", values="och", aggfunc="sum", fill_value=0).reset_index()
    pivot.columns = ["game_id", "team_tricode"] + [f"och_{c}" for c in pivot.columns[2:]]
    ranks = active[active["rank"].isin([1, 2, 3])][["game_id", "team_tricode", "rank", "cls"]]
    wide = ranks.pivot_table(index=["game_id", "team_tricode"], columns="rank", values="cls", aggfunc="first").reset_index()
    names = {1: "closest", 2: "second", 3: "third", 1.0: "closest", 2.0: "second", 3.0: "third"}
    wide = wide.rename(columns=names)
    trad = read(
        """
        SELECT game_id, team_tricode,
               field_goals_attempted - field_goals_made AS misses,
               three_pointers_attempted - three_pointers_made AS three_misses
        FROM games.box_traditional_team_1
        """
    )
    adv = read(
        """
        SELECT game_id, team_tricode, offensive_rebound_percentage AS oreb_pct,
               offensive_rating, defensive_rating, net_rating, possessions
        FROM games.box_advanced_team
        """
    )
    misc = read("SELECT game_id, team_tricode, points_fast_break AS fb_pts FROM games.box_misc_team")
    sched = read(
        """
        SELECT game_id, home_team_team_tricode AS home, away_team_team_tricode AS away
        FROM league.schedule_0
        """
    ).drop_duplicates("game_id")
    f = pivot.merge(wide, on=["game_id", "team_tricode"], how="left").merge(trad, on=["game_id", "team_tricode"])
    f = f.merge(adv, on=["game_id", "team_tricode"]).merge(misc, on=["game_id", "team_tricode"])
    f = f.merge(sched, on="game_id")
    f["opponent"] = np.where(f["team_tricode"] == f["home"], f["away"], f["home"])
    opp = misc.rename(columns={"team_tricode": "opponent", "fb_pts": "opp_fb_pts"})
    f = f.merge(opp, on=["game_id", "opponent"], how="left")
    for c in ("och_big", "och_forward", "och_guard"):
        if c not in f.columns:
            f[c] = 0.0
    f["misses"] = f["misses"].replace(0, np.nan)
    f["crash_per_miss"] = (f["och_big"] + f["och_forward"] + f["och_guard"]) / f["misses"]
    for c in ("big", "forward", "guard"):
        f[f"{c}_per_miss"] = f[f"och_{c}"] / f["misses"]
    f["three_miss_share"] = f["three_misses"] / f["misses"]
    f["opp_fb_per_100"] = 100 * f["opp_fb_pts"] / f["possessions"]
    f["second_is_guard"] = np.where(f["second"].isna(), np.nan, (f["second"] == "guard").astype(float))
    f["third_is_guard"] = np.where(f["third"].isna(), np.nan, (f["third"] == "guard").astype(float))
    return f.replace([np.inf, -np.inf], np.nan)


def rebounding(f: pd.DataFrame) -> dict:
    cols = ["big_per_miss", "forward_per_miss", "guard_per_miss", "crash_per_miss", "three_miss_share",
            "second_is_guard", "oreb_pct", "opp_fb_per_100", "defensive_rating", "net_rating", "offensive_rating"]
    d = f.dropna(subset=cols).copy()
    dm = demean(d, cols, "team_tricode")
    out = {"team_games": int(len(d))}
    # who the extra body is, holding the total number of bodies fixed
    res = {}
    X = dm[["second_is_guard", "crash_per_miss"]].to_numpy()
    for y in ("oreb_pct", "opp_fb_per_100", "defensive_rating", "net_rating"):
        b, se = ols(X, dm[y].to_numpy())
        res[y] = {"second_closest_is_a_guard": {"coef": float(b[1]), "se": float(se[1])}}
    out["marginal_body"] = res
    out["how_often_second_closest_is_a_guard"] = float(d["second_is_guard"].mean())
    out["closest_mix"] = d["closest"].value_counts(normalize=True).round(3).to_dict()
    out["second_mix"] = d["second"].value_counts(normalize=True).round(3).to_dict()
    # split the position effect by whether the misses were mostly threes
    med = float(d["three_miss_share"].median())
    split = {}
    for label, part in (("mostly_twos", d[d["three_miss_share"] <= med]), ("mostly_threes", d[d["three_miss_share"] > med])):
        dm2 = demean(part, ["big_per_miss", "forward_per_miss", "guard_per_miss", "oreb_pct", "opp_fb_per_100", "defensive_rating", "net_rating"], "team_tricode")
        block = {}
        X = dm2[["big_per_miss", "forward_per_miss", "guard_per_miss"]].to_numpy()
        for y in ("oreb_pct", "opp_fb_per_100", "defensive_rating", "net_rating"):
            b, se = ols(X, dm2[y].to_numpy())
            block[y] = {k: {"coef": float(b[i + 1]), "se": float(se[i + 1])} for i, k in enumerate(("big", "forward", "guard"))}
        block["team_games"] = int(len(part))
        block["mean_three_miss_share"] = float(part["three_miss_share"].mean())
        split[label] = block
    out["by_miss_mix"] = split
    out["three_miss_share_median"] = med
    return out


def missed_shot_rebounds() -> dict:
    """Who actually comes up with a missed two versus a missed three."""
    pbp = read(
        """
        SELECT game_id, action_number, period, clock, team_id, person_id, action_type,
               shot_value, shot_distance, score_home, score_away, description
        FROM games.pbp
        WHERE game_id IN (SELECT game_id FROM games.loaded)
          AND action_type IN ('Made Shot', 'Missed Shot', 'Rebound')
        """
    )
    roster = read("SELECT player_id, position FROM league.rosters").drop_duplicates("player_id")
    roster["cls"] = roster["position"].map(pos_class)
    cls = roster.set_index("player_id")["cls"]
    rows = []
    for _, g in pbp.groupby("game_id", sort=False):
        g = g.sort_values("action_number")
        score_h, score_a = 0.0, 0.0
        pending = None
        for rec in g.itertuples(index=False):
            if rec.action_type == "Made Shot" and rec.score_home not in (None, ""):
                try:
                    score_h, score_a = float(rec.score_home), float(rec.score_away)
                except ValueError:
                    pass
            if rec.action_type == "Missed Shot":
                pending = rec
                continue
            if rec.action_type != "Rebound" or pending is None:
                continue
            margin = abs(score_h - score_a)
            clock = str(rec.clock)
            mins = 99
            if clock.startswith("PT") and "M" in clock:
                mins = int(clock.split("M")[0][2:] or 0)
            junk = rec.period >= 4 and mins < 3 and margin > 20
            dist = pending.shot_distance
            if not junk and dist is not None and dist <= 40:
                same = int(rec.team_id) == int(pending.team_id)
                kind = "three" if int(pending.shot_value or 2) == 3 else ("rim" if dist < 5 else "two")
                try:
                    pid = int(rec.person_id)
                except (TypeError, ValueError):
                    pid = None
                if pid is not None and pid == int(rec.team_id):
                    who = "team"
                else:
                    who = cls.get(pid, "unknown")
                rows.append({"kind": kind, "oreb": int(same), "who": who if same else None})
            pending = None
    ev = pd.DataFrame(rows)
    out = {}
    for kind, d in ev.groupby("kind"):
        got = d[d["oreb"] == 1]
        out[kind] = {
            "misses": int(len(d)),
            "offensive_rebound_rate": float(d["oreb"].mean()),
            "who_got_it": got["who"].value_counts(normalize=True).round(3).to_dict(),
        }
    return out


def team_report(poss: pd.DataFrame, crash: pd.DataFrame, rule: dict) -> pd.DataFrame:
    """Points per game a team would pick up by stopping the pushes that lose points.

    Only situations where pushing lost to walking it up on both halves are counted.
    Extra pushes are not given a point value here: a push the team did not make
    is not a layup they turned down.
    """
    lose = {r["situation"]: r["gap_pushed_minus_walked"] for r in rule["push_versus_walk"] if r["gap_pushed_minus_walked"] < 0 and r["same_sign_both_halves"]}
    rows = []
    crash_team = crash.groupby("team_tricode")[["crash_per_miss", "big_per_miss", "forward_per_miss", "guard_per_miss"]].mean()
    target_frontcourt = float((crash_team["big_per_miss"] + crash_team["forward_per_miss"]).quantile(0.75))
    for team, d in poss.groupby("team_tricode"):
        games = d["game_id"].nunique()
        ct = crash_team.loc[team] if team in crash_team.index else None
        front = float(ct["big_per_miss"] + ct["forward_per_miss"]) if ct is not None else np.nan
        lost_pts = 0.0
        for sit, gap in lose.items():
            n = int(((d["situation"] == sit) & d["pushed"]).sum())
            lost_pts += n * (-gap) / games
        rows.append({
            "team": team,
            "games": int(games),
            "push_rate_after_steal": float(d.loc[d.situation == "after_steal", "pushed"].mean()),
            "push_rate_after_short_miss": float(d.loc[d.situation == "after_short_miss", "pushed"].mean()),
            "push_rate_after_long_miss": float(d.loc[d.situation == "after_long_miss", "pushed"].mean()),
            "push_rate_after_opponent_score": float(d.loc[d.situation == "after_opponent_score", "pushed"].mean()),
            "early_floater_share_of_early_shots": float((d.loc[d.shot_early, "zone"] == "other").mean()),
            "frontcourt_per_miss": front,
            "guard_per_miss": float(ct["guard_per_miss"]) if ct is not None else np.nan,
            "frontcourt_gap_vs_top_quarter": target_frontcourt - front,
            "points_per_game_lost_pushing_when_it_loses": round(lost_pts, 2),
        })
    rep = pd.DataFrame(rows).sort_values("frontcourt_gap_vs_top_quarter", ascending=False)
    rep["frontcourt_target_per_miss"] = target_frontcourt
    return rep


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    poss = live_possessions()
    crash = crash_frame()
    rule = decision_rule(poss)
    reb = rebounding(crash)
    shots = missed_shot_rebounds()
    report = team_report(poss, crash, rule)
    report.to_csv(OUT / "team_report.csv", index=False)
    summary = {
        "decision_rule": rule,
        "rebounding": reb,
        "who_gets_the_rebound": shots,
        "league_push_rates": poss.groupby("situation")["pushed"].mean().round(3).to_dict(),
        "reading": (
            "Push means the first action of the possession came within 8 seconds. Walked means it did not. "
            "A recommendation has to show up on both the even-numbered games and the odd-numbered games. "
            "Frontcourt crashers are bigs and forwards within 3.5 feet of a miss. The top-quarter target is the "
            "75th percentile team, not a theoretical maximum."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("PUSH VERSUS WALK")
    for r in rule["push_versus_walk"]:
        print(f"  {r['situation']}: gap {r['gap_pushed_minus_walked']:+.3f} same_sign {r['same_sign_both_halves']} clears_noise {r['clears_the_noise_both_halves']} {r['halves']}")
    print("\nWORSE THAN WALKING")
    print(rule["worse_than_walking_it_up_on_both_halves"])
    print("\nBEST EARLY SHOT")
    print(pd.DataFrame(rule["best_early_shot"]).to_string(index=False))
    print("\nCELLS")
    print(pd.DataFrame(rule["cells"])[["situation", "choice", "ppp_even", "n_even", "ppp_odd", "n_odd"]].round(3).to_string(index=False))
    print("\nMARGINAL BODY (second-closest is a guard, holding total crashers fixed)")
    print(json.dumps(reb["marginal_body"], indent=2))
    print("closest", reb["closest_mix"], "second", reb["second_mix"], "share second is guard", round(reb["how_often_second_closest_is_a_guard"], 3))
    print("\nWHO GETS THE REBOUND")
    print(json.dumps(shots, indent=2))
    print("\nMISS MIX defensive_rating guard coef")
    for label, block in reb["by_miss_mix"].items():
        g = block["defensive_rating"]["guard"]
        b = block["defensive_rating"]["big"]
        print(f"  {label} n={block['team_games']} three-miss share {block['mean_three_miss_share']:.2f} guard def {g['coef']:.1f}±{g['se']:.1f} big def {b['coef']:.1f}±{b['se']:.1f} oreb guard {block['oreb_pct']['guard']['coef']:.3f} big {block['oreb_pct']['big']['coef']:.3f}")
    print("\nTEAMS, points per game available")
    cols = ["team", "games", "push_rate_after_steal", "push_rate_after_short_miss", "push_rate_after_long_miss",
            "push_rate_after_opponent_score", "early_floater_share_of_early_shots", "frontcourt_per_miss",
            "guard_per_miss", "frontcourt_gap_vs_top_quarter", "points_per_game_lost_pushing_when_it_loses"]
    print(report[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
