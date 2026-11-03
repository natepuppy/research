"""What moves the margin. Possessions rebuilt from play-by-play for every
loaded game, then team-game features from the possessions and the box-score
tables, then within-team correlations with the game's net rating.

Writes results/scoreboard/possessions.csv, team_games.csv, summary.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

DSN = "postgresql://nathanclark@127.0.0.1:5432/nba"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "scoreboard"

OFFENSIVE = {"Made Shot", "Missed Shot", "Heave", "Turnover", "Free Throw"}
CLOCK = re.compile(r"PT(\d+)M([\d.]+)S")


def read(sql: str, params=None) -> pd.DataFrame:
    with psycopg.connect(DSN) as conn:
        return pd.read_sql(sql, conn, params=params)


def clock_seconds(text) -> float:
    m = CLOCK.match(str(text))
    if not m:
        return np.nan
    return int(m.group(1)) * 60 + float(m.group(2))


def period_len(period: int) -> int:
    return 720 if period <= 4 else 300


def build_possessions(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per possession. Team side is the pbp `location` (h/v)."""
    rows = []
    pbp = pbp.sort_values(["game_id", "action_number", "action_id"], kind="stable")
    for game_id, g in pbp.groupby("game_id", sort=False):
        remaining = g["clock"].map(clock_seconds).to_numpy()
        periods = g["period"].to_numpy()
        elapsed = np.array([period_len(p) - r for p, r in zip(periods, remaining)])
        atype = g["action_type"].to_numpy()
        sub = g["sub_type"].fillna("").to_numpy()
        loc = g["location"].fillna("").to_numpy()
        desc = g["description"].fillna("").to_numpy()
        shot_value = g["shot_value"].fillna(0).to_numpy()
        shot_result = g["shot_result"].fillna("").to_numpy()
        score_h = g["score_home"].to_numpy()
        score_v = g["score_away"].to_numpy()
        action_no = g["action_number"].to_numpy()

        cur = None  # current possession dict
        last_end_time = 0.0
        last_end_type = "period_start"
        last_detail = {}  # how the previous possession ended: shot distance, value, rebounder
        last_miss = {}  # most recent missed shot or free throw
        person = g["person_id"].to_numpy() if "person_id" in g.columns else np.full(len(g), np.nan)
        dist = g["shot_distance"].to_numpy()
        last_period = None
        timeout_pending = ""  # side that called a timeout since last possession ended
        dead_foul_pending = False
        margin_h = 0  # home minus away before the possession
        n_in_period = 0
        prev_poss_in_period = []

        def close(end_time, end_type, detail=None):
            nonlocal cur, last_end_time, last_end_type, timeout_pending, dead_foul_pending, last_detail
            if cur is None:
                return
            last_detail = dict(detail or {})
            if end_type in ("def_reb", "def_reb_ft"):
                last_detail.update(last_miss)
            cur["end_time"] = end_time
            cur["ended_by"] = end_type
            cur["duration"] = max(end_time - cur["start_time"], 0.0)
            rows.append(cur)
            cur = None
            last_end_time = end_time
            last_end_type = end_type
            timeout_pending = ""
            dead_foul_pending = False

        def open_poss(side, t, period):
            nonlocal cur, n_in_period
            n_in_period += 1
            cur = {
                "game_id": game_id,
                "period": period,
                "side": side,
                "start_time": last_end_time,
                "start_remaining": period_len(period) - last_end_time,
                "start_type": last_end_type,
                "timeout_before_by": ("own" if timeout_pending == side else "opp") if timeout_pending else "",
                "dead_foul_before": dead_foul_pending,
                "margin_at_start": margin_h if side == "h" else -margin_h,
                "first_action_time": np.nan,
                "first_action": "",
                "first_shot_dist": np.nan,
                "first_shot_value": np.nan,
                "first_shot_made": np.nan,
                "first_shot_time": np.nan,
                "points": 0,
                "fga": 0,
                "fg3a": 0,
                "fta": 0,
                "oreb": 0,
                "tov": 0,
                "steal_against": 0,
                "timeout_during": 0,
                "n_in_period": n_in_period,
                "prev_rebounder": last_detail.get("rebounder", np.nan),
                "prev_actor": last_detail.get("actor", np.nan),
                "prev_shot_dist": last_detail.get("dist", np.nan),
                "prev_shot_value": last_detail.get("value", np.nan),
                "prev_was_ft": last_detail.get("ft", 0),
            }

        for i in range(len(g)):
            a = atype[i]
            p = periods[i]
            t = elapsed[i]
            if last_period is None or p != last_period:
                if last_period is not None:
                    close(period_len(last_period), "period_end")
                last_period = p
                last_end_time = 0.0
                last_end_type = "period_start"
                timeout_pending = ""
                dead_foul_pending = False
                n_in_period = 0
            if a == "Timeout":
                if cur is not None and loc[i] == cur["side"]:
                    cur["timeout_during"] = 1
                    # an offensive timeout resets the possession clock for the offense
                else:
                    timeout_pending = loc[i]
                continue
            if a == "Foul":
                if cur is None and sub[i] not in ("Shooting",):
                    dead_foul_pending = True
                continue
            if a not in OFFENSIVE and a != "Rebound":
                continue
            if a == "Free Throw" and "Technical" in sub[i]:
                # technical free throws do not change possession
                made = not desc[i].startswith("MISS")
                if made:
                    if loc[i] == "h":
                        margin_h += 1
                    else:
                        margin_h -= 1
                continue
            side = loc[i]
            if not side:
                continue
            if a == "Rebound":
                if cur is None:
                    continue
                if side == cur["side"]:
                    cur["oreb"] += 1
                else:
                    close(t, "def_reb_ft" if cur.get("_last_was_ft") else "def_reb", {"rebounder": person[i]})
                continue
            # offensive event
            if cur is not None and cur["side"] != side:
                # possession changed without a rebound record (rare). Close the old one.
                close(t, "unknown")
            if cur is None:
                open_poss(side, t, p)
            if np.isnan(cur["first_action_time"]):
                cur["first_action_time"] = t
                cur["first_action"] = a
                if a in ("Made Shot", "Missed Shot"):
                    cur["first_shot_dist"] = g["shot_distance"].iloc[i]
            cur["_last_was_ft"] = a == "Free Throw"
            if a in ("Made Shot", "Missed Shot", "Heave"):
                if np.isnan(cur["first_shot_time"]):
                    cur["first_shot_time"] = t
                    cur["first_shot_dist"] = dist[i]
                    cur["first_shot_value"] = int(shot_value[i]) if shot_value[i] else 2
                    cur["first_shot_made"] = int(a == "Made Shot" or shot_result[i] == "Made")
                cur["fga"] += 1
                if shot_value[i] == 3:
                    cur["fg3a"] += 1
                if a == "Made Shot" or shot_result[i] == "Made":
                    pts = int(shot_value[i]) if shot_value[i] else 2
                    cur["points"] += pts
                    margin_h += pts if side == "h" else -pts
                    # possession ends unless an and-one follows; handled by next events
                    cur["_pending_end"] = ("made_fg", t, {"actor": person[i], "dist": dist[i], "value": pts})
                else:
                    cur["_pending_end"] = None
                    last_miss = {"dist": dist[i], "value": int(shot_value[i]) if shot_value[i] else 2, "ft": 0, "actor": person[i]}
            elif a == "Free Throw":
                cur["fta"] += 1
                made = not desc[i].startswith("MISS")
                if made:
                    cur["points"] += 1
                    margin_h += 1 if side == "h" else -1
                else:
                    last_miss = {"dist": 0, "value": 1, "ft": 1, "actor": person[i]}
                last_of_trip = bool(re.search(r"(\d) of \1", sub[i])) or "1 of 1" in sub[i]
                if last_of_trip and made:
                    cur["_pending_end"] = ("made_ft", t, {"actor": person[i]})
                else:
                    cur["_pending_end"] = None
            elif a == "Turnover":
                cur["tov"] += 1
                # steal is a blank action_type row with the same action_number
                j = i + 1
                stolen = False
                while j < len(g) and action_no[j] == action_no[i]:
                    if "STEAL" in desc[j]:
                        stolen = True
                    j += 1
                cur["steal_against"] = int(stolen)
                stealer = np.nan
                if stolen:
                    j = i + 1
                    while j < len(g) and action_no[j] == action_no[i]:
                        if "STEAL" in desc[j]:
                            stealer = person[j]
                        j += 1
                close(t, "steal" if stolen else "dead_tov", {"actor": stealer if stolen else person[i], "rebounder": stealer})
                continue
            # a made shot or final made free throw ends the possession unless the next
            # offensive event is by the same team at the same clock (and-one free throws)
            pend = cur.get("_pending_end")
            if pend:
                j = i + 1
                same_team_followup = False
                while j < len(g) and periods[j] == p:
                    if atype[j] in OFFENSIVE and loc[j] == side and abs(elapsed[j] - t) < 0.01 and not (atype[j] == "Free Throw" and "Technical" in sub[j]):
                        same_team_followup = True
                        break
                    if atype[j] in OFFENSIVE or atype[j] == "Rebound":
                        break
                    j += 1
                if not same_team_followup:
                    close(pend[1], pend[0], pend[2])
        if last_period is not None:
            close(period_len(last_period), "period_end")
    df = pd.DataFrame(rows)
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    df["seconds_to_first_action"] = df["first_action_time"] - df["start_time"]
    df["seconds_to_first_shot"] = df["first_shot_time"] - df["start_time"]
    return df


def team_game_from_possessions(poss: pd.DataFrame) -> pd.DataFrame:
    poss = poss[poss["ended_by"] != "period_end"].copy()
    poss["early"] = (poss["seconds_to_first_action"] <= 6).astype(int)
    poss["late"] = (poss["seconds_to_first_action"] >= 18).astype(int)
    poss["after_make"] = poss["start_type"].isin(["made_fg", "made_ft"]).astype(int)
    poss["after_dreb"] = poss["start_type"].isin(["def_reb", "def_reb_ft"]).astype(int)
    poss["after_live_tov"] = (poss["start_type"] == "steal").astype(int)
    poss["scored"] = (poss["points"] > 0).astype(int)
    poss["ft_trip"] = (poss["fta"] > 0).astype(int)
    poss["first_is_3"] = ((poss["first_action"].isin(["Made Shot", "Missed Shot"])) & (poss["first_shot_dist"] >= 22)).astype(int)

    def agg(d: pd.DataFrame) -> pd.Series:
        n = len(d)
        out = {
            "poss": n,
            "ppp": d["points"].sum() / n,
            "early_share": d["early"].mean(),
            "late_share": d["late"].mean(),
            "ppp_early": d.loc[d["early"] == 1, "points"].mean(),
            "ppp_late": d.loc[d["late"] == 1, "points"].mean(),
            "median_sec_to_action": d["seconds_to_first_action"].median(),
            "after_make_share": d["after_make"].mean(),
            "ppp_after_make": d.loc[d["after_make"] == 1, "points"].mean(),
            "sec_after_make": d.loc[d["after_make"] == 1, "seconds_to_first_action"].median(),
            "early_after_make_share": d.loc[d["after_make"] == 1, "early"].mean(),
            "ppp_after_dreb": d.loc[d["after_dreb"] == 1, "points"].mean(),
            "sec_after_dreb": d.loc[d["after_dreb"] == 1, "seconds_to_first_action"].median(),
            "early_after_dreb_share": d.loc[d["after_dreb"] == 1, "early"].mean(),
            "ppp_after_steal": d.loc[d["after_live_tov"] == 1, "points"].mean(),
            "steal_start_share": d["after_live_tov"].mean(),
            "oreb_per_poss": d["oreb"].sum() / n,
            "tov_per_poss": d["tov"].sum() / n,
            "ft_trip_share": d["ft_trip"].mean(),
            "fg3a_share": d["fg3a"].sum() / max(d["fga"].sum(), 1),
            "timeouts_called": d["timeout_during"].sum() + (d["timeout_before_by"] == "own").sum(),
            "ppp_after_own_timeout": d.loc[d["timeout_before_by"] == "own", "points"].mean(),
            "ppp_after_opp_timeout": d.loc[d["timeout_before_by"] == "opp", "points"].mean(),
        }
        return pd.Series(out)

    off = poss.groupby(["game_id", "side"]).apply(agg, include_groups=False).reset_index()
    off = off.rename(columns={c: f"o_{c}" for c in off.columns if c not in ("game_id", "side")})
    deff = off.copy()
    deff["side"] = deff["side"].map({"h": "v", "v": "h"})
    deff = deff.rename(columns={c: "d_" + c[2:] for c in deff.columns if c.startswith("o_")})
    return off.merge(deff, on=["game_id", "side"])


def box_features() -> pd.DataFrame:
    sched = read(
        """
        SELECT game_id, home_team_team_tricode AS home, away_team_team_tricode AS away,
               home_team_score::float AS home_score, away_team_score::float AS away_score,
               game_date_est
        FROM league.schedule_0
        """
    )
    sched = sched.drop_duplicates("game_id")
    adv = read(
        """
        SELECT game_id, team_tricode, offensive_rating, defensive_rating, net_rating, pace, possessions,
               effective_field_goal_percentage AS efg, true_shooting_percentage AS ts,
               assist_percentage, assist_to_turnover, offensive_rebound_percentage AS oreb_pct,
               defensive_rebound_percentage AS dreb_pct, turnover_ratio
        FROM games.box_advanced_team
        """
    )
    ff = read(
        """
        SELECT game_id, team_tricode, free_throw_attempt_rate AS ftr, team_turnover_percentage AS tov_pct,
               opp_effective_field_goal_percentage AS opp_efg, opp_free_throw_attempt_rate AS opp_ftr,
               opp_team_turnover_percentage AS opp_tov_pct, opp_offensive_rebound_percentage AS opp_oreb_pct
        FROM games.box_four_factors_team
        """
    )
    hustle = read(
        """
        SELECT game_id, team_tricode, contested_shots, contested_shots2pt, contested_shots3pt, deflections,
               charges_drawn, screen_assists, screen_assist_points, loose_balls_recovered_total AS loose_balls,
               offensive_box_outs, defensive_box_outs, box_outs
        FROM games.box_hustle_team
        """
    )
    track = read(
        """
        SELECT game_id, team_tricode, speed, distance, rebound_chances_offensive, rebound_chances_defensive,
               touches, secondary_assists, free_throw_assists, passes, assists AS trk_assists,
               contested_field_goals_attempted AS contested_fga, contested_field_goal_percentage AS contested_fg_pct,
               uncontested_field_goals_attempted AS uncontested_fga, uncontested_field_goals_percentage AS uncontested_fg_pct,
               defended_at_rim_field_goals_attempted AS rim_defended_fga, defended_at_rim_field_goal_percentage AS rim_defended_pct
        FROM games.box_tracking_team
        """
    )
    trad = read(
        """
        SELECT game_id, team_tricode, field_goals_attempted AS fga, three_pointers_attempted AS fg3a,
               free_throws_attempted AS fta, rebounds_offensive AS oreb, assists, steals, blocks, turnovers,
               fouls_personal AS fouls, points
        FROM games.box_traditional_team_1
        """
    )
    # defensive assignment stability, from the matchup box: for each defending team,
    # possession-weighted mean assignment length and distinct defenders per opponent scorer
    match = read(
        """
        WITH m AS (
          SELECT game_id, team_tricode AS off_team, person_id_off, person_id_def,
                 partial_possessions AS poss, matchup_minutes_sort AS sec,
                 partial_possessions / sum(partial_possessions) OVER (PARTITION BY game_id, person_id_off) AS share
          FROM games.box_matchups_team
          WHERE partial_possessions > 0 AND matchup_minutes_sort > 0
        ),
        per_scorer AS (
          SELECT game_id, off_team, person_id_off,
                 sum(poss) AS poss, count(*) AS n_def,
                 exp(-sum(share * ln(share))) AS n_eff,
                 sum(poss * sec) / sum(poss) AS wmean_sec
          FROM m
          GROUP BY 1, 2, 3
        )
        SELECT game_id, off_team,
               sum(poss * wmean_sec) / sum(poss) AS opp_assignment_sec,
               sum(poss * n_eff) / sum(poss) AS opp_defenders_eff,
               sum(poss * n_def) / sum(poss) AS opp_defenders_n
        FROM per_scorer
        GROUP BY 1, 2
        """
    )
    # the "off_team" in the matchup box is the team of the offensive player (team header)
    # so the defending team is the other one. Attach via schedule.
    match = match.merge(sched[["game_id", "home", "away"]], on="game_id")
    match["team_tricode"] = np.where(match["off_team"] == match["home"], match["away"], match["home"])
    match = match.drop(columns=["off_team", "home", "away"])
    match = match.rename(columns={
        "opp_assignment_sec": "def_assignment_sec",
        "opp_defenders_eff": "def_defenders_eff",
        "opp_defenders_n": "def_defenders_n",
    })
    # and the same from the offense's view: how many defenders they faced, how long
    own = read(
        """
        WITH m AS (
          SELECT game_id, team_tricode AS off_team, person_id_off, person_id_def,
                 partial_possessions AS poss, matchup_minutes_sort AS sec
          FROM games.box_matchups_team
          WHERE partial_possessions > 0 AND matchup_minutes_sort > 0
        )
        SELECT game_id, off_team AS team_tricode,
               sum(poss * sec) / sum(poss) AS off_faced_assignment_sec
        FROM m GROUP BY 1, 2
        """
    )
    sub = read(
        """
        SELECT game_id, location AS side, count(*) AS substitutions
        FROM games.pbp WHERE action_type = 'Substitution' GROUP BY 1, 2
        """
    )
    fouls = read(
        """
        SELECT game_id, location AS side,
               count(*) FILTER (WHERE sub_type IN ('Personal Take', 'Transition Take')) AS take_fouls,
               count(*) FILTER (WHERE sub_type = 'Shooting') AS shooting_fouls,
               count(*) FILTER (WHERE sub_type = 'Offensive' OR sub_type = 'Offensive Charge') AS offensive_fouls,
               count(*) FILTER (WHERE sub_type = 'Technical') AS technicals
        FROM games.pbp WHERE action_type = 'Foul' GROUP BY 1, 2
        """
    )
    frame = adv.merge(ff, on=["game_id", "team_tricode"], how="left")
    frame = frame.merge(hustle, on=["game_id", "team_tricode"], how="left")
    frame = frame.merge(track, on=["game_id", "team_tricode"], how="left")
    frame = frame.merge(trad, on=["game_id", "team_tricode"], how="left")
    frame = frame.merge(match, on=["game_id", "team_tricode"], how="left")
    frame = frame.merge(own, on=["game_id", "team_tricode"], how="left")
    frame = frame.merge(sched, on="game_id", how="left")
    frame["side"] = np.where(frame["team_tricode"] == frame["home"], "h", "v")
    frame["opponent"] = np.where(frame["side"] == "h", frame["away"], frame["home"])
    frame["margin"] = np.where(frame["side"] == "h", frame["home_score"] - frame["away_score"], frame["away_score"] - frame["home_score"])
    frame = frame.merge(sub, on=["game_id", "side"], how="left").merge(fouls, on=["game_id", "side"], how="left")
    # opponent's hustle and tracking, so defense features can use what the other side did
    opp_cols = ["contested_shots", "deflections", "passes", "touches", "speed", "distance", "fg3a", "fga", "fta", "oreb", "substitutions", "take_fouls", "shooting_fouls"]
    opp = frame[["game_id", "team_tricode"] + opp_cols].rename(columns={"team_tricode": "opponent", **{c: f"opp_{c}" for c in opp_cols}})
    frame = frame.merge(opp, on=["game_id", "opponent"], how="left")
    return frame


def demean(df: pd.DataFrame, cols: list[str], by: str) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = out[c] - out.groupby(by)[c].transform("mean")
    return out


def corr_table(df: pd.DataFrame, target: str, features: list[str], min_n: int = 200) -> pd.DataFrame:
    rows = []
    half_a = df["game_id"].str[-1].astype(int) % 2 == 0
    for f in features:
        sub = df[[f, target, "game_id"]].dropna()
        if len(sub) < min_n or sub[f].std() == 0:
            continue
        r = sub[f].corr(sub[target])
        ra = sub.loc[half_a.reindex(sub.index).fillna(False).astype(bool), f].corr(sub.loc[half_a.reindex(sub.index).fillna(False).astype(bool), target])
        rb = sub.loc[~half_a.reindex(sub.index).fillna(False).astype(bool), f].corr(sub.loc[~half_a.reindex(sub.index).fillna(False).astype(bool), target])
        rows.append({"feature": f, "target": target, "r": r, "r_half_a": ra, "r_half_b": rb, "n": len(sub)})
    return pd.DataFrame(rows).sort_values("r", key=lambda s: -s.abs())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading pbp")
    pbp = read(
        """
        SELECT game_id, action_number, action_id, clock, period, location, action_type, sub_type,
               shot_result, shot_distance, shot_value, score_home, score_away, description, person_id
        FROM games.pbp
        WHERE game_id IN (SELECT game_id FROM games.loaded)
        """
    )
    poss = build_possessions(pbp)
    poss.to_csv(OUT / "possessions.csv", index=False)
    print(f"possessions {len(poss)} in {poss['game_id'].nunique()} games")

    tg = team_game_from_possessions(poss)
    box = box_features()
    frame = box.merge(tg, on=["game_id", "side"], how="inner")
    # sanity: possessions from pbp vs box
    frame["poss_ratio"] = frame["o_poss"] / frame["possessions"]
    frame.to_csv(OUT / "team_games.csv", index=False)
    print(f"team games {len(frame)}; poss ratio median {frame['poss_ratio'].median():.3f}")

    id_cols = {"game_id", "team_tricode", "home", "away", "home_score", "away_score", "game_date_est", "side", "opponent"}
    numeric = [c for c in frame.columns if c not in id_cols and pd.api.types.is_numeric_dtype(frame[c])]
    targets = ["net_rating", "offensive_rating", "defensive_rating", "margin"]
    features = [c for c in numeric if c not in targets]
    dm = demean(frame, features + targets, "team_tricode")
    tables = {t: corr_table(dm, t, features) for t in targets}
    for t, tab in tables.items():
        tab.to_csv(OUT / f"within_team_corr_{t}.csv", index=False)

    # also across-team season means: does the team trait predict the team's season net rating
    season = frame.groupby("team_tricode")[features + targets].mean()
    season_rows = []
    for f in features:
        if season[f].std() == 0 or season[f].isna().any():
            continue
        season_rows.append({"feature": f, "r_net": season[f].corr(season["net_rating"]), "r_off": season[f].corr(season["offensive_rating"]), "r_def": season[f].corr(season["defensive_rating"])})
    season_tab = pd.DataFrame(season_rows).sort_values("r_net", key=lambda s: -s.abs())
    season_tab.to_csv(OUT / "across_team_corr.csv", index=False)

    # possession-level facts
    live = poss[poss["ended_by"] != "period_end"].copy()
    live["sec_bin"] = pd.cut(live["seconds_to_first_action"], [-1, 4, 8, 12, 16, 20, 30], labels=["0-4", "4-8", "8-12", "12-16", "16-20", "20+"])
    by_start = live.groupby("start_type")["points"].agg(["mean", "count"]).reset_index()
    by_start_sec = live.groupby(["start_type", "sec_bin"], observed=True)["points"].agg(["mean", "count"]).reset_index()
    timeout = live.groupby(["start_type", "timeout_before_by"])["points"].agg(["mean", "count"]).reset_index()
    # run-stopping: team scored on, calls timeout. Compare the next possession of the team that called it
    # against the same start type with no timeout, matched on period and margin bucket.
    live["margin_bin"] = pd.cut(live["margin_at_start"], [-100, -15, -8, -3, 3, 8, 15, 100])
    dead = live[live["start_type"].isin(["made_fg", "made_ft", "dead_tov"])]
    grp = dead.groupby(["start_type", "period", "margin_bin"], observed=True)
    base = grp["points"].transform("mean")
    dead = dead.assign(vs_matched=dead["points"] - base)
    to_effect = dead.groupby("timeout_before_by")["vs_matched"].agg(["mean", "count", "sem"]).reset_index()

    summary = {
        "games": int(poss["game_id"].nunique()),
        "possessions": int(len(live)),
        "poss_ratio_median": float(frame["poss_ratio"].median()),
        "ppp_by_start_type": by_start.to_dict("records"),
        "ppp_by_start_and_seconds": by_start_sec.to_dict("records"),
        "ppp_by_timeout": timeout.to_dict("records"),
        "timeout_vs_matched_dead_ball": to_effect.to_dict("records"),
        "within_team_top": {t: tab.head(40).to_dict("records") for t, tab in tables.items()},
        "across_team_top": season_tab.head(40).to_dict("records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    for t in ("net_rating", "offensive_rating", "defensive_rating"):
        print(f"\n== within-team correlations with {t}")
        print(tables[t].head(30).to_string(index=False))
    print("\n== across-team season means vs net rating")
    print(season_tab.head(30).to_string(index=False))
    print("\n== ppp by start type")
    print(by_start.to_string(index=False))
    print("\n== timeout vs matched dead ball")
    print(to_effect.to_string(index=False))


if __name__ == "__main__":
    main()
