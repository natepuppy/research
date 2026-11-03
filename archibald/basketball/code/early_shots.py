"""Which early shots are worth taking, and how many games the two rules would
have flipped.

Early = first shot within 8 seconds of the possession starting. Zones from
shot distance: rim (<5 ft), paint (5-14), mid-range (14 ft to the arc), three.
The alternative to an early shot is the same team running its half-court
offense: points per possession when the first shot comes after 8 seconds, from
the same start type.

Flip simulation uses the within-team effect sizes already estimated, applied
game by game to the actual margins. Assumptions are printed with the result.
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


def zone(dist, value):
    if value == 3:
        return "three"
    if dist < 5:
        return "rim (<5 ft)"
    if dist < 14:
        return "paint (5-14 ft)"
    return "mid-range (14+ ft, 2)"


def early_shot_table(poss: pd.DataFrame) -> dict:
    live = poss[(poss["ended_by"] != "period_end")].copy()
    live = live[~(((live["period"] >= 4) & (live["start_remaining"] < 180)) | (live["margin_at_start"].abs() > 20))]
    shots = live[live["first_shot_time"].notna()].copy()
    shots["early"] = shots["seconds_to_first_shot"] <= EARLY
    shots["zone"] = [zone(d, v) for d, v in zip(shots["first_shot_dist"], shots["first_shot_value"])]
    shots["pps"] = shots["first_shot_made"] * shots["first_shot_value"]
    out = {}
    for st in ("def_reb", "steal", "made_fg", "made_ft", "dead_tov"):
        d = shots[shots["start_type"] == st]
        late = d[~d["early"]]
        baseline = float(late["points"].mean())  # whole possession, half-court alternative
        rows = []
        for z, dz in d[d["early"]].groupby("zone"):
            rows.append({
                "zone": z,
                "early_shots": int(len(dz)),
                "share_of_early_shots": float(len(dz) / d["early"].sum()),
                "fg_pct": float(dz["first_shot_made"].mean()),
                "points_per_shot": float(dz["pps"].mean()),
                "possession_ppp": float(dz["points"].mean()),
                "se": float(dz["points"].sem()),
                "vs_half_court": float(dz["points"].mean() - baseline),
            })
        late_rows = []
        for z, dz in late.groupby("zone"):
            late_rows.append({"zone": z, "shots": int(len(dz)), "fg_pct": float(dz["first_shot_made"].mean()), "points_per_shot": float(dz["pps"].mean()), "possession_ppp": float(dz["points"].mean())})
        out[st] = {
            "early_shots": int(d["early"].sum()),
            "half_court_ppp_same_start": baseline,
            "early_by_zone": rows,
            "late_by_zone": late_rows,
            "early_zone_mix": d[d["early"]]["zone"].value_counts(normalize=True).round(3).to_dict(),
            "late_zone_mix": late["zone"].value_counts(normalize=True).round(3).to_dict(),
        }
    # the 8-second rule as a whole: possessions where the first shot came early vs those where
    # the team got into the clock, by start type, including possessions with no shot (turnovers)
    live["early_any"] = live["seconds_to_first_action"] <= EARLY
    rule = {}
    for st in ("def_reb", "steal", "made_fg"):
        d = live[live["start_type"] == st]
        rule[st] = {
            "early_ppp": float(d.loc[d["early_any"], "points"].mean()),
            "early_tov_rate": float((d.loc[d["early_any"], "first_action"] == "Turnover").mean()),
            "slow_ppp": float(d.loc[~d["early_any"], "points"].mean()),
            "slow_tov_rate": float((d.loc[~d["early_any"], "tov"] > 0).mean()),
        }
    out["rule_summary"] = rule
    # seconds curve for the first shot after a rebound, by zone
    dreb = shots[shots["start_type"] == "def_reb"].copy()
    dreb["sec_bin"] = pd.cut(dreb["seconds_to_first_shot"], [0, 4, 8, 12, 16, 24], labels=["0-4", "4-8", "8-12", "12-16", "16-24"])
    curve = dreb.groupby(["zone", "sec_bin"], observed=True)["points"].agg(["mean", "count"]).reset_index()
    out["after_rebound_curve"] = curve.round(3).to_dict("records")
    return out


def flips(poss: pd.DataFrame) -> dict:
    live = poss[poss["ended_by"] != "period_end"].copy()
    live["early"] = live["seconds_to_first_action"] <= EARLY
    sched = read("SELECT game_id, home_team_team_tricode AS home, away_team_team_tricode AS away, home_team_score::float AS hs, away_team_score::float AS as_ FROM league.schedule_0").drop_duplicates("game_id")
    # per team-game counts from the sample
    recs = []
    for (gid, side), d in live.groupby(["game_id", "side"]):
        mk = d[d["start_type"].isin(["made_fg", "made_ft"])]
        dr = d[d["start_type"].isin(["def_reb", "def_reb_ft"])]
        recs.append({
            "game_id": gid, "side": side, "poss": len(d),
            "early_after_make": int(mk["early"].sum()),
            "slow_after_dreb": int((~dr["early"]).sum()),
            "early_after_dreb_ppp": float(dr.loc[dr["early"], "points"].mean()) if dr["early"].sum() else np.nan,
            "slow_after_dreb_ppp": float(dr.loc[~dr["early"], "points"].mean()) if (~dr["early"]).sum() else np.nan,
        })
    tg = pd.DataFrame(recs).merge(sched, on="game_id")
    tg["margin"] = np.where(tg["side"] == "h", tg["hs"] - tg["as_"], tg["as_"] - tg["hs"])
    tg["team"] = np.where(tg["side"] == "h", tg["home"], tg["away"])
    # league-wide gains per possession from the possession table
    mk = live[live["start_type"].isin(["made_fg", "made_ft"])]
    gain_stop_push = float(mk.loc[~mk["early"], "points"].mean() - mk.loc[mk["early"], "points"].mean())
    dr = live[live["start_type"].isin(["def_reb", "def_reb_ft"])]
    gain_push = float(dr.loc[dr["early"], "points"].mean() - dr.loc[~dr["early"], "points"].mean())
    sample_games = tg["game_id"].nunique()

    def count_flips(shift: pd.Series) -> dict:
        lost = tg["margin"] < 0
        flipped = lost & (tg["margin"] + shift > 0)
        # ties become wins half the time
        tied = lost & (tg["margin"] + shift == 0)
        return {"team_games": int(len(tg)), "losses": int(lost.sum()), "flipped": float(flipped.sum() + 0.5 * tied.sum()),
                "flipped_per_82_games_one_team": float((flipped.sum() + 0.5 * tied.sum()) / len(tg) * 82),
                "mean_shift_points": float(shift.mean())}

    scenarios = {}
    s1 = tg["early_after_make"] * gain_stop_push
    scenarios["A. never push after being scored on"] = {**count_flips(s1), "assumption": f"every early possession after an opponent score becomes a normal one; gain {gain_stop_push:.3f} per possession, {tg['early_after_make'].mean():.1f} such possessions per game"}
    for frac in (0.25, 0.5):
        s2 = tg["slow_after_dreb"] * frac * gain_push
        scenarios[f"B. push after {int(frac*100)}% more of defensive rebounds"] = {**count_flips(s2), "assumption": f"{int(frac*100)}% of slow after-rebound possessions become early at the league-average early gain of {gain_push:.3f}; {tg['slow_after_dreb'].mean():.1f} slow after-rebound possessions per game. Assumes the extra pushes are as good as the ones teams already choose, which is optimistic."}
    scenarios["A + B(25%)"] = {**count_flips(s1 + tg["slow_after_dreb"] * 0.25 * gain_push), "assumption": "both rules"}
    # C: crash more. within-team coefficient on net rating per crasher per miss = 12.06 per 100 possessions (SE 3.7)
    for shift_c in (0.05, 0.10):
        s3 = 12.06 * shift_c * tg["poss"] / 100
        scenarios[f"C. +{shift_c:.2f} crashers per miss"] = {**count_flips(s3), "assumption": f"+{shift_c:.2f} bodies near the ball per own miss (league SD within team is 0.14; median-to-top-8 is about 0.10). Net +12.1 per 100 possessions per unit, SE 3.7. Applied to the team's own possessions."}
    scenarios["A + B(25%) + C(0.10)"] = {**count_flips(s1 + tg["slow_after_dreb"] * 0.25 * gain_push + 12.06 * 0.10 * tg["poss"] / 100), "assumption": "all three"}
    # full-season version of C using the season game logs (1,230 games), rebounding only
    gl = read("SELECT game_id, team_abbreviation AS team, pts, plus_minus, fga, fta, oreb, tov FROM league.game_log_team_regular_season")
    gl["poss"] = gl["fga"] - gl["oreb"] + gl["tov"] + 0.44 * gl["fta"]
    season = {}
    for shift_c in (0.05, 0.10):
        shift = 12.06 * shift_c * gl["poss"] / 100
        lost = gl["plus_minus"] < 0
        flipped = lost & (gl["plus_minus"] + shift > 0)
        season[f"+{shift_c:.2f} crashers per miss"] = {
            "team_games": int(len(gl)), "losses": int(lost.sum()), "flipped": int(flipped.sum()),
            "flipped_per_team_per_season": float(flipped.sum() / 30),
            "by_team": gl.assign(f=flipped).groupby("team")["f"].sum().astype(int).to_dict(),
        }
    return {
        "sample_games": int(sample_games),
        "per_possession_gains": {"stop_pushing_after_opponent_score": gain_stop_push, "push_after_rebound": gain_push},
        "sample_scenarios": scenarios,
        "full_season_rebounding": season,
        "reading": "Flipped = a loss that becomes a win when the estimated point shift is added to the final margin, one team adopting while the opponent does not. Margins are whole-game, so overtime games count at their final margin.",
    }


def main() -> None:
    poss = pd.read_csv(ROOT / "results" / "scoreboard" / "possessions.csv", dtype={"game_id": str})
    out = {"early_shots": early_shot_table(poss), "flips": flips(poss)}
    (OUT / "early_shots_and_flips.json").write_text(json.dumps(out, indent=2, default=float))
    e = out["early_shots"]
    for st in ("def_reb", "steal", "made_fg"):
        print(f"\n== {st}: half-court alternative {e[st]['half_court_ppp_same_start']:.3f} PPP, {e[st]['early_shots']} early shots")
        print(pd.DataFrame(e[st]["early_by_zone"]).round(3).to_string(index=False))
        print(" late:"); print(pd.DataFrame(e[st]["late_by_zone"]).round(3).to_string(index=False))
    print("\n== rule summary"); print(pd.DataFrame(e["rule_summary"]).T.round(3).to_string())
    print("\n== after-rebound curve"); print(pd.DataFrame(e["after_rebound_curve"]).pivot(index="zone", columns="sec_bin", values="mean").round(3).to_string())
    f = out["flips"]
    print("\n== flips, sample", f["sample_games"], "games;", f["per_possession_gains"])
    for k, v in f["sample_scenarios"].items():
        print(f"  {k}: flipped {v['flipped']:.1f} of {v['losses']} losses in {v['team_games']} team-games -> {v['flipped_per_82_games_one_team']:.1f} wins per 82 for one team; mean shift {v['mean_shift_points']:.2f} pts")
    for k, v in f["full_season_rebounding"].items():
        print(f"  season, {k}: flipped {v['flipped']} of {v['losses']} losses -> {v['flipped_per_team_per_season']:.1f} wins per team")


if __name__ == "__main__":
    main()
