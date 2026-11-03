"""Two lineup questions.

1. Reactive substitution. When the opponent's best defender sits, is the star's
   floor time worth more? Rebuilds who is on the floor for every possession in
   the loaded games from starters plus substitution rows, then compares the
   star's team on offense and defense in the four states
   (star on/off x opponent stopper on/off), within game.

2. Composition. Among lineups containing a high-usage star, do units built from
   low-usage defenders outperform units built from medium scorers? Uses the
   league five-man lineup table with season player stats, within team.

Writes results/lineups/.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

DSN = "postgresql://nathanclark@127.0.0.1:5432/nba"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "lineups"
CLOCK = re.compile(r"PT(\d+)M([\d.]+)S")
OFFENSIVE = {"Made Shot", "Missed Shot", "Heave", "Turnover", "Free Throw"}


def read(sql: str) -> pd.DataFrame:
    with psycopg.connect(DSN) as conn:
        return pd.read_sql(sql, conn)


def clock_seconds(text) -> float:
    m = CLOCK.match(str(text))
    return int(m.group(1)) * 60 + float(m.group(2)) if m else np.nan


def period_len(p: int) -> int:
    return 720 if p <= 4 else 300


def norm(s: str) -> str:
    text = unicodedata.normalize("NFKD", str(s))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z]", "", text.lower())


def resolve_incoming(desc: str, roster: pd.DataFrame, out_id: int):
    m = re.match(r"SUB:\s*(.+?)\s+FOR\s+(.+)$", desc)
    if not m:
        return None
    name = norm(m.group(1))
    cands = roster[(roster["fam"] == name) & (roster["person_id"] != out_id)]
    if len(cands) == 1:
        return int(cands["person_id"].iloc[0])
    cands = roster[roster["fam"].str.endswith(name) | roster["fam"].str.startswith(name)]
    cands = cands[cands["person_id"] != out_id]
    if len(cands) >= 1:
        return int(cands["person_id"].iloc[0])
    return None


def floor_by_event(pbp: pd.DataFrame, box: pd.DataFrame, sides: dict) -> pd.DataFrame:
    """For every pbp row, the set of players on the floor for h and v."""
    out_rows = []
    box = box.copy()
    box["fam"] = box["family_name"].map(norm)
    for game_id, g in pbp.groupby("game_id", sort=False):
        g = g.sort_values(["action_number", "action_id"], kind="stable")
        b = box[box["game_id"] == game_id]
        side_of_team = sides[game_id]
        rosters = {s: b[b["team_tricode"].map(side_of_team) == s] for s in ("h", "v")}
        roster_ids = {s: set(rosters[s]["person_id"].astype(int)) for s in ("h", "v")}
        starters = {s: set(rosters[s].loc[rosters[s]["position"].fillna("") != "", "person_id"].astype(int)) for s in ("h", "v")}
        prev_end = {s: set(starters[s]) for s in ("h", "v")}
        for period, gp in g.groupby("period", sort=True):
            # infer who starts the period: acted before being subbed in
            acted_before_in = {"h": set(), "v": set()}
            subbed_in_first = {"h": set(), "v": set()}
            seen_in = {"h": set(), "v": set()}
            for r in gp.itertuples(index=False):
                s = r.location
                if s not in ("h", "v"):
                    continue
                pid = r.person_id
                if r.action_type == "Substitution":
                    out_id = int(pid) if pd.notna(pid) else None
                    in_id = resolve_incoming(r.description, rosters[s], out_id)
                    if out_id is not None and out_id not in seen_in[s]:
                        acted_before_in[s].add(out_id)
                    if in_id is not None:
                        seen_in[s].add(in_id)
                        if in_id not in acted_before_in[s]:
                            subbed_in_first[s].add(in_id)
                elif pd.notna(pid) and pid and int(pid) in roster_ids[s]:
                    pid = int(pid)
                    if pid not in seen_in[s]:
                        acted_before_in[s].add(pid)
            start = {}
            for s in ("h", "v"):
                if period == 1:
                    lineup = set(starters[s])
                else:
                    lineup = set(acted_before_in[s])
                    if len(lineup) < 5:
                        carry = [p for p in prev_end[s] if p not in subbed_in_first[s] and p not in lineup]
                        lineup |= set(carry[: 5 - len(lineup)])
                    if len(lineup) > 5:
                        # more than five with evidence: a sub row was missed. Keep the first five to act.
                        order = []
                        for r in gp.itertuples(index=False):
                            if r.location == s and pd.notna(r.person_id) and int(r.person_id) in lineup and int(r.person_id) not in order:
                                order.append(int(r.person_id))
                        lineup = set(order[:5])
                start[s] = lineup
            on = {s: set(start[s]) for s in ("h", "v")}
            for r in gp.itertuples(index=False):
                s = r.location
                if r.action_type == "Substitution" and s in ("h", "v"):
                    out_id = int(r.person_id) if pd.notna(r.person_id) else None
                    in_id = resolve_incoming(r.description, rosters[s], out_id)
                    if out_id in on[s]:
                        on[s].discard(out_id)
                    if in_id is not None:
                        on[s].add(in_id)
                out_rows.append((game_id, r.action_number, r.action_id, frozenset(on["h"]), frozenset(on["v"])))
            prev_end = {s: set(on[s]) for s in ("h", "v")}
    return pd.DataFrame(out_rows, columns=["game_id", "action_number", "action_id", "on_h", "on_v"])


def possessions_with_floor(pbp: pd.DataFrame, floor: pd.DataFrame) -> pd.DataFrame:
    """Possession = run of offensive events by one side. Floor taken at the first event."""
    pbp = pbp.merge(floor, on=["game_id", "action_number", "action_id"], how="left")
    pbp = pbp.sort_values(["game_id", "action_number", "action_id"], kind="stable")
    rows = []
    for game_id, g in pbp.groupby("game_id", sort=False):
        cur = None
        for r in g.itertuples(index=False):
            a = r.action_type
            s = r.location
            if a == "Free Throw" and "Technical" in str(r.sub_type):
                continue
            if a == "Rebound":
                if cur is not None and s != cur["side"]:
                    rows.append(cur)
                    cur = None
                continue
            if a not in OFFENSIVE or s not in ("h", "v"):
                continue
            if cur is not None and (cur["side"] != s or cur["period"] != r.period):
                rows.append(cur)
                cur = None
            if cur is None:
                cur = {"game_id": game_id, "period": r.period, "side": s, "points": 0, "on_h": r.on_h, "on_v": r.on_v, "fga": 0}
            if a in ("Made Shot",) or (a in ("Missed Shot", "Heave") and r.shot_result == "Made"):
                cur["points"] += int(r.shot_value) if r.shot_value else 2
            if a in ("Made Shot", "Missed Shot", "Heave"):
                cur["fga"] += 1
            if a == "Free Throw" and not str(r.description).startswith("MISS"):
                cur["points"] += 1
            if a == "Turnover" or a == "Made Shot":
                # made shot might continue with and-one; the next event decides. Keep simple: close on turnover only,
                # and on made shot unless a same-side free throw follows (handled by run logic since FT is same side).
                if a == "Turnover":
                    rows.append(cur)
                    cur = None
        if cur is not None:
            rows.append(cur)
    return pd.DataFrame(rows)


def player_quality() -> pd.DataFrame:
    base = read(
        """
        SELECT a.player_id, a.player_name, a.team_abbreviation AS team, a.gp, a.min, a.usg_pct, a.ts_pct, a.pie,
               b.pts AS ppg
        FROM league.player_advanced_regular_season a
        JOIN league.player_base_regular_season b USING (player_id)
        """
    ).sort_values("gp", ascending=False).drop_duplicates("player_id")
    deff = read(
        """
        SELECT def_player_id AS player_id,
               sum(player_pts) / sum(partial_poss) AS def_ppp,
               sum(team_pts) / sum(partial_poss) AS def_team_ppp,
               sum(partial_poss) AS def_poss
        FROM league.matchups_regular_season GROUP BY 1
        """
    )
    contest = read(
        "SELECT close_def_person_id AS player_id, pct_plusminus AS contest_pm, d_fga FROM league.pt_defend_overall_regular_season"
    )
    q = base.merge(deff, on="player_id", how="left").merge(contest, on="player_id", how="left")
    # defender score: z of points allowed (lower better) and contest plus-minus (lower better), rotation players only
    rot = q[(q["def_poss"] >= 400) & (q["min"] >= 15)]
    mu, sd = rot["def_ppp"].mean(), rot["def_ppp"].std()
    mu2, sd2 = rot["contest_pm"].mean(), rot["contest_pm"].std()
    q["def_score"] = -((q["def_ppp"] - mu) / sd + (q["contest_pm"] - mu2) / sd2) / 2
    return q


def reactive(poss: pd.DataFrame, quality: pd.DataFrame, sides: dict, teams: dict) -> dict:
    """Four-state comparison per game for each team's star vs the opponent's best defender."""
    q = quality.set_index("player_id")
    recs = []
    for game_id, g in poss.groupby("game_id"):
        side_of_team = sides[game_id]
        team_of_side = {v: k for k, v in side_of_team.items()}
        rosters = teams[game_id]
        for s in ("h", "v"):
            o = "v" if s == "h" else "h"
            mine = [p for p in rosters[s] if p in q.index]
            theirs = [p for p in rosters[o] if p in q.index and q.at[p, "def_poss"] >= 400 and q.at[p, "min"] >= 15]
            if not mine or not theirs:
                continue
            star = max(mine, key=lambda p: (q.at[p, "usg_pct"] if pd.notna(q.at[p, "usg_pct"]) else 0) * (q.at[p, "ppg"] if pd.notna(q.at[p, "ppg"]) else 0))
            if q.at[star, "ppg"] < 20:
                continue
            stopper = max(theirs, key=lambda p: q.at[p, "def_score"] if pd.notna(q.at[p, "def_score"]) else -9)
            on_col = f"on_{s}"
            opp_col = f"on_{o}"
            for r in g.itertuples(index=False):
                son = star in getattr(r, on_col)
                don = stopper in getattr(r, opp_col)
                recs.append({
                    "game_id": game_id, "team": team_of_side[s], "star": q.at[star, "player_name"], "stopper": q.at[stopper, "player_name"],
                    "offense": r.side == s, "star_on": son, "stopper_on": don, "points": r.points,
                })
    d = pd.DataFrame(recs)
    d["state"] = np.where(d.star_on, "star on", "star off") + ", " + np.where(d.stopper_on, "stopper on", "stopper off")
    # within game-team demeaning of PPP so the comparison is between states inside the same game
    off = d[d.offense].copy()
    deff = d[~d.offense].copy()
    for frame in (off, deff):
        frame["ppp_dm"] = frame["points"] - frame.groupby(["game_id", "team"])["points"].transform("mean")

    def table(frame):
        t = frame.groupby("state").agg(poss=("points", "size"), ppp=("points", "mean"), ppp_vs_game=("ppp_dm", "mean"), se=("ppp_dm", "sem")).reset_index()
        return t.to_dict("records")

    # the two interaction tests: star's value with stopper on vs off
    def star_value(frame):
        out = {}
        for d_on in (True, False):
            sub = frame[frame.stopper_on == d_on]
            a = sub[sub.star_on]["ppp_dm"]
            b = sub[~sub.star_on]["ppp_dm"]
            out["stopper on" if d_on else "stopper off"] = {
                "star_on_minus_off": float(a.mean() - b.mean()),
                "se": float(np.sqrt(a.sem() ** 2 + b.sem() ** 2)),
                "poss_on": int(len(a)), "poss_off": int(len(b)),
            }
        return out

    # do coaches already react? P(star on | stopper on) vs P(star on | stopper off), within game
    both = d.drop_duplicates(["game_id", "team", "offense", "star_on", "stopper_on", "points"]) if False else d
    react = both.groupby(["game_id", "team", "stopper_on"])["star_on"].mean().unstack("stopper_on")
    react = react.dropna()
    return {
        "games": int(d["game_id"].nunique()),
        "team_games": int(d.groupby(["game_id", "team"]).ngroups),
        "offense_by_state": table(off),
        "defense_by_state": table(deff),
        "star_offense_value": star_value(off),
        "star_defense_value": star_value(deff),
        "net_star_value": {
            k: {"net_per_100": 100 * (star_value(off)[k]["star_on_minus_off"] - star_value(deff)[k]["star_on_minus_off"])}
            for k in ("stopper on", "stopper off")
        },
        "coach_reaction": {
            "p_star_on_when_stopper_on": float(react[True].mean()),
            "p_star_on_when_stopper_off": float(react[False].mean()),
            "mean_within_game_difference": float((react[True] - react[False]).mean()),
            "se": float((react[True] - react[False]).sem()),
            "team_games": int(len(react)),
        },
        "state_share": d.groupby("state").size().div(len(d)).to_dict(),
    }


def composition(quality: pd.DataFrame) -> dict:
    lu = read(
        """
        SELECT group_id, group_name, team_abbreviation AS team, gp, min, off_rating, def_rating, net_rating, poss, pace, ts_pct
        FROM league.lineups_advanced_regular_season
        """
    )
    q = quality.set_index("player_id")
    rows = []
    for r in lu.itertuples(index=False):
        ids = [int(x) for x in r.group_id.strip("-").split("-")]
        if any(i not in q.index for i in ids):
            continue
        usg = np.array([q.at[i, "usg_pct"] for i in ids], dtype=float)
        ppg = np.array([q.at[i, "ppg"] for i in ids], dtype=float)
        ts = np.array([q.at[i, "ts_pct"] for i in ids], dtype=float)
        dsc = np.array([q.at[i, "def_score"] for i in ids], dtype=float)
        pie = np.array([q.at[i, "pie"] for i in ids], dtype=float)
        if np.isnan(usg).any() or np.isnan(ts).any():
            continue
        k = int(np.argmax(usg))
        others = np.ones(5, dtype=bool)
        others[k] = False
        rows.append({
            "team": r.team, "group_name": r.group_name, "min": r.min, "gp": r.gp,
            "off_rating": r.off_rating, "def_rating": r.def_rating, "net_rating": r.net_rating,
            "star_usg": usg[k], "star_ppg": ppg[k],
            "others_usg_sum": usg[others].sum(), "others_ppg_sum": ppg[others].sum(),
            "others_ts": ts[others].mean(), "others_def": np.nanmean(dsc[others]),
            "others_pie": pie[others].sum(), "total_usg": usg.sum(), "total_pie": pie.sum(),
            "n_scorers_15": int((ppg[others] >= 15).sum()),
            "n_defenders_good": int((dsc[others] >= 0.5).sum()),
        })
    L = pd.DataFrame(rows)
    L = L[L["min"] >= 30]
    L.to_csv(OUT / "lineups_with_composition.csv", index=False)

    def wls(X, y, w):
        X = np.column_stack([np.ones(len(y)), X])
        sw = np.sqrt(w)
        b, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
        r = y - X @ b
        s2 = np.sum(w * r * r) / (len(y) - X.shape[1])
        se = np.sqrt(np.diag(s2 * np.linalg.inv((X * sw[:, None]).T @ (X * sw[:, None]))))
        return b, se

    def demean(df, cols, w):
        out = df.copy()
        for c in cols:
            wm = df.groupby("team").apply(lambda d: np.average(d[c], weights=d[w]), include_groups=False)
            out[c] = df[c] - df["team"].map(wm)
        return out

    star_units = L[L["star_usg"] >= 0.28].copy()
    cols = ["off_rating", "def_rating", "net_rating", "others_usg_sum", "others_ts", "others_def", "others_pie", "others_ppg_sum"]
    dm = demean(star_units, cols, "min")
    w = dm["min"].to_numpy()
    results = {"star_lineups": int(len(star_units)), "teams": int(star_units["team"].nunique()), "minutes": float(star_units["min"].sum())}
    X = dm[["others_usg_sum", "others_def", "others_ts"]].to_numpy()
    for y in ("off_rating", "def_rating", "net_rating"):
        b, se = wls(X, dm[y].to_numpy(), w)
        results[y] = {
            "per_10pts_of_teammate_usage": {"coef": 0.10 * b[1], "se": 0.10 * se[1]},
            "per_1sd_teammate_defense": {"coef": b[2], "se": se[2]},
            "per_5pts_teammate_ts": {"coef": 0.05 * b[3], "se": 0.05 * se[3]},
        }
    # the direct contrast: star + defenders (others_usg_sum low, others_def high) vs star + scorers
    star_units["type"] = np.select(
        [
            (star_units["n_scorers_15"] == 0) & (star_units["n_defenders_good"] >= 2),
            (star_units["n_scorers_15"] >= 2),
        ],
        ["star + defenders, no 15-ppg teammate", "star + two 15-ppg scorers"],
        "mixed",
    )
    contrast = star_units.groupby("type").apply(
        lambda d: pd.Series({
            "lineups": len(d), "minutes": d["min"].sum(),
            "off_rating": np.average(d["off_rating"], weights=d["min"]),
            "def_rating": np.average(d["def_rating"], weights=d["min"]),
            "net_rating": np.average(d["net_rating"], weights=d["min"]),
        }), include_groups=False,
    ).reset_index()
    results["contrast_raw"] = contrast.to_dict("records")
    # same contrast within team
    dm2 = star_units.copy()
    for c in ("off_rating", "def_rating", "net_rating"):
        wm = star_units.groupby("team").apply(lambda d: np.average(d[c], weights=d["min"]), include_groups=False)
        dm2[c] = star_units[c] - star_units["team"].map(wm)
    contrast_dm = dm2.groupby("type").apply(
        lambda d: pd.Series({
            "lineups": len(d), "minutes": d["min"].sum(),
            "off_vs_team": np.average(d["off_rating"], weights=d["min"]),
            "def_vs_team": np.average(d["def_rating"], weights=d["min"]),
            "net_vs_team": np.average(d["net_rating"], weights=d["min"]),
        }), include_groups=False,
    ).reset_index()
    results["contrast_within_team"] = contrast_dm.to_dict("records")
    # all lineups: usage sum diminishing returns
    all_dm = demean(L, ["off_rating", "total_usg", "total_pie", "others_ts"], "min")
    b, se = wls(np.column_stack([all_dm["total_usg"], all_dm["total_usg"] ** 2, all_dm["total_pie"]]), all_dm["off_rating"].to_numpy(), all_dm["min"].to_numpy())
    results["all_lineups_usage_curve"] = {"lineups": int(len(L)), "linear": b[1], "quadratic": b[2], "se_quadratic": se[2]}
    return results


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pbp = read(
        """
        SELECT game_id, action_number, action_id, clock, period, location, action_type, sub_type, person_id,
               shot_result, shot_value, description
        FROM games.pbp WHERE game_id IN (SELECT game_id FROM games.loaded)
        """
    )
    box = read("SELECT game_id, team_tricode, person_id, family_name, name_i, position, minutes FROM games.box_traditional_player")
    box = box[box["minutes"].fillna("").str.contains(":")]
    sched = read("SELECT game_id, home_team_team_tricode AS home, away_team_team_tricode AS away FROM league.schedule_0").drop_duplicates("game_id")
    sides = {r.game_id: {r.home: "h", r.away: "v"} for r in sched.itertuples(index=False)}
    teams = {}
    for game_id, b in box.groupby("game_id"):
        if game_id not in sides:
            continue
        teams[game_id] = {s: set(b.loc[b["team_tricode"].map(sides[game_id]) == s, "person_id"].astype(int)) for s in ("h", "v")}
    pbp = pbp[pbp["game_id"].isin(teams)]
    print("rebuilding floor")
    floor = floor_by_event(pbp, box, sides)
    bad = floor[(floor["on_h"].map(len) != 5) | (floor["on_v"].map(len) != 5)]
    print(f"rows {len(floor)}, rows with a side not at 5 players: {len(bad)} ({len(bad)/len(floor):.1%})")
    poss = possessions_with_floor(pbp, floor)
    poss = poss[(poss["on_h"].map(len) == 5) & (poss["on_v"].map(len) == 5)]
    print(f"possessions {len(poss)}")
    quality = player_quality()
    quality.to_csv(OUT / "player_quality.csv", index=False)
    react = reactive(poss, quality, sides, teams)
    comp = composition(quality)
    summary = {"reactive_substitution": react, "composition": comp}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
