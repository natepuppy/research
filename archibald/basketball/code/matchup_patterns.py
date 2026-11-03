"""First-pass correlations in the 2025-26 NBA tables.

The season matchup table is one row per offensive player vs defender, full
regular season. Partial possessions sum to the offensive player's on-court
possessions and are split in proportion to tracked matchup time, so "time
guarded" and "possessions attacked" are the same split. The choice variable
that is not the same split is percentage_total_time_both_on on the per-game
matchup table: of the minutes both players shared the floor, what fraction
was this assignment.

Game-level tables cover 387 regular-season games, mostly the first part of
the season. Playoff player-versus-defender matchups were not returned by the
API. Switches are present as a column and are zero in every row.

Writes results/patterns/summary.json plus pair and player feature tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

DSN = "postgresql://nathanclark@127.0.0.1:5432/nba"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "patterns"

PLAY_TYPES = (
    "isolation",
    "prballhandler",
    "prrollman",
    "postup",
    "spotup",
    "cut",
    "handoff",
    "offscreen",
    "transition",
    "offrebound",
    "misc",
)
DEFENSE_MISSING = {"cut", "offrebound", "transition", "misc"}
ZONES = (
    ("Restricted area", "restricted_area", 2),
    ("Paint, non-RA", "in_the_paint_non_ra", 2),
    ("Midrange", "mid_range", 2),
    ("Corner 3", "corner_3", 3),
    ("Above-the-break 3", "above_the_break_3", 3),
)
PAIR_MIN_POSS = 30.0
BASE_MIN_POSS = 200.0


def connect():
    return psycopg.connect(DSN)


def read(sql: str) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql(sql, conn)


def wmean(x, w) -> float:
    w = np.asarray(w, dtype=float)
    x = np.asarray(x, dtype=float)
    sw = w.sum()
    if sw <= 0 or len(x) == 0:
        return float("nan")
    return float(np.sum(w * x) / sw)


def wcorr(x, y, w) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[mask], y[mask], w[mask]
    if len(x) < 3:
        return float("nan")
    xc = x - wmean(x, w)
    yc = y - wmean(y, w)
    denom = np.sqrt(np.sum(w * xc * xc) * np.sum(w * yc * yc))
    if denom <= 0:
        return float("nan")
    return float(np.sum(w * xc * yc) / denom)


def wr2(y, yhat, w) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    w = np.asarray(w, dtype=float)
    yc = y - wmean(y, w)
    ss_tot = np.sum(w * yc * yc)
    if ss_tot <= 0:
        return float("nan")
    resid = y - yhat
    return float(1 - np.sum(w * resid * resid) / ss_tot)


def wls(X, y, w):
    """Weighted least squares. X includes the intercept column."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(y) & np.isfinite(w) & (w > 0) & np.all(np.isfinite(X), axis=1)
    X, y, w = X[mask], y[mask], w[mask]
    sw = np.sqrt(w)
    beta, _, _, _ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    n, k = X.shape
    dof = max(n - k, 1)
    sigma2 = float(np.sum(w * resid * resid) / dof)
    xtwx = (X * sw[:, None]).T @ (X * sw[:, None])
    try:
        se = np.sqrt(np.clip(np.diag(sigma2 * np.linalg.inv(xtwx)), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    return {
        "n": int(n),
        "beta": [float(b) for b in beta],
        "se": [float(s) for s in se],
        "r2": wr2(y, yhat, w),
        "y_mean": wmean(y, w),
    }


def fit_nested(y, off, deff, w) -> dict:
    y = np.asarray(y, dtype=float)
    off = np.asarray(off, dtype=float)
    deff = np.asarray(deff, dtype=float)
    w = np.asarray(w, dtype=float)
    base = wls(np.column_stack([np.ones(len(y)), off]), y, w)
    both = wls(np.column_stack([np.ones(len(y)), off, deff]), y, w)
    slope = wls(np.column_stack([np.ones(len(y)), deff]), y - off, w)
    return {
        "n_pairs": both["n"],
        "poss": float(np.nansum(w)),
        "y_mean": both["y_mean"],
        "r2_offense_only": base["r2"],
        "r2_offense_and_defender": both["r2"],
        "incremental_r2": both["r2"] - base["r2"] if np.isfinite(both["r2"]) else None,
        "defender_coef_on_raw": both["beta"][2],
        "defender_coef_se": both["se"][2],
        "offense_coef": both["beta"][1],
        "slope_of_residual_on_defender": slope["beta"][1],
        "slope_se": slope["se"][1],
        "residual_slope_r2": slope["r2"],
    }


def wdemean(values, weights, groups) -> np.ndarray:
    frame = pd.DataFrame({"v": values, "w": weights, "g": groups})
    frame["wv"] = frame["v"] * frame["w"]
    agg = frame.groupby("g", sort=False)[["wv", "w"]].sum()
    mu = frame["g"].map(agg["wv"] / agg["w"])
    return frame["v"].to_numpy(dtype=float) - mu.to_numpy(dtype=float)


def entropy(shares: np.ndarray) -> float:
    p = shares[shares > 0]
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def round_doc(obj):
    if isinstance(obj, dict):
        return {k: round_doc(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_doc(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        if not np.isfinite(obj):
            return None
        return round(float(obj), 4)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def parse_minutes(text: pd.Series) -> pd.Series:
    parts = text.fillna("").astype(str).str.split(":", n=1, expand=True)
    mins = pd.to_numeric(parts[0], errors="coerce")
    secs = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else 0
    return mins * 60 + secs.fillna(0)


def tier_ppg(ppg: float) -> str:
    if ppg >= 20:
        return "star (20+ ppg)"
    if ppg >= 10:
        return "medium (10–20 ppg)"
    return "role (<10 ppg)"


def load_players() -> pd.DataFrame:
    players = read(
        """
        SELECT a.player_id, a.player_name, a.gp, a.min, a.usg_pct, a.ts_pct,
               a.off_rating, a.def_rating, a.poss, a.pie,
               b.pts AS ppg, b.fga AS fga_pg,
               u.pct_pts, u.pct_fga
        FROM league.player_advanced_regular_season a
        JOIN league.player_base_regular_season b USING (player_id)
        JOIN league.player_usage_regular_season u USING (player_id)
        """
    )
    players = players.sort_values("gp", ascending=False).drop_duplicates("player_id")
    playoff = read(
        """
        SELECT player_id, gp AS po_gp, min AS po_min, usg_pct AS po_usg,
               ts_pct AS po_ts, off_rating AS po_off, def_rating AS po_def
        FROM league.player_advanced_playoffs
        """
    )
    playoff = playoff.sort_values("po_gp", ascending=False).drop_duplicates("player_id")
    touches = read(
        """
        SELECT player_id, touches, time_of_poss, pts_per_touch, paint_touches,
               post_touches, elbow_touches
        FROM league.tracking_possessions_player_regular_season
        """
    )
    touches = touches.sort_values("touches", ascending=False).drop_duplicates("player_id")
    return players.merge(playoff, on="player_id", how="left").merge(touches, on="player_id", how="left")


def load_matchups() -> pd.DataFrame:
    return read(
        """
        SELECT off_player_id, off_player_name, def_player_id, def_player_name,
               gp, partial_poss, player_pts, team_pts,
               matchup_fga, matchup_fgm, matchup_fg3_a, matchup_fg3_m,
               matchup_fta, matchup_ftm, matchup_tov, matchup_ast,
               matchup_time_sec, help_fga, help_fgm, sfl
        FROM league.matchups_regular_season
        WHERE partial_poss > 0
        """
    )


def prepare_pairs(matchups: pd.DataFrame, players: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    m = matchups.copy()
    off_tot = m.groupby("off_player_id")[["player_pts", "team_pts", "partial_poss", "matchup_fga", "matchup_fta"]].sum()
    off_tot = off_tot.rename(columns=lambda c: f"off_{c}")
    def_tot = m.groupby("def_player_id")[["player_pts", "team_pts", "partial_poss", "matchup_fga", "matchup_fta"]].sum()
    def_tot = def_tot.rename(columns=lambda c: f"def_{c}")
    m = m.join(off_tot, on="off_player_id").join(def_tot, on="def_player_id")

    league_player = m["player_pts"].sum() / m["partial_poss"].sum()
    league_team = m["team_pts"].sum() / m["partial_poss"].sum()
    league_fga = m["matchup_fga"].sum() / m["partial_poss"].sum()

    poss = m["partial_poss"]
    m["off_ppp"] = (m["off_player_pts"] - m["player_pts"]) / (m["off_partial_poss"] - poss)
    m["def_ppp"] = (m["def_player_pts"] - m["player_pts"]) / (m["def_partial_poss"] - poss)
    m["off_team"] = (m["off_team_pts"] - m["team_pts"]) / (m["off_partial_poss"] - poss)
    m["def_team"] = (m["def_team_pts"] - m["team_pts"]) / (m["def_partial_poss"] - poss)
    m["off_fga"] = (m["off_matchup_fga"] - m["matchup_fga"]) / (m["off_partial_poss"] - poss)
    m["def_fga"] = (m["def_matchup_fga"] - m["matchup_fga"]) / (m["def_partial_poss"] - poss)
    m["off_fta"] = (m["off_matchup_fta"] - m["matchup_fta"]) / (m["off_partial_poss"] - poss)

    shot_denom = m["matchup_fga"] + 0.44 * m["matchup_fta"]
    m["player_ppp"] = m["player_pts"] / poss
    m["team_ppp"] = m["team_pts"] / poss
    m["fga_per_poss"] = m["matchup_fga"] / poss
    m["ts"] = np.where(shot_denom > 0, m["player_pts"] / (2 * shot_denom), np.nan)
    m["help_rate"] = m["help_fga"] / poss
    m["tov_per_poss"] = m["matchup_tov"] / poss

    keep = (
        (poss >= PAIR_MIN_POSS)
        & (m["off_partial_poss"] >= BASE_MIN_POSS)
        & (m["def_partial_poss"] >= BASE_MIN_POSS)
        & np.isfinite(m["off_ppp"])
        & np.isfinite(m["def_ppp"])
    )
    pairs = m.loc[keep].copy()
    meta = players[["player_id", "ppg", "usg_pct", "ts_pct", "gp", "min"]].rename(
        columns={"player_id": "off_player_id", "ppg": "off_ppg", "usg_pct": "off_usg", "ts_pct": "off_ts", "gp": "off_gp", "min": "off_min"}
    )
    defs = players[["player_id", "ppg", "usg_pct", "def_rating"]].rename(
        columns={"player_id": "def_player_id", "ppg": "def_ppg", "usg_pct": "def_usg", "def_rating": "def_rating_season"}
    )
    pairs = pairs.merge(meta, on="off_player_id", how="left").merge(defs, on="def_player_id", how="left")
    pairs["tier"] = pairs["off_ppg"].map(lambda v: tier_ppg(v) if pd.notna(v) else "unknown")
    info = {
        "league_player_points_per_poss": float(league_player),
        "league_team_points_per_poss": float(league_team),
        "league_fga_per_poss": float(league_fga),
        "pair_min_poss": PAIR_MIN_POSS,
        "rows_in_fit": int(len(pairs)),
        "note": (
            "Rates are per on-court team possession, not per touch. "
            "A star at 0.30 scores about 0.30 points per team possession while he is on the floor. "
            "Team points per possession are the team's scoring on those same possessions, about 1.15."
        ),
    }
    return pairs, info


def identity_block(pairs: pd.DataFrame) -> dict:
    w = pairs["partial_poss"].to_numpy()
    overall = {
        "scorer_points_per_poss": fit_nested(pairs["player_ppp"], pairs["off_ppp"], pairs["def_ppp"], w),
        "team_points_per_poss": fit_nested(pairs["team_ppp"], pairs["off_team"], pairs["def_team"], w),
        "fga_per_poss": fit_nested(pairs["fga_per_poss"], pairs["off_fga"], pairs["def_fga"], w),
    }
    ts_rows = pairs[pairs["matchup_fga"] + 0.44 * pairs["matchup_fta"] >= 12].copy()
    # True shooting has no clean leave-one-out baseline built above; compare to the player's season TS.
    if len(ts_rows):
        ts_fit = wls(
            np.column_stack([
                np.ones(len(ts_rows)),
                ts_rows["off_ts"].to_numpy(),
                (ts_rows["def_ppp"] - pairs["def_ppp"].mean()).to_numpy(),
            ]),
            ts_rows["ts"].to_numpy(),
            ts_rows["partial_poss"].to_numpy(),
        )
        overall["true_shooting"] = {
            "n_pairs": ts_fit["n"],
            "r2": ts_fit["r2"],
            "defender_coef": ts_fit["beta"][2],
            "defender_coef_se": ts_fit["se"][2],
            "offense_coef": ts_fit["beta"][1],
            "note": "Defender coefficient is the change in true shooting for a 1.0 increase in the defender's points-allowed rate. Season true shooting is the offensive baseline.",
        }

    by_tier = {}
    for tier, chunk in pairs.groupby("tier"):
        if len(chunk) < 40:
            continue
        by_tier[tier] = {
            "scorer_points": fit_nested(chunk["player_ppp"], chunk["off_ppp"], chunk["def_ppp"], chunk["partial_poss"]),
            "team_points": fit_nested(chunk["team_ppp"], chunk["off_team"], chunk["def_team"], chunk["partial_poss"]),
            "fga_rate": fit_nested(chunk["fga_per_poss"], chunk["off_fga"], chunk["def_fga"], chunk["partial_poss"]),
        }

    help_fit = wls(
        np.column_stack([np.ones(len(pairs)), pairs["off_ppp"], pairs["help_rate"]]),
        pairs["player_ppp"],
        w,
    )
    help_team = wls(
        np.column_stack([np.ones(len(pairs)), pairs["off_team"], pairs["help_rate"]]),
        pairs["team_ppp"],
        w,
    )
    return {
        "overall": overall,
        "by_scoring_tier": by_tier,
        "help_defense": {
            "scorer_points_coef_on_help_fga_per_poss": help_fit["beta"][2],
            "scorer_points_se": help_fit["se"][2],
            "team_points_coef_on_help_fga_per_poss": help_team["beta"][2],
            "team_points_se": help_team["se"][2],
            "n": help_fit["n"],
            "note": "Help field-goal attempts per possession, after the offensive player's own scoring rate. A negative scorer coefficient with a flat team coefficient means the shot moved to a teammate.",
        },
    }


def examples(pairs: pd.DataFrame) -> dict:
    show = pairs[pairs["off_ppg"].ge(18) & pairs["off_gp"].ge(40)].copy()
    show["vs_own_baseline"] = show["player_ppp"] - show["off_ppp"]
    cols = [
        "off_player_name", "def_player_name", "gp", "partial_poss", "player_pts",
        "player_ppp", "off_ppp", "vs_own_baseline", "team_ppp", "def_ppp", "def_ppg",
    ]

    def pack(frame):
        rows = []
        for rec in frame[cols].to_dict("records"):
            rec["partial_poss"] = round(float(rec["partial_poss"]), 1)
            rec["player_pts"] = round(float(rec["player_pts"]), 1)
            for key in ("player_ppp", "off_ppp", "vs_own_baseline", "team_ppp", "def_ppp"):
                rec[key] = round(float(rec[key]), 3)
            rec["def_ppg"] = None if pd.isna(rec["def_ppg"]) else round(float(rec["def_ppg"]), 1)
            rec["gp"] = int(rec["gp"])
            rows.append(rec)
        return rows

    return {
        "largest_drop_vs_own_baseline": pack(show.nsmallest(12, "vs_own_baseline")),
        "largest_gain_vs_own_baseline": pack(show.nlargest(8, "vs_own_baseline")),
        "filter": "Offensive player at least 18 ppg and 40 games. Matchup at least 30 partial possessions. Drop is points per on-court possession minus that player's season rate.",
    }


def concentration(matchups: pd.DataFrame, players: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    rows = []
    for off_id, chunk in matchups.groupby("off_player_id"):
        poss = chunk["partial_poss"].to_numpy()
        total = poss.sum()
        if total < 400:
            continue
        shares = np.sort(poss / total)[::-1]
        h = entropy(poss)
        rows.append({
            "player_id": off_id,
            "top1_share": float(shares[0]),
            "top3_share": float(shares[:3].sum()),
            "n_defenders": int(len(shares)),
            "n_eff": float(np.exp(h)),
            "entropy_nats": h,
        })
    conc = pd.DataFrame(rows).merge(
        players[["player_id", "player_name", "ppg", "usg_pct", "gp"]],
        on="player_id",
        how="left",
    )
    conc["tier"] = conc["ppg"].map(lambda v: tier_ppg(v) if pd.notna(v) else "unknown")

    def summarize(frame):
        return {
            "players": int(len(frame)),
            "median_top1_share": float(frame["top1_share"].median()),
            "median_top3_share": float(frame["top3_share"].median()),
            "median_effective_defenders": float(frame["n_eff"].median()),
            "p10_top1": float(frame["top1_share"].quantile(0.1)),
            "p90_top1": float(frame["top1_share"].quantile(0.9)),
        }

    by_tier = {tier: summarize(chunk) for tier, chunk in conc.groupby("tier")}
    hist_edges = [0, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 1]
    hist_labels = ["under 6%", "6–8%", "8–10%", "10–12%", "12–15%", "15–20%", "20–30%", "over 30%"]
    counts = pd.cut(conc["top1_share"], hist_edges, labels=hist_labels, right=False).value_counts().reindex(hist_labels).fillna(0)
    return {
        "min_on_court_poss": 400,
        "overall": summarize(conc),
        "by_tier": by_tier,
        "top1_share_hist": [{"bin": lab, "players": int(n)} for lab, n in counts.items()],
        "note": (
            "Shares are of tracked matchup possessions, which themselves cover about 37% of playing time "
            "but are scaled so they sum to on-court possessions. Top-1 share is the primary defender's fraction of that."
        ),
    }, conc


def commit_and_duration(players: pd.DataFrame, pairs_baselines: pd.DataFrame) -> dict:
    games = read(
        """
        SELECT game_id, person_id_off, person_id_def,
               matchup_minutes_sort AS sec, partial_possessions AS poss,
               percentage_total_time_both_on AS commit,
               percentage_offensive_total_time AS off_share,
               player_points, team_points,
               matchup_field_goals_attempted AS fga
        FROM games.box_matchups_team
        WHERE partial_possessions > 0 AND matchup_minutes_sort > 0
        """
    )
    totals = read(
        """
        SELECT off_player_id AS player_id,
               sum(player_pts) / sum(partial_poss) AS off_ppp,
               sum(team_pts) / sum(partial_poss) AS off_team
        FROM league.matchups_regular_season
        GROUP BY 1
        """
    )
    def_totals = read(
        """
        SELECT def_player_id AS player_id,
               sum(player_pts) / sum(partial_poss) AS def_ppp,
               sum(team_pts) / sum(partial_poss) AS def_team,
               sum(matchup_fga) / sum(partial_poss) AS def_fga
        FROM league.matchups_regular_season
        GROUP BY 1
        """
    )
    usg = players.set_index("player_id")["usg_pct"]
    ppg = players.set_index("player_id")["ppg"]
    games = games.merge(totals.rename(columns={"player_id": "person_id_off"}), on="person_id_off", how="left")
    games = games.merge(def_totals.rename(columns={"player_id": "person_id_def"}), on="person_id_def", how="left")
    games["off_usg"] = games["person_id_off"].map(usg)
    games["off_ppg"] = games["person_id_off"].map(ppg)
    games["minutes"] = games["sec"] / 60.0
    games["player_ppp"] = games["player_points"] / games["poss"]
    games["resid"] = games["player_ppp"] - games["off_ppp"]

    # Commitment: when both are on the floor, fraction of that overlap assigned.
    commit = games[(games["commit"] > 0.05) & (games["commit"] <= 1) & games["def_ppp"].notna() & games["off_ppp"].notna()].copy()
    commit["overlap_sec"] = commit["sec"] / commit["commit"]
    # One row per pair across the 387-game sample, weighted by overlap.
    commit["commit_w"] = commit["commit"] * commit["overlap_sec"]
    pair = commit.groupby(["person_id_off", "person_id_def"], as_index=False).agg(
        commit_w=("commit_w", "sum"),
        overlap_sec=("overlap_sec", "sum"),
        def_ppp=("def_ppp", "first"),
        off_ppp=("off_ppp", "first"),
        off_usg=("off_usg", "first"),
    )
    pair["commit"] = pair["commit_w"] / pair["overlap_sec"]
    pair = pair[pair["overlap_sec"] >= 60]
    # Within the offensive player: do leakier defenders get a larger share of the overlap?
    off_commit = wcorr(
        wdemean(pair["commit"], pair["overlap_sec"], pair["person_id_off"]),
        wdemean(pair["def_ppp"], pair["overlap_sec"], pair["person_id_off"]),
        pair["overlap_sec"],
    )
    def_commit = wcorr(
        wdemean(pair["commit"], pair["overlap_sec"], pair["person_id_def"]),
        wdemean(pair["off_ppp"], pair["overlap_sec"], pair["person_id_def"]),
        pair["overlap_sec"],
    )
    # Slopes, same demeaning, so the intercept is zero and the coef is the within-player slope.
    hunt = wls(
        np.column_stack([
            np.ones(len(pair)),
            wdemean(pair["def_ppp"], pair["overlap_sec"], pair["person_id_off"]),
        ]),
        wdemean(pair["commit"], pair["overlap_sec"], pair["person_id_off"]),
        pair["overlap_sec"],
    )
    stick = wls(
        np.column_stack([
            np.ones(len(pair)),
            wdemean(pair["off_ppp"], pair["overlap_sec"], pair["person_id_def"]),
        ]),
        wdemean(pair["commit"], pair["overlap_sec"], pair["person_id_def"]),
        pair["overlap_sec"],
    )

    # Duration. Keep stints of at least 2 possessions so a single make does not dominate.
    dur = games[(games["poss"] >= 2) & games["off_ppp"].notna() & games["def_ppp"].notna()].copy()
    bins = [0, 1, 2, 4, 7, 20]
    labels = ["under 1 min", "1–2 min", "2–4 min", "4–7 min", "7+ min"]
    dur["bin"] = pd.cut(dur["minutes"], bins, labels=labels, right=False)
    bin_rows = []
    for lab, chunk in dur.groupby("bin", observed=True):
        bin_rows.append({
            "assignment": str(lab),
            "rows": int(len(chunk)),
            "poss": round(float(chunk["poss"].sum()), 1),
            "scorer_points_vs_own_rate": round(wmean(chunk["resid"], chunk["poss"]), 4),
            "scorer_points_per_poss": round(wmean(chunk["player_ppp"], chunk["poss"]), 4),
            "team_points_per_poss": round(float(np.average(chunk["team_points"] / chunk["poss"], weights=chunk["poss"])), 4),
        })
    dur_fit = wls(
        np.column_stack([
            np.ones(len(dur)),
            dur["minutes"],
            dur["def_ppp"],
            dur["off_ppp"],
        ]),
        dur["player_ppp"],
        dur["poss"],
    )
    # Stars only, same regression.
    stars = dur[dur["off_ppg"].ge(20)]
    star_fit = wls(
        np.column_stack([np.ones(len(stars)), stars["minutes"], stars["def_ppp"], stars["off_ppp"]]),
        stars["player_ppp"],
        stars["poss"],
    ) if len(stars) > 50 else None

    return {
        "games": int(games["game_id"].nunique()),
        "commit": {
            "pairs": int(len(pair)),
            "within_offensive_player_corr_commit_vs_defender_ppp": off_commit,
            "within_offensive_player_slope": hunt["beta"][1],
            "within_offensive_player_slope_se": hunt["se"][1],
            "within_defender_corr_commit_vs_scorer_ppp": def_commit,
            "within_defender_slope": stick["beta"][1],
            "within_defender_slope_se": stick["se"][1],
            "reading": (
                "Commit is the fraction of shared floor time spent as the matchup. "
                "Positive within-scorer correlation means a larger share of the overlap is assigned "
                "when the defender gives up more points to everyone else (the offense hunts, or the defense fails to hide him). "
                "Positive within-defender correlation means a defender spends a larger share of the overlap on scorers who score more on everyone else."
            ),
        },
        "duration": {
            "bins": bin_rows,
            "minutes_coef_on_scorer_ppp": dur_fit["beta"][1],
            "minutes_coef_se": dur_fit["se"][1],
            "defender_coef": dur_fit["beta"][2],
            "n_stints": dur_fit["n"],
            "r2": dur_fit["r2"],
            "stars_minutes_coef": None if star_fit is None else star_fit["beta"][1],
            "stars_minutes_se": None if star_fit is None else star_fit["se"][1],
            "stars_n": None if star_fit is None else star_fit["n"],
            "note": (
                "Each row is the whole assignment in one game, not a running clock. "
                "The coefficient is the change in the scorer's points per possession per extra minute of that assignment, "
                "holding his season rate and the defender's season rate fixed. "
                "It is not a within-stint decay curve. The tracking feed has no timestamp for when the assignment started."
            ),
        },
    }


def fatigue(players: pd.DataFrame) -> dict:
    games = read(
        """
        SELECT game_id, person_id_off, person_id_def,
               matchup_minutes_sort AS sec, partial_possessions AS poss
        FROM games.box_matchups_team
        WHERE matchup_minutes_sort > 0
        """
    )
    box = read(
        """
        SELECT t.game_id, t.person_id, t.minutes, t.points,
               u.usage_percentage, a.possessions
        FROM games.box_traditional_player t
        LEFT JOIN games.box_usage_player u USING (game_id, person_id)
        LEFT JOIN games.box_advanced_player a USING (game_id, person_id)
        WHERE t.minutes LIKE '%:%'
        """
    )
    box["sec_played"] = parse_minutes(box["minutes"])
    usg = players.set_index("player_id")["usg_pct"]
    season_usg = players.set_index("player_id")["usg_pct"]
    games["off_usg"] = games["person_id_off"].map(usg)
    games["star_sec"] = np.where(games["off_usg"].ge(0.25), games["sec"], 0.0)
    # Highest-usage opponent matched in that game, per defender.
    games["off_usg_filled"] = games["off_usg"].fillna(-1)
    idx = games.groupby(["game_id", "person_id_def"])["off_usg_filled"].idxmax()
    top = games.loc[idx, ["game_id", "person_id_def", "sec"]].rename(columns={"sec": "sec_on_their_best"})
    grouped = games.groupby(["game_id", "person_id_def"], as_index=False).agg(
        def_sec=("sec", "sum"),
        star_sec=("star_sec", "sum"),
    )
    grouped = grouped.merge(top, on=["game_id", "person_id_def"], how="left")
    merged = box.merge(grouped, left_on=["game_id", "person_id"], right_on=["game_id", "person_id_def"], how="inner")
    merged["season_usg"] = merged["person_id"].map(season_usg)
    merged = merged[(merged["sec_played"] >= 12 * 60) & merged["season_usg"].ge(0.18) & (merged["def_sec"] > 0)]
    counts = merged.groupby("person_id")["game_id"].transform("count")
    merged = merged[counts >= 6].copy()
    merged["star_share"] = merged["star_sec"] / merged["def_sec"]
    merged["best_share"] = merged["sec_on_their_best"] / merged["def_sec"]
    merged["ppm"] = merged["points"] / (merged["sec_played"] / 60.0)
    w = np.ones(len(merged))
    y_ppm = wdemean(merged["ppm"], w, merged["person_id"])
    y_usg = wdemean(merged["usage_percentage"], w, merged["person_id"])
    x_star = wdemean(merged["star_share"], w, merged["person_id"])
    x_best = wdemean(merged["best_share"], w, merged["person_id"])
    x_min = wdemean(merged["sec_played"] / 60.0, w, merged["person_id"])
    ppm_star = wls(np.column_stack([np.ones(len(merged)), x_star, x_min]), y_ppm, w)
    usg_star = wls(np.column_stack([np.ones(len(merged)), x_star, x_min]), y_usg, w)
    ppm_best = wls(np.column_stack([np.ones(len(merged)), x_best, x_min]), y_ppm, w)
    return {
        "player_games": int(len(merged)),
        "players": int(merged["person_id"].nunique()),
        "games": int(merged["game_id"].nunique()),
        "who": "Players with season usage at least 18% and at least 6 games of 12+ minutes in the 387-game sample.",
        "points_per_minute_on_star_share": ppm_star["beta"][1],
        "points_per_minute_se": ppm_star["se"][1],
        "usage_on_star_share": usg_star["beta"][1],
        "usage_se": usg_star["se"][1],
        "points_per_minute_on_best_opponent_share": ppm_best["beta"][1],
        "points_per_minute_best_se": ppm_best["se"][1],
        "mean_star_share": float(merged["star_share"].mean()),
        "note": (
            "Within-player. Star share is the fraction of this player's tracked defensive matchup time "
            "spent on opponents whose season usage is at least 25%. "
            "The coefficient is the change in his own points per minute when that fraction goes from 0 to 1, "
            "holding his minutes played fixed. "
            "Games against a team with a star are also games against that team's defense, so this is not a pure fatigue effect."
        ),
    }


def defender_load(players: pd.DataFrame, matchups: pd.DataFrame, conc: pd.DataFrame) -> dict:
    def_poss = matchups.groupby("def_player_id")["partial_poss"].sum().rename("def_poss")
    off_poss = matchups.groupby("off_player_id")["partial_poss"].sum().rename("off_poss")
    frame = players.merge(def_poss, left_on="player_id", right_index=True, how="left")
    frame = frame.merge(off_poss, left_on="player_id", right_index=True, how="left")
    frame = frame.merge(conc[["player_id", "top1_share", "top3_share", "n_eff", "entropy_nats"]], on="player_id", how="left")
    frame["minutes"] = frame["gp"] * frame["min"]
    frame = frame[frame["minutes"].gt(200) & frame["def_poss"].notna() & frame["usg_pct"].notna()]
    frame["def_poss_per_36"] = frame["def_poss"] / frame["minutes"] * 36
    frame["off_poss_per_36"] = frame["off_poss"] / frame["minutes"] * 36
    w = frame["minutes"].to_numpy()
    corr_usg = wcorr(frame["usg_pct"], frame["def_poss_per_36"], w)
    corr_touch = wcorr(frame["usg_pct"], frame["touches"], frame["minutes"].where(frame["touches"].notna(), 0))
    # High-usage players do not necessarily guard the ball. Split the correlation by ppg.
    return {
        "players": int(len(frame)),
        "corr_usage_vs_def_poss_per_36": corr_usg,
        "corr_usage_vs_touches_per_game": wcorr(frame["usg_pct"], frame["touches"], w),
        "corr_ppg_vs_def_poss_per_36": wcorr(frame["ppg"], frame["def_poss_per_36"], w),
        "corr_usage_vs_off_poss_per_36": wcorr(frame["usg_pct"], frame["off_poss_per_36"], w),
        "median_def_poss_per_36": float(frame["def_poss_per_36"].median()),
        "median_off_poss_per_36": float(frame["off_poss_per_36"].median()),
        "note": (
            "Defensive possessions per 36 are the on-court possessions of the players he was assigned to, scaled as in the matchup table. "
            "Offensive possessions per 36 are his own on-court possessions and mostly track minutes, so usage should be nearly uncorrelated with them. "
            "Touches are per game from the tracking table."
        ),
        "_frame": frame,
    }


def shot_zones() -> dict:
    frames = {}
    for season, table in (
        ("regular_season", "league.player_shot_locations_regular_season"),
        ("playoffs", "league.player_shot_locations_playoffs"),
    ):
        loc = read(f"SELECT * FROM {table}")
        gp_table = "league.player_base_regular_season" if season == "regular_season" else "league.player_base_playoffs"
        gp = read(f"SELECT player_id, gp, pts FROM {gp_table}")
        gp = gp.sort_values("gp", ascending=False).drop_duplicates("player_id")
        loc = loc.merge(gp, on="player_id", how="left")
        frames[season] = loc

    def zone_table(loc: pd.DataFrame) -> list[dict]:
        rows = []
        loc = loc[loc["gp"].fillna(0) >= 1]
        for label, key, value in ZONES:
            fga = loc[f"{key}_fga"] * loc["gp"]
            fgm = loc[f"{key}_fgm"] * loc["gp"]
            attempts = float(fga.sum())
            makes = float(fgm.sum())
            rows.append({
                "zone": label,
                "shot_value": value,
                "attempts": round(attempts),
                "fg_pct": makes / attempts if attempts else None,
                "points_per_shot": value * makes / attempts if attempts else None,
            })
        total_att = sum(r["attempts"] for r in rows)
        for r in rows:
            r["share_of_attempts"] = r["attempts"] / total_att if total_att else None
        return rows

    rs = frames["regular_season"]
    rs = rs[rs["gp"].fillna(0) >= 20].copy()
    # Across players: do players who shoot a zone well shoot it more often?
    zone_corrs = []
    for label, key, value in ZONES:
        fga = rs[f"{key}_fga"]
        fg = rs[f"{key}_fg_pct"]
        use = fga.ge(1.0) & fg.notna()
        if use.sum() < 30:
            continue
        pps = fg[use] * value
        freq = fga[use] / sum(rs.loc[use, f"{k}_fga"] for _, k, _ in ZONES)
        zone_corrs.append({
            "zone": label,
            "players": int(use.sum()),
            "corr_frequency_vs_points_per_shot": wcorr(freq, pps, fga[use] * rs.loc[use, "gp"]),
        })

    # Each player's best zone by points per shot, among zones with 40 attempts.
    best_rows = []
    for rec in rs.itertuples(index=False):
        options = []
        total_fga = 0
        for label, key, value in ZONES:
            fga = getattr(rec, f"{key}_fga") * rec.gp
            fg = getattr(rec, f"{key}_fg_pct")
            total_fga += getattr(rec, f"{key}_fga")
            if fga >= 40 and pd.notna(fg):
                options.append((label, value * fg, getattr(rec, f"{key}_fga") / max(total_fga, 1e-9)))
        if not options:
            continue
        # frequency needs the full total; recompute
        total = sum(getattr(rec, f"{key}_fga") for _, key, _ in ZONES)
        options = []
        for label, key, value in ZONES:
            fga = getattr(rec, f"{key}_fga") * rec.gp
            fg = getattr(rec, f"{key}_fg_pct")
            if fga >= 40 and pd.notna(fg) and total > 0:
                options.append((label, value * fg, getattr(rec, f"{key}_fga") / total, fga))
        if not options:
            continue
        label, pps, freq, fga = max(options, key=lambda t: t[1])
        best_rows.append({"player_id": rec.player_id, "best_zone": label, "best_pps": pps, "best_freq": freq})
    best = pd.DataFrame(best_rows)
    best_counts = best["best_zone"].value_counts().to_dict() if len(best) else {}

    po = frames["playoffs"]
    po = po[po["gp"].fillna(0) >= 5]
    # Frequency shift, players in both.
    shifts = []
    merged = rs.merge(po, on="player_id", suffixes=("_rs", "_po"))
    for label, key, _value in ZONES:
        rs_tot = sum(merged[f"{k}_fga_rs"] for _, k, _ in ZONES)
        po_tot = sum(merged[f"{k}_fga_po"] for _, k, _ in ZONES)
        # per player
        rs_share = merged[f"{key}_fga_rs"] / rs_tot.replace(0, np.nan)
        po_share = merged[f"{key}_fga_po"] / po_tot.replace(0, np.nan)
        # rs_tot is a series already
        delta = (po_share - rs_share).dropna()
        shifts.append({
            "zone": label,
            "players": int(delta.shape[0]),
            "mean_share_change_playoffs_minus_rs": float(delta.mean()) if len(delta) else None,
        })

    return {
        "regular_season": zone_table(frames["regular_season"]),
        "playoffs": zone_table(frames["playoffs"]),
        "corr_player_frequency_vs_efficiency": zone_corrs,
        "best_zone_counts": {k: int(v) for k, v in best_counts.items()},
        "players_with_a_best_zone": int(len(best)),
        "median_frequency_of_best_zone": float(best["best_freq"].median()) if len(best) else None,
        "playoff_share_shift": shifts,
        "_best": best,
    }


def play_types() -> dict:
    rows = []
    player_long = []
    for pt in PLAY_TYPES:
        for side in ("offensive", "defensive"):
            if side == "defensive" and pt in DEFENSE_MISSING:
                continue
            for season in ("regular_season", "playoffs"):
                table = f"league.synergy_{pt}_{side}_player_{season}"
                try:
                    df = read(
                        f"""
                        SELECT player_id, player_name, gp, poss, ppp, poss_pct, percentile, pts
                        FROM {table}
                        """
                    )
                except Exception:
                    continue
                if df.empty:
                    continue
                df = df[df["poss"].fillna(0) > 0]
                weight = df["poss"] * df["gp"]
                rows.append({
                    "play_type": pt,
                    "side": side,
                    "season": season,
                    "players": int(len(df)),
                    "ppp": wmean(df["ppp"], weight),
                    "poss_per_game": wmean(df["poss"], df["gp"]),
                    "mean_poss_pct": wmean(df["poss_pct"], weight),
                })
                if side == "offensive" and season == "regular_season":
                    part = df[["player_id", "gp", "poss", "ppp", "poss_pct"]].copy()
                    part["play_type"] = pt
                    part["weight"] = part["poss"] * part["gp"]
                    player_long.append(part)
    long = pd.concat(player_long, ignore_index=True)
    # Within player, spread between best and worst play type with enough volume.
    long = long[long["weight"] >= 40]
    spreads = []
    mixes = []
    for pid, chunk in long.groupby("player_id"):
        if len(chunk) < 3:
            continue
        spreads.append(float(chunk["ppp"].max() - chunk["ppp"].min()))
        mixes.append(entropy(chunk["weight"].to_numpy()))
    # Which play type's PPP varies most across players.
    spread_by_type = []
    for pt, chunk in long.groupby("play_type"):
        if len(chunk) < 30:
            continue
        spread_by_type.append({
            "play_type": pt,
            "players": int(len(chunk)),
            "ppp_p10": float(chunk["ppp"].quantile(0.1)),
            "ppp_p50": float(chunk["ppp"].quantile(0.5)),
            "ppp_p90": float(chunk["ppp"].quantile(0.9)),
        })
    return {
        "league": rows,
        "within_player_best_minus_worst_ppp": {
            "players": len(spreads),
            "median": float(np.median(spreads)) if spreads else None,
            "p90": float(np.quantile(spreads, 0.9)) if spreads else None,
        },
        "ppp_spread_across_players": spread_by_type,
        "note": "Synergy tables are per-game rates. Weights are possessions per game times games. A player needs 40 such possessions in a play type to enter the within-player spread. Defensive player rows do not exist for cuts, putbacks, transition, or misc.",
    }


def closest_defender() -> dict:
    rows = []
    frames = {}
    specs = (
        ("overall", "league.pt_defend_overall_regular_season", "d_fga", "d_fg_pct", "normal_fg_pct", "pct_plusminus"),
        ("under 6 ft", "league.pt_defend_less_than_6ft_regular_season", "fga_lt_06", "lt_06_pct", "ns_lt_06_pct", "plusminus"),
        ("under 10 ft", "league.pt_defend_less_than_10ft_regular_season", "fga_lt_10", "lt_10_pct", "ns_lt_10_pct", "plusminus"),
        ("over 15 ft", "league.pt_defend_greater_than_15ft_regular_season", "fga_gt_15", "gt_15_pct", "ns_gt_15_pct", "plusminus"),
        ("twos", "league.pt_defend_2_pointers_regular_season", "fg2_a", "fg2_pct", "ns_fg2_pct", "plusminus"),
        ("threes", "league.pt_defend_3_pointers_regular_season", "fg3_a", "fg3_pct", "ns_fg3_pct", "plusminus"),
    )
    for cat, table, fga_col, pct_col, normal_col, plus_col in specs:
        df = read(
            f"""
            SELECT close_def_person_id AS player_id, g,
                   {fga_col} AS d_fga, {pct_col} AS d_fg_pct,
                   {normal_col} AS normal_fg_pct, {plus_col} AS pct_plusminus
            FROM {table}
            """
        )
        df = df[df["d_fga"].fillna(0) > 0]
        weight = df["d_fga"] * df["g"].fillna(1)
        rows.append({
            "distance": cat,
            "players": int(len(df)),
            "mean_fg_pct_allowed": wmean(df["d_fg_pct"], weight),
            "mean_shooter_normal_fg_pct": wmean(df["normal_fg_pct"], weight),
            "mean_pct_plusminus": wmean(df["pct_plusminus"], weight),
            "sd_pct_plusminus": float(df["pct_plusminus"].std()),
        })
        if cat == "overall":
            frames["overall"] = df.assign(weight=weight)
    po = read(
        """
        SELECT close_def_person_id AS player_id, g, d_fga, pct_plusminus
        FROM league.pt_defend_overall_playoffs
        """
    )
    rs = frames["overall"][["player_id", "pct_plusminus", "weight", "d_fg_pct"]].merge(
        po.rename(columns={"pct_plusminus": "po_plus", "d_fga": "po_fga", "g": "po_g"}),
        on="player_id",
        how="inner",
    )
    rs = rs[rs["po_g"].fillna(0) >= 5]
    return {
        "by_distance": rows,
        "playoff_vs_rs_corr_of_pct_plusminus": wcorr(rs["pct_plusminus"], rs["po_plus"], rs["weight"]) if len(rs) else None,
        "playoff_players": int(len(rs)),
        "mean_playoff_minus_rs": float((rs["po_plus"] - rs["pct_plusminus"]).mean()) if len(rs) else None,
        "_overall": frames["overall"],
        "note": "pct_plusminus is the closest defender's field-goal percentage minus the shooter's usual percentage, on shots they contested. Negative means the shooter made less than usual. This is the contest at the shot, not the assigned matchup.",
    }


def playoff_player_shift(players: pd.DataFrame) -> dict:
    both = players[(players["gp"] >= 20) & (players["po_gp"] >= 5)].copy()
    return {
        "players": int(len(both)),
        "mean_usage_change": float((both["po_usg"] - both["usg_pct"]).mean()),
        "mean_ts_change": float((both["po_ts"] - both["ts_pct"]).mean()),
        "mean_off_rating_change": float((both["po_off"] - both["off_rating"]).mean()),
        "mean_def_rating_change": float((both["po_def"] - both["def_rating"]).mean()),
        "corr_rs_usage_vs_playoff_usage": float(both["usg_pct"].corr(both["po_usg"])),
        "corr_rs_ts_vs_playoff_ts": float(both["ts_pct"].corr(both["po_ts"])),
        "note": "Playoff minus regular season. Defensive rating down is better. No playoff matchup table was loaded, so who-guards-whom cannot be compared.",
    }


def link_contest_to_matchup(closest: pd.DataFrame, matchups: pd.DataFrame) -> dict:
    allowed = matchups.groupby("def_player_id").apply(
        lambda d: pd.Series({
            "def_ppp": d["player_pts"].sum() / d["partial_poss"].sum(),
            "poss": d["partial_poss"].sum(),
        }),
        include_groups=False,
    ).reset_index()
    merged = closest.merge(allowed, left_on="player_id", right_on="def_player_id", how="inner")
    merged = merged[merged["poss"] >= 400]
    return {
        "players": int(len(merged)),
        "corr_contest_plusminus_vs_matchup_ppp_allowed": wcorr(merged["pct_plusminus"], merged["def_ppp"], merged["weight"]),
        "note": "Positive correlation means defenders who let shooters beat their usual percentage also give up more points as the assigned matchup. The two measurements are not the same event.",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading")
    players = load_players()
    matchups = load_matchups()
    pairs, scale = prepare_pairs(matchups, players)
    print(f"pairs {len(pairs)}")

    identity = identity_block(pairs)
    ex = examples(pairs)
    conc_summary, conc = concentration(matchups, players)
    commit = commit_and_duration(players, pairs)
    tired = fatigue(players)
    load = defender_load(players, matchups, conc)
    frame = load.pop("_frame")
    zones = shot_zones()
    best = zones.pop("_best")
    actions = play_types()
    contest = closest_defender()
    overall_contest = contest.pop("_overall")
    contest_link = link_contest_to_matchup(overall_contest, matchups)
    po_shift = playoff_player_shift(players)

    # Coverage of tracked minutes.
    cover = read(
        """
        SELECT sum(matchup_time_sec) / 60.0 AS matchup_min,
               count(DISTINCT off_player_id) AS players
        FROM league.matchups_regular_season
        """
    )
    played = float((players["gp"] * players["min"]).sum())
    coverage = {
        "regular_season_matchup_rows": int(len(matchups)),
        "tracked_matchup_minutes": float(cover["matchup_min"].iloc[0]),
        "player_minutes_in_advanced_table": played,
        "matchup_minutes_over_player_minutes": float(cover["matchup_min"].iloc[0]) / played,
        "game_sample": "387 regular-season games with box matchups. Ids span the season but cluster in the early games plus a late slice.",
        "switches": "games.box_matchups_team.switches_on and games.box_defensive_player.switches_on are 0 on every row.",
        "playoff_matchups": "The playoff matchup request returned no rows.",
        "lineups": "Five-man lineup tables have no opponent lineup. They cannot answer which unit to answer a unit with.",
        "partial_possessions": scale["note"],
    }

    # Feature tables for a later model.
    feat_cols = [
        "off_player_id", "off_player_name", "def_player_id", "def_player_name",
        "gp", "partial_poss", "matchup_time_sec", "player_pts", "team_pts",
        "player_ppp", "team_ppp", "fga_per_poss", "ts", "tov_per_poss", "help_rate",
        "off_ppp", "def_ppp", "off_team", "def_team", "off_fga", "def_fga",
        "off_ppg", "off_usg", "off_ts", "def_ppg", "def_usg", "tier",
    ]
    pairs[feat_cols].to_csv(OUT / "matchup_pairs.csv", index=False)

    within_game = read(
        """
        WITH s AS (
          SELECT game_id, person_id_off AS player_id,
                 partial_possessions / sum(partial_possessions)
                   OVER (PARTITION BY game_id, person_id_off) AS share,
                 sum(partial_possessions) OVER (PARTITION BY game_id, person_id_off) AS poss
          FROM games.box_matchups_team
          WHERE partial_possessions > 0
        ),
        g AS (
          SELECT game_id, player_id,
                 max(share) AS top1,
                 count(*) AS n_def,
                 exp(-sum(share * ln(share))) AS n_eff
          FROM s
          WHERE poss >= 20 AND share > 0
          GROUP BY 1, 2
        )
        SELECT player_id,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY top1) AS game_median_top1_share,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY n_eff) AS game_median_n_eff,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY n_def) AS game_median_n_defenders,
               count(*) AS game_sample_games
        FROM g
        GROUP BY 1
        """
    )
    player_feat = frame.merge(best, on="player_id", how="left").merge(within_game, on="player_id", how="left")
    keep_cols = [
        "player_id", "player_name", "gp", "min", "ppg", "usg_pct", "ts_pct",
        "off_rating", "def_rating", "touches", "pts_per_touch", "time_of_poss",
        "def_poss_per_36", "off_poss_per_36", "top1_share", "top3_share", "n_eff",
        "entropy_nats", "game_median_top1_share", "game_median_n_eff",
        "game_median_n_defenders", "game_sample_games",
        "best_zone", "best_pps", "best_freq",
        "po_gp", "po_usg", "po_ts", "po_off", "po_def",
    ]
    player_feat[keep_cols].to_csv(OUT / "player_features.csv", index=False)

    summary = {
        "scale": scale,
        "coverage": coverage,
        "does_the_defender_matter": identity,
        "examples": ex,
        "how_concentrated_is_the_assignment": conc_summary,
        "who_is_assigned_when_both_are_on": commit,
        "scorer_guarding_a_scorer": tired,
        "offensive_load_vs_defensive_load": {k: v for k, v in load.items()},
        "shot_zones": zones,
        "play_types": actions,
        "closest_defender": contest,
        "contest_vs_assignment": contest_link,
        "playoffs_vs_regular_season": po_shift,
    }
    text = json.dumps(round_doc(summary), indent=2)
    (OUT / "summary.json").write_text(text)
    print(text)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
