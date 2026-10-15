"""S4-46 extensions. Where does the extra seed cost of an anytime-valid comparison come from?

Two properties are bundled in the betting test of s4_anytime.py: it is valid at
every stopping time (anytime), and it is valid for any distribution on a declared
bounded interval (distribution-free). This script separates them by adding tests
that have one property but not the other:

  t_fixed      one-look t-test at a pre-committed n              (neither)
  t_peek       t-test after every seed, stop at first p < alpha  (neither; invalid)
  gs_pocock    group-sequential t, look after every seed, constant boundary
               calibrated by Monte Carlo under the Gaussian null (anytime up to
               a pre-committed maximum n; parametric)
  gs_obf       same with an O'Brien-Fleming shaped boundary
  seq_t        anytime-valid t-test e-process: likelihood ratio of the t-statistic
               under a Gaussian prior on the standardized effect vs. the central
               t density (Perez-Ortiz, Lardy, de Heide, Grunwald 2022). Anytime,
               parametric, no maximum n.
  bet_fixed    the betting martingale of s4_anytime.py evaluated once at a fixed n,
               rejecting when wealth >= 1/alpha (Markov)   (distribution-free; not anytime)
  bet_plugin   the same martingale stopped at first crossing (both; Ville)
  eb_cs        predictable-plug-in empirical-Bernstein confidence sequence for a
               bounded mean (Waudby-Smith & Ramdas 2023), reject when it excludes
               the null (both)

Data:
  * beta differences at a fixed gap, three concentrations, as in the paper
  * real seeds: BBF Atari-100k human-normalized scores (Schwarzer et al. 2023,
    google-research/bigger_better_faster/scores), 55 games, 12-30 runs per game.
    Scores are clipped to [0, 1] (rliable's optimality-gap transform), so a
    paired difference is in [-1, 1] by construction.

    python code/s4_extensions.py --quick
    python code/s4_extensions.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import roots_hermitenorm

HERE = Path(__file__).resolve().parent  # code/
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

OUT = PROJECT / "results"
DATA = PROJECT / "data" / "bbf"


# ----------------------------------------------------------------------------
# Test statistics along a path. Every function takes a 1-D array of paired
# differences x (raw scale) and returns, per n, whether the test has rejected by n.
# ----------------------------------------------------------------------------


def t_stats_along_path(x: np.ndarray) -> np.ndarray:
    """T_n = sqrt(n) mean / sd for n = 1..len(x); T_1 = 0, sd = 0 -> +-inf handled."""
    n = np.arange(1, len(x) + 1)
    csum = np.cumsum(x)
    csq = np.cumsum(x * x)
    mean = csum / n
    var = np.maximum((csq - n * mean**2) / np.maximum(n - 1, 1), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.sqrt(n) * mean / np.sqrt(var)
    t[0] = 0.0
    zero_var = var < 1e-18
    t[zero_var & (np.abs(mean) > 1e-12)] = np.inf
    t[zero_var & (np.abs(mean) <= 1e-12)] = 0.0
    return t


def t_pvalues(t: np.ndarray) -> np.ndarray:
    n = np.arange(1, len(t) + 1)
    p = 2.0 * stats.t.sf(np.abs(t), df=np.maximum(n - 1, 1))
    p[0] = 1.0
    return p


_GS_CACHE: dict = {}


def group_sequential_design(nmax: int, alpha: float, looks: int, rng: np.random.Generator, paths: int = 20000) -> dict:
    """K equally spaced looks up to a planned maximum nmax. Two classical boundaries on the
    two-sided z-scale z_k = Phi^{-1}(1 - p_k / 2), where p_k is the t-test p-value at look k:
      Pocock          z_k >= c_P                 (constant)
      O'Brien-Fleming z_k >= c_O sqrt(K / k)     (strict early, lenient late)
    Constants are calibrated by Monte Carlo under the Gaussian null so the probability of
    crossing at any look is alpha. The design is anytime-valid only up to nmax, and only under
    the parametric model: it is the clinical-trials answer to peeking."""
    key = (nmax, alpha, looks)
    if key in _GS_CACHE:
        return _GS_CACHE[key]
    look_n = np.unique(np.maximum(2, np.round(nmax * np.arange(1, looks + 1) / looks)).astype(int))
    k = np.arange(1, len(look_n) + 1)
    max_p = np.empty(paths)
    max_o = np.empty(paths)
    for i in range(paths):
        t = t_stats_along_path(rng.normal(size=nmax))
        z = stats.norm.isf(t_pvalues(t)[look_n - 1] / 2.0)
        max_p[i] = z.max()
        max_o[i] = (z * np.sqrt(k / len(look_n))).max()
    design = {
        "nmax": int(nmax),
        "look_n": look_n.tolist(),
        "pocock_c": float(np.quantile(max_p, 1 - alpha)),
        "obf_c": float(np.quantile(max_o, 1 - alpha)),
    }
    _GS_CACHE[key] = design
    return design


def group_sequential_first_reject(p: np.ndarray, design: dict, shape: str) -> int:
    """1-based n of the first look that crosses, or 0."""
    look_n = np.asarray(design["look_n"])
    look_n = look_n[look_n <= len(p)]
    z = stats.norm.isf(p[look_n - 1] / 2.0)
    k = np.arange(1, len(look_n) + 1)
    if shape == "pocock":
        bound = np.full(len(look_n), design["pocock_c"])
    else:
        bound = design["obf_c"] * np.sqrt(len(design["look_n"]) / k)
    hit = np.flatnonzero(z >= bound)
    return int(look_n[hit[0]]) if hit.size else 0


def fixed_t_n80(draw_path, rng: np.random.Generator, cap: int, alpha: float, paths: int = 300, power: float = 0.8) -> int | None:
    mat = np.stack([t_pvalues(t_stats_along_path(np.asarray(draw_path(rng))[:cap])) < alpha for _ in range(paths)])
    hit = np.flatnonzero(mat.mean(axis=0) >= power)
    return int(hit[0] + 1) if hit.size else None


_GH_NODES, _GH_WEIGHTS = roots_hermitenorm(40)
_GH_WEIGHTS = _GH_WEIGHTS / _GH_WEIGHTS.sum()


def sequential_t_log_evalue(t: np.ndarray, prior_sd: float) -> np.ndarray:
    """log E_n for the t-test e-process with delta ~ N(0, prior_sd^2) on the standardized effect.

    E_n = int f_nct(T_n; n-1, delta sqrt(n)) dPi(delta) / f_t(T_n; n-1).
    The density ratio of the scale-maximal-invariant depends on the data only through T_n,
    and the sequence is a nonnegative martingale under H0 w.r.t. the invariant filtration.
    """
    n = np.arange(1, len(t) + 1)
    df = np.maximum(n - 1, 1)
    deltas = prior_sd * _GH_NODES  # (q,)
    tt = np.clip(t, -1e6, 1e6)
    with np.errstate(all="ignore"):
        num = stats.nct.pdf(tt[:, None], df[:, None], deltas[None, :] * np.sqrt(n)[:, None])
        num = np.nansum(num * _GH_WEIGHTS[None, :], axis=1)
        den = stats.t.pdf(tt, df)
        log_e = np.log(np.maximum(num, 1e-300)) - np.log(np.maximum(den, 1e-300))
    log_e[0] = 0.0
    # Degenerate (zero variance) steps: a huge |T| with finite n is overwhelming evidence; cap it.
    log_e = np.where(np.isfinite(tt), log_e, 50.0)
    return log_e


def betting_log_wealth(unit: np.ndarray, null: float, bound_fraction: float = 0.99) -> np.ndarray:
    """Plug-in betting martingale from probes/push_further.PlugInTest, vectorized along the path."""
    n_steps = len(unit)
    scale = max(null, 1.0 - null, 1e-3)
    bound = bound_fraction / scale
    log_w = np.zeros(n_steps)
    total = 0.0
    sumsq = 0.0
    acc = 0.0
    for i in range(n_steps):
        if i == 0:
            lam = 0.0
        else:
            mu = total / i
            moment = sumsq / i - 2.0 * null * mu + null**2
            lam = float(np.clip((mu - null) / max(moment, 1e-6), -bound, bound))
        acc += float(np.log(max(1.0 + lam * (unit[i] - null), 1e-12)))
        log_w[i] = acc
        total += unit[i]
        sumsq += unit[i] * unit[i]
    return log_w


def eb_cs_excludes(unit: np.ndarray, null: float, alpha: float, lam_cap: float = 0.5) -> np.ndarray:
    """Predictable plug-in empirical-Bernstein CS (Waudby-Smith & Ramdas 2023, Thm 2), for x in [0,1].
    Returns per-n whether the running intersection has excluded `null`."""
    n_steps = len(unit)
    log_term = np.log(2.0 / alpha)
    mu_hat = 0.5
    var_hat = 0.25
    sum_lam = 0.0
    sum_lam_x = 0.0
    sum_v_psi = 0.0
    total = 0.5  # prior pseudo-count for mu_hat as in WS&R
    sumsq = 0.25
    lower, upper = 0.0, 1.0
    out = np.zeros(n_steps, dtype=bool)
    excluded = False
    for i in range(n_steps):
        t = i + 1
        lam = min(np.sqrt(2.0 * log_term / (var_hat * t * np.log(t + 1.0))), lam_cap)
        x = unit[i]
        v = 4.0 * (x - mu_hat) ** 2
        psi = (-np.log(1.0 - lam) - lam) / 4.0
        sum_lam += lam
        sum_lam_x += lam * x
        sum_v_psi += v * psi
        center = sum_lam_x / sum_lam
        width = (log_term + sum_v_psi) / sum_lam
        lower = max(lower, center - width)
        upper = min(upper, center + width)
        excluded = excluded or (null < lower) or (null > upper)
        out[i] = excluded
        total += x
        sumsq += x * x
        mu_hat = total / (t + 1)
        var_hat = max(sumsq / (t + 1) - mu_hat**2, 1e-4)
    return out


def path_results(x: np.ndarray, low: float, high: float, alpha: float, gs: dict, prior_sd: float) -> dict:
    """First rejection index (1-based n) for every test on one path; 0 means never."""
    cap = len(x)
    unit = (x - low) / (high - low)
    violated = bool(np.any(unit < -1e-9) or np.any(unit > 1 + 1e-9))
    unit = np.clip(unit, 0.0, 1.0)
    null = (0.0 - low) / (high - low)
    t = t_stats_along_path(x)
    p = t_pvalues(t)
    n = np.arange(1, cap + 1)
    thr = np.log(1.0 / alpha)

    def first(mask: np.ndarray) -> int:
        idx = np.flatnonzero(mask)
        return int(idx[0] + 1) if idx.size else 0

    log_w = betting_log_wealth(unit, null)
    del n
    out = {
        "violated_bound": int(violated),
        "t_peek": first(p < alpha),
        "gs_pocock": group_sequential_first_reject(p, gs, "pocock"),
        "gs_obf": group_sequential_first_reject(p, gs, "obf"),
        "seq_t": first(sequential_t_log_evalue(t, prior_sd) >= thr),
        "seq_t_narrow": first(sequential_t_log_evalue(t, 0.3) >= thr),
        "bet_plugin": first(log_w >= thr),
        "eb_cs": first(eb_cs_excludes(unit, null, alpha)),
        # Fixed-n tests: store the per-n decision arrays so n80 can be read off directly.
        "t_fixed_reject": (p < alpha).astype(np.int8),
        "bet_fixed_reject": (log_w >= thr).astype(np.int8),
    }
    return out


def _worker(args):
    x, low, high, alpha, gs, prior_sd = args
    return path_results(x, low, high, alpha, gs, prior_sd)


SEQUENTIAL = ["t_peek", "gs_pocock", "gs_obf", "seq_t", "seq_t_narrow", "bet_plugin", "eb_cs"]
FIXED = ["t_fixed", "bet_fixed"]


def summarize(results: list[dict], cap: int, gs: dict, power: float = 0.8) -> dict:
    """n80 (smallest n with P(rejected by n) >= power), rejection rate within cap, mean stopping n."""
    paths = len(results)
    summary = {"paths": paths, "cap": cap, "violated_bound_rate": float(np.mean([r["violated_bound"] for r in results]))}
    for name in SEQUENTIAL:
        stops = np.array([r[name] for r in results])
        rejected = stops > 0
        by_n = np.array([np.mean((stops > 0) & (stops <= n)) for n in range(1, cap + 1)])
        hit = np.flatnonzero(by_n >= power)
        stop_cap = gs["nmax"] if name.startswith("gs_") else cap
        summary[name] = {
            "reject_rate": float(rejected.mean()),
            "n80": int(hit[0] + 1) if hit.size else None,
            "mean_n": float(np.where(rejected, stops, stop_cap).mean()),
        }
    for name in FIXED:
        mat = np.stack([r[name + "_reject"] for r in results])  # (paths, cap)
        rate = mat.mean(axis=0)
        hit = np.flatnonzero(rate >= power)
        summary[name] = {
            "n80": int(hit[0] + 1) if hit.size else None,
            "reject_rate_at_cap": float(rate[-1]),
            "reject_rate_curve": rate.tolist(),
        }
    return summary


def run_setting(name: str, draw_path, low: float, high: float, alpha: float, cap: int, paths: int, prior_sd: float, pool: Pool, rng: np.random.Generator, is_null: bool = False) -> dict:
    t0 = time.time()
    # Group-sequential plan: 5 looks, planned maximum = twice the one-look t-test's n80 for this
    # setting (a generous power analysis). On a null there is no n80; plan for the cap.
    n80 = None if is_null else fixed_t_n80(draw_path, rng, cap, alpha)
    nmax = cap if n80 is None else int(min(cap, max(10, 2 * n80)))
    gs = group_sequential_design(nmax, alpha, looks=5, rng=np.random.default_rng(12345))
    jobs = [(np.asarray(draw_path(rng), dtype=np.float64)[:cap], low, high, alpha, gs, prior_sd) for _ in range(paths)]
    results = pool.map(_worker, jobs, chunksize=4)
    summary = summarize(results, cap, gs)
    summary["name"] = name
    summary["bound"] = [low, high]
    summary["gs_design"] = gs
    sample = np.concatenate([j[0] for j in jobs[:50]])
    summary["data_mean"] = float(sample.mean())
    summary["data_sd"] = float(sample.std())
    summary["data_range"] = [float(sample.min()), float(sample.max())]
    if is_null:
        print(f"{name:40} sd {summary['data_sd']:.3f}  FPR " + "  ".join(f"{k}:{summary[k]['reject_rate']:.3f}" for k in SEQUENTIAL) + f"  t_fixed@10:{summary['t_fixed']['reject_rate_curve'][9]:.3f}  ({time.time() - t0:.0f}s)")
    else:
        print(f"{name:40} gap {summary['data_mean']:+.3f} sd {summary['data_sd']:.3f} viol {summary['violated_bound_rate']:.2f}  n80 " + "  ".join(f"{k}:{summary[k]['n80'] if summary[k]['n80'] is not None else '>cap'}" for k in ["t_fixed", "gs_pocock", "gs_obf", "seq_t", "bet_fixed", "bet_plugin", "eb_cs"]) + f"  E[n] seq_t {summary['seq_t']['mean_n']:.0f} gs_p {summary['gs_pocock']['mean_n']:.0f} bet {summary['bet_plugin']['mean_n']:.0f}  ({time.time() - t0:.0f}s)")
    return summary


# ----------------------------------------------------------------------------
# Data sources
# ----------------------------------------------------------------------------


def beta_gap_path(total: float, gap: float):
    a = 0.5 * (1.0 + gap) * total
    b = total - a

    def draw(rng: np.random.Generator) -> np.ndarray:
        return rng.beta(a, b, size=400) - rng.beta(b, a, size=400)

    return draw


def load_bbf(name: str) -> pd.DataFrame:
    frame = pd.read_csv(DATA / f"{name}.csv")
    frame["game"] = frame["game"].str.lower().str.replace("_", "", regex=False)
    frame["clipped"] = frame["GameScoreNormalized"].clip(0.0, 1.0)
    return frame


def bbf_aggregate_path(left: pd.DataFrame, right: pd.DataFrame):
    """One 'seed' = one independent run per game for each algorithm, aggregated as the mean
    clipped human-normalized score over the 55 games (rliable's 1 - optimality gap)."""
    games = sorted(set(left["game"]) & set(right["game"]))
    left_pools = [left.loc[left["game"] == g, "clipped"].to_numpy() for g in games]
    right_pools = [right.loc[right["game"] == g, "clipped"].to_numpy() for g in games]

    def draw(rng: np.random.Generator) -> np.ndarray:
        n = 400
        agg_l = np.zeros(n)
        agg_r = np.zeros(n)
        for pool_l, pool_r in zip(left_pools, right_pools):
            agg_l += pool_l[rng.integers(0, len(pool_l), size=n)]
            agg_r += pool_r[rng.integers(0, len(pool_r), size=n)]
        return (agg_l - agg_r) / len(games)

    return draw


def bbf_game_path(left: pd.DataFrame, right: pd.DataFrame, game: str):
    pool_l = left.loc[left["game"] == game, "clipped"].to_numpy()
    pool_r = right.loc[right["game"] == game, "clipped"].to_numpy()

    def draw(rng: np.random.Generator) -> np.ndarray:
        n = 400
        return pool_l[rng.integers(0, len(pool_l), size=n)] - pool_r[rng.integers(0, len(pool_r), size=n)]

    return draw


def bbf_null_path(frame: pd.DataFrame):
    """Same algorithm against itself with independent run draws: a true null with real seed noise."""
    return bbf_aggregate_path(frame, frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--prior-sd", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    alpha = args.alpha
    cap = 200
    paths = 120 if args.quick else 400
    null_paths = 300 if args.quick else 1000
    rng = np.random.default_rng(0)

    t0 = time.time()
    settings = []
    with Pool(args.workers) as pool:
        # --- Validity: nulls ---
        nulls = []
        for total in (8.0, 24.0, 80.0):
            nulls.append(run_setting(f"null_beta_conc{total:g}", beta_gap_path(total, 0.0), -1.0, 1.0, alpha, cap, null_paths, args.prior_sd, pool, rng, is_null=True))
        nulls.append(run_setting("null_gaussian_sd0.1", lambda r: r.normal(0, 0.1, size=400), -1.0, 1.0, alpha, cap, null_paths, args.prior_sd, pool, rng, is_null=True))
        # Heavy-tailed null: scaled t(3), bounded by clipping to the declared interval.
        nulls.append(run_setting("null_t3_sd0.1", lambda r: np.clip(0.1 * r.standard_t(3, size=400) / np.sqrt(3), -1, 1), -1.0, 1.0, alpha, cap, null_paths, args.prior_sd, pool, rng, is_null=True))
        # Skewed null: centered exponential, clipped.
        nulls.append(run_setting("null_skewed_exp_sd0.1", lambda r: np.clip(0.1 * (r.exponential(size=400) - 1.0), -1, 1), -1.0, 1.0, alpha, cap, null_paths, args.prior_sd, pool, rng, is_null=True))
        bbf = load_bbf("RR2_BBF")
        nulls.append(run_setting("null_bbf_vs_bbf_aggregate", bbf_null_path(bbf), -1.0, 1.0, alpha, cap, null_paths, args.prior_sd, pool, rng, is_null=True))
        for game in ("pong", "kangaroo", "frostbite"):
            nulls.append(run_setting(f"null_bbf_vs_bbf_{game}", bbf_game_path(bbf, bbf, game), -1.0, 1.0, alpha, cap, null_paths, args.prior_sd, pool, rng, is_null=True))
        settings += nulls

        # --- Power: beta gap 0.14 as in the paper, declared bound [-1, 1] and a tighter true bound ---
        for total in (8.0, 24.0, 80.0):
            settings.append(run_setting(f"beta_gap0.14_conc{total:g}", beta_gap_path(total, 0.14), -1.0, 1.0, alpha, cap, paths, args.prior_sd, pool, rng))
        settings.append(run_setting("beta_gap0.14_conc80_bound0.5", beta_gap_path(80.0, 0.14), -0.5, 0.5, alpha, cap, paths, args.prior_sd, pool, rng))

        # --- Power: real BBF ablations, aggregate seeds ---
        for other in ("RR2_BBF-SPR", "RR2_BBF-Resets", "RR2_BBF-WD", "RR2_BBF+sticky_100k", "RR8_BBF"):
            right = load_bbf(other)
            settings.append(run_setting(f"bbf_vs_{other}_aggregate", bbf_aggregate_path(bbf, right), -1.0, 1.0, alpha, cap, paths, args.prior_sd, pool, rng))
        # Declared-bound sweep on the aggregate comparison that has a moderate gap.
        right = load_bbf("RR2_BBF-SPR")
        for half in (0.5, 0.25, 0.1, 0.05):
            settings.append(run_setting(f"bbf_vs_RR2_BBF-SPR_aggregate_bound{half:g}", bbf_aggregate_path(bbf, right), -half, half, alpha, cap, paths, args.prior_sd, pool, rng))
        # --- Power: single games. Chosen before testing by one rule: BBF's clipped score has mean in
        # (0.05, 0.95) and sd > 0.02 (so the clipped score is informative), spanning small to large gaps.
        rr8 = load_bbf("RR8_BBF")
        for game in ("alien", "hero", "qbert", "zaxxon", "kangaroo", "frostbite", "choppercommand", "asterix"):
            settings.append(run_setting(f"bbf_vs_RR2_BBF-SPR_{game}", bbf_game_path(bbf, right, game), -1.0, 1.0, alpha, cap, paths, args.prior_sd, pool, rng))
            settings.append(run_setting(f"bbf_vs_RR8_BBF_{game}", bbf_game_path(bbf, rr8, game), -1.0, 1.0, alpha, cap, paths, args.prior_sd, pool, rng))

    (OUT / "extensions_summary.json").write_text(json.dumps({"alpha": alpha, "cap": cap, "prior_sd": args.prior_sd, "settings": settings}, indent=1))
    print(f"wrote {OUT / 'extensions_summary.json'}  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
