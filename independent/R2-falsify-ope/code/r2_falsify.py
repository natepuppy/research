"""R2 pilot: do falsification tests catch wrong OPE estimates that estimator agreement misses?

Grid: failure mode × MDP seed × target × dataset size. For every cell we compute five OPE estimates,
their exact errors (the MDP is known), and seven log-only test statistics. Pre-registered kill test in README.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import confmdp as cm  # noqa: E402
from lib.io_utils import kill_test, load_csv, save_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FAILURES = ["none", "confounding", "miscal", "abstraction", "support"]
TARGETS = ["softgreedy", "uniform", "shifted"]
WRONG_TAU = 0.5          # per-row absolute error (returns are on a 0..15 scale) above which an estimate counts as wrong
BIAS_TAU = 0.5           # replicate-mean absolute error (returns are on a 0..10 scale) above which an estimator is called biased
KILL_ESTIMATORS = ["FQE", "WIS", "DR"]   # the kill test scores the estimators people actually use for long horizons; IS/PDIS are logged only


def target_policies(m: cm.MDP, rng: np.random.Generator) -> dict:
    # a soft-greedy policy w.r.t. the u-marginal model
    Pm = (1 - m.pu) * m.P[0] + m.pu * m.P[1]
    Rm = (1 - m.pu) * m.R[0] + m.pu * m.R[1]
    V = np.zeros(cm.S)
    for _ in range(cm.H):
        Q = Rm + Pm @ V
        V = Q.max(1)
    soft = np.exp(2.0 * (Q - Q.max(1, keepdims=True)))
    soft /= soft.sum(1, keepdims=True)
    beh_marg = (1 - m.pu) * m.beh[0] + m.pu * m.beh[1]
    shifted = 0.6 * beh_marg + 0.4 * np.eye(cm.A)[0][None, :]
    return dict(softgreedy=soft, uniform=np.full((cm.S, cm.A), 1.0 / cm.A), shifted=shifted)


def nc_policy(pi: np.ndarray) -> np.ndarray:
    pi_nc = pi.copy()
    pi_nc[cm.INERT] = np.eye(cm.A)[2][None, :]          # in inert states do action 2 always; nothing can change
    return pi_nc


def run_cell(failure: str, mdp_seed: int, n: int, rep: int, rng: np.random.Generator) -> list[dict]:
    m = cm.make_mdp(mdp_seed, kappa=1.5, support_gap=(failure == "support"))
    data = cm.simulate(m, n, 1000 * mdp_seed + n + 97 * rep)
    mode = failure
    x, n_x = cm.observed_state(data, mode)
    beh_hat = cm.estimate_behaviour(x, data["a"], n_x)
    if failure == "miscal":
        beh_hat = 0.5 * beh_hat + 0.5 / cm.A
    rows = []
    pols = target_policies(m, rng)
    for tname, pi in pols.items():
        pi_obs = cm.lift_target(pi, mode)
        pi_true = cm.unlift_target(pi_obs, mode)
        truth = cm.true_value(m, pi_true)
        pi_nc_obs = cm.lift_target(nc_policy(pi), mode)
        assert abs(cm.true_value(m, cm.unlift_target(pi_nc_obs, mode)) - truth) < 1e-9
        ests = cm.all_estimates(x, data["a"], data["r"], n_x, pi_obs, beh_hat, x[:, 0])
        tests = cm.falsification_tests(x, data["a"], data["r"], n_x, pi_obs, pi_nc_obs, beh_hat, data["nco"], rng, ests)
        for e in cm.ESTIMATORS:
            rows.append(dict(failure=failure, mdp_seed=mdp_seed, n=n, rep=rep, target=tname, estimator=e, value=ests[e], truth=truth,
                             error=ests[e] - truth, abs_error=abs(ests[e] - truth), wrong=abs(ests[e] - truth) > WRONG_TAU,
                             behaviour_truth=cm.true_value_u(m, m.beh), empirical=float(data["r"].sum(1).mean()),
                             **{k: v for k, v in tests.items() if not k.startswith("ncp_")}, ncp_own=tests[f"ncp_{e}"]))
    return rows


def auroc(stat: np.ndarray, label: np.ndarray) -> float:
    pos, neg = stat[label], stat[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def summarize(rows: list[dict] | None = None) -> None:
    rows = rows if rows is not None else load_csv(RESULTS / "runs.csv")
    rows = [r for r in rows if r["estimator"] in KILL_ESTIMATORS]
    tests = cm.TESTS + ["ncp_own"]
    f = np.array([r["failure"] for r in rows])
    abs_err = np.array([r["abs_error"] for r in rows])
    # 'biased' = the estimator's error averaged over data replicates is large and not explained by its spread
    groups: dict[tuple, list[float]] = {}
    for r in rows:
        groups.setdefault((r["failure"], r["mdp_seed"], r["n"], r["target"], r["estimator"]), []).append(r["error"])
    def is_biased(key):
        e = np.array(groups[key])
        se = e.std(ddof=1) / np.sqrt(len(e)) if len(e) > 1 else 0.0
        return abs(e.mean()) > BIAS_TAU and abs(e.mean()) > 2 * se
    wrong = np.array([is_biased((r["failure"], r["mdp_seed"], r["n"], r["target"], r["estimator"])) for r in rows])
    stats = {t: np.array([float(r[t]) for r in rows]) for t in tests}
    # robust z-scores against the no-failure configurations, then thresholds at 10% false alarm
    def rz(t):
        ref = stats[t][f == "none"]
        med = np.median(ref); mad = np.median(np.abs(ref - med)) * 1.4826 + 1e-9
        return (stats[t] - med) / mad
    battery = np.max(np.stack([rz(t) for t in ("self", "nco_is", "nco_model", "ncp", "placebo")]), axis=0)
    standard = np.max(np.stack([rz("agree"), rz("ess")]), axis=0)
    stats["battery"], stats["standard"] = battery, standard
    thr = {t: np.quantile(stats[t][f == "none"], 0.9) for t in tests}
    thr["battery"] = np.quantile(battery[f == "none"], 0.9); thr["standard"] = np.quantile(standard[f == "none"], 0.9)
    summary = []
    for fail in FAILURES:
        sel = f == fail
        row = dict(failure=fail, n_rows=int(sel.sum()), frac_wrong=float(wrong[sel].mean()), median_abs_error=float(np.median(abs_err[sel])))
        for t in tests + ["battery", "standard"]:
            row[f"power_{t}"] = float(np.mean(stats[t][sel] > thr[t]))
            row[f"auroc_{t}"] = auroc(stats[t][sel], wrong[sel]) if fail != "none" else float("nan")
        summary.append(row)
    pooled = f != "none"
    row = dict(failure="pooled_failures", n_rows=int(pooled.sum()), frac_wrong=float(wrong[pooled].mean()), median_abs_error=float(np.median(abs_err[pooled])))
    for t in tests + ["battery", "standard"]:
        row[f"power_{t}"] = float(np.mean(stats[t][pooled] > thr[t]))
        row[f"auroc_{t}"] = auroc(stats[t][pooled], wrong[pooled])
    summary.append(row)
    save_csv(RESULTS / "summary.csv", summary)
    # also: AUROC for wrong-vs-right over *all* rows including none (detecting wrong estimates in general)
    all_auroc = {t: auroc(stats[t], wrong) for t in tests + ["battery", "standard"]}
    save_csv(RESULTS / "auroc_all.csv", [dict(test=t, auroc_wrong_vs_right_all=v) for t, v in all_auroc.items()])

    # kill test
    k1 = all_auroc["battery"] >= all_auroc["standard"] + 0.10
    k2_detail = {}
    for fail in ("confounding", "miscal", "abstraction", "support"):
        r = next(s for s in summary if s["failure"] == fail)
        best = max(("self", "nco_is", "nco_model", "ncp", "placebo"), key=lambda t: r[f"power_{t}"])
        k2_detail[fail] = (best, r[f"power_{best}"])
    k2 = all(v[1] >= 0.6 for v in k2_detail.values())
    conf = next(s for s in summary if s["failure"] == "confounding")
    k3 = conf["power_agree"] <= 0.2
    text = (f"K1 (battery AUROC wrong-vs-right {all_auroc['battery']:.2f} >= standard checks {all_auroc['standard']:.2f} + 0.10) = {k1}. "
            f"K2 (best test power at 10% false alarm: " + ", ".join(f"{k}: {v[0]} {v[1]:.2f}" for k, v in k2_detail.items()) + f") = {k2}. "
            f"K3 (agreement power against confounding {conf['power_agree']:.2f} <= 0.2) = {k3}.")
    verdict = True if (k1 and k2) else (False if (not k1 and not k2) else None)
    line = kill_test(verdict, text)
    (RESULTS / "verdict.txt").write_text(line + "\n")
    print("\nfailure        wrong  | power at 10% FA:  " + "  ".join(f"{t:>9s}" for t in tests + ["battery", "standard"]))
    for s in summary:
        print(f"{s['failure']:14s} {s['frac_wrong']:.2f}   | " + "  ".join(f"{s[f'power_{t}']:9.2f}" for t in tests + ["battery", "standard"]))
    print("\nAUROC (wrong vs right, all rows): " + "  ".join(f"{t}={v:.2f}" for t, v in all_auroc.items()))


def run(args) -> None:
    RESULTS.mkdir(exist_ok=True)
    runs_path = RESULTS / "runs.csv"
    rows = load_csv(runs_path) if args.resume else []
    done = {(r["failure"], r["mdp_seed"], r["n"], r["rep"]) for r in rows}
    rng = np.random.default_rng(0)
    t0 = time.time()
    for mdp_seed in range(args.mdp_seeds):
        for n in args.sizes:
            for failure in FAILURES:
                for rep in range(args.reps):
                    if (failure, mdp_seed, n, rep) in done:
                        continue
                    new = run_cell(failure, mdp_seed, n, rep, rng)
                    rows.extend(new)
                    if rep:
                        continue
                    fq = [r for r in new if r["estimator"] == "FQE"]
                    print(f"mdp {mdp_seed} n={n:5d} {failure:12s} FQE err " + " ".join(f"{r['error']:+5.2f}" for r in fq)
                          + f"  agree={fq[0]['agree']:.2f} self={fq[0]['self']:.1f} nco_is={fq[0]['nco_is']:.1f} nco_model={fq[0]['nco_model']:.1f} ncp={fq[0]['ncp']:.1f}  [{time.time() - t0:4.0f}s]")
                save_csv(runs_path, rows)
    summarize(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--mdp-seeds", type=int, default=20)
    ap.add_argument("--sizes", type=int, nargs="+", default=[500, 2000])
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    if args.smoke:
        args.mdp_seeds, args.sizes, args.reps = 2, [500], 2
    if args.full:
        args.mdp_seeds, args.sizes, args.reps = 40, [500, 2000, 8000], 5
    if args.summarize:
        summarize()
        return
    run(args)


if __name__ == "__main__":
    main()
