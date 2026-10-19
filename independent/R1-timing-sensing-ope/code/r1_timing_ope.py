"""R1 pilot: can we evaluate a policy that chooses *when to check* from logs where the clinician chose?

Estimators (all tabular FQE on the same coarse state discretisation):
    bin1h, bin4h   : resample to a fixed grid, carry-forward obs, dose per bin   (the common pipeline)
    exo            : decision points = logged checks, action = dose only          (Insulin4RL protocol)
    timing         : action = (dose, wait), raw sojourns, gamma**tau               (SMDP FQE)
    timing+mask    : + compound transitions that skip zero-dose checks, unweighted
    timing+ipw     : + the same with inverse-probability weights for the logger's 'keep waiting' decisions

Targets (deterministic rules on the glucose band):
    mode   : clinician's modal dose and wait
    sparse : modal dose, one wait class longer      (looks less often)   -> identifiable by masking
    dense  : modal dose, one wait class shorter     (looks more often)   -> not identifiable from these logs
    dosehi : one dose level higher, modal wait
    doselo : one dose level lower, modal wait

Ground truth: simulate each target. Pre-registered kill test is in README.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import ctsim as cs  # noqa: E402
from lib import ope_timing as ope  # noqa: E402
from lib.io_utils import kill_test, load_csv, save_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"          # overridden by --tag

TARGETS = {
    "mode": (0, 0), "sparse": (0, 1), "dense": (0, -1), "dosehi": (1, 0), "doselo": (-1, 0),
}
ESTIMATORS = ["bin1h", "bin4h", "exo", "timing", "timing+mask", "timing+ipw", "timing+ipw10"]


def estimate(name: str, ev: cs.Events, ds: int, ws: int) -> dict:
    if name == "bin1h":
        return ope.est_binned(ev, ds, ws, 1.0)
    if name == "bin4h":
        return ope.est_binned(ev, ds, ws, 4.0)
    if name == "exo":
        return ope.est_exogenous(ev, ds, ws)
    if name == "timing":
        return ope.est_timing_aware(ev, ds, ws, mask="none")
    if name == "timing+mask":
        return ope.est_timing_aware(ev, ds, ws, mask="naive")
    if name == "timing+ipw":
        return ope.est_timing_aware(ev, ds, ws, mask="ipw")
    if name == "timing+ipw10":
        return ope.est_timing_aware(ev, ds, ws, mask="ipw", max_w=10.0)
    raise ValueError(name)


def ground_truth(n: int, seeds: list[int]) -> dict:
    truth = {}
    for name, (ds, ws) in TARGETS.items():
        vals = [cs.simulate(cs.make_target(ds, ws), n, 10_000 + s).returns.mean() for s in seeds]
        truth[name] = (float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals))))
    for name, (ds, every) in {"grid1h_mode": (0, 1.0), "grid4h_mode": (0, 4.0), "grid1h_dosehi": (1, 1.0), "grid4h_dosehi": (1, 4.0),
                              "grid1h_doselo": (-1, 1.0), "grid4h_doselo": (-1, 4.0)}.items():
        vals = [cs.simulate(cs.make_grid_policy(ds, every), n, 10_000 + s).returns.mean() for s in seeds]
        truth[name] = (float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals))))
    vals = [cs.simulate(cs.pi_behaviour, n, 10_000 + s).returns.mean() for s in seeds]
    truth["behaviour"] = (float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals))))
    return truth


def run(args) -> None:
    RESULTS.mkdir(exist_ok=True)
    runs_path = RESULTS / "runs.csv"
    rows = load_csv(runs_path) if args.resume else []
    done = {(r["seed"], r["target"], r["estimator"]) for r in rows}
    t0 = time.time()
    truth = ground_truth(args.n_truth, list(range(args.truth_seeds)))
    save_csv(RESULTS / "truth.csv", [dict(policy=k, value=v[0], se=v[1]) for k, v in truth.items()])
    print("ground truth:", {k: round(v[0], 2) for k, v in truth.items()})
    for seed in range(args.seeds):
        ev = cs.simulate(cs.pi_behaviour, args.n_episodes, seed)
        emp = float(ev.returns.mean())
        for target, (ds, ws) in TARGETS.items():
            for est in ESTIMATORS:
                if (seed, target, est) in done:
                    continue
                out = estimate(est, ev, ds, ws)
                rows.append(dict(seed=seed, target=target, estimator=est, value=out["value"], truth=truth[target][0],
                                 bias=out["value"] - truth[target][0], unsupported=out["unsupported_lookup_frac"],
                                 n_visited_sa=out["n_visited_sa"], n_transitions=out["n_transitions"],
                                 behaviour_empirical=emp, n_episodes=args.n_episodes))
                print(f"seed {seed} {target:7s} {est:12s} V={out['value']:7.2f} truth={truth[target][0]:7.2f} "
                      f"bias={out['value'] - truth[target][0]:+6.2f} unsupported={out['unsupported_lookup_frac']:.3f}  [{time.time() - t0:5.0f}s]")
                save_csv(runs_path, rows)
    summarize(rows, truth)


def summarize(rows: list[dict] | None = None, truth: dict | None = None) -> None:
    rows = rows if rows is not None else load_csv(RESULTS / "runs.csv")
    if truth is None:
        truth = {r["policy"]: (r["value"], r["se"]) for r in load_csv(RESULTS / "truth.csv")}
    summary = []
    for target in TARGETS:
        for est in ESTIMATORS:
            b = np.array([r["bias"] for r in rows if r["target"] == target and r["estimator"] == est])
            u = np.array([r["unsupported"] for r in rows if r["target"] == target and r["estimator"] == est])
            if len(b) == 0:
                continue
            summary.append(dict(target=target, estimator=est, truth=truth[target][0], mean_bias=b.mean(), sd_bias=b.std(ddof=1) if len(b) > 1 else 0.0,
                                mean_abs_bias=np.abs(b).mean(), unsupported=u.mean(), n_seeds=len(b)))
    save_csv(RESULTS / "summary.csv", summary)

    def mab(target, est):
        return next(r["mean_abs_bias"] for r in summary if r["target"] == target and r["estimator"] == est)

    def unsup(target, est):
        return next(r["unsupported"] for r in summary if r["target"] == target and r["estimator"] == est)

    def per_seed(target, est):
        return {r["seed"]: abs(r["bias"]) for r in rows if r["target"] == target and r["estimator"] == est}

    # K1: on the sparse target, timing+ipw beats the better of bin1h / exo by 2x, in most seeds
    ref = min(mab("sparse", "bin1h"), mab("sparse", "exo"))
    k1_mean = mab("sparse", "timing+ipw") < 0.5 * ref
    ps_ipw, ps_bin, ps_exo = per_seed("sparse", "timing+ipw"), per_seed("sparse", "bin1h"), per_seed("sparse", "exo")
    wins = sum(ps_ipw[s] < 0.5 * min(ps_bin[s], ps_exo[s]) for s in ps_ipw)
    k1 = k1_mean and wins >= max(1, int(np.ceil(0.8 * len(ps_ipw))))
    # K2: the support diagnostic separates dense (not identifiable) from sparse (identifiable after masking)
    k2 = unsup("dense", "timing+ipw") >= 3.0 * max(unsup("sparse", "timing+ipw"), 1e-3)
    # K3: binned estimates sit closer to the degenerate grid policy's truth than to the target's truth
    k3_hits = []
    for target, grid_key in [("mode", "grid{d}h_mode"), ("dosehi", "grid{d}h_dosehi"), ("doselo", "grid{d}h_doselo")]:
        for d in (1, 4):
            v = np.mean([r["value"] for r in rows if r["target"] == target and r["estimator"] == f"bin{d}h"])
            k3_hits.append(abs(v - truth[grid_key.format(d=d)][0]) < abs(v - truth[target][0]))
    k3 = sum(k3_hits) >= len(k3_hits) / 2
    text = (f"K1 (timing+ipw halves the sparse-target error of the better of bin1h/exo: {mab('sparse', 'timing+ipw'):.2f} vs {ref:.2f}, "
            f"{wins}/{len(ps_ipw)} seeds) = {k1}. K2 (support flag dense/sparse = {unsup('dense', 'timing+ipw'):.3f}/{unsup('sparse', 'timing+ipw'):.3f}) = {k2}. "
            f"K3 (binned FQE closer to the fixed-grid policy's truth than to the target's, {sum(k3_hits)}/{len(k3_hits)}) = {k3}. "
            f"Naive masking bias on sparse = {next(r['mean_bias'] for r in summary if r['target'] == 'sparse' and r['estimator'] == 'timing+mask'):+.2f}, "
            f"IPW masking = {next(r['mean_bias'] for r in summary if r['target'] == 'sparse' and r['estimator'] == 'timing+ipw'):+.2f}.")
    verdict = True if (k1 and k2) else (False if not k1 and not k2 else None)
    line = kill_test(verdict, text)
    (RESULTS / "verdict.txt").write_text(line + "\n")
    print("\ntarget   estimator     truth   mean_bias  sd   |bias|  unsupported")
    for r in summary:
        print(f"{r['target']:7s} {r['estimator']:12s} {r['truth']:7.2f} {r['mean_bias']:+8.2f} {r['sd_bias']:5.2f} {r['mean_abs_bias']:6.2f}  {r['unsupported']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-episodes", type=int, default=3000)
    ap.add_argument("--n-truth", type=int, default=5000)
    ap.add_argument("--truth-seeds", type=int, default=3)
    ap.add_argument("--eps-wait", type=float, default=cs.EPS_WAIT, help="logger's wait randomisation; 0 = strict protocol")
    ap.add_argument("--tag", type=str, default="", help="write results to results/<tag>/")
    args = ap.parse_args()
    global RESULTS
    if args.tag:
        RESULTS = ROOT / "results" / args.tag
    cs.EPS_WAIT = args.eps_wait
    if args.smoke:
        args.seeds, args.n_episodes, args.n_truth, args.truth_seeds = 1, 400, 1000, 1
    if args.full:
        args.seeds, args.n_episodes, args.n_truth, args.truth_seeds = 20, 10000, 20000, 5
    if args.summarize:
        summarize()
        return
    run(args)


if __name__ == "__main__":
    main()
