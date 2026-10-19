"""R3 pilot: does every preprocessing step (not just time binning) make an offline-RL pipeline
over-rate the policy it just learned, when it evaluates on the same preprocessed data?

For each phi = (time, state, action) and each dataset seed:
    1. learn a greedy policy on phi(logs)                      (model-based Q-iteration, count >= 5)
    2. FQE on phi(held-out logs) of that policy                 -> V_phi   (what the pipeline reports)
    3. deploy the policy in the simulator                       -> V_true
    4. timing-aware FQE on the *raw* held-out logs of the policy -> V_raw  (the proposed protocol)
optimism = V_phi - V_true ; raw_err = V_raw - V_true. Pre-registered kill test in README.md.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import ctsim as cs  # noqa: E402
from lib import pipelines as pp  # noqa: E402
from lib.io_utils import kill_test, load_csv, save_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TIMES = ["raw", "bin1h", "bin4h"]
STATES = ["fine", "coarse", "binary"]
ACTIONS = ["fine", "coarse"]


def run(args) -> None:
    RESULTS.mkdir(exist_ok=True)
    runs_path = RESULTS / "runs.csv"
    rows = load_csv(runs_path) if args.resume else []
    done = {(r["seed"], r["time"], r["state"], r["action"]) for r in rows}
    t0 = time.time()
    for seed in range(args.seeds):
        ev = cs.simulate(cs.pi_behaviour, args.n_episodes, seed)                 # policy learning
        ev_test = cs.simulate(cs.pi_behaviour, args.n_episodes, 500 + seed)      # held-out logs for every evaluation
        beh_emp = float(ev_test.returns.mean())
        for tm, sm, am in itertools.product(TIMES, STATES, ACTIONS):
            if (seed, tm, sm, am) in done:
                continue
            tr, n_s, n_a = pp.phi_transitions(ev, tm, sm, am)
            greedy, _ = pp.q_iteration(tr, n_s, n_a)
            tr_test, _, _ = pp.phi_transitions(ev_test, tm, sm, am)
            est = pp.fqe_in_phi(tr_test, n_s, n_a, greedy)
            pol = pp.deploy_policy(greedy, tm, sm, am)
            dep = cs.simulate(pol, args.n_deploy, 50_000 + seed)
            v_true = float(dep.returns.mean())
            v_true_se = float(dep.returns.std() / np.sqrt(args.n_deploy))
            raw = pp.raw_fqe_of(ev_test, pp.raw_target_table(greedy, tm, sm, am))
            rows.append(dict(seed=seed, time=tm, state=sm, action=am, v_phi=est["value"], v_true=v_true, v_true_se=v_true_se,
                             v_raw=raw["value"], optimism=est["value"] - v_true, raw_err=raw["value"] - v_true,
                             unsupported_phi=est["unsupported_lookup_frac"], unsupported_raw=raw["unsupported_lookup_frac"],
                             checks_per_episode=len(dep.t) / args.n_deploy, behaviour_empirical=beh_emp, n_episodes=args.n_episodes))
            print(f"seed {seed} {tm:5s} {sm:6s} {am:6s}  V_phi={est['value']:7.2f}  V_true={v_true:7.2f}  V_raw={raw['value']:7.2f}  "
                  f"optimism={est['value'] - v_true:+6.2f}  raw_err={raw['value'] - v_true:+6.2f}  [{time.time() - t0:5.0f}s]")
            save_csv(runs_path, rows)
    summarize(rows)


def summarize(rows: list[dict] | None = None) -> None:
    rows = rows if rows is not None else load_csv(RESULTS / "runs.csv")
    summary = []
    for tm, sm, am in itertools.product(TIMES, STATES, ACTIONS):
        sel = [r for r in rows if r["time"] == tm and r["state"] == sm and r["action"] == am]
        if not sel:
            continue
        o = np.array([r["optimism"] for r in sel]); e = np.array([r["raw_err"] for r in sel]); vt = np.array([r["v_true"] for r in sel])
        summary.append(dict(time=tm, state=sm, action=am, n_seeds=len(sel), v_true=vt.mean(), mean_optimism=o.mean(),
                            sd_optimism=o.std(ddof=1) if len(o) > 1 else 0.0, mean_abs_optimism=np.abs(o).mean(),
                            mean_raw_err=e.mean(), mean_abs_raw_err=np.abs(e).mean(),
                            raw_better_frac=float(np.mean(np.abs(e) < np.abs(o)))))
    save_csv(RESULTS / "summary.csv", summary)

    def get(tm, sm, am, key):
        return next(r[key] for r in summary if r["time"] == tm and r["state"] == sm and r["action"] == am)

    # K1 (sanity): time binning inflates. mean optimism of binned pipelines minus raw pipelines, at fine state/action
    binned = np.mean([get(t, "fine", "fine", "mean_optimism") for t in ("bin1h", "bin4h")])
    k1 = binned - get("raw", "fine", "fine", "mean_optimism") > 2.0
    # K2: within raw-time pipelines, coarser state or action produces more |optimism| than fine/fine by > 2 seed-SDs
    base = get("raw", "fine", "fine", "mean_abs_optimism")
    sd = max(get("raw", "fine", "fine", "sd_optimism"), 0.25)
    k2_hits = {f"{sm}/{am}": get("raw", sm, am, "mean_abs_optimism") - base > 2 * sd
               for sm, am in itertools.product(STATES, ACTIONS) if (sm, am) != ("fine", "fine")}
    k2 = sum(k2_hits.values()) >= 2
    # K3: raw-log evaluation beats in-phi evaluation for most binned pipelines
    fr = [get(t, sm, am, "raw_better_frac") for t in ("bin1h", "bin4h") for sm, am in itertools.product(STATES, ACTIONS)]
    k3 = np.mean(fr) > 0.5
    text = (f"K1 (binning inflates: binned - raw optimism = {binned - get('raw', 'fine', 'fine', 'mean_optimism'):+.2f} > 2) = {k1}. "
            f"K2 (coarser state/action adds optimism beyond seed noise, hits: {sum(k2_hits.values())}/{len(k2_hits)} "
            + ", ".join(k for k, v in k2_hits.items() if v) + f") = {k2}. "
            f"K3 (raw-log FQE closer to truth than in-phi FQE for binned pipelines in {np.mean(fr):.0%} of runs) = {k3}.")
    verdict = True if (k2 and k3) else (False if (not k2 and not k3) else None)
    line = kill_test(verdict, text)
    (RESULTS / "verdict.txt").write_text(line + "\n")
    print("\ntime   state   action  V_true  optimism(sd)   |opt|   raw_err  |raw_err|  raw_better")
    for r in summary:
        print(f"{r['time']:5s}  {r['state']:6s}  {r['action']:6s}  {r['v_true']:6.2f}  {r['mean_optimism']:+6.2f} ({r['sd_optimism']:4.2f})  "
              f"{r['mean_abs_optimism']:5.2f}  {r['mean_raw_err']:+6.2f}  {r['mean_abs_raw_err']:6.2f}   {r['raw_better_frac']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-episodes", type=int, default=3000)
    ap.add_argument("--n-deploy", type=int, default=3000)
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.n_episodes, args.n_deploy = 1, 1500, 1000
    if args.full:
        args.seeds, args.n_episodes, args.n_deploy = 20, 10000, 10000
    if args.summarize:
        summarize()
        return
    run(args)


if __name__ == "__main__":
    main()
