"""R10: How far behind is the critic?

Sweep actor and critic learning rates in a batched DDPG on Pendulum. At every evaluation, compare the
critic's Q(s, pi_t(s)) against exact Monte-Carlo Q^{pi_{t-k}}(s, pi_t(s)) for k = 0..max_lag past
policies, and record the best-matching lag k*. Pre-registered kill test in README.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import ac_batched as A  # noqa: E402
from lib.io_utils import kill_test, save_csv, load_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LR_ACTOR = [1e-4, 3e-4, 1e-3]
LR_CRITIC = [1e-4, 3e-4, 1e-3, 3e-3]


def run(args):
    torch.set_num_threads(args.threads)
    cfg = A.Config(total_steps=args.T, eval_every=args.eval_every, max_lag=args.max_lag)
    la, lc, seeds = [], [], []
    for i, a in enumerate(LR_ACTOR):
        for j, c in enumerate(LR_CRITIC):
            for s in range(args.seeds):
                la.append(a); lc.append(c); seeds.append(1000 * i + 100 * j + s)
    print(f"{len(seeds)} runs x {args.T} steps", flush=True)
    P = A.BatchedDDPG(cfg, la, lc, seeds)
    t0 = time.time()
    for step in range(1, args.T + 1):
        P.step()
        if step % max(cfg.eval_every, 1000) == 0:
            last = [r for r in P.log if r["step"] == step]
            by = {}
            for r in last:
                by.setdefault((r["lr_actor"], r["lr_critic"]), []).append(r)
            print(f"step {step} [{time.time() - t0:.0f}s]  " + "  ".join(
                f"a{a:g}/c{c:g}: ret={np.mean([r['ret'] for r in v]):.0f} err0={np.mean([r['err0'] for r in v]):.1f} k*={np.mean([r['kstar'] for r in v]):.2f}"
                for (a, c), v in sorted(by.items())), flush=True)
    rows = [dict(r, seed=seeds[r["run"]]) for r in P.log]
    save_csv(RESULTS / "runs.csv", rows)
    (summarize_profile if args.tag else summarize)(rows)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(rows=None):
    rows = rows if rows is not None else load_csv(RESULTS / "runs.csv")
    steps = sorted({r["step"] for r in rows})
    T = max(steps)
    # analysis window: evals with a full lag history available and after the first third of training
    win = [r for r in rows if r["n_lags"] >= 5 and r["step"] >= T / 3]
    cells = sorted({(r["lr_actor"], r["lr_critic"]) for r in rows})
    summary = []
    for (a, c) in cells:
        sel = [r for r in win if r["lr_actor"] == a and r["lr_critic"] == c]
        fin = [r for r in rows if r["lr_actor"] == a and r["lr_critic"] == c and r["step"] >= T - 2 * (steps[1] - steps[0])]
        summary.append(dict(lr_actor=a, lr_critic=c, ratio=c / a,
                            mean_kstar=float(np.mean([r["kstar"] for r in sel])),
                            frac_lag_pos=float(np.mean([r["kstar"] > 0 for r in sel])),
                            err0=float(np.mean([r["err0"] for r in sel])), err_kstar=float(np.mean([r["err_kstar"] for r in sel])),
                            rel_improvement=float(1 - np.mean([r["err_kstar"] for r in sel]) / np.mean([r["err0"] for r in sel])),
                            bias0=float(np.mean([r["bias0"] for r in sel])), drift=float(np.mean([r["drift"] for r in sel])),
                            qscale=float(np.mean([r["qscale"] for r in sel])),
                            final_ret=float(np.mean([r["ret"] for r in fin])), n=len(sel)))
    save_csv(RESULTS / "summary.csv", summary)
    # K1: a lag exists when the critic is not faster than the actor
    slow = [s for s in summary if s["ratio"] <= 1.0]
    frac_pos = float(np.mean([s["frac_lag_pos"] for s in slow]))
    rel_imp = float(np.mean([s["rel_improvement"] for s in slow]))
    k1 = frac_pos >= 0.6 and rel_imp >= 0.2
    # K2: lag decreases with the critic/actor learning-rate ratio
    rho_ratio = spearman([np.log(s["ratio"]) for s in summary], [s["mean_kstar"] for s in summary])
    k2 = rho_ratio <= -0.6
    # K3: lag (and current-policy error) predict final return across LR cells
    rho_perf = spearman([s["mean_kstar"] for s in summary], [s["final_ret"] for s in summary])
    rho_err = spearman([s["err0"] for s in summary], [s["final_ret"] for s in summary])
    k3 = rho_perf <= -0.5 or rho_err <= -0.5
    verdict = True if (k1 and k2) else (None if k1 else False)
    text = (f"K1 (ratio<=1 cells: fraction of (run,eval) with k*>0 = {frac_pos:.2f} >= 0.6 and error reduction from best past policy = {rel_imp:.2f} >= 0.2) = {k1}. "
            f"K2 (Spearman of mean k* on log lr_critic/lr_actor = {rho_ratio:+.2f} <= -0.6) = {k2}. "
            f"K3 (Spearman of final return on k* = {rho_perf:+.2f} or on err0 = {rho_err:+.2f}, <= -0.5) = {k3}.")
    line = kill_test(verdict, text)
    (RESULTS / "verdict.txt").write_text(line + "\n")
    print("\nlr_a    lr_c    ratio   k*    P(k*>0)  err0   err_k*  rel.imp  bias0   drift  final ret")
    for s in summary:
        print(f"{s['lr_actor']:<7g} {s['lr_critic']:<7g} {s['ratio']:<6.2f} {s['mean_kstar']:5.2f}  {s['frac_lag_pos']:5.2f}   {s['err0']:6.1f} {s['err_kstar']:6.1f}  {s['rel_improvement']:5.2f}  {s['bias0']:+6.1f}  {s['drift']:5.1f}  {s['final_ret']:7.1f}")


def summarize_profile(rows=None):
    """Fine-resolution follow-up (pre-registered after the pilot, see README): test the *mean* error profile over
    lag k, which is immune to the argmin selection bias that inflates per-eval k*."""
    rows = rows if rows is not None else load_csv(RESULTS / "runs.csv")
    T = max(r["step"] for r in rows)
    K = int(max(r["n_lags"] for r in rows))
    win = [r for r in rows if r["n_lags"] >= K and r["step"] >= T / 3]
    dt = sorted({r["step"] for r in rows})[1] - sorted({r["step"] for r in rows})[0]
    cells = sorted({(r["lr_actor"], r["lr_critic"]) for r in rows})
    summary = []
    for a, c in cells:
        sel = [r for r in win if r["lr_actor"] == a and r["lr_critic"] == c]
        prof = [float(np.mean([r[f"err_lag{k}"] for r in sel])) for k in range(K)]
        kmin = int(np.argmin(prof))
        # bootstrap over runs for the argmin
        runs = sorted({r["run"] for r in sel}); rng = np.random.default_rng(0); boots = []
        for _ in range(200):
            pick = rng.integers(0, len(runs), len(runs)); sub = [r for r in sel if r["run"] in {runs[i] for i in pick}]
            boots.append(int(np.argmin([np.mean([r[f"err_lag{k}"] for r in sub]) for k in range(K)])))
        summary.append(dict(lr_actor=a, lr_critic=c, ratio=c / a, kmin=kmin, lag_steps=kmin * dt, frac_boot_kmin_pos=float(np.mean(np.array(boots) > 0)),
                            err0=prof[0], err_kmin=prof[kmin], rel_improvement=1 - prof[kmin] / prof[0], drift=float(np.mean([r["drift"] for r in sel])),
                            **{f"prof{k}": prof[k] for k in range(K)}))
    save_csv(RESULTS / "summary.csv", summary)
    slow = [s for s in summary if s["ratio"] <= 1.0]
    f1 = all(s["kmin"] > 0 and s["frac_boot_kmin_pos"] >= 0.8 for s in slow)
    rho = spearman([np.log(s["ratio"]) for s in summary], [s["kmin"] for s in summary])
    f2 = rho <= -0.6
    verdict = True if (f1 and f2) else (None if f1 else False)
    text = (f"FINE RESOLUTION ({dt}-step evals, {K - 1} lags). F1 (every cell with ratio <= 1 has the mean error profile minimised at k > 0 with bootstrap support >= 0.8; "
            f"kmin = {[s['kmin'] for s in slow]}, support = {[round(s['frac_boot_kmin_pos'], 2) for s in slow]}) = {f1}. "
            f"F2 (Spearman of kmin on log ratio = {rho:+.2f} <= -0.6) = {f2}.")
    line = kill_test(verdict, text)
    (RESULTS / "verdict.txt").write_text(line + "\n")
    print("\nlr_a    lr_c    ratio  kmin  lag(steps)  boot P(kmin>0)  err0   err_kmin  rel.imp  drift   profile k=0..")
    for s in summary:
        print(f"{s['lr_actor']:<7g} {s['lr_critic']:<7g} {s['ratio']:<6.2f} {s['kmin']:3d}   {s['lag_steps']:6.0f}      {s['frac_boot_kmin_pos']:4.2f}        {s['err0']:6.1f} {s['err_kmin']:6.1f}   {s['rel_improvement']:5.2f}  {s['drift']:5.1f}   " + " ".join(f"{s[f'prof{k}']:.1f}" for k in range(K)))


def main():
    global RESULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="", help="write to results/<tag>/ and use the mean-profile summary")
    ap.add_argument("--max-lag", type=int, default=6)
    ap.add_argument("--T", type=int, default=20_000)
    ap.add_argument("--eval-every", type=int, default=1_000)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.tag:
        RESULTS = RESULTS / args.tag
    if args.summarize:
        (summarize_profile if args.tag else summarize)(); return
    if args.smoke:
        args.T, args.eval_every, args.seeds = 3000, 500, 1
    RESULTS.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
