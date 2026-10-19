"""R8 pilot: after reward centering, is the remaining cost of gamma -> 1 set by the chain's mixing time?

Claim. Centering removes the r_bar/(1-gamma) offset. What is left is the TD operator restricted to
the complement of the constant direction, whose slowest mode has contraction 1 - gamma*lambda_2.
So the time to learn the centered values should scale like 1/(1 - gamma*lambda_2): flat in gamma on
fast-mixing chains, growing like 1/(1-gamma) on slow-mixing chains until it saturates at the mixing time.
Value scaling (Pop-Art style) fixes magnitude, not conditioning, so it should not remove that growth.

Design. Ergodic chains with controlled lambda_2 (P = alpha*I + (1-alpha)*P_rand, lambda_2 ~ alpha).
Tabular TD(0) along one Markov trajectory with
    plain     : standard discounted TD
    centered  : TD with a running average-reward estimate subtracted from every reward (Naik et al. 2024 style)
    centered+scaled : centered, with targets normalised by a running std of the TD targets (Pop-Art style)
For each (alpha, gamma, method) the step size is swept and the best is kept. Error = RMS error of the
centered value estimate relative to the std of the true centered value. Convergence time = first step after
which the error stays below 0.2 for the rest of the run. Pre-registered kill test in README.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io_utils import kill_test, load_csv, save_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
N_STATES = 40
ALPHAS = [0.0, 0.5, 0.9, 0.97, 0.99]
GAMMAS = [0.9, 0.99, 0.999, 0.9999]
METHODS = ["plain", "centered", "centered+scaled"]
STEP_SIZES = [0.4, 0.2, 0.1, 0.05, 0.02, 0.01]
THRESH = 0.2


def make_chain(alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    P_rand = rng.dirichlet(np.ones(N_STATES) * 0.5, size=N_STATES)
    P = alpha * np.eye(N_STATES) + (1 - alpha) * P_rand
    r = rng.uniform(0.0, 1.0, N_STATES) + 1.0                       # offset so plain TD has something to pay for
    ev, evec = np.linalg.eig(P.T)
    d = np.real(evec[:, np.argmax(np.real(ev))]); d = d / d.sum()
    lam2 = float(np.sort(np.abs(ev))[-2])
    return P, r, d, lam2


def true_values(P, r, d, gamma):
    rbar = float(d @ r)
    v = np.linalg.solve(np.eye(N_STATES) - gamma * P, r)
    v_c = np.linalg.solve(np.eye(N_STATES) - gamma * P, r - rbar)   # centered values: finite as gamma -> 1
    return v, v_c, rbar


def run_grid(args):
    RESULTS.mkdir(exist_ok=True)
    rows = []
    t0 = time.time()
    for seed in range(args.seeds):
        for alpha in ALPHAS:
            P, r, d, lam2 = make_chain(alpha, seed)
            cum = np.cumsum(P, axis=1)
            for gamma in GAMMAS:
                _, v_c, rbar = true_values(P, r, d, gamma)
                scale = float(np.sqrt(d @ (v_c - d @ v_c) ** 2)) + 1e-9
                # configs: methods x step sizes, all walking the SAME trajectory (common random numbers)
                cfgs = [(m, a) for m in METHODS for a in STEP_SIZES]
                C = len(cfgs)
                V = np.zeros((C, N_STATES)); rhat = np.zeros(C); sig = np.ones(C); mu_t = np.zeros(C)
                eta = np.array([a * 0.1 for _, a in cfgs])      # centering step (Naik: eta * alpha)
                alph = np.array([a for _, a in cfgs])
                is_c = np.array([m != "plain" for m, _ in cfgs]); is_s = np.array([m == "centered+scaled" for m, _ in cfgs])
                rng = np.random.default_rng(1000 * seed + int(alpha * 1000))
                s = int(rng.integers(N_STATES))
                err = np.full((C, args.steps // args.check_every + 1), np.nan)
                ci = np.arange(C)
                for t in range(1, args.steps + 1):
                    u = rng.uniform()
                    s2 = int(np.searchsorted(cum[s], u))
                    rew = r[s]
                    delta = rew - rhat * is_c + gamma * V[:, s2] - V[:, s]
                    # Pop-Art-style scaling: normalise the TD error by a running std of the targets
                    target = rew - rhat * is_c + gamma * V[:, s2]
                    mu_t = np.where(is_s, mu_t + 0.01 * (target - mu_t), mu_t)
                    sig = np.where(is_s, np.sqrt(np.maximum(sig ** 2 + 0.01 * ((target - mu_t) ** 2 - sig ** 2), 1e-4)), 1.0)
                    V[ci, s] += alph * delta / np.where(is_s, sig, 1.0)
                    rhat += eta * delta * is_c
                    s = s2
                    if t % args.check_every == 0:
                        Vc = V - (d @ V.T)[:, None]            # compare centered estimates (remove any constant)
                        err[:, t // args.check_every] = np.sqrt(((Vc - (v_c - d @ v_c)) ** 2 @ d)) / scale
                # convergence time per config, then best step size per method
                for m in METHODS:
                    best = None
                    for a in STEP_SIZES:
                        c = cfgs.index((m, a))
                        e = err[c, 1:]
                        ok = e < THRESH
                        # first checkpoint after which it stays below threshold
                        tconv = np.inf
                        for k in range(len(ok)):
                            if ok[k:].all():
                                tconv = (k + 1) * args.check_every; break
                        final = float(np.nanmean(e[-5:]))
                        if best is None or (tconv, final) < (best[1], best[2]):
                            best = (a, tconv, final)
                    rows.append(dict(seed=seed, alpha=alpha, lambda2=lam2, gamma=gamma, method=m, step_size=best[0],
                                     t_conv=best[1] if np.isfinite(best[1]) else args.steps * 2, converged=bool(np.isfinite(best[1])),
                                     final_err=best[2], pred=1.0 / (1.0 - gamma * lam2), horizon=1.0 / (1.0 - gamma), rbar=rbar))
                print(f"seed {seed} alpha={alpha:.2f} (l2={lam2:.3f}) gamma={gamma}: " + "  ".join(
                    f"{m}: t={rows[-3 + i]['t_conv']:.0f} (a={rows[-3 + i]['step_size']}, err={rows[-3 + i]['final_err']:.2f})" for i, m in enumerate(METHODS))
                    + f"  [{time.time() - t0:.0f}s]")
                save_csv(RESULTS / "runs.csv", rows)
    summarize(rows)


def summarize(rows=None):
    rows = rows if rows is not None else load_csv(RESULTS / "runs.csv")
    summary = []
    for alpha in ALPHAS:
        for gamma in GAMMAS:
            for m in METHODS:
                sel = [r for r in rows if r["alpha"] == alpha and r["gamma"] == gamma and r["method"] == m]
                if not sel:
                    continue
                t = np.array([r["t_conv"] for r in sel], float)
                summary.append(dict(alpha=alpha, gamma=gamma, method=m, lambda2=float(np.mean([r["lambda2"] for r in sel])),
                                    t_conv_gmean=float(np.exp(np.mean(np.log(t)))), frac_converged=float(np.mean([r["converged"] for r in sel])),
                                    final_err=float(np.mean([r["final_err"] for r in sel])), pred=float(np.mean([r["pred"] for r in sel]))))
    save_csv(RESULTS / "summary.csv", summary)

    def tc(alpha, gamma, m):
        return next(s["t_conv_gmean"] for s in summary if s["alpha"] == alpha and s["gamma"] == gamma and s["method"] == m)

    # K1: centered: gamma 0.9999 vs 0.99 ratio < 1.5 on the fast chain and > 3 on the slowest chain
    r_fast = tc(0.0, 0.9999, "centered") / tc(0.0, 0.99, "centered")
    r_slow = tc(0.99, 0.9999, "centered") / tc(0.99, 0.99, "centered")
    k1 = r_fast < 1.5 and r_slow > 3.0
    # K2: scaling does not remove the slow-chain growth
    r_slow_s = tc(0.99, 0.9999, "centered+scaled") / tc(0.99, 0.99, "centered+scaled")
    k2 = r_slow_s > 2.0
    # K3: log t_conv (centered) is predicted by log 1/(1 - gamma*lambda2) with R^2 > 0.8 across the grid
    xs = np.log([s["pred"] for s in summary if s["method"] == "centered"]); ys = np.log([s["t_conv_gmean"] for s in summary if s["method"] == "centered"])
    xh = np.log([1 / (1 - s["gamma"]) for s in summary if s["method"] == "centered"])
    r2_pred = float(np.corrcoef(xs, ys)[0, 1] ** 2); r2_hor = float(np.corrcoef(xh, ys)[0, 1] ** 2)
    k3 = r2_pred > 0.8 and r2_pred > r2_hor + 0.1
    text = (f"K1 (centered gamma-ratio 0.9999/0.99: fast chain {r_fast:.2f} < 1.5, slow chain {r_slow:.2f} > 3) = {k1}. "
            f"K2 (same ratio with scaling on slow chain {r_slow_s:.2f} > 2) = {k2}. "
            f"K3 (R2 of log t_conv on log 1/(1-gamma*lambda2) = {r2_pred:.2f} > 0.8 and beats 1/(1-gamma) R2 = {r2_hor:.2f} by 0.1) = {k3}.")
    verdict = True if (k1 and k3) else (False if (not k1 and not k3) else None)
    line = kill_test(verdict, text)
    (RESULTS / "verdict.txt").write_text(line + "\n")
    print("\nalpha  l2     gamma    plain        centered     cent+scaled   pred 1/(1-g*l2)")
    for alpha in ALPHAS:
        for gamma in GAMMAS:
            s = [x for x in summary if x["alpha"] == alpha and x["gamma"] == gamma]
            if len(s) == 3:
                print(f"{alpha:.2f}  {s[0]['lambda2']:.3f}  {gamma:<7} " + "  ".join(f"{x['t_conv_gmean']:9.0f}({x['final_err']:.2f})" for x in s) + f"   {s[0]['pred']:8.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--check-every", type=int, default=2_000)
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.steps, args.check_every = 1, 40_000, 1_000
    if args.summarize:
        summarize(); return
    run_grid(args)


if __name__ == "__main__":
    main()
