"""R21: Polyak target tables as an implicit discount schedule. Deterministic synchronous tabular updates."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io_utils import kill_test, save_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GAMMA = 0.99
TAUS = [0.1, 0.01, 0.001]
ALPHAS = [1.0, 0.1, 0.01]
NS = [1, 5]
DEPTHS = [10, 30, 100]
CHAIN_D = 200


def cells():
    out = [dict(tau=t, alpha=a, n=n, mode="soft") for t in TAUS for a in ALPHAS for n in NS]
    out += [dict(tau=t, alpha=1.0, n=n, mode="hard") for t in TAUS for n in NS]  # hard sync with K = 1/tau
    return out


def nstep_targets(Qbar_next_vals, rewards, n, S_corr):
    """Deterministic single-action corridor of S_corr states (index 0..S_corr-1), moving +1 each step, terminal after the
    last. rewards[i] = reward on leaving state i. Qbar_next_vals[c, i] = target table value at state i. Returns y[c, i]."""
    key = (n, S_corr, id(rewards))
    if key not in _CACHE:
        Rn = np.zeros(S_corr); boot = np.zeros(S_corr, bool); nxt = np.zeros(S_corr, int)
        for i in range(S_corr):
            disc = 1.0
            for j in range(i, min(i + n, S_corr)):
                Rn[i] += disc * rewards[j]; disc *= GAMMA
            if i + n < S_corr:
                boot[i] = True; nxt[i] = i + n
        _CACHE[key] = (Rn, boot, nxt)
    Rn, boot, nxt = _CACHE[key]
    return Rn[None] + (GAMMA ** n) * boot[None] * Qbar_next_vals[:, nxt]


_CACHE: dict = {}


def run(max_steps, log):
    cs = cells(); C = len(cs)
    tau = np.array([c["tau"] for c in cs]); alpha = np.array([c["alpha"] for c in cs]); n_arr = np.array([c["n"] for c in cs])
    hard = np.array([c["mode"] == "hard" for c in cs]); K = np.where(hard, np.round(1 / tau).astype(int), 1)
    ns = sorted(set(NS))
    # ---- chain: reward 1 per step, D states, V(s0) traced
    Q = np.zeros((C, CHAIN_D)); Qb = np.zeros((C, CHAIN_D)); rew = np.ones(CHAIN_D)
    htrace = []  # (t, cell, h)
    # ---- forks: state 0 start; corridor states 1..d-1; reward R on leaving d-1. 'now' pays 1.
    forks = {}
    for d in DEPTHS:
        R = 2.0 / GAMMA ** d
        forks[d] = dict(Qw=np.zeros((C, d)), Qbw=np.zeros((C, d)), Qnow=np.zeros(C), Qbnow=np.zeros(C), R=R,
                        rewards=np.array([0.0] * (d - 1) + [R]), tstar=np.full(C, -1))
    t0 = time.time()
    for t in range(1, max_steps + 1):
        # chain
        y = np.zeros_like(Q)
        for n in ns:
            m = n_arr == n
            y[m] = nstep_targets(Qb[m], rew, n, CHAIN_D)
        Q += alpha[:, None] * (y - Q)
        for d, f in forks.items():
            # corridor index i: state i+1 ... use arrays over corridor positions 0..d-1 where position 0 == start's 'wait' edge
            # wait: moving from start into corridor is position 0 (reward 0), positions 1..d-1 follow, reward R on leaving d-1
            yw = np.zeros_like(f["Qw"])
            for n in ns:
                m = n_arr == n
                yw[m] = nstep_targets(f["Qbw"][m], f["rewards"], n, d)
            f["Qw"] += alpha[:, None] * (yw - f["Qw"])
            f["Qnow"] += alpha * (1.0 - f["Qnow"])
            newly = (f["Qw"][:, 0] > f["Qnow"]) & (f["tstar"] < 0)
            f["tstar"][newly] = t
        # target tables
        soft = ~hard
        Qb[soft] += tau[soft, None] * (Q[soft] - Qb[soft])
        sync = hard & (t % K == 0)
        Qb[sync] = Q[sync]
        for f in forks.values():
            f["Qbw"][soft] += tau[soft, None] * (f["Qw"][soft] - f["Qbw"][soft]); f["Qbw"][sync] = f["Qw"][sync]
            f["Qbnow"][soft] += tau[soft] * (f["Qnow"][soft] - f["Qbnow"][soft]); f["Qbnow"][sync] = f["Qnow"][sync]
        if t % 10 == 0 or t < 10:
            V0 = Q[:, 0]
            h = np.log(np.clip(1 - (1 - GAMMA) * V0, 1e-12, 1)) / np.log(GAMMA)
            for c in range(C):
                htrace.append((t, c, float(h[c])))
        if t % 20000 == 0:
            log(f"t={t} [{time.time() - t0:.0f}s] fork100 switched {np.mean(forks[100]['tstar'] >= 0):.2f}")
        if all((f["tstar"] >= 0).all() for f in forks.values()) and t > 2000:
            break
    # ---- summarise
    hrows = [dict(t=tt, cell=c, h=h) for tt, c, h in htrace]
    save_csv(RESULTS / "horizon.csv", hrows)
    frows = []
    for d, f in forks.items():
        for c in range(C):
            m = int(np.ceil(d / cs[c]["n"]))
            if cs[c]["mode"] == "soft":
                it, ia = 1 / cs[c]["tau"], 1 / cs[c]["alpha"]
                pred = (m - 1) * it + m * ia - max(it, ia) / 3
            else:
                pred = float("nan")
            frows.append(dict(d=d, m=m, **cs[c], K=int(K[c]) if hard[c] else 0, tstar=int(f["tstar"][c]), tstar_pred=pred))
    save_csv(RESULTS / "fork.csv", frows)
    summ = []
    H = np.array([(tt, c, h) for tt, c, h in htrace])
    for c in range(C):
        sel = H[(H[:, 1] == c) & (H[:, 2] >= 5) & (H[:, 2] <= 100)]
        if len(sel) >= 5:
            A = np.column_stack([sel[:, 0], np.ones(len(sel))]); b, *_ = np.linalg.lstsq(A, sel[:, 2], rcond=None)
            pred = A @ b; r2 = 1 - ((sel[:, 2] - pred) ** 2).sum() / max(((sel[:, 2] - sel[:, 2].mean()) ** 2).sum(), 1e-12)
            slope = b[0]
        else:
            slope, r2 = float("nan"), float("nan")
        rate_pred = cs[c]["n"] / (1 / cs[c]["tau"] + 1 / cs[c]["alpha"]) if cs[c]["mode"] == "soft" else cs[c]["n"] / (1 / cs[c]["tau"] + 1)
        summ.append(dict(**cs[c], h_slope=slope, h_slope_pred=rate_pred, ratio=slope / rate_pred, h_r2=r2))
    save_csv(RESULTS / "summary.csv", summ)
    verdict(summ, frows, cs, log)


def verdict(summ, frows, cs, log):
    soft = [s for s in summ if s["mode"] == "soft"]
    k1_ok = [(0.8 <= s["ratio"] <= 1.2) and s["h_r2"] >= 0.98 for s in soft]
    k1 = sum(k1_ok) >= 16
    fs = [f for f in frows if f["mode"] == "soft"]
    x = np.log([f["tstar_pred"] for f in fs]); y = np.log([f["tstar"] for f in fs])
    A = np.column_stack([x, np.ones_like(x)]); b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = 1 - ((y - A @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    ratios = [f["tstar"] / f["tstar_pred"] for f in fs]
    n_in = sum(0.7 <= r <= 1.4 for r in ratios)
    k2 = (0.9 <= b[0] <= 1.1) and r2 >= 0.98 and n_in >= 48

    def fork(d, tau, alpha, n, mode):
        return next(f for f in frows if (f["d"], f["tau"], f["alpha"], f["n"], f["mode"]) == (d, tau, alpha, n, mode))

    k3_parts = []
    for ta in [0.1, 0.01]:
        for n in NS:
            for d in DEPTHS:
                m = int(np.ceil(d / n))
                pred_ratio = (2 * m - 1 - 1 / 3) / ((m - 1 - 1 / 3) + m * ta)
                r = fork(d, ta, ta, n, "soft")["tstar"] / fork(d, ta, 1.0, n, "soft")["tstar"] / pred_ratio
                k3_parts.append(0.8 <= r <= 1.25)
    k3 = all(k3_parts)
    k4_parts = []
    for tau in TAUS:
        for n in NS:
            for d in DEPTHS:
                if int(np.ceil(d / n)) >= 6:
                    r = fork(d, tau, 1.0, n, "hard")["tstar"] / fork(d, tau, 1.0, n, "soft")["tstar"]
                    k4_parts.append((tau, n, d, round(r, 2)))
    k4 = all(0.7 <= r[3] <= 1.3 for r in k4_parts)
    n_ok = sum([k1, k2, k3])
    v = True if n_ok == 3 else (None if n_ok == 2 else False)
    text = (f"K1 (horizon slope within [0.8,1.2] of n/(1/tau+1/alpha) and R2>=0.98 in {sum(k1_ok)}/18 soft cells; ratios {[round(float(s['ratio']), 2) for s in soft]}) = {k1}. "
            f"K2 (fork switch: log t* on log pred slope {b[0]:.3f}, R2 {r2:.4f}, {n_in}/54 ratios in [0.7,1.4]) = {k2}. "
            f"K3 (tau=alpha cells vs alpha=1 cells match the predicted ratio within [0.8,1.25]: {sum(k3_parts)}/12) = {k3}. "
            f"K4 (hard K=1/tau vs soft, m>=6: ratios {[r[3] for r in k4_parts]}) = {k4}.")
    line = kill_test(v, text)
    (RESULTS / "verdict.txt").write_text(line + "\n")
    log("\n tau    alpha  n  mode   h_slope  pred   ratio   r2")
    for s in summ:
        log(f"{s['tau']:.3f}  {s['alpha']:.2f}  {s['n']}  {s['mode']:4s}  {s['h_slope']:.4f}  {s['h_slope_pred']:.4f}  {s['ratio']:.2f}  {s['h_r2']:.3f}")
    log("\n   d  tau    alpha  n  mode    t*     pred   ratio")
    for f in frows:
        log(f" {f['d']:3d}  {f['tau']:.3f}  {f['alpha']:.2f}  {f['n']}  {f['mode']:4s}  {f['tstar']:6d}  {f['tstar_pred']:8.1f}  {f['tstar'] / f['tstar_pred'] if f['tstar_pred'] == f['tstar_pred'] else float('nan'):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=250_000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.max_steps = 3000
    RESULTS.mkdir(exist_ok=True)
    run(args.max_steps, lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
