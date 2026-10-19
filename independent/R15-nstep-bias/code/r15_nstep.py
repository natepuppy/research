"""R15: Why does uncorrected n-step work? A bias accounting.

Tabular Q-learning with a FIFO replay buffer, a target table and *uncorrected* n-step targets (behaviour actions
from the buffer, no importance weights) on a stochastic gridworld with controllable reward noise. For every
(n, noise, epsilon_behaviour, method) we measure, against the exact Q*, the overestimation of the start state's
greedy value and the greedy policy's true return. Hypothesis: the uncorrected n-step target has a pessimistic
fixed-point bias (it blends Q^mu into Q*) that cancels Q-learning's max-overestimation; so (i) overestimation
falls with n, (ii) the best n rises with noise, (iii) with double-Q (max-bias largely removed) the n-dependence
weakens. Pre-registered kill test in README.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io_utils import kill_test, save_csv, load_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GAMMA = 0.95
SLIP = 0.1
W = 8
S = W * W
A = 4
MOVES = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])
GOAL = S - 1
EP_CAP = 100
NS = [1, 2, 3, 5, 10]
SIGMAS = [0.0, 0.5, 1.5]
EPS_B = [0.1, 0.3]
METHODS = ["q", "double"]
ALPHA = 0.1
BUF = 5000
BATCH = 32
K_TARGET = 100


def build_mdp():
    P = np.zeros((S, A, S)); R = np.zeros((S, A))
    for s in range(S):
        r, c = divmod(s, W)
        for a in range(A):
            for a2 in range(A):
                p = (1 - SLIP) if a2 == a else SLIP / (A - 1)
                r2, c2 = r + MOVES[a2][0], c + MOVES[a2][1]
                s2 = r2 * W + c2 if (0 <= r2 < W and 0 <= c2 < W) else s
                if s == GOAL:
                    s2 = s
                P[s, a, s2] += p
            R[s, a] = P[s, a, GOAL] if s != GOAL else 0.0
    return P, R


def optimal_q(P, R):
    V = np.zeros(S)
    for _ in range(5000):
        Q = R + GAMMA * P @ V
        V2 = Q.max(1)
        if np.abs(V2 - V).max() < 1e-12:
            break
        V = V2
    return Q


def greedy_return(P, R, Q_tab, s0=0):
    """True value at s0 of the greedy policy of Q_tab (ties broken by first index). Vectorised over agents."""
    C = Q_tab.shape[0]
    g = Q_tab.argmax(-1)                                   # (C, S)
    Ppi = P[np.arange(S)[None, :], g]                      # (C, S, S)
    Rpi = R[np.arange(S)[None, :], g]                      # (C, S)
    V = np.linalg.solve(np.eye(S)[None] - GAMMA * Ppi, Rpi[..., None])[..., 0]
    mu = np.zeros(S); mu[s0] = 1.0
    d = np.linalg.solve(np.eye(S)[None] - GAMMA * np.swapaxes(Ppi, 1, 2), np.broadcast_to(mu, (C, S))[..., None])[..., 0] * (1 - GAMMA)
    return V[:, s0], d


def run(args):
    P, R = build_mdp(); Qstar = optimal_q(P, R); Vstar0 = Qstar[0].max()
    cfgs = [(n, sg, e, m) for n in NS for sg in SIGMAS for e in EPS_B for m in METHODS]
    C = len(cfgs) * args.seeds
    n_arr = np.array([c[0] for c in cfgs for _ in range(args.seeds)]); sig = np.array([c[1] for c in cfgs for _ in range(args.seeds)])
    eps = np.array([c[2] for c in cfgs for _ in range(args.seeds)]); dbl = np.array([c[3] == "double" for c in cfgs for _ in range(args.seeds)])
    seeds = np.array([s for _ in cfgs for s in range(args.seeds)])
    rng = np.random.default_rng(args.seed_base)
    Q = rng.random((C, S, A)) * 1e-6; Qt = Q.copy()            # tiny noise: random tie-breaking
    # replay (circular): per slot store s, a, r, s2, done, ep, t
    bs = np.zeros((C, BUF), np.int64); ba = np.zeros((C, BUF), np.int64); br = np.zeros((C, BUF)); bs2 = np.zeros((C, BUF), np.int64)
    bd = np.zeros((C, BUF)); bep = np.full((C, BUF), -1, np.int64); bt = np.full((C, BUF), -10 ** 9, np.int64)
    s = rng.integers(0, S - 1, C); ep = np.zeros(C, np.int64); ept = np.zeros(C, np.int64)   # exploring starts for behaviour
    ar = np.arange(C)
    cum_P = P.cumsum(-1)
    log = []
    t0 = time.time()
    for t in range(1, args.steps + 1):
        # act: epsilon-greedy on online Q
        greedy = Q[ar, s].argmax(-1)
        explore = rng.random(C) < eps
        a = np.where(explore, rng.integers(0, A, C), greedy)
        s2 = (rng.random(C)[:, None] > cum_P[s, a]).sum(-1).clip(max=S - 1)
        r = R[s, a] * 0 + (s2 == GOAL).astype(float) * (s != GOAL) + sig * rng.standard_normal(C)   # reward on entering goal, plus noise
        done = (s2 == GOAL).astype(float)
        ept += 1
        trunc = ept >= EP_CAP
        slot = (t - 1) % BUF
        bs[:, slot] = s; ba[:, slot] = a; br[:, slot] = r; bs2[:, slot] = s2; bd[:, slot] = done; bep[:, slot] = ep; bt[:, slot] = t
        reset = (done > 0) | trunc
        s = np.where(reset, rng.integers(0, S - 1, C), s2)       # exploring starts (uniform over non-goal states)
        ep = ep + reset; ept = np.where(reset, 0, ept)
        # learn
        if t >= 500:
            nfill = min(t, BUF)
            idx = rng.integers(0, nfill, (C, BATCH))
            base_t = bt[ar[:, None], idx]; base_ep = bep[ar[:, None], idx]
            y = np.zeros((C, BATCH)); disc = np.ones((C, BATCH)); alive = np.ones((C, BATCH), bool)
            last_s2 = bs2[ar[:, None], idx]; last_d = bd[ar[:, None], idx]
            for j in range(int(n_arr.max())):
                pos = (idx + j) % BUF
                ok = alive & (j < n_arr[:, None]) & (bt[ar[:, None], pos] == base_t + j) & (bep[ar[:, None], pos] == base_ep)
                y += np.where(ok, disc * br[ar[:, None], pos], 0.0)
                last_s2 = np.where(ok, bs2[ar[:, None], pos], last_s2); last_d = np.where(ok, bd[ar[:, None], pos], last_d)
                disc = np.where(ok, disc * GAMMA, disc)
                alive = ok & (bd[ar[:, None], pos] < 0.5)
            # bootstrap from the last valid s2 with the discount accumulated so far (disc = gamma^j)
            qn_t = Qt[ar[:, None], last_s2]                                   # (C, B, A)
            if dbl.any():
                a_on = Q[ar[:, None], last_s2].argmax(-1)
                boot_dbl = np.take_along_axis(qn_t, a_on[..., None], -1)[..., 0]
            boot_q = qn_t.max(-1)
            boot = np.where(dbl[:, None], boot_dbl, boot_q) if dbl.any() else boot_q
            y += disc * (1 - last_d) * boot
            s_b = bs[ar[:, None], idx]; a_b = ba[ar[:, None], idx]
            delta = y - Q[ar[:, None], s_b, a_b]
            upd = np.zeros_like(Q)
            np.add.at(upd, (ar[:, None].repeat(BATCH, 1), s_b, a_b), ALPHA * delta)
            Q += upd
            if t % K_TARGET == 0:
                Qt[:] = Q
        if t % args.eval_every == 0 or t == args.steps:
            gr, dvis = greedy_return(P, R, Q)
            over0 = Q[:, 0].max(-1) - Qstar[0, Q[:, 0].argmax(-1)]           # overestimation of the start greedy value
            over_all = ((Q.max(-1) - Qstar.max(-1)) * dvis).sum(-1)          # weighted by the greedy policy's visitation from s0
            for i in range(C):
                log.append(dict(step=t, n=int(n_arr[i]), sigma=float(sig[i]), eps=float(eps[i]), method="double" if dbl[i] else "q", seed=int(seeds[i]),
                                greedy_return=float(gr[i]), over0=float(over0[i]), over_all=float(over_all[i]), vstar0=float(Vstar0)))
            if t % (args.eval_every * 5) == 0 or t == args.steps:
                msg = []
                for m in METHODS:
                    for sg in SIGMAS:
                        sel = [(n, np.mean([l["over0"] for l in log if l["step"] == t and l["method"] == m and l["sigma"] == sg and l["n"] == n])) for n in NS]
                        msg.append(f"{m}/s{sg:g}: " + " ".join(f"n{n}:{v:+.2f}" for n, v in sel))
                print(f"t={t} [{time.time() - t0:.0f}s] over0  " + " | ".join(msg), flush=True)
    save_csv(RESULTS / "runs.csv", log)
    summarize(log)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(rows=None):
    rows = rows if rows is not None else load_csv(RESULTS / "runs.csv")
    T = max(r["step"] for r in rows)
    steps = sorted({r["step"] for r in rows})
    late = [r for r in rows if r["step"] >= T - 4 * (steps[1] - steps[0])]      # last 5 checkpoints
    summary = []
    for m in METHODS:
        for sg in SIGMAS:
            for e in EPS_B:
                for n in NS:
                    sel = [r for r in late if r["method"] == m and r["sigma"] == sg and r["eps"] == e and r["n"] == n]
                    auc = [r for r in rows if r["method"] == m and r["sigma"] == sg and r["eps"] == e and r["n"] == n]
                    summary.append(dict(method=m, sigma=sg, eps=e, n=n, over0=float(np.mean([r["over0"] for r in sel])), over_all=float(np.mean([r["over_all"] for r in sel])),
                                        final_return=float(np.mean([r["greedy_return"] for r in sel])), auc_return=float(np.mean([r["greedy_return"] for r in auc])),
                                        vstar0=sel[0]["vstar0"]))
    save_csv(RESULTS / "summary.csv", summary)

    def cell(m, sg, e):
        return sorted([s for s in summary if s["method"] == m and s["sigma"] == sg and s["eps"] == e], key=lambda s: s["n"])

    # K1: with plain Q, overestimation falls with n in every noisy cell, and n >= 3 is pessimistic at sigma = 0
    rhos = {(sg, e): spearman([s["n"] for s in cell("q", sg, e)], [s["over_all"] for s in cell("q", sg, e)]) for sg in SIGMAS if sg > 0 for e in EPS_B}
    pess = all(s["over_all"] <= 0 for e in EPS_B for s in cell("q", 0.0, e) if s["n"] >= 3)
    k1 = all(v <= -0.8 for v in rhos.values()) and pess
    # K2: the best n (by AUC of greedy return) is larger at sigma = 3 than at sigma = 0, for both eps, plain Q
    best = {(sg, e): max(cell("q", sg, e), key=lambda s: s["auc_return"])["n"] for sg in SIGMAS for e in EPS_B}
    sg_hi = max(SIGMAS)
    k2 = all(best[(sg_hi, e)] > best[(0.0, e)] for e in EPS_B)
    # K3: double-Q weakens the n-dependence of overestimation (slope of over_all on log n, averaged over noisy cells)
    def slope(m):
        sl = []
        for sg in SIGMAS:
            if sg > 0:
                for e in EPS_B:
                    c = cell(m, sg, e); sl.append(np.polyfit(np.log([s["n"] for s in c]), [s["over_all"] for s in c], 1)[0])
        return float(np.mean(sl))
    sq, sd = slope("q"), slope("double")
    k3 = abs(sd) < 0.5 * abs(sq)
    verdict = True if (k1 and k2) else (None if (k1 or k2) else False)
    text = (f"K1 (plain Q: Spearman of overestimation on n per noisy cell = " + ", ".join(f"s{sg:g}/e{e:g}:{v:+.2f}" for (sg, e), v in rhos.items()) +
            f" all <= -0.8; n>=3 pessimistic at sigma=0 = {pess}) = {k1}. "
            f"K2 (best n by AUC, plain Q: " + ", ".join(f"s{sg:g}/e{e:g}:n={v}" for (sg, e), v in best.items()) + f"; best n larger at sigma={sg_hi:g} than at 0 for both eps) = {k2}. "
            f"K3 (slope of overestimation on log n: plain Q {sq:+.3f}, double {sd:+.3f}; |double| < half |plain|) = {k3}.")
    line = kill_test(verdict, text)
    (RESULTS / "verdict.txt").write_text(line + "\n")
    print(f"\nV*(s0) = {summary[0]['vstar0']:.3f}")
    print("method  sigma  eps   n   over0    over_all  final ret  AUC ret")
    for s in summary:
        print(f"{s['method']:7s} {s['sigma']:4.1f}  {s['eps']:4.2f} {s['n']:3d}  {s['over0']:+7.3f}  {s['over_all']:+7.3f}   {s['final_return']:6.3f}    {s['auc_return']:6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--eval-every", type=int, default=2_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--sigmas", type=float, nargs="+", default=None, help="override reward-noise grid (follow-up band)")
    ap.add_argument("--tag", default=None, help="write results to results/<tag>/")
    args = ap.parse_args()
    global SIGMAS, RESULTS
    if args.sigmas is not None:
        SIGMAS = list(args.sigmas)
    if args.tag:
        RESULTS = RESULTS / args.tag
    if args.summarize:
        summarize(); return
    if args.smoke:
        args.steps, args.eval_every, args.seeds = 6000, 2000, 1
    RESULTS.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
