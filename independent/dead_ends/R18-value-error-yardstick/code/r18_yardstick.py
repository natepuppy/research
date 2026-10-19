"""R18: which value-error statistic predicts the greedy policy's suboptimality? Garnet corpus, Q-learning Q-hats, exact truth."""

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
S, A, BR = 50, 4, 3
GAMMA = 0.9
N_MDP = 24
ALPHAS = [0.05, 0.2]
EPSS = [0.1, 0.5]
SIGMAS = [0.0, 0.5]
RESTART = 50
STATS = ["sup", "rms", "rms_dstar", "rms_dhat", "bell_sup", "bell_rms_dhat", "over_dhat", "abs_over_dhat", "cross_unif", "cross_dstar", "cross_dhat"]


def garnet(rng, sparse):
    succ = np.zeros((S, A, BR), int); prob = np.zeros((S, A, BR))
    for s in range(S):
        for a in range(A):
            succ[s, a] = rng.choice(S, BR, replace=False)
            p = rng.random(BR); prob[s, a] = p / p.sum()
    R = rng.normal(size=(S, A))
    if sparse:
        R *= rng.random((S, A)) < 0.05
    P = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            np.add.at(P[s, a], succ[s, a], prob[s, a])
    return dict(succ=succ, prob=prob, R=R, P=P)


def q_star(m):
    Q = np.zeros((S, A))
    for _ in range(5000):
        Qn = m["R"] + GAMMA * m["P"] @ Q.max(1)
        if np.abs(Qn - Q).max() < 1e-12:
            return Qn
        Q = Qn
    return Q


def policy_eval(m, pi):
    Ppi = m["P"][np.arange(S), pi]; Rpi = m["R"][np.arange(S), pi]
    V = np.linalg.solve(np.eye(S) - GAMMA * Ppi, Rpi)
    d = (1 - GAMMA) * np.linalg.solve(np.eye(S) - GAMMA * Ppi.T, np.full(S, 1 / S))
    return V, d


def stats_for(m, Qs, Qh, Vs, dstar, astar):
    err = Qh - Qs
    pi = Qh.argmax(1)
    V, dhat = policy_eval(m, pi)
    delta = float(np.mean(Vs - V))
    TQ = m["R"] + GAMMA * m["P"] @ Qh.max(1)
    bell = Qh - TQ
    cross = (pi != astar).astype(float)
    over = Qh.max(1) - Vs
    return dict(delta=delta, sup=float(np.abs(err).max()), rms=float(np.sqrt((err ** 2).mean())),
                rms_dstar=float(np.sqrt((dstar[:, None] * err ** 2).sum() / A)), rms_dhat=float(np.sqrt((dhat[:, None] * err ** 2).sum() / A)),
                bell_sup=float(np.abs(bell).max()), bell_rms_dhat=float(np.sqrt((dhat[:, None] * bell ** 2).sum() / A)),
                over_dhat=float((dhat * over).sum()), abs_over_dhat=float(abs((dhat * over).sum())),
                cross_unif=float(cross.mean()), cross_dstar=float((dstar * cross).sum()), cross_dhat=float((dhat * cross).sum()))


def run(args):
    rng = np.random.default_rng(args.seed_base)
    mdps = [garnet(rng, sparse=(i % 2 == 1)) for i in range(args.n_mdp)]
    truth = []
    for m in mdps:
        Qs = q_star(m); astar = Qs.argmax(1); Vs, dstar = policy_eval(m, astar)
        truth.append((Qs, Vs, dstar, astar))
    learners = [(al, e, sg, sd) for al in ALPHAS for e in EPSS for sg in SIGMAS for sd in range(args.seeds)]
    agents = [(mi, li) for mi in range(len(mdps)) for li in range(len(learners))]
    C = len(agents)
    alpha = np.array([learners[li][0] for _, li in agents]); eps = np.array([learners[li][1] for _, li in agents]); sig = np.array([learners[li][2] for _, li in agents])
    succ = np.stack([mdps[mi]["succ"] for mi, _ in agents]); prob = np.stack([mdps[mi]["prob"] for mi, _ in agents]); Rm = np.stack([mdps[mi]["R"] for mi, _ in agents])
    cprob = np.cumsum(prob, -1)
    Q = rng.normal(scale=1e-6, size=(C, S, A))
    state = rng.integers(0, S, C)
    ar = np.arange(C)
    ckpts = [int(args.steps * k / 10) for k in range(1, 11)]
    rows = []
    t0 = time.time()
    for t in range(1, args.steps + 1):
        greedy = Q[ar, state].argmax(-1)
        a = np.where(rng.random(C) < eps, rng.integers(0, A, C), greedy)
        u = rng.random(C)
        br = (u[:, None] > cprob[ar, state, a]).sum(-1).clip(max=BR - 1)
        s2 = succ[ar, state, a, br]
        r = Rm[ar, state, a] + sig * rng.normal(size=C)
        delta = r + GAMMA * Q[ar, s2].max(-1) - Q[ar, state, a]
        Q[ar, state, a] += alpha * delta
        state = np.where(t % RESTART == 0, rng.integers(0, S, C), s2)
        if t in ckpts:
            for j, (mi, li) in enumerate(agents):
                Qs, Vs, dstar, astar = truth[mi]
                st = stats_for(mdps[mi], Qs, Q[j], Vs, dstar, astar)
                al, e, sg, sd = learners[li]
                rows.append(dict(mdp=mi, sparse=(mi % 2 == 1), alpha=al, eps=e, sigma=sg, seed=sd, step=t, **st))
            print(f"t={t} [{time.time() - t0:.0f}s]  mean delta {np.mean([r['delta'] for r in rows if r['step'] == t]):.3f}  "
                  f"mean sup {np.mean([r['sup'] for r in rows if r['step'] == t]):.3f}", flush=True)
    save_csv(RESULTS / "runs.csv", rows)
    summarize(rows)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(rows=None):
    rows = rows if rows is not None else load_csv(RESULTS / "runs.csv")
    groups = {}
    for r in rows:
        groups.setdefault((r["mdp"], r["step"]), []).append(r)
    pooled = {}
    for r in rows:
        pooled.setdefault(r["mdp"], []).append(r)
    summary = []
    for st in STATS:
        tc = [spearman([r[st] for r in g], [r["delta"] for r in g]) for g in groups.values()]
        pl = [spearman([r[st] for r in g], [r["delta"] for r in g]) for g in pooled.values()]
        tc_d = [spearman([r[st] for r in g], [r["delta"] for r in g]) for (mi, _), g in groups.items() if mi % 2 == 0]
        tc_s = [spearman([r[st] for r in g], [r["delta"] for r in g]) for (mi, _), g in groups.items() if mi % 2 == 1]
        summary.append(dict(stat=st, rho_time_controlled=float(np.nanmean(tc)), rho_tc_sd=float(np.nanstd(tc)), rho_pooled=float(np.nanmean(pl)),
                            rho_tc_dense=float(np.nanmean(tc_d)), rho_tc_sparse=float(np.nanmean(tc_s))))
    save_csv(RESULTS / "summary.csv", summary)
    rho = {s["stat"]: s["rho_time_controlled"] for s in summary}
    k1 = rho["sup"] <= 0.5
    k2 = rho["cross_dhat"] >= 0.8
    best_norm = max(rho[s] for s in STATS[:8])
    k3 = (rho["cross_dhat"] - best_norm >= 0.1) and (rho["cross_dhat"] - rho["cross_unif"] >= 0.05)
    npass = sum([k1, k2, k3])
    verdict = True if npass == 3 else (None if npass == 2 else False)
    text = (f"K1 (time-controlled Spearman(sup, delta) = {rho['sup']:+.2f} <= 0.5) = {k1}. "
            f"K2 (Spearman(cross_dhat, delta) = {rho['cross_dhat']:+.2f} >= 0.8) = {k2}. "
            f"K3 (cross_dhat beats best norm-type statistic {best_norm:+.2f} by >= 0.1, and cross_unif {rho['cross_unif']:+.2f} by >= 0.05) = {k3}. "
            f"[secondary: |Spearman(over_dhat, delta)| = {abs(rho['over_dhat']):.2f} <= 0.5 = {abs(rho['over_dhat']) <= 0.5}]")
    line = kill_test(verdict, text)
    (RESULTS / "verdict.txt").write_text(line + "\n")
    print("\nstat            time-ctrl (sd)    pooled   dense   sparse")
    for s in summary:
        print(f"{s['stat']:14s}  {s['rho_time_controlled']:+.2f} ({s['rho_tc_sd']:.2f})     {s['rho_pooled']:+.2f}    {s['rho_tc_dense']:+.2f}   {s['rho_tc_sparse']:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-mdp", type=int, default=N_MDP)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize(); return
    if args.smoke:
        args.steps, args.seeds, args.n_mdp = 10_000, 1, 4
    RESULTS.mkdir(exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
