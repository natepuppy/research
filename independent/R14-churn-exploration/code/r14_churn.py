"""R14: Is policy churn the exploration you did not ask for? Target period and epsilon as substitutes.

Batched DQN on CartPole with constant epsilon (no annealing) and a per-run hard target period K. Churn is
measured on every update. Pre-registered kill test in README.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import dqn_batched as D  # noqa: E402
from lib.dqn_churn import ChurnDQN  # noqa: E402
from lib.io_utils import kill_test, save_csv, load_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
KS = [20, 100, 500, 2500]
EPS = [0.0, 0.05, 0.2]


def run(args):
    torch.set_num_threads(args.threads)
    cfg = D.Config(total_steps=args.T, eval_every=args.eval_every)
    Ks, eps, si, se, sb = [], [], [], [], []
    for K in KS:
        for e in EPS:
            for s in range(args.seeds):
                Ks.append(K); eps.append(e); si.append(s); se.append(500 + s); sb.append(900 + s)
    print(f"{len(Ks)} runs x {args.T} steps", flush=True)
    P = ChurnDQN(cfg, si, se, sb, Ks, eps)
    t0 = time.time()
    for step in range(1, args.T + 1):
        P.step()
        if step % cfg.eval_every == 0 and step % 5000 == 0:
            msg = []
            for K in KS:
                for e in EPS:
                    idx = [i for i in range(len(Ks)) if Ks[i] == K and eps[i] == e]
                    msg.append(f"K{K}/e{e:g}: {np.mean([P.eval_log[i][-1][1] for i in idx]):.0f} ch={100 * np.mean([P.churn_log[i][-1][1] for i in idx]):.1f}%")
            print(f"step {step} [{time.time() - t0:.0f}s]  " + "  ".join(msg), flush=True)
    rows = []
    for i in range(len(Ks)):
        for (st, sc), (st2, ch) in zip(P.eval_log[i], P.churn_log[i]):
            rows.append(dict(run=i, K=Ks[i], eps=eps[i], seed=si[i], step=st, score=sc, churn=ch))
    save_csv(RESULTS / "runs.csv", rows)
    summarize(rows)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(rows=None):
    rows = rows if rows is not None else load_csv(RESULTS / "runs.csv")
    T = max(r["step"] for r in rows)
    runs = sorted({r["run"] for r in rows})
    per_run = []
    for i in runs:
        sel = sorted([r for r in rows if r["run"] == i], key=lambda r: r["step"])
        sc = [r["score"] for r in sel]; ch = [r["churn"] for r in sel if r["step"] > 500]
        per_run.append(dict(run=i, K=sel[0]["K"], eps=sel[0]["eps"], seed=sel[0]["seed"], final=float(np.mean(sc[-5:])), auc=float(np.mean(sc)),
                            churn=float(np.mean(ch)), churn_early=float(np.mean(ch[: max(1, len(ch) // 3)]))))
    save_csv(RESULTS / "per_run.csv", per_run)
    summary = []
    for K in KS:
        for e in EPS:
            sel = [p for p in per_run if p["K"] == K and p["eps"] == e]
            summary.append(dict(K=K, eps=e, n=len(sel), final=float(np.mean([p["final"] for p in sel])), final_sd=float(np.std([p["final"] for p in sel])),
                                auc=float(np.mean([p["auc"] for p in sel])), churn=float(np.mean([p["churn"] for p in sel])),
                                churn_early=float(np.mean([p["churn_early"] for p in sel]))))
    save_csv(RESULTS / "summary.csv", summary)

    def g(K, e, key):
        return next(s[key] for s in summary if s["K"] == K and s["eps"] == e)

    # K1: churn falls with the target period (across the 12 cells)
    rho_K = spearman(np.log([s["K"] for s in summary]), [s["churn"] for s in summary])
    k1 = rho_K <= -0.8
    # K2: the value of epsilon grows with K (substitution): gap(K) = AUC(eps=0.2) - AUC(eps=0)
    gaps = {K: g(K, 0.2, "auc") - g(K, 0.0, "auc") for K in KS}
    rho_gap = spearman(np.log(list(gaps.keys())), list(gaps.values()))
    k2 = rho_gap >= 0.8 and (gaps[KS[-1]] - gaps[KS[0]]) >= 20
    # K3: with eps = 0, performance tracks churn across runs (mediator)
    e0 = [p for p in per_run if p["eps"] == 0.0]
    rho_med = spearman([p["churn_early"] for p in e0], [p["auc"] for p in e0])
    k3 = rho_med >= 0.4
    verdict = True if (k1 and k2) else (None if (k1 and k3) else False)
    text = (f"K1 (Spearman of mean churn on log K over 12 cells = {rho_K:+.2f} <= -0.8) = {k1}. "
            f"K2 (epsilon's AUC gain, eps 0.2 minus 0, by K: " + ", ".join(f"K{K}:{v:+.0f}" for K, v in gaps.items()) + f"; Spearman on log K = {rho_gap:+.2f} >= 0.8 and gain(K={KS[-1]}) - gain(K={KS[0]}) >= 20) = {k2}. "
            f"K3 (eps = 0 runs: Spearman of AUC on early churn = {rho_med:+.2f} >= 0.4) = {k3}.")
    line = kill_test(verdict, text)
    (RESULTS / "verdict.txt").write_text(line + "\n")
    print("\n   K    eps   final (sd)      AUC    churn%  early churn%")
    for s in summary:
        print(f"{s['K']:5d}  {s['eps']:4.2f}  {s['final']:6.1f} ({s['final_sd']:4.1f})  {s['auc']:6.1f}  {100 * s['churn']:5.2f}   {100 * s['churn_early']:5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=30_000)
    ap.add_argument("--eval-every", type=int, default=1_000)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize(); return
    if args.smoke:
        args.T, args.eval_every, args.seeds = 5000, 1000, 1
    RESULTS.mkdir(exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
