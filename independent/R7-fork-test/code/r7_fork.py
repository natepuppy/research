"""R7 pilot: when is a deep-RL run's fate sealed, and by which noise?

Parents: N seeds of batched DQN on CartPole, T steps, full snapshots at each fork time.
Children: at fork time t_f, K children per parent per condition continue from the snapshot with
    all      : environment/exploration AND minibatch generators reseeded
    explore  : only the environment/exploration generator reseeded (minibatch stream copied)
    batch    : only the minibatch generator reseeded (environment stream copied)
Lock-in curve: intraclass correlation of children's final scores grouped by parent, per t_f and
condition. Visible-fate curve: R^2 of child final score on the parent's evaluation score at t_f.
Pre-registered kill test in README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import dqn_batched as D  # noqa: E402
from lib.io_utils import kill_test, save_csv, load_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CONDITIONS = ["all", "explore", "batch"]


def run(args):
    torch.set_num_threads(args.threads)
    RESULTS.mkdir(exist_ok=True)
    cfg = D.Config(total_steps=args.T, eval_every=args.eval_every)
    forks = [int(f) for f in args.forks]
    N, K = args.parents, args.children
    t0 = time.time()
    # ---- parents
    P = D.BatchedDQN(cfg, [1000 + i for i in range(N)], [2000 + i for i in range(N)], [3000 + i for i in range(N)])
    snaps = {}
    for step in range(1, args.T + 1):
        if (step - 1) in forks:
            snaps[step - 1] = [P.snapshot([i]) for i in range(N)]
        P.step()
    parent_final = [P.final_score(i) for i in range(N)]
    parent_log = [P.eval_log[i] for i in range(N)]
    print(f"parents done [{time.time() - t0:.0f}s] finals: " + " ".join(f"{v:.0f}" for v in parent_final))
    rows = []
    # ---- children, one batch per fork time
    for tf in forks:
        reps, rs_env, rs_batch, meta = [], [], [], []
        for i in range(N):
            for cond in CONDITIONS:
                for k in range(K):
                    meta.append((i, cond, k))
                    seed = 100_000 + tf * 100 + i * 10 + k
                    rs_env.append(seed if cond in ("all", "explore") else None)
                    rs_batch.append(seed + 50_000 if cond in ("all", "batch") else None)
        reps = [len(CONDITIONS) * K] * N
        C = D.BatchedDQN.from_snapshots(cfg, snaps[tf], reps, rs_env, rs_batch)
        for _ in range(args.T - tf):
            C.step()
        for j, (i, cond, k) in enumerate(meta):
            vis = [v for (s, v) in parent_log[i] if s <= tf]
            rows.append(dict(fork=tf, parent=i, condition=cond, child=k, final=C.final_score(j), parent_final=parent_final[i],
                             parent_visible=(vis[-1] if vis else float("nan")), parent_recent=(float(np.mean(snaps[tf][i]["recent"][0][-5:])) if snaps[tf][i]["recent"][0] else float("nan"))))
        save_csv(RESULTS / "runs.csv", rows)
        by = {c: np.array([r["final"] for r in rows if r["fork"] == tf and r["condition"] == c]) for c in CONDITIONS}
        print(f"fork {tf:6d} done [{time.time() - t0:.0f}s]  child mean final: " + "  ".join(f"{c}={by[c].mean():.0f}" for c in CONDITIONS))
    json.dump(dict(T=args.T, parent_final=parent_final, parent_log=parent_log), open(RESULTS / "parents.json", "w"))
    summarize(rows)


def icc(groups: list[np.ndarray]) -> float:
    k = min(len(g) for g in groups)
    g = np.array([x[:k] for x in groups])
    n = len(g)
    grand = g.mean()
    msb = k * ((g.mean(1) - grand) ** 2).sum() / (n - 1)
    msw = ((g - g.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return float((msb - msw) / (msb + (k - 1) * msw + 1e-12))


def r2(x: np.ndarray, y: np.ndarray) -> float:
    m = ~np.isnan(x) & ~np.isnan(y)
    if m.sum() < 3 or np.std(x[m]) == 0:
        return float("nan")
    return float(np.corrcoef(x[m], y[m])[0, 1] ** 2)


def summarize(rows=None):
    rows = rows if rows is not None else load_csv(RESULTS / "runs.csv")
    rng = np.random.default_rng(0)
    forks = sorted({r["fork"] for r in rows})
    parents = sorted({r["parent"] for r in rows})
    # T is the parent training length (README: forks <= T/2 count as "early"), not the last fork time.
    pj = RESULTS / "parents.json"
    T = float(json.load(open(pj)).get("T", 30_000)) if pj.exists() else 30_000.0
    summary = []
    for tf in forks:
        for c in CONDITIONS:
            sel = [r for r in rows if r["fork"] == tf and r["condition"] == c]
            groups = [np.array([r["final"] for r in sel if r["parent"] == p]) for p in parents]
            groups = [g for g in groups if len(g) >= 2]
            val = icc(groups)
            boots = [icc([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(300)]
            x_vis = np.array([r["parent_visible"] for r in sel]); x_fin = np.array([r["parent_final"] for r in sel]); y = np.array([r["final"] for r in sel])
            summary.append(dict(fork=tf, condition=c, icc=val, icc_lo=float(np.quantile(boots, 0.05)), icc_hi=float(np.quantile(boots, 0.95)),
                                r2_visible=r2(x_vis, y), r2_parent_final=r2(x_fin, y), child_mean=float(y.mean()), child_sd=float(y.std()),
                                n_children=len(sel)))
    save_csv(RESULTS / "summary.csv", summary)

    def get(tf, c, key):
        return next(s[key] for s in summary if s["fork"] == tf and s["condition"] == c)

    # K1: lock-in exists early
    early = [tf for tf in forks if tf <= T / 2 and get(tf, "all", "icc") >= 0.5 and get(tf, "all", "icc_lo") > 0.2]
    k1 = len(early) > 0
    t_lock = min(early) if early else None
    # K2: the two noise sources differ
    diffs = [(tf, get(tf, "explore", "icc") - get(tf, "batch", "icc")) for tf in forks if tf > 0]
    pos = [d for _, d in diffs if d >= 0.15]; neg = [d for _, d in diffs if d <= -0.15]
    k2 = len(pos) >= 2 or len(neg) >= 2
    # K3: fate decided before it is visible
    k3 = False
    if t_lock is not None:
        k3 = get(t_lock, "all", "r2_visible") <= get(t_lock, "all", "icc") - 0.2
    verdict = True if (k1 and (k2 or k3)) else (None if k1 else False)
    text = (f"K1 (ICC_all >= 0.5 with CI lower > 0.2 at some fork <= T/2; first such fork = {t_lock}) = {k1}. "
            f"K2 (|ICC_explore - ICC_batch| >= 0.15 at >= 2 forks; diffs = " + ", ".join(f"{tf}:{d:+.2f}" for tf, d in diffs) + f") = {k2}. "
            f"K3 (at lock-in fork, R2 of child final on parent's visible score <= ICC - 0.2" + (f": {get(t_lock, 'all', 'r2_visible'):.2f} vs {get(t_lock, 'all', 'icc'):.2f}" if t_lock is not None else "") + f") = {k3}.")
    line = kill_test(verdict, text)
    (RESULTS / "verdict.txt").write_text(line + "\n")
    print("\nfork    cond     ICC [90% CI]        R2(visible)  R2(parent final)  child mean (sd)")
    for s in summary:
        print(f"{s['fork']:6d}  {s['condition']:8s} {s['icc']:5.2f} [{s['icc_lo']:5.2f},{s['icc_hi']:5.2f}]   {s['r2_visible']:5.2f}        {s['r2_parent_final']:5.2f}          {s['child_mean']:6.1f} ({s['child_sd']:5.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--parents", type=int, default=16)
    ap.add_argument("--children", type=int, default=3)
    ap.add_argument("--T", type=int, default=30_000)
    ap.add_argument("--eval-every", type=int, default=2_500)
    ap.add_argument("--forks", nargs="+", default=[0, 2500, 5000, 10000, 15000, 20000, 25000])
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if args.smoke:
        args.parents, args.children, args.T, args.eval_every, args.forks = 4, 2, 4000, 1000, [0, 2000]
    if args.summarize:
        summarize(); return
    run(args)


if __name__ == "__main__":
    main()
