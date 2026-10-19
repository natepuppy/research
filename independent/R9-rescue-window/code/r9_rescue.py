"""R9 pilot: unlucky or doomed? Can a bad deep-RL run be rescued by a cheap intervention at step t,
and is there a time after which it cannot?

Parents: the same N seeds as R7 (identical construction, so the snapshots are the same runs).
Children at fork t_f, K per parent per condition, all with fresh exploration + minibatch seeds:
    control     : continue (this is R7's 'all' condition)
    reset_last  : re-initialise the output layer and its Adam state (Nikishin-style partial reset)
    reset_all   : fresh network and optimiser, keep the replay buffer (Nikishin et al. 2022 full reset)
    lr_drop     : halve the learning rate for the rest of training
Parents are split by their own final score into bad (bottom third) and good (top third).
Rescue = child final >= the median final of the good parents. Pre-registered kill test in README.md.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import dqn_batched as D  # noqa: E402
from lib.io_utils import kill_test, load_csv, save_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CONDITIONS = ["control", "reset_last", "reset_all", "lr_drop"]


def reset_all(C: D.BatchedDQN, idx: list[int]):
    t = torch.tensor(idx)
    gens = [torch.Generator().manual_seed(999 + i) for i in idx]
    fresh = D.init_params(len(idx), C.cfg.hidden, gens)
    for dst, dst_t, src in zip(C.p.tensors(), C.p_targ.tensors(), fresh.tensors()):
        dst[t] = src; dst_t[t] = src.clone()
    for k in range(6):
        C.opt.m[k][t] = 0.0; C.opt.v[k][t] = 0.0
    C.opt.t[t] = 0.0


def run(args):
    torch.set_num_threads(args.threads)
    RESULTS.mkdir(exist_ok=True)
    cfg = D.Config(total_steps=args.T, eval_every=args.eval_every)
    forks = [int(f) for f in args.forks]
    N, K = args.parents, args.children
    t0 = time.time()
    P = D.BatchedDQN(cfg, [1000 + i for i in range(N)], [2000 + i for i in range(N)], [3000 + i for i in range(N)])
    snaps = {}
    for step in range(1, args.T + 1):
        if (step - 1) in forks:
            snaps[step - 1] = [P.snapshot([i]) for i in range(N)]
        P.step()
    parent_final = [P.final_score(i) for i in range(N)]
    print(f"parents done [{time.time() - t0:.0f}s] finals: " + " ".join(f"{v:.0f}" for v in parent_final))
    rows = []
    for tf in forks:
        rs_env, rs_batch, meta = [], [], []
        for i in range(N):
            for cond in CONDITIONS:
                for k in range(K):
                    meta.append((i, cond, k))
                    seed = 200_000 + tf * 100 + i * 10 + k
                    rs_env.append(seed); rs_batch.append(seed + 50_000)
        C = D.BatchedDQN.from_snapshots(cfg, snaps[tf], [len(CONDITIONS) * K] * N, rs_env, rs_batch)
        idx_last = [j for j, (_, c, _) in enumerate(meta) if c == "reset_last"]
        idx_all = [j for j, (_, c, _) in enumerate(meta) if c == "reset_all"]
        idx_lr = torch.tensor([j for j, (_, c, _) in enumerate(meta) if c == "lr_drop"])
        if idx_last:
            C.reset_last_layer(idx_last)
        if idx_all:
            reset_all(C, idx_all)
        # per-run learning rate: make lr a tensor
        lr = torch.full((C.R,), cfg.lr); lr[idx_lr] = cfg.lr * 0.5
        C.opt.lr = lr.view(-1, *([1] * 0))  # broadcast later
        orig_step = C.opt.step

        def step_with_lr(p, grads, _o=C.opt):
            _o.t += 1
            for k, (w, g) in enumerate(zip(p.tensors(), grads)):
                _o.m[k].mul_(_o.b1).add_(g, alpha=1 - _o.b1)
                _o.v[k].mul_(_o.b2).addcmul_(g, g, value=1 - _o.b2)
                shape = (-1,) + (1,) * (w.dim() - 1)
                mhat = _o.m[k] / (1 - _o.b1 ** _o.t).view(shape)
                vhat = _o.v[k] / (1 - _o.b2 ** _o.t).view(shape)
                w.sub_(lr.view(shape) * mhat / (vhat.sqrt() + _o.eps))
        C.opt.step = step_with_lr
        for _ in range(args.T - tf):
            C.step()
        for j, (i, cond, k) in enumerate(meta):
            rows.append(dict(fork=tf, parent=i, condition=cond, child=k, final=C.final_score(j), parent_final=parent_final[i]))
        save_csv(RESULTS / "runs.csv", rows)
        by = {c: np.array([r["final"] for r in rows if r["fork"] == tf and r["condition"] == c]) for c in CONDITIONS}
        print(f"fork {tf:6d} done [{time.time() - t0:.0f}s]  child mean final: " + "  ".join(f"{c}={by[c].mean():.0f}" for c in CONDITIONS))
    json.dump(dict(T=args.T, parent_final=parent_final), open(RESULTS / "parents.json", "w"))
    summarize(rows)


def summarize(rows=None):
    rows = rows if rows is not None else load_csv(RESULTS / "runs.csv")
    forks = sorted({r["fork"] for r in rows})
    pf = {r["parent"]: r["parent_final"] for r in rows}
    vals = np.array(sorted(pf.values()))
    lo, hi = np.quantile(vals, 1 / 3), np.quantile(vals, 2 / 3)
    bad = {p for p, v in pf.items() if v <= lo}; good = {p for p, v in pf.items() if v >= hi}
    good_median = float(np.median([pf[p] for p in good]))
    summary = []
    for tf in forks:
        for c in CONDITIONS:
            sel = [r for r in rows if r["fork"] == tf and r["condition"] == c]
            b = np.array([r["final"] for r in sel if r["parent"] in bad]); g = np.array([r["final"] for r in sel if r["parent"] in good])
            summary.append(dict(fork=tf, condition=c, rescue_bad=float(np.mean(b >= good_median)), mean_bad=float(b.mean()), mean_good=float(g.mean()),
                                harm_good=float(np.mean(g < good_median)), n_bad=len(b), n_good=len(g)))
    save_csv(RESULTS / "summary.csv", summary)

    def get(tf, c, key):
        return next(s[key] for s in summary if s["fork"] == tf and s["condition"] == c)

    gains = {c: {tf: get(tf, c, "rescue_bad") - get(tf, "control", "rescue_bad") for tf in forks} for c in CONDITIONS if c != "control"}
    # T is the parent training length (README: forks <= 0.6T), not the last fork time.
    pj = RESULTS / "parents.json"
    T = float(json.load(open(pj)).get("T", 30_000)) if pj.exists() else 30_000.0
    k1_hits = {c: max(v for tf, v in gs.items() if tf <= T * 0.6) for c, gs in gains.items()}
    k1 = any(v >= 0.2 for v in k1_hits.values())
    best_c = max(k1_hits, key=k1_hits.get)
    # K2: the window closes: rescue probability (best intervention) at the last fork is at least 0.2 below its best early value
    k2 = get(forks[0], best_c, "rescue_bad") - get(forks[-1], best_c, "rescue_bad") >= 0.2 or \
        max(get(tf, best_c, "rescue_bad") for tf in forks[:2]) - get(forks[-1], best_c, "rescue_bad") >= 0.2
    # K3: the best intervention does not harm good parents (harm rate within 0.15 of control at every fork)
    k3 = all(get(tf, best_c, "harm_good") - get(tf, "control", "harm_good") <= 0.15 for tf in forks)
    text = (f"bad/good split at {lo:.0f}/{hi:.0f}, good median {good_median:.0f}. "
            f"K1 (some intervention raises rescue prob of bad parents by >= 0.2 over control at a fork <= 0.6T; best = {best_c}, gains " +
            ", ".join(f"{tf}:{v:+.2f}" for tf, v in gains[best_c].items()) + f") = {k1}. "
            f"K2 (window closes: {best_c} rescue prob drops >= 0.2 from early to last fork) = {k2}. "
            f"K3 ({best_c} does not harm good parents) = {k3}.")
    verdict = True if (k1 and k2) else (False if not k1 else None)
    line = kill_test(verdict, text)
    (RESULTS / "verdict.txt").write_text(line + "\n")
    print("\nfork    condition    P(rescue | bad)  mean bad  mean good  P(harm | good)")
    for s in summary:
        print(f"{s['fork']:6d}  {s['condition']:11s}  {s['rescue_bad']:6.2f}          {s['mean_bad']:6.1f}    {s['mean_good']:6.1f}     {s['harm_good']:5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--parents", type=int, default=16)
    ap.add_argument("--children", type=int, default=3)
    ap.add_argument("--T", type=int, default=30_000)
    ap.add_argument("--eval-every", type=int, default=2_500)
    ap.add_argument("--forks", nargs="+", default=[5000, 10000, 15000, 20000, 25000])
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if args.smoke:
        args.parents, args.children, args.T, args.eval_every, args.forks = 6, 2, 4000, 1000, [1000, 3000]
    if args.summarize:
        summarize(); return
    run(args)


if __name__ == "__main__":
    main()
