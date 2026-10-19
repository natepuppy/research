"""R17: minibatch correlation vs reuse. Samplers {iid, win4, win1, recent} x target period {100, 2500} x seeds."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import dqn_batched as D  # noqa: E402
from lib.dqn_sampler import SamplerDQN  # noqa: E402
from lib.io_utils import kill_test, load_csv, save_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SAMPLERS = ["iid", "win4", "win1", "recent"]
KS = [100, 2500]


def run(args):
    torch.set_num_threads(max(1, torch.get_num_threads()))
    cells = [(si, K) for si in range(len(SAMPLERS)) for K in KS]
    seeds = list(range(args.seeds))
    runs = [(si, K, sd) for (si, K) in cells for sd in seeds]
    R = len(runs)
    cfg = D.Config(total_steps=args.steps, eval_every=args.eval_every, eval_episodes=4)
    agent = SamplerDQN(cfg, seeds_init=[1000 + sd for (_, _, sd) in runs], seeds_env=[2000 + sd for (_, _, sd) in runs],
                       seeds_batch=[3000 + sd for (_, _, sd) in runs], samplers=[si for (si, _, _) in runs], target_every=[K for (_, K, _) in runs],
                       cos_every=args.eval_every)
    print(f"{R} runs x {args.steps} steps", flush=True)
    t0 = time.time()
    for t in range(1, args.steps + 1):
        agent.step()
        if t % 5000 == 0:
            msg = []
            for (si, K) in cells:
                idx = [j for j, (a, b, _) in enumerate(runs) if a == si and b == K]
                last = np.mean([agent.eval_log[j][-1][1] for j in idx if agent.eval_log[j]])
                msg.append(f"{SAMPLERS[si]}/K{K}: {last:5.1f}")
            print(f"step {t} [{time.time() - t0:.0f}s]  " + "  ".join(msg), flush=True)
    rows, per = [], []
    for j, (si, K, sd) in enumerate(runs):
        ev = agent.eval_log[j]; cs = dict(agent.cos_log[j])
        for step, score in ev:
            rows.append(dict(run=j, sampler=SAMPLERS[si], K=K, seed=sd, step=step, score=score, grad_cos=cs.get(step, float("nan"))))
        scores = [v for _, v in ev]
        per.append(dict(run=j, sampler=SAMPLERS[si], K=K, seed=sd, final=float(np.mean(scores[-3:])), auc=float(np.mean(scores)),
                        grad_cos=float(np.nanmean(list(cs.values()))) if cs else float("nan")))
    save_csv(RESULTS / "runs.csv", rows); save_csv(RESULTS / "per_run.csv", per)
    summarize(per)


def summarize(per=None):
    per = per if per is not None else load_csv(RESULTS / "per_run.csv")
    summary = []
    for K in KS:
        for s in SAMPLERS:
            sel = [r for r in per if r["K"] == K and r["sampler"] == s]
            summary.append(dict(K=K, sampler=s, final=float(np.mean([r["final"] for r in sel])), final_sd=float(np.std([r["final"] for r in sel])),
                                auc=float(np.mean([r["auc"] for r in sel])), auc_sd=float(np.std([r["auc"] for r in sel])),
                                grad_cos=float(np.nanmean([r["grad_cos"] for r in sel]))))
    save_csv(RESULTS / "summary.csv", summary)

    def auc(K, s):
        return {r["seed"]: r["auc"] for r in per if r["K"] == K and r["sampler"] == s}

    def gap(K, a, b):
        A, B = auc(K, a), auc(K, b)
        d = [A[k] - B[k] for k in sorted(A)]
        return float(np.mean(d)), int(sum(x > 0 for x in d)), len(d)

    g100, w100, n = gap(100, "iid", "win1")
    g2500, w2500, _ = gap(2500, "iid", "win1")
    k1 = g100 >= 20 and w100 >= 12
    k2 = g2500 <= 0.5 * g100
    r100, _, _ = gap(100, "win1", "recent"); r2500, _, _ = gap(2500, "win1", "recent")
    k3 = r100 >= 20 and r2500 >= 20
    verdict = True if (k1 and k2) else (None if (k1 or k2) else False)
    text = (f"K1 (K=100: AUC iid - win1 = {g100:+.1f} >= 20, iid wins {w100}/{n} >= 12) = {k1}. "
            f"K2 (gap at K=2500 = {g2500:+.1f} <= half of gap at K=100 = {0.5 * g100:+.1f}) = {k2}. "
            f"K3 (reuse control, win1 - recent: K100 {r100:+.1f}, K2500 {r2500:+.1f}, both >= 20) = {k3}.")
    line = kill_test(verdict, text)
    (RESULTS / "verdict.txt").write_text(line + "\n")
    print("\n   K   sampler   final (sd)       AUC (sd)      grad cos")
    for r in summary:
        print(f"{r['K']:5d}  {r['sampler']:7s}  {r['final']:6.1f} ({r['final_sd']:5.1f})   {r['auc']:6.1f} ({r['auc_sd']:5.1f})   {r['grad_cos']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--eval-every", type=int, default=1_000)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize(); return
    if args.smoke:
        args.steps, args.eval_every, args.seeds = 3000, 1000, 2
    RESULTS.mkdir(exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
