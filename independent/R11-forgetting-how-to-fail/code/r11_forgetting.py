"""R11: Forgetting how to fail.

Batched DQN on CartPole, three conditions (control / terminal reservoir / random reservoir), many seeds.
Tracks the greedy evaluation score, the failure-Q inflation on a fixed probe set (true value exactly 1),
and the terminal fraction of the replay buffer. Pre-registered kill test in README.md.
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
from lib.dqn_reservoir import ReservoirDQN  # noqa: E402
from lib.io_utils import kill_test, save_csv, load_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CONDITIONS = {0: "control", 1: "reservoir_term", 2: "reservoir_rand"}
HIGH, LOW = 175.0, 100.0        # a collapse: an eval <= LOW after the run has had an eval >= HIGH


def run(args):
    torch.set_num_threads(args.threads)
    cfg = D.Config(total_steps=args.T, eval_every=args.eval_every)
    kinds, si, se, sb = [], [], [], []
    for k in CONDITIONS:
        for s in range(args.seeds):
            kinds.append(k); si.append(s); se.append(500 + s); sb.append(900 + s)      # same seeds across conditions
    print(f"{len(kinds)} runs x {args.T} steps", flush=True)
    P = ReservoirDQN(cfg, si, se, sb, kinds)
    t0 = time.time()
    for step in range(1, args.T + 1):
        P.step()
        if step % cfg.eval_every == 0:
            msg = []
            for k, name in CONDITIONS.items():
                idx = [i for i in range(len(kinds)) if kinds[i] == k]
                msg.append(f"{name}: score={np.mean([P.eval_log[i][-1][1] for i in idx]):.0f} failQ={np.mean([P.probe_log[i][-1][1] for i in idx]):.1f} term%={100 * np.mean([P.probe_log[i][-1][2] for i in idx]):.1f}")
            print(f"step {step} [{time.time() - t0:.0f}s]  " + "  ".join(msg), flush=True)
    rows = []
    for i in range(len(kinds)):
        for (st, sc), (st2, fq, tf) in zip(P.eval_log[i], P.probe_log[i]):
            assert st == st2
            rows.append(dict(run=i, condition=CONDITIONS[kinds[i]], seed=si[i], step=st, score=sc, failq=fq, term_frac=tf))
    save_csv(RESULTS / "runs.csv", rows)
    summarize(rows)


def collapses(scores: list[float]) -> list[int]:
    """Indices of evals that count as a collapse."""
    out, armed = [], False
    for j, s in enumerate(scores):
        if s >= HIGH:
            armed = True
        elif armed and s <= LOW:
            out.append(j); armed = False
    return out


def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    if labels.all() or (~labels).all():
        return float("nan")
    pos, neg = scores[labels], scores[~labels]
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(rows=None):
    rows = rows if rows is not None else load_csv(RESULTS / "runs.csv")
    runs = sorted({r["run"] for r in rows})
    per_run = []
    pooled_scores, pooled_labels, pooled_fq, pooled_tf = [], [], [], []
    for i in runs:
        sel = sorted([r for r in rows if r["run"] == i], key=lambda r: r["step"])
        sc = [r["score"] for r in sel]; fq = [r["failq"] for r in sel]; tf = [r["term_frac"] for r in sel]
        col = collapses(sc)
        cond = sel[0]["condition"]
        per_run.append(dict(run=i, condition=cond, seed=sel[0]["seed"], n_collapse=len(col), final=float(np.mean(sc[-5:])),
                            auc=float(np.mean(sc)), max_failq=float(np.max(fq)), mean_failq=float(np.mean(fq[len(fq) // 4:])),
                            min_term_frac=float(np.min(tf))))
        if cond == "control":
            # K1 windows: at eval j (score >= HIGH, i.e. the run is currently good), does a collapse occur at j+1 or j+2?
            for j in range(len(sc) - 2):
                if sc[j] >= HIGH:
                    pooled_scores.append(fq[j]); pooled_labels.append(any(c in (j + 1, j + 2) for c in col))
            pooled_fq += fq[3:]; pooled_tf += tf[3:]
    save_csv(RESULTS / "per_run.csv", per_run)
    summary = []
    for cond in CONDITIONS.values():
        sel = [r for r in per_run if r["condition"] == cond]
        summary.append(dict(condition=cond, n=len(sel), collapse_rate=float(np.mean([r["n_collapse"] for r in sel])),
                            frac_runs_collapsing=float(np.mean([r["n_collapse"] > 0 for r in sel])),
                            final=float(np.mean([r["final"] for r in sel])), final_sd=float(np.std([r["final"] for r in sel])),
                            auc=float(np.mean([r["auc"] for r in sel])), max_failq=float(np.mean([r["max_failq"] for r in sel])),
                            mean_failq=float(np.mean([r["mean_failq"] for r in sel])), min_term_frac=float(np.mean([r["min_term_frac"] for r in sel]))))
    save_csv(RESULTS / "summary.csv", summary)

    def g(cond, key):
        return next(s[key] for s in summary if s["condition"] == cond)

    # K1: mechanism. Failure-Q inflation predicts an imminent collapse in control runs, and tracks the loss of terminals.
    au = auroc(pooled_scores, pooled_labels)
    rho_tf = spearman(pooled_fq, pooled_tf)
    k1 = (au >= 0.7) and (rho_tf <= -0.5)
    # K2: fix. Terminal reservoir cuts collapses by >= 40% and does not cost final score.
    ctrl = g("control", "collapse_rate")
    red_term = (1 - g("reservoir_term", "collapse_rate") / ctrl) if ctrl > 0 else float("nan")
    k2 = bool(red_term >= 0.4 and g("reservoir_term", "final") >= g("control", "final") - 5)
    # K3: specificity. The random reservoir gets less than half the reduction.
    red_rand = (1 - g("reservoir_rand", "collapse_rate") / ctrl) if ctrl > 0 else float("nan")
    k3 = bool(red_rand < 0.5 * red_term)
    verdict = True if (k1 and k2) else (None if (k1 or k2) else False)
    text = (f"K1 (control runs: AUROC of failure-Q for a collapse within the next 2 evals = {au:.2f} >= 0.7, n={len(pooled_scores)}, positives={int(np.sum(pooled_labels))}; "
            f"Spearman failure-Q vs buffer terminal fraction = {rho_tf:+.2f} <= -0.5) = {k1}. "
            f"K2 (terminal reservoir: collapse reduction {red_term:+.0%} >= 40%, final {g('reservoir_term', 'final'):.0f} vs control {g('control', 'final'):.0f}) = {k2}. "
            f"K3 (random reservoir reduction {red_rand:+.0%} < half of terminal's) = {k3}.")
    line = kill_test(verdict, text)
    (RESULTS / "verdict.txt").write_text(line + "\n")
    print("\ncondition        n  collapses/run  P(any collapse)  final (sd)     AUC   max failQ  mean failQ  min term%")
    for s in summary:
        print(f"{s['condition']:16s} {s['n']:2d}  {s['collapse_rate']:5.2f}          {s['frac_runs_collapsing']:4.2f}         {s['final']:5.1f} ({s['final_sd']:4.1f})  {s['auc']:5.1f}  {s['max_failq']:6.1f}     {s['mean_failq']:6.1f}     {100 * s['min_term_frac']:5.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=40_000)
    ap.add_argument("--eval-every", type=int, default=1_000)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize(); return
    if args.smoke:
        args.T, args.eval_every, args.seeds = 4000, 1000, 2
    RESULTS.mkdir(exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
