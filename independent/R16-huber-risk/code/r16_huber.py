"""R16: Huber-TD as a risk preference. Tabular fork MDPs with exact Huber fixed points; optional CartPole DQN check."""

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
GAMMA = 0.95
L = 3
M_SAFE = 1.0
GS = [0.1, 0.3]
PS = [0.05, 0.1, 0.3]
KAPPAS = [0.1, 0.3, 1.0, 3.0, float("inf")]
SKEWS = ["lottery", "hazard"]
ALPHA = 0.02
S = 2 * L + 2            # s0, safe chain L, risky chain L, terminal
TERM = S - 1


def lottery_params(skew, g, p):
    """Returns (outcomes, probs) for the risky terminal reward."""
    if skew == "lottery":
        M = (M_SAFE + g) / p
        return np.array([M, 0.0]), np.array([p, 1 - p])
    M = (M_SAFE + g) / p                      # same jump size as the lottery of equal g, p
    base = M_SAFE - g + p * M
    return np.array([base - M, base]), np.array([p, 1 - p])


def next_state(s, a):
    if s == 0:
        return 1 if a == 0 else L + 1
    if s == L or s == 2 * L:
        return TERM
    return s + 1


def huber_root(outcomes, probs, kappa):
    """q solving sum_i probs_i * clip(outcomes_i - q, -kappa, kappa) = 0 (monotone decreasing in q)."""
    if not np.isfinite(kappa):
        return float(np.dot(outcomes, probs))
    lo, hi = float(outcomes.min()), float(outcomes.max())
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f = float(np.dot(probs, np.clip(outcomes - mid, -kappa, kappa)))
        if f > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def huber_fixed_point(skew, g, p, kappa):
    """Iterate the Huber Bellman operator on the fork MDP to convergence; returns Q (S x 2)."""
    out_r, pr_r = lottery_params(skew, g, p)
    Q = np.zeros((S, 2))
    for _ in range(10_000):
        Qn = Q.copy()
        for s in range(S - 1):
            for a in range(2):
                s2 = next_state(s, a)
                if s == L:
                    outs, prs = np.array([M_SAFE]), np.array([1.0])
                elif s == 2 * L:
                    outs, prs = out_r, pr_r
                else:
                    outs, prs = np.array([GAMMA * Q[s2].max()]), np.array([1.0])
                Qn[s, a] = huber_root(outs, prs, kappa)
        if np.abs(Qn - Q).max() < 1e-12:
            return Qn
        Q = Qn
    return Q


SINGLE = False     # --single-action: chain states have one action (no max over duplicate noisy estimates)


def run_tabular(args):
    cells = [(sk, g, p, k) for sk in SKEWS for g in GS for p in PS for k in KAPPAS]
    seeds = list(range(args.seeds))
    agents = [(ci, sd) for ci in range(len(cells)) for sd in seeds]
    C = len(agents)
    kappa = np.array([cells[ci][3] for ci, _ in agents])
    outs = np.array([lottery_params(*cells[ci][:3])[0] for ci, _ in agents])        # (C, 2)
    prs = np.array([lottery_params(*cells[ci][:3])[1][0] for ci, _ in agents])      # P(first outcome)
    rng = np.random.default_rng(args.seed_base)
    Q = np.zeros((C, S, 2))
    state = rng.integers(0, TERM, C)
    ns_table = np.array([[next_state(s, a) if s < TERM else TERM for a in range(2)] for s in range(S)])
    acc = np.zeros((C, 2)); n_acc = 0
    t0 = time.time()
    ar = np.arange(C)
    chain = np.array([s != 0 for s in range(S)])            # chain states: single action if SINGLE
    for t in range(1, args.steps + 1):
        a = rng.integers(0, 2, C)
        if SINGLE:
            a = np.where(chain[state], 0, a)
        s2 = ns_table[state, a]
        u = rng.random(C)
        r = np.where(state == L, M_SAFE, 0.0)
        risky_last = state == 2 * L
        r = np.where(risky_last, np.where(u < prs, outs[:, 0], outs[:, 1]), r)
        done = s2 == TERM
        q2 = Q[ar, s2, 0] if SINGLE else Q[ar, s2].max(-1)       # s2 is never s0, so single-action max is Q[s2, 0]
        target = r + GAMMA * (~done) * q2
        delta = target - Q[ar, state, a]
        Q[ar, state, a] += ALPHA * np.clip(delta, -kappa, kappa)
        state = np.where(done, rng.integers(0, TERM, C), s2)
        if t > 0.75 * args.steps:
            acc += Q[:, 0]; n_acc += 1
        if t % 50_000 == 0:
            print(f"t={t} [{time.time() - t0:.0f}s]", flush=True)
    qbar = acc / n_acc
    rows = []
    for j, (ci, sd) in enumerate(agents):
        sk, g, p, k = cells[ci]
        true = huber_fixed_point(sk, g, p, float("inf"))
        pred = huber_fixed_point(sk, g, p, k)
        rows.append(dict(skew=sk, g=g, p=p, kappa=k, seed=sd, q_safe=float(qbar[j, 0]), q_risky=float(qbar[j, 1]),
                         true_risky=float(true[0, 1]), pred_risky=float(pred[0, 1]), bias=float(qbar[j, 1] - true[0, 1]),
                         pred_bias=float(pred[0, 1] - true[0, 1]), greedy_risky=bool(qbar[j, 1] > qbar[j, 0]),
                         optimal_risky=bool(sk == "lottery")))
    save_csv(RESULTS / "runs.csv", rows)
    summarize(rows)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(rows=None):
    rows = rows if rows is not None else load_csv(RESULTS / "runs.csv")
    cells = sorted({(r["skew"], r["g"], r["p"], r["kappa"]) for r in rows}, key=lambda c: (c[0], c[1], c[2], c[3]))
    summary = []
    for sk, g, p, k in cells:
        sel = [r for r in rows if (r["skew"], r["g"], r["p"], r["kappa"]) == (sk, g, p, k)]
        wrong = np.mean([r["greedy_risky"] != r["optimal_risky"] for r in sel])
        summary.append(dict(skew=sk, g=g, p=p, kappa=k, bias=float(np.mean([r["bias"] for r in sel])), bias_sd=float(np.std([r["bias"] for r in sel])),
                            pred_bias=sel[0]["pred_bias"], wrong_frac=float(wrong), pred_wrong=bool(abs(sel[0]["pred_bias"]) > g)))
    save_csv(RESULTS / "summary.csv", summary)
    fin = [s for s in summary if np.isfinite(s["kappa"])]
    # K1
    sign_ok = all((s["bias"] < 0) if s["skew"] == "lottery" else (s["bias"] > 0) for s in fin)
    rhos = []
    for sk in SKEWS:
        for g in GS:
            for p in PS:
                c = [s for s in summary if (s["skew"], s["g"], s["p"]) == (sk, g, p)]
                rhos.append(spearman([abs(s["bias"]) for s in c], [min(s["kappa"], 1e6) for s in c]))
    mono_ok = all(r <= -0.8 for r in rhos)
    k1 = sign_ok and mono_ok
    # K2
    x = np.array([s["pred_bias"] for s in summary]); y = np.array([s["bias"] for s in summary])
    r2 = 1 - np.sum((y - x) ** 2) / np.sum((y - y.mean()) ** 2)
    k2 = r2 >= 0.9
    # K3
    agree = np.mean([(s["wrong_frac"] > 0.5) == s["pred_wrong"] for s in summary])
    inf_wrong = any(s["wrong_frac"] > 0.5 for s in summary if not np.isfinite(s["kappa"]))
    k3 = agree >= 0.9 and not inf_wrong
    npass = sum([k1, k2, k3])
    verdict = True if npass == 3 else (None if npass == 2 else False)
    text = (f"K1 (sign = -skew in all finite-kappa cells: {sign_ok}; |bias| non-increasing in kappa, Spearman range [{min(rhos):+.2f}, {max(rhos):+.2f}] all <= -0.8: {mono_ok}) = {k1}. "
            f"K2 (identity-fit R^2 of learned bias on Huber fixed-point bias = {r2:.3f} >= 0.9) = {k2}. "
            f"K3 (wrong-policy cells agree with |pred bias| > g in {agree:.0%} of cells >= 90%; any squared-loss cell wrong = {inf_wrong}) = {k3}.")
    line = kill_test(verdict, text)
    (RESULTS / "verdict.txt").write_text(line + "\n")
    print("\nskew     g     p    kappa    bias (sd)        pred     wrong%  pred_wrong")
    for s in summary:
        print(f"{s['skew']:8s} {s['g']:.2f}  {s['p']:.2f}  {s['kappa']:5.1f}  {s['bias']:+.4f} ({s['bias_sd']:.4f})  {s['pred_bias']:+.4f}   {s['wrong_frac']:5.0%}   {s['pred_wrong']}")


# ----------------------------------------------------------------------------- DQN check

def run_dqn(args):
    import torch
    from lib import dqn_batched as D
    from lib.dqn_loss import LossDQN
    out = RESULTS / "dqn"; out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    runs = [(k, sd) for k in (0, 1) for sd in seeds]
    cfg = D.Config(total_steps=args.steps, eval_every=1000, eval_episodes=4)
    agent = LossDQN(cfg, [1000 + sd for _, sd in runs], [2000 + sd for _, sd in runs], [3000 + sd for _, sd in runs], loss_kind=[k for k, _ in runs])
    names = ["huber", "mse"]
    t0 = time.time()
    for t in range(1, args.steps + 1):
        agent.step()
        if t % 1000 == 0:
            agent.probe()
        if t % 5000 == 0:
            msg = []
            for k in (0, 1):
                idx = [j for j, (kk, _) in enumerate(runs) if kk == k]
                msg.append(f"{names[k]}: ret {np.mean([agent.eval_log[j][-1][1] for j in idx]):5.1f} failQ {np.mean([agent.probe_log[j][-1][1] for j in idx]):5.2f} clip {np.mean([agent.probe_log[j][-1][2] for j in idx]):.2f}")
            print(f"step {t} [{time.time() - t0:.0f}s]  " + "  ".join(msg), flush=True)
    rows, per = [], []
    for j, (k, sd) in enumerate(runs):
        for (step, score), (_, fq, cf) in zip(agent.eval_log[j], agent.probe_log[j]):
            rows.append(dict(loss=names[k], seed=sd, step=step, score=score, fail_q=fq, clip_frac=cf))
        sc = [v for _, v in agent.eval_log[j]]; fq = [v for _, v, _ in agent.probe_log[j]]
        per.append(dict(loss=names[k], seed=sd, final=float(np.mean(sc[-3:])), auc=float(np.mean(sc)), fail_q_final=float(np.mean(fq[-3:])),
                        fail_q_max=float(np.max(fq)), clip_frac=float(np.mean([c for _, _, c in agent.probe_log[j]]))))
    save_csv(out / "runs.csv", rows); save_csv(out / "per_run.csv", per)
    H = {r["seed"]: r for r in per if r["loss"] == "huber"}; Q = {r["seed"]: r for r in per if r["loss"] == "mse"}
    wins = sum(H[s]["fail_q_final"] > Q[s]["fail_q_final"] for s in seeds)
    text = (f"K4 (secondary; final failure-Q higher under Huber than squared loss in {wins}/{len(seeds)} paired seeds, >= 12): "
            f"huber failQ {np.mean([H[s]['fail_q_final'] for s in seeds]):.2f} (max {np.mean([H[s]['fail_q_max'] for s in seeds]):.2f}), "
            f"mse failQ {np.mean([Q[s]['fail_q_final'] for s in seeds]):.2f} (max {np.mean([Q[s]['fail_q_max'] for s in seeds]):.2f}); "
            f"final return huber {np.mean([H[s]['final'] for s in seeds]):.1f} vs mse {np.mean([Q[s]['final'] for s in seeds]):.1f}; "
            f"AUC {np.mean([H[s]['auc'] for s in seeds]):.1f} vs {np.mean([Q[s]['auc'] for s in seeds]):.1f}; "
            f"fraction of TD errors beyond kappa=1: {np.mean([H[s]['clip_frac'] for s in seeds]):.2f}. K4 = {wins >= 12}.")
    print(text)
    (out / "verdict.txt").write_text(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--dqn", action="store_true")
    ap.add_argument("--single-action", action="store_true", help="chain states have a single action (pre-registered follow-up)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    global SINGLE, RESULTS
    SINGLE = args.single_action
    if args.tag:
        RESULTS = RESULTS / args.tag
    if args.summarize:
        summarize(); return
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.dqn:
        if args.steps == 300_000:
            args.steps = 30_000
        if args.seeds == 8:
            args.seeds = 16
        if args.smoke:
            args.steps, args.seeds = 3000, 2
        run_dqn(args); return
    if args.smoke:
        args.steps, args.seeds = 20_000, 2
    run_tabular(args)


if __name__ == "__main__":
    main()
