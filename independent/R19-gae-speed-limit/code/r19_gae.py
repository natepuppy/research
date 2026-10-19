"""R19: iteration latency of GAE(lambda) credit assignment in an iterated PPO-like loop.

Tabular softmax policy (one SGD step per batch on normalised GAE advantages), exact tabular critic
fit to lambda-return targets each iteration, segment length T with bootstrap at segment boundaries.
Vectorised over runs = (lambda, T, seed) for each corridor depth D.
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
DS = [8, 16, 32, 64, 128]
LAMS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]
TS = [0, 16]  # 0 = infinity
N_EP = 32
STEP = 0.5
CRITIC_ON = True  # post-hoc switch: False freezes V at 0 (pure Monte-Carlo-with-decay policy gradient)
V_THRESH = 0.25
PI_THRESH = 0.75


def success_prob(pi_fwd, pA, cap):
    """Exact P(reach end of corridor A within cap steps) for per-state forward probs pi_fwd[R, D-1]."""
    R, Dm1 = pi_fwd.shape
    dist = np.zeros((R, Dm1 + 1)); dist[:, 0] = 1.0  # pos index 0 == corridor pos 1; index Dm1 == end
    for _ in range(cap - 1):
        move = dist[:, :Dm1] * pi_fwd
        stay = dist[:, :Dm1] * (1 - pi_fwd)
        new = np.zeros_like(dist)
        new[:, :Dm1] += stay
        new[:, 1:] += move
        new[:, Dm1] += dist[:, Dm1]
        dist = new
    return pA * dist[:, Dm1]


def run_depth(D, seeds, max_iter, seed_base, log):
    combos = [(lam, T, sd) for lam in LAMS for T in TS for sd in range(seeds)]
    R = len(combos)
    lam = np.array([c[0] for c in combos]); Tseg = np.array([c[1] if c[1] > 0 else 10 ** 9 for c in combos])
    S = 1 + 2 * (D - 1)  # s0, A pos 1..D-1, B pos 1..D-1
    cap = 3 * D
    rng = np.random.default_rng(seed_base + D)
    logits = np.zeros((R, S, 2)); V = np.zeros((R, S))
    ar = np.arange(R)
    t_V = np.full(R, -1); t_pi = np.full(R, -1); final_p = np.zeros(R)
    t0 = time.time()
    for it in range(1, max_iter + 1):
        pi = np.exp(logits - logits.max(-1, keepdims=True)); pi /= pi.sum(-1, keepdims=True)
        # ---- rollout N_EP episodes per run
        s = np.zeros((R, N_EP), np.int64); alive = np.ones((R, N_EP), bool)
        S_hist = np.zeros((cap, R, N_EP), np.int64); A_hist = np.zeros((cap, R, N_EP), np.int64)
        Rw = np.zeros((cap, R, N_EP)); Done = np.zeros((cap, R, N_EP), bool); Valid = np.zeros((cap, R, N_EP), bool)
        S2 = np.zeros((cap, R, N_EP), np.int64)
        for k in range(cap):
            p1 = pi[ar[:, None], s, 1]
            a = (rng.random((R, N_EP)) < p1).astype(np.int64)
            # transitions
            at_s0 = s == 0
            inA = (s >= 1) & (s <= D - 1)
            pos = np.where(inA, s, s - (D - 1))  # corridor position 1..D-1 (for s>0)
            fwd = a == 1
            newpos = np.where(fwd, pos + 1, pos)
            reach = fwd & (pos == D - 1) & ~at_s0
            s_next = np.where(at_s0, np.where(a == 0, 1, D), np.where(inA, newpos, newpos + (D - 1)))
            s_next = np.where(reach, 0, s_next)  # terminal -> placeholder (masked by Done)
            r = (reach & inA).astype(float)
            S_hist[k] = s; A_hist[k] = a; Rw[k] = r; Done[k] = reach & alive; Valid[k] = alive; S2[k] = s_next
            alive = alive & ~reach
            s = np.where(alive, s_next, 0)
        # ---- GAE / lambda returns backward with segment boundaries
        adv = np.zeros((cap, R, N_EP)); G = np.zeros((cap, R, N_EP))
        last = np.zeros((R, N_EP))
        Vs = V[ar[None, :, None], S_hist]  # (cap, R, N_EP)
        Vn = V[ar[None, :, None], S2]
        for k in range(cap - 1, -1, -1):
            boundary = ((k + 1) % Tseg[:, None] == 0) | (k == cap - 1)
            nxt = np.where(Done[k], 0.0, Vn[k])
            delta = Rw[k] + nxt - Vs[k]
            carry = np.where(Done[k] | boundary, 0.0, lam[:, None] * last)
            adv[k] = delta + carry
            G[k] = adv[k] + Vs[k]
            last = adv[k]
        # ---- exact critic fit: per-run per-state mean of targets over valid steps
        flatS = S_hist.transpose(1, 0, 2).reshape(R, -1); flatG = G.transpose(1, 0, 2).reshape(R, -1); flatV = Valid.transpose(1, 0, 2).reshape(R, -1)
        sums = np.zeros((R, S)); cnt = np.zeros((R, S))
        rr = np.repeat(ar[:, None], flatS.shape[1], 1)
        np.add.at(sums, (rr, flatS), np.where(flatV, flatG, 0.0)); np.add.at(cnt, (rr, flatS), flatV.astype(float))
        if CRITIC_ON:
            V = np.where(cnt > 0, sums / np.maximum(cnt, 1), V)
        # ---- policy step on normalised advantages (old V used for adv), mean per episode
        flatA = adv.transpose(1, 0, 2).reshape(R, -1); flatAct = A_hist.transpose(1, 0, 2).reshape(R, -1)
        m = np.where(flatV, flatA, 0).sum(1) / flatV.sum(1)
        sd = np.sqrt(np.where(flatV, (flatA - m[:, None]) ** 2, 0).sum(1) / flatV.sum(1)) + 1e-8
        An = np.where(flatV, (flatA - m[:, None]) / sd[:, None], 0.0)
        grad = np.zeros((R, S, 2))
        pis = pi[rr, flatS]  # (R, M, 2)
        onehot = np.eye(2)[flatAct]
        np.add.at(grad, (rr, flatS), An[..., None] * (onehot - pis))
        logits += STEP * grad / N_EP
        # ---- measurements
        pi_new = np.exp(logits - logits.max(-1, keepdims=True)); pi_new /= pi_new.sum(-1, keepdims=True)
        psucc = success_prob(pi_new[:, 1:D, 1], pi_new[:, 0, 0], cap)
        final_p = psucc
        hitV = (V[:, 0] >= V_THRESH) & (t_V < 0); t_V[hitV] = it
        hitP = (psucc >= PI_THRESH) & (t_pi < 0); t_pi[hitP] = it
        if it % 50 == 0 or it == 1:
            log(f"D={D} it={it} [{time.time() - t0:.0f}s] V-hit {np.mean(t_V >= 0):.2f} pi-hit {np.mean(t_pi >= 0):.2f} mean psucc {psucc.mean():.2f}")
    rows = []
    for j, (l, T, sd) in enumerate(combos):
        rows.append(dict(D=D, lam=l, T=T, seed=sd, t_V=int(t_V[j]) if t_V[j] > 0 else max_iter, t_pi=int(t_pi[j]) if t_pi[j] > 0 else max_iter,
                         V_hit=bool(t_V[j] > 0), pi_hit=bool(t_pi[j] > 0), final_psucc=float(final_p[j])))
    return rows


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def ols(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    pred = X1 @ b
    r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return b, r2


def summarize(rows=None):
    rows = rows if rows is not None else load_csv(RESULTS / "runs.csv")
    cells = sorted({(r["D"], r["lam"], r["T"]) for r in rows})
    summary = []
    for D, l, T in cells:
        sel = [r for r in rows if (r["D"], r["lam"], r["T"]) == (D, l, T)]
        summary.append(dict(D=D, lam=l, T=T, n=len(sel), t_V=float(np.median([r["t_V"] for r in sel])), t_pi=float(np.median([r["t_pi"] for r in sel])),
                            V_hit=float(np.mean([r["V_hit"] for r in sel])), pi_hit=float(np.mean([r["pi_hit"] for r in sel])),
                            final_psucc=float(np.mean([r["final_psucc"] for r in sel])), pred=float(D * (1 - l)) if l < 1 else 0.0))
    save_csv(RESULTS / "summary.csv", summary)

    def cell(D, l, T):
        return next(s for s in summary if (s["D"], s["lam"], s["T"]) == (D, l, T))

    k1_cells = [s for s in summary if s["T"] == 0 and s["lam"] < 1 and s["pred"] >= 4]
    X = np.array([[np.log(s["D"]), np.log(1 - s["lam"])] for s in k1_cells]); y = np.log([s["t_V"] for s in k1_cells])
    b1, r2_1 = ols(X, y)
    k1 = (0.7 <= b1[1] <= 1.3) and (0.7 <= b1[2] <= 1.3) and r2_1 >= 0.9
    k2_cells = [cell(D, 1.0, 0) for D in DS]
    b2, _ = ols(np.log([s["D"] for s in k2_cells])[:, None], np.log([s["t_V"] for s in k2_cells]))
    k2 = b2[1] < 0.3
    k3_cells = [cell(D, 1.0, 16) for D in DS if D >= 32]
    b3, _ = ols(np.log([s["D"] for s in k3_cells])[:, None], np.log([s["t_V"] for s in k3_cells]))
    ratios = [cell(D, 1.0, 16)["t_V"] / max(cell(D, 0.95, 0)["t_V"], 1e-9) for D in DS if D >= 32]
    k3 = (0.7 <= b3[1] <= 1.3) and all(1 / 1.5 <= r <= 1.5 for r in ratios)
    fin = [s for s in summary if s["V_hit"] > 0.5 and s["pi_hit"] > 0.5]
    rho = spearman([s["t_pi"] for s in fin], [s["t_V"] for s in fin])
    k4 = rho >= 0.8
    n_ok = sum([k1, k2, k4])
    verdict = True if n_ok == 3 else (None if n_ok == 2 else False)
    text = (f"K1 (T=inf, lam<1, D(1-lam)>=4, {len(k1_cells)} cells: log t_V slopes on log D = {b1[1]:.2f}, on log(1-lam) = {b1[2]:.2f}, R2 = {r2_1:.3f}) = {k1}. "
            f"K2 (lam=1, T=inf: slope on log D = {b2[1]:.2f} < 0.3) = {k2}. "
            f"K3 (T=16, lam=1, D>=32: slope {b3[1]:.2f}; ratios to lam=0.95,T=inf = {[round(r, 2) for r in ratios]}) = {k3}. "
            f"K4 (Spearman(t_pi, t_V) over {len(fin)} cells = {rho:+.2f} >= 0.8) = {k4}.")
    line = kill_test(verdict, text)
    (RESULTS / "verdict.txt").write_text(line + "\n")
    print("\n   D   lam   T    t_V    t_pi  Vhit pihit  final_p   pred")
    for s in summary:
        print(f"{s['D']:4d}  {s['lam']:.2f}  {s['T']:2d}  {s['t_V']:5.0f}  {s['t_pi']:6.0f}  {s['V_hit']:.2f}  {s['pi_hit']:.2f}   {s['final_psucc']:.2f}  {s['pred']:6.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=600)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize(); return
    ds = DS
    if args.smoke:
        args.seeds, args.max_iter, ds = 2, 150, [8, 32]
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for D in ds:
        rows += run_depth(D, args.seeds, args.max_iter, args.seed_base, lambda m: print(m, flush=True))
        save_csv(RESULTS / "runs.csv", rows)
    if not args.smoke:
        summarize(rows)
    else:
        try:
            summarize(rows)
        except Exception as e:  # smoke grid lacks some cells
            print("summary skipped in smoke:", e)


if __name__ == "__main__":
    main()
