"""R20: discovery time vs propagation time in sparse corridors. Tabular Q-learning with replay, target table, n-step, count bonus."""

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
DS = [10, 20, 30]
KS = [1, 200, 2000]
NS = [1, 5]
BETAS = [0.0, 0.1]
GAMMA = 0.99
EPS = 0.1
ALPHA = 0.1
BUF = 20_000
BATCH = 32
LEARN_START = 200
A = 2


def run(args):
    cells = [(D, K, n, b) for D in DS for K in KS for n in NS for b in BETAS]
    agents = [(ci, sd) for ci in range(len(cells)) for sd in range(args.seeds)]
    C = len(agents)
    D = np.array([cells[ci][0] for ci, _ in agents]); K = np.array([cells[ci][1] for ci, _ in agents])
    n_arr = np.array([cells[ci][2] for ci, _ in agents]); beta = np.array([cells[ci][3] for ci, _ in agents])
    Smax = max(DS) + 1
    rng = np.random.default_rng(args.seed_base)
    Q = np.zeros((C, Smax, A)); Qt = Q.copy()
    N = np.zeros((C, Smax))
    bs = np.zeros((C, BUF), np.int64); ba = np.zeros((C, BUF), np.int64); br = np.zeros((C, BUF)); bs2 = np.zeros((C, BUF), np.int64)
    bd = np.zeros((C, BUF)); bep = np.full((C, BUF), -1, np.int64); bt = np.full((C, BUF), -10 ** 9, np.int64)
    s = np.zeros(C, np.int64); ep = np.zeros(C, np.int64); ept = np.zeros(C, np.int64)
    ar = np.arange(C)
    T_d = np.full(C, -1); T_succ = np.full(C, -1); n_upd = np.zeros(C, np.int64)
    t0 = time.time()
    for t in range(1, args.steps + 1):
        q = Q[ar, s]
        tie = (q.max(-1) - q.min(-1)) < 1e-4
        greedy = np.where(tie, rng.integers(0, A, C), q.argmax(-1))
        a = np.where(rng.random(C) < EPS, rng.integers(0, A, C), greedy)
        s2 = np.clip(s + np.where(a == 1, 1, -1), 0, D)
        done = (s2 == D)
        r = done.astype(float)
        N[ar, s2] += 1
        r_store = r + beta / np.sqrt(N[ar, s2])
        first = done & (T_d < 0)
        T_d[first] = t
        ept += 1
        trunc = ept >= 4 * D
        slot = (t - 1) % BUF
        bs[:, slot] = s; ba[:, slot] = a; br[:, slot] = r_store; bs2[:, slot] = s2; bd[:, slot] = done; bep[:, slot] = ep; bt[:, slot] = t
        reset = done | trunc
        s = np.where(reset, 0, s2); ep = ep + reset; ept = np.where(reset, 0, ept)
        if t >= LEARN_START:
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
            y += disc * (1 - last_d) * Qt[ar[:, None], last_s2].max(-1)
            s_b = bs[ar[:, None], idx]; a_b = ba[ar[:, None], idx]
            delta = y - Q[ar[:, None], s_b, a_b]
            upd = np.zeros_like(Q)
            np.add.at(upd, (ar[:, None].repeat(BATCH, 1), s_b, a_b), ALPHA * delta)
            Q += upd
            n_upd += 1
            sync = (n_upd % K) == 0
            Qt[sync] = Q[sync]
        if t % args.eval_every == 0 or t == args.steps:
            # exact greedy rollout from 0 (deterministic corridor), ties -> fail
            pos = np.zeros(C, np.int64); reached = np.zeros(C, bool)
            for _ in range(4 * int(D.max())):
                qq = Q[ar, pos]
                tie = (qq.max(-1) - qq.min(-1)) < 1e-4
                mv = np.where(qq.argmax(-1) == 1, 1, -1)
                pos = np.where(reached | tie, pos, np.clip(pos + mv, 0, D))
                reached |= (pos == D)
            newly = reached & (T_succ < 0) & (T_d >= 0)
            T_succ[newly] = t
            if t % (args.eval_every * 20) == 0 or t == args.steps:
                print(f"t={t} [{time.time() - t0:.0f}s] discovered {np.mean(T_d >= 0):.2f}  solved {np.mean(T_succ >= 0):.2f}", flush=True)
    rows = []
    for j, (ci, sd) in enumerate(agents):
        Dj, Kj, nj, bj = cells[ci]
        rows.append(dict(D=Dj, K=Kj, n=nj, beta=bj, seed=sd, T_d=int(T_d[j]), T_succ=int(T_succ[j]),
                         T_p=int(T_succ[j] - T_d[j]) if (T_succ[j] >= 0 and T_d[j] >= 0) else -1,
                         discovered=bool(T_d[j] >= 0), solved=bool(T_succ[j] >= 0)))
    save_csv(RESULTS / "runs.csv", rows)
    summarize(rows)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(rows=None):
    rows = rows if rows is not None else load_csv(RESULTS / "runs.csv")
    cells = sorted({(r["D"], r["K"], r["n"], r["beta"]) for r in rows})
    summary = []
    for D, K, n, b in cells:
        sel = [r for r in rows if (r["D"], r["K"], r["n"], r["beta"]) == (D, K, n, b)]
        disc = [r for r in sel if r["discovered"]]; solved = [r for r in sel if r["solved"]]
        summary.append(dict(D=D, K=K, n=n, beta=b, n_disc=len(disc), n_solved=len(solved),
                            T_d=float(np.median([r["T_d"] for r in disc])) if disc else float("nan"),
                            T_p=float(np.median([r["T_p"] for r in solved])) if solved else float("nan"),
                            share=float(np.median([r["T_p"] / (r["T_d"] + r["T_p"]) for r in solved])) if solved else float("nan"),
                            law=float((np.ceil(D / n) - 1) * K + 1)))
    save_csv(RESULTS / "summary.csv", summary)

    def cell(D, K, n, b):
        return next(s for s in summary if (s["D"], s["K"], s["n"], s["beta"]) == (D, K, n, b))

    k1_cells = [cell(D, 2000, 1, 0.0) for D in DS]
    k1 = all(np.isfinite(c["T_p"]) and c["T_p"] > c["T_d"] for c in k1_cells)
    law_cells = [s for s in summary if s["beta"] == 0.0 and s["n_solved"] >= 4]
    rho = spearman([s["T_p"] for s in law_cells], [s["law"] for s in law_cells]) if len(law_cells) >= 4 else float("nan")
    k2 = rho >= 0.8
    k3_parts = []
    for D in DS:
        a, b = cell(D, 2000, 1, 0.0), cell(D, 2000, 1, 0.1)
        k3_parts.append(np.isfinite(a["T_d"]) and np.isfinite(b["T_d"]) and a["T_d"] >= 2 * b["T_d"])
    k3 = all(k3_parts)
    verdict = True if (k1 and k2) else (None if (k1 or k2) else False)
    text = (f"K1 (K=2000, n=1, no bonus: median T_p > median T_d for every D; T_p/T_d = " + ", ".join(f"D{c['D']}:{c['T_p']:.0f}/{c['T_d']:.0f}" for c in k1_cells) + f") = {k1}. "
            f"K2 (Spearman of median T_p on (ceil(D/n)-1)K+1 over {len(law_cells)} no-bonus cells = {rho:+.2f} >= 0.8) = {k2}. "
            f"K3 (bonus halves T_d at K=2000,n=1 for every D: {k3_parts}) = {k3}.")
    line = kill_test(verdict, text)
    (RESULTS / "verdict.txt").write_text(line + "\n")
    print("\n  D     K  n  beta  disc solved   T_d      T_p     share   law")
    for s in summary:
        print(f"{s['D']:3d}  {s['K']:4d}  {s['n']}  {s['beta']:.1f}   {s['n_disc']:2d}   {s['n_solved']:2d}   {s['T_d']:7.0f}  {s['T_p']:8.0f}   {s['share']:.2f}   {s['law']:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize(); return
    if args.smoke:
        args.steps, args.seeds = 20_000, 2
    RESULTS.mkdir(exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
