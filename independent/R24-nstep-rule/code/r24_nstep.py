"""R24: n-step as a latency-bias trade-off. Tabular Q with uniform replay from a fixed eps-greedy dataset,
uncorrected n-step, hard target table. Vectorised over runs."""

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
L = 20
GAMMA = 0.99
EPSS = [0.01, 0.03, 0.05, 0.1]
GAPS = [0.1, 0.2, 0.3]
NS = [1, 2, 3, 5, 8, 12, 20]
KS = [200, 2000]
N_DATA = 20_000
BATCH = 32
ALPHA = 0.1
S = L + 1   # 0 = s0, 1..L corridor; terminal handled by done flag
A = 2       # s0: 0=A, 1=B ; corridor: 0=forward, 1=fall
QA = GAMMA ** L


def nstar(eps, gap):
    f = eps / 2; g = QA - gap
    best = 0
    for n in NS:
        c = int(np.ceil((L + 1) / n))
        if QA * (1 - f) ** ((L + 1) - c) > g:
            best = n
    return best


def make_dataset(eps, rng):
    f = eps / 2
    s_l, a_l, r_l, s2_l, d_l, ep_l, t_l, isB_l = [], [], [], [], [], [], [], []
    ep = 0; t = 0
    while t < N_DATA:
        # episode
        if rng.random() < 0.5:  # B
            s_l.append(0); a_l.append(1); r_l.append(0.0); s2_l.append(0); d_l.append(1.0); ep_l.append(ep); t_l.append(t); isB_l.append(1.0); t += 1
        else:
            s_l.append(0); a_l.append(0); r_l.append(0.0); s2_l.append(1); d_l.append(0.0); ep_l.append(ep); t_l.append(t); isB_l.append(0.0); t += 1
            s = 1
            while True:
                if rng.random() < f:
                    s_l.append(s); a_l.append(1); r_l.append(0.0); s2_l.append(0); d_l.append(1.0); ep_l.append(ep); t_l.append(t); isB_l.append(0.0); t += 1
                    break
                if s == L:
                    s_l.append(s); a_l.append(0); r_l.append(1.0); s2_l.append(0); d_l.append(1.0); ep_l.append(ep); t_l.append(t); isB_l.append(0.0); t += 1
                    break
                s_l.append(s); a_l.append(0); r_l.append(0.0); s2_l.append(s + 1); d_l.append(0.0); ep_l.append(ep); t_l.append(t); isB_l.append(0.0); t += 1
                s += 1
        ep += 1
    cut = N_DATA
    return (np.array(s_l[:cut]), np.array(a_l[:cut]), np.array(r_l[:cut]), np.array(s2_l[:cut]), np.array(d_l[:cut]),
            np.array(ep_l[:cut]), np.array(t_l[:cut]), np.array(isB_l[:cut]))


def run(args):
    cells = [(e, g, n, K) for e in EPSS for g in GAPS for n in NS for K in KS]
    runs = [(ci, sd) for ci in range(len(cells)) for sd in range(args.seeds)]
    C = len(runs)
    eps = np.array([cells[ci][0] for ci, _ in runs]); gap = np.array([cells[ci][1] for ci, _ in runs])
    n_arr = np.array([cells[ci][2] for ci, _ in runs]); K = np.array([cells[ci][3] for ci, _ in runs])
    gB = QA - gap
    # datasets per (eps, seed)
    keys = sorted({(cells[ci][0], sd) for ci, sd in runs})
    data = {k: make_dataset(k[0], np.random.default_rng(10_000 + 97 * int(k[0] * 1000) + k[1])) for k in keys}
    ds_idx = np.array([keys.index((cells[ci][0], sd)) for ci, sd in runs])
    D = len(keys)
    bs = np.stack([data[k][0] for k in keys]); ba = np.stack([data[k][1] for k in keys]); br = np.stack([data[k][2] for k in keys])
    bs2 = np.stack([data[k][3] for k in keys]); bd = np.stack([data[k][4] for k in keys]); bep = np.stack([data[k][5] for k in keys])
    bt = np.stack([data[k][6] for k in keys]); bB = np.stack([data[k][7] for k in keys])
    rng = np.random.default_rng(args.seed_base)
    Q = np.zeros((C, S, A)); Qt = Q.copy()
    ar = np.arange(C); di = ds_idx[:, None]
    T = np.full(C, -1); t0 = time.time()
    tail_start = int(0.8 * args.updates); tail_n = np.zeros(C); tail_c = np.zeros(C)
    nmax = int(n_arr.max())
    for t in range(1, args.updates + 1):
        idx = rng.integers(0, N_DATA, (C, BATCH))
        base_t = bt[di, idx]; base_ep = bep[di, idx]
        y = np.zeros((C, BATCH)); disc = np.ones((C, BATCH)); alive = np.ones((C, BATCH), bool)
        last_s2 = bs2[di, idx]; last_d = bd[di, idx]
        for j in range(nmax):
            pos = np.minimum(idx + j, N_DATA - 1)
            ok = alive & (j < n_arr[:, None]) & (bt[di, pos] == base_t + j) & (bep[di, pos] == base_ep)
            rew = br[di, pos] + bB[di, pos] * gB[:, None]
            y += np.where(ok, disc * rew, 0.0)
            last_s2 = np.where(ok, bs2[di, pos], last_s2); last_d = np.where(ok, bd[di, pos], last_d)
            disc = np.where(ok, disc * GAMMA, disc)
            alive = ok & (bd[di, pos] < 0.5)
        y += disc * (1 - last_d) * Qt[ar[:, None], last_s2].max(-1)
        s_b = bs[di, idx]; a_b = ba[di, idx]
        delta = y - Q[ar[:, None], s_b, a_b]
        upd = np.zeros_like(Q)
        np.add.at(upd, (np.repeat(ar[:, None], BATCH, 1), s_b, a_b), ALPHA * delta)
        Q += upd
        sync = (t % K) == 0
        Qt[sync] = Q[sync]
        if t % args.eval_every == 0 or t == args.updates:
            # exact greedy rollout: A at s0 and forward at every corridor state (strict >)
            okA = Q[:, 0, 0] > Q[:, 0, 1]
            fwd = np.all(Q[:, 1:, 0] > Q[:, 1:, 1], axis=1)
            correct = okA & fwd
            newly = correct & (T < 0); T[newly] = t
            if t > tail_start:
                tail_n += 1; tail_c += correct
            if t % (args.eval_every * 50) == 0 or t == args.updates:
                print(f"t={t} [{time.time() - t0:.0f}s] correct now {correct.mean():.2f}", flush=True)
    rows = []
    for j, (ci, sd) in enumerate(runs):
        e, g, n, k = cells[ci]
        rows.append(dict(eps=e, gap=g, n=n, K=k, seed=sd, T=int(T[j]), final_correct=bool(tail_c[j] / max(tail_n[j], 1) >= 0.8), tail_frac=float(tail_c[j] / max(tail_n[j], 1)),
                         QA_learned=float(Q[j, 0, 0]), QB_learned=float(Q[j, 0, 1])))
    save_csv(RESULTS / "runs.csv", rows)
    summarize(rows)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(rows=None):
    rows = rows if rows is not None else load_csv(RESULTS / "runs.csv")
    cells = sorted({(r["eps"], r["gap"], r["n"], r["K"]) for r in rows})
    summ = []
    for e, g, n, k in cells:
        sel = [r for r in rows if (r["eps"], r["gap"], r["n"], r["K"]) == (e, g, n, k)]
        fc = sum(r["final_correct"] for r in sel)
        Ts = [r["T"] for r in sel if r["T"] > 0]
        summ.append(dict(eps=e, gap=g, n=n, K=k, n_seeds=len(sel), final_correct=fc, T=float(np.median(Ts)) if Ts else float("nan"),
                         QA=float(np.median([r["QA_learned"] for r in sel])), law=float((np.ceil((L + 1) / n) - 1) * k + 1),
                         bias_pred=float(QA * (1 - e / 2) ** ((L + 1) - np.ceil((L + 1) / n)))))
    save_csv(RESULTS / "summary.csv", summ)
    ns_rows = []
    for e in EPSS:
        for g in GAPS:
            star = nstar(e, g)
            rec = {}
            for k in KS:
                good = [s["n"] for s in summ if (s["eps"], s["gap"], s["K"]) == (e, g, k) and s["final_correct"] >= 6]
                rec[k] = max(good) if good else 0
            ns_rows.append(dict(eps=e, gap=g, nstar=star, nhat_K200=rec[200], nhat_K2000=rec[2000]))
    save_csv(RESULTS / "nstar.csv", ns_rows)

    def neighbour_ok(nh, star):
        grid = [0] + NS
        i = grid.index(star)
        allowed = {grid[max(i - 1, 0)], grid[i], grid[min(i + 1, len(grid) - 1)]}
        return nh in allowed

    k1_n = sum(neighbour_ok(r["nhat_K2000"], r["nstar"]) for r in ns_rows); k1 = k1_n >= 9
    good = [s for s in summ if s["final_correct"] >= 6 and np.isfinite(s["T"])]
    rho = spearman([s["T"] for s in good], [s["law"] for s in good]) if len(good) >= 4 else float("nan")
    k2 = rho >= 0.85
    k3_n = sum(r["nhat_K200"] == r["nhat_K2000"] for r in ns_rows); k3 = k3_n >= 9
    v = True if (k1 and k2) else (None if (k1 or k2) else False)
    text = (f"K1 (largest safe n at K=2000 within one grid step of n*: {k1_n}/12; (eps, gap, n*, nhat) = " + ", ".join(f"({r['eps']},{r['gap']},{r['nstar']},{r['nhat_K2000']})" for r in ns_rows) + f") = {k1}. "
            f"K2 (Spearman of median T on (ceil(21/n)-1)K+1 over {len(good)} correct cells = {rho:+.2f} >= 0.85) = {k2}. "
            f"K3 (nhat equal at K=200 and K=2000: {k3_n}/12) = {k3}.")
    line = kill_test(v, text)
    (RESULTS / "verdict.txt").write_text(line + "\n")
    print("\n eps    gap   n     K   correct   T(med)    law    QA_learned  bias_pred")
    for s in summ:
        print(f" {s['eps']:.2f}  {s['gap']:.1f}  {s['n']:2d}  {s['K']:4d}    {s['final_correct']}/8   {s['T']:7.0f}  {s['law']:6.0f}     {s['QA']:.3f}      {s['bias_pred']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=60_000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize(); return
    if args.smoke:
        args.updates, args.seeds = 8000, 2
    RESULTS.mkdir(exist_ok=True)
    print("n* table:", {(e, g): nstar(e, g) for e in EPSS for g in GAPS}, flush=True)
    run(args)


if __name__ == "__main__":
    main()
