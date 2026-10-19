"""R23: the C51 value support as a horizon. Exact tabular C51 vs clipped value iteration on patient/myopic forks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io_utils import kill_test, save_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GAMMA = 0.99
FORKS = [(5, 12), (10, 25), (20, 50), (40, 100)]
VMAXS = [3, 5, 8, 10, 15, 20, 30, 50, 100, 200]
N_ATOMS = 51
ALPHA = 0.1


def build(M, P):
    """States: 0 = s0; 1..M myopic corridor; M+1..M+P patient corridor. Returns per (state, action): reward, next, terminal.
    s0 has 2 actions; corridor states have 1 action (duplicated so arrays are rectangular)."""
    S = 1 + M + P
    nxt = np.zeros((S, 2), int); rew = np.zeros((S, 2)); term = np.zeros((S, 2), bool)
    rew[0, 0] = 1.0; nxt[0, 0] = 1 if M >= 1 else -1; term[0, 0] = M == 0           # myopic
    rew[0, 1] = 0.0; nxt[0, 1] = M + 1; term[0, 1] = False                           # patient
    for i in range(1, M + 1):
        for a in range(2):
            rew[i, a] = 1.0; nxt[i, a] = i + 1 if i < M else 0; term[i, a] = i == M
    for j in range(M + 1, M + P + 1):
        for a in range(2):
            rew[j, a] = 1.0; nxt[j, a] = j + 1 if j < M + P else 0; term[j, a] = j == M + P
    return S, rew, nxt, term


def clipped_vi(S, rew, nxt, term, vmax, iters=20000):
    Q = np.zeros((S, 2))
    for _ in range(iters):
        V = Q.max(1)
        Qn = np.minimum(vmax, rew + GAMMA * np.where(term, 0.0, V[nxt]))
        if np.abs(Qn - Q).max() < 1e-12:
            Q = Qn; break
        Q = Qn
    return Q


def scalar_clipped(S, rew, nxt, term, vmax, sweeps):
    Q = np.zeros((S, 2))
    for _ in range(sweeps):
        V = Q.max(1)
        y = np.minimum(vmax, rew + GAMMA * np.where(term, 0.0, V[nxt]))
        Q += ALPHA * (y - Q)
    return Q


def c51(S, rew, nxt, term, vmax, sweeps):
    z = np.linspace(-vmax, vmax, N_ATOMS); dz = z[1] - z[0]
    p = np.zeros((S, 2, N_ATOMS)); p[:, :, N_ATOMS // 2] = 1.0  # all mass at 0
    for _ in range(sweeps):
        means = (p * z).sum(-1)                      # (S, 2)
        astar = means.argmax(1)                      # greedy next action per state
        # target atoms per (s, a): Tz = clip(r + gamma z) (non-terminal) or clip(r) at all atoms (terminal)
        Tz = np.clip(rew[:, :, None] + GAMMA * z[None, None, :] * (~term)[:, :, None], -vmax, vmax)  # (S, 2, A)
        pn = p[nxt, astar[nxt]]                      # (S, 2, A) successor distribution under greedy action
        pn = np.where(term[:, :, None], np.eye(N_ATOMS)[N_ATOMS // 2][None, None, :], pn)  # terminal: any dist, atoms all map to clip(r)
        b = (Tz - (-vmax)) / dz; l = np.floor(b).astype(int); u = np.ceil(b).astype(int)
        l = np.clip(l, 0, N_ATOMS - 1); u = np.clip(u, 0, N_ATOMS - 1)
        m = np.zeros_like(p)
        wl = pn * (u - b); wu = pn * (b - l)
        same = (l == u)
        wl = np.where(same, pn, wl); wu = np.where(same, 0.0, wu)
        idx_s = np.arange(S)[:, None, None]; idx_a = np.arange(2)[None, :, None]
        np.add.at(m, (np.broadcast_to(idx_s, l.shape), np.broadcast_to(idx_a, l.shape), l), wl)
        np.add.at(m, (np.broadcast_to(idx_s, u.shape), np.broadcast_to(idx_a, u.shape), u), wu)
        p += ALPHA * (m - p)
    return (p * z).sum(-1)


def run(sweeps, log):
    rows = []
    for M, P in FORKS:
        S, rew, nxt, term = build(M, P)
        VM = (1 - GAMMA ** M) / (1 - GAMMA); VP = (1 - GAMMA ** P) / (1 - GAMMA)
        q_true = (1 + GAMMA * VM, GAMMA * VP)
        for vmax in VMAXS:
            Qc = clipped_vi(S, rew, nxt, term, vmax)
            dpred = Qc[0, 1] - Qc[0, 0]
            pred = "patient" if dpred > 0.05 else ("myopic" if dpred < -0.05 else "indifferent")
            Qd = c51(S, rew, nxt, term, vmax, sweeps)
            dc = Qd[0, 1] - Qd[0, 0]
            choice = "patient" if dc > 0.05 else ("myopic" if dc < -0.05 else "indifferent")
            Qs = scalar_clipped(S, rew, nxt, term, vmax, sweeps)
            ds = Qs[0, 1] - Qs[0, 0]
            schoice = "patient" if ds > 0.05 else ("myopic" if ds < -0.05 else "indifferent")
            both_sat = vmax < q_true[0]; none_sat = vmax > q_true[1]
            rows.append(dict(M=M, P=P, vmax=vmax, q_true_myopic=q_true[0], q_true_patient=q_true[1],
                             qc_myopic=Qc[0, 0], qc_patient=Qc[0, 1], pred=pred,
                             c51_myopic=Qd[0, 0], c51_patient=Qd[0, 1], c51_choice=choice,
                             scalar_myopic=Qs[0, 0], scalar_patient=Qs[0, 1], scalar_choice=schoice,
                             both_saturate=bool(both_sat), none_saturate=bool(none_sat)))
            log(f"M={M:3d} P={P:3d} vmax={vmax:4d}  true ({q_true[0]:6.2f},{q_true[1]:6.2f})  clippedVI ({Qc[0,0]:6.2f},{Qc[0,1]:6.2f}) -> {pred:11s}  C51 ({Qd[0,0]:6.2f},{Qd[0,1]:6.2f}) -> {choice:11s} scalar -> {schoice}")
    save_csv(RESULTS / "cells.csv", rows)
    verdict(rows, log)


def verdict(rows, log):
    def agree(r, learned):
        if r["pred"] == "indifferent":
            return learned == "indifferent"
        return learned == r["pred"]
    k1_n = sum(agree(r, r["c51_choice"]) for r in rows); k1 = k1_n / len(rows) >= 0.95
    pairs = []
    for r in rows:
        for a in ("myopic", "patient"):
            pairs.append(abs(r[f"c51_{a}"] - r[f"qc_{a}"]) <= 0.05 * r["vmax"])
    k2 = np.mean(pairs) >= 0.95
    k3_parts = [(r["c51_choice"] == "myopic") for r in rows if r["both_saturate"]] + [(r["c51_choice"] == "patient") for r in rows if r["none_saturate"]]
    k3 = all(k3_parts)
    k4_n = sum(r["c51_choice"] == r["scalar_choice"] for r in rows); k4 = k4_n == len(rows)
    n_ok = sum([k1, k2, k3])
    v = True if n_ok == 3 else (None if n_ok == 2 else False)
    text = (f"K1 (C51 choice matches clipped-VI prediction in {k1_n}/{len(rows)} cells) = {k1}. "
            f"K2 (C51 mean within 5% of V_max of the clipped fixed point in {int(np.sum(pairs))}/{len(pairs)} (cell, action) pairs) = {k2}. "
            f"K3 (myopic in all {sum(r['both_saturate'] for r in rows)} both-saturate cells and patient in all {sum(r['none_saturate'] for r in rows)} none-saturate cells: {sum(k3_parts)}/{len(k3_parts)}) = {k3}. "
            f"K4 (scalar clipped-Q agrees with C51 in {k4_n}/{len(rows)}) = {k4}.")
    line = kill_test(v, text)
    (RESULTS / "verdict.txt").write_text(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", type=int, default=20000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.sweeps = 300
    RESULTS.mkdir(exist_ok=True)
    run(args.sweeps, lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
