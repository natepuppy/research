"""R13: Does the distribution-mismatch coefficient predict how fast policy gradient learns?

Tabular softmax policy gradient with exact gradients on gridworld mazes. The agent is trained from a start
distribution mu and evaluated from a fixed start rho. Agarwal et al. (2021) bound the iteration complexity by
D_mu = || d^{pi*}_rho / mu ||_inf. We compute D_mu exactly, together with a variant that compares against the
*initial policy's* visitation under mu, D_0 = || d^{pi*}_rho / d^{pi_0}_mu ||_inf, and ask which predicts the
number of iterations until V^pi(rho) >= 0.9 V*(rho). Pre-registered kill test in README.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io_utils import kill_test, save_csv, load_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GAMMA = 0.95
SLIP = 0.1
ETA = 2.0            # softmax PG step size (exact gradient)
MAX_ITERS = 50_000
TARGET_FRAC = 0.9
A_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

LAYOUTS = {
    "open": [
        "..........",
        "..........",
        "..........",
        "..........",
        "..........",
        "..........",
        "..........",
        "..........",
        "..........",
        ".........G"],
    "wall": [
        "..........",
        "..........",
        "..........",
        "..........",
        "########..",
        "..........",
        "..........",
        "..........",
        "..........",
        ".........G"],
    "snake": [
        "..........",
        "#########.",
        "..........",
        ".#########",
        "..........",
        "#########.",
        "..........",
        ".#########",
        "..........",
        ".........G"],
    "rooms": [
        "....#.....",
        "....#.....",
        "..........",
        "....#.....",
        "##.####.##",
        "....#.....",
        "....#.....",
        "..........",
        "....#.....",
        "....#....G"],
}
START = (0, 0)


class Grid:
    def __init__(self, layout: list[str]):
        self.h, self.w = len(layout), len(layout[0])
        self.cells = [(r, c) for r in range(self.h) for c in range(self.w) if layout[r][c] != "#"]
        self.idx = {rc: i for i, rc in enumerate(self.cells)}
        self.S = len(self.cells)
        self.goal = self.idx[next((r, c) for r in range(self.h) for c in range(self.w) if layout[r][c] == "G")]
        self.start = self.idx[START]
        A = len(A_MOVES)
        P = np.zeros((self.S, A, self.S)); R = np.zeros((self.S, A))
        for s, (r, c) in enumerate(self.cells):
            for a, (dr, dc) in enumerate(A_MOVES):
                for a2, (dr2, dc2) in enumerate(A_MOVES):
                    p = (1 - SLIP) if a2 == a else SLIP / (A - 1)
                    r2, c2 = r + dr2, c + dc2
                    s2 = self.idx.get((r2, c2), s) if (0 <= r2 < self.h and 0 <= c2 < self.w) else s
                    if s == self.goal:
                        s2 = s                      # absorbing goal
                    P[s, a, s2] += p
            if s == self.goal:
                R[s, :] = 0.0
        # reward on entering the goal
        for s in range(self.S):
            for a in range(A):
                R[s, a] = P[s, a, self.goal] * (1.0 if s != self.goal else 0.0)
        self.P, self.R, self.A = P, R, A

    def value(self, pi: np.ndarray) -> np.ndarray:
        """pi: (S, A) -> V (S,)"""
        Ppi = np.einsum("sa,sat->st", pi, self.P); Rpi = (pi * self.R).sum(1)
        return np.linalg.solve(np.eye(self.S) - GAMMA * Ppi, Rpi)

    def q(self, pi: np.ndarray, V: np.ndarray) -> np.ndarray:
        return self.R + GAMMA * self.P @ V

    def visitation(self, pi: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Discounted state visitation d^pi_mu (normalised to sum 1)."""
        Ppi = np.einsum("sa,sat->st", pi, self.P)
        d = np.linalg.solve(np.eye(self.S) - GAMMA * Ppi.T, mu)
        return d * (1 - GAMMA)

    def optimal(self):
        V = np.zeros(self.S)
        for _ in range(2000):
            Q = self.R + GAMMA * self.P @ V
            V2 = Q.max(1)
            if np.abs(V2 - V).max() < 1e-12:
                break
            V = V2
        pi = np.zeros((self.S, self.A)); pi[np.arange(self.S), Q.argmax(1)] = 1.0
        return V, pi


def softmax_pg(g: Grid, mu: np.ndarray, rho: np.ndarray, v_star_rho: float, eta: float, max_iters: int):
    """Exact-gradient softmax PG. Returns iterations to reach TARGET_FRAC of V*(rho), or -1."""
    theta = np.zeros((g.S, g.A))
    for it in range(1, max_iters + 1):
        pi = np.exp(theta - theta.max(1, keepdims=True)); pi /= pi.sum(1, keepdims=True)
        V = g.value(pi)
        if V @ rho >= TARGET_FRAC * v_star_rho:
            return it
        Q = g.q(pi, V); Adv = Q - V[:, None]
        d = g.visitation(pi, mu)
        grad = d[:, None] * pi * Adv / (1 - GAMMA)
        theta += eta * grad
    return -1


def make_mus(g: Grid, pi_star: np.ndarray, rho: np.ndarray) -> dict[str, np.ndarray]:
    S = g.S
    delta = np.zeros(S); delta[g.start] = 1.0
    uniform = np.ones(S) / S
    d_star = g.visitation(pi_star, rho)
    mus = {}
    for beta in [0.0, 0.02, 0.1, 0.3, 0.6, 1.0]:
        mus[f"mix{beta:g}"] = (1 - beta) * delta + beta * uniform
    mus["dstar"] = d_star / d_star.sum()
    # anti-coverage: uniform over states far from the optimal path (low d*), with a little mass on the start
    far = (d_star < np.quantile(d_star, 0.5)).astype(float); far /= far.sum()
    mus["far"] = 0.9 * far + 0.1 * delta
    mus["far_only"] = far
    return mus


def coeff(num: np.ndarray, den: np.ndarray) -> float:
    m = num > 1e-12
    if (den[m] <= 1e-12).any():
        return float("inf")
    return float((num[m] / den[m]).max())


def run(args):
    rows = []
    t0 = time.time()
    for name, layout in LAYOUTS.items():
        g = Grid(layout)
        V_star, pi_star = g.optimal()
        rho = np.zeros(g.S); rho[g.start] = 1.0
        v_star_rho = float(V_star @ rho)
        d_star = g.visitation(pi_star, rho)
        pi0 = np.ones((g.S, g.A)) / g.A
        for mname, mu in make_mus(g, pi_star, rho).items():
            d0 = g.visitation(pi0, mu)
            D_mu = coeff(d_star, mu)
            D_0 = coeff(d_star, d0)
            # a "soft" version: chi-square-like expected ratio under d*
            E_mu = float(np.sum(d_star[mu > 1e-12] ** 2 / mu[mu > 1e-12])) if np.isfinite(D_mu) else float("inf")
            E_0 = float(np.sum(d_star ** 2 / d0))
            t = softmax_pg(g, mu, rho, v_star_rho, args.eta, args.max_iters)
            rows.append(dict(layout=name, mu=mname, S=g.S, v_star=v_star_rho, D_mu=D_mu, D_0=D_0, E_mu=E_mu, E_0=E_0,
                             t_iters=t if t > 0 else args.max_iters, converged=t > 0))
            print(f"{name:9s} {mname:9s} D_mu={D_mu:9.2f} D_0={D_0:9.2f} E_0={E_0:7.2f}  t={t} [{time.time() - t0:.0f}s]", flush=True)
    save_csv(RESULTS / "runs.csv", rows)
    summarize(rows)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(rows=None):
    rows = rows if rows is not None else load_csv(RESULTS / "runs.csv")
    conv = [r for r in rows if r["converged"]]
    fin = [r for r in conv if np.isfinite(r["D_mu"]) and r["D_mu"] != float("inf")]
    rho_mu = spearman(np.log([r["D_mu"] for r in fin]), np.log([r["t_iters"] for r in fin]))
    rho_0 = spearman(np.log([r["D_0"] for r in conv]), np.log([r["t_iters"] for r in conv]))
    rho_E0 = spearman(np.log([r["E_0"] for r in conv]), np.log([r["t_iters"] for r in conv]))
    # within-layout monotonicity for D_0
    within = []
    for name in LAYOUTS:
        sel = [r for r in conv if r["layout"] == name]
        if len(sel) >= 4:
            within.append(spearman(np.log([r["D_0"] for r in sel]), np.log([r["t_iters"] for r in sel])))
    k1 = rho_mu >= 0.7
    k2 = rho_0 >= 0.7 and rho_0 >= rho_mu + 0.1 and len(conv) == len(rows)
    k3 = all(w >= 0.6 for w in within)
    verdict = True if ((k1 or k2) and k3) else (None if (k1 or k2) else False)
    text = (f"K1 (Spearman of log t on log D_mu over the {len(fin)} configs with finite D_mu = {rho_mu:+.2f} >= 0.7) = {k1}. "
            f"K2 (Spearman of log t on log D_0 over all {len(conv)} converged configs = {rho_0:+.2f} >= 0.7, beats D_mu by 0.1, all configs converged = {len(conv) == len(rows)}) = {k2}. "
            f"K3 (within-layout Spearman for D_0 all >= 0.6: {[round(w, 2) for w in within]}) = {k3}. "
            f"[E_0 (expected ratio) Spearman = {rho_E0:+.2f}]")
    line = kill_test(verdict, text)
    (RESULTS / "verdict.txt").write_text(line + "\n")
    summary = [dict(layout=r["layout"], mu=r["mu"], D_mu=r["D_mu"], D_0=r["D_0"], E_0=r["E_0"], t_iters=r["t_iters"]) for r in rows]
    save_csv(RESULTS / "summary.csv", summary)
    print("\nlayout     mu         D_mu       D_0      E_0     iters")
    for r in rows:
        print(f"{r['layout']:9s} {r['mu']:9s} {r['D_mu']:9.2f} {r['D_0']:9.2f} {r['E_0']:7.2f}  {r['t_iters']:7d}{'' if r['converged'] else ' (cap)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eta", type=float, default=ETA)
    ap.add_argument("--max-iters", type=int, default=MAX_ITERS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize(); return
    if args.smoke:
        args.max_iters = 2000
    RESULTS.mkdir(exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
