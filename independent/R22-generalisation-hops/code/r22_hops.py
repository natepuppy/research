"""R22: hops per target sync under function approximation, as a function of input encoding.
Batched offline DQN (R runs as one tensor program) on a fixed corridor dataset."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.dqn_batched import Adam, Params, forward  # noqa: E402
from lib.io_utils import kill_test, load_csv, save_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
L = 40
GAMMA = 0.99
ENCS = ["onehot", "random", "thermo", "coord"]
KS = [50, 200, 800]
N_DATA = 20_000
BATCH = 64
HIDDEN = 64
LR = 1e-3
FRONT_EVERY = 50
GAP: tuple | None = None  # (lo, hi) inclusive block of states with no data (follow-up)
torch.set_num_threads(4)


def encode(enc: str, states: torch.Tensor, code: torch.Tensor | None) -> torch.Tensor:
    """states: (...) long in 0..L-1 -> features (..., F)."""
    if enc == "onehot":
        return torch.nn.functional.one_hot(states, L).float()
    if enc == "thermo":
        k = torch.arange(1, L + 1)
        return (states[..., None] >= k).float()
    if enc == "coord":
        return (states.float() / L)[..., None]
    if enc == "random":
        return code[states]
    raise ValueError(enc)


FEAT = {"onehot": L, "thermo": L, "coord": 1, "random": 16}


def init_params(R, fin, gens):
    def layer(i, o):
        bound = 1 / math.sqrt(i)
        W = torch.stack([(torch.rand(i, o, generator=g) * 2 - 1) * bound for g in gens])
        b = torch.stack([(torch.rand(o, generator=g) * 2 - 1) * bound for g in gens])
        return W, b
    W1, b1 = layer(fin, HIDDEN); W2, b2 = layer(HIDDEN, HIDDEN); W3, b3 = layer(HIDDEN, 2)
    return Params(W1, b1, W2, b2, W3, b3)


V_STAR = torch.tensor([GAMMA ** (L - 1 - s) for s in range(L)])


def value_front(V: torch.Tensor) -> torch.Tensor:
    """V: (R, L). Largest d such that V(s) >= 0.5 V*(s) for all s in L-d..L-1 (contiguous from the goal)."""
    arrived = (V >= 0.5 * V_STAR).flip(1)  # index 0 == state L-1
    return torch.cumprod(arrived.long(), dim=1).sum(1)


def front_contiguous(greedy: torch.Tensor) -> torch.Tensor:
    R = greedy.shape[0]
    ok = torch.zeros(R, L + 1, dtype=torch.bool); ok[:, 0] = True
    for d in range(1, L + 1):
        pos = torch.full((R,), L - d, dtype=torch.long); reached = torch.zeros(R, dtype=torch.bool)
        for _ in range(2 * d):
            a = greedy[torch.arange(R), pos.clamp(max=L - 1)]
            pos = torch.where(reached, pos, torch.where(a == 1, pos + 1, (pos - 1).clamp(min=0)))
            reached |= pos >= L
        ok[:, d] = reached
    cum = torch.cumprod(ok.long(), dim=1)
    return cum.sum(1) - 1


def run_encoding(enc: str, seeds: int, max_updates: int, log):
    combos = [(K, sd) for K in KS for sd in range(seeds)]
    R = len(combos)
    gens = [torch.Generator().manual_seed(1000 * sd + 7) for _, sd in combos]
    fin = FEAT[enc]
    codes = torch.stack([(torch.rand(L, 16, generator=g) < 0.5).float() * 2 - 1 for g in gens]) if enc == "random" else None
    p = init_params(R, fin, gens); pt = p.clone(); opt = Adam(p, lr=LR)
    Kt = torch.tensor([K for K, _ in combos])
    # fixed dataset per seed: uniform (s, a)
    gd = [torch.Generator().manual_seed(500 + sd) for _, sd in combos]
    if GAP:
        allowed = torch.tensor([x for x in range(L) if not (GAP[0] <= x <= GAP[1])])
        S = torch.stack([allowed[torch.randint(0, len(allowed), (N_DATA,), generator=g)] for g in gd])
    else:
        S = torch.stack([torch.randint(0, L, (N_DATA,), generator=g) for g in gd])
    A = torch.stack([torch.randint(0, 2, (N_DATA,), generator=g) for g in gd])
    S2 = torch.where(A == 1, S + 1, (S - 1).clamp(min=0))
    D = (S2 >= L).float(); Rw = D.clone(); S2c = S2.clamp(max=L - 1)
    all_states = torch.arange(L)
    ar = torch.arange(R)[:, None]

    def feats(states):  # states (R, B) -> (R, B, F)
        if enc == "random":
            return torch.stack([codes[i][states[i]] for i in range(R)])
        return encode(enc, states, None)

    X_all = feats(all_states[None].expand(R, L))
    T_p = torch.full((R,), -1, dtype=torch.long)
    fronts = []
    t0 = time.time()
    for t in range(1, max_updates + 1):
        idx = torch.stack([torch.randint(0, N_DATA, (BATCH,), generator=g) for g in gd])
        s, a, r, s2, d = S[ar, idx], A[ar, idx], Rw[ar, idx], S2c[ar, idx], D[ar, idx]
        with torch.no_grad():
            q2 = forward(pt, feats(s2)).max(-1).values
            y = r + GAMMA * (1 - d) * q2
        ps = [w.detach().requires_grad_(True) for w in p.tensors()]
        pp = Params(*ps)
        q = forward(pp, feats(s)).gather(-1, a[..., None])[..., 0]
        loss = torch.nn.functional.smooth_l1_loss(q, y, reduction="none").mean(-1).sum()
        grads = torch.autograd.grad(loss, ps)
        norms = torch.sqrt(sum((g.flatten(1) ** 2).sum(1) for g in grads))
        scale = torch.clamp(10.0 / (norms + 1e-6), max=1.0)
        grads = [g * scale.view((-1,) + (1,) * (g.dim() - 1)) for g in grads]
        opt.step(p, grads)
        sync = (t % Kt) == 0
        if sync.any():
            for w_t, w in zip(pt.tensors(), p.tensors()):
                w_t[sync] = w[sync].clone()
        if t % FRONT_EVERY == 0:
            with torch.no_grad():
                qall = forward(p, X_all)
            greedy = qall.argmax(-1)
            fr_pi = front_contiguous(greedy)
            fr = value_front(qall.max(-1).values)
            for i in range(R):
                fronts.append(dict(enc=enc, K=combos[i][0], seed=combos[i][1], t=t, front=int(fr[i]), front_pi=int(fr_pi[i])))
            newly = (fr >= L) & (T_p < 0)
            T_p[newly] = t
            if t % 5000 == 0:
                log(f"{enc} t={t} [{time.time() - t0:.0f}s] solved {float((T_p >= 0).float().mean()):.2f} mean front {float(fr.float().mean()):.1f}")
            if (T_p >= 0).all():
                break
    rows = [dict(enc=enc, K=K, seed=sd, T_p=int(T_p[i]) if T_p[i] > 0 else max_updates, solved=bool(T_p[i] > 0)) for i, (K, sd) in enumerate(combos)]
    with torch.no_grad():
        Vfin = forward(p, X_all).max(-1).values
    vals = [dict(enc=enc, K=combos[i][0], seed=combos[i][1], s=s_, V=float(Vfin[i, s_]), Vstar=float(V_STAR[s_])) for i in range(R) for s_ in range(L)]
    return rows, fronts, vals


def summarize(rows=None):
    rows = rows if rows is not None else load_csv(RESULTS / "runs.csv")
    summ = []
    for enc in ENCS:
        for K in KS:
            sel = [r for r in rows if r["enc"] == enc and r["K"] == K]
            if not sel:
                continue
            med = float(np.median([r["T_p"] for r in sel]))
            summ.append(dict(enc=enc, K=K, n=len(sel), T_p=med, solved=float(np.mean([r["solved"] for r in sel])), eta=K * L / med))
    save_csv(RESULTS / "summary.csv", summ)

    def cell(enc, K):
        return next(s for s in summ if s["enc"] == enc and s["K"] == K)

    fits = {}
    for enc in ENCS:
        cs_ = [cell(enc, K) for K in KS]
        x = np.array([c["K"] for c in cs_], float); y = np.array([c["T_p"] for c in cs_])
        Xm = np.column_stack([x, np.ones_like(x)]); b, *_ = np.linalg.lstsq(Xm, y, rcond=None)
        r2 = 1 - ((y - Xm @ b) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        eta = L / b[0] if b[0] > 1e-9 else float("inf")
        fits[enc] = dict(enc=enc, slope=float(b[0]), intercept=float(b[1]), r2=float(r2), eta=float(eta))
    save_csv(RESULTS / "fits.csv", list(fits.values()))
    k1 = (0.67 <= fits["onehot"]["eta"] <= 1.5) and fits["onehot"]["r2"] >= 0.95
    k2 = fits["coord"]["eta"] >= 2 * fits["onehot"]["eta"] and fits["coord"]["eta"] >= fits["thermo"]["eta"] >= fits["onehot"]["eta"]
    k3 = fits["random"]["eta"] <= 1.5 * fits["onehot"]["eta"]
    v = True if (k1 and k2) else (None if (k1 or k2) else False)
    text = (f"K1 (onehot: eta = L/slope = {fits['onehot']['eta']:.2f} in [0.67, 1.5], slope {fits['onehot']['slope']:.1f} updates per unit K, intercept {fits['onehot']['intercept']:.0f}, R2 = {fits['onehot']['r2']:.3f}) = {k1}. "
            f"K2 (eta(coord) >= 2 eta(onehot) and coord >= thermo >= onehot; etas " + ", ".join(f"{e}={fits[e]['eta']:.2f}" for e in ENCS) + f"; slopes " + ", ".join(f"{e}={fits[e]['slope']:.1f}" for e in ENCS) + f") = {k2}. "
            f"K3 (eta(random) <= 1.5 eta(onehot)) = {k3}.")
    line = kill_test(v, text)
    (RESULTS / "verdict.txt").write_text(line + "\n")
    print("\n enc      K    n   T_p    solved  eta")
    for s in summ:
        print(f" {s['enc']:7s} {s['K']:4d}  {s['n']:2d}  {s['T_p']:6.0f}   {s['solved']:.2f}   {s['eta']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--max-updates", type=int, default=60_000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--encs", default=",".join(ENCS))
    ap.add_argument("--gap", action="store_true", help="follow-up: no data for states 15..24, K=200 only")
    args = ap.parse_args()
    global GAP, KS, RESULTS
    if args.gap:
        GAP = (15, 24); KS = [200]; RESULTS = RESULTS / "gap"
    if args.summarize:
        summarize(); return
    encs = args.encs.split(",")
    if args.smoke:
        args.seeds, args.max_updates = 2, 6000
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows, fronts, vals = [], [], []
    for enc in encs:
        r, f, v = run_encoding(enc, args.seeds, args.max_updates, lambda m: print(m, flush=True))
        rows += r; fronts += f; vals += v
        save_csv(RESULTS / "runs.csv", rows); save_csv(RESULTS / "front.csv", fronts); save_csv(RESULTS / "values.csv", vals)
    try:
        summarize(rows)
    except Exception as e:
        print("summary skipped:", e)


if __name__ == "__main__":
    main()
