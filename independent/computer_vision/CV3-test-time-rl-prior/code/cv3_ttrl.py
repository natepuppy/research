"""CV3: test-time RL amplifies the prior. Linear softmax classifier with a spurious cue of controllable
strength; TTRV-style adaptation (frequency reward, group-standardised advantages, policy gradient) on an
unlabelled shifted test set; harm/gain decomposition and an augmentation-instability gate. numpy only."""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message="All-NaN slice encountered")  # conflicting subset is empty at phi = 0
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io_utils import kill_test, load_csv, save_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

C, D_CORE, D_SP = 5, 10, 3
D = D_CORE + D_SP
STRENGTHS = [0.6, 0.8, 0.95]
PHIS = [0.0, 0.25, 0.5]
SHIFT_NORM = 1.0
AUG_SIGMA = 0.5
GATE_FRAC = 0.3

CFG = dict(seeds=8, n_train=5000, n_test=2000, train_steps=400, adapt_steps=50, group=16, lr_adapt=1e-2, n_aug=16)
SMOKE = dict(seeds=2, n_train=5000, n_test=1000, train_steps=400, adapt_steps=20, group=16, lr_adapt=1e-2, n_aug=16)


# ----------------------------------------------------------------------------- data

def make_world(rng: np.random.Generator) -> dict:
    return dict(mu=rng.normal(size=(C, D_CORE)) * 0.5, pat=rng.normal(size=(C, D_SP)),
                shift=rng.normal(size=D_CORE) * SHIFT_NORM / np.sqrt(D_CORE))


def make_data(n: int, world: dict, s: float, phi: float, shifted: bool, rng: np.random.Generator):
    """s: P(spurious block matches the class) on training data; phi: fraction of conflicting examples on test."""
    y = rng.integers(0, C, n)
    x = world["mu"][y] + rng.normal(size=(n, D_CORE))
    if shifted:
        x += world["shift"]
    if phi is None:                      # training: match w.p. s else a random other class
        match = rng.random(n) < s
    else:                                # test: conflicting fraction phi
        match = rng.random(n) >= phi
    other = (y + rng.integers(1, C, n)) % C
    cue = np.where(match, y, other)
    sp = world["pat"][cue] + rng.normal(size=(n, D_SP)) * 0.3
    return np.c_[x, sp], y, ~match


# ----------------------------------------------------------------------------- model

def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    p = np.exp(z); return p / p.sum(1, keepdims=True)


def predict(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    return softmax(np.c_[x, np.ones(len(x))] @ W)


class Adam:
    def __init__(self, shape, lr):
        self.lr, self.t = lr, 0; self.m = np.zeros(shape); self.v = np.zeros(shape)

    def step(self, W, g):
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * g; self.v = 0.999 * self.v + 0.001 * g ** 2
        W -= self.lr * (self.m / (1 - 0.9 ** self.t)) / (np.sqrt(self.v / (1 - 0.999 ** self.t)) + 1e-8)


def train_base(x: np.ndarray, y: np.ndarray, steps: int, rng: np.random.Generator) -> np.ndarray:
    W = rng.normal(size=(D + 1, C)) * 0.01
    xb = np.c_[x, np.ones(len(x))]
    opt = Adam(W.shape, 3e-2)
    for _ in range(steps):
        p = softmax(xb @ W); d = p.copy(); d[np.arange(len(y)), y] -= 1
        opt.step(W, xb.T @ d / len(y) + 1e-3 * W)
    return W


def ttrv_adapt(W: np.ndarray, x: np.ndarray, steps: int, group: int, lr: float, rng: np.random.Generator,
               mask: np.ndarray | None = None) -> np.ndarray:
    """TTRV recipe: sample `group` answers per example, reward = frequency of the sampled answer among the
    group, standardise advantages within the group, policy-gradient step. Labels never used."""
    W = W.copy()
    xs = x if mask is None else x[mask]
    xb = np.c_[xs, np.ones(len(xs))]
    opt = Adam(W.shape, lr)
    for _ in range(steps):
        p = softmax(xb @ W)
        a = (p.cumsum(1)[:, None, :] > rng.random((len(xs), group, 1))).argmax(2)          # (n, group)
        onehot = np.zeros((len(xs), group, C)); np.put_along_axis(onehot, a[..., None], 1.0, axis=2)
        freq = onehot.mean(1)                                                                # (n, C)
        r = np.take_along_axis(freq, a, axis=1)                                              # reward = frequency
        adv = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-6)
        d = -(adv[..., None] * (onehot - p[:, None, :])).mean(1)                             # d/dlogits
        opt.step(W, xb.T @ d / len(xs))
    return W


# ----------------------------------------------------------------------------- measurement

def auroc(score: np.ndarray, y: np.ndarray) -> float:
    y = y.astype(bool)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    s_sorted = score[order]
    _, first, counts = np.unique(s_sorted, return_index=True, return_counts=True)
    for f, c in zip(first, counts):
        if c > 1:
            ranks[order[f:f + c]] = f + 1 + (c - 1) / 2
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def gate_score(W: np.ndarray, x: np.ndarray, n_aug: int, rng: np.random.Generator) -> np.ndarray:
    clean = predict(W, x).argmax(1)
    diff = np.zeros(len(x))
    for _ in range(n_aug):
        diff += predict(W, x + rng.normal(size=x.shape) * AUG_SIGMA).argmax(1) != clean
    return diff / n_aug


def run_cell(s: float, phi: float, seed: int, cfg: dict) -> dict:
    rng = np.random.default_rng(seed * 1000 + int(s * 100) + int(phi * 10))
    world = make_world(rng)
    xtr, ytr, _ = make_data(cfg["n_train"], world, s, None, False, rng)
    W0 = train_base(xtr, ytr, cfg["train_steps"], rng)
    xte, yte, conflict = make_data(cfg["n_test"], world, s, phi, True, rng)

    p0 = predict(W0, xte); pred0 = p0.argmax(1); correct0 = pred0 == yte
    conf_score = 1 - p0.max(1)
    gate = gate_score(W0, xte, cfg["n_aug"], rng)

    adapt_rng = int(rng.integers(1 << 31))
    W1 = ttrv_adapt(W0, xte, cfg["adapt_steps"], cfg["group"], cfg["lr_adapt"], np.random.default_rng(adapt_rng))
    correct1 = predict(W1, xte).argmax(1) == yte
    harm = correct0 & ~correct1; gain = ~correct0 & correct1

    keep = gate <= np.quantile(gate, 1 - GATE_FRAC)
    Wg = ttrv_adapt(W0, xte, cfg["adapt_steps"], cfg["group"], cfg["lr_adapt"], np.random.default_rng(adapt_rng), mask=keep)
    correctg = predict(Wg, xte).argmax(1) == yte
    harm_g = correct0 & ~correctg; gain_g = ~correct0 & correctg

    aligned = ~conflict
    sub = lambda m, c: float(c[m].mean()) if m.any() else float("nan")  # noqa: E731
    return dict(
        s=s, phi=phi, seed=seed,
        acc0=float(correct0.mean()), acc1=float(correct1.mean()), accg=float(correctg.mean()),
        d_acc=float(correct1.mean() - correct0.mean()),
        d_acc_conflict=sub(conflict, correct1) - sub(conflict, correct0),
        d_acc_aligned=sub(aligned, correct1) - sub(aligned, correct0),
        n_harm=int(harm.sum()), n_gain=int(gain.sum()),
        n_harm_gated=int(harm_g.sum()), n_gain_aligned=int((gain & aligned).sum()),
        n_gain_aligned_gated=int((gain_g & aligned).sum()),
        auroc_gate=auroc(gate[correct0], harm[correct0]), auroc_conf=auroc(conf_score[correct0], harm[correct0]),
        mode_wrong_frac=float((~correct0).mean()), d_acc_gated=float(correctg.mean() - correct0.mean()),
    )


# ----------------------------------------------------------------------------- summary & verdict

def summarize(rows: list[dict]) -> tuple[list[dict], str]:
    cells = sorted({(r["s"], r["phi"]) for r in rows})
    summ = []
    for s, phi in cells:
        rs = [r for r in rows if r["s"] == s and r["phi"] == phi]
        med = lambda k: float(np.nanmedian([r[k] for r in rs]))  # noqa: E731
        summ.append(dict(s=s, phi=phi, n_seeds=len(rs), acc0=med("acc0"), acc1=med("acc1"), accg=med("accg"),
                         d_acc=med("d_acc"), d_acc_conflict=med("d_acc_conflict"), d_acc_aligned=med("d_acc_aligned"),
                         neg_conflict_seeds=int(sum(r["d_acc_conflict"] < 0 for r in rs)),
                         n_harm=med("n_harm"), n_gain=med("n_gain"), n_harm_gated=med("n_harm_gated"),
                         n_gain_aligned=med("n_gain_aligned"), n_gain_aligned_gated=med("n_gain_aligned_gated"),
                         auroc_gate=med("auroc_gate"), auroc_conf=med("auroc_conf"), d_acc_gated=med("d_acc_gated")))
    get = lambda s, phi, k: next(x[k] for x in summ if x["s"] == s and x["phi"] == phi)  # noqa: E731
    n_seeds = max(x["n_seeds"] for x in summ)
    need = 6 if n_seeds >= 8 else n_seeds

    # K1 at phi = 0.25
    k1_neg = all(get(s, 0.25, "neg_conflict_seeds") >= need for s in STRENGTHS)
    dc = [get(s, 0.25, "d_acc_conflict") for s in STRENGTHS]
    k1 = k1_neg and all(dc[i + 1] <= dc[i] for i in range(len(dc) - 1))
    # K2 at s = 0.8
    k2 = get(0.8, 0.0, "d_acc") >= 0 and get(0.8, 0.5, "d_acc") <= 0
    # K3 pooled over cells with >= 20 harmed examples
    hc = [x for x in summ if x["n_harm"] >= 20]
    au_g = float(np.nanmedian([x["auroc_gate"] for x in hc])) if hc else float("nan")
    au_c = float(np.nanmedian([x["auroc_conf"] for x in hc])) if hc else float("nan")
    harm_removed = 1 - get(0.8, 0.25, "n_harm_gated") / max(get(0.8, 0.25, "n_harm"), 1)
    gain_kept = get(0.8, 0.25, "n_gain_aligned_gated") / max(get(0.8, 0.25, "n_gain_aligned"), 1)
    k3 = bool(hc) and au_g >= 0.70 and au_g >= au_c and harm_removed >= 0.5 and gain_kept >= 0.7
    verdict = True if (k1 and k2) else (None if (k1 or k2) else False)
    text = (f"K1 (phi=0.25: conflicting-subset d_acc negative in >= {need}/{n_seeds} seeds for every s: "
            + ", ".join(f"s={s}: {get(s, 0.25, 'neg_conflict_seeds')}/{n_seeds}, median {get(s, 0.25, 'd_acc_conflict'):+.3f}" for s in STRENGTHS)
            + f"; monotone in s) = {k1}. "
            f"K2 (s=0.8: net d_acc {get(0.8, 0.0, 'd_acc'):+.3f} at phi=0 >= 0 and {get(0.8, 0.5, 'd_acc'):+.3f} at phi=0.5 <= 0; "
            f"phi=0.25: {get(0.8, 0.25, 'd_acc'):+.3f}) = {k2}. "
            f"K3 (pooled AUROC gate {au_g:.3f} >= 0.70 and >= confidence {au_c:.3f}; at (0.8, 0.25) harm removed {harm_removed:.2f} >= 0.5, "
            f"aligned gain kept {gain_kept:.2f} >= 0.7) = {k3}.")
    return summ, kill_test(verdict, text)


def print_table(summ: list[dict]) -> None:
    print("\n  s     phi   acc0   acc1   d_acc   d_conf  d_align  harm  gain  AUROC(gate/conf)  acc_gated  harm_gated")
    for x in summ:
        print(f" {x['s']:<5} {x['phi']:<5} {x['acc0']:.3f}  {x['acc1']:.3f}  {x['d_acc']:+.3f}  {x['d_acc_conflict']:+.3f}  "
              f"{x['d_acc_aligned']:+.3f}   {x['n_harm']:>4.0f}  {x['n_gain']:>4.0f}  {x['auroc_gate']:.3f}/{x['auroc_conf']:.3f}       "
              f"{x['accg']:.3f}      {x['n_harm_gated']:>4.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.summarize:
        rows = load_csv(RESULTS / "runs.csv")
    else:
        cfg = SMOKE if args.smoke else CFG
        rows = []
        t0 = time.time()
        for s in STRENGTHS:
            for phi in PHIS:
                for seed in range(cfg["seeds"]):
                    rows.append(run_cell(s, phi, seed, cfg))
                r = rows[-1]
                print(f"s={s} phi={phi} [{time.time() - t0:.0f}s] last seed: acc {r['acc0']:.3f} -> {r['acc1']:.3f} "
                      f"(conflict {r['d_acc_conflict']:+.3f}, aligned {r['d_acc_aligned']:+.3f}) gated {r['accg']:.3f}", flush=True)
        if not args.smoke:
            save_csv(RESULTS / "runs.csv", rows)
    summ, line = summarize(rows)
    print_table(summ)
    if not args.smoke:
        save_csv(RESULTS / "summary.csv", summ)
        (RESULTS / "verdict.txt").write_text(line + "\n")


if __name__ == "__main__":
    main()
