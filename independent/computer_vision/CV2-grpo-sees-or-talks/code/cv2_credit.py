"""CV2: where does the gain from outcome-reward RL live? Two-module network (perception -> reasoner)
pretrained by SFT, then group-relative policy gradient on the final-answer reward with each module
frozen or trainable. Perception measured by a linear probe on the latent, reasoning by noise-free accuracy.
Hand-written gradients; numpy only."""

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

NA, K, NQ, ND = 3, 5, 3, 10          # attributes, count levels, questions, distractor dims
D = NA * K + ND + NQ
H1, H2, C = 32, 32, NA
NU_SFT = 0.3
NUS_RL = [0.3, 0.6, 0.9]
ARMS = ["all", "reasoner", "perception", "sft_ref"]

CFG = dict(seeds=8, n_sft=2000, sft_steps=300, rl_steps=300, batch=64, group=8, lr_sft=3e-3, lr_rl=1e-3, n_eval=4000)
SMOKE = dict(seeds=2, n_sft=2000, sft_steps=300, rl_steps=60, batch=64, group=8, lr_sft=3e-3, lr_rl=1e-3, n_eval=1000)


# ----------------------------------------------------------------------------- data

def make_batch(n: int, nu: float, rng: np.random.Generator):
    counts = rng.integers(0, K, (n, NA))
    # resample rows with ties in the sort order (so largest/smallest/middle are unique)
    for _ in range(50):
        srt = np.sort(counts, axis=1)
        bad = (srt[:, 1:] == srt[:, :-1]).any(1)
        if not bad.any():
            break
        counts[bad] = rng.integers(0, K, (bad.sum(), NA))
    q = rng.integers(0, NQ, n)
    order = np.argsort(counts, axis=1)              # ascending attribute indices
    y = np.where(q == 0, order[:, -1], np.where(q == 1, order[:, 0], order[:, 1]))
    x = np.zeros((n, D))
    for a in range(NA):
        blk = np.zeros((n, K)); blk[np.arange(n), counts[:, a]] = 1.0
        x[:, a * K:(a + 1) * K] = blk + rng.normal(size=(n, K)) * nu
    x[:, NA * K:NA * K + ND] = rng.normal(size=(n, ND))
    x[np.arange(n), NA * K + ND + q] = 1.0
    return x, y, counts


# ----------------------------------------------------------------------------- model

def init_params(rng: np.random.Generator) -> dict:
    s = lambda *shape: rng.normal(size=shape) / np.sqrt(shape[0])  # noqa: E731
    return dict(W1=s(D, H1), b1=np.zeros(H1), W2=s(H1 + NQ, H2), b2=np.zeros(H2), W3=s(H2, C), b3=np.zeros(C))


PERCEPTION = ("W1", "b1")
REASONER = ("W2", "b2", "W3", "b3")


def forward(p: dict, x: np.ndarray):
    z = np.tanh(x @ p["W1"] + p["b1"])
    u = np.c_[z, x[:, NA * K + ND:]]
    h = np.tanh(u @ p["W2"] + p["b2"])
    logits = h @ p["W3"] + p["b3"]
    logits -= logits.max(1, keepdims=True)
    prob = np.exp(logits); prob /= prob.sum(1, keepdims=True)
    return z, u, h, prob


def backward(p: dict, x: np.ndarray, cache, dlogits: np.ndarray) -> dict:
    z, u, h, _ = cache
    n = x.shape[0]
    g = {}
    g["W3"] = h.T @ dlogits / n; g["b3"] = dlogits.mean(0)
    dh = dlogits @ p["W3"].T * (1 - h ** 2)
    g["W2"] = u.T @ dh / n; g["b2"] = dh.mean(0)
    dz = (dh @ p["W2"].T)[:, :H1] * (1 - z ** 2)
    g["W1"] = x.T @ dz / n; g["b1"] = dz.mean(0)
    return g


class Adam:
    def __init__(self, p: dict, lr: float, trainable: tuple):
        self.lr, self.t = lr, 0
        self.m = {k: np.zeros_like(v) for k, v in p.items()}
        self.v = {k: np.zeros_like(v) for k, v in p.items()}
        self.trainable = trainable

    def step(self, p: dict, g: dict) -> None:
        self.t += 1
        for k in self.trainable:
            self.m[k] = 0.9 * self.m[k] + 0.1 * g[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * g[k] ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t); vh = self.v[k] / (1 - 0.999 ** self.t)
            p[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def sft(p: dict, x: np.ndarray, y: np.ndarray, steps: int, batch: int, lr: float, rng: np.random.Generator,
        trainable=PERCEPTION + REASONER) -> None:
    opt = Adam(p, lr, trainable)
    for _ in range(steps):
        idx = rng.integers(0, len(x), batch)
        cache = forward(p, x[idx])
        prob = cache[3]
        d = prob.copy(); d[np.arange(batch), y[idx]] -= 1.0
        opt.step(p, backward(p, x[idx], cache, d))


def grpo(p: dict, nu: float, steps: int, batch: int, group: int, lr: float, rng: np.random.Generator,
         trainable: tuple) -> None:
    """Group-relative policy gradient on the final-answer reward. Fresh unlabeled-looking batches each step;
    labels are used only to compute the reward."""
    opt = Adam(p, lr, trainable)
    for _ in range(steps):
        x, y, _ = make_batch(batch, nu, rng)
        cache = forward(p, x)
        prob = cache[3]
        a = (prob.cumsum(1)[:, None, :] > rng.random((batch, group, 1))).argmax(2)   # (batch, group)
        r = (a == y[:, None]).astype(float)
        adv = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-6)
        # d/dlogits of -adv * log prob[a], averaged over the group
        onehot = np.zeros((batch, group, C)); np.put_along_axis(onehot, a[..., None], 1.0, axis=2)
        d = -(adv[..., None] * (onehot - prob[:, None, :])).mean(1)
        opt.step(p, backward(p, x, cache, d))


# ----------------------------------------------------------------------------- measurement

def probe_r2(z: np.ndarray, counts: np.ndarray) -> float:
    n = len(z) // 2
    Xa = np.c_[z[:n], np.ones(n)]; Xb = np.c_[z[n:], np.ones(len(z) - n)]
    W = np.linalg.solve(Xa.T @ Xa + 1e-3 * np.eye(Xa.shape[1]), Xa.T @ counts[:n])
    pred = Xb @ W; yb = counts[n:]
    ss_res = ((yb - pred) ** 2).sum(0); ss_tot = ((yb - yb.mean(0)) ** 2).sum(0)
    return float(np.mean(1 - ss_res / ss_tot))


def measure(p: dict, nu: float, n: int, rng: np.random.Generator) -> dict:
    x, y, counts = make_batch(n, nu, rng)
    z, _, _, prob = forward(p, x)
    acc = float((prob.argmax(1) == y).mean())
    ent = float(-(prob * np.log(prob + 1e-12)).sum(1).mean())
    x0, y0, _ = make_batch(n, 0.0, rng)
    acc0 = float((forward(p, x0)[3].argmax(1) == y0).mean())
    return dict(acc=acc, probe=probe_r2(z, counts), reason_acc=acc0, entropy=ent)


# ----------------------------------------------------------------------------- one cell x seed

def run_cell(nu_rl: float, seed: int, cfg: dict) -> list[dict]:
    rng = np.random.default_rng(seed * 100 + int(nu_rl * 10))
    p0 = init_params(rng)
    xs, ys, _ = make_batch(cfg["n_sft"], NU_SFT, rng)
    sft(p0, xs, ys, cfg["sft_steps"], cfg["batch"], cfg["lr_sft"], rng)
    eval_seed = int(rng.integers(1 << 31))
    base = measure(p0, nu_rl, cfg["n_eval"], np.random.default_rng(eval_seed))
    rows = [dict(nu_rl=nu_rl, seed=seed, arm="sft", **base)]
    for arm in ARMS:
        p = {k: v.copy() for k, v in p0.items()}
        r = np.random.default_rng(seed * 100 + int(nu_rl * 10) + 7)
        if arm == "sft_ref":
            x, y, _ = make_batch(cfg["rl_steps"] * cfg["batch"], nu_rl, r)
            sft(p, x, y, cfg["rl_steps"], cfg["batch"], cfg["lr_rl"], r)
        else:
            trainable = {"all": PERCEPTION + REASONER, "reasoner": REASONER, "perception": PERCEPTION}[arm]
            grpo(p, nu_rl, cfg["rl_steps"], cfg["batch"], cfg["group"], cfg["lr_rl"], r, trainable)
        m = measure(p, nu_rl, cfg["n_eval"], np.random.default_rng(eval_seed))
        rows.append(dict(nu_rl=nu_rl, seed=seed, arm=arm, **m,
                         d_acc=m["acc"] - base["acc"], d_probe=m["probe"] - base["probe"],
                         d_reason=m["reason_acc"] - base["reason_acc"], d_entropy=m["entropy"] - base["entropy"]))
    return rows


# ----------------------------------------------------------------------------- summary & verdict

def summarize(rows: list[dict]) -> tuple[list[dict], str]:
    nus = sorted({r["nu_rl"] for r in rows})
    summ = []
    for nu in nus:
        for arm in ["sft"] + ARMS:
            rs = [r for r in rows if r["nu_rl"] == nu and r["arm"] == arm]
            med = lambda k: float(np.median([r.get(k, 0.0) for r in rs]))  # noqa: E731
            summ.append(dict(nu_rl=nu, arm=arm, n_seeds=len(rs), acc=med("acc"), probe=med("probe"),
                             reason_acc=med("reason_acc"), entropy=med("entropy"), d_acc=med("d_acc"),
                             d_probe=med("d_probe"), d_reason=med("d_reason"), d_entropy=med("d_entropy")))
    get = lambda nu, arm, k: next(s[k] for s in summ if s["nu_rl"] == nu and s["arm"] == arm)  # noqa: E731

    k1 = all(get(nu, "all", "d_acc") >= 0.05 for nu in (0.3, 0.6) if nu in nus)
    k2_cells = [nu for nu in nus if get(nu, "all", "d_acc") >= 0.03]
    k2 = bool(k2_cells) and all(get(nu, "reasoner", "d_acc") >= 0.7 * get(nu, "all", "d_acc")
                                and get(nu, "perception", "d_acc") <= 0.3 * get(nu, "all", "d_acc") for nu in k2_cells)
    k3 = (all(abs(get(nu, "all", "d_probe")) < 0.05 for nu in nus)
          and (0.6 not in nus or get(0.6, "sft_ref", "d_probe") >= 0.05))
    verdict = True if (k1 and k2) else (None if (k1 or k2) else False)
    text = ("K1 (median d_acc[all] >= 0.05 at nu=0.3, 0.6: "
            + ", ".join(f"nu={nu}: {get(nu, 'all', 'd_acc'):+.3f}" for nu in nus) + f") = {k1}. "
            f"K2 (cells with d_acc[all] >= 0.03: {k2_cells}; reasoner-only >= 0.7x and perception-only <= 0.3x: "
            + "; ".join(f"nu={nu}: all {get(nu, 'all', 'd_acc'):+.3f}, reasoner {get(nu, 'reasoner', 'd_acc'):+.3f}, "
                        f"perception {get(nu, 'perception', 'd_acc'):+.3f}" for nu in nus) + f") = {k2}. "
            f"K3 (|d_probe[all]| < 0.05 everywhere and d_probe[sft_ref] >= 0.05 at nu=0.6: "
            + ", ".join(f"nu={nu}: all {get(nu, 'all', 'd_probe'):+.3f} / sft_ref {get(nu, 'sft_ref', 'd_probe'):+.3f}"
                        for nu in nus) + f") = {k3}.")
    return summ, kill_test(verdict, text)


def print_table(summ: list[dict]) -> None:
    print("\n nu_rl  arm         acc     d_acc   probeR2  d_probe  reason  d_reason  entropy")
    for s in summ:
        print(f" {s['nu_rl']:<5}  {s['arm']:<10}  {s['acc']:.3f}  {s['d_acc']:+.3f}  {s['probe']:.3f}   "
              f"{s['d_probe']:+.3f}   {s['reason_acc']:.3f}  {s['d_reason']:+.3f}    {s['entropy']:.3f}")


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
        for nu in NUS_RL:
            for seed in range(cfg["seeds"]):
                rows.extend(run_cell(nu, seed, cfg))
            last = [r for r in rows if r["nu_rl"] == nu and r["seed"] == cfg["seeds"] - 1]
            print(f"nu_rl={nu} [{time.time() - t0:.0f}s] last seed: " +
                  ", ".join(f"{r['arm']} {r['acc']:.3f}" for r in last), flush=True)
        if not args.smoke:
            save_csv(RESULTS / "runs.csv", rows)
    summ, line = summarize(rows)
    print_table(summ)
    if not args.smoke:
        save_csv(RESULTS / "summary.csv", summ)
        (RESULTS / "verdict.txt").write_text(line + "\n")


if __name__ == "__main__":
    main()
