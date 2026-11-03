"""CV1: dead-ends in zoom trajectories. Synthetic zoom grid, fixed stochastic base policy,
irrecoverability from Monte-Carlo continuations, logistic dead-end detector, gated re-crop.
Vectorised over episodes."""

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

G = 8            # grid side
R = 2            # readable view side
C = 10           # symbol classes (chance = 1/C)
B = 5            # action budget
P_READ = 0.95
TAU = 0.3        # softmax temperature over cues
DECAY = 0.7      # sigma_d = sigma * DECAY**depth
DELTA = 0.15     # dead-end threshold on P(success | prefix)
MIN_B = 3        # budget needed for a re-crop to be useful: 1 re-crop + 2 zooms to a readable view
SIGMAS = [0.5, 1.0, 1.5]
RHOS = [0.0, 0.3]
NF = 5 + 2 * B   # detector feature count

CFG = dict(seeds=8, n_train=1000, n_test=500, n_eval=2000, m_label=32, m_ref=256)
SMOKE = dict(seeds=2, n_train=200, n_test=100, n_eval=300, m_label=16, m_ref=64)


# ----------------------------------------------------------------------------- environment

def new_states(n: int, rng: np.random.Generator) -> dict:
    return dict(
        x0=np.zeros(n, int), y0=np.zeros(n, int), side=np.full(n, G, int), b=np.full(n, B, int),
        tx=rng.integers(0, G, n), ty=rng.integers(0, G, n), nrec=np.zeros(n, int),
        done=np.zeros(n, bool), succ=np.zeros(n, bool), acts=np.zeros(n, int),
        hm=np.zeros((n, B)), hc=np.zeros((n, B)), nz=np.zeros(n, int),
    )


def take(st: dict, idx: np.ndarray) -> dict:
    return {k: v[idx] for k, v in st.items()}


def depth_of(side: np.ndarray) -> np.ndarray:
    return np.log2(G // side).astype(int)


def in_view(st: dict) -> np.ndarray:
    return ((st["tx"] >= st["x0"]) & (st["tx"] < st["x0"] + st["side"])
            & (st["ty"] >= st["y0"]) & (st["ty"] < st["y0"] + st["side"]))


def quadrant_cues(st: dict, idx: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    half = st["side"][idx] // 2
    x0, y0, tx, ty = st["x0"][idx], st["y0"][idx], st["tx"][idx], st["ty"][idx]
    inq = np.zeros((idx.size, 4))
    for q in range(4):
        dx, dy = q % 2, q // 2
        inq[:, q] = ((tx >= x0 + dx * half) & (tx < x0 + (dx + 1) * half)
                     & (ty >= y0 + dy * half) & (ty < y0 + (dy + 1) * half))
    sd = sigma * DECAY ** depth_of(st["side"][idx])
    return inq + rng.normal(size=inq.shape) * sd[:, None]


def features(st: dict, idx: np.ndarray, cues: np.ndarray) -> np.ndarray:
    srt = np.sort(cues, axis=1)
    f = np.zeros((idx.size, NF))
    f[:, 0] = depth_of(st["side"][idx])
    f[:, 1] = st["b"][idx]
    f[:, 2] = st["nrec"][idx]
    f[:, 3] = srt[:, -1]
    f[:, 4] = srt[:, -1] - srt[:, -2]
    f[:, 5:5 + B] = st["hm"][idx]
    f[:, 5 + B:] = st["hc"][idx]
    return f


def recrop(st: dict, idx: np.ndarray) -> None:
    st["x0"][idx] = 0; st["y0"][idx] = 0; st["side"][idx] = G
    st["b"][idx] -= 1; st["nrec"][idx] += 1; st["acts"][idx] += 1


def guess(st: dict, idx: np.ndarray, rng: np.random.Generator) -> None:
    st["succ"][idx] = rng.random(idx.size) < 1.0 / C
    st["done"][idx] = True; st["acts"][idx] += 1


def run(st: dict, sigma: float, rho: float, rng: np.random.Generator, mode: str = "base",
        det=None, thr: float | None = None, record: bool = False):
    """Advance every episode in `st` to termination. mode in {base, gated, heur, oracle}.
    If record, return the list of prefix snapshots (decision points at non-readable views)."""
    snaps = []
    for _ in range(2 * B + 2):
        act = ~st["done"]
        if not act.any():
            break
        readable = st["side"] <= R
        vis = in_view(st)

        # readable & visible -> read
        idx = np.flatnonzero(act & readable & vis)
        if idx.size:
            st["succ"][idx] = rng.random(idx.size) < P_READ
            st["done"][idx] = True; st["acts"][idx] += 1
        # readable & not visible -> base policy: re-crop w.p. rho (if budget) else guess
        idx = np.flatnonzero(act & readable & ~vis)
        if idx.size:
            rc = (rng.random(idx.size) < rho) & (st["b"][idx] >= 1)
            recrop(st, idx[rc]); guess(st, idx[~rc], rng)
        # non-readable & no budget -> guess
        idx = np.flatnonzero(act & ~readable & (st["b"] == 0))
        if idx.size:
            guess(st, idx, rng)
        # non-readable with budget -> decision point
        idx = np.flatnonzero(act & ~readable & (st["b"] >= 1))
        if idx.size:
            cues = quadrant_cues(st, idx, sigma, rng)
            feats = features(st, idx, cues)
            if record:
                snaps.append(dict(ep=idx.copy(), feats=feats,
                                  state={k: st[k][idx].copy() for k in ("x0", "y0", "side", "b", "tx", "ty", "nrec")}))
            can = (st["b"][idx] >= MIN_B) & (feats[:, 0] >= 1)
            if mode == "gated":
                p = det(feats)
                inter = can & (p > 0.5)
            elif mode == "heur":
                inter = can & (feats[:, 4] < thr)
            elif mode == "oracle":
                inter = can & ~vis[idx]
            else:
                inter = np.zeros(idx.size, bool)
            recrop(st, idx[inter])
            z = idx[~inter]; cz = cues[~inter]
            if z.size:
                logits = cz / TAU
                logits -= logits.max(1, keepdims=True)
                p = np.exp(logits); p /= p.sum(1, keepdims=True)
                q = (p.cumsum(1) > rng.random((z.size, 1))).argmax(1)
                srt = np.sort(cz, axis=1)
                k = st["nz"][z]
                st["hm"][z, k] = srt[:, -1] - srt[:, -2]
                st["hc"][z, k] = cz[np.arange(z.size), q]
                st["nz"][z] += 1
                half = st["side"][z] // 2
                st["x0"][z] += (q % 2) * half; st["y0"][z] += (q // 2) * half
                st["side"][z] = half; st["b"][z] -= 1; st["acts"][z] += 1
    assert st["done"].all()
    return snaps


def continuation_prob(state: dict, m: int, sigma: float, rho: float, rng: np.random.Generator) -> np.ndarray:
    """P(success | prefix) from m base-policy continuations per snapshot state."""
    n = state["x0"].size
    st = new_states(n * m, rng)
    for k, v in state.items():
        st[k] = np.repeat(v, m)
    run(st, sigma, rho, rng, mode="base")
    return st["succ"].reshape(n, m).mean(1)


def gather(snaps: list) -> tuple[np.ndarray, np.ndarray, dict]:
    feats = np.concatenate([s["feats"] for s in snaps])
    ep = np.concatenate([s["ep"] for s in snaps])
    state = {k: np.concatenate([s["state"][k] for s in snaps]) for k in snaps[0]["state"]}
    return feats, ep, state


# ----------------------------------------------------------------------------- detector

class LogReg:
    def __init__(self, X: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 600, lr: float = 0.5):
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-8
        Xs = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        w = np.zeros(Xs.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-Xs @ w))
            w -= lr * (Xs.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0.0])
        self.w = w

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Xs = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        return 1 / (1 + np.exp(-Xs @ self.w))


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    y = y.astype(bool)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    # average ranks over ties
    s_sorted = score[order]
    _, first, counts = np.unique(s_sorted, return_index=True, return_counts=True)
    for f, c in zip(first, counts):
        if c > 1:
            ranks[order[f:f + c]] = f + 1 + (c - 1) / 2
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ----------------------------------------------------------------------------- one cell x seed

def run_cell(sigma: float, rho: float, seed: int, cfg: dict) -> dict:
    rng = np.random.default_rng(seed * 1000 + int(sigma * 10) * 10 + int(rho * 10))

    # training prefixes and labels
    st = new_states(cfg["n_train"], rng)
    f_tr, _, s_tr = gather(run(st, sigma, rho, rng, record=True))
    p_tr = continuation_prob(s_tr, cfg["m_label"], sigma, rho, rng)
    keep = s_tr["b"] >= MIN_B
    y_tr = (p_tr[keep] <= DELTA).astype(float)
    det = LogReg(f_tr[keep], y_tr)
    thr = float(np.median(f_tr[keep & (f_tr[:, 0] >= 1), 4]))

    # held-out prefixes: labels, reference, detector scores
    st = new_states(cfg["n_test"], rng)
    f_te, ep_te, s_te = gather(run(st, sigma, rho, rng, record=True))
    fail_ep = ~st["succ"]
    p_lab = continuation_prob(s_te, cfg["m_label"], sigma, rho, rng)
    p_ref = continuation_prob(s_te, cfg["m_ref"], sigma, rho, rng)
    pre = (s_te["b"] >= MIN_B) & (f_te[:, 0] >= 1)
    y_te = p_ref[pre] <= DELTA
    au_det = auroc(det(f_te[pre]), y_te)
    au_mar = auroc(-f_te[pre, 4], y_te)
    label_agree = float(np.mean((p_lab[pre] <= DELTA) == y_te))

    # K1: failed held-out episodes passing through a dead-end prefix; depth of the first
    de = pre & (p_ref <= DELTA)
    de_eps = np.unique(ep_te[de])
    failed = np.flatnonzero(fail_ep)
    through = np.isin(failed, de_eps).mean() if failed.size else float("nan")
    first_depth = []
    for e in de_eps:
        m = (ep_te == e) & de
        first_depth.append(f_te[m, 0].min())
    med_first = float(np.median(first_depth)) if first_depth else float("nan")

    # paired evaluation of the four arms on common initial states
    init_seed = int(rng.integers(1 << 31))
    out = {}
    for mode in ("base", "gated", "heur", "oracle"):
        r = np.random.default_rng(init_seed)
        st = new_states(cfg["n_eval"], r)
        run(st, sigma, rho, r, mode=mode, det=det, thr=thr)
        out[f"{mode}_succ"] = float(st["succ"].mean())
        out[f"{mode}_acts"] = float(st["acts"].mean())

    return dict(sigma=sigma, rho=rho, seed=seed, n_prefix=int(pre.sum()), frac_de=float(y_te.mean()),
                label_agree=label_agree, auroc_det=au_det, auroc_margin=au_mar,
                fail_through_de=float(through), med_first_de_depth=med_first, heur_thr=thr, **out)


# ----------------------------------------------------------------------------- summary & verdict

def summarize(rows: list[dict]) -> tuple[list[dict], str]:
    cells = sorted({(r["sigma"], r["rho"]) for r in rows})
    summ = []
    for sigma, rho in cells:
        rs = [r for r in rows if r["sigma"] == sigma and r["rho"] == rho]
        med = lambda k: float(np.nanmedian([r[k] for r in rs]))  # noqa: E731
        summ.append(dict(
            sigma=sigma, rho=rho, n_seeds=len(rs), frac_de=med("frac_de"), auroc_det=med("auroc_det"),
            auroc_margin=med("auroc_margin"), fail_through_de=med("fail_through_de"),
            med_first_de_depth=med("med_first_de_depth"),
            base=med("base_succ"), gated=med("gated_succ"), heur=med("heur_succ"), oracle=med("oracle_succ"),
            gain_gated=float(np.median([r["gated_succ"] - r["base_succ"] for r in rs])),
            gain_heur=float(np.median([r["heur_succ"] - r["base_succ"] for r in rs])),
            gain_oracle=float(np.median([r["oracle_succ"] - r["base_succ"] for r in rs])),
            extra_acts_gated=float(np.median([r["gated_acts"] - r["base_acts"] for r in rs])),
        ))
    z = [s for s in summ if s["rho"] == 0.0]
    k1 = (np.mean([s["fail_through_de"] for s in z]) >= 0.5) and (np.median([s["med_first_de_depth"] for s in z]) <= 1)
    k2 = all(s["auroc_det"] >= 0.80 and s["auroc_det"] >= s["auroc_margin"] + 0.05 for s in z)
    k3_cells = sum(s["gain_gated"] >= 0.05 and s["gain_gated"] >= 0.5 * s["gain_oracle"] and s["extra_acts_gated"] <= 1
                   for s in z)
    k3 = k3_cells >= 2
    verdict = True if (k1 and k3) else (None if (k1 or k3) else False)
    text = (f"K1 (rho=0: failed episodes through a dead-end = {np.mean([s['fail_through_de'] for s in z]):.2f} >= 0.5; "
            f"median first dead-end depth = {np.median([s['med_first_de_depth'] for s in z]):.0f} <= 1) = {k1}. "
            f"K2 (rho=0 cells AUROC det/margin: "
            + ", ".join(f"sigma={s['sigma']}: {s['auroc_det']:.3f}/{s['auroc_margin']:.3f}" for s in z)
            + f") = {k2}. K3 (gated gain >= 0.05, >= half oracle gain, <= 1 extra action: {k3_cells}/3 cells; "
            + ", ".join(f"sigma={s['sigma']}: gated +{s['gain_gated']:.3f}, heur +{s['gain_heur']:.3f}, oracle +{s['gain_oracle']:.3f}, "
                        f"acts +{s['extra_acts_gated']:.2f}" for s in z)
            + f") = {k3}.")
    return summ, kill_test(verdict, text)


def print_table(summ: list[dict]) -> None:
    print("\n sigma  rho   frac_de  AUROC(det/margin)  fail->DE  base   gated  heur   oracle  +acts")
    for s in summ:
        print(f" {s['sigma']:<5} {s['rho']:<5} {s['frac_de']:.2f}     {s['auroc_det']:.3f}/{s['auroc_margin']:.3f}        "
              f"{s['fail_through_de']:.2f}     {s['base']:.3f}  {s['gated']:.3f}  {s['heur']:.3f}  {s['oracle']:.3f}   "
              f"{s['extra_acts_gated']:+.2f}")


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
        for sigma in SIGMAS:
            for rho in RHOS:
                for seed in range(cfg["seeds"]):
                    rows.append(run_cell(sigma, rho, seed, cfg))
                r = rows[-1]
                print(f"sigma={sigma} rho={rho} [{time.time() - t0:.0f}s] last seed: base {r['base_succ']:.3f} "
                      f"gated {r['gated_succ']:.3f} oracle {r['oracle_succ']:.3f} AUROC {r['auroc_det']:.3f}", flush=True)
        if not args.smoke:
            save_csv(RESULTS / "runs.csv", rows)
    summ, line = summarize(rows)
    print_table(summ)
    if not args.smoke:
        save_csv(RESULTS / "summary.csv", summ)
        (RESULTS / "verdict.txt").write_text(line + "\n")


if __name__ == "__main__":
    main()
