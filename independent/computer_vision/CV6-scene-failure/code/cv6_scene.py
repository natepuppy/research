"""CV6: scene-level failure detection for a feed-forward reconstructor.

Synthetic view-sets. An intruder view comes from another scene. A duplicate view
belongs to the scene but was placed a second time, so appearance still matches
and confidence stays high. A probe on the fused token is compared with anchor
agreement and with confidence. Flagged views are dropped when they look unlike
the scene, and split into their own component when they look like it."""

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

PLACED, MISPLACED, INTRUDER = 0, 1, 2
N_VIEWS = 12
# A genuine view left out of the reconstruction costs more than a bad alignment.
ERR_GOOD, ERR_BAD, ERR_LOST = 3.0, 18.0, 40.0
# Flagged views below this anchor agreement are treated as intruders and dropped.
# Views at or above it still look like the scene, so they become a second component.
ANCHOR_SPLIT = 0.62

# Noise differs by domain so the probe is scored on a held-out family, not a reshuffle.
DOMAINS = [
    dict(name="scannet", fusion_scale=1.00, fusion_bias=0.00, appear_noise=0.20),
    dict(name="dl3dv", fusion_scale=1.20, fusion_bias=0.20, appear_noise=0.22),
    dict(name="tanks", fusion_scale=0.85, fusion_bias=-0.15, appear_noise=0.18),
]

CFG = dict(seeds=8, n_sets=80)
SMOKE = dict(seeds=2, n_sets=40)


def one_set(rng: np.random.Generator, kind: str, domain: dict) -> dict:
    placement = np.zeros(N_VIEWS, dtype=int)
    if kind == "intruder":
        idx = rng.choice(N_VIEWS, int(rng.integers(2, 5)), replace=False)
        placement[idx] = INTRUDER
    elif kind == "duplicate":
        idx = rng.choice(N_VIEWS, 4, replace=False)
        placement[idx] = MISPLACED
    # Fused-token residual: low when the view sits in the scene, high when it does not.
    fusion_mean = np.array([0.25, 1.85, 2.15])[placement]
    fusion = domain["fusion_bias"] + domain["fusion_scale"] * fusion_mean
    fusion = fusion + rng.normal(0.0, 0.38, N_VIEWS)
    # Anchor agreement reads appearance. Duplicates belong to the scene, so they agree.
    anchor = np.where(placement == INTRUDER, 0.50, 0.74)
    anchor = np.clip(anchor + rng.normal(0.0, domain["appear_noise"], N_VIEWS), 0.0, 1.0)
    # Confidence is local ease. A duplicate is confidently wrong; an intruder is only a bit lower.
    conf = np.array([0.72, 0.86, 0.66])[placement] + 0.05 * rng.random(N_VIEWS)
    conf = np.clip(conf + rng.normal(0.0, 0.04, N_VIEWS), 0.0, 1.0)
    noise = rng.normal(0.0, 1.0, (N_VIEWS, 2))
    return dict(kind=kind, domain=domain["name"], placement=placement,
                fusion=fusion, anchor=anchor, conf=conf, noise=noise)


def draw_domain(rng: np.random.Generator, domain: dict, n_per_kind: int) -> list[dict]:
    sets = []
    for kind in ("clean", "intruder", "duplicate"):
        for _ in range(n_per_kind):
            sets.append(one_set(rng, kind, domain))
    return sets


def features(s: dict) -> np.ndarray:
    return np.column_stack([s["fusion"], s["anchor"], s["conf"], s["noise"]])


def fit_probe(sets: list[dict]) -> dict:
    """L2 logistic regression. Sample weights keep the rare failed views in the fit."""
    x = np.concatenate([features(s) for s in sets])
    y = np.concatenate([s["placement"] != PLACED for s in sets]).astype(float)
    mu, sd = x.mean(0), x.std(0) + 1e-6
    z = np.column_stack([(x - mu) / sd, np.ones(len(y))])
    rate = float(np.clip(y.mean(), 1e-3, 1 - 1e-3))
    sw = np.where(y > 0.5, 0.5 / rate, 0.5 / (1.0 - rate))
    w = np.zeros(z.shape[1])
    l2 = 2.0
    for _ in range(20):
        prob = 1.0 / (1.0 + np.exp(-np.clip(z @ w, -20.0, 20.0)))
        var = np.clip(prob * (1.0 - prob), 1e-4, None)
        ww = sw * var
        hess = (z * ww[:, None]).T @ z + l2 * np.eye(z.shape[1])
        grad = z.T @ (sw * (y - prob)) - l2 * w
        w = w + np.linalg.solve(hess, grad)
    return dict(mu=mu, sd=sd, w=w)


def apply_probe(s: dict, model: dict) -> np.ndarray:
    z = (features(s) - model["mu"]) / model["sd"]
    z = np.column_stack([z, np.ones(len(s["fusion"]))])
    return z @ model["w"]


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    y = y.astype(bool)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    _, first, counts = np.unique(score[order], return_index=True, return_counts=True)
    for start, count in zip(first, counts):
        if count > 1:
            ranks[order[start:start + count]] = start + 1 + (count - 1) / 2.0
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def collect(sets: list[dict], kinds: tuple[str, ...], positive) -> dict[str, float]:
    scores = {k: [] for k in ("probe", "anchor", "conf", "fusion")}
    labels = []
    for s in sets:
        if s["kind"] not in kinds:
            continue
        scores["probe"].append(s["probe"])
        scores["anchor"].append(-s["anchor"])
        scores["conf"].append(-s["conf"])
        scores["fusion"].append(s["fusion"])
        labels.append(positive(s["placement"]))
    y = np.concatenate(labels)
    return {k: auroc(np.concatenate(v), y) for k, v in scores.items()}


def set_error(placement: np.ndarray, groups: np.ndarray) -> float:
    """Mean error over genuine views. A view scores ERR_GOOD only in a pure group."""
    errs = []
    for i in np.where(placement != INTRUDER)[0]:
        g = groups[i]
        if g < 0:
            errs.append(ERR_LOST)
            continue
        members = placement[groups == g]
        errs.append(ERR_GOOD if members.size and np.all(members == placement[i]) else ERR_BAD)
    return float(np.mean(errs))


def repair(s: dict, z_thr: float) -> tuple[float, int]:
    """Within-set z-score, so a domain shift in the raw probe does not move the cut."""
    probe = s["probe"]
    z = (probe - probe.mean()) / (probe.std() + 1e-6)
    flag = z >= z_thr
    groups = np.zeros(len(flag), dtype=int)
    groups[flag & (s["anchor"] < ANCHOR_SPLIT)] = -1
    groups[flag & (s["anchor"] >= ANCHOR_SPLIT)] = 1
    forwards = 2 if flag.any() else 1
    full = set_error(s["placement"], np.zeros(len(flag), dtype=int))
    fixed = set_error(s["placement"], groups)
    return (full - fixed) / full, forwards


def fit_z(sets: list[dict]) -> float:
    best_thr, best = 1.0, -1.0
    for thr in np.linspace(0.4, 2.2, 19):
        vals = [repair(s, thr)[0] for s in sets if s["kind"] != "clean"]
        mean = float(np.mean(vals))
        if mean > best:
            best_thr, best = float(thr), mean
    return best_thr


def run_cell(seed: int, cfg: dict) -> dict:
    held = seed % len(DOMAINS)
    rng = np.random.default_rng(1000 + seed)
    train = []
    for i, domain in enumerate(DOMAINS):
        if i != held:
            train += draw_domain(rng, domain, cfg["n_sets"])
    test = draw_domain(rng, DOMAINS[held], cfg["n_sets"])
    model = fit_probe(train)
    for s in train + test:
        s["probe"] = apply_probe(s, model)
    z_thr = fit_z(train)

    intr = collect(test, ("clean", "intruder"), lambda p: p == INTRUDER)
    dup = collect(test, ("clean", "duplicate"), lambda p: p == MISPLACED)
    any_fail = collect(test, ("clean", "intruder", "duplicate"), lambda p: p != PLACED)
    reductions, forwards = [], []
    for s in test:
        if s["kind"] == "clean":
            continue
        red, fwd = repair(s, z_thr)
        reductions.append(red)
        forwards.append(fwd)
    return dict(
        seed=seed, held=DOMAINS[held]["name"], z_thr=z_thr,
        auroc_probe_intr=intr["probe"], auroc_anchor_intr=intr["anchor"],
        auroc_conf_intr=intr["conf"], auroc_fusion_intr=intr["fusion"],
        auroc_probe_dup=dup["probe"], auroc_anchor_dup=dup["anchor"],
        auroc_conf_dup=dup["conf"], auroc_fusion_dup=dup["fusion"],
        auroc_probe_any=any_fail["probe"],
        reduction=float(np.mean(reductions)), forwards=float(np.mean(forwards)),
    )


def summarize(rows: list[dict]) -> tuple[list[dict], str]:
    med = lambda k: float(np.nanmedian([r[k] for r in rows]))  # noqa: E731
    n = len(rows)
    need = 6 if n >= 8 else n
    k1_seeds = sum(
        r["auroc_probe_intr"] >= 0.85 and r["auroc_probe_intr"] >= r["auroc_anchor_intr"] + 0.10
        for r in rows
    )
    k2_seeds = sum(r["auroc_probe_dup"] >= 0.80 for r in rows)
    k3_seeds = sum(r["reduction"] >= 0.40 and r["forwards"] <= 2.0 for r in rows)
    k1, k2, k3 = k1_seeds >= need, k2_seeds >= need, k3_seeds >= need
    if k1 and k2:
        verdict = True
    elif k1 or k2:
        verdict = None
    else:
        verdict = False
    summ = [dict(
        n_seeds=n,
        auroc_probe_intr=med("auroc_probe_intr"), auroc_anchor_intr=med("auroc_anchor_intr"),
        auroc_conf_intr=med("auroc_conf_intr"), auroc_fusion_intr=med("auroc_fusion_intr"),
        auroc_probe_dup=med("auroc_probe_dup"), auroc_anchor_dup=med("auroc_anchor_dup"),
        auroc_conf_dup=med("auroc_conf_dup"), auroc_fusion_dup=med("auroc_fusion_dup"),
        auroc_probe_any=med("auroc_probe_any"),
        reduction=med("reduction"), forwards=med("forwards"), z_thr=med("z_thr"),
        k1_seeds=k1_seeds, k2_seeds=k2_seeds, k3_seeds=k3_seeds,
    )]
    text = (
        f"K1 (held-out intruder AUROC of the probe >= 0.85 and >= anchor + 0.10: {k1_seeds}/{n}; "
        f"probe {med('auroc_probe_intr'):.3f}, anchor {med('auroc_anchor_intr'):.3f}, "
        f"confidence {med('auroc_conf_intr'):.3f}, fusion {med('auroc_fusion_intr'):.3f}) = {k1}. "
        f"K2 (held-out duplicate AUROC of the probe >= 0.80: {k2_seeds}/{n}; "
        f"probe {med('auroc_probe_dup'):.3f}, anchor {med('auroc_anchor_dup'):.3f}, "
        f"confidence {med('auroc_conf_dup'):.3f}, fusion {med('auroc_fusion_dup'):.3f}) = {k2}. "
        f"K3 (mean pose-error reduction >= 40% at <= 2 forwards: {k3_seeds}/{n}; "
        f"reduction {med('reduction'):.2f}, forwards {med('forwards'):.2f}) = {k3}."
    )
    return summ, kill_test(verdict, text)


def print_table(summ: list[dict]) -> None:
    s = summ[0]
    print("\n intruder probe/anchor/conf/fusion   duplicate probe/anchor/conf/fusion   reduction  fwd")
    print(f" {s['auroc_probe_intr']:.3f}/{s['auroc_anchor_intr']:.3f}/"
          f"{s['auroc_conf_intr']:.3f}/{s['auroc_fusion_intr']:.3f}                  "
          f"{s['auroc_probe_dup']:.3f}/{s['auroc_anchor_dup']:.3f}/"
          f"{s['auroc_conf_dup']:.3f}/{s['auroc_fusion_dup']:.3f}                   "
          f"{s['reduction']:.2f}       {s['forwards']:.2f}")


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
        t0 = time.time()
        rows = [run_cell(seed, cfg) for seed in range(cfg["seeds"])]
        print(f"{len(rows)} seeds [{time.time() - t0:.1f}s]", flush=True)
        if not args.smoke:
            save_csv(RESULTS / "runs.csv", rows)
    summ, line = summarize(rows)
    print_table(summ)
    if not args.smoke:
        save_csv(RESULTS / "summary.csv", summ)
        (RESULTS / "verdict.txt").write_text(line + "\n")


if __name__ == "__main__":
    main()
