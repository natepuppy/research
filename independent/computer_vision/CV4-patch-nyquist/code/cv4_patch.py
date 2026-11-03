"""CV4: a patch-size law for unresolvable objects. Synthetic scenes, several vision towers
that differ only in patch size. Errors come from object size and neighbor gap in units of
patches, plus a weak clutter term. Tests whether that unit transfers, and whether giving
every object its own tile beats a global 2x upsample and a zoom-the-largest crop."""

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

PATCHES = [8, 16, 32]
K_OBJ = 4

CFG = dict(seeds=8, n_scenes=4000)
SMOKE = dict(seeds=2, n_scenes=600)


def p_fail(size: np.ndarray, gap: np.ndarray, patch: float, clutter: np.ndarray) -> np.ndarray:
    """Steep in the shorter of (size, gap) measured in patches. Clutter adds a small floor."""
    short = np.minimum(size, gap) / patch
    base = 1.0 / (1.0 + np.exp(6.0 * (short - 1.0)))
    return np.clip(0.92 * base + 0.012 * clutter, 0.0, 0.98)


def make_scenes(n: int, rng: np.random.Generator) -> dict:
    # Log-uniform size and gap: many objects sit below a coarse patch, which is the regime
    # a global 2x upsample does not clear.
    size = np.exp(rng.uniform(np.log(3.0), np.log(96.0), (n, K_OBJ)))
    gap = np.exp(rng.uniform(np.log(3.0), np.log(96.0), (n, K_OBJ)))
    clutter = rng.integers(0, 8, (n, K_OBJ))
    return dict(size=size, gap=gap, clutter=clutter)


def draw_fail(size, gap, patch, clutter, rng) -> np.ndarray:
    return rng.random(size.shape) < p_fail(size, gap, patch, clutter)


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    y = y.astype(bool)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    s_sorted = score[order]
    _, first, counts = np.unique(s_sorted, return_index=True, return_counts=True)
    for f, c in zip(first, counts):
        if c > 1:
            ranks[order[f:f + c]] = f + 1 + (c - 1) / 2.0
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def threshold_in_patches(size, gap, fail, patch: float) -> float:
    """min(size, gap)/P at which the fail rate crosses 0.5."""
    sp = np.minimum(size, gap) / patch
    if fail.size < 50:
        return float("nan")
    grid = np.linspace(0.4, 1.8, 29)
    rates = []
    for t in grid:
        w = np.abs(sp - t) <= 0.08
        rates.append(fail[w].mean() if w.sum() >= 20 else np.nan)
    rates = np.array(rates)
    below = np.where(np.isfinite(rates) & (rates >= 0.5))[0]
    above = np.where(np.isfinite(rates) & (rates < 0.5))[0]
    if below.size == 0 or above.size == 0:
        return float("nan")
    # the crossing is the largest t that is still at or above 0.5, walking upward
    cross = below[below < above.max()]
    return float(grid[cross.max()]) if cross.size else float("nan")


CLEAR = 2.0  # how many patches an intervention must leave the object, past the decision boundary


def _scale_to_clear(size, gap, patch):
    return np.maximum(1.0, CLEAR * patch / np.minimum(size, gap))


def zoom_largest(size, gap, patch):
    """A learned-zoom stand-in: only the biggest object in the scene is rescaled."""
    size2, gap2 = size.copy(), gap.copy()
    big = size.argmax(1)
    rows = np.arange(size.shape[0])
    scale = _scale_to_clear(size2[rows, big], gap2[rows, big], patch)
    size2[rows, big] *= scale
    gap2[rows, big] *= scale
    return size2, gap2


def retile(size, gap, patch):
    """Every object gets its own tile and is rescaled past the decision boundary."""
    scale = _scale_to_clear(size, gap, patch)
    return size * scale, gap * scale


def run_seed(seed: int, cfg: dict) -> list[dict]:
    """One set of scenes, scored by every tower. The scenes are shared so a pooled comparison
    of patch units against raw pixels is meaningful: within one tower the two are monotone."""
    rng = np.random.default_rng(seed * 100)
    sc = make_scenes(cfg["n_scenes"], rng)
    size, gap, clutter = sc["size"], sc["gap"], sc["clutter"]
    rows = []
    parts_p, parts_px, parts_y = [], [], []
    for patch in PATCHES:
        sub = np.random.default_rng(seed * 100 + patch)
        fail = draw_fail(size, gap, patch, clutter, sub)
        flat_fail = fail.ravel()
        short_px = np.minimum(size, gap).ravel()
        short_p = short_px / patch
        conf = (0.15 * clutter + sub.normal(scale=0.5, size=clutter.shape)).ravel()
        flag = np.minimum(size, gap) < patch

        def on_flag(sz, gp, flag=flag, clutter=clutter, patch=patch):
            return float(p_fail(sz, gp, patch, clutter)[flag].mean()) if flag.any() else float("nan")

        zs, zg = zoom_largest(size, gap, patch)
        rs, rg = retile(size, gap, patch)
        rows.append(dict(
            patch=patch, seed=seed, n_scenes=cfg["n_scenes"],
            fail_rate=float(fail.mean()), n_flagged=int(flag.sum()),
            auroc_patch=auroc(-short_p, flat_fail), auroc_pixels=auroc(-short_px, flat_fail),
            auroc_clutter=auroc(clutter.ravel().astype(float), flat_fail),
            auroc_conf=auroc(conf, flat_fail),
            threshold=threshold_in_patches(size.ravel(), gap.ravel(), flat_fail, patch),
            err_base=on_flag(size, gap), err_highres=on_flag(size * 2.0, gap * 2.0),
            err_zoom=on_flag(zs, zg), err_retile=on_flag(rs, rg),
        ))
        parts_p.append(short_p)
        parts_px.append(short_px)
        parts_y.append(flat_fail)
    y = np.concatenate(parts_y)
    au_p = auroc(-np.concatenate(parts_p), y)
    au_x = auroc(-np.concatenate(parts_px), y)
    for row in rows:
        row["auroc_pooled_patch"] = au_p
        row["auroc_pooled_pixels"] = au_x
    return rows


def summarize(rows: list[dict]) -> tuple[list[dict], str]:
    summ = []
    for patch in PATCHES:
        rs = [r for r in rows if r["patch"] == patch]
        med = lambda k, rs=rs: float(np.nanmedian([r[k] for r in rs]))  # noqa: E731
        summ.append(dict(
            patch=patch, n_seeds=len(rs), fail_rate=med("fail_rate"),
            auroc_patch=med("auroc_patch"), auroc_pixels=med("auroc_pixels"),
            auroc_clutter=med("auroc_clutter"), auroc_conf=med("auroc_conf"),
            threshold=med("threshold"),
            err_base=med("err_base"), err_highres=med("err_highres"),
            err_zoom=med("err_zoom"), err_retile=med("err_retile"),
            auroc_pooled_patch=med("auroc_pooled_patch"), auroc_pooled_pixels=med("auroc_pooled_pixels"),
        ))
    pooled_gap = summ[0]["auroc_pooled_patch"] - summ[0]["auroc_pooled_pixels"]
    k1 = (pooled_gap >= 0.05) and all(
        s["auroc_patch"] >= 0.80 and s["auroc_patch"] >= s["auroc_clutter"] + 0.10
        and s["auroc_patch"] >= s["auroc_conf"] + 0.10 for s in summ
    )
    thrs = [s["threshold"] for s in summ]
    k2 = all(np.isfinite(t) and 0.75 <= t <= 1.25 for t in thrs) and (max(thrs) - min(thrs) <= 0.25)
    k3 = all(
        (s["err_retile"] <= s["err_highres"] - 0.08) and (s["err_retile"] <= s["err_zoom"] - 0.08)
        for s in summ
    )
    n_pass = sum((k1, k2, k3))
    verdict = True if n_pass == 3 else (None if n_pass == 2 else False)
    text = (
        "K1 (pooled AUROC of min(size, gap)/P beats raw pixels by >= 0.05, and every tower's "
        f"patch AUROC >= 0.80 and beats clutter and confidence by 0.10; pooled "
        f"{summ[0]['auroc_pooled_patch']:.3f} vs pixels {summ[0]['auroc_pooled_pixels']:.3f}; "
        + ", ".join(f"P={s['patch']}: {s['auroc_patch']:.3f} vs clutter {s['auroc_clutter']:.3f}, "
                    f"conf {s['auroc_conf']:.3f}" for s in summ)
        + f") = {k1}. "
        "K2 (fail-rate crossing of min(size, gap)/P inside [0.75, 1.25] and within 0.25 across towers: "
        + ", ".join(f"P={s['patch']}: {s['threshold']:.2f}" for s in summ)
        + f") = {k2}. "
        "K3 (on sub-patch objects, retile error <= 2x upsample and <= zoom-largest by 0.08 on every tower: "
        + ", ".join(f"P={s['patch']}: base {s['err_base']:.3f}, x2 {s['err_highres']:.3f}, "
                    f"zoom {s['err_zoom']:.3f}, retile {s['err_retile']:.3f}" for s in summ)
        + f") = {k3}."
    )
    return summ, kill_test(verdict, text)


def print_table(summ: list[dict]) -> None:
    print("\n P    fail   AUROC patch/px/clutter/conf   thr    flagged: base   x2     zoom   retile")
    for s in summ:
        print(f" {s['patch']:<4} {s['fail_rate']:.3f}  {s['auroc_patch']:.3f}/{s['auroc_pixels']:.3f}/"
              f"{s['auroc_clutter']:.3f}/{s['auroc_conf']:.3f}   {s['threshold']:.2f}   "
              f"{s['err_base']:.3f}  {s['err_highres']:.3f}  {s['err_zoom']:.3f}  {s['err_retile']:.3f}")


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
        for seed in range(cfg["seeds"]):
            rows.extend(run_seed(seed, cfg))
            print(f"seed={seed} [{time.time() - t0:.1f}s]", flush=True)
        if not args.smoke:
            save_csv(RESULTS / "runs.csv", rows)
    summ, line = summarize(rows)
    print_table(summ)
    if not args.smoke:
        save_csv(RESULTS / "summary.csv", summ)
        (RESULTS / "verdict.txt").write_text(line + "\n")


if __name__ == "__main__":
    main()
