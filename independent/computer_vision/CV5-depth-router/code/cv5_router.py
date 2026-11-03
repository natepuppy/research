"""CV5: a frozen depth model and a vision-language model fail on different spatial items.
Synthetic items with a depth gap, a size, an occlusion and a detector-miss flag.
Geometry is right when the gap is large, the object is big and the detector saw it.
The VLM is right when the question is semantic or the depth map is flat.
A threshold fit on a few hundred items routes between them."""

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

CFG = dict(seeds=8, n_test=4000, n_val=400)
SMOKE = dict(seeds=2, n_test=1500, n_val=400)


def make_items(n: int, rng: np.random.Generator) -> dict:
    depth_gap = rng.random(n)
    size = rng.random(n)
    occlusion = rng.random(n)
    semantic = rng.random(n) < 0.35  # the detector has no category for this question
    # detector confidence is high only when it saw the object
    det_conf = np.where(semantic, 0.05, 0.45 + 0.5 * size) * (1.0 - 0.4 * occlusion)
    det_conf = np.clip(det_conf + rng.normal(scale=0.03, size=n), 0.0, 1.0)
    return dict(depth_gap=depth_gap, size=size, occlusion=occlusion, semantic=semantic, det_conf=det_conf)


def labels(it: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    geom_p = np.full(it["depth_gap"].shape, 0.12)
    # Loose on purpose: geometry is reliable on a large, obvious slice, not a rare corner.
    easy = (it["depth_gap"] > 0.30) & (it["size"] > 0.20) & (it["occlusion"] < 0.80) & (~it["semantic"])
    geom_p[easy] = 0.92
    vlm_p = np.full(it["depth_gap"].shape, 0.38)
    vlm_p[it["semantic"]] = 0.84
    flat = (it["depth_gap"] < 0.22) & (~it["semantic"])
    vlm_p[flat] = 0.70
    geom_ok = rng.random(len(geom_p)) < geom_p
    vlm_ok = rng.random(len(vlm_p)) < vlm_p
    return geom_ok, vlm_ok


def route_acc(use_geom: np.ndarray, geom_ok: np.ndarray, vlm_ok: np.ndarray) -> float:
    return float(np.where(use_geom, geom_ok, vlm_ok).mean())


def fit_threshold(it: dict, geom_ok: np.ndarray, vlm_ok: np.ndarray):
    """Grid search. Use geometry when the gap, the size and the detector confidence all clear a bar."""
    best, best_acc = (0.45, 0.35, 0.4), -1.0
    for gap_t in np.linspace(0.25, 0.70, 10):
        for size_t in np.linspace(0.15, 0.60, 10):
            for conf_t in np.linspace(0.15, 0.70, 12):
                use = (it["depth_gap"] >= gap_t) & (it["size"] >= size_t) & (it["det_conf"] >= conf_t)
                acc = route_acc(use, geom_ok, vlm_ok)
                if acc > best_acc:
                    best, best_acc = (float(gap_t), float(size_t), float(conf_t)), acc
    return best, best_acc


def apply_threshold(it: dict, thr) -> np.ndarray:
    gap_t, size_t, conf_t = thr
    return (it["depth_gap"] >= gap_t) & (it["size"] >= size_t) & (it["det_conf"] >= conf_t)


def run_cell(seed: int, cfg: dict) -> dict:
    rng = np.random.default_rng(seed + 17)
    val = make_items(cfg["n_val"], rng)
    g_val, v_val = labels(val, rng)
    thr, val_acc = fit_threshold(val, g_val, v_val)

    rng_te = np.random.default_rng(seed + 900)
    te = make_items(cfg["n_test"], rng_te)
    g, v = labels(te, rng_te)
    use = apply_threshold(te, thr)
    routed = np.where(use, g, v)

    only_g = g & ~v
    only_v = v & ~g
    exactly_one = only_g | only_v
    gap_g = float(np.median(te["depth_gap"][only_g])) if only_g.any() else float("nan")
    gap_v = float(np.median(te["depth_gap"][only_v])) if only_v.any() else float("nan")

    # the threshold a test-set search would have picked, to see what the val fit leaves on the table
    oracle_thr, _ = fit_threshold(te, g, v)
    oracle = route_acc(apply_threshold(te, oracle_thr), g, v)
    gain = float(routed.mean()) - max(float(g.mean()), float(v.mean()))
    oracle_gain = oracle - max(float(g.mean()), float(v.mean()))

    return dict(
        seed=seed,
        geom_acc=float(g.mean()), vlm_acc=float(v.mean()), router_acc=float(routed.mean()),
        oracle_acc=oracle, gain=gain, oracle_gain=oracle_gain,
        frac_exactly_one=float(exactly_one.mean()),
        median_gap_geom_only=gap_g, median_gap_vlm_only=gap_v,
        val_acc=val_acc, gap_t=thr[0], size_t=thr[1], conf_t=thr[2],
        frac_routed_to_geom=float(use.mean()),
    )


def summarize(rows: list[dict]) -> tuple[list[dict], str]:
    med = lambda k: float(np.nanmedian([r[k] for r in rows]))  # noqa: E731
    n = len(rows)
    need = 6 if n >= 8 else n
    k1_seeds = sum(
        r["frac_exactly_one"] >= 0.30 and (r["median_gap_geom_only"] - r["median_gap_vlm_only"]) >= 0.15
        for r in rows
    )
    k1 = k1_seeds >= need
    k2_seeds = sum(r["gain"] >= 0.05 for r in rows)
    k2 = k2_seeds >= need
    # the val-fit router captures most of the gain an oracle threshold would have
    k3_seeds = sum(r["oracle_gain"] > 0 and r["gain"] >= 0.8 * r["oracle_gain"] for r in rows)
    k3 = k3_seeds >= need
    if k1 and k2:
        verdict = True
    elif k1 or k2:
        verdict = None
    else:
        verdict = False
    summ = [dict(
        n_seeds=n, geom_acc=med("geom_acc"), vlm_acc=med("vlm_acc"), router_acc=med("router_acc"),
        oracle_acc=med("oracle_acc"), gain=med("gain"), oracle_gain=med("oracle_gain"),
        frac_exactly_one=med("frac_exactly_one"),
        median_gap_geom_only=med("median_gap_geom_only"), median_gap_vlm_only=med("median_gap_vlm_only"),
        gap_t=med("gap_t"), size_t=med("size_t"), conf_t=med("conf_t"),
        frac_routed_to_geom=med("frac_routed_to_geom"),
        k1_seeds=k1_seeds, k2_seeds=k2_seeds, k3_seeds=k3_seeds,
    )]
    text = (
        f"K1 (exactly one system correct >= 0.30 and geometry-only depth gap exceeds VLM-only by >= 0.15: "
        f"{k1_seeds}/{n}; median exactly-one {med('frac_exactly_one'):.2f}, "
        f"gap geom-only {med('median_gap_geom_only'):.2f} vs vlm-only {med('median_gap_vlm_only'):.2f}) = {k1}. "
        f"K2 (router beats the better system by >= 0.05: {k2_seeds}/{n}; "
        f"geom {med('geom_acc'):.3f}, vlm {med('vlm_acc'):.3f}, router {med('router_acc'):.3f}, "
        f"gain {med('gain'):+.3f}) = {k2}. "
        f"K3 (val-fit threshold keeps >= 80% of the test-oracle gain: {k3_seeds}/{n}; "
        f"oracle gain {med('oracle_gain'):+.3f}) = {k3}."
    )
    return summ, kill_test(verdict, text)


def print_table(summ: list[dict]) -> None:
    s = summ[0]
    print("\n geom   vlm    router  oracle   gain   exactly-one  gap(geom/vlm)  thr(gap,size,conf)")
    print(f" {s['geom_acc']:.3f}  {s['vlm_acc']:.3f}  {s['router_acc']:.3f}   {s['oracle_acc']:.3f}   "
          f"{s['gain']:+.3f}  {s['frac_exactly_one']:.2f}         "
          f"{s['median_gap_geom_only']:.2f}/{s['median_gap_vlm_only']:.2f}       "
          f"{s['gap_t']:.2f}, {s['size_t']:.2f}, {s['conf_t']:.2f}")


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
