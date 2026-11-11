"""CV10: are SAM 3's small-object misses presence vetoes of localized proposals, and does crop presence fix them? See README."""

from __future__ import annotations

import argparse
import ctypes
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io_utils import kill_test, load_csv, save_csv  # noqa: E402
from lib.sam3_utils import (  # noqa: E402
    Sam3Runner, crop_boxes, image_path, iou_many, load_lvis, nms_masks, paste, phrase, raster, sigmoid, size_bin,
)

OUT = Path(__file__).resolve().parents[1] / "results" / "real"
N_IMAGES = 500
THR = 0.5
MAX_PRESENT = 6
MAX_ABSENT = 6
CENTER = 4  # index of the center crop in crop_boxes()


def run(limit: int) -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    cats, images, anns, ids, _ = load_lvis(limit)
    ids = ids[:limit]
    OUT.mkdir(parents=True, exist_ok=True)
    pairs, insts, absents = load_csv(OUT / "pairs.csv"), load_csv(OUT / "instances.csv"), load_csv(OUT / "absent.csv")
    done = {int(r["image"]) for r in pairs}
    runner = Sam3Runner()
    started = time.time()
    for n, iid in enumerate(ids):
        if iid in done:
            continue
        rng = random.Random(iid)
        info = images[iid]
        image = Image.open(image_path(info)).convert("RGB")
        w, h = image.size
        ex = set(info["not_exhaustive_category_ids"])
        by_cat: dict[int, list[dict]] = {}
        for a in anns[iid]:
            if a["category_id"] not in ex:
                by_cat.setdefault(a["category_id"], []).append(a)
        present = sorted(by_cat, key=lambda c: -sum(a["area"] for a in by_cat[c]))[:MAX_PRESENT]
        absent = rng.sample(sorted(info["neg_category_ids"]), min(MAX_ABSENT, len(info["neg_category_ids"])))
        gt = {c: [(a, raster(a["segmentation"], w, h)) for a in by_cat[c]] for c in present}

        # Full pass.
        runner.set_image(image)
        full: dict[int, tuple] = {}
        for c in present:
            full[c] = runner.prompt(phrase(cats[c]))
        for c in absent:
            full[c] = runner.prompt(phrase(cats[c]), want_masks=False)

        # Crop passes: presence per crop for every category; kept masks mapped back for present categories (tiling).
        boxes = crop_boxes(w, h)
        crop_pres: dict[int, list[float]] = {c: [] for c in present + absent}
        crop_maxscore: dict[int, list[float]] = {c: [] for c in present + absent}
        tile_pool: dict[int, list[tuple[np.ndarray, float]]] = {c: [] for c in present}
        for box in boxes:
            runner.set_image(image.crop(box))
            for c in present + absent:
                want = c in gt
                pres, logits, masks = runner.prompt(phrase(cats[c]), want_masks=want)
                scores = sigmoid(logits) * sigmoid(pres)
                crop_pres[c].append(pres)
                crop_maxscore[c].append(float(scores.max()))
                if want:
                    keep = scores > THR
                    if keep.any():
                        for m, s in zip(paste(masks[keep], box, w, h), scores[keep]):
                            tile_pool[c].append((m, float(s)))

        new_pairs, new_insts, new_abs = [], [], []
        for c in present:
            pres, logits, masks = full[c]
            probs = sigmoid(logits)
            scores = probs * sigmoid(pres)
            cp = crop_pres[c]
            p_local = max([pres] + cp)
            p_center = max(pres, cp[CENTER])
            p_corners = max([pres] + cp[:CENTER])
            # Tiling pool: full kept + crop kept, NMS by mask IoU.
            keep_full = scores > THR
            pool_masks = [m for m in masks[keep_full]] + [m for m, _ in tile_pool[c]]
            pool_scores = list(scores[keep_full]) + [s for _, s in tile_pool[c]]
            if pool_masks:
                pm = np.stack(pool_masks)
                kept_idx = nms_masks(pm, np.array(pool_scores), 0.5)
                pm = pm[kept_idx]
            else:
                pm = np.zeros((0, h, w), dtype=bool)
            largest = max(gt[c], key=lambda t: t[0]["area"])
            largest_prob = 0.0
            for a, g in gt[c]:
                ious = iou_many(masks, g)
                matching = ious > 0.5
                match_prob = float(probs[matching].max()) if matching.any() else 0.0
                match_score = match_prob * float(sigmoid(pres))
                if a is largest[0]:
                    largest_prob = match_prob
                hit = match_score > THR
                outcome = "hit" if hit else ("veto" if match_prob > THR else ("weak" if matching.any() else "unlocalized"))
                new_insts.append(dict(
                    image=iid, cat=phrase(cats[c]), freq=cats[c]["frequency"], ann=a["id"], area=a["area"], bin=size_bin(a["area"]),
                    outcome=outcome, best_iou=float(ious.max()) if len(ious) else 0.0, match_prob=match_prob, match_score=match_score,
                    presence=pres, presence_local=p_local, crop_presence_max=max(cp),
                    hit=int(hit), hit_local=int(match_prob * sigmoid(p_local) > THR), hit_center=int(match_prob * sigmoid(p_center) > THR),
                    hit_corners=int(match_prob * sigmoid(p_corners) > THR),
                    hit_tiling=int(bool((iou_many(pm, g) > 0.5).any())) if len(pm) else 0,
                ))
            new_pairs.append(dict(
                image=iid, cat=phrase(cats[c]), freq=cats[c]["frequency"], n_inst=len(gt[c]), largest_area=largest[0]["area"],
                largest_bin=size_bin(largest[0]["area"]), presence=pres, presence_local=p_local, largest_match_prob=largest_prob,
                n_kept_full=int(keep_full.sum()), n_kept_local=int((probs * sigmoid(p_local) > THR).sum()), n_kept_tiling=int(len(pm)),
                **{f"crop_presence_{i}": v for i, v in enumerate(cp)},
            ))
        for c in absent:
            pres, logits, _ = full[c]
            max_prob = float(sigmoid(logits).max())
            max_score = max_prob * float(sigmoid(pres))
            cp = crop_pres[c]
            p_local = max([pres] + cp)
            p_center = max(pres, cp[CENTER])
            p_corners = max([pres] + cp[:CENTER])
            new_abs.append(dict(
                image=iid, cat=phrase(cats[c]), freq=cats[c]["frequency"], presence=pres, presence_local=p_local, crop_presence_max=max(cp),
                max_prob=max_prob, max_score=max_score, fp=int(max_score > THR), fp_local=int(max_prob * sigmoid(p_local) > THR),
                fp_center=int(max_prob * sigmoid(p_center) > THR), fp_corners=int(max_prob * sigmoid(p_corners) > THR),
                fp_tiling=int(max_score > THR or max(crop_maxscore[c]) > THR),
            ))
        pairs.extend(new_pairs)
        insts.extend(new_insts)
        absents.extend(new_abs)
        save_csv(OUT / "pairs.csv", pairs)
        save_csv(OUT / "instances.csv", insts)
        save_csv(OUT / "absent.csv", absents)
        print(f"{n:03d} {image_path(info).name} pairs={len(pairs)} inst={len(insts)} abs={len(absents)} [{time.time() - started:.0f}s]", flush=True)
    verdict(pairs, insts, absents)


def mean(xs) -> float:
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def stats(pairs: list[dict], insts: list[dict], absents: list[dict]) -> dict:
    out: dict = dict(n_images=len({r["image"] for r in pairs}), n_pairs=len(pairs), n_instances=len(insts), n_absent=len(absents))
    for b in ("small", "medium", "large"):
        pb = [r for r in pairs if r["largest_bin"] == b]
        out[f"pairs_{b}"] = len(pb)
        out[f"veto_rate_{b}"] = mean(r["presence"] < 0 for r in pb)
        out[f"loc_rate_{b}"] = mean(r["largest_match_prob"] > THR for r in pb)
        ib = [r for r in insts if r["bin"] == b]
        out[f"inst_{b}"] = len(ib)
        for m in ("hit", "hit_local", "hit_center", "hit_corners", "hit_tiling"):
            out[f"recall_{b}_{m[4:] or 'full'}"] = mean(r[m] for r in ib)
        miss = [r for r in ib if not r["hit"]]
        out[f"miss_{b}"] = len(miss)
        for o in ("veto", "weak", "unlocalized"):
            out[f"share_{o}_{b}"] = mean(r["outcome"] == o for r in miss)
        out[f"veto_crop_positive_{b}"] = mean(r["crop_presence_max"] >= 0 for r in miss if r["outcome"] == "veto")
    for m in ("fp", "fp_local", "fp_center", "fp_corners", "fp_tiling"):
        out[m] = mean(r[m] for r in absents)
    # Threshold baseline: largest tau <= 0.5 whose absent FP rate matches or exceeds local presence's.
    small = [r for r in insts if r["bin"] == "small"]
    abs_scores = np.array([r["max_score"] for r in absents], dtype=float)
    small_scores = np.array([r["match_score"] for r in small], dtype=float)
    tau = THR
    for t in np.arange(THR, 0.0, -0.005):
        if (abs_scores > t).mean() >= out["fp_local"]:
            tau = float(t)
            break
    else:
        tau = 0.005
    out["tau_matched"] = tau
    out["fp_at_tau"] = float((abs_scores > tau).mean()) if len(abs_scores) else float("nan")
    out["recall_small_at_tau"] = float((small_scores > tau).mean()) if len(small_scores) else float("nan")
    out["recall_small_gain_local"] = out["recall_small_local"] - out["recall_small_full"]
    out["fp_rise_local"] = out["fp_local"] - out["fp"]
    out["recall_small_local_minus_tau"] = out["recall_small_local"] - out["recall_small_at_tau"]
    out["mean_kept_full"] = mean(r["n_kept_full"] for r in pairs)
    out["mean_kept_local"] = mean(r["n_kept_local"] for r in pairs)
    out["mean_kept_tiling"] = mean(r["n_kept_tiling"] for r in pairs)
    return out


def criteria(s: dict) -> tuple[bool, bool, bool]:
    rise = s["veto_rate_small"] - s["veto_rate_large"]
    drop = s["loc_rate_large"] - s["loc_rate_small"]
    k1 = s["veto_rate_small"] >= 0.20 and s["veto_rate_small"] >= 3 * s["veto_rate_large"] and drop < rise
    k2 = s["share_veto_small"] >= 0.40
    k3 = s["recall_small_gain_local"] >= 0.10 and s["fp_rise_local"] <= 0.05 and s["recall_small_local_minus_tau"] >= 0.05
    return k1, k2, k3


def verdict(pairs: list[dict], insts: list[dict], absents: list[dict]) -> None:
    point = stats(pairs, insts, absents)
    imgs = sorted({r["image"] for r in pairs})
    by_img = {i: ([], [], []) for i in imgs}
    for r in pairs:
        by_img[r["image"]][0].append(r)
    for r in insts:
        by_img[r["image"]][1].append(r)
    for r in absents:
        by_img[r["image"]][2].append(r)
    seeds = []
    for seed in range(8):
        take = np.random.default_rng(seed).choice(len(imgs), size=len(imgs), replace=True)
        p = [r for i in take for r in by_img[imgs[i]][0]]
        s_ = [r for i in take for r in by_img[imgs[i]][1]]
        a = [r for i in take for r in by_img[imgs[i]][2]]
        s = stats(p, s_, a)
        s["k1"], s["k2"], s["k3"] = criteria(s)
        seeds.append(s)
    k = [sum(bool(s[f"k{i}"]) for s in seeds) for i in (1, 2, 3)]
    passed = [c >= 6 for c in k]
    outcome = True if all(passed) else (None if sum(passed) == 2 else False)
    text = (
        f"K1 (present pairs with presence < 0: small >= 0.20 and >= 3x large, and localization drop < veto rise: {k[0]}/8; "
        f"veto small {point['veto_rate_small']:.3f} medium {point['veto_rate_medium']:.3f} large {point['veto_rate_large']:.3f}; "
        f"largest-instance localized small {point['loc_rate_small']:.3f} medium {point['loc_rate_medium']:.3f} large {point['loc_rate_large']:.3f}) = {passed[0]}. "
        f"K2 (veto share of small misses >= 0.40: {k[1]}/8; small misses {point['miss_small']}: veto {point['share_veto_small']:.3f}, "
        f"weak {point['share_weak_small']:.3f}, unlocalized {point['share_unlocalized_small']:.3f}; "
        f"medium veto {point['share_veto_medium']:.3f}, large veto {point['share_veto_large']:.3f}) = {passed[1]}. "
        f"K3 (local presence: small recall +>= 0.10, absent FP rise <= 0.05, beats matched threshold by >= 0.05: {k[2]}/8; "
        f"small recall full {point['recall_small_full']:.3f} -> local {point['recall_small_local']:.3f} (tiling {point['recall_small_tiling']:.3f}, "
        f"center {point['recall_small_center']:.3f}, corners {point['recall_small_corners']:.3f}); "
        f"absent FP full {point['fp']:.3f} -> local {point['fp_local']:.3f} (tiling {point['fp_tiling']:.3f}, center {point['fp_center']:.3f}, corners {point['fp_corners']:.3f}); "
        f"threshold tau {point['tau_matched']:.3f} gives FP {point['fp_at_tau']:.3f} and small recall {point['recall_small_at_tau']:.3f}) = {passed[2]}. "
        f"Medium recall full {point['recall_medium_full']:.3f} local {point['recall_medium_local']:.3f} tiling {point['recall_medium_tiling']:.3f}; "
        f"large recall full {point['recall_large_full']:.3f} local {point['recall_large_local']:.3f} tiling {point['recall_large_tiling']:.3f}. "
        f"Vetoed small instances whose crop presence is >= 0: {point['veto_crop_positive_small']:.3f}. "
        f"Kept masks per present pair: full {point['mean_kept_full']:.2f}, local {point['mean_kept_local']:.2f}, tiling {point['mean_kept_tiling']:.2f}. "
        f"Images {point['n_images']}, pairs {point['n_pairs']}, instances {point['n_instances']} (small {point['inst_small']}), absent episodes {point['n_absent']}. "
        f"SAM 3 fp16, LVIS v1 val on COCO val2017 images."
    )
    (OUT / "verdict.txt").write_text(kill_test(outcome, text) + "\n")
    save_csv(OUT / "summary.csv", [point])
    save_csv(OUT / "seeds.csv", seeds)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=N_IMAGES)
    p.add_argument("--summarize", action="store_true")
    a = p.parse_args()
    if a.summarize:
        verdict(load_csv(OUT / "pairs.csv"), load_csv(OUT / "instances.csv"), load_csv(OUT / "absent.csv"))
    else:
        run(a.limit)


if __name__ == "__main__":
    main()
