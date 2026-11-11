"""CV8 phase 3: phase-2 protocol on images that were not in it, with the hard gate and the geometry gate."""

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
from lib.sam3_utils import DATA, MIN_AREA, Sam3Runner, iou, kept, sigmoid, union  # noqa: E402
from cv8_lvis import background_box, covers, load_lvis, phrase, raster  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "real_heldout"
PRIOR = Path(__file__).resolve().parents[1] / "results" / "real_lvis" / "episodes.csv"
N_IMAGES = 500
HARD = 0.10


def gate_sets(logits, presence, masks, text_presence, box, shape):
    """Ungated kept masks, hard-gate masks, geometry-gate masks."""
    keep = kept(logits, presence, 0.5)
    text_ok = float(sigmoid(text_presence)) >= HARD
    h, w = shape
    rect = np.zeros(shape, bool)
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    rect[max(0, y0):min(h, max(y0 + 1, y1)), max(0, x0):min(w, max(x0 + 1, x1))] = True
    hard, geom = [], []
    for m, k in zip(masks, keep):
        if not k:
            continue
        if text_ok:
            hard.append(m)
            geom.append(m)
        elif iou(m, rect) < 0.5:
            geom.append(m)
    return keep, hard, geom


def pack(masks, shape):
    if not masks:
        return np.zeros(shape, bool)
    return union(np.stack(masks), [True] * len(masks))


def run(limit: int) -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    cats, images, anns, ids = load_lvis(1)
    prior = {int(r["image"]) for r in load_csv(PRIOR)}
    ids = [i for i in ids if i not in prior][:limit]
    if len(ids) < limit:
        raise SystemExit(f"only {len(ids)} held-out LVIS images on disk; need {limit}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_csv(OUT / "episodes.csv")
    done = {int(r["image"]) for r in rows}
    runner = Sam3Runner()
    started = time.time()
    for n, iid in enumerate(ids):
        if iid in done:
            continue
        rng = random.Random(iid)
        info = images[iid]
        image = Image.open(DATA / "val2017" / info["coco_url"].split("/")[-1]).convert("RGB")
        w, h = image.size
        by_cat: dict[int, list] = {}
        for a in anns[iid]:
            by_cat.setdefault(a["category_id"], []).append(a)
        gt_union = {c: union(np.stack([raster(a["segmentation"], w, h) for a in items]), [True] * len(items)) for c, items in by_cat.items()}
        occupied = np.zeros((h, w), bool)
        for m in gt_union.values():
            occupied |= m
        anchor = max((a for a in anns[iid] if a["area"] >= MIN_AREA), key=lambda a: a["area"])
        anchor_mask = raster(anchor["segmentation"], w, h)
        ax, ay, aw, ah = anchor["bbox"]
        anchor_box = [float(ax), float(ay), float(ax + aw), float(ay + ah)]
        anchor_cat = anchor["category_id"]
        n_gt_anchor = sum(a["area"] >= MIN_AREA for a in by_cat[anchor_cat])
        present = sorted((c for c in by_cat if c not in info["not_exhaustive_category_ids"]), key=lambda c: -sum(a["area"] for a in by_cat[c]))
        absent = rng.sample(sorted(info["neg_category_ids"]), min(3, len(info["neg_category_ids"])))
        bg_box = background_box(occupied, int(round(aw)), int(round(ah)))
        runner.set_image(image)
        text_presence: dict[int, float] = {}
        new = []

        def add(cond, c, pres, n_kept, covers_anchor, iou_cat, n_gt, **extra):
            new.append(dict(image=iid, cond=cond, cat=phrase(cats[c]), freq=cats[c]["frequency"], presence=pres,
                            n_kept=n_kept, covers_anchor=covers_anchor, iou_cat=iou_cat, n_gt=n_gt, **extra))

        for c in present[:3]:
            pres, logits, masks = runner.prompt(phrase(cats[c]))
            text_presence[c] = pres
            keep = kept(logits, pres, 0.5)
            add("T+", c, pres, int(keep.sum()), int(covers(masks, keep, anchor_mask)), iou(union(masks, keep), gt_union[c]),
                sum(a["area"] >= MIN_AREA for a in by_cat[c]))
        if anchor_cat not in text_presence:
            text_presence[anchor_cat] = runner.prompt(phrase(cats[anchor_cat]))[0]
        for c in absent:
            pres, logits, masks = runner.prompt(phrase(cats[c]))
            text_presence[c] = pres
            keep = kept(logits, pres, 0.5)
            add("T-", c, pres, int(keep.sum()), 0, 0.0, 0)
            pres, logits, masks = runner.prompt(phrase(cats[c]), [anchor_box], [1])
            keep, hard, geom = gate_sets(logits, pres, masks, text_presence[c], anchor_box, (h, w))
            gt = gt_union.get(c, np.zeros((h, w), bool))
            add("B+T-", c, pres, int(keep.sum()), int(covers(masks, keep, anchor_mask)), iou(union(masks, keep), gt), 0,
                n_kept_hard=len(hard), n_kept_geom=len(geom), iou_hard=iou(pack(hard, (h, w)), gt), iou_geom=iou(pack(geom, (h, w)), gt),
                gate_presence=text_presence[c])
        pres, logits, masks = runner.prompt(phrase(cats[absent[0]]), [bg_box], [1])
        keep = kept(logits, pres, 0.5)
        add("Bg+T-", absent[0], pres, int(keep.sum()), 0, 0.0, 0)
        pres, logits, masks = runner.prompt(phrase(cats[anchor_cat]), [anchor_box], [1])
        keep, hard, geom = gate_sets(logits, pres, masks, text_presence[anchor_cat], anchor_box, (h, w))
        gt = gt_union[anchor_cat]
        add("B+T+", anchor_cat, pres, int(keep.sum()), int(covers(masks, keep, anchor_mask)), iou(union(masks, keep), gt), n_gt_anchor,
            n_kept_hard=len(hard), n_kept_geom=len(geom), iou_hard=iou(pack(hard, (h, w)), gt), iou_geom=iou(pack(geom, (h, w)), gt),
            gate_presence=text_presence[anchor_cat])
        rows.extend(new)
        save_csv(OUT / "episodes.csv", rows)
        print(f"{n:03d} {info['coco_url'].split('/')[-1]} rows={len(rows)} [{time.time() - started:.0f}s]", flush=True)
    verdict(rows)


def mean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def stats(rows):
    g = lambda cond: [r for r in rows if r["cond"] == cond]  # noqa: E731
    tn, btn, btp, bg = g("T-"), g("B+T-"), g("B+T+"), g("Bg+T-")
    fp, fp_h, fp_g = mean(r["n_kept"] > 0 for r in btn), mean(r["n_kept_hard"] > 0 for r in btn), mean(r["n_kept_geom"] > 0 for r in btn)
    iou_u, iou_h, iou_g = mean(r["iou_cat"] for r in btp), mean(r["iou_hard"] for r in btp), mean(r["iou_geom"] for r in btp)
    mae = mean(abs(r["n_kept"] - r["n_gt"]) for r in btp)
    mae_h = mean(abs(r["n_kept_hard"] - r["n_gt"]) for r in btp)
    return dict(
        n_images=len({r["image"] for r in rows}),
        text_absent_ge0=mean(r["presence"] >= 0 for r in tn),
        box_absent_ge0=mean(r["presence"] >= 0 for r in btn),
        btn_covers=mean(r["covers_anchor"] for r in btn),
        fp=fp, fp_hard=fp_h, fp_geom=fp_g,
        drop_hard=float(1 - fp_h / fp) if fp else float("nan"),
        drop_geom=float(1 - fp_g / fp) if fp else float("nan"),
        iou=iou_u, iou_hard=iou_h, iou_geom=iou_g,
        cost_hard=iou_u - iou_h, cost_geom=iou_u - iou_g,
        bg_ge0=mean(r["presence"] >= 0 for r in bg), bg_any=mean(r["n_kept"] > 0 for r in bg), bg_presence=mean(r["presence"] for r in bg),
        false_count=fp, false_count_hard=fp_h, false_count_drop=float(1 - fp_h / fp) if fp else float("nan"),
        mae=mae, mae_hard=mae_h, mae_cost=mae_h - mae,
    )


def criteria(s):
    h1 = s["text_absent_ge0"] <= 0.10 and s["box_absent_ge0"] >= 0.90
    h2 = s["btn_covers"] >= 0.70
    h3 = s["drop_hard"] >= 0.80 and s["cost_hard"] <= 0.02
    h4 = s["drop_geom"] >= 0.80 and s["cost_geom"] <= 0.02
    h5 = s["bg_ge0"] >= 0.90
    h6 = s["false_count"] >= 0.90 and s["false_count_drop"] >= 0.80 and s["mae_cost"] <= 0.30
    return h1, h2, h3, h4, h5, h6


def verdict(rows):
    point = stats(rows)
    imgs = sorted({r["image"] for r in rows})
    by = {}
    for r in rows:
        by.setdefault(r["image"], []).append(r)
    seeds = []
    for seed in range(8):
        take = np.random.default_rng(seed).choice(len(imgs), size=len(imgs), replace=True)
        s = stats([r for i in take for r in by[imgs[i]]])
        (s["h1"], s["h2"], s["h3"], s["h4"], s["h5"], s["h6"]) = criteria(s)
        seeds.append(s)
    k = [sum(bool(s[f"h{i}"]) for s in seeds) for i in range(1, 7)]
    passed = [c >= 6 for c in k]
    core = passed[:3] + [passed[4]]
    outcome = True if all(core) else (None if sum(core) == 3 else False)
    text = (
        f"H1 ({k[0]}/8; text {point['text_absent_ge0']:.3f}, box {point['box_absent_ge0']:.3f}) = {passed[0]}. "
        f"H2 ({k[1]}/8; covers {point['btn_covers']:.3f}) = {passed[1]}. "
        f"H3 ({k[2]}/8; fp {point['fp']:.3f} -> {point['fp_hard']:.3f}, drop {point['drop_hard']:.3f}; IoU {point['iou']:.3f} -> {point['iou_hard']:.3f}, cost {point['cost_hard']:.3f}) = {passed[2]}. "
        f"H4 ({k[3]}/8; fp {point['fp_geom']:.3f}, drop {point['drop_geom']:.3f}; IoU {point['iou_geom']:.3f}, cost {point['cost_geom']:.3f}) = {passed[3]}. "
        f"H5 ({k[4]}/8; background presence >= 0 {point['bg_ge0']:.3f}, mean {point['bg_presence']:.2f}, any mask {point['bg_any']:.3f}) = {passed[4]}. "
        f"H6 ({k[5]}/8; false count {point['false_count']:.3f} -> {point['false_count_hard']:.3f}; MAE {point['mae']:.3f} -> {point['mae_hard']:.3f}, cost {point['mae_cost']:.3f}) = {passed[5]}. "
        f"Tag uses H1, H2, H3, H5. Images {point['n_images']}. Hard gate tau 0.10. SAM 3 fp16, LVIS verified-absent, held-out images."
    )
    (OUT / "verdict.txt").write_text(kill_test(outcome, text) + "\n")
    save_csv(OUT / "summary.csv", [point])
    save_csv(OUT / "seeds.csv", seeds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=N_IMAGES)
    p.add_argument("--summarize", action="store_true")
    a = p.parse_args()
    if a.summarize:
        verdict(load_csv(OUT / "episodes.csv"))
    else:
        run(a.limit)


if __name__ == "__main__":
    main()
