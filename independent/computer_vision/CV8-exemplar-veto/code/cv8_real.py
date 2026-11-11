"""CV8: does a positive exemplar pin SAM 3's presence head? See README for the kill test."""

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
from lib.sam3_utils import DATA, MIN_AREA, Sam3Runner, iou, kept, load_coco, raster, union, xyxy  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "real"
N_IMAGES = 500
THR = 0.5


def covers(masks, keep, gt) -> bool:
    return any(k and iou(m, gt) > 0.5 for m, k in zip(masks, keep))


def run(limit: int) -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    cats, images, anns, ids = load_coco(limit)
    ids = ids[:limit]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_csv(OUT / "episodes.csv")
    done = {int(r["image"]) for r in rows}
    runner = Sam3Runner()
    all_cats = sorted(cats)
    started = time.time()
    for n, iid in enumerate(ids):
        if iid in done:
            continue
        rng = random.Random(iid)
        info = images[iid]
        image = Image.open(DATA / "val2017" / info["file_name"]).convert("RGB")
        w, h = image.size
        by_cat: dict[int, list[dict]] = {}
        for a in anns[iid]:
            by_cat.setdefault(a["category_id"], []).append(a)
        gt_union = {c: union(np.stack([raster(a["segmentation"], w, h) for a in items]), [True] * len(items)) for c, items in by_cat.items()}
        anchor = max((a for a in anns[iid] if a["area"] >= MIN_AREA), key=lambda a: a["area"])
        anchor_mask = raster(anchor["segmentation"], w, h)
        anchor_box = xyxy(anchor["bbox"])
        anchor_cat = anchor["category_id"]
        present = sorted(by_cat, key=lambda c: -sum(a["area"] for a in by_cat[c]))
        absent = rng.sample([c for c in all_cats if c not in by_cat], 3)
        runner.set_image(image)
        text_presence: dict[int, float] = {}
        new = []

        def record(cond, c, pres, logits, masks, gate=None):
            keep = kept(logits, pres, THR)
            row = dict(image=iid, cond=cond, cat=cats[c]["name"], anchor_cat=cats[anchor_cat]["name"], presence=pres,
                       n_kept=int(keep.sum()), covers_anchor=int(covers(masks, keep, anchor_mask)),
                       iou_cat=iou(union(masks, keep), gt_union[c]) if c in gt_union else 0.0)
            if gate is not None:
                keep_g = kept(logits, gate, THR)
                row.update(gate_presence=gate, n_kept_gated=int(keep_g.sum()), covers_anchor_gated=int(covers(masks, keep_g, anchor_mask)),
                           iou_cat_gated=iou(union(masks, keep_g), gt_union[c]) if c in gt_union else 0.0)
            new.append(row)

        for c in present[:3]:
            pres, logits, masks = runner.prompt(cats[c]["name"])
            text_presence[c] = pres
            record("T+", c, pres, logits, masks)
        if anchor_cat not in text_presence:
            text_presence[anchor_cat] = runner.prompt(cats[anchor_cat]["name"])[0]
        for c in absent:
            pres, logits, masks = runner.prompt(cats[c]["name"])
            text_presence[c] = pres
            record("T-", c, pres, logits, masks)
            pres, logits, masks = runner.prompt(cats[c]["name"], [anchor_box], [1])
            record("B+T-", c, pres, logits, masks, gate=text_presence[c])
        for c in absent[:2]:
            pres, logits, masks = runner.prompt(cats[c]["name"], [anchor_box], [0])
            record("B-T-", c, pres, logits, masks)
        pres, logits, masks = runner.prompt(cats[anchor_cat]["name"], [anchor_box], [1])
        record("B+T+", anchor_cat, pres, logits, masks, gate=text_presence[anchor_cat])
        others = [c for c in present if c != anchor_cat]
        if others:
            c = others[0]
            pres, logits, masks = runner.prompt(cats[c]["name"], [anchor_box], [1])
            record("B+Tx", c, pres, logits, masks)
        pres, logits, masks = runner.prompt(None, [anchor_box], [1])
        record("B", anchor_cat, pres, logits, masks)
        rows.extend(new)
        save_csv(OUT / "episodes.csv", rows)
        print(f"{n:03d} {info['file_name']} rows={len(rows)} [{time.time() - started:.0f}s]", flush=True)
    verdict(rows)


def auroc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def stats(rows: list[dict]) -> dict:
    g = lambda cond: [r for r in rows if r["cond"] == cond]  # noqa: E731
    tp, tn, btp, btn, btx, bntn, b = g("T+"), g("T-"), g("B+T+"), g("B+T-"), g("B+Tx"), g("B-T-"), g("B")
    fp_before = np.mean([r["n_kept"] > 0 for r in btn]) if btn else float("nan")
    fp_after = np.mean([r["n_kept_gated"] > 0 for r in btn]) if btn else float("nan")
    return dict(
        n_images=len({r["image"] for r in rows}),
        auroc_text=auroc([r["presence"] for r in tp], [r["presence"] for r in tn]),
        auroc_box=auroc([r["presence"] for r in btp], [r["presence"] for r in btn]),
        btn_covers_anchor=float(np.mean([r["covers_anchor"] for r in btn])) if btn else float("nan"),
        btx_covers_anchor=float(np.mean([r["covers_anchor"] for r in btx])) if btx else float("nan"),
        btx_text_iou_over_half=float(np.mean([r["iou_cat"] > 0.5 for r in btx])) if btx else float("nan"),
        btn_fp_rate=float(fp_before), btn_fp_rate_gated=float(fp_after),
        btn_fp_relative_drop=float(1 - fp_after / fp_before) if fp_before else float("nan"),
        btp_iou=float(np.mean([r["iou_cat"] for r in btp])) if btp else float("nan"),
        btp_iou_gated=float(np.mean([r["iou_cat_gated"] for r in btp])) if btp else float("nan"),
        btn_presence_at_clamp=float(np.mean([r["presence"] >= 9.99 for r in btn])) if btn else float("nan"),
        bntn_presence_at_neg_clamp=float(np.mean([r["presence"] <= -9.99 for r in bntn])) if bntn else float("nan"),
        bntn_any_mask=float(np.mean([r["n_kept"] > 0 for r in bntn])) if bntn else float("nan"),
        b_presence_mean=float(np.mean([r["presence"] for r in b])) if b else float("nan"),
        text_presence_present_mean=float(np.mean([r["presence"] for r in tp])) if tp else float("nan"),
        text_presence_absent_mean=float(np.mean([r["presence"] for r in tn])) if tn else float("nan"),
    )


def criteria(s: dict) -> tuple[bool, bool, bool]:
    k1 = s["auroc_text"] >= 0.95 and s["auroc_box"] <= 0.60
    k2 = s["btn_covers_anchor"] >= 0.70 and s["btx_covers_anchor"] >= 2 * s["btx_text_iou_over_half"]
    k3 = s["btn_fp_relative_drop"] >= 0.80 and (s["btp_iou"] - s["btp_iou_gated"]) <= 0.02
    return k1, k2, k3


def verdict(rows: list[dict]) -> None:
    point = stats(rows)
    imgs = sorted({r["image"] for r in rows})
    by_img: dict[int, list[dict]] = {}
    for r in rows:
        by_img.setdefault(r["image"], []).append(r)
    seeds = []
    for seed in range(8):
        take = np.random.default_rng(seed).choice(len(imgs), size=len(imgs), replace=True)
        sample = [r for i in take for r in by_img[imgs[i]]]
        s = stats(sample)
        s["k1"], s["k2"], s["k3"] = criteria(s)
        seeds.append(s)
    k = [sum(bool(s[f"k{i}"]) for s in seeds) for i in (1, 2, 3)]
    passed = [c >= 6 for c in k]
    outcome = True if all(passed) else (None if sum(passed) == 2 else False)
    text = (
        f"K1 (presence AUROC text-only >= 0.95 and with a positive box <= 0.60: {k[0]}/8; "
        f"text {point['auroc_text']:.3f}, box {point['auroc_box']:.3f}) = {passed[0]}. "
        f"K2 (absent phrase + box covers the anchor >= 0.70: {point['btn_covers_anchor']:.3f}; conflict box {point['btx_covers_anchor']:.3f} vs text {point['btx_text_iou_over_half']:.3f}: {k[1]}/8) = {passed[1]}. "
        f"K3 (gate cuts absent-phrase false-positive episodes by >= 80%: {point['btn_fp_rate']:.3f} -> {point['btn_fp_rate_gated']:.3f}, "
        f"drop {point['btn_fp_relative_drop']:.3f}; consistent IoU {point['btp_iou']:.3f} -> {point['btp_iou_gated']:.3f}: {k[2]}/8) = {passed[2]}. "
        f"Mechanism: absent phrase + positive box at +10 in {point['btn_presence_at_clamp']:.3f} of episodes; absent phrase + negative box at -10 in "
        f"{point['bntn_presence_at_neg_clamp']:.3f}, any mask {point['bntn_any_mask']:.3f}; box-only presence mean {point['b_presence_mean']:.2f}. "
        f"Images {point['n_images']}. SAM 3 fp16, COCO val2017, 80 categories."
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
        verdict(load_csv(OUT / "episodes.csv"))
    else:
        run(a.limit)


if __name__ == "__main__":
    main()
