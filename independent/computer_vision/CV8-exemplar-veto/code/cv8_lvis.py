"""CV8 phase 2A: verified-absent phrases from LVIS, a background-box control, gate cost by frequency."""

from __future__ import annotations

import argparse
import ctypes
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io_utils import kill_test, load_csv, save_csv  # noqa: E402
from lib.sam3_utils import DATA, MIN_AREA, Sam3Runner, iou, kept, raster, union  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "real_lvis"
LVIS = Path(r"C:\Users\natha\dev\cv-data\lvis\lvis_v1_val.json")
N_IMAGES = 500
THR = 0.5


def phrase(cat: dict) -> str:
    return cat["synonyms"][0].replace("_", " ")


def load_lvis(min_images: int):
    d = json.loads(LVIS.read_text(encoding="utf-8"))
    cats = {c["id"]: c for c in d["categories"]}
    have = {p.name for p in (DATA / "val2017").glob("*.jpg")}
    images = {i["id"]: i for i in d["images"] if i["coco_url"].split("/")[-1] in have}
    anns: dict[int, list[dict]] = {}
    for a in d["annotations"]:
        if a["image_id"] in images and isinstance(a["segmentation"], list):
            anns.setdefault(a["image_id"], []).append(a)
    ids = [i for i in sorted(images) if i in anns and images[i]["neg_category_ids"] and any(a["area"] >= MIN_AREA for a in anns[i])]
    if len(ids) < min_images:
        raise SystemExit(f"only {len(ids)} LVIS images on disk; need {min_images}")
    return cats, images, anns, ids


def background_box(occupied: np.ndarray, bw: int, bh: int) -> list[float]:
    """Box of size (bw, bh) whose overlap with annotated pixels is smallest, on a 12x12 grid of positions."""
    h, w = occupied.shape
    bw, bh = min(bw, w), min(bh, h)
    integral = np.pad(occupied.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best, best_box = None, None
    for y in np.linspace(0, h - bh, 12).astype(int):
        for x in np.linspace(0, w - bw, 12).astype(int):
            s = integral[y + bh, x + bw] - integral[y, x + bw] - integral[y + bh, x] + integral[y, x]
            if best is None or s < best:
                best, best_box = s, [float(x), float(y), float(x + bw), float(y + bh)]
    return best_box


def covers(masks, keep, gt) -> bool:
    return any(k and iou(m, gt) > 0.5 for m, k in zip(masks, keep))


def run(limit: int) -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    cats, images, anns, ids = load_lvis(limit)
    ids = ids[:limit]
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
        by_cat: dict[int, list[dict]] = {}
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
        exhaustive = [c for c in by_cat if c not in info["not_exhaustive_category_ids"]]
        present = sorted(exhaustive, key=lambda c: -sum(a["area"] for a in by_cat[c]))
        absent = rng.sample(sorted(info["neg_category_ids"]), min(3, len(info["neg_category_ids"])))
        bg_box = background_box(occupied, int(round(aw)), int(round(ah)))
        bg_mask = np.zeros((h, w), bool)
        bg_mask[int(bg_box[1]):int(bg_box[3]), int(bg_box[0]):int(bg_box[2])] = True
        runner.set_image(image)
        text_presence: dict[int, float] = {}
        new = []

        def record(cond, c, pres, logits, masks, gate=None, target=None):
            keep = kept(logits, pres, THR)
            tgt = anchor_mask if target is None else target
            row = dict(image=iid, cond=cond, cat=phrase(cats[c]), freq=cats[c]["frequency"], anchor_cat=phrase(cats[anchor_cat]),
                       anchor_freq=cats[anchor_cat]["frequency"], presence=pres, n_kept=int(keep.sum()),
                       covers_anchor=int(covers(masks, keep, tgt)),
                       iou_cat=iou(union(masks, keep), gt_union[c]) if c in gt_union else 0.0)
            if gate is not None:
                keep_g = kept(logits, gate, THR)
                row.update(gate_presence=gate, n_kept_gated=int(keep_g.sum()),
                           iou_cat_gated=iou(union(masks, keep_g), gt_union[c]) if c in gt_union else 0.0)
            new.append(row)

        for c in present[:3]:
            pres, logits, masks = runner.prompt(phrase(cats[c]))
            text_presence[c] = pres
            record("T+", c, pres, logits, masks)
        if anchor_cat not in text_presence:
            text_presence[anchor_cat] = runner.prompt(phrase(cats[anchor_cat]))[0]
        for c in absent:
            pres, logits, masks = runner.prompt(phrase(cats[c]))
            text_presence[c] = pres
            record("T-", c, pres, logits, masks)
            pres, logits, masks = runner.prompt(phrase(cats[c]), [anchor_box], [1])
            record("B+T-", c, pres, logits, masks, gate=text_presence[c])
        pres, logits, masks = runner.prompt(phrase(cats[absent[0]]), [bg_box], [1])
        record("Bg+T-", absent[0], pres, logits, masks, target=bg_mask)
        pres, logits, masks = runner.prompt(phrase(cats[anchor_cat]), [anchor_box], [1])
        record("B+T+", anchor_cat, pres, logits, masks, gate=text_presence[anchor_cat])
        rows.extend(new)
        save_csv(OUT / "episodes.csv", rows)
        print(f"{n:03d} {info['coco_url'].split('/')[-1]} rows={len(rows)} [{time.time() - started:.0f}s]", flush=True)
    verdict(rows)


def mean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def stats(rows: list[dict]) -> dict:
    g = lambda cond: [r for r in rows if r["cond"] == cond]  # noqa: E731
    tp, tn, btp, btn, bg = g("T+"), g("T-"), g("B+T+"), g("B+T-"), g("Bg+T-")
    fp_before = mean(r["n_kept"] > 0 for r in btn)
    fp_after = mean(r["n_kept_gated"] > 0 for r in btn)
    out = dict(
        n_images=len({r["image"] for r in rows}),
        text_absent_presence_ge0=mean(r["presence"] >= 0 for r in tn),
        text_present_presence_ge0=mean(r["presence"] >= 0 for r in tp),
        box_absent_presence_ge0=mean(r["presence"] >= 0 for r in btn),
        box_absent_at_clamp=mean(r["presence"] >= 9.99 for r in btn),
        btn_covers_anchor=mean(r["covers_anchor"] for r in btn),
        btn_fp_rate=fp_before, btn_fp_rate_gated=fp_after,
        btn_fp_relative_drop=float(1 - fp_after / fp_before) if fp_before else float("nan"),
        btp_iou=mean(r["iou_cat"] for r in btp), btp_iou_gated=mean(r["iou_cat_gated"] for r in btp),
        bg_presence_mean=mean(r["presence"] for r in bg), bg_presence_ge0=mean(r["presence"] >= 0 for r in bg),
        bg_at_clamp=mean(r["presence"] >= 9.99 for r in bg), bg_any_mask=mean(r["n_kept"] > 0 for r in bg),
        bg_covers_box=mean(r["covers_anchor"] for r in bg),
        text_present_mean=mean(r["presence"] for r in tp), text_absent_mean=mean(r["presence"] for r in tn),
    )
    for f in "rcf":
        sub = [r for r in btp if r["anchor_freq"] == f]
        out[f"btp_n_{f}"] = len(sub)
        out[f"btp_iou_{f}"] = mean(r["iou_cat"] for r in sub)
        out[f"btp_iou_gated_{f}"] = mean(r["iou_cat_gated"] for r in sub)
        out[f"text_presence_ge0_present_{f}"] = mean(r["presence"] >= 0 for r in tp if r["freq"] == f)
    out["btp_cost"] = out["btp_iou"] - out["btp_iou_gated"]
    return out


def criteria(s: dict) -> tuple[bool, bool, bool]:
    l1 = s["text_absent_presence_ge0"] <= 0.10 and s["box_absent_presence_ge0"] >= 0.90
    l2 = s["btn_covers_anchor"] >= 0.70
    l3 = s["btn_fp_relative_drop"] >= 0.80 and s["btp_cost"] <= 0.03
    return l1, l2, l3


def verdict(rows: list[dict]) -> None:
    point = stats(rows)
    imgs = sorted({r["image"] for r in rows})
    by_img: dict[int, list[dict]] = {}
    for r in rows:
        by_img.setdefault(r["image"], []).append(r)
    seeds = []
    for seed in range(8):
        take = np.random.default_rng(seed).choice(len(imgs), size=len(imgs), replace=True)
        s = stats([r for i in take for r in by_img[imgs[i]]])
        s["l1"], s["l2"], s["l3"] = criteria(s)
        seeds.append(s)
    k = [sum(bool(s[f"l{i}"]) for s in seeds) for i in (1, 2, 3)]
    passed = [c >= 6 for c in k]
    text = (
        f"L1 (verified-absent presence >= 0: text-only <= 0.10, with box >= 0.90: {k[0]}/8; text {point['text_absent_presence_ge0']:.3f}, "
        f"box {point['box_absent_presence_ge0']:.3f}, at +10 {point['box_absent_at_clamp']:.3f}) = {passed[0]}. "
        f"L2 (anchor covered under absent phrase >= 0.70: {k[1]}/8; {point['btn_covers_anchor']:.3f}) = {passed[1]}. "
        f"L3 (gate drop >= 0.80 and cost <= 0.03: {k[2]}/8; fp {point['btn_fp_rate']:.3f} -> {point['btn_fp_rate_gated']:.3f}, drop {point['btn_fp_relative_drop']:.3f}; "
        f"IoU {point['btp_iou']:.3f} -> {point['btp_iou_gated']:.3f}) = {passed[2]}. "
        f"Gate cost by anchor frequency: r {point['btp_iou_r']:.3f}->{point['btp_iou_gated_r']:.3f} (n={point['btp_n_r']}), "
        f"c {point['btp_iou_c']:.3f}->{point['btp_iou_gated_c']:.3f} (n={point['btp_n_c']}), f {point['btp_iou_f']:.3f}->{point['btp_iou_gated_f']:.3f} (n={point['btp_n_f']}). "
        f"Background box + absent phrase: presence mean {point['bg_presence_mean']:.2f}, >=0 {point['bg_presence_ge0']:.3f}, at +10 {point['bg_at_clamp']:.3f}, "
        f"any mask {point['bg_any_mask']:.3f}, box region covered {point['bg_covers_box']:.3f}. "
        f"Text-only presence >= 0 for present categories {point['text_present_presence_ge0']:.3f} (r {point['text_presence_ge0_present_r']:.3f}, c {point['text_presence_ge0_present_c']:.3f}, f {point['text_presence_ge0_present_f']:.3f}). "
        f"Images {point['n_images']}. SAM 3 fp16, LVIS v1 val annotations on COCO val2017 images."
    )
    tag = all(passed)
    (OUT / "verdict.txt").write_text(kill_test(True if tag else (None if sum(passed) == 2 else False), text) + "\n")
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
