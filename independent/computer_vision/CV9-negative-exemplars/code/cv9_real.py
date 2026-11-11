"""CV9: what a positive box generalizes to, and what a negative box removes. See README."""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io_utils import kill_test, load_csv, save_csv  # noqa: E402
from lib.sam3_utils import DATA, MIN_AREA, Sam3Runner, iou, kept, load_coco, raster, xyxy  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "real"
N_IMAGES = 500
THR = 0.5


def mask_box(m: np.ndarray) -> list[float]:
    ys, xs = np.where(m)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def split(masks, keep, gts):
    """Kept masks into true positives (cover a GT instance) and false positives (large, cover none)."""
    tps, fps = [], []
    for m, k in zip(masks, keep):
        if not k:
            continue
        if any(iou(m, g) > 0.5 for g in gts):
            tps.append(m)
        elif m.sum() >= MIN_AREA:
            fps.append(m)
    return tps, fps


def survives(target, masks, keep) -> bool:
    return any(k and iou(m, target) > 0.5 for m, k in zip(masks, keep))


def run(limit: int) -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    cats, images, anns, ids = load_coco(limit)
    ids = ids[:limit]
    OUT.mkdir(parents=True, exist_ok=True)
    pos_rows = load_csv(OUT / "positives.csv")
    neg_rows = load_csv(OUT / "negatives.csv")
    done = {int(r["image"]) for r in pos_rows} | {int(r["image"]) for r in neg_rows}
    seen = load_csv(OUT / "images_done.csv")
    done |= {int(r["image"]) for r in seen}
    runner = Sam3Runner()
    started = time.time()
    for n, iid in enumerate(ids):
        if iid in done:
            continue
        info = images[iid]
        image = Image.open(DATA / "val2017" / info["file_name"]).convert("RGB")
        w, h = image.size
        by_cat: dict[int, list[dict]] = {}
        for a in anns[iid]:
            by_cat.setdefault(a["category_id"], []).append(a)
        gt_masks = {c: [raster(a["segmentation"], w, h) for a in items] for c, items in by_cat.items()}
        runner.set_image(image)

        # positive generalization
        multi = [c for c in by_cat if sum(a["area"] >= MIN_AREA for a in by_cat[c]) >= 2]
        multi.sort(key=lambda c: -sum(a["area"] for a in by_cat[c]))
        for c in multi[:2]:
            inst = [a for a in by_cat[c] if a["area"] >= MIN_AREA]
            ex = max(inst, key=lambda a: a["area"])
            ex_mask = raster(ex["segmentation"], w, h)
            others_gt = [raster(a["segmentation"], w, h) for a in inst if a is not ex]
            foreign = np.zeros((h, w), bool)
            for k, ms in gt_masks.items():
                if k != c:
                    for m in ms:
                        foreign |= m
            for mode, text, boxes, labels in (("text", cats[c]["name"], None, None), ("box", None, [xyxy(ex["bbox"])], [1]), ("text+box", cats[c]["name"], [xyxy(ex["bbox"])], [1])):
                pres, logits, masks = runner.prompt(text, boxes, labels)
                keep = kept(logits, pres, THR)
                ks = [m for m, k in zip(masks, keep) if k]
                recall = np.mean([any(iou(m, g) > 0.5 for m in ks) for g in others_gt]) if others_gt else float("nan")
                leak = [bool(m.sum()) and np.logical_and(m, foreign).sum() / m.sum() > 0.5 for m in ks if iou(m, ex_mask) <= 0.5]
                pos_rows.append(dict(image=iid, cat=cats[c]["name"], mode=mode, presence=pres, n_kept=len(ks), n_others=len(others_gt),
                                     recall=float(recall), n_leak=int(sum(leak)), leak=float(np.mean(leak)) if leak else float("nan")))

        # negatives on the text pass's own false positives
        for c in sorted(by_cat):
            gts = gt_masks[c]
            pres, logits, masks = runner.prompt(cats[c]["name"])
            keep = kept(logits, pres, THR)
            tps, fps = split(masks, keep, gts)
            if not tps or not fps:
                continue
            target = max(fps, key=lambda m: m.sum())
            others = [f for f in fps if f is not target]
            tbox = mask_box(target)
            abox = mask_box(max(tps, key=lambda m: m.sum()))
            row = dict(image=iid, cat=cats[c]["name"], presence_text=pres, n_tp=len(tps), n_fp=len(fps))
            arms = {
                "native": (cats[c]["name"], [tbox], [0]),
                "anchored": (cats[c]["name"], [abox, tbox], [1, 0]),
            }
            for arm, (text, boxes, labels) in arms.items():
                p2, l2, m2 = runner.prompt(text, boxes, labels)
                k2 = kept(l2, p2, THR)
                row[f"{arm}_presence"] = p2
                row[f"{arm}_target_removed"] = int(not survives(target, m2, k2))
                row[f"{arm}_other_fp_removed"] = float(np.mean([not survives(o, m2, k2) for o in others])) if others else float("nan")
                row[f"{arm}_tp_lost"] = float(np.mean([not survives(t, m2, k2) for t in tps]))
            # subtraction: run the target box as its own positive concept, drop text-pass masks it explains
            p3, l3, m3 = runner.prompt(None, [tbox], [1])
            k3 = kept(l3, p3, THR)
            app = [m for m, k in zip(m3, k3) if k]
            explained = lambda m: any(iou(m, a) > 0.5 for a in app)  # noqa: E731
            row["subtraction_n_app"] = len(app)
            row["subtraction_target_removed"] = int(explained(target))
            row["subtraction_other_fp_removed"] = float(np.mean([explained(o) for o in others])) if others else float("nan")
            row["subtraction_tp_lost"] = float(np.mean([explained(t) for t in tps]))
            neg_rows.append(row)
        seen.append(dict(image=iid))
        save_csv(OUT / "positives.csv", pos_rows)
        save_csv(OUT / "negatives.csv", neg_rows)
        save_csv(OUT / "images_done.csv", seen)
        print(f"{n:03d} {info['file_name']} pos={len(pos_rows)} neg={len(neg_rows)} [{time.time() - started:.0f}s]", flush=True)
    verdict(pos_rows, neg_rows)


def nanmean(xs):
    xs = [x for x in xs if x == x]
    return float(np.mean(xs)) if xs else float("nan")


def stats(pos: list[dict], neg: list[dict]) -> dict:
    out = dict(n_pos_episodes=len({(r["image"], r["cat"]) for r in pos}), n_neg_episodes=len(neg))
    for mode in ("text", "box", "text+box"):
        rs = [r for r in pos if r["mode"] == mode]
        out[f"{mode}_recall"] = nanmean([r["recall"] for r in rs])
        out[f"{mode}_leak"] = nanmean([r["leak"] for r in rs if r["n_kept"] > 0])
    for arm in ("native", "anchored", "subtraction"):
        out[f"{arm}_target_removed"] = nanmean([r[f"{arm}_target_removed"] for r in neg])
        out[f"{arm}_other_fp_removed"] = nanmean([r[f"{arm}_other_fp_removed"] for r in neg])
        out[f"{arm}_tp_lost"] = nanmean([r[f"{arm}_tp_lost"] for r in neg])
    out["native_presence_drop"] = nanmean([r["presence_text"] - r["native_presence"] for r in neg])
    return out


def criteria(s: dict) -> tuple[bool, bool, bool]:
    k1 = s["native_target_removed"] >= 0.50 and s["native_other_fp_removed"] <= s["native_target_removed"] - 0.20
    k2 = s["subtraction_other_fp_removed"] >= s["native_other_fp_removed"] + 0.20 and s["subtraction_tp_lost"] <= 0.10
    k3 = s["box_leak"] >= 0.20 and s["text_leak"] <= 0.05 and s["box_recall"] >= 0.80
    return k1, k2, k3


def verdict(pos: list[dict], neg: list[dict]) -> None:
    point = stats(pos, neg)
    imgs = sorted({r["image"] for r in pos} | {r["image"] for r in neg})
    seeds = []
    for seed in range(8):
        take = np.random.default_rng(seed).choice(len(imgs), size=len(imgs), replace=True)
        count = Counter(imgs[i] for i in take)
        ps = [r for r in pos for _ in range(count.get(r["image"], 0))]
        ns = [r for r in neg for _ in range(count.get(r["image"], 0))]
        s = stats(ps, ns)
        s["k1"], s["k2"], s["k3"] = criteria(s)
        seeds.append(s)
    k = [sum(bool(s[f"k{i}"]) for s in seeds) for i in (1, 2, 3)]
    passed = [c >= 6 for c in k]
    outcome = True if passed[0] and passed[1] else (None if passed[0] or passed[1] else False)
    text = (
        f"K1 (native negative removes its target >= 0.50 and other false positives >= 20 points less often: {k[0]}/8; "
        f"target {point['native_target_removed']:.3f}, others {point['native_other_fp_removed']:.3f}, TP lost {point['native_tp_lost']:.3f}) = {passed[0]}. "
        f"K2 (subtraction removes other false positives >= 20 points more often than native with TP loss <= 0.10: {k[1]}/8; "
        f"others {point['subtraction_other_fp_removed']:.3f}, target {point['subtraction_target_removed']:.3f}, TP lost {point['subtraction_tp_lost']:.3f}) = {passed[1]}. "
        f"K3 (box-only leak >= 0.20, text leak <= 0.05, box recall >= 0.80: {k[2]}/8; box leak {point['box_leak']:.3f} recall {point['box_recall']:.3f}, "
        f"text leak {point['text_leak']:.3f} recall {point['text_recall']:.3f}, text+box leak {point['text+box_leak']:.3f} recall {point['text+box_recall']:.3f}) = {passed[2]}. "
        f"Anchored arm: target {point['anchored_target_removed']:.3f}, others {point['anchored_other_fp_removed']:.3f}, TP lost {point['anchored_tp_lost']:.3f}. "
        f"Negative episodes {point['n_neg_episodes']}, positive episodes {point['n_pos_episodes']}. SAM 3 fp16, COCO val2017."
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
        verdict(load_csv(OUT / "positives.csv"), load_csv(OUT / "negatives.csv"))
    else:
        run(a.limit)


if __name__ == "__main__":
    main()
