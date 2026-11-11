"""CV8 phase 3: does the pin survive more boxes, and does it survive a box that no longer fits the object?"""

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
from lib.sam3_utils import DATA, MIN_AREA, Sam3Runner, kept  # noqa: E402
from cv8_lvis import covers, load_lvis, phrase, raster  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "real_ablate"
PRIOR = Path(__file__).resolve().parents[1] / "results" / "real_lvis" / "episodes.csv"
N_IMAGES = 400


def clip_box(box, w, h):
    x0, y0, x1, y1 = box
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(w), x1), min(float(h), y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return [x0, y0, x1, y1]


def scale_box(box, factor, w, h):
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = (x1 - x0) * factor / 2, (y1 - y0) * factor / 2
    return clip_box([cx - bw, cy - bh, cx + bw, cy + bh], w, h)


def shift_box(box, w, h):
    x0, y0, x1, y1 = box
    dx = 0.5 * (x1 - x0)
    return clip_box([x0 + dx, y0, x1 + dx, y1], w, h)


def run(limit: int) -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    cats, images, anns, ids = load_lvis(1)
    pool = {int(r["image"]) for r in load_csv(PRIOR)}
    ids = [i for i in ids if i in pool][:limit]
    if len(ids) < limit:
        raise SystemExit(f"only {len(ids)} phase-2 images on disk; need {limit}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_csv(OUT / "episodes.csv")
    done = {int(r["image"]) for r in rows}
    runner = Sam3Runner()
    started = time.time()
    for n, iid in enumerate(ids):
        if iid in done:
            continue
        info = images[iid]
        image = Image.open(DATA / "val2017" / info["coco_url"].split("/")[-1]).convert("RGB")
        w, h = image.size
        inst = [a for a in anns[iid] if a["area"] >= MIN_AREA]
        inst.sort(key=lambda a: -a["area"])
        anchor = inst[0]
        ax, ay, aw, ah = anchor["bbox"]
        anchor_box = [float(ax), float(ay), float(ax + aw), float(ay + ah)]
        anchor_mask = raster(anchor["segmentation"], w, h)
        others = []
        for a in inst[1:3]:
            x, y, bw, bh = a["bbox"]
            others.append([float(x), float(y), float(x + bw), float(y + bh)])
        absent = random.Random(iid).choice(sorted(info["neg_category_ids"]))
        text = phrase(cats[absent])
        boxes = {
            "b1": [anchor_box],
            "b2": [anchor_box, *others[:1]] if len(others) >= 1 else None,
            "b3": [anchor_box, *others[:2]] if len(others) >= 2 else None,
            "half": [scale_box(anchor_box, 0.5, w, h)],
            "double": [scale_box(anchor_box, 2.0, w, h)],
            "shift": [shift_box(anchor_box, w, h)],
        }
        runner.set_image(image)
        text_presence = runner.prompt(text)[0]
        new = [dict(image=iid, cond="text", presence=text_presence, n_kept=0, covers_anchor=0, n_boxes=0)]
        for cond, boxlist in boxes.items():
            if boxlist is None or any(b is None for b in boxlist):
                continue
            pres, logits, masks = runner.prompt(text, boxlist, [1] * len(boxlist))
            keep = kept(logits, pres, 0.5)
            new.append(dict(image=iid, cond=cond, presence=pres, n_kept=int(keep.sum()),
                            covers_anchor=int(covers(masks, keep, anchor_mask)), n_boxes=len(boxlist)))
        rows.extend(new)
        save_csv(OUT / "episodes.csv", rows)
        print(f"{n:03d} {info['coco_url'].split('/')[-1]} [{time.time() - started:.0f}s]", flush=True)
    verdict(rows)


def mean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def rate(rows, cond):
    sub = [r for r in rows if r["cond"] == cond]
    return mean(r["presence"] >= 0 for r in sub), mean(r["covers_anchor"] for r in sub), len(sub)


def stats(rows):
    out = dict(n_images=len({r["image"] for r in rows}))
    for cond in ("b1", "b2", "b3", "half", "double", "shift"):
        ge0, cover, n = rate(rows, cond)
        out[f"{cond}_ge0"] = ge0
        out[f"{cond}_cover"] = cover
        out[f"{cond}_n"] = n
    return out


def criteria(s):
    a1 = s["b1_ge0"] >= 0.90 and s["b2_ge0"] >= 0.90 and s["b3_ge0"] >= 0.90
    a2 = s["half_ge0"] >= 0.90 and s["double_ge0"] >= 0.90 and s["shift_ge0"] >= 0.90
    a3 = s["half_cover"] <= 0.50 and s["half_ge0"] >= 0.90
    return a1, a2, a3


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
        s["a1"], s["a2"], s["a3"] = criteria(s)
        seeds.append(s)
    k = [sum(bool(s[f"a{i}"]) for s in seeds) for i in (1, 2, 3)]
    passed = [c >= 6 for c in k]
    outcome = True if all(passed) else (None if sum(passed) == 2 else False)
    text = (
        f"A1 (presence >= 0 for 1, 2 and 3 positive boxes, each >= 0.90: {k[0]}/8; "
        f"b1 {point['b1_ge0']:.3f} n={point['b1_n']}, b2 {point['b2_ge0']:.3f} n={point['b2_n']}, b3 {point['b3_ge0']:.3f} n={point['b3_n']}) = {passed[0]}. "
        f"A2 (presence >= 0 for half, double and shifted boxes >= 0.90: {k[1]}/8; "
        f"half {point['half_ge0']:.3f}, double {point['double_ge0']:.3f}, shift {point['shift_ge0']:.3f}) = {passed[1]}. "
        f"A3 (half-scale covers the anchor <= 0.50 while presence stays >= 0.90: {k[2]}/8; cover {point['half_cover']:.3f}, "
        f"double cover {point['double_cover']:.3f}, shift cover {point['shift_cover']:.3f}) = {passed[2]}. "
        f"Images {point['n_images']}. One LVIS verified-absent phrase per image, SAM 3 fp16."
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
