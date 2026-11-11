"""CV8 phase 3: exemplar counting on FSC-147. An absent class plus an exemplar box should not count."""

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
from lib.sam3_utils import Sam3Runner, kept, sigmoid  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "real_count"
FSC = Path(r"C:\Users\natha\dev\cv-data\fsc147")
N_IMAGES = 400
HARD = 0.10


def exemplar(entry) -> list[float]:
    """First exemplar box, scaled from the original image into the 384px image."""
    corners = entry["box_examples_coordinates"][0]
    xs = [float(c[0]) for c in corners]
    ys = [float(c[1]) for c in corners]
    rw, rh = float(entry["ratio_w"]), float(entry["ratio_h"])
    return [min(xs) * rw, min(ys) * rh, max(xs) * rw, max(ys) * rh]


def find_images() -> Path:
    for folder in (FSC / "images_384_VarV2", FSC / "FSC147_384_V2" / "images_384_VarV2"):
        if folder.exists():
            return folder
    found = list(FSC.rglob("images_384_VarV2"))
    if found:
        return found[0]
    raise SystemExit(f"no FSC-147 images under {FSC}")


def run(limit: int) -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    ann = json.loads((FSC / "annotation_FSC147_384.json").read_text(encoding="utf-8"))
    split = json.loads((FSC / "Train_Test_Val_FSC_147.json").read_text(encoding="utf-8"))["test"]
    classes = {}
    for line in (FSC / "ImageClasses_FSC147.txt").read_text(encoding="utf-8").splitlines():
        name, cls = line.split("\t")
        classes[name] = cls
    vocab = sorted(set(classes.values()))
    folder = find_images()
    names = [n for n in split if (folder / n).exists() and n in ann and n in classes][:limit]
    if len(names) < min(limit, 50):
        raise SystemExit(f"only {len(names)} FSC-147 test images on disk")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_csv(OUT / "episodes.csv")
    done = {r["image"] for r in rows}
    runner = Sam3Runner()
    started = time.time()
    for n, name in enumerate(names):
        if name in done:
            continue
        entry = ann[name]
        image = Image.open(folder / name).convert("RGB")
        x0, y0, x1, y1 = exemplar(entry)
        w, h = image.size
        box = [max(0.0, x0), max(0.0, y0), min(float(w - 1), x1), min(float(h - 1), y1)]
        gt = len(entry["points"])
        true = classes[name]
        absent = random.Random(name).choice([c for c in vocab if c != true])
        runner.set_image(image)
        p_true, logits_t, masks_t = runner.prompt(true)
        p_abs = runner.prompt(absent)[0]
        new = []
        for cond, text, use_box, text_p in (("text", true, False, p_true), ("box", true, True, p_true), ("absent_box", absent, True, p_abs)):
            if use_box:
                pres, logits, masks = runner.prompt(text, [box], [1])
            else:
                pres, logits, masks = p_true, logits_t, masks_t
            keep = kept(logits, pres, 0.5)
            new.append(dict(image=name, cond=cond, cls=true, absent=absent, gt=gt, presence=pres, text_presence=text_p,
                            n_kept=int(np.sum(keep)), n_kept_hard=int(np.sum(keep)) if float(sigmoid(text_p)) >= HARD else 0))
        rows.extend(new)
        save_csv(OUT / "episodes.csv", rows)
        print(f"{n:03d} {name} gt={gt} [{time.time() - started:.0f}s]", flush=True)
    verdict(rows)


def mean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def stats(rows):
    g = lambda cond: [r for r in rows if r["cond"] == cond]  # noqa: E731
    text, box, absent = g("text"), g("box"), g("absent_box")
    rate = mean(r["n_kept"] >= 1 for r in absent)
    rate_h = mean(r["n_kept_hard"] >= 1 for r in absent)
    mae = mean(abs(r["n_kept"] - r["gt"]) for r in box)
    mae_h = mean(abs(r["n_kept_hard"] - r["gt"]) for r in box)
    return dict(
        n_images=len({r["image"] for r in rows}),
        text_mae=mean(abs(r["n_kept"] - r["gt"]) for r in text),
        box_mae=mae, box_mae_hard=mae_h, mae_cost=mae_h - mae,
        false_count=rate, false_count_hard=rate_h,
        false_drop=float(1 - rate_h / rate) if rate else float("nan"),
        absent_presence_ge0=mean(r["presence"] >= 0 for r in absent),
        gt_mean=mean(r["gt"] for r in box),
    )


def criteria(s):
    f1 = s["false_count"] >= 0.80
    f2 = s["false_drop"] >= 0.70
    f3 = s["mae_cost"] <= 2.0
    return f1, f2, f3


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
        s["f1"], s["f2"], s["f3"] = criteria(s)
        seeds.append(s)
    k = [sum(bool(s[f"f{i}"]) for s in seeds) for i in (1, 2, 3)]
    passed = [c >= 6 for c in k]
    outcome = True if passed[0] and passed[1] else (None if passed[0] else False)
    text = (
        f"F1 (absent class + exemplar counts >= 1 in >= 0.80: {k[0]}/8; {point['false_count']:.3f}, presence >= 0 {point['absent_presence_ge0']:.3f}) = {passed[0]}. "
        f"F2 (hard gate drops that by >= 0.70: {k[1]}/8; {point['false_count']:.3f} -> {point['false_count_hard']:.3f}, drop {point['false_drop']:.3f}) = {passed[1]}. "
        f"F3 (true-class MAE rises by <= 2.0: {k[2]}/8; box MAE {point['box_mae']:.3f} -> {point['box_mae_hard']:.3f}, cost {point['mae_cost']:.3f}; "
        f"text-only MAE {point['text_mae']:.3f}; mean gt count {point['gt_mean']:.2f}) = {passed[2]}. "
        f"Tag uses F1 and F2. Images {point['n_images']}. FSC-147 test split, SAM 3 fp16, hard gate 0.10."
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
