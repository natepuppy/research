"""CV8 phase 4: the held-out LVIS protocol in fp32 on 100 images, compared episode by episode with the stored fp16 rows."""

from __future__ import annotations

import argparse
import ctypes
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io_utils import kill_test, load_csv, save_csv  # noqa: E402
from lib.sam3_utils import DATA, MIN_AREA, Sam3Runner, kept, sigmoid  # noqa: E402
from cv8_lvis import load_lvis, phrase  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "results"
OUT = ROOT / "real_precision"
FP16 = ROOT / "real_heldout" / "episodes.csv"
N_IMAGES = 100
HARD = 0.10


def run(limit: int) -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    cats, images, anns, _ = load_lvis(1)
    stored = load_csv(FP16)
    half = {(r["image"], r["cond"], r["cat"]): r for r in stored}
    ids = sorted({r["image"] for r in stored})[:limit]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_csv(OUT / "episodes.csv")
    done = {int(r["image"]) for r in rows}
    runner = Sam3Runner(dtype=torch.float32)
    started = time.time()
    for n, iid in enumerate(ids):
        if iid in done:
            continue
        rng = random.Random(iid)
        info = images[iid]
        image = Image.open(DATA / "val2017" / info["coco_url"].split("/")[-1]).convert("RGB")
        anchor = max((a for a in anns[iid] if a["area"] >= MIN_AREA), key=lambda a: a["area"])
        ax, ay, aw, ah = anchor["bbox"]
        anchor_box = [float(ax), float(ay), float(ax + aw), float(ay + ah)]
        anchor_name = phrase(cats[anchor["category_id"]])
        absent = rng.sample(sorted(info["neg_category_ids"]), min(3, len(info["neg_category_ids"])))
        runner.set_image(image)
        new = []

        def add(cond, name, pres, logits, pres16, kept16):
            new.append(dict(image=iid, cond=cond, cat=name, presence=pres, presence16=pres16,
                            n_kept=int(kept(logits, pres, 0.5).sum()), n_kept16=kept16))

        for c in absent:
            name = phrase(cats[c])
            pres, logits, _ = runner.prompt(name)
            ref = half[(iid, "T-", name)]
            add("T-", name, pres, logits, ref["presence"], ref["n_kept"])
            pres, logits, _ = runner.prompt(name, [anchor_box], [1])
            ref = half[(iid, "B+T-", name)]
            add("B+T-", name, pres, logits, ref["presence"], ref["n_kept"])
        ref = half[(iid, "B+T+", anchor_name)]
        pres, logits, _ = runner.prompt(anchor_name)
        add("Ta", anchor_name, pres, logits, ref["gate_presence"], -1)
        pres, logits, _ = runner.prompt(anchor_name, [anchor_box], [1])
        add("B+T+", anchor_name, pres, logits, ref["presence"], ref["n_kept"])
        rows.extend(new)
        save_csv(OUT / "episodes.csv", rows)
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"{n:03d} {info['coco_url'].split('/')[-1]} rows={len(rows)} peak={peak:.2f}GB [{time.time() - started:.0f}s]", flush=True)
    verdict(rows)


def mean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def stats(rows):
    g = lambda cond: [r for r in rows if r["cond"] == cond]  # noqa: E731
    tn, btn = g("T-"), g("B+T-")
    gate = lambda v: float(sigmoid(v)) >= HARD  # noqa: E731
    diff = [abs(r["presence"] - r["presence16"]) for r in rows]
    counted = [r for r in rows if r["n_kept16"] >= 0]
    return dict(
        n_images=len({r["image"] for r in rows}), n_episodes=len(rows),
        text_absent_ge0=mean(r["presence"] >= 0 for r in tn),
        box_absent_ge0=mean(r["presence"] >= 0 for r in btn),
        box_absent_at_clamp=mean(r["presence"] >= 9.99 for r in btn),
        box_absent_any_mask=mean(r["n_kept"] > 0 for r in btn),
        text_absent_ge0_16=mean(r["presence16"] >= 0 for r in tn),
        box_absent_ge0_16=mean(r["presence16"] >= 0 for r in btn),
        gate_agree=mean(gate(r["presence"]) == gate(r["presence16"]) for r in tn),
        sign_agree=mean((r["presence"] >= 0) == (r["presence16"] >= 0) for r in rows),
        n_kept_agree=mean(r["n_kept"] == r["n_kept16"] for r in counted),
        mean_abs_diff=mean(diff), max_abs_diff=float(max(diff)) if diff else float("nan"),
    )


def criteria(s):
    p1 = s["text_absent_ge0"] <= 0.10 and s["box_absent_ge0"] >= 0.90
    p2 = s["gate_agree"] >= 0.95
    p3 = s["mean_abs_diff"] <= 0.5
    return p1, p2, p3


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
        s["p1"], s["p2"], s["p3"] = criteria(s)
        seeds.append(s)
    k = [sum(bool(s[f"p{i}"]) for s in seeds) for i in (1, 2, 3)]
    passed = [c >= 6 for c in k]
    outcome = True if passed[0] and passed[1] else (None if passed[0] else False)
    text = (
        f"P1 (fp32 verified-absent presence >= 0: text-only <= 0.10, with box >= 0.90: {k[0]}/8; text {point['text_absent_ge0']:.3f}, "
        f"box {point['box_absent_ge0']:.3f}; fp16 on the same episodes text {point['text_absent_ge0_16']:.3f}, box {point['box_absent_ge0_16']:.3f}) = {passed[0]}. "
        f"P2 (hard-gate decision identical in fp32 and fp16 >= 0.95 of absent-phrase episodes: {k[1]}/8; {point['gate_agree']:.3f}) = {passed[1]}. "
        f"P3 (mean |fp32 - fp16| presence logit <= 0.5: {k[2]}/8; mean {point['mean_abs_diff']:.4f}, max {point['max_abs_diff']:.3f}) = {passed[2]}. "
        f"fp32 absent phrase + box: at +10 {point['box_absent_at_clamp']:.3f}, any mask {point['box_absent_any_mask']:.3f}. "
        f"Presence sign identical {point['sign_agree']:.3f}; kept-mask count identical {point['n_kept_agree']:.3f}. "
        f"Images {point['n_images']}, episodes {point['n_episodes']}. Tag uses P1 and P2. SAM 3 fp32 vs stored fp16, held-out LVIS."
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
