"""CV11: does SAM 3's presence head follow the scene prior while its queries do not? See README for the kill test."""

from __future__ import annotations

import argparse
import ctypes
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io_utils import kill_test, load_csv, save_csv  # noqa: E402
from lib.sam3_utils import Sam3Runner, image_path, iou_many, load_lvis, phrase, raster, sigmoid, size_bin  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "real"
YOLO_WEIGHTS = Path(r"C:\Users\natha\dev\cv-data\yolov8s-worldv2.pt")
N_IMAGES = 500
MAX_PRESENT = 10
MAX_ABSENT = 10
MIN_COUNT = 20
ALPHA = 5.0
YW_MISS = 0.25


class ScenePrior:
    """Leave-one-image-out PMI from all LVIS val annotations."""

    def __init__(self, all_present: dict[int, set[int]]):
        self.sets = all_present
        self.n = len(all_present)
        self.nc: Counter = Counter()
        self.ncp: Counter = Counter()
        for s in all_present.values():
            self.nc.update(s)
            items = sorted(s)
            for i, a in enumerate(items):
                for b in items[i + 1:]:
                    self.ncp[(a, b)] += 1

    def pair(self, a: int, b: int) -> int:
        return self.ncp[(a, b) if a < b else (b, a)]

    def scores(self, image_id: int, c: int) -> tuple[float, float, int]:
        """(max PMI, mean PMI, leave-one-out count of c) over the categories present in the image other than c."""
        s = self.sets.get(image_id, set())
        n = self.n - 1
        nc = self.nc[c] - (c in s)
        pc = nc / n if n else 0.0
        others = [p for p in s if p != c]
        if not others or pc <= 0:
            return 0.0, 0.0, nc
        vals = []
        for p in others:
            np_ = self.nc[p] - 1
            ncp = self.pair(c, p) - (c in s)
            p_c_given_p = (ncp + ALPHA * pc) / (np_ + ALPHA)
            vals.append(math.log(p_c_given_p / pc))
        return float(max(vals)), float(np.mean(vals)), nc


def clip_similarity(phrases: list[str]) -> np.ndarray:
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    embs = []
    with torch.inference_mode():
        for i in range(0, len(phrases), 256):
            batch = tok([f"a photo of a {p}" for p in phrases[i:i + 256]], padding=True, return_tensors="pt")
            e = model.get_text_features(**batch)
            if not torch.is_tensor(e):  # transformers >= 5 returns a model output; the projected embedding is pooler_output
                e = e.pooler_output if getattr(e, "pooler_output", None) is not None else e[0]
            embs.append(torch.nn.functional.normalize(e, dim=-1))
    e = torch.cat(embs)
    return (e @ e.T).numpy()


def box_iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def run(limit: int) -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    cats, images, anns, ids, all_present = load_lvis(limit)
    ids = ids[:limit]
    cat_ids = sorted(cats)
    index = {c: i for i, c in enumerate(cat_ids)}
    phrases = [phrase(cats[c]) for c in cat_ids]
    prior = ScenePrior(all_present)
    sim = clip_similarity(phrases)
    from ultralytics import YOLOWorld

    yolo = YOLOWorld(str(YOLO_WEIGHTS))
    yolo.set_classes(phrases)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_csv(OUT / "pairs.csv")
    done = {int(r["image"]) for r in rows}
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
        scene = all_present.get(iid, set())

        det = yolo.predict(image, conf=0.001, max_det=1000, verbose=False, device=0)[0].boxes
        det_cls = det.cls.cpu().numpy().astype(int)
        det_conf = det.conf.cpu().numpy()
        det_xyxy = det.xyxy.cpu().numpy()

        runner.set_image(image)
        new = []
        for c in present + absent:
            is_present = c in by_cat
            pres, logits, masks = runner.prompt(phrase(cats[c]), want_masks=is_present)
            probs = sigmoid(logits)
            ctx_max, ctx_mean, n_c = prior.scores(iid, c)
            others = [p for p in scene if p != c]
            text_sim = float(max(sim[index[c], index[p]] for p in others)) if others else 0.0
            mine = det_cls == index[c]
            yw_max = float(det_conf[mine].max()) if mine.any() else 0.0
            row = dict(image=iid, cat=phrase(cats[c]), cat_id=c, freq=cats[c]["frequency"], present=int(is_present), n_c=n_c,
                       context_max=ctx_max, context_mean=ctx_mean, text_sim=text_sim, n_scene=len(others),
                       presence=pres, max_prob=float(probs.max()), yw_max=yw_max)
            if is_present:
                largest = max(by_cat[c], key=lambda a: a["area"])
                g = raster(largest["segmentation"], w, h)
                matching = iou_many(masks, g) > 0.5
                bx, by, bw, bh = largest["bbox"]
                gbox = (bx, by, bx + bw, by + bh)
                yw_m = [float(cf) for cf, bb, m in zip(det_conf, det_xyxy, mine) if m and box_iou(bb, gbox) > 0.5]
                row.update(n_inst=len(by_cat[c]), largest_area=largest["area"], largest_bin=size_bin(largest["area"]),
                           matched_prob=float(probs[matching].max()) if matching.any() else 0.0,
                           yw_matched=max(yw_m) if yw_m else 0.0)
            new.append(row)
        rows.extend(new)
        save_csv(OUT / "pairs.csv", rows)
        print(f"{n:03d} {image_path(info).name} rows={len(rows)} [{time.time() - started:.0f}s]", flush=True)
    verdict(rows)


def rank(x) -> np.ndarray:
    return pd.Series(np.asarray(x, dtype=float)).rank().to_numpy()


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def partial_spearman(x, y, z) -> float:
    if len(x) < 4:
        return float("nan")
    rx, ry, rz = rank(x), rank(y), rank(z)
    design = np.stack([np.ones_like(rz), rz], 1)
    ex = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    ey = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    if np.std(ex) == 0 or np.std(ey) == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def auroc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def mean(xs) -> float:
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def tercile_ratio(present: list[dict], miss, ctx_key: str = "context_max") -> tuple[float, float, float]:
    """Pooled (over size bins) miss rate in the bottom vs top context tercile; returns (bottom, top, ratio)."""
    if len(present) < 9:
        return float("nan"), float("nan"), float("nan")
    ctx = np.array([r[ctx_key] for r in present], float)
    lo, hi = np.quantile(ctx, [1 / 3, 2 / 3])
    num_b = num_t = den = 0.0
    for b in ("small", "medium", "large"):
        rows = [r for r in present if r["largest_bin"] == b]
        bottom = [r for r in rows if r[ctx_key] <= lo]
        top = [r for r in rows if r[ctx_key] > hi]
        if not bottom or not top:
            continue
        num_b += len(rows) * mean(miss(r) for r in bottom)
        num_t += len(rows) * mean(miss(r) for r in top)
        den += len(rows)
    if den == 0:
        return float("nan"), float("nan"), float("nan")
    rb, rt = num_b / den, num_t / den
    return rb, rt, (rb / rt if rt > 0 else float("inf"))


def stats(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["n_c"] >= MIN_COUNT]
    absent = [r for r in scored if not r["present"]]
    present = [r for r in scored if r["present"]]
    out = dict(n_images=len({r["image"] for r in rows}), n_pairs=len(rows), n_scored_absent=len(absent), n_scored_present=len(present))
    ctx_a = [r["context_max"] for r in absent]
    out["rho_ctx_presence_absent"] = spearman(ctx_a, [r["presence"] for r in absent])
    out["partial_rho_absent"] = partial_spearman(ctx_a, [r["presence"] for r in absent], [r["text_sim"] for r in absent])
    out["rho_sim_presence_absent"] = spearman([r["text_sim"] for r in absent], [r["presence"] for r in absent])
    out["rho_ctx_maxprob_absent"] = spearman(ctx_a, [r["max_prob"] for r in absent])
    out["rho_ctxmean_presence_absent"] = spearman([r["context_mean"] for r in absent], [r["presence"] for r in absent])
    out["auroc_ctx_false_alarm"] = auroc([r["context_max"] for r in absent if r["presence"] >= 0], [r["context_max"] for r in absent if r["presence"] < 0])
    out["false_alarm_rate"] = mean(r["presence"] >= 0 for r in absent)
    out["rho_ctx_yw_absent"] = spearman(ctx_a, logit([r["yw_max"] for r in absent]))
    out["partial_rho_yw_absent"] = partial_spearman(ctx_a, logit([r["yw_max"] for r in absent]), [r["text_sim"] for r in absent])
    ctx_p = [r["context_max"] for r in present]
    out["rho_ctx_presence_present"] = spearman(ctx_p, [r["presence"] for r in present])
    out["rho_ctx_matched_present"] = spearman(ctx_p, [r["matched_prob"] for r in present])
    out["rho_ctx_maxprob_present"] = spearman(ctx_p, [r["max_prob"] for r in present])
    out["rho_ctx_yw_present"] = spearman(ctx_p, logit([r["yw_max"] for r in present]))
    out["rho_ctx_yw_matched_present"] = spearman(ctx_p, logit([r["yw_matched"] for r in present]))
    out["veto_rate_present"] = mean(r["presence"] < 0 for r in present)
    out["veto_bottom"], out["veto_top"], out["veto_ratio"] = tercile_ratio(present, lambda r: r["presence"] < 0)
    out["veto_bottom_mean"], out["veto_top_mean"], out["veto_ratio_mean"] = tercile_ratio(present, lambda r: r["presence"] < 0, "context_mean")
    out["yw_miss_bottom"], out["yw_miss_top"], out["yw_miss_ratio"] = tercile_ratio(present, lambda r: r["yw_max"] < YW_MISS)
    out["matched_weak_bottom"], out["matched_weak_top"], out["matched_weak_ratio"] = tercile_ratio(present, lambda r: r["matched_prob"] <= 0.5)
    for f in "rcf":
        a_f = [r for r in rows if not r["present"] and r["freq"] == f]
        p_f = [r for r in rows if r["present"] and r["freq"] == f]
        out[f"n_absent_{f}"], out[f"n_present_{f}"] = len(a_f), len(p_f)
        out[f"rho_ctx_presence_absent_{f}"] = spearman([r["context_max"] for r in a_f], [r["presence"] for r in a_f])
        out[f"veto_rate_present_{f}"] = mean(r["presence"] < 0 for r in p_f)
    return out


def criteria(s: dict) -> tuple[bool, bool, bool]:
    c1 = s["rho_ctx_presence_absent"] >= 0.20 and s["partial_rho_absent"] >= 0.10
    c2 = s["veto_ratio"] >= 1.5 and s["veto_bottom"] >= 0.05
    c3 = (s["rho_ctx_maxprob_absent"] <= 0.5 * s["rho_ctx_presence_absent"]
          and s["rho_ctx_matched_present"] <= 0.5 * s["rho_ctx_presence_present"])
    return c1, c2, c3


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
        s["c1"], s["c2"], s["c3"] = criteria(s)
        seeds.append(s)
    k = [sum(bool(s[f"c{i}"]) for s in seeds) for i in (1, 2, 3)]
    passed = [c >= 6 for c in k]
    outcome = True if all(passed) else (None if sum(passed) == 2 else False)
    text = (
        f"C1 (absent pairs: rho(context, presence) >= 0.20 and partial rho | text similarity >= 0.10: {k[0]}/8; "
        f"rho {point['rho_ctx_presence_absent']:.3f}, partial {point['partial_rho_absent']:.3f}, rho(similarity, presence) {point['rho_sim_presence_absent']:.3f}, "
        f"AUROC context -> false alarm {point['auroc_ctx_false_alarm']:.3f}, false-alarm rate {point['false_alarm_rate']:.3f}) = {passed[0]}. "
        f"C2 (present pairs, within size bins: veto rate bottom context tercile >= 1.5x top and >= 0.05: {k[1]}/8; "
        f"bottom {point['veto_bottom']:.3f}, top {point['veto_top']:.3f}, ratio {point['veto_ratio']:.2f}; overall veto {point['veto_rate_present']:.3f}) = {passed[1]}. "
        f"C3 (queries less contextual than presence: {k[2]}/8; absent rho(context, max query) {point['rho_ctx_maxprob_absent']:.3f} vs presence {point['rho_ctx_presence_absent']:.3f}; "
        f"present rho(context, matched query) {point['rho_ctx_matched_present']:.3f} vs presence {point['rho_ctx_presence_present']:.3f}) = {passed[2]}. "
        f"Mean-PMI variant: rho {point['rho_ctxmean_presence_absent']:.3f}, veto ratio {point['veto_ratio_mean']:.2f}. "
        f"Matched-query weak (<= 0.5) bottom/top tercile {point['matched_weak_bottom']:.3f}/{point['matched_weak_top']:.3f}. "
        f"YOLO-World: absent rho(context, logit max conf) {point['rho_ctx_yw_absent']:.3f} (partial {point['partial_rho_yw_absent']:.3f}); "
        f"present miss (<0.25) bottom/top {point['yw_miss_bottom']:.3f}/{point['yw_miss_top']:.3f} ratio {point['yw_miss_ratio']:.2f}; "
        f"present rho(context, max) {point['rho_ctx_yw_present']:.3f}, rho(context, matched) {point['rho_ctx_yw_matched_present']:.3f}. "
        f"By frequency, absent rho r/c/f {point['rho_ctx_presence_absent_r']:.3f}/{point['rho_ctx_presence_absent_c']:.3f}/{point['rho_ctx_presence_absent_f']:.3f}; "
        f"present veto r/c/f {point['veto_rate_present_r']:.3f}/{point['veto_rate_present_c']:.3f}/{point['veto_rate_present_f']:.3f}. "
        f"Images {point['n_images']}, pairs {point['n_pairs']}, scored absent {point['n_scored_absent']}, scored present {point['n_scored_present']}. "
        f"SAM 3 fp16 and YOLO-World v2-S, LVIS v1 val on COCO val2017 images."
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
        verdict(load_csv(OUT / "pairs.csv"))
    else:
        run(a.limit)


if __name__ == "__main__":
    main()
