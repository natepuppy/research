"""CV8 phase 2B: a detector's boxes as SAM 3 exemplars. Does SAM 3 confirm the detector's errors, and does the gate fix it?"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io_utils import kill_test, load_csv, save_csv  # noqa: E402
from lib.sam3_utils import DATA, Sam3Runner, kept, load_coco, sigmoid, xyxy  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "real_detector"
N_IMAGES = 500
MODEL_NAME = "Faster R-CNN R50-FPN v2 (torchvision COCO)"
THR = 0.5
DET_SCORE = 0.30
MAX_DET = 8


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def detect(ids, images, cats, limit) -> list[dict]:
    """Run Faster R-CNN once over the images; cached in detections.csv."""
    path = OUT / "detections.csv"
    rows = load_csv(path)
    have = {int(r["image"]) for r in rows}
    todo = [i for i in ids if i not in have]
    if not todo:
        return rows
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
    from torchvision.transforms.functional import to_tensor

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    names = weights.meta["categories"]  # index = COCO category id (with gaps named N/A)
    model = fasterrcnn_resnet50_fpn_v2(weights=weights).cuda().eval()
    valid = {c["name"] for c in cats.values()}
    for n, iid in enumerate(todo):
        image = Image.open(DATA / "val2017" / images[iid]["file_name"]).convert("RGB")
        with torch.inference_mode():
            out = model([to_tensor(image).cuda()])[0]
        count = 0
        for b, s, l in zip(out["boxes"].cpu().tolist(), out["scores"].cpu().tolist(), out["labels"].cpu().tolist()):
            if s < DET_SCORE or count >= MAX_DET:
                break
            name = names[l]
            if name not in valid:
                continue
            rows.append(dict(image=iid, det=count, cls=name, score=s, x0=b[0], y0=b[1], x1=b[2], y1=b[3]))
            count += 1
        if n % 50 == 0:
            print(f"det {n}/{len(todo)}", flush=True)
    save_csv(path, rows)
    del model
    torch.cuda.empty_cache()
    return rows


def run(limit: int, exclude: set[int] | None = None, dest: Path | None = None, protocol: str = "phase2") -> None:
    global OUT
    if dest is not None:
        OUT = dest
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
    cats, images, anns, ids = load_coco(1)
    if exclude:
        ids = [i for i in ids if i not in exclude]
    if len(ids) < limit:
        raise SystemExit(f"only {len(ids)} COCO images available after exclusion; need {limit}")
    ids = ids[:limit]
    OUT.mkdir(parents=True, exist_ok=True)
    dets = detect(ids, images, cats, limit)
    by_img: dict[int, list[dict]] = {}
    for d in dets:
        by_img.setdefault(int(d["image"]), []).append(d)
    name_to_id = {c["name"]: cid for cid, c in cats.items()}
    rows = load_csv(OUT / "episodes.csv")
    done = {int(r["image"]) for r in rows}
    runner = Sam3Runner()
    started = time.time()
    for n, iid in enumerate(ids):
        if iid in done or iid not in by_img:
            if iid not in by_img:
                done.add(iid)
            continue
        info = images[iid]
        image = Image.open(DATA / "val2017" / info["file_name"]).convert("RGB")
        gt_boxes: dict[int, list] = {}
        for a in anns[iid]:
            gt_boxes.setdefault(a["category_id"], []).append(xyxy(a["bbox"]))
        runner.set_image(image)
        text_presence: dict[int, float] = {}
        new = []
        for d in by_img[iid]:
            cid = name_to_id[d["cls"]]
            box = [d["x0"], d["y0"], d["x1"], d["y1"]]
            same = gt_boxes.get(cid, [])
            best = max((box_iou(box, g) for g in same), default=0.0)
            if cid not in text_presence:
                text_presence[cid] = runner.prompt(d["cls"])[0]
            pres, logits, masks = runner.prompt(d["cls"], [box], [1])
            keep = kept(logits, pres, THR)
            keep_g = kept(logits, text_presence[cid], THR)
            new.append(dict(
                image=iid, det=d["det"], cls=d["cls"], det_score=d["score"], gt_iou=best, correct=int(best >= 0.5),
                class_absent=int(len(same) == 0), presence=pres, text_presence=text_presence[cid],
                accepted=int(keep.sum() > 0), accepted_gated=int(keep_g.sum() > 0),
                max_score=float((sigmoid(logits) * sigmoid(pres)).max()),
                max_score_gated=float((sigmoid(logits) * sigmoid(text_presence[cid])).max()),
            ))
        rows.extend(new)
        save_csv(OUT / "episodes.csv", rows)
        print(f"{n:03d} {info['file_name']} dets={len(new)} rows={len(rows)} [{time.time() - started:.0f}s]", flush=True)
    verdict(rows, protocol)


def mean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def auroc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def stats(rows: list[dict]) -> dict:
    correct = [r for r in rows if r["correct"]]
    wrong = [r for r in rows if not r["correct"]]
    absent = [r for r in wrong if r["class_absent"]]
    present_wrong = [r for r in wrong if not r["class_absent"]]
    acc_abs, acc_abs_g = mean(r["accepted"] for r in absent), mean(r["accepted_gated"] for r in absent)
    out = dict(
        n_images=len({r["image"] for r in rows}), n_det=len(rows), n_correct=len(correct), n_wrong=len(wrong), n_class_absent=len(absent),
        accept_correct=mean(r["accepted"] for r in correct), accept_correct_gated=mean(r["accepted_gated"] for r in correct),
        accept_absent=acc_abs, accept_absent_gated=acc_abs_g,
        accept_absent_relative_drop=float(1 - acc_abs_g / acc_abs) if acc_abs else float("nan"),
        accept_present_wrong=mean(r["accepted"] for r in present_wrong), accept_present_wrong_gated=mean(r["accepted_gated"] for r in present_wrong),
        presence_absent_ge0=mean(r["presence"] >= 0 for r in absent), text_presence_absent_ge0=mean(r["text_presence"] >= 0 for r in absent),
        text_presence_correct_ge0=mean(r["text_presence"] >= 0 for r in correct),
        det_score_correct_mean=mean(r["det_score"] for r in correct), det_score_absent_mean=mean(r["det_score"] for r in absent),
        auroc_det_all=auroc([r["det_score"] for r in correct], [r["det_score"] for r in wrong]),
        auroc_sam_all=auroc([r["max_score"] for r in correct], [r["max_score"] for r in wrong]),
        auroc_gated_all=auroc([r["max_score_gated"] for r in correct], [r["max_score_gated"] for r in wrong]),
        auroc_det_absent=auroc([r["det_score"] for r in correct], [r["det_score"] for r in absent]),
        auroc_sam_absent=auroc([r["max_score"] for r in correct], [r["max_score"] for r in absent]),
        auroc_gated_absent=auroc([r["max_score_gated"] for r in correct], [r["max_score_gated"] for r in absent]),
    )
    out["accept_correct_cost"] = out["accept_correct"] - out["accept_correct_gated"]
    return out


def criteria(s: dict, protocol: str) -> tuple[bool, bool]:
    if protocol == "confirm":
        return s["accept_absent"] >= 0.90, s["auroc_gated_absent"] >= 0.90 and s["auroc_gated_absent"] > s["auroc_det_absent"]
    return s["accept_absent"] >= 0.90, s["accept_absent_relative_drop"] >= 0.70 and s["accept_correct_cost"] <= 0.05


def verdict(rows: list[dict], protocol: str = "phase2") -> None:
    point = stats(rows)
    imgs = sorted({r["image"] for r in rows})
    by_img: dict[int, list[dict]] = {}
    for r in rows:
        by_img.setdefault(r["image"], []).append(r)
    seeds = []
    for seed in range(8):
        take = np.random.default_rng(seed).choice(len(imgs), size=len(imgs), replace=True)
        s = stats([r for i in take for r in by_img[imgs[i]]])
        s["d1"], s["d2"] = criteria(s, protocol)
        seeds.append(s)
    k = [sum(bool(s[f"d{i}"]) for s in seeds) for i in (1, 2)]
    passed = [c >= 6 for c in k]
    absent = [r for r in rows if not r["correct"] and r["class_absent"]]
    correct = [r for r in rows if r["correct"]]
    hard_abs = mean(r["accepted"] and sigmoid(r["text_presence"]) >= 0.10 for r in absent)
    hard_cor = mean(r["accepted"] and sigmoid(r["text_presence"]) >= 0.10 for r in correct)
    rule = (
        f"D2 (gate drops that by >= 0.70 relative with correct-acceptance cost <= 0.05: {k[1]}/8; absent {point['accept_absent']:.3f} -> {point['accept_absent_gated']:.3f} "
        f"(drop {point['accept_absent_relative_drop']:.3f}); correct {point['accept_correct']:.3f} -> {point['accept_correct_gated']:.3f}, n={point['n_correct']}) = {passed[1]}. "
        if protocol == "phase2" else
        f"R2 (gated AUROC correct-vs-class-absent >= 0.90 and above the detector score: {k[1]}/8; gated {point['auroc_gated_absent']:.3f}, detector {point['auroc_det_absent']:.3f}) = {passed[1]}. "
    )
    text = (
        f"R1/D1 (class-absent wrong detections accepted >= 0.90 without gate: {k[0]}/8; {point['accept_absent']:.3f}, n={point['n_class_absent']}) = {passed[0]}. "
        + rule +
        f"Hard gate 0.10, reported: class-absent acceptance {hard_abs:.3f}, correct acceptance {hard_cor:.3f}. "
        f"Present-class wrong (confounded by missing labels, n={point['n_wrong'] - point['n_class_absent']}): {point['accept_present_wrong']:.3f} -> {point['accept_present_wrong_gated']:.3f}. "
        f"Exemplar-pass presence >= 0 on class-absent {point['presence_absent_ge0']:.3f}; text-only presence >= 0 on class-absent {point['text_presence_absent_ge0']:.3f}, on correct {point['text_presence_correct_ge0']:.3f}. "
        f"AUROC correct-vs-wrong: detector score {point['auroc_det_all']:.3f}, SAM 3 ungated {point['auroc_sam_all']:.3f}, gated {point['auroc_gated_all']:.3f}; "
        f"correct-vs-class-absent: detector {point['auroc_det_absent']:.3f}, ungated {point['auroc_sam_absent']:.3f}, gated {point['auroc_gated_absent']:.3f}. "
        f"Detector mean score correct {point['det_score_correct_mean']:.3f}, class-absent {point['det_score_absent_mean']:.3f}. "
        f"{point['n_det']} detections on {point['n_images']} images. {MODEL_NAME}, score >= {DET_SCORE}, max {MAX_DET}/image; SAM 3 fp16. Protocol {protocol}."
    )
    (OUT / "verdict.txt").write_text(kill_test(all(passed) if (all(passed) or not any(passed)) else None, text) + "\n")
    save_csv(OUT / "summary.csv", [point])
    save_csv(OUT / "seeds.csv", seeds)


def main() -> None:
    global OUT
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=N_IMAGES)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--exclude", type=str, default="")
    p.add_argument("--dest", type=str, default="")
    p.add_argument("--protocol", choices=("phase2", "confirm"), default="phase2")
    a = p.parse_args()
    if a.protocol == "confirm" and not a.dest:
        a.dest = str(Path(__file__).resolve().parents[1] / "results" / "real_detector_heldout")
    if a.dest:
        OUT = Path(a.dest)
    exclude = {int(r["image"]) for r in load_csv(Path(a.exclude))} if a.exclude else set()
    if a.summarize:
        verdict(load_csv(OUT / "episodes.csv"), a.protocol)
    else:
        run(a.limit, exclude, OUT if a.dest else None, a.protocol)


if __name__ == "__main__":
    main()
