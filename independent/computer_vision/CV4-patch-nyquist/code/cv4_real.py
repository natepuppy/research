"""Real-image test of the patch law. CountBenchQA, Qwen3-VL-2B, YOLO-World boxes.

The synthetic pilot draws failures from the law. This script does not. A failure is a wrong
count from the model. Object size and gap are measured on the pixels the model is shown.
Thresholds below are the README's, restricted to the one tower this card can run at two
resolutions. They were written before the run.

One tower, two resolutions, token side fixed at 32 pixels. A second family was not started.
With one token size, min(size, gap) / patch and min(size, gap) in raw pixels are the same
ranking inside a resolution, so the README's 0.05 pooled gap is not identifiable here and
is logged rather than required. The 2x arm is doubled and then capped at 1024^2, which
stays larger than the 640^2 question image.

K1. AUROC of min(size, gap) / patch against failure is >= 0.80 at each resolution.
K2. The fail rate crosses one half inside [0.75, 1.25] patches at both resolutions, and the
    two crossings differ by <= 0.25.
K3. On at least 20 images whose patch ratio is below 1, retile error is no higher than the
    2x upsample and at least 0.08 below zoom-the-largest.
SUPPORTED if all three. INCONCLUSIVE if exactly two. KILLED otherwise, including when fewer
than 60 images have a box for the asked object (then the curve was not measured).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io_utils import kill_test, load_csv, save_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "real"
PATCH = 16
MERGE = 2
FACTOR = PATCH * MERGE  # pixels per visual token on a side
RESOLUTIONS = {"high": 640 * 640, "low": 320 * 320}
# A doubled 640^2 image is 1280^2. Cap the 2x arm at 1024^2 (1024 visual tokens) so it
# stays strictly larger than the question image and still fits the card. Set before the run.
UPSAMPLE_CAP = 1024 * 1024


def noun_of(question: str) -> str:
    match = re.search(r"how many (.+?)(?: are| is| do| does|\?)", question, re.I)
    return (match.group(1) if match else "object").strip()


def fit_image(image: Image.Image, max_pixels: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = min(1.0, (max_pixels / max(width * height, 1)) ** 0.5)
    nw = max(FACTOR, int(width * scale) // FACTOR * FACTOR)
    nh = max(FACTOR, int(height * scale) // FACTOR * FACTOR)
    if (nw, nh) != (width, height):
        image = image.resize((nw, nh), Image.BICUBIC)
    return image


def edge_gap(boxes: np.ndarray) -> float:
    """Minimum edge-to-edge gap between boxes. 0 if any pair overlaps. inf if one box."""
    if len(boxes) < 2:
        return float("inf")
    best = float("inf")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            ax1, ay1, ax2, ay2 = boxes[i]
            bx1, by1, bx2, by2 = boxes[j]
            dx = max(0.0, max(bx1 - ax2, ax1 - bx2))
            dy = max(0.0, max(by1 - ay2, ay1 - by2))
            gap = max(dx, dy) if dx == 0 or dy == 0 else (dx ** 2 + dy ** 2) ** 0.5
            if dx > 0 and dy > 0:
                gap = (dx ** 2 + dy ** 2) ** 0.5
            elif dx > 0:
                gap = dx
            elif dy > 0:
                gap = dy
            else:
                gap = 0.0
            best = min(best, gap)
    return best


def geometry(boxes: np.ndarray) -> tuple[float, float, float]:
    if len(boxes) == 0:
        return float("nan"), float("nan"), float("nan")
    sides = np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    size = float(np.min(sides))
    gap = edge_gap(boxes)
    usable_gap = size if not np.isfinite(gap) else gap
    return size, float(gap) if np.isfinite(gap) else -1.0, min(size, usable_gap)


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Higher score means more failure-like. label 1 = failed."""
    mask = np.isfinite(score)
    score, label = score[mask], label[mask]
    n_pos, n_neg = int(label.sum()), int((1 - label).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    sorted_score = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_score[j + 1] == sorted_score[i]:
            j += 1
        if j > i:
            mean = ranks[order[i : j + 1]].mean()
            ranks[order[i : j + 1]] = mean
        i = j + 1
    sum_pos = ranks[label == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def crossing(ratio: np.ndarray, failed: np.ndarray) -> float:
    """Patch ratio at which the fail rate crosses 0.5, interpolated across quantile bins."""
    mask = np.isfinite(ratio)
    ratio, failed = ratio[mask], failed[mask]
    if len(ratio) < 40:
        return float("nan")
    edges = np.quantile(ratio, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    edges = np.unique(edges)
    if len(edges) < 3:
        return float("nan")
    centers, rates = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (ratio >= lo) & (ratio <= hi if hi == edges[-1] else ratio < hi)
        if sel.sum() < 8:
            continue
        centers.append(float(np.median(ratio[sel])))
        rates.append(float(failed[sel].mean()))
    for (c0, r0), (c1, r1) in zip(zip(centers, rates), zip(centers[1:], rates[1:])):
        if (r0 - 0.5) * (r1 - 0.5) <= 0 and r0 != r1:
            return float(c0 + (0.5 - r0) / (r1 - r0) * (c1 - c0))
    return float("nan")


def first_int(text: str) -> int | None:
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else None


def load_rows(limit: int) -> list[dict]:
    from datasets import load_dataset

    data = load_dataset("vikhyatk/CountBenchQA", split="test")
    rows = []
    for index, item in enumerate(data):
        if limit and index >= limit:
            break
        image = item["image"].convert("RGB")
        question = item["question"]
        answer = int(item["number"])
        rows.append(dict(index=index, image=image, question=question, answer=answer, noun=noun_of(question)))
    return rows


def detect(detector, image: Image.Image, noun: str) -> np.ndarray:
    detector.set_classes([noun])
    found = detector.predict(image, verbose=False, conf=0.15, device="cpu")[0]
    if found.boxes is None or len(found.boxes) == 0:
        return np.zeros((0, 4), dtype=float)
    return found.boxes.xyxy.cpu().numpy().astype(float)


def ask(model, processor, image: Image.Image, question: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question + " Reply with only the integer."},
        ],
    }]
    # The image is already a multiple of FACTOR. Stop the processor from resizing it again.
    pixels = max(image.width * image.height, FACTOR * FACTOR)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        if hasattr(image_processor, "min_pixels"):
            image_processor.min_pixels = FACTOR * FACTOR
        if hasattr(image_processor, "max_pixels"):
            image_processor.max_pixels = pixels
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    if hasattr(inputs, "to"):
        inputs = inputs.to(model.device)
    else:
        inputs = {key: value.to(model.device) if torch.is_tensor(value) else value for key, value in inputs.items()}
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")
    if not getattr(ask, "logged_grid", False) and "image_grid_thw" in inputs:
        ask.logged_grid = True
        print(f"first grid {inputs['image_grid_thw'].tolist()} image {image.size}", flush=True)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    new_tokens = output[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def retile(image: Image.Image, boxes: np.ndarray) -> Image.Image:
    """One tile per box, each object given at least two tokens of room, stacked horizontally."""
    tiles = []
    for x1, y1, x2, y2 in boxes:
        side = max(2 * FACTOR, x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        left = int(round(cx - side / 2))
        top = int(round(cy - side / 2))
        crop = image.crop((left, top, left + int(side), top + int(side)))
        tiles.append(crop.resize((2 * FACTOR, 2 * FACTOR), Image.BICUBIC))
    if not tiles:
        return image
    canvas = Image.new("RGB", (2 * FACTOR * len(tiles), 2 * FACTOR), (255, 255, 255))
    for i, tile in enumerate(tiles):
        canvas.paste(tile, (i * 2 * FACTOR, 0))
    return canvas


def zoom_largest(image: Image.Image, boxes: np.ndarray) -> Image.Image:
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    x1, y1, x2, y2 = boxes[int(np.argmax(areas))]
    side = max(2 * FACTOR, x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    return image.crop((left, top, left + int(side), top + int(side))).resize((4 * FACTOR, 4 * FACTOR), Image.BICUBIC)


def evaluate(rows: list[dict]) -> None:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from ultralytics import YOLO

    OUT.mkdir(parents=True, exist_ok=True)
    done = {int(r["index"]) for r in load_csv(OUT / "runs.csv")}
    detector = YOLO("yolov8s-worldv2.pt")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    written = list(load_csv(OUT / "runs.csv"))
    started = time.time()
    for row in rows:
        if row["index"] in done:
            continue
        record = dict(index=row["index"], noun=row["noun"], gold=row["answer"], question=row["question"])
        for name, pixels in RESOLUTIONS.items():
            shown = fit_image(row["image"], pixels)
            boxes = detect(detector, shown, row["noun"])
            size, gap, shorter = geometry(boxes)
            text = ask(model, processor, shown, row["question"])
            pred = first_int(text)
            record[f"{name}_pred"] = -1 if pred is None else pred
            record[f"{name}_ok"] = int(pred == row["answer"])
            record[f"{name}_nbox"] = int(len(boxes))
            record[f"{name}_size"] = size
            record[f"{name}_gap"] = gap
            record[f"{name}_ratio"] = shorter / FACTOR
            record[f"{name}_pixels"] = shorter
            record[f"{name}_text"] = text
            if name == "high" and len(boxes) and shorter / FACTOR < 1:
                up = shown.resize((shown.width * 2, shown.height * 2), Image.BICUBIC)
                up_pred = first_int(ask(model, processor, fit_image(up, UPSAMPLE_CAP), row["question"]))
                tile_pred = first_int(ask(model, processor, retile(shown, boxes), f"How many {row['noun']} are there in total?"))
                zoom_pred = first_int(ask(model, processor, zoom_largest(shown, boxes), row["question"]))
                record["up_ok"] = int(up_pred == row["answer"])
                record["tile_ok"] = int(tile_pred == row["answer"])
                record["zoom_ok"] = int(zoom_pred == row["answer"])
            torch.cuda.empty_cache()
        written.append(record)
        save_csv(OUT / "runs.csv", written)
        print(
            f"{row['index']:03d} gold={row['answer']} high={record['high_pred']} "
            f"ratio={record['high_ratio']:.2f} boxes={record['high_nbox']} [{time.time() - started:.0f}s]",
            flush=True,
        )
    verdict(written)


def verdict(rows: list[dict]) -> None:
    rows = [r for r in rows if int(r.get("high_nbox", 0)) > 0]
    failed_high = np.array([1 - int(r["high_ok"]) for r in rows], float)
    failed_low = np.array([1 - int(r["low_ok"]) for r in rows], float)
    ratio_high = np.array([float(r["high_ratio"]) for r in rows], float)
    ratio_low = np.array([float(r["low_ratio"]) for r in rows], float)
    pixels_high = np.array([float(r["high_pixels"]) for r in rows], float)
    pixels_low = np.array([float(r["low_pixels"]) for r in rows], float)
    # A small object should score as more failure-like, so AUROC gets the negation.
    auc_high = auroc(-ratio_high, failed_high)
    auc_low = auroc(-ratio_low, failed_low)
    auc_patch = auroc(-np.concatenate([ratio_high, ratio_low]), np.concatenate([failed_high, failed_low]))
    auc_pixels = auroc(-np.concatenate([pixels_high, pixels_low]), np.concatenate([failed_high, failed_low]))
    cross_high, cross_low = crossing(ratio_high, failed_high), crossing(ratio_low, failed_low)
    flagged = [r for r in rows if float(r["high_ratio"]) < 1 and "tile_ok" in r and r["tile_ok"] != ""]
    def rate(key: str) -> float:
        if not flagged:
            return float("nan")
        return 1 - float(np.mean([int(r[key]) for r in flagged]))
    base, up, tile, zoom = rate("high_ok"), rate("up_ok"), rate("tile_ok"), rate("zoom_ok")
    measured = len(rows) >= 60
    k1 = measured and auc_high >= 0.80 and auc_low >= 0.80
    k2 = (
        measured
        and cross_high == cross_high
        and cross_low == cross_low
        and 0.75 <= cross_high <= 1.25
        and 0.75 <= cross_low <= 1.25
        and abs(cross_high - cross_low) <= 0.25
    )
    k3 = measured and len(flagged) >= 20 and tile <= up and (zoom - tile) >= 0.08
    n_pass = int(bool(k1)) + int(bool(k2)) + int(bool(k3))
    if not measured or n_pass <= 1:
        outcome = False
    elif n_pass == 3:
        outcome = True
    else:
        outcome = None
    text = (
        f"K1 (patch AUROC high {auc_high:.3f}, low {auc_low:.3f}, each >= 0.80; "
        f"pooled patch {auc_patch:.3f} vs pixels {auc_pixels:.3f} is logged, not required) = {bool(k1)}. "
        f"K2 (crossings high {cross_high:.2f}, low {cross_low:.2f} inside [0.75, 1.25] and within 0.25) = {bool(k2)}. "
        f"K3 (flagged n={len(flagged)} error base {base:.3f}, x2 {up:.3f}, zoom {zoom:.3f}, retile {tile:.3f}) = {bool(k3)}. "
        f"Images with a box: {len(rows)}. One tower, token side {FACTOR}px, 2x cap {UPSAMPLE_CAP}px."
    )
    line = kill_test(outcome, text)
    (OUT / "verdict.txt").write_text(line + "\n")
    save_csv(OUT / "summary.csv", [dict(
        n=len(rows), auc_high=auc_high, auc_low=auc_low, auc_patch=auc_patch, auc_pixels=auc_pixels,
        cross_high=cross_high, cross_low=cross_low,
        err_base=base, err_up=up, err_zoom=zoom, err_tile=tile, n_flagged=len(flagged),
    )])
    print(line, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 means the whole dataset")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        verdict(load_csv(OUT / "runs.csv"))
        return
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")
    evaluate(load_rows(args.limit))


if __name__ == "__main__":
    main()
