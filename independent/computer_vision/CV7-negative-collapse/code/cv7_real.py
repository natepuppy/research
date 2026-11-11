"""SAM 3 on COCO val, with one negative box.

CASE, COCO-20i, PASCAL-5i and the published LoRA are not in this run. The README
requires those benchmarks for a SUPPORTED verdict, so this script can come back
INCONCLUSIVE or KILLED and cannot come back SUPPORTED. Thresholds below are the
README's COCO bars. They were fixed before the run.

An episode is one COCO val image. The text and one positive box name a category.
The ground truth is the union of that category's masks. The negative, when present,
is a box from a different category. A fifth of the episodes have no negative.

Scores are sigmoid(query logit) times sigmoid(presence). A mask is kept at 0.5.
Presence-only keeps the positive-pass masks and logits and swaps in the presence
logit from the pass that saw the negative. Similarity is the cosine between a
positive-pass query and the mean query, from a third forward, that fires on the
negative box. The logit drop uses the same query index in the two forwards.
Queries whose positive-pass and negative-pass scores are both at or below 0.10
are left out of the correlation. A score is sigmoid(logit) times sigmoid(presence).

K1, COCO only. Absolute correlation of logit drop with that similarity < 0.20,
and presence share >= 0.50, on >= 6 of 8 bootstrap resamples. A resample with
fewer than 10 dropped episodes fails. If fewer than half the test episodes drop
by >= 0.02, the collapse was not observed and this criterion fails.
K2, COCO only. Rejection recovers >= 60% of the drop, on >= 6 of 8, with the
same drop rules. The CASE 70% bar was not measured, so the README's K2 cannot pass.
K3 is reported on COCO: label-oracle gap and the zero-negative gap. The CASE
clause was not measured, so K3 is not marked passed.
The first 30 episodes, in image-id order, are validation and choose the
rejection threshold. The rest are the test set. SUPPORTED is not available.
INCONCLUSIVE if the test set has fewer than 40 episodes, or if the COCO K1 or
the COCO K2 passes. KILLED if neither COCO criterion passes.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io_utils import kill_test, load_csv, save_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "real"
DATA = Path(r"C:\Users\natha\dev\cv-data\coco")
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
SCORE_THR = 0.5
MASK_THR = 0.5
QUERY_FLOOR = 0.10
DROP_FLOOR = 0.02
FIX_GRID = (0.2, 0.4, 0.6, 0.8)
N_EPISODES = 120
VAL_N = 30
MODEL_KEYS = ("pixel_values", "input_ids", "attention_mask", "input_boxes", "input_boxes_labels")


def ensure_annotations() -> Path:
    target = DATA / "annotations" / "instances_val2017.json"
    if target.exists():
        return target
    DATA.mkdir(parents=True, exist_ok=True)
    archive = DATA / "annotations_trainval2017.zip"
    if not archive.exists() or archive.stat().st_size < 1_000_000:
        print("downloading COCO annotations", flush=True)
        subprocess.check_call(["curl.exe", "-L", "--retry", "10", "--retry-all-errors", "-C", "-", "-o", str(archive), ANN_URL])
    print("extracting instances_val2017.json", flush=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extract("annotations/instances_val2017.json", DATA)
    return target


def polygons(segmentation) -> list[list[float]]:
    if not isinstance(segmentation, list):
        return []
    return [poly for poly in segmentation if isinstance(poly, list) and len(poly) >= 6]


def raster(segmentation, width: int, height: int) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for poly in polygons(segmentation):
        draw.polygon(list(zip(poly[0::2], poly[1::2])), fill=1)
    return np.array(canvas, dtype=bool)


def build_episodes(limit: int) -> list[dict]:
    payload = json.loads(ensure_annotations().read_text(encoding="utf-8"))
    names = {row["id"]: row["name"] for row in payload["categories"]}
    images = {row["id"]: row for row in payload["images"]}
    by_image: dict[int, list[dict]] = {}
    for ann in payload["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        if not polygons(ann["segmentation"]):
            continue
        by_image.setdefault(ann["image_id"], []).append(ann)
    episodes = []
    for image_id in sorted(by_image):
        groups: dict[int, list[dict]] = {}
        for ann in by_image[image_id]:
            if ann["bbox"][2] * ann["bbox"][3] < 32 * 32:
                continue
            groups.setdefault(ann["category_id"], []).append(ann)
        if len(groups) < 2:
            continue
        target_id = max(groups, key=lambda key: sum(item["area"] for item in groups[key]))
        others = [ann for key, items in groups.items() if key != target_id for ann in items]
        if not others:
            continue
        negative = max(others, key=lambda item: item["area"])
        positive = max(groups[target_id], key=lambda item: item["area"])
        info = images[image_id]
        zero = len(episodes) % 5 == 4
        episodes.append(dict(
            index=len(episodes), image_id=image_id, file_name=info["file_name"],
            width=info["width"], height=info["height"],
            target=names[target_id], target_id=target_id,
            negative=names[negative["category_id"]], neg_id=negative["id"],
            pos_box=xyxy(positive["bbox"]),
            neg_box=xyxy(negative["bbox"]),
            zero_neg=int(zero),
        ))
        if len(episodes) >= limit:
            break
    keep = {episode["image_id"] for episode in episodes}
    return episodes, {key: value for key, value in by_image.items() if key in keep}


def xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]


def ground_truth(by_image: dict, episode: dict) -> tuple[np.ndarray, np.ndarray]:
    height, width = episode["height"], episode["width"]
    target = np.zeros((height, width), dtype=bool)
    negative = np.zeros((height, width), dtype=bool)
    for ann in by_image[episode["image_id"]]:
        if ann["category_id"] != episode["target_id"] and ann["id"] != episode["neg_id"]:
            continue
        mask = raster(ann["segmentation"], width, height)
        if ann["category_id"] == episode["target_id"]:
            target |= mask
        if ann["id"] == episode["neg_id"]:
            negative |= mask
    if not episode["zero_neg"] and not negative.any():
        raise RuntimeError(f"episode {episode['index']} has no negative mask")
    return target, negative


def fetch_image(episode: dict) -> Image.Image:
    folder = DATA / "val2017"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / episode["file_name"]
    if not path.exists() or path.stat().st_size < 1000:
        url = f"http://images.cocodataset.org/val2017/{episode['file_name']}"
        urllib.request.urlretrieve(url, path)
    return Image.open(path).convert("RGB")


def move(batch, device: str):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def image_size(value) -> tuple[int, int]:
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    height, width = value[0]
    return int(height), int(width)


def forward(model, processor, image, text, boxes, labels, device, holder):
    kwargs = dict(images=image, text=text, return_tensors="pt")
    if boxes is not None:
        kwargs["input_boxes"] = [boxes]
        kwargs["input_boxes_labels"] = [labels]
    batch = processor(**kwargs)
    size = image_size(batch["original_sizes"])
    inputs = move({key: batch[key] for key in MODEL_KEYS if key in batch}, device)
    dtype = next(model.parameters()).dtype
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=dtype)
    if "input_boxes" in inputs:
        inputs["input_boxes"] = inputs["input_boxes"].to(dtype=dtype)
    with torch.inference_mode():
        output = model(**inputs)
    hidden = holder["q"][0].float()
    return output, hidden, size


def presence_value(output) -> torch.Tensor:
    return output.presence_logits.float().reshape(-1)[0]


def scored_masks(output, presence, size):
    logits = output.pred_logits[0].float()
    if presence is None:
        presence = presence_value(output)
    scores = logits.sigmoid() * torch.sigmoid(presence)
    masks = output.pred_masks[0].float().sigmoid()
    masks = F.interpolate(masks[:, None], size=size, mode="bilinear", align_corners=False)[:, 0] > MASK_THR
    return logits, scores, masks.cpu().numpy()


def paint(scores, masks, keep) -> np.ndarray:
    canvas = np.zeros(masks.shape[-2:], dtype=bool)
    for score, mask, use in zip(scores, masks, keep):
        if use and float(score) > SCORE_THR:
            canvas |= mask
    return canvas


def mask_iou(pred: np.ndarray, truth: np.ndarray) -> float:
    inter = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


def iou(pred: np.ndarray, truth: np.ndarray) -> float:
    if not pred.any() and not truth.any():
        return 1.0
    return mask_iou(pred, truth)


def cosine(query: torch.Tensor, prototype: torch.Tensor) -> np.ndarray:
    query = F.normalize(query, dim=-1)
    prototype = F.normalize(prototype, dim=0)
    return (query @ prototype).float().cpu().numpy()


def stay_awake() -> None:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)


def run(limit: int) -> None:
    from transformers import Sam3Model, Sam3Processor

    stay_awake()
    episodes, by_image = build_episodes(limit)
    if limit >= N_EPISODES and len(episodes) < N_EPISODES:
        raise SystemExit(f"only {len(episodes)} usable COCO episodes")
    OUT.mkdir(parents=True, exist_ok=True)
    done = {int(row["index"]) for row in load_csv(OUT / "runs.csv")}
    written = list(load_csv(OUT / "runs.csv"))
    proposals = list(load_csv(OUT / "proposals.csv"))
    device = "cuda"
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.float16).to(device).eval()
    holder: dict[str, torch.Tensor] = {}

    def capture(_module, _inputs, output):
        holder["q"] = output.intermediate_hidden_states[-1].detach()

    model.detr_decoder.register_forward_hook(capture)
    started = time.time()
    for episode in episodes:
        if episode["index"] in done:
            continue
        image = fetch_image(episode)
        target, negative_mask = ground_truth(by_image, episode)
        positive, hidden_pos, size = forward(
            model, processor, image, episode["target"], [episode["pos_box"]], [1], device, holder
        )
        if size != (episode["height"], episode["width"]):
            raise RuntimeError(f"processor size {size} != image {(episode['height'], episode['width'])}")
        logits_pos, scores_pos, masks_pos = scored_masks(positive, None, size)
        pos_iou = iou(paint(scores_pos, masks_pos, np.ones(len(scores_pos), dtype=bool)), target)
        record = dict(
            index=episode["index"], image_id=episode["image_id"], target=episode["target"],
            negative="" if episode["zero_neg"] else episode["negative"], zero_neg=episode["zero_neg"],
            pos_iou=pos_iou, presence_pos=float(presence_value(positive)),
        )
        if episode["zero_neg"]:
            record.update(neg_iou=pos_iou, presence_only_iou=pos_iou, presence_neg=record["presence_pos"], oracle_iou=pos_iou)
            for thr in FIX_GRID:
                record[f"fix_{thr}"] = pos_iou
        else:
            collapsed, hidden_neg, _ = forward(
                model, processor, image, episode["target"],
                [episode["pos_box"], episode["neg_box"]], [1, 0], device, holder,
            )
            logits_neg, scores_neg, masks_neg = scored_masks(collapsed, None, size)
            _, scores_presence, _ = scored_masks(positive, presence_value(collapsed), size)
            neg_concept, hidden_concept, _ = forward(
                model, processor, image, episode["negative"], [episode["neg_box"]], [1], device, holder
            )
            _, concept_scores, _ = scored_masks(neg_concept, None, size)
            fired = concept_scores > SCORE_THR
            if not fired.any():
                fired[int(concept_scores.argmax())] = True
            prototype = hidden_concept[fired.cpu()].mean(dim=0)
            similarity = cosine(hidden_pos, prototype)
            drop = (logits_pos - logits_neg).float().cpu().numpy()
            score_pos = scores_pos.detach().float().cpu().numpy()
            score_neg = scores_neg.detach().float().cpu().numpy()
            keep_query = np.maximum(score_pos, score_neg) > QUERY_FLOOR
            for query in np.flatnonzero(keep_query):
                proposals.append(dict(index=episode["index"], drop=float(drop[query]), sim=float(similarity[query])))
            distractor = np.array([mask_iou(mask, negative_mask) > 0.5 for mask in masks_pos])
            record.update(
                neg_iou=iou(paint(scores_neg, masks_neg, np.ones(len(scores_neg), dtype=bool)), target),
                presence_only_iou=iou(paint(scores_presence, masks_pos, np.ones(len(scores_pos), dtype=bool)), target),
                presence_neg=float(presence_value(collapsed)),
                oracle_iou=iou(paint(scores_pos, masks_pos, ~distractor), target),
            )
            for thr in FIX_GRID:
                record[f"fix_{thr}"] = iou(paint(scores_pos, masks_pos, similarity < thr), target)
        written.append(record)
        save_csv(OUT / "runs.csv", written)
        save_csv(OUT / "proposals.csv", proposals)
        print(
            f"{episode['index']:03d} {episode['target']} pos={record['pos_iou']:.2f} "
            f"neg={record.get('neg_iou', record['pos_iou']):.2f} [{time.time() - started:.0f}s]",
            flush=True,
        )
        torch.cuda.empty_cache()
    test_n = sum(1 for row in written if int(row["index"]) >= VAL_N)
    if test_n >= 40:
        verdict(written, proposals)
    else:
        print(f"recorded {len(written)} episodes; verdict waits for 40 test episodes", flush=True)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 8:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom == 0:
        return float("nan")
    return float((x * y).sum() / denom)


def choose_threshold(rows: list[dict]) -> float:
    usable = [row for row in rows if int(row["zero_neg"]) == 0]
    best_thr, best = FIX_GRID[0], -1.0
    for thr in FIX_GRID:
        score = float(np.mean([float(row[f"fix_{thr}"]) for row in usable])) if usable else -1.0
        if score > best:
            best, best_thr = score, thr
    return best_thr


def episode_stats(rows: list[dict], proposals: list[dict], thr: float) -> dict:
    dropped = []
    shares, recoveries = [], []
    for row in rows:
        if int(row["zero_neg"]) == 1:
            continue
        gap = float(row["pos_iou"]) - float(row["neg_iou"])
        if gap < DROP_FLOOR:
            continue
        dropped.append(gap)
        shares.append((float(row["pos_iou"]) - float(row["presence_only_iou"])) / gap)
        recoveries.append((float(row[f"fix_{thr}"]) - float(row["neg_iou"])) / gap)
    counts = Counter(int(row["index"]) for row in rows if int(row["zero_neg"]) == 0)
    paired = []
    for item in proposals:
        paired.extend([item] * counts.get(int(item["index"]), 0))
    zeros = [row for row in rows if int(row["zero_neg"]) == 1]
    zero_gap = float(np.mean([abs(float(row[f"fix_{thr}"]) - float(row["pos_iou"])) for row in zeros])) if zeros else float("nan")
    oracle_gap = float(np.mean([abs(float(row["oracle_iou"]) - float(row[f"fix_{thr}"])) for row in rows if int(row["zero_neg"]) == 0]))
    return dict(
        n=len(rows), n_drop=len(dropped),
        pos=float(np.mean([float(row["pos_iou"]) for row in rows])),
        neg=float(np.mean([float(row["neg_iou"]) for row in rows if int(row["zero_neg"]) == 0])),
        share=float(np.mean(shares)) if shares else float("nan"),
        recovery=float(np.mean(recoveries)) if recoveries else float("nan"),
        rho=pearson(np.array([float(item["drop"]) for item in paired]), np.array([float(item["sim"]) for item in paired])),
        zero_gap=zero_gap, oracle_gap=oracle_gap,
    )


def criterion(stats: dict) -> tuple[bool, bool]:
    enough = stats["n_drop"] >= 10
    k1 = enough and stats["rho"] == stats["rho"] and abs(stats["rho"]) < 0.20 and stats["share"] == stats["share"] and stats["share"] >= 0.50
    k2 = enough and stats["recovery"] == stats["recovery"] and stats["recovery"] >= 0.60
    return k1, k2


def verdict(rows: list[dict], proposals: list[dict]) -> None:
    rows = sorted(rows, key=lambda row: int(row["index"]))
    val = [row for row in rows if int(row["index"]) < VAL_N]
    test = [row for row in rows if int(row["index"]) >= VAL_N]
    measured = len(test) >= 40
    thr = choose_threshold(val) if measured else FIX_GRID[0]
    point = episode_stats(test, proposals, thr) if measured else {}
    seeds = []
    if measured:
        for seed in range(8):
            take = np.random.default_rng(seed).choice(len(test), size=len(test), replace=True)
            sample = [test[i] for i in take]
            stats = episode_stats(sample, proposals, thr)
            stats["k1"], stats["k2"] = criterion(stats)
            seeds.append(stats)
        # The half-drop rule is on the test set itself. A resample also fails below 10 drops.
        if point["n_drop"] < 0.5 * point["n"]:
            for stats in seeds:
                stats["k1"] = False
                stats["k2"] = False
    coco_k1 = measured and sum(bool(seed["k1"]) for seed in seeds) >= 6
    coco_k2 = measured and sum(bool(seed["k2"]) for seed in seeds) >= 6
    outcome = None if (not measured or coco_k1 or coco_k2) else False
    text = (
        f"COCO K1 (|rho| < 0.20 and presence share >= 0.50 on >= 6/8: "
        f"{sum(bool(seed['k1']) for seed in seeds)}/8; rho {point.get('rho', float('nan')):.3f}, "
        f"share {point.get('share', float('nan')):.3f}) = {bool(coco_k1)}. "
        f"COCO K2 (recovery >= 0.60 on >= 6/8: {sum(bool(seed['k2']) for seed in seeds)}/8; "
        f"recovery {point.get('recovery', float('nan')):.3f}) = {bool(coco_k2)}. "
        f"README K1 and K2 also require CASE, which was not measured, so neither README criterion passes and SUPPORTED is unavailable. "
        f"K3 (CASE oracle bar not measured; COCO |oracle-fix| {point.get('oracle_gap', float('nan')) * 100:.2f} mIoU, "
        f"zero-negative gap {point.get('zero_gap', float('nan')) * 100:.2f} mIoU; the 1 mIoU zero-negative bar is reported, not used as a pass) = False. "
        f"Test n={len(test)} val n={len(val)} threshold {thr}. "
        f"Mean IoU positive {point.get('pos', float('nan')):.3f}, with the negative {point.get('neg', float('nan')):.3f}, "
        f"episodes with a drop {point.get('n_drop', 0)} of {point.get('n', 0)}. "
        f"SAM 3 fp16, one negative box, COCO val only. COCO-20i, PASCAL-5i and the LoRA were not run."
    )
    line = kill_test(outcome, text)
    (OUT / "verdict.txt").write_text(line + "\n")
    save_csv(OUT / "summary.csv", [dict(threshold=thr, k1_seeds=sum(bool(seed["k1"]) for seed in seeds), **point)])
    save_csv(OUT / "seeds.csv", seeds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=N_EPISODES)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        verdict(load_csv(OUT / "runs.csv"), load_csv(OUT / "proposals.csv"))
        return
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")
    run(args.limit)


if __name__ == "__main__":
    main()
