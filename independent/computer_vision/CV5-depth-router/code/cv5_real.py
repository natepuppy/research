"""Which object is closer: Depth Anything V2 Small against Qwen3-VL-2B.

Labels are metric depth from TUM RGB-D (freiburg1_desk and freiburg1_xyz), not the
depth model. What'sUp was not downloaded. Thresholds are the README's, written before
the run. A gap of 0.15 is 15% of that frame's own depth range (95th minus 5th
percentile of valid depth).

The depth model's output sign is frozen from the correlation of its map with metric
depth on the first five frames. That chooses the convention, not the router.

K1. On a test bootstrap, exactly one system is correct on >= 0.30 of pairs, and the
    median ground-truth gap on geometry-only pairs exceeds the median on VLM-only
    pairs by >= 0.15. Required on >= 6 of 8 bootstraps.
K2. A router fit on the chronological first 25% beats the better system by >= 0.05
    on >= 6 of 8 bootstraps. It trusts geometry only when the predicted gap is at
    least t_gap and the smaller box is at least t_size of the image side.
K3. That frozen router keeps >= 80% of the gain of the same grid refit on the
    bootstrap, on >= 6 of 8. Reported.
SUPPORTED if K1 and K2. INCONCLUSIVE if exactly one. KILLED if neither.
INCONCLUSIVE if the test split has fewer than 80 pairs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.io_utils import kill_test, load_csv, save_csv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "real"
DATA = Path(r"C:\Users\natha\dev\cv-data\tum")
SEQUENCES = ("rgbd_dataset_freiburg1_desk", "rgbd_dataset_freiburg1_xyz")
URLS = {
    "rgbd_dataset_freiburg1_desk": "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz",
    "rgbd_dataset_freiburg1_xyz": "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz",
}
GAP_GRID = (0.05, 0.10, 0.20, 0.30)
SIDE_GRID = (0.05, 0.10, 0.20)


def ensure_sequences() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        folder = DATA / name
        if (folder / "rgb.txt").exists():
            continue
        archive = DATA / f"{name}.tgz"
        if not archive.exists() or archive.stat().st_size < 1_000_000:
            print(f"downloading {name}", flush=True)
            subprocess.check_call(["curl.exe", "-L", "--retry", "10", "--retry-all-errors", "-C", "-", "-o", str(archive), url])
        print(f"extracting {name}", flush=True)
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(DATA)


def read_list(path: Path) -> list[tuple[float, str]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        stamp, name = line.split()
        rows.append((float(stamp), name))
    return rows


def associate(seq: Path) -> list[tuple[float, Path, Path]]:
    rgb = read_list(seq / "rgb.txt")
    depth = read_list(seq / "depth.txt")
    pairs = []
    j = 0
    for stamp, rgb_name in rgb:
        while j + 1 < len(depth) and abs(depth[j + 1][0] - stamp) < abs(depth[j][0] - stamp):
            j += 1
        if abs(depth[j][0] - stamp) <= 0.02:
            pairs.append((stamp, seq / rgb_name, seq / depth[j][1]))
    return pairs


def read_depth(path: Path) -> np.ndarray:
    return np.array(Image.open(path), dtype=np.float32) / 5000.0


def depth_range(depth: np.ndarray) -> float:
    valid = depth[depth > 0]
    if len(valid) < 50:
        return float("nan")
    return float(np.percentile(valid, 95) - np.percentile(valid, 5))


def box_depth(depth: np.ndarray, box: np.ndarray) -> float:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(depth.shape[1], x2), min(depth.shape[0], y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return float("nan")
    crop = depth[y1:y2, x1:x2]
    valid = crop[crop > 0]
    if len(valid) < 30:
        return float("nan")
    return float(np.median(valid))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def build_pairs(limit_frames: int) -> list[dict]:
    from ultralytics import YOLO

    detector = YOLO("yolov8s.pt")
    pairs = []
    for seq_name in SEQUENCES:
        frames = associate(DATA / seq_name)[::8][:limit_frames]
        for frame_i, (stamp, rgb_path, depth_path) in enumerate(frames):
            image = Image.open(rgb_path).convert("RGB")
            depth = read_depth(depth_path)
            span = depth_range(depth)
            found = detector.predict(image, verbose=False, conf=0.25, device="cpu")[0]
            if found.boxes is None or len(found.boxes) < 2 or not np.isfinite(span) or span <= 0:
                continue
            boxes = found.boxes.xyxy.cpu().numpy()
            classes = found.boxes.cls.cpu().numpy().astype(int)
            raw_names = [detector.names[int(c)] for c in classes]
            names = []
            for box, name in zip(boxes, raw_names):
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                horizontal = "left" if cx < image.width / 2 else "right"
                vertical = "upper" if cy < image.height / 2 else "lower"
                names.append(f"{vertical} {horizontal} {name}")
            chosen = 0
            order = np.argsort(-(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
            for a in order:
                if chosen >= 2:
                    break
                for b in order:
                    if b <= a or iou(boxes[a], boxes[b]) > 0.3:
                        continue
                    za, zb = box_depth(depth, boxes[a]), box_depth(depth, boxes[b])
                    if not np.isfinite(za) or not np.isfinite(zb) or abs(za - zb) < 1e-4:
                        continue
                    side = min(
                        boxes[a, 2] - boxes[a, 0], boxes[a, 3] - boxes[a, 1],
                        boxes[b, 2] - boxes[b, 0], boxes[b, 3] - boxes[b, 1],
                    ) / max(image.size)
                    pairs.append(dict(
                        index=len(pairs), seq=seq_name, frame=frame_i, stamp=stamp,
                        rgb=str(rgb_path), name_a=names[a], name_b=names[b],
                        box_a=boxes[a].tolist(), box_b=boxes[b].tolist(),
                        z_a=za, z_b=zb, gap_gt=abs(za - zb) / span, side=float(side),
                        gt=0 if za < zb else 1,
                    ))
                    chosen += 1
                    break
            print(f"{seq_name} frame {frame_i} pairs={len(pairs)}", flush=True)
    return pairs


def calibrate_sign(model, processor, pairs: list[dict]) -> int:
    seen = []
    corr = []
    for pair in pairs:
        if pair["rgb"] in seen:
            continue
        seen.append(pair["rgb"])
        if len(seen) > 5:
            break
        image = Image.open(pair["rgb"]).convert("RGB")
        pred = predict_depth(model, processor, image)
        metric = read_depth(associate_depth(pair))
        pred = torch.nn.functional.interpolate(
            torch.from_numpy(np.ascontiguousarray(pred))[None, None],
            size=metric.shape, mode="bilinear", align_corners=False,
        )[0, 0].numpy()
        valid = metric > 0
        if valid.sum() < 100:
            continue
        sample = np.flatnonzero(valid)[::50]
        a, b = pred.reshape(-1)[sample], metric.reshape(-1)[sample]
        if a.std() == 0 or b.std() == 0:
            continue
        corr.append(float(np.corrcoef(a, b)[0, 1]))
    if not corr:
        raise RuntimeError("Depth Anything and TUM depth did not overlap on the calibration frames.")
    sign = 1 if float(np.mean(corr)) >= 0 else -1
    print(f"depth convention correlation {corr} sign {sign} (1 means larger prediction is farther)", flush=True)
    (OUT / "depth_sign.txt").write_text(f"correlation={corr}\nsign={sign}\n")
    return sign


def associate_depth(pair: dict) -> Path:
    seq = DATA / pair["seq"]
    for stamp, rgb, depth in associate(seq):
        if abs(stamp - float(pair["stamp"])) < 1e-6:
            return depth
    raise FileNotFoundError(pair["rgb"])


def predict_depth(model, processor, image: Image.Image) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    with torch.inference_mode():
        pred = model(**inputs).predicted_depth
    pred = torch.nn.functional.interpolate(pred[:, None], size=image.size[::-1], mode="bicubic", align_corners=False)
    return pred[0, 0].float().cpu().numpy()


def fit_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    scale = min(1.0, (640 * 640 / max(image.width * image.height, 1)) ** 0.5)
    nw, nh = max(32, int(image.width * scale)), max(32, int(image.height * scale))
    return image.resize((nw, nh), Image.BICUBIC)


def ask(model, processor, image: Image.Image, name_a: str, name_b: str) -> str:
    question = (
        f"Which object is closer to the camera: A, the {name_a}, or B, the {name_b}? "
        "Reply with only A or B."
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": fit_image(image)},
        {"type": "text", "text": question},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    text = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return text.strip()


def parse_choice(text: str) -> int | None:
    for char in text.upper():
        if char == "A":
            return 0
        if char == "B":
            return 1
    return None


def run_models(pairs: list[dict]) -> list[dict]:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoProcessor, Qwen3VLForConditionalGeneration

    OUT.mkdir(parents=True, exist_ok=True)
    done = {int(r["index"]) for r in load_csv(OUT / "runs.csv")}
    written = list(load_csv(OUT / "runs.csv"))
    pending = [p for p in pairs if p["index"] not in done]
    if not pending and written:
        return written
    depth_processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    depth_model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Small-hf", torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    sign = calibrate_sign(depth_model, depth_processor, pairs)
    cache: dict[str, np.ndarray] = {}
    for pair in pending:
        if pair["rgb"] not in cache:
            cache[pair["rgb"]] = predict_depth(depth_model, depth_processor, Image.open(pair["rgb"]).convert("RGB"))
    del depth_model
    torch.cuda.empty_cache()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    started = time.time()
    for pair in pending:
        pred = cache[pair["rgb"]]
        da, db = box_depth(pred, np.array(pair["box_a"])), box_depth(pred, np.array(pair["box_b"]))
        # sign +1: larger prediction is farther, so the closer box is the smaller value
        geom = None if not (np.isfinite(da) and np.isfinite(db)) else (0 if sign * da < sign * db else 1)
        finite = pred[np.isfinite(pred)]
        span = float(np.percentile(finite, 95) - np.percentile(finite, 5)) if len(finite) else float("nan")
        text = ask(model, processor, Image.open(pair["rgb"]).convert("RGB"), pair["name_a"], pair["name_b"])
        choice = parse_choice(text)
        record = dict(
            index=pair["index"], seq=pair["seq"], frame=pair["frame"],
            name_a=pair["name_a"], name_b=pair["name_b"], gt=pair["gt"],
            gap_gt=pair["gap_gt"], gap_pred=(abs(da - db) / span) if np.isfinite(span) else float("nan"),
            side=pair["side"], geom_ok=int(geom == pair["gt"]) if geom is not None else 0,
            geom_valid=int(geom is not None), vlm_ok=int(choice == pair["gt"]) if choice is not None else 0,
            vlm_valid=int(choice is not None), text=text, sign=sign,
        )
        written.append(record)
        save_csv(OUT / "runs.csv", written)
        print(f"{pair['index']:03d} gt={pair['gt']} geom={geom} vlm={text!r} [{time.time() - started:.0f}s]", flush=True)
        torch.cuda.empty_cache()
    return written


def choose(gap, side, geom, vlm, t_gap, t_side):
    use_geom = (gap >= t_gap) & (side >= t_side)
    return np.where(use_geom, geom, vlm)


def best_grid(gap, side, geom, vlm):
    best = (-1.0, GAP_GRID[0], SIDE_GRID[0])
    for t_gap in GAP_GRID:
        for t_side in SIDE_GRID:
            acc = float(choose(gap, side, geom, vlm, t_gap, t_side).mean())
            if acc > best[0]:
                best = (acc, t_gap, t_side)
    return best


def seed_metrics(rows, rng, t_gap, t_side) -> dict:
    take = rng.choice(len(rows), size=len(rows), replace=True)
    gap = np.array([float(rows[i]["gap_pred"]) for i in take])
    side = np.array([float(rows[i]["side"]) for i in take])
    geom = np.array([int(rows[i]["geom_ok"]) for i in take], float)
    vlm = np.array([int(rows[i]["vlm_ok"]) for i in take], float)
    gt_gap = np.array([float(rows[i]["gap_gt"]) for i in take])
    exactly = (geom != vlm).mean()
    only_g, only_v = (geom == 1) & (vlm == 0), (vlm == 1) & (geom == 0)
    med_g = float(np.median(gt_gap[only_g])) if only_g.any() else float("nan")
    med_v = float(np.median(gt_gap[only_v])) if only_v.any() else float("nan")
    routed = choose(gap, side, geom, vlm, t_gap, t_side)
    better = max(float(geom.mean()), float(vlm.mean()))
    oracle_acc, _, _ = best_grid(gap, side, geom, vlm)
    gain, oracle_gain = float(routed.mean()) - better, oracle_acc - better
    keep = gain / oracle_gain if oracle_gain > 1e-6 else float("nan")
    return dict(
        exactly=float(exactly), gap_delta=med_g - med_v, router=float(routed.mean()),
        better=better, geom=float(geom.mean()), vlm=float(vlm.mean()), keep=keep,
        k1=exactly >= 0.30 and (med_g - med_v) >= 0.15,
        k2=float(routed.mean()) - better >= 0.05,
        k3=keep == keep and keep >= 0.80,
    )


def verdict(rows: list[dict]) -> None:
    rows = [r for r in rows if int(r.get("geom_valid", 0)) == 1 and int(r.get("vlm_valid", 0)) == 1]
    rows = sorted(rows, key=lambda r: (str(r["seq"]), int(r["frame"]), int(r["index"])))
    cut = max(1, int(len(rows) * 0.25))
    val, test = rows[:cut], rows[cut:]
    measured = len(test) >= 80
    if measured:
        gap = np.array([float(r["gap_pred"]) for r in val])
        side = np.array([float(r["side"]) for r in val])
        geom = np.array([int(r["geom_ok"]) for r in val], float)
        vlm = np.array([int(r["vlm_ok"]) for r in val], float)
        _, t_gap, t_side = best_grid(gap, side, geom, vlm)
        seeds = [seed_metrics(test, np.random.default_rng(seed), t_gap, t_side) for seed in range(8)]
    else:
        t_gap, t_side, seeds = float("nan"), float("nan"), []
    k1 = measured and sum(bool(s["k1"]) for s in seeds) >= 6
    k2 = measured and sum(bool(s["k2"]) for s in seeds) >= 6
    k3 = measured and sum(bool(s["k3"]) for s in seeds) >= 6
    if not measured:
        outcome = None
    elif k1 and k2:
        outcome = True
    elif k1 or k2:
        outcome = None
    else:
        outcome = False
    point = seeds[0] if seeds else {}
    text = (
        f"K1 (exactly-one and gap split on >= 6/8: {sum(bool(s['k1']) for s in seeds)}/8) = {bool(k1)}. "
        f"K2 (router beats the better system by >= 0.05 on >= 6/8: {sum(bool(s['k2']) for s in seeds)}/8) = {bool(k2)}. "
        f"K3 (frozen router keeps >= 80% of the oracle gain on >= 6/8: {sum(bool(s['k3']) for s in seeds)}/8) = {bool(k3)}. "
        f"Val n={len(val)} test n={len(test)} thresholds gap {t_gap} side {t_side}. "
        f"Seed 0 geom {point.get('geom', float('nan')):.3f} vlm {point.get('vlm', float('nan')):.3f} "
        f"router {point.get('router', float('nan')):.3f} exactly {point.get('exactly', float('nan')):.3f} "
        f"gap_delta {point.get('gap_delta', float('nan')):.3f}. TUM RGB-D, not What'sUp."
    )
    line = kill_test(None if not measured else outcome, text)
    (OUT / "verdict.txt").write_text(line + "\n")
    save_csv(OUT / "summary.csv", [dict(
        n_val=len(val), n_test=len(test), t_gap=t_gap, t_side=t_side,
        k1_seeds=sum(bool(s["k1"]) for s in seeds), k2_seeds=sum(bool(s["k2"]) for s in seeds),
        k3_seeds=sum(bool(s["k3"]) for s in seeds), **{f"s0_{k}": point.get(k, "") for k in ("geom", "vlm", "router", "exactly", "gap_delta", "keep")},
    )])
    save_csv(OUT / "seeds.csv", seeds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-frames", type=int, default=40)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        verdict(load_csv(OUT / "runs.csv"))
        return
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")
    ensure_sequences()
    pairs = build_pairs(args.limit_frames)
    if not pairs:
        raise SystemExit("No object pairs with valid depth.")
    verdict(run_models(pairs))


if __name__ == "__main__":
    main()
