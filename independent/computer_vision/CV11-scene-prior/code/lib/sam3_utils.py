"""SAM 3 on LVIS val (COCO val2017 images): one vision pass per view, one decoder pass per prompt."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

DATA = Path(r"C:\Users\natha\dev\cv-data\coco")
LVIS = Path(r"C:\Users\natha\dev\cv-data\lvis\lvis_v1_val.json")
MIN_AREA = 32 * 32
SMALL, LARGE = 32 * 32, 96 * 96


def size_bin(area: float) -> str:
    return "small" if area < SMALL else ("large" if area > LARGE else "medium")


def phrase(cat: dict) -> str:
    return cat["synonyms"][0].replace("_", " ")


def load_lvis(min_images: int, need_exhaustive: bool = True):
    """Categories, image records, polygon annotations and the ordered image ids that pass the design filter.

    Also returns the full LVIS val payload's per-image category sets (all 19.8k images) for co-occurrence statistics."""
    d = json.loads(LVIS.read_text(encoding="utf-8"))
    cats = {c["id"]: c for c in d["categories"]}
    have = {p.name for p in (DATA / "val2017").glob("*.jpg")}
    images = {i["id"]: i for i in d["images"] if i["coco_url"].split("/")[-1] in have}
    anns: dict[int, list[dict]] = {}
    all_present: dict[int, set[int]] = {}
    for a in d["annotations"]:
        all_present.setdefault(a["image_id"], set()).add(a["category_id"])
        if a["image_id"] in images and isinstance(a["segmentation"], list):
            anns.setdefault(a["image_id"], []).append(a)
    ids = []
    for i in sorted(images):
        info = images[i]
        if i not in anns or not info["neg_category_ids"] or not any(a["area"] >= MIN_AREA for a in anns[i]):
            continue
        if need_exhaustive:
            ex = set(info["not_exhaustive_category_ids"])
            if not any(a["category_id"] not in ex for a in anns[i]):
                continue
        ids.append(i)
    if len(ids) < min_images:
        raise SystemExit(f"only {len(ids)} LVIS images on disk pass the filter; need {min_images}")
    return cats, images, anns, ids, all_present


def image_path(info: dict) -> Path:
    return DATA / "val2017" / info["coco_url"].split("/")[-1]


def raster(seg, w: int, h: int) -> np.ndarray:
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    for poly in seg:
        if isinstance(poly, list) and len(poly) >= 6:
            draw.polygon(list(zip(poly[0::2], poly[1::2])), fill=1)
    return np.array(canvas, dtype=bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def iou_many(masks: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """IoU of each mask in (Q,H,W) with one (H,W) mask."""
    if masks.shape[0] == 0:
        return np.zeros(0)
    inter = np.logical_and(masks, gt[None]).sum((1, 2)).astype(np.float64)
    areas = masks.sum((1, 2)).astype(np.float64)
    union = areas + gt.sum() - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


class Sam3Runner:
    """Holds the model and a cache of the current view's vision features."""

    def __init__(self, device: str = "cuda"):
        from transformers import Sam3Model, Sam3Processor

        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        self.model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.float16).to(device).eval()
        self.device = device
        self.dtype = torch.float16
        self.vision = None
        self.size = None

    def set_image(self, image: Image.Image) -> None:
        batch = self.processor(images=image, text="object", return_tensors="pt")
        with torch.inference_mode():
            self.vision = self.model.get_vision_features(batch["pixel_values"].to(self.device, self.dtype))
        self.size = (image.height, image.width)

    def prompt(self, text: str, want_masks: bool = True):
        """Returns presence logit, per-query logits (Q,), and binary masks (Q,H,W) at view size (or None)."""
        batch = self.processor(text=text, return_tensors="pt", original_sizes=[[self.size[0], self.size[1]]])
        inputs = {k: batch[k].to(self.device) for k in ("input_ids", "attention_mask") if k in batch}
        with torch.inference_mode():
            out = self.model(vision_embeds=self.vision, **inputs)
            presence = float(out.presence_logits.float().reshape(-1)[0])
            logits = out.pred_logits[0].float().reshape(-1).cpu().numpy()
            masks = None
            if want_masks:
                masks = (F.interpolate(out.pred_masks[0].float()[:, None], size=self.size, mode="bilinear")[:, 0] > 0).cpu().numpy()
        return presence, logits, masks


def crop_boxes(w: int, h: int, frac: float = 0.6) -> list[tuple[int, int, int, int]]:
    """Five crops of frac*w x frac*h: four corners and the center. Every pixel is in at least one crop."""
    cw, ch = int(round(frac * w)), int(round(frac * h))
    xs, ys = [0, w - cw], [0, h - ch]
    boxes = [(x, y, x + cw, y + ch) for y in ys for x in xs]
    boxes.append(((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch))
    return boxes


def paste(masks: np.ndarray, box: tuple[int, int, int, int], w: int, h: int) -> np.ndarray:
    """Place crop-sized masks (Q,ch,cw) into full-image canvases (Q,h,w)."""
    x0, y0, x1, y1 = box
    canvas = np.zeros((masks.shape[0], h, w), dtype=bool)
    canvas[:, y0:y1, x0:x1] = masks
    return canvas


def nms_masks(masks: np.ndarray, scores: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Greedy mask-IoU non-maximum suppression; returns indices kept, by descending score."""
    order = np.argsort(-scores)
    keep: list[int] = []
    for i in order:
        if all(iou(masks[i], masks[j]) <= thr for j in keep):
            keep.append(int(i))
    return np.array(keep, dtype=int)
