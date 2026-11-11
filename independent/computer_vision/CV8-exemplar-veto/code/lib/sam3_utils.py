"""SAM 3 on COCO val: one vision pass per image, one decoder pass per prompt."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

DATA = Path(r"C:\Users\natha\dev\cv-data\coco")
MIN_AREA = 32 * 32


def load_coco(min_images: int):
    payload = json.loads((DATA / "annotations" / "instances_val2017.json").read_text(encoding="utf-8"))
    cats = {c["id"]: c for c in payload["categories"]}
    images = {i["id"]: i for i in payload["images"]}
    anns: dict[int, list[dict]] = {}
    for a in payload["annotations"]:
        if a.get("iscrowd", 0) or not isinstance(a["segmentation"], list):
            continue
        anns.setdefault(a["image_id"], []).append(a)
    have = {p.name for p in (DATA / "val2017").glob("*.jpg")}
    ids = [i for i in sorted(anns) if images[i]["file_name"] in have and any(a["area"] >= MIN_AREA for a in anns[i])]
    if len(ids) < min_images:
        raise SystemExit(f"only {len(ids)} COCO images on disk; need {min_images}")
    return cats, images, anns, ids


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


def xyxy(bbox) -> list[float]:
    x, y, w, h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]


class Sam3Runner:
    """Holds the model and a cache of the current image's vision features."""

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

    def prompt(self, text: str | None, boxes=None, labels=None):
        """Returns presence logit, per-query logits (Q,), and binary masks (Q,H,W) at image size."""
        kw = dict(text=text, return_tensors="pt", original_sizes=[[self.size[0], self.size[1]]])
        if boxes is not None:
            kw["input_boxes"] = [boxes]
            kw["input_boxes_labels"] = [labels]
        batch = self.processor(**kw)
        inputs = {k: batch[k].to(self.device) for k in ("input_ids", "attention_mask", "input_boxes", "input_boxes_labels") if k in batch}
        if "input_boxes" in inputs:
            inputs["input_boxes"] = inputs["input_boxes"].to(self.dtype)
        with torch.inference_mode():
            out = self.model(vision_embeds=self.vision, **inputs)
            presence = float(out.presence_logits.float().reshape(-1)[0])
            logits = out.pred_logits[0].float().cpu().numpy()
            masks = F.interpolate(out.pred_masks[0].float()[:, None], size=self.size, mode="bilinear")[:, 0] > 0
        return presence, logits, masks.cpu().numpy()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def kept(logits, presence, thr: float = 0.5):
    return sigmoid(logits) * sigmoid(presence) > thr


def union(masks, keep) -> np.ndarray:
    canvas = np.zeros(masks.shape[-2:], dtype=bool)
    for m, k in zip(masks, keep):
        if k:
            canvas |= m
    return canvas
