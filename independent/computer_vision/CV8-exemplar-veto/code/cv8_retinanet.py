"""CV8 phase 3: the same detector test with RetinaNet instead of Faster R-CNN. Criteria are R1 and R2."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv8_detector as D  # noqa: E402
from lib.io_utils import load_csv, save_csv  # noqa: E402
from lib.sam3_utils import DATA  # noqa: E402


def detect(ids, images, cats, limit) -> list[dict]:
    path = D.OUT / "detections.csv"
    rows = load_csv(path)
    have = {int(r["image"]) for r in rows}
    todo = [i for i in ids if i not in have]
    if not todo:
        return rows
    from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights, retinanet_resnet50_fpn_v2
    from torchvision.transforms.functional import to_tensor

    weights = RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
    names = weights.meta["categories"]
    model = retinanet_resnet50_fpn_v2(weights=weights).cuda().eval()
    valid = {c["name"] for c in cats.values()}
    for n, iid in enumerate(todo):
        image = Image.open(DATA / "val2017" / images[iid]["file_name"]).convert("RGB")
        with torch.inference_mode():
            out = model([to_tensor(image).cuda()])[0]
        count = 0
        for b, s, lab in zip(out["boxes"].cpu().tolist(), out["scores"].cpu().tolist(), out["labels"].cpu().tolist()):
            if s < D.DET_SCORE or count >= D.MAX_DET:
                break
            name = names[lab]
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


if __name__ == "__main__":
    D.detect = detect
    D.MODEL_NAME = "RetinaNet R50-FPN v2 (torchvision COCO)"
    D.OUT = Path(__file__).resolve().parents[1] / "results" / "real_retinanet"
    sys.argv = [sys.argv[0], "--protocol", "confirm", "--dest", str(D.OUT), *sys.argv[1:]]
    D.main()
