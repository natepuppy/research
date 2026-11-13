"""Qualitative figure. Runs SAM 3 (fp16) on held-out LVIS images chosen by a rule that does not look at the outcome:
the first N held-out images by id that are landscape and whose anchor box covers 5-40% of the frame.
The phrase is the first seeded verified-absent phrase, the one the held-out run used. Writes fig/qualitative.pdf and .json.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
from lib.sam3_utils import DATA, MIN_AREA, Sam3Runner, kept, raster, sigmoid, union  # noqa: E402
from cv8_lvis import background_box, load_lvis, phrase  # noqa: E402

N_ROWS = 2
HARD = 0.10
COLORS = np.array([[230, 57, 70], [29, 120, 200], [244, 162, 30], [42, 157, 90], [140, 80, 200], [0, 170, 170]], float) / 255

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix", "font.size": 7.5,
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.01})


def overlay(image: np.ndarray, masks: np.ndarray, keep) -> np.ndarray:
    out = image.astype(float) / 255
    for i, m in enumerate([m for m, k in zip(masks, keep) if k]):
        out[m] = 0.45 * out[m] + 0.55 * COLORS[i % len(COLORS)]
    return out


def main() -> None:
    cats, images, anns, _ = load_lvis(1)
    held = sorted(int(i) for i in pd.read_csv(HERE.parent / "results" / "real_heldout" / "episodes.csv").image.unique())
    chosen = []
    for iid in held:
        info = images[iid]
        anchor = max((a for a in anns[iid] if a["area"] >= MIN_AREA), key=lambda a: a["area"])
        frac = anchor["bbox"][2] * anchor["bbox"][3] / (info["width"] * info["height"])
        if info["width"] > info["height"] and 0.05 <= frac <= 0.40:
            chosen.append((iid, anchor))
        if len(chosen) == N_ROWS:
            break
    runner = Sam3Runner()
    fig, axes = plt.subplots(N_ROWS, 4, figsize=(6.9, 1.42 * N_ROWS), gridspec_kw=dict(wspace=0.03, hspace=0.16))
    log = []
    for row, (iid, anchor) in enumerate(chosen):
        info = images[iid]
        pil = Image.open(DATA / "val2017" / info["coco_url"].split("/")[-1]).convert("RGB")
        image = np.asarray(pil)
        w, h = pil.size
        ax_, ay, aw, ah = anchor["bbox"]
        box = [float(ax_), float(ay), float(ax_ + aw), float(ay + ah)]
        occupied = np.zeros((h, w), bool)
        for a in anns[iid]:
            occupied |= raster(a["segmentation"], w, h)
        bg = background_box(occupied, int(round(aw)), int(round(ah)))
        absent = random.Random(iid).sample(sorted(info["neg_category_ids"]), min(3, len(info["neg_category_ids"])))
        text = phrase(cats[absent[0]])
        anchor_name = phrase(cats[anchor["category_id"]])
        runner.set_image(pil)
        p_text, l_text, m_text = runner.prompt(text)
        p_box, l_box, m_box = runner.prompt(text, [box], [1])
        p_bg, l_bg, m_bg = runner.prompt(text, [bg], [1])
        gate_open = float(sigmoid(p_text)) >= HARD
        k_text, k_box, k_bg = kept(l_text, p_text), kept(l_box, p_box), kept(l_bg, p_bg)
        k_gate = k_box if gate_open else np.zeros_like(k_box)
        covered = float((union(m_box, k_box) & raster(anchor["segmentation"], w, h)).sum() / max(1, raster(anchor["segmentation"], w, h).sum()))
        def note(pres, keep):
            n = int(keep.sum())
            return f"$s_p = {pres:+.1f}$, {n} mask{'' if n == 1 else 's'}"

        panels = [
            (m_text, k_text, None, "Text only", note(p_text, k_text)),
            (m_box, k_box, box, "+ box on the largest object", note(p_box, k_box)),
            (m_bg, k_bg, bg, "+ box on background", note(p_bg, k_bg)),
            (m_box, k_gate, box, "+ box, with the presence gate",
             f"$\\sigma(s_p^{{\\rm text}}) = {float(sigmoid(p_text)):.3f} < 0.10$, {int(k_gate.sum())} masks" if not gate_open
             else f"$\\sigma(s_p^{{\\rm text}}) = {float(sigmoid(p_text)):.3f}$, {int(k_gate.sum())} masks"),
        ]
        for col, (masks, keep, rect, title, label) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(overlay(image, masks, keep))
            if rect is not None:
                ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2] - rect[0], rect[3] - rect[1], fill=False,
                                               edgecolor="#39ff14", linewidth=1.3, linestyle="--"))
            if row == 0:
                ax.set_title(title, fontsize=7.5, pad=3)
            ax.set_xlabel(label, fontsize=7, labelpad=2)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
        axes[row, 0].set_ylabel(f"\u201c{text}\u201d", fontsize=7.5)
        log.append(dict(image=int(iid), phrase=text, anchor=anchor_name, presence_text=p_text, presence_box=p_box, presence_bg=p_bg,
                        n_text=int(k_text.sum()), n_box=int(k_box.sum()), n_bg=int(k_bg.sum()), n_gated=int(k_gate.sum()),
                        anchor_pixels_covered=covered, text_prob=float(sigmoid(p_text))))
    (HERE / "fig").mkdir(exist_ok=True)
    fig.savefig(HERE / "fig" / "qualitative.pdf", dpi=220)
    fig.savefig(HERE / "fig" / "qualitative.png", dpi=160)
    (HERE / "fig" / "qualitative.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
