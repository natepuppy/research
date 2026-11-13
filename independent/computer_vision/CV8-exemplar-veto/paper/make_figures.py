"""Plots for the paper, drawn from results/*/episodes.csv. Writes paper/fig/*.pdf."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from make_tables import HARD, Boot, hard_curve, load, sigmoid  # noqa: E402

FIG = Path(__file__).resolve().parent / "fig"
COL = 3.25  # CVPR column width in inches

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix", "font.size": 7.5,
    "axes.labelsize": 7.5, "axes.titlesize": 7.5, "legend.fontsize": 6.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.5, "lines.linewidth": 1.1, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})
BLUE, RED, ORANGE, GREEN, GREY = "#2b6cb0", "#c53030", "#dd6b20", "#2f855a", "#4a5568"


def presence() -> None:
    df = load("real_heldout")
    bins = np.linspace(-10, 10, 41)
    fig, axes = plt.subplots(2, 1, figsize=(COL, 2.75), sharex=True, gridspec_kw=dict(hspace=0.38))

    def draw(ax, cond, color, label, style="-"):
        v = df[df.cond == cond].presence.to_numpy()
        h, _ = np.histogram(np.clip(v, -10, 10), bins=bins)
        ax.stairs(h / len(v), bins, color=color, label=label, linestyle=style, linewidth=1.1)

    draw(axes[0], "T-", RED, "absent phrase")
    draw(axes[0], "T+", BLUE, "present phrase")
    draw(axes[1], "B+T-", RED, "absent phrase + box on largest object")
    draw(axes[1], "Bg+T-", ORANGE, "absent phrase + box on background", "--")
    draw(axes[1], "B+T+", BLUE, "present phrase + its box", ":")
    axes[0].set_title("Text only", loc="left", pad=3)
    axes[1].set_title("Text + positive exemplar box", loc="left", pad=3)
    for ax in axes:
        ax.axvline(0, color=GREY, linewidth=0.5, linestyle=":")
        ax.set_ylabel("fraction of episodes")
    axes[0].set_ylim(0, 0.135)
    axes[0].legend(frameon=False, loc="upper center", ncol=2, handlelength=1.6, borderaxespad=0.1)
    axes[1].legend(frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.6), handlelength=1.6)
    axes[1].set_xlabel(r"presence logit $s_p$ (clamped to $[-10, 10]$)")
    axes[1].set_xlim(-10, 10)
    fig.savefig(FIG / "presence.pdf")
    plt.close(fig)


def gate_curve() -> None:
    pool, held = load("real_lvis"), load("real_heldout")
    fig, ax = plt.subplots(figsize=(COL, 2.05))
    for df, color, label, marker in ((pool, GREY, "hard gate, selection images", "o"), (held, BLUE, "hard gate, held-out images", "s")):
        curve = hard_curve(df, Boot(df))
        x = [100 * c["fp"][0] for c in curve]
        y = [c["cost"][0] for c in curve]
        ax.plot(x, y, color=color, marker=marker, markersize=2.6, label=label)
        if df is held:
            # tau=0.5 lies on the multiplicative-gate marker, so it is left unlabeled.
            offsets = {0.8: (5, 2), 0.2: (6, -9), 0.05: (8, 5), 0.01: (4, 4)}
            for c in curve:
                if c["tau"] in offsets:
                    ax.annotate(rf"$\tau$={c['tau']:g}", (100 * c["fp"][0], c["cost"][0]), textcoords="offset points",
                                xytext=offsets[c["tau"]], fontsize=6, color=BLUE)
            at = [c for c in curve if c["tau"] == HARD][0]
            ax.plot(100 * at["fp"][0], at["cost"][0], marker="*", markersize=9, color=BLUE, markeredgecolor="black", markeredgewidth=0.4,
                    linestyle="none", label=r"preregistered point, $\tau$=0.10", zorder=5)
    btn, btp = pool[pool.cond == "B+T-"], pool[pool.cond == "B+T+"]
    ax.plot(100 * (btn.n_kept_gated > 0).mean(), (btp.iou_cat - btp.iou_cat_gated).mean(), marker="X", markersize=5.5, color=RED,
            linestyle="none", label="multiplicative gate (fails)")
    btn, btp = held[held.cond == "B+T-"], held[held.cond == "B+T+"]
    ax.plot(100 * (btn.n_kept_geom > 0).mean(), (btp.iou_cat - btp.iou_geom).mean(), marker="D", markersize=4.5, color=ORANGE,
            linestyle="none", label="geometry gate (fails)")
    ax.axhline(0.02, color=GREY, linewidth=0.5, linestyle=":")
    ax.text(88, 0.023, "cost bound 0.02", fontsize=6, color=GREY, ha="right")
    ax.set_xscale("log")
    ax.set_xlim(0.3, 100)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50, 100], labels=["0.5", "1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("absent-phrase episodes that still return a mask (%)")
    ax.set_ylabel("mIoU cost on consistent prompts")
    ax.legend(frameon=False, loc="upper right", handlelength=1.6)
    fig.savefig(FIG / "gate.pdf")
    plt.close(fig)


def roc(score, pos, neg):
    s = np.concatenate([score[pos], score[neg]])
    y = np.concatenate([np.ones(pos.sum()), np.zeros(neg.sum())])
    thr = np.unique(s)[::-1]
    tpr = [(s[y == 1] >= t).mean() for t in thr]
    fpr = [(s[y == 0] >= t).mean() for t in thr]
    return np.array([0, *fpr, 1]), np.array([0, *tpr, 1])


def detector_roc() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.75), sharey=True, gridspec_kw=dict(wspace=0.12))
    for ax, name, title in ((axes[0], "real_detector_heldout", "Faster R-CNN, held-out"), (axes[1], "real_retinanet", "RetinaNet")):
        df = load(name)
        pos = (df.correct == 1).to_numpy()
        neg = ((df.correct == 0) & (df.class_absent == 1)).to_numpy()
        for col, color, label, style in (("det_score", GREY, "detector score", "--"), ("max_score", RED, "SAM 3", ":"),
                                         ("max_score_gated", BLUE, "SAM 3, gated", "-")):
            x, y = roc(df[col].to_numpy(float), pos, neg)
            auc = float(np.trapezoid(y, x))
            ax.plot(x, y, color=color, linestyle=style, label=f"{label} ({auc:.3f})")
        ax.plot([0, 1], [0, 1], color="#cbd5e0", linewidth=0.5)
        ax.set_title(title, pad=2)
        ax.set_xlabel("class-absent boxes accepted")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], labels=["0", "0.25", "0.5", "0.75", "1"])
        ax.set_ylim(0, 1.005)
        ax.set_aspect("equal")
        ax.legend(frameon=False, loc="lower right", handlelength=1.5, fontsize=5.8)
    axes[0].set_ylabel("correct boxes accepted")
    fig.savefig(FIG / "roc.pdf")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    presence()
    gate_curve()
    detector_roc()
    print("wrote", sorted(p.name for p in FIG.glob("*.pdf")))
    assert abs(float(sigmoid(0)) - 0.5) < 1e-12
