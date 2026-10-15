"""Figure 5: betting-test multiplier over the one-look t-test against h / sd, pooled over every real-data
power setting (BBF Atari-100k, TD-MPC2 DMControl and Meta-World, fresh DQN runs), plus the parametric
anytime e-process multiplier for reference, and Proposition 2's prediction.

    python code/make_figures_real.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent  # code/
PROJECT = HERE.parent
RES = PROJECT / "results"
FIG = PROJECT / "paper" / "figures"


def dataset_of(name: str) -> str | None:
    if name.startswith("null") or name.startswith("beta"):
        return None
    if name.startswith("bbf_vs") and "aggregate" in name and "bound" not in name:
        return "BBF aggregate (55 games)"
    if name.startswith("bbf_vs") and "aggregate" not in name:
        return "BBF single game"
    if name.startswith("dmcontrol") and "subset5" in name:
        return "DMControl, 5-task slice"
    if name.startswith("dmcontrol") and "bound" not in name:
        return "DMControl aggregate (39 tasks)"
    if name.startswith("metaworld") and "subset5" in name:
        return "Meta-World, 5-task slice"
    if name.startswith("metaworld") and "bound" not in name:
        return "Meta-World aggregate (50 tasks)"
    if name.startswith("fresh") and name.endswith("bootstrap"):
        return "fresh DQN runs (bootstrap pool)"
    return None


def main() -> None:
    settings = []
    for f in ("extensions_summary.json", "real_data_summary.json"):
        p = RES / f
        if p.exists():
            settings += json.load(open(p))["settings"]
    alpha = 0.05
    markers = {"BBF aggregate (55 games)": "o", "BBF single game": "s", "DMControl aggregate (39 tasks)": "^", "DMControl, 5-task slice": "v",
               "Meta-World aggregate (50 tasks)": "D", "Meta-World, 5-task slice": "d", "fresh DQN runs (bootstrap pool)": "P"}
    colors = {k: c for k, c in zip(markers, plt.cm.tab10.colors)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    seen = set()
    for s in settings:
        ds = dataset_of(s["name"])
        if ds is None:
            continue
        t = s["t_fixed"]["n80"]
        b = s["bet_plugin"]["n80"]
        e = s["seq_t"]["n80"]
        if t is None or t == 0:
            continue
        h = (s["bound"][1] - s["bound"][0]) / 2
        ratio = h / s["data_sd"]
        lab = ds if ds not in seen else None
        seen.add(ds)
        if b is not None:
            axes[0].scatter(ratio, b / t, marker=markers[ds], color=colors[ds], s=36, label=lab, alpha=0.85)
        else:
            axes[0].scatter(ratio, 200 / t, marker=markers[ds], color=colors[ds], s=36, facecolors="none", alpha=0.85, label=lab)
        if e is not None:
            axes[1].scatter(ratio, e / t, marker=markers[ds], color=colors[ds], s=36, alpha=0.85, label=lab)
        # Proposition 2 prediction for the betting count: h log(1/alpha) / delta, against the t-test's count.
        delta = abs(s["data_mean"])
        if delta > 0:
            axes[0].scatter(ratio, (h * np.log(1 / alpha) / delta) / t, marker="_", color="k", s=40, alpha=0.5)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("declared half-width / sd of paired differences")
        ax.axhline(1, color="gray", lw=0.8, ls=":")
    axes[0].set_ylabel("n80 multiplier over one-look t-test")
    axes[0].set_title("anytime betting test (black ticks: Prop. 2 prediction;\nopen markers: not reached within 200 seeds, lower bound)", fontsize=9)
    axes[1].set_title("anytime t e-process", fontsize=9)
    axes[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_multiplier_vs_ratio.png", dpi=150)
    print("wrote", FIG / "fig5_multiplier_vs_ratio.png")


if __name__ == "__main__":
    main()
