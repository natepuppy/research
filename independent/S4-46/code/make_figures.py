"""Figures for the S4-46 paper from results/extensions_summary.json.

    python code/make_figures.py
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
FIG.mkdir(exist_ok=True, parents=True)

METHODS = ["t_fixed", "gs_pocock", "gs_obf", "seq_t", "bet_fixed", "bet_plugin", "eb_cs"]
LABEL = {
    "t_fixed": "t-test, one look",
    "t_peek": "t-test, peek every seed",
    "gs_pocock": "group-seq. Pocock (5 looks)",
    "gs_obf": "group-seq. O'Brien-Fleming (5 looks)",
    "seq_t": "anytime t e-process",
    "seq_t_narrow": "anytime t e-process (prior sd 0.3)",
    "bet_fixed": "betting, one look",
    "bet_plugin": "betting, anytime",
    "eb_cs": "empirical-Bernstein CS",
}
COLOR = {
    "t_fixed": "tab:blue",
    "t_peek": "tab:red",
    "gs_pocock": "tab:cyan",
    "gs_obf": "tab:cyan",
    "seq_t": "tab:purple",
    "seq_t_narrow": "tab:purple",
    "bet_fixed": "tab:orange",
    "bet_plugin": "tab:brown",
    "eb_cs": "tab:gray",
}


def load():
    return json.loads((RES / "extensions_summary.json").read_text())


def n80(s, m, cap):
    v = s[m]["n80"]
    return cap * 1.15 if v is None else v


def fig_decomposition(data):
    cap = data["cap"]
    names = ["beta_gap0.14_conc8", "beta_gap0.14_conc24", "beta_gap0.14_conc80", "bbf_vs_RR2_BBF-SPR_aggregate", "bbf_vs_RR8_BBF_aggregate", "bbf_vs_RR2_BBF-Resets_aggregate"]
    titles = ["beta, noise fills [-1,1]", "beta, intermediate", "beta, noise ≪ bound", "BBF vs BBF−SPR (55-game aggregate)", "BBF RR2 vs RR8 (aggregate)", "BBF vs BBF−Resets (aggregate)"]
    sets = {s["name"]: s for s in data["settings"]}
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
    for ax, name, title in zip(axes.ravel(), names, titles):
        s = sets[name]
        vals = [n80(s, m, cap) for m in METHODS]
        bars = ax.bar(range(len(METHODS)), vals, color=[COLOR[m] for m in METHODS])
        for b, m in zip(bars, METHODS):
            v = s[m]["n80"]
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), ">cap" if v is None else str(v), ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([LABEL[m] for m in METHODS], rotation=35, ha="right", fontsize=7)
        ax.set_title(f"{title}\ngap {s['data_mean']:+.3f}, sd {s['data_sd']:.3f}", fontsize=9)
        ax.set_ylabel("seeds to 80% power")
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_n80_decomposition.png", dpi=160)
    plt.close(fig)


def fig_bound_sweep(data):
    sets = {s["name"]: s for s in data["settings"]}
    rows = [("bbf_vs_RR2_BBF-SPR_aggregate", 1.0)] + [(f"bbf_vs_RR2_BBF-SPR_aggregate_bound{h:g}", h) for h in (0.5, 0.25, 0.1, 0.05)]
    halves = [h for _, h in rows]
    bet = [sets[n]["bet_plugin"]["n80"] for n, _ in rows]
    viol = [sets[n]["violated_bound_rate"] for n, _ in rows]
    t_n = sets[rows[0][0]]["t_fixed"]["n80"]
    seq_n = sets[rows[0][0]]["seq_t"]["n80"]
    sd = sets[rows[0][0]]["data_sd"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(np.array(halves) / sd, bet, "o-", color=COLOR["bet_plugin"], label="betting, anytime")
    for i, (h, b, v) in enumerate(zip(halves, bet, viol)):
        offset = (6, 4) if i % 2 == 0 else (6, -22)
        if i == len(halves) - 1:
            offset = (-10, 14)
        ax.annotate(f"bound ±{h:g}\n{v:.0%} of paths violate it", (h / sd, b), textcoords="offset points", xytext=offset, fontsize=7, color="tab:red" if v > 0 else "black")
    ax.axhline(t_n, color=COLOR["t_fixed"], ls="--", label=f"t-test, one look ({t_n})")
    ax.axhline(seq_n, color=COLOR["seq_t"], ls="--", label=f"anytime t e-process ({seq_n})")
    ax.set_xscale("log")
    ax.set_xlabel("declared half-width / sd of the paired differences")
    ax.set_ylabel("seeds to 80% power")
    ax.set_title("BBF vs BBF−SPR, 55-game aggregate: the betting test's\ncost is set by how loose the declared bound is", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_bound_sweep.png", dpi=160)
    plt.close(fig)


def fig_nulls(data):
    sets = [s for s in data["settings"] if s["name"].startswith("null")]
    methods = ["t_peek", "gs_pocock", "gs_obf", "seq_t", "bet_plugin", "eb_cs"]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    x = np.arange(len(sets))
    width = 0.8 / (len(methods) + 1)
    for j, m in enumerate(methods):
        ax.bar(x + (j - len(methods) / 2) * width, [s[m]["reject_rate"] for s in sets], width, label=LABEL[m], color=COLOR[m])
    ax.bar(x + (len(methods) - len(methods) / 2) * width, [s["t_fixed"]["reject_rate_curve"][9] for s in sets], width, label="t-test, one look at n=10", color=COLOR["t_fixed"])
    ax.axhline(data["alpha"], color="k", ls="--", lw=0.8, label="α")
    ax.set_xticks(x)
    ax.set_xticklabels([s["name"].replace("null_", "").replace("_", "\n") for s in sets], fontsize=7)
    ax.set_ylabel("false-rejection rate (any time ≤ 200 seeds)")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.legend(fontsize=7, ncol=4)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_nulls.png", dpi=160)
    plt.close(fig)


def fig_games(data):
    cap = data["cap"]
    sets = [s for s in data["settings"] if ("_RR2_BBF-SPR_" in s["name"] or "_RR8_BBF_" in s["name"]) and "aggregate" not in s["name"]]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for s in sets:
        a = s["seq_t"]["n80"]
        b = s["bet_plugin"]["n80"]
        if a is None or b is None:
            continue
        ratio = s["data_sd"]
        sc = ax.scatter(a, b, c=[ratio], vmin=0, vmax=0.45, cmap="viridis", s=40, edgecolor="k", lw=0.5)
        ax.annotate(s["name"].split("_")[-1], (a, b), fontsize=6, textcoords="offset points", xytext=(3, 3))
    lim = [0, cap]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("anytime t e-process: seeds to 80% power")
    ax.set_ylabel("betting (bound [-1,1]): seeds to 80% power")
    ax.set_title("Single Atari games, BBF ablations.\nColor: sd of paired differences", fontsize=9)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("sd of paired difference")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_games.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    data = load()
    for fn in (fig_decomposition, fig_bound_sweep, fig_nulls, fig_games):
        fn(data)
        print("wrote", fn.__name__)
