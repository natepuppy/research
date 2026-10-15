"""Figures for the N1-1 paper from results/*.json written by n1_extensions.py.

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

ALGO_LABEL = {
    "td": "TD(0)",
    "gtd2": "GTD2",
    "gtd2(beta=10a)": "GTD2 (β=10α)",
    "tdc": "TDC",
    "tdc(beta=10a)": "TDC (β=10α)",
    "etd": "ETD(0)",
    "sdtd": "SD-TD (ours)",
}
COLORS = {"td": "tab:red", "gtd2": "tab:blue", "tdc": "tab:green", "etd": "tab:purple", "sdtd": "black"}


def baird_curves() -> None:
    curves = json.loads((RES / "linear_curves_baird.json").read_text())
    summary = json.loads((RES / "linear_summary.json").read_text())
    fixed = json.loads((RES / "linear_fixed_points.json").read_text())
    best = {(r["family"], r["algo"]): r["best_alpha"] for r in summary}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    for ax, fam, title in zip(axes, ["baird", "baird_reward"], ["Baird (reward 0)", "Baird with random rewards"]):
        for algo in ["td", "gtd2", "tdc", "etd", "sdtd"]:
            key = algo if algo not in ("gtd2", "tdc") else algo
            alpha = best[(fam, key)]
            sel = [c for c in curves if c["system"] == fam and c["algo"] == algo and c["beta_ratio"] == 1.0 and abs(c["alpha"] - alpha) < 1e-12]
            if not sel:
                continue
            longest = max(sel, key=lambda c: len(c["curve"]))
            steps = np.array([p[0] for p in longest["curve"]])
            mat = np.full((len(sel), len(steps)), np.nan)
            for i, c in enumerate(sel):
                vals = np.array([p[1] for p in c["curve"]], dtype=float)
                mat[i, : len(vals)] = vals  # diverged runs are truncated; remaining entries stay NaN
            med = np.nanmedian(mat, axis=0)
            ax.plot(steps, med, label=f"{ALGO_LABEL[algo]} α={alpha:g}", color=COLORS[algo], lw=1.6)
        zero = [r["rmsve_zero"] for r in fixed if r["system"] == fam]
        if zero and np.median(zero) > 1e-9:
            ax.axhline(np.median(zero), color="gray", ls=":", lw=1.2, label="predict 0 everywhere")
        ax.set_yscale("log")
        ax.set_xlabel("transitions")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("RMSVE (median over seeds and systems)")
    axes[1].legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_baird_curves.png", dpi=160)
    plt.close(fig)


def family_summary() -> None:
    summary = json.loads((RES / "linear_summary.json").read_text())
    fams = ["baird", "baird_reward", "baird_structure", "random", "on_policy"]
    fam_label = {"baird": "Baird", "baird_reward": "Baird+reward", "baird_structure": "Baird-structure\n(diverging)", "random": "random softmax", "on_policy": "on-policy"}
    algos = ["td", "gtd2", "tdc", "etd", "sdtd"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    width = 0.16
    x = np.arange(len(fams))
    for j, algo in enumerate(algos):
        div = [next(r["diverged_at_best"] for r in summary if r["family"] == f and r["algo"] == algo) for f in fams]
        rm = [next(r["median_tail_rmsve_at_best"] for r in summary if r["family"] == f and r["algo"] == algo) for f in fams]
        axes[0].bar(x + (j - 2) * width, div, width, label=ALGO_LABEL[algo], color=COLORS[algo])
        axes[1].bar(x + (j - 2) * width, rm, width, label=ALGO_LABEL[algo], color=COLORS[algo])
    axes[0].set_ylabel("fraction of runs diverged (best α)")
    axes[1].set_ylabel("median RMSVE at best α (clipped at 1e-2; NaN = all diverged)")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-2, None)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([fam_label[f] for f in fams], fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_family_summary.png", dpi=160)
    plt.close(fig)


def fixed_points() -> None:
    fp = json.loads((RES / "linear_fixed_points.json").read_text())
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for fam, marker, color in [("baird_reward", "s", "tab:red"), ("baird_structure", "o", "tab:orange"), ("random", "^", "tab:blue"), ("on_policy", "x", "tab:green")]:
        rows = [r for r in fp if r["system"] == fam]
        if not rows:
            continue
        td = np.array([r["rmsve_td_fixed_point"] for r in rows])
        dm = np.array([r["rmsve_damped_fixed_point"] for r in rows])
        zero = np.array([r["rmsve_zero"] for r in rows])
        ax.scatter(td / np.maximum(zero, 1e-12), dm / np.maximum(zero, 1e-12), marker=marker, color=color, label=fam, alpha=0.8)
    ax.plot([0, 1.05], [0, 1.05], "k--", lw=0.8)
    ax.set_xlabel("RMSVE of TD fixed point / RMSVE of predicting 0")
    ax.set_ylabel("RMSVE of damped fixed point / RMSVE of predicting 0")
    ax.set_title("Off-policy: damping replaces an exact TD fixed point\nby (almost) the zero predictor. On-policy: no change.")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_fixed_points.png", dpi=160)
    plt.close(fig)


def cartpole() -> None:
    data = json.loads((RES / "cartpole_runs.json").read_text())
    runs = data["runs"]
    groups = [
        ("no target, lr 3e-3, γ=0.99", lambda r: r["tau"] >= 1 and r["damp_k"] == 0 and abs(r["lr"] - 3e-3) < 1e-9 and r["gamma"] == 0.99, "tab:red"),
        ("target net τ=0.005, lr 3e-3", lambda r: r["tau"] < 1 and r["damp_k"] == 0, "tab:blue"),
        ("no target, lr 3e-3, γ=0.9", lambda r: r["tau"] >= 1 and r["damp_k"] == 0 and abs(r["lr"] - 3e-3) < 1e-9 and r["gamma"] == 0.9, "tab:green"),
        ("SD-DQN k=2, lr 3e-3, γ=0.99", lambda r: r["damp_k"] == 2 and abs(r["lr"] - 3e-3) < 1e-9, "black"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for label, pred, color in groups:
        sel = [r for r in runs if pred(r)]
        for i, r in enumerate(sel):
            steps = [row["step"] for row in r["log"]]
            axes[0].plot(steps, [row["q_abs"] for row in r["log"]], color=color, alpha=0.7, lw=1, label=label if i == 0 else None)
            axes[1].plot(steps, [row["lr_times_sharpness"] for row in r["log"]], color=color, alpha=0.7, lw=1)
            axes[2].plot(steps, [row["lam_max_S"] for row in r["log"]], color=color, alpha=0.7, lw=1)
            axes[3].plot(steps, [row["curl_fraction"] for row in r["log"]], color=color, alpha=0.7, lw=1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("mean |Q| on batch")
    axes[1].set_ylabel("lr × sharpness of the TD loss gradient")
    axes[1].axhline(2, color="gray", lw=0.8, ls="--")
    axes[1].set_yscale("log")
    axes[2].set_ylabel("tracked λ_max of sym(Jacobian), TD field")
    axes[2].axhline(0, color="gray", lw=0.8)
    axes[2].set_yscale("symlog", linthresh=1e-2)
    axes[3].set_ylabel("curl fraction ||A||_F / ||K||_F")
    for ax in axes:
        ax.set_xlabel("environment steps")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_cartpole.png", dpi=160)
    plt.close(fig)


def nonlinear() -> None:
    rows = json.loads((RES / "nonlinear_sweep.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 3.4))
    labels = []
    vals = {0.01: [], 0.05: []}
    for r in rows:
        if r["lr"] == 0.01:
            labels.append(f"{r['method']} k={r['k']}" if r["k"] else "no correction")
        vals[r["lr"]].append(r["stable"] / r["n_seeds"])
    x = np.arange(len(labels))
    ax.bar(x - 0.18, vals[0.01], 0.36, label="lr 0.01")
    ax.bar(x + 0.18, vals[0.05], 0.36, label="lr 0.05")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, fontsize=8)
    ax.set_ylabel(f"fraction of {rows[0]['n_seeds']} seeds stable")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_nonlinear.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    for fn in (baird_curves, family_summary, fixed_points, cartpole, nonlinear):
        try:
            fn()
            print("wrote", fn.__name__)
        except FileNotFoundError as exc:
            print("skip", fn.__name__, exc)
