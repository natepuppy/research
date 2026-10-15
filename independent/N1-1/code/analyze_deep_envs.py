"""Mechanism statistics for DQN divergence across environments (CartPole + n1_deep_envs.py runs).

For every (env, config) group: divergence count, final |Q| (median), lr x sharpness during the growth
phase and at the last finite measurement before blow-up, sign of the tracked lambda_max(S), curl fraction.

For diverging runs, "growth onset" is the first measurement where |Q| exceeds 3 x its value at the first
measurement; lr x sharpness at onset separates an outward field (<< 2) from a step-size overshoot (~2).

    python code/analyze_deep_envs.py
Writes results/deep_envs_summary.json and paper/figures/fig6_deep_envs.png.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent  # code/
PROJECT = HERE.parent
RES = PROJECT / "results"
FIG = PROJECT / "paper" / "figures"


def col(run: dict, key: str) -> np.ndarray:
    return np.array([np.nan if e.get(key) is None else e.get(key) for e in run["log"]], dtype=float)


def finite(a: np.ndarray) -> np.ndarray:
    return a[np.isfinite(a)]


def load() -> list[dict]:
    runs = []
    cp = json.load(open(RES / "cartpole_runs.json"))["runs"]
    for r in cp:
        runs.append({**r, "env": "cartpole"})
    runs += json.load(open(RES / "deep_envs_runs.json"))["runs"]
    return runs


def group_stats(rs: list[dict]) -> dict:
    out: dict = {"n": len(rs)}
    div = [r for r in rs if r.get("diverged_at") is not None]
    out["diverged"] = len(div)
    out["onset_steps"] = sorted(r["diverged_at"] for r in div)
    q_final = []
    for r in rs:
        q = finite(col(r, "q_abs"))
        q_final.append(q[-1] if len(q) else np.nan)
    out["q_final_median"] = float(np.nanmedian(q_final))
    out["q_final_max"] = float(np.nanmax(q_final))
    ls_median = [float(np.nanmedian(col(r, "lr_times_sharpness"))) for r in rs]
    out["lr_sharp_median_over_runs"] = float(np.nanmedian(ls_median))
    out["lam_pos_runs"] = int(sum(1 for r in rs if np.nanmax(col(r, "lam_max_S")) > 0))
    out["lam_median"] = float(np.nanmedian([np.nanmedian(col(r, "lam_max_S")) for r in rs]))
    out["curl_median"] = float(np.nanmedian([np.nanmedian(col(r, "curl_fraction")) for r in rs]))
    out["final_return_median"] = float(np.nanmedian([r["final_return"] for r in rs]))
    onset_ls, last_ls = [], []
    for r in div:
        q = col(r, "q_abs")
        ls = col(r, "lr_times_sharpness")
        ok = np.isfinite(q)
        if ok.sum() == 0:
            continue
        q0 = q[ok][0]
        idx = np.flatnonzero(ok & (q > 3 * max(q0, 1e-3)))
        if idx.size:
            onset_ls.append(float(ls[idx[0]]))
        last_ls.append(float(finite(ls)[-1]) if len(finite(ls)) else float("nan"))
    out["lr_sharp_at_growth_onset"] = onset_ls
    out["lr_sharp_last_finite"] = last_ls
    return out


def main() -> None:
    runs = load()
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in runs:
        groups[(r["env"], r["lr"], r["tau"], r["gamma"], r["damp_k"])].append(r)
    summary = []
    for key in sorted(groups):
        s = group_stats(groups[key])
        s.update(dict(zip(("env", "lr", "tau", "gamma", "damp_k"), key)))
        summary.append(s)
        onset = s["lr_sharp_at_growth_onset"]
        last = s["lr_sharp_last_finite"]
        print(
            f"{s['env']:14s} lr {s['lr']:<6g} tau {s['tau']:<6g} g {s['gamma']} k {s['damp_k']} | div {s['diverged']}/{s['n']}"
            f" onset {s['onset_steps'][:4]}{'...' if len(s['onset_steps']) > 4 else ''} | |Q| fin {s['q_final_median']:.1f}"
            f" | lr*sh med {s['lr_sharp_median_over_runs']:.3f}"
            f" | at growth onset {np.median(onset) if onset else float('nan'):.2f} [{min(onset) if onset else float('nan'):.2f},{max(onset) if onset else float('nan'):.2f}]"
            f" | last finite {np.median(last) if last else float('nan'):.2f}"
            f" | lam>0 {s['lam_pos_runs']}/{s['n']} | ret {s['final_return_median']:.1f}"
        )
    (RES / "deep_envs_summary.json").write_text(json.dumps(summary, indent=1))

    # Figure: per environment, lr x sharpness at growth onset (diverging, no-target, k=0 runs) and the
    # full distribution of lr x sharpness in non-diverging no-target runs, against the threshold 2.
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    envs = ["cartpole", "acrobot", "mountaincar", "breakout", "space_invaders"]
    ax = axes[0]
    xs, ys, cs = [], [], []
    for i, env in enumerate(envs):
        for key, rs in groups.items():
            if key[0] != env or key[2] != 1.0 or key[3] != 0.99 or key[4] != 0:
                continue
            for r in rs:
                q = col(r, "q_abs")
                ls = col(r, "lr_times_sharpness")
                ok = np.isfinite(q)
                if not ok.any():
                    continue
                diverged = r.get("diverged_at") is not None
                if diverged:
                    idx = np.flatnonzero(ok & (q > 3 * max(q[ok][0], 1e-3)))
                    if not idx.size:
                        continue
                    val = ls[idx[0]]
                else:
                    val = np.nanmedian(ls)
                xs.append(i + (0.15 if diverged else -0.15) + 0.04 * np.random.default_rng(r["seed"]).normal())
                ys.append(max(val, 1e-4))
                cs.append("tab:red" if diverged else "tab:gray")
    ax.scatter(xs, ys, c=cs, s=14, alpha=0.7)
    ax.axhline(2.0, ls="--", c="k", lw=1)
    ax.set_yscale("log")
    ax.set_xticks(range(len(envs)))
    ax.set_xticklabels(["CartPole", "Acrobot", "MountainCar", "Breakout", "SpaceInv."], fontsize=8)
    ax.set_ylabel("lr × sharpness")
    ax.set_title("no target net, γ=0.99: red = at growth onset (diverging)\ngray = median over run (stable); dashed = 2", fontsize=8)
    ax = axes[1]
    labels, divs, qs = [], [], []
    for env in envs:
        for key, rs in sorted(groups.items()):
            if key[0] != env:
                continue
            tag = f"{env[:5]} lr{key[1]:g}" + (" targ" if key[2] < 1 else "") + (f" γ{key[3]}" if key[3] != 0.99 else "") + (f" SD k{key[4]}" if key[4] else "")
            labels.append(tag)
            divs.append(sum(1 for r in rs if r.get("diverged_at") is not None) / len(rs))
    ax.barh(range(len(labels)), divs, color="tab:red")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of runs diverged")
    ax.set_title("divergence fraction per configuration", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_deep_envs.png", dpi=150)
    print("wrote", FIG / "fig6_deep_envs.png")


if __name__ == "__main__":
    main()
