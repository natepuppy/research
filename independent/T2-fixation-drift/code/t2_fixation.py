"""T2. Selection vs drift: a fixation theory of rare correct modes under finite-group GRPO.

Pilot version (E1-lite). A tiny transformer (lib/tiny_lm.py) with single-token
answers, so every prompt's success probability p_i is exact from the logits.
We pretrain it so that p_i(0) is spread over (0.002, 0.95), then run exact
GRPO with group size G and watch each prompt fix (p -> 1), go extinct
(p -> 0) or wander.

The model being tested. In log-odds l = log p/(1-p), one GRPO step on a
2-outcome tabular policy with group-normalised advantages moves the correct
logit by exactly

    delta l = eta * sqrt(k (G - k)) / G,   k ~ Binomial(G, p),

which is zero when the group is all-pass or all-fail (derived in
`tabular_step_check`, verified numerically). Shared parameters add a second
term: updates on *other* prompts move l_i even when prompt i is not in the
batch. We model that as additive noise with mean mu_int and std sigma_int
per step (hitchhiking), and we *measure* both from held-out prompts that are
never trained on. We also measure the effective eta from the regression of
observed delta l on sqrt(k(G-k)) for prompts in the batch. The resulting
chain has no parameter fitted to fixation outcomes; it predicts the fixation
fraction per (p_0 bin, G, eta) and the full kernel H(q | p_0) out of sample.

Predictions checked here
  P1  H(q | p_0) at boundary p_0 (0.01-0.05) is bimodal: fixation or not,
      little mass in between.
  P2  Fixation probability rises with G at fixed eta with diminishing
      returns; at fixed rollout budget B*G the coverage of boundary prompts
      has an interior optimum or plateau.
  P4  Allocating per-prompt group size G_i ~ 1/p_hat_i (p_hat from the
      running reward history) at matched budget expands coverage over
      uniform G.

Kill test: observed fixation fractions do not track the calibrated chain
(MAE > 0.2 or the ordering in G disagrees), or the boundary kernel is
unimodal.

    python code/t2_fixation.py --smoke
    python code/t2_fixation.py
    python code/t2_fixation.py --full
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent  # code/
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

from lib.io_utils import kill_test, load_csv, save_csv  # noqa: E402
from lib.tiny_lm import TinyLM, grpo_loss, make_family, make_prompts, n_params, pretrain, success_probs  # noqa: E402

VOCAB, PROMPT_LEN, N_ANSWERS = 32, 8, 16
BIN_EDGES = np.array([0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
FIX_LOGIT = 2.0  # q > 0.88 counts as fixed
WINDOW = 50


def logit(p: torch.Tensor | np.ndarray):
    if isinstance(p, torch.Tensor):
        p = p.clamp(1e-6, 1 - 1e-6)
        return torch.log(p) - torch.log1p(-p)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def tabular_step_check() -> float:
    """Numerically confirm delta theta_correct = eta sqrt(k(G-k)) / G for 2-outcome tabular GRPO (group-mean loss)."""
    G, eta = 8, 1.0
    worst = 0.0
    for p in (0.05, 0.3, 0.7):
        theta = torch.tensor([math.log(p), math.log(1 - p)], requires_grad=True)
        for k in range(1, G):
            r = torch.tensor([1.0] * k + [0.0] * (G - k))
            adv = (r - r.mean()) / r.std(unbiased=False)
            idx = torch.tensor([0] * k + [1] * (G - k))
            logp = theta.log_softmax(0)[idx]
            (grad,) = torch.autograd.grad((adv * logp).mean(), theta)
            worst = max(worst, abs(float(grad[0]) - eta * math.sqrt(k * (G - k)) / G))
    return worst


def own_term(k, G):
    """Size of the group-normalised GRPO step on the correct log-odds, up to eta."""
    return np.sqrt(k * (G - k)) / G


def build_base(seed: int, n_train: int, n_held: int, pretrain_steps: int, device) -> dict:
    torch.manual_seed(seed)
    family = make_family(PROMPT_LEN, N_ANSWERS, VOCAB, seed=seed)
    prompts = make_prompts(n_train + n_held, PROMPT_LEN, VOCAB, seed=seed + 1).to(device)
    answers = family.answers(prompts.cpu()).to(device)
    # target log-odds spread uniformly over [-6.5, 3]; held-out prompts get the same spread
    generator = torch.Generator().manual_seed(seed + 2)
    target_logit = torch.rand(n_train + n_held, generator=generator) * 9.5 - 6.5
    target_p = torch.sigmoid(target_logit).to(device)
    model = TinyLM(VOCAB, PROMPT_LEN, N_ANSWERS).to(device)
    loss = pretrain(model, prompts, answers, target_p, steps=pretrain_steps, seed=seed)
    return {"model": model, "prompts": prompts, "answers": answers, "n_train": n_train, "pretrain_loss": loss}


def run_grpo(base: dict, G: int, lr: float, steps: int, batch_prompts: int, seed: int, adaptive: bool = False, budget: int | None = None) -> dict:
    """Returns per-prompt p trajectories (steps+1, N) and per-step own-update records."""
    import copy

    model = copy.deepcopy(base["model"])
    prompts, answers, n_train = base["prompts"], base["answers"], base["n_train"]
    N = prompts.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    traj = torch.zeros(steps + 1, N)
    traj[0] = success_probs(model, prompts, answers).cpu()
    own = []  # (step, prompt, k, G_i)
    p_hat = torch.full((n_train,), float("nan"))
    for t in range(steps):
        idx = torch.randperm(n_train, generator=gen)[:batch_prompts]
        if adaptive:
            est = p_hat[idx].clone()
            est[torch.isnan(est)] = float(torch.nanmean(p_hat)) if not torch.isnan(p_hat).all() else 0.3
            raw = 1.0 / est.clamp_min(0.02)
            sizes = (raw / raw.sum() * budget).round().clamp(2, 32).long()
        else:
            sizes = torch.full((batch_prompts,), G, dtype=torch.long)
        loss, info = grpo_loss(model, prompts[idx], answers[idx], sizes, generator=gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            traj[t + 1] = success_probs(model, prompts, answers).cpu()
        # k per prompt is the group's observed correct count; p_hat is the EMA of the
        # observed group mean reward, which is all a real trainer would have.
        for i, k, g in zip(idx.tolist(), info["group_correct"].tolist(), sizes.tolist()):
            own.append((t, i, int(k), int(g)))
            obs_p = k / g
            p_hat[i] = obs_p if torch.isnan(p_hat[i]) else 0.7 * p_hat[i] + 0.3 * obs_p
    return {"traj": traj, "own": own}


P_BINS = np.array([0.0, 0.01, 0.03, 0.1, 0.3, 0.7, 1.01])  # interference is measured per current-p bin


def calibrate(traj: torch.Tensor, own: list, n_train: int) -> list[dict]:
    """Per-window interference (mu_int, sigma_int) from held-out prompts, binned by the
    held-out prompt's *current* p, and eta_eff from in-batch prompts.

    The p-binning matters: pooled over all held-out prompts the drift averages to
    ~0, but rare modes drift down and common modes drift up (updates on other
    prompts sharpen the shared representation). Nothing here is fitted to the
    fixation outcomes of the trained prompts.
    """
    L = logit(traj.numpy())
    steps = L.shape[0] - 1
    held = L[:, n_train:]
    dheld = np.diff(held, axis=0)  # (steps, n_held)
    pheld = traj[:-1, n_train:].numpy()
    own_arr = np.array(own) if own else np.zeros((0, 4))
    windows = []
    for start in range(0, steps, WINDOW):
        end = min(start + WINDOW, steps)
        d = dheld[start:end]
        p = pheld[start:end]
        mu_int = float(d.mean())
        sigma_int = float(d.std())
        mu_bins, sigma_bins = [], []
        for lo, hi in zip(P_BINS[:-1], P_BINS[1:]):
            m = (p >= lo) & (p < hi)
            if m.sum() >= 20:
                mu_bins.append(float(d[m].mean()))
                sigma_bins.append(float(d[m].std()))
            else:
                mu_bins.append(mu_int)
                sigma_bins.append(sigma_int)
        sel = own_arr[(own_arr[:, 0] >= start) & (own_arr[:, 0] < end)] if len(own_arr) else own_arr
        eta_eff = float("nan")
        if len(sel):
            t_idx, p_idx, k, g = sel[:, 0].astype(int), sel[:, 1].astype(int), sel[:, 2], sel[:, 3]
            dl = L[t_idx + 1, p_idx] - L[t_idx, p_idx] - mu_int
            x = own_term(k, g)
            mixed = x > 0
            if mixed.sum() > 5 and (x[mixed] ** 2).sum() > 0:
                eta_eff = float((x[mixed] * dl[mixed]).sum() / (x[mixed] ** 2).sum())
        row = {"start": start, "end": end, "mu_int": mu_int, "sigma_int": sigma_int, "eta_eff": eta_eff}
        for b, (m, s) in enumerate(zip(mu_bins, sigma_bins)):
            row[f"mu_int_b{b}"] = m
            row[f"sigma_int_b{b}"] = s
        windows.append(row)
    return windows


def chain_predict(p0: float, G: int, windows: list[dict], include_prob: float, n_paths: int = 2000, seed: int = 0, p_dependent: bool = True) -> np.ndarray:
    """Simulate the calibrated 2-outcome chain; returns final log-odds for n_paths.

    p_dependent=True uses the per-p-bin interference; False uses the pooled
    (mu_int, sigma_int), which is the naive neutral-drift model.
    """
    rng = np.random.default_rng(seed)
    l = np.full(n_paths, logit(np.array(p0)))
    n_bins = len(P_BINS) - 1
    for w in windows:
        eta = w["eta_eff"] if np.isfinite(w["eta_eff"]) else 0.0
        mu_b = np.array([w.get(f"mu_int_b{b}", w["mu_int"]) for b in range(n_bins)])
        sig_b = np.array([w.get(f"sigma_int_b{b}", w["sigma_int"]) for b in range(n_bins)])
        for _ in range(w["end"] - w["start"]):
            p = 1.0 / (1.0 + np.exp(-l))
            k = rng.binomial(G, p)
            own = eta * own_term(k, G) * (rng.random(n_paths) < include_prob)
            if p_dependent:
                b = np.clip(np.digitize(p, P_BINS) - 1, 0, n_bins - 1)
                mu, sig = mu_b[b], sig_b[b]
            else:
                mu, sig = w["mu_int"], w["sigma_int"]
            l = l + own + mu + sig * rng.standard_normal(n_paths)
    return l


def bin_index(p0: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(p0, BIN_EDGES) - 1, 0, len(BIN_EDGES) - 2)


def kernel_shape(l_final: np.ndarray, l0: np.ndarray) -> dict:
    fixed = (l_final > FIX_LOGIT).mean()
    intermediate = ((l_final > l0 + 1.0) & (l_final <= FIX_LOGIT)).mean()
    stayed = (np.abs(l_final - l0) <= 1.0).mean()
    declined = (l_final < l0 - 1.0).mean()
    bimodal = fixed >= 0.15 and (stayed + declined) >= 0.15 and intermediate < min(fixed, stayed + declined)
    return {"fixed": float(fixed), "intermediate": float(intermediate), "stayed": float(stayed), "declined": float(declined), "bimodal": bool(bimodal)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=PROJECT / "results")
    parser.add_argument("--resume", action="store_true", help="skip seeds already present in results/cells.csv")
    parser.add_argument("--summarize", action="store_true", help="only re-summarise existing results")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    if args.smoke:
        n_train, n_held, pre_steps, steps, B, seeds = 256, 64, 100, 20, 32, [0]
        groups, lrs = [2, 8], [1e-3]
    elif args.full:
        n_train, n_held, pre_steps, steps, B, seeds = 4096, 1024, 6000, 800, 64, list(range(10))
        groups, lrs = [2, 4, 8, 16, 32], [3e-4, 1e-3]
    else:
        n_train, n_held, pre_steps, steps, B, seeds = 2048, 512, 3000, 1500, 64, [0, 1, 2]
        groups, lrs = [2, 4, 8, 16, 32], [3e-4, 1e-3]
    budget = B * 8

    check = tabular_step_check()
    print(f"tabular GRPO step check: max |grad - sqrt(k(G-k))| = {check:.2e}")

    cells, prompts_rows, window_rows, budget_rows = [], [], [], []
    if args.resume or args.summarize:
        cells = load_csv(args.out / "cells.csv")
        budget_rows = load_csv(args.out / "budget.csv")
        window_rows = load_csv(args.out / "windows.csv")
        prompts_rows = load_csv(args.out / "prompts_seed0.csv")
    done_seeds = {r["seed"] for r in cells}
    for seed in seeds:
        if args.summarize or seed in done_seeds:
            continue
        t0 = time.time()
        base = build_base(seed, n_train, n_held, pre_steps, device)
        p0 = success_probs(base["model"], base["prompts"], base["answers"]).cpu().numpy()
        print(f"seed {seed}: base {n_params(base['model'])} params, pretrain loss {base['pretrain_loss']:.3f}, p0 quantiles {np.quantile(p0[:n_train], [0.05, 0.25, 0.5, 0.75, 0.95]).round(3)} ({time.time()-t0:.0f}s)", flush=True)
        configs = [(G, lr, False) for G in groups for lr in lrs]
        # P2 at fixed rollout budget: B_G = budget / G prompts per step
        # P4: adaptive group size at the same budget as G = 8
        configs.append((8, lrs[-1], True))
        for G, lr, adaptive in configs:
            t1 = time.time()
            run = run_grpo(base, G, lr, steps, B, seed, adaptive=adaptive, budget=budget)
            traj = run["traj"]
            windows = calibrate(traj, run["own"], n_train)
            l0 = logit(traj[0, :n_train].numpy())
            lT = logit(traj[-1, :n_train].numpy())
            bins = bin_index(traj[0, :n_train].numpy())
            tag = "adaptive" if adaptive else f"G{G}"
            for w in windows:
                window_rows.append({"seed": seed, "config": tag, "lr": lr, **w})
            include = B / n_train
            for b in range(len(BIN_EDGES) - 1):
                mask = bins == b
                if mask.sum() < 5:
                    continue
                shape = kernel_shape(lT[mask], l0[mask])
                row = {"seed": seed, "config": tag, "G": G, "lr": lr, "adaptive": adaptive, "bin": b, "p0_lo": BIN_EDGES[b], "p0_hi": BIN_EDGES[b + 1], "n": int(mask.sum()), "obs_fixed": shape["fixed"], "obs_intermediate": shape["intermediate"], "obs_stayed": shape["stayed"], "obs_declined": shape["declined"], "obs_bimodal": shape["bimodal"]}
                if not adaptive:
                    centre = float(np.sqrt(BIN_EDGES[b] * BIN_EDGES[b + 1]))
                    l_centre = logit(np.array(centre))
                    pred = chain_predict(centre, G, windows, include, seed=seed, p_dependent=True)
                    pshape = kernel_shape(pred, np.full_like(pred, l_centre))
                    naive = chain_predict(centre, G, windows, include, seed=seed, p_dependent=False)
                    nshape = kernel_shape(naive, np.full_like(naive, l_centre))
                    row.update({"pred_fixed": pshape["fixed"], "pred_intermediate": pshape["intermediate"], "pred_bimodal": pshape["bimodal"], "naive_fixed": nshape["fixed"], "naive_bimodal": nshape["bimodal"]})
                cells.append(row)
            # boundary coverage: fraction of prompts with p0 in [0.01, 0.05] that fixed
            boundary = (traj[0, :n_train].numpy() >= 0.01) & (traj[0, :n_train].numpy() < 0.05)
            budget_rows.append({"seed": seed, "config": tag, "G": G, "lr": lr, "adaptive": adaptive, "rollouts_per_step": int(B * G) if not adaptive else budget, "boundary_fixed": float((lT[boundary] > FIX_LOGIT).mean()) if boundary.any() else float("nan"), "mean_final_p": float(traj[-1, :n_train].mean()), "heldout_final_p": float(traj[-1, n_train:].mean()), "sigma_int_mean": float(np.mean([w["sigma_int"] for w in windows])), "eta_eff_mean": float(np.nanmean([w["eta_eff"] for w in windows]))})
            if seed == seeds[0]:
                for i in range(n_train):
                    prompts_rows.append({"seed": seed, "config": tag, "lr": lr, "prompt": i, "p0": float(traj[0, i]), "pT": float(traj[-1, i])})
            bl = budget_rows[-1]
            print(f"  {tag:9s} lr {lr:.0e}: boundary fixed {bl['boundary_fixed']:.2f}  mean p {bl['mean_final_p']:.3f}  held-out p {bl['heldout_final_p']:.3f}  sigma_int {bl['sigma_int_mean']:.3f}  eta_eff {bl['eta_eff_mean']:.4f} ({time.time()-t1:.0f}s)", flush=True)
        save_csv(args.out / "cells.csv", cells)
        save_csv(args.out / "budget.csv", budget_rows)
        save_csv(args.out / "windows.csv", window_rows)
        save_csv(args.out / "prompts_seed0.csv", prompts_rows)

    # ---- verdict ----
    fixed_cells = [c for c in cells if not c["adaptive"] and "pred_fixed" in c]
    mae = float(np.mean([abs(c["obs_fixed"] - c["pred_fixed"]) for c in fixed_cells])) if fixed_cells else float("nan")
    naive_mae = float(np.mean([abs(c["obs_fixed"] - c["naive_fixed"]) for c in fixed_cells])) if fixed_cells else float("nan")
    bias = float(np.mean([c["obs_fixed"] - c["pred_fixed"] for c in fixed_cells])) if fixed_cells else float("nan")
    boundary_cells = [c for c in cells if not c["adaptive"] and c["bin"] in (2, 3)]  # p0 in [0.01, 0.05)
    bimodal_frac = float(np.mean([c["obs_bimodal"] for c in boundary_cells])) if boundary_cells else float("nan")
    # G ordering: fixation at the boundary should be non-decreasing in G at each lr
    orderings = []
    for lr in lrs:
        means = []
        for G in groups:
            vals = [c["obs_fixed"] for c in boundary_cells if c["G"] == G and c["lr"] == lr]
            if vals:
                means.append((G, float(np.mean(vals))))
        if len(means) >= 2 and np.std([m[1] for m in means]) > 1e-9:
            rho = np.corrcoef([m[0] for m in means], [m[1] for m in means])[0, 1]
            orderings.append(rho)
    g_rho = float(np.nanmean(orderings)) if orderings else float("nan")
    adaptive_gain = float("nan")
    uni = [r["boundary_fixed"] for r in budget_rows if r["config"] == "G8" and r["lr"] == lrs[-1]]
    ada = [r["boundary_fixed"] for r in budget_rows if r["adaptive"]]
    if uni and ada:
        adaptive_gain = float(np.nanmean(ada) - np.nanmean(uni))
    text = (
        f"calibrated-chain MAE on fixation fraction {mae:.3f} (bias {bias:+.3f}) over {len(fixed_cells)} cells with p-dependent interference, "
        f"vs {naive_mae:.3f} with pooled neutral drift; boundary kernel bimodal in {bimodal_frac:.0%} of cells; "
        f"corr(G, boundary fixation) {g_rho:+.2f}; adaptive-G gain at matched budget {adaptive_gain:+.3f}."
    )
    if not fixed_cells or not boundary_cells:
        supported = None
    elif mae <= 0.1 and bimodal_frac >= 0.5 and g_rho > 0:
        supported = True
    elif mae > 0.2 or bimodal_frac < 0.25 or g_rho < 0:
        supported = False
    else:
        supported = None
    verdict = kill_test(supported, text)
    (args.out / "verdict.txt").write_text(verdict + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
