"""Probes for the reformulated hypotheses. Not a paper experiment.

  * N1: does extra rotation stabilize Baird, or only damping the unstable
    symmetric mode?
  * S4-46: seeds the betting sequence needs versus the smallest t-test with
    the same power.
  * S4-1: does a nonlinear value function on Baird show progressive sharpening?

    python probes/explore_reformulations.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.anytime import BettingCS, null_mean_on_unit_interval, rescale_to_unit  # noqa: E402
from lib.baird import GAMMA, LOWER, SOLID_PROBABILITY, expected_semi_gradient, feature_matrix, initial_weights  # noqa: E402
from lib.nets import FuncMLP  # noqa: E402
from lib.sharpness import top_hessian_eigenvalue  # noqa: E402


def baird_jacobian() -> torch.Tensor:
    origin = torch.zeros(8, dtype=torch.float64)
    columns = []
    for index in range(8):
        direction = torch.zeros(8, dtype=torch.float64)
        direction[index] = 1.0
        _, tangent = torch.autograd.functional.jvp(expected_semi_gradient, origin, direction)
        columns.append(tangent)
    return torch.stack(columns, dim=1)


def spectral_radius_step(operator: torch.Tensor, alpha: float) -> float:
    """Largest |1 + alpha * eigenvalue| of the discrete update map."""
    eigvals = torch.linalg.eigvals(torch.eye(operator.shape[0], dtype=operator.dtype) + alpha * operator)
    return float(eigvals.abs().max())


def max_real(operator: torch.Tensor) -> float:
    return float(torch.linalg.eigvals(operator).real.max())


def rotation_sweep(jacobian: torch.Tensor) -> None:
    symmetric = 0.5 * (jacobian + jacobian.T)
    skew = 0.5 * (jacobian - jacobian.T)
    print("\n== N1 rotation scale β in update S + β A ==")
    print(f"{'beta':>6} {'max_real':>10} {'stable_at_0.05':>16} {'norm_400':>10}")
    w0 = torch.tensor(initial_weights(), dtype=torch.float64)
    for beta in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        operator = symmetric + beta * skew
        radius = spectral_radius_step(operator, 0.05)
        weights = w0.clone()
        for _ in range(400):
            weights = weights + 0.05 * (operator @ weights)
            if float(weights.norm()) > 1e6:
                break
        print(f"{beta:6.1f} {max_real(operator):10.3f} {radius:16.3f} {float(weights.norm()):10.1f}")

    eigvals, evecs = torch.linalg.eigh(symmetric)
    unstable = evecs[:, int(torch.argmax(eigvals))]
    print(f"\nunstable symmetric eigenvalue {float(eigvals.max()):.3f}")
    print("== damp that mode: step += -kappa (u·w) u, on top of the full TD field ==")
    print(f"{'kappa':>6} {'max_real':>10} {'radius':>8} {'norm_400':>10}")
    for kappa in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        # dw = J w - kappa (u·w) u, so the operator is J - kappa u u^T
        operator = jacobian - kappa * torch.outer(unstable, unstable)
        weights = w0.clone()
        for _ in range(400):
            weights = weights + 0.05 * (operator @ weights)
            if float(weights.norm()) > 1e6:
                break
        print(
            f"{kappa:6.1f} {max_real(operator):10.3f} {spectral_radius_step(operator, 0.05):8.3f} "
            f"{float(weights.norm()):10.1f}"
        )


def _first_n_at_power(hit_steps: np.ndarray, target: float, cap: int) -> int | None:
    """hit_steps[trial] is the first n that rejected, or cap+1 if it never did."""
    for n in range(2, cap + 1):
        if float(np.mean(hit_steps <= n)) >= target:
            return n
    return None


def power_curve() -> None:
    """Seeds to 80% power. Differences of two betas, bounds [-1, 1], null mean 0."""
    rng = np.random.default_rng(0)
    alpha = 0.05
    trials = 60
    cap = 100
    print("\n== S4-46 seeds to 80% power, t-test vs betting sequence ==", flush=True)
    print(f"{'gap':>6} {'t_n':>6} {'cs_n':>6} {'cs/t':>6}", flush=True)
    # mean of Beta(a, b) minus mean of Beta(b, a) equals (a-b)/(a+b).
    pairs = [(12, 12), (14, 12), (16, 12), (20, 12), (15, 10)]
    null = null_mean_on_unit_interval(-1.0, 1.0)
    for left_a, left_b in pairs:
        gap = left_a / (left_a + left_b) - left_b / (left_a + left_b)
        left = rng.beta(left_a, left_b, size=(trials, cap))
        right = rng.beta(left_b, left_a, size=(trials, cap))
        diffs = left - right
        t_hits = np.full(trials, cap + 1)
        cs_hits = np.full(trials, cap + 1)
        for trial in range(trials):
            series = diffs[trial]
            for n in range(2, cap + 1):
                sample = series[:n]
                if float(np.std(sample)) < 1e-8:
                    rejected = abs(float(np.mean(sample))) > 1e-8
                else:
                    pvalue = float(stats.ttest_1samp(sample, 0.0).pvalue)
                    rejected = pvalue == pvalue and pvalue < alpha
                if rejected:
                    t_hits[trial] = n
                    break
            unit = rescale_to_unit(series, -1.0, 1.0)
            sequence = BettingCS(alpha=alpha, n_grid=81, n_lambda=7)
            for n, value in enumerate(unit, start=1):
                sequence.update(float(value))
                if sequence.excludes(null):
                    cs_hits[trial] = n
                    break
        t_n = _first_n_at_power(t_hits, 0.8, cap)
        cs_n = _first_n_at_power(cs_hits, 0.8, cap)
        ratio = (cs_n / t_n) if t_n and cs_n else float("nan")
        print(f"{gap:6.3f} {str(t_n):>6} {str(cs_n):>6} {ratio:6.2f}", flush=True)


def nonlinear_baird_sharpness() -> None:
    """Off-policy semi-gradient TD through a small MLP on Baird features."""
    print("\n== S4-1 nonlinear Baird, top Hessian eigenvalue of the TD loss ==")
    phi = torch.tensor(feature_matrix(), dtype=torch.float32)
    device = torch.device("cpu")
    for lr in (0.05, 0.2):
        torch.manual_seed(0)
        net = FuncMLP([8, 32, 1], device, seed=0)
        theta = net.theta.detach().clone()
        rng = np.random.default_rng(0)
        last_lambda = float("nan")
        diverged_at = None
        for step in range(1, 401):
            state = int(rng.integers(0, 7))
            solid = bool(rng.random() < SOLID_PROBABILITY)
            nxt = LOWER if solid else int(rng.integers(0, LOWER))
            rho = (1.0 / SOLID_PROBABILITY) if solid else 0.0
            features = phi[state][None]
            nxt_features = phi[nxt][None]
            leaf = theta.detach().requires_grad_(True)
            prediction = net.forward(leaf, features).squeeze()
            bootstrap = net.forward(leaf, nxt_features).squeeze()
            # Semi-gradient: backup value is stop-grad, importance-weighted.
            loss = 0.5 * rho * (prediction - (GAMMA * bootstrap).detach()).pow(2)
            (grad,) = torch.autograd.grad(loss, leaf)
            theta = (leaf.detach() - lr * grad.detach()).detach()
            if float(theta.norm()) > 1e3 or not torch.isfinite(theta).all():
                diverged_at = step
                break
            if step % 80 == 0:
                def hvp(vector: torch.Tensor, theta=theta, features=features, nxt_features=nxt_features, rho=rho) -> torch.Tensor:
                    parameter = theta.detach().requires_grad_(True)
                    pred = net.forward(parameter, features).squeeze()
                    boot = net.forward(parameter, nxt_features).squeeze()
                    value = 0.5 * rho * (pred - GAMMA * boot).pow(2)
                    (derivative,) = torch.autograd.grad(value, parameter, create_graph=True)
                    (product,) = torch.autograd.grad((derivative * vector).sum(), parameter)
                    return product.detach()

                last_lambda = top_hessian_eigenvalue(hvp, theta.numel(), steps=12)
                edge = 2.0 / lr
                print(
                    f"lr {lr:.2f} step {step:3d}  lambda {last_lambda:8.2f}  "
                    f"ratio {last_lambda / edge:6.3f}  norm {float(theta.norm()):7.2f}"
                )
        print(f"lr {lr:.2f} diverged_at {diverged_at} last_lambda {last_lambda:.2f}")


def main() -> None:
    jacobian = baird_jacobian()
    rotation_sweep(jacobian)
    power_curve()
    nonlinear_baird_sharpness()


if __name__ == "__main__":
    main()
