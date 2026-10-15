"""Second correction, after damping the unstable symmetric modes.

At a large step size the symmetric part can be nonpositive and the iteration
still diverges, because the imaginary part makes the explicit step expand.
This probe asks which extra change stops those systems:

  * shrink the rotational piece of the update, keeping as much of it as the
    step size allows
  * take the damped field implicitly

It also asks whether removing several symmetric modes, not just the top one,
stops the nonlinear Baird value function at the learning rate where one mode
was not enough.

    python probes/push_again.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.baird import GAMMA, feature_matrix  # noqa: E402
from lib.nets import FuncMLP  # noqa: E402
from probes.push_further import (  # noqa: E402
    PlugInTest,
    damp_positive_modes,
    discrete_diverges,
    family_aliased,
    family_baird,
    family_cycle,
    family_random,
    family_self_loop,
    step_radius,
)


def split_parts(operator: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    symmetric = 0.5 * (operator + operator.T)
    skew = 0.5 * (operator - operator.T)
    return symmetric, skew


def best_rotation_scale(operator: np.ndarray, alpha: float) -> float | None:
    """Largest scale in [0, 1] on the skew part that keeps the explicit step stable.

    The symmetric part is left as given. None means even deleting the skew part
    leaves the explicit step unstable.
    """
    symmetric, skew = split_parts(operator)
    best = None
    for scale in np.linspace(0.0, 1.0, 21):
        if not discrete_diverges(symmetric + scale * skew, alpha):
            best = float(scale)
    return best


def implicit_diverges(operator: np.ndarray, alpha: float) -> bool:
    """Spectral radius of (I - alpha K)^{-1}, the implicit Euler step."""
    eigenvalues = np.linalg.eigvals(operator)
    denom = 1.0 - alpha * eigenvalues
    if np.any(np.abs(denom) < 1e-10):
        return True
    return float(np.max(np.abs(1.0 / denom))) > 1.0 + 1e-8


def two_stage(alpha: float, rng: np.random.Generator) -> None:
    print(f"\n== step size {alpha}: damp, then shrink rotation, then implicit ==")
    print(f"{'family':12} {'div':>5} {'damp':>6} {'shrink':>7} {'med_β':>7} {'impl':>6} {'β=0 still out':>14}")
    families = {
        "baird": family_baird(rng, 120),
        "random": family_random(rng, 250),
        "cycle": family_cycle(rng, 120),
        "self-loop": family_self_loop(rng, 80),
        "aliased": family_aliased(rng, 120),
    }
    for name, keys in families.items():
        diverging = [key for key in keys if discrete_diverges(key, alpha)]
        if not diverging:
            print(f"{name:12} {0:5d}")
            continue
        damp_ok = 0
        shrink_ok = 0
        implicit_ok = 0
        beta_zero_fails = 0
        betas = []
        for key in diverging:
            damped = damp_positive_modes(key)
            damp_ok += int(not discrete_diverges(damped, alpha))
            beta = best_rotation_scale(damped, alpha)
            if beta is None:
                beta_zero_fails += 1
            else:
                shrink_ok += 1
                betas.append(beta)
            implicit_ok += int(not implicit_diverges(damped, alpha))
        median_beta = float(np.median(betas)) if betas else float("nan")
        print(
            f"{name:12} {len(diverging):5d} {damp_ok:6d} {shrink_ok:7d} {median_beta:7.2f} "
            f"{implicit_ok:6d} {beta_zero_fails:14d}"
        )


def apply_symmetric(net: FuncMLP, theta: torch.Tensor, vector: torch.Tensor, phi: torch.Tensor, gamma: float) -> torch.Tensor:
    """Symmetric part of the Jacobian of the semi-gradient, times `vector`."""
    theta_leaf = theta.detach().requires_grad_(True)
    omega = theta.detach().clone().requires_grad_(True)
    prediction = net.forward(theta_leaf, phi).squeeze(-1)
    bootstrap = net.forward(omega, phi[6][None]).squeeze(-1)
    backup = gamma * bootstrap.expand_as(prediction)
    loss = (prediction - backup).pow(2).mean()
    (grad,) = torch.autograd.grad(loss, theta_leaf, create_graph=True)
    (_, through_target) = torch.autograd.functional.jvp(
        lambda om: _grad_online(net, theta, om, phi, gamma),
        theta.detach(),
        vector,
    )
    (through_target_T,) = torch.autograd.grad((grad * vector).sum(), omega, retain_graph=True)
    (through_online,) = torch.autograd.grad((grad * vector).sum(), theta_leaf)
    return through_online.detach() + 0.5 * (through_target.detach() + through_target_T.detach())


def _grad_online(net: FuncMLP, theta: torch.Tensor, omega: torch.Tensor, phi: torch.Tensor, gamma: float) -> torch.Tensor:
    theta_fixed = theta.detach().requires_grad_(True)
    prediction = net.forward(theta_fixed, phi).squeeze(-1)
    bootstrap = net.forward(omega, phi[6][None]).squeeze(-1)
    loss = (prediction - gamma * bootstrap.expand_as(prediction)).pow(2).mean()
    (grad,) = torch.autograd.grad(loss, theta_fixed, create_graph=True)
    return grad


def top_symmetric_modes(net: FuncMLP, theta: torch.Tensor, phi: torch.Tensor, gamma: float, count: int) -> list[torch.Tensor]:
    """Top eigenvectors of the symmetric part, by deflated power iteration."""
    modes = []
    for _ in range(count):
        vector = torch.randn(theta.numel())
        vector = vector / vector.norm()
        for _round in range(6):
            updated = apply_symmetric(net, theta, vector, phi, gamma)
            for earlier in modes:
                updated = updated - torch.dot(earlier, updated) * earlier
            norm = updated.norm()
            if float(norm) < 1e-8:
                break
            vector = (updated / norm).detach()
        modes.append(vector.detach())
    return modes


def semi_grad(net: FuncMLP, theta: torch.Tensor, phi: torch.Tensor, gamma: float) -> torch.Tensor:
    leaf = theta.detach().requires_grad_(True)
    prediction = net.forward(leaf, phi).squeeze(-1)
    backup = gamma * prediction[6].detach()
    loss = (prediction - backup).pow(2).mean()
    (grad,) = torch.autograd.grad(loss, leaf)
    return grad.detach()


def nonlinear_several_modes() -> None:
    print("\n== nonlinear Baird at lr 0.05, projecting several symmetric modes ==")
    phi = torch.tensor(feature_matrix(), dtype=torch.float32)
    gamma = GAMMA
    for n_modes in (0, 1, 3):
        torch.manual_seed(0)
        net = FuncMLP([8, 32, 1], torch.device("cpu"), seed=0)
        theta = net.theta.detach().clone()
        modes = top_symmetric_modes(net, theta, phi, gamma, max(n_modes, 1)) if n_modes else []
        diverged = None
        for step in range(1, 301):
            grad = semi_grad(net, theta, phi, gamma)
            for mode in modes[:n_modes]:
                grad = grad - torch.dot(mode, grad) * mode
            theta = theta - 0.05 * grad
            if step % 50 == 0 and n_modes:
                modes = top_symmetric_modes(net, theta, phi, gamma, n_modes)
            if float(theta.norm()) > 1e3 or not torch.isfinite(theta).all():
                diverged = step
                break
        with torch.no_grad():
            values = net.forward(theta, phi).squeeze(-1)
        print(
            f"modes {n_modes}: diverged_at {diverged}  norm {float(theta.norm()):.2f}  "
            f"mean |V| {float(values.abs().mean()):.3f}"
        )


def fixed_n_power(diffs: np.ndarray, alpha: float, power: float) -> int | None:
    trials, cap = diffs.shape
    for n in range(4, cap + 1):
        hits = 0
        for trial in range(trials):
            sample = diffs[trial, :n]
            if float(np.std(sample)) < 1e-8:
                hits += int(abs(float(sample.mean())) > 1e-8)
            else:
                pvalue = float(stats.ttest_1samp(sample, 0.0).pvalue)
                hits += int(pvalue == pvalue and pvalue < alpha)
        if hits / trials >= power:
            return n
    return None


def plugin_n_power(diffs: np.ndarray, alpha: float, power: float) -> int | None:
    trials, cap = diffs.shape
    hits = np.full(trials, cap + 1)
    for trial in range(trials):
        unit = (diffs[trial] + 1.0) / 2.0
        test = PlugInTest(null=0.5, alpha=alpha)
        for n, value in enumerate(unit, start=1):
            test.update(float(value))
            if test.rejects():
                hits[trial] = n
                break
    for n in range(2, cap + 1):
        if float(np.mean(hits <= n)) >= power:
            return n
    return None


def plugin_fpr(rng: np.random.Generator, total: float, alpha: float, paths: int, steps: int) -> float:
    misses = 0
    for _ in range(paths):
        diffs = rng.beta(total / 2, total / 2, size=steps) - rng.beta(total / 2, total / 2, size=steps)
        unit = np.clip((diffs + 1.0) / 2.0, 0.0, 1.0)
        test = PlugInTest(null=0.5, alpha=alpha)
        for value in unit:
            test.update(float(value))
            if test.rejects():
                misses += 1
                break
    return misses / paths


def variance_curve() -> None:
    """Honest sample size: one-look t-test versus the plug-in stopping time.

    `total` is the beta concentration. Larger total means a tighter difference
    around the same mean gap, so the gap is a smaller fraction of the [-1, 1] bound.
    """
    print("\n== fixed t-test n versus plug-in stopping time, gap 0.14 ==")
    rng = np.random.default_rng(2)
    alpha = 0.05
    trials = 60
    cap = 200
    gap = 0.14
    print(f"{'concentration':>14} {'t_fixed':>8} {'plugin':>8} {'ratio':>7} {'null_fpr':>9}")
    for total in (8.0, 24.0, 80.0):
        left_a = 0.5 * (1.0 + gap) * total
        left_b = total - left_a
        diffs = rng.beta(left_a, left_b, size=(trials, cap)) - rng.beta(left_b, left_a, size=(trials, cap))
        diffs = np.clip(diffs, -1.0, 1.0)
        t_n = fixed_n_power(diffs, alpha, 0.8)
        plug_n = plugin_n_power(diffs, alpha, 0.8)
        fpr = plugin_fpr(rng, total, alpha, paths=200, steps=cap)
        ratio = "na" if t_n is None or plug_n is None else f"{plug_n / t_n:.2f}"
        print(f"{total:14.0f} {str(t_n):>8} {str(plug_n):>8} {ratio:>7} {fpr:9.3f}")


def main() -> None:
    rng = np.random.default_rng(0)
    for alpha in (0.05, 0.5):
        two_stage(alpha, rng)
    nonlinear_several_modes()
    variance_curve()


if __name__ == "__main__":
    main()
