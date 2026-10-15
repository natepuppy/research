"""Tweaks aimed at a stronger paper than the first full grids.

1. Random off-policy linear MDPs: does damping the unstable symmetric mode
   stabilize systems that diverge, and does stripping the skew part?
2. The same damping on the nonlinear Baird value function that diverged.
3. Betting sequence versus a t-test when the effect size is not known in
   advance, which is the comparison a paper can actually recommend.

    python probes/tweak_followups.py
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
from lib.baird import GAMMA, expected_semi_gradient, feature_matrix  # noqa: E402
from lib.nets import FuncMLP  # noqa: E402


def softmax(logits: np.ndarray, axis: int) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=axis, keepdims=True)


def offpolicy_key(phi: np.ndarray, transitions: np.ndarray, target: np.ndarray, behavior: np.ndarray, gamma: float) -> np.ndarray:
    """Expected semi-gradient matrix. transitions[s, a, s'], policies[s, a]."""
    n_s, n_a = behavior.shape
    behavior_kernel = np.einsum("sa,sap->sp", behavior, transitions)
    eigvals, eigvecs = np.linalg.eig(behavior_kernel.T)
    index = int(np.argmin(np.abs(eigvals - 1.0)))
    stationary = np.real(eigvecs[:, index])
    stationary = np.abs(stationary)
    stationary = stationary / stationary.sum()
    dimension = phi.shape[1]
    key = np.zeros((dimension, dimension))
    for state in range(n_s):
        for action in range(n_a):
            weight = stationary[state] * target[state, action]
            next_phi = transitions[state, action] @ phi
            key += weight * np.outer(phi[state], gamma * next_phi - phi[state])
    return key


def max_real(operator: np.ndarray) -> float:
    return float(np.real(np.linalg.eigvals(operator)).max())


def diverges(operator: np.ndarray, alpha: float = 0.05, steps: int = 400, start_norm: float = 1.0) -> bool:
    weights = np.ones(operator.shape[0])
    weights = weights / np.linalg.norm(weights) * start_norm
    for _ in range(steps):
        weights = weights + alpha * (operator @ weights)
        norm = float(np.linalg.norm(weights))
        if norm > 50.0 or not np.isfinite(norm):
            return True
    return float(np.linalg.norm(weights)) > 5.0 * start_norm


def baird_transitions(n_states: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solid action goes to the last state. Dashed action jumps uniformly among the others.

    Target policy is always solid. Behavior takes solid with probability 1/n_states.
    This is Baird's transition structure at n_states=7, and the same structure
    at other sizes.
    """
    transitions = np.zeros((n_states, 2, n_states))
    transitions[:, 0, -1] = 1.0
    if n_states > 1:
        transitions[:, 1, :-1] = 1.0 / (n_states - 1)
    target = np.zeros((n_states, 2))
    target[:, 0] = 1.0
    behavior = np.zeros((n_states, 2))
    behavior[:, 0] = 1.0 / n_states
    behavior[:, 1] = 1.0 - 1.0 / n_states
    return transitions, target, behavior


def random_systems(n: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    diverging = 0
    damp_fixes = 0
    skew_fixes = 0
    both = 0
    damp_breaks_stable = 0
    stable = 0
    for _ in range(n):
        n_states = int(rng.choice([4, 5, 7]))
        dimension = n_states + 1
        phi = rng.normal(size=(n_states, dimension))
        transitions, target, behavior = baird_transitions(n_states)
        key = offpolicy_key(phi, transitions, target, behavior, gamma=0.99)
        symmetric = 0.5 * (key + key.T)
        evals, evecs = np.linalg.eigh(symmetric)
        mode = evecs[:, int(np.argmax(evals))]
        damped = key - 2.0 * np.outer(mode, mode)
        plain_bad = diverges(key)
        if plain_bad:
            diverging += 1
            damp_ok = not diverges(damped)
            skew_ok = not diverges(symmetric)
            damp_fixes += int(damp_ok)
            skew_fixes += int(skew_ok)
            both += int(damp_ok and skew_ok)
        else:
            stable += 1
            damp_breaks_stable += int(diverges(damped))
    print("\n== random off-policy linear MDPs ==")
    print(f"systems {n}  diverging {diverging}  already stable {stable}")
    if diverging:
        print(
            f"of the diverging ones: damping stabilizes {damp_fixes / diverging:.2f}, "
            f"dropping the skew part stabilizes {skew_fixes / diverging:.2f}, "
            f"both {both / diverging:.2f}"
        )
    if stable:
        print(f"damping turns a stable system into a diverging one in {damp_breaks_stable / stable:.2f} of cases")


def nonlinear_baird() -> None:
    """Project the semi-gradient off the top symmetric mode. Rewards are 0, so V* is 0."""
    print("\n== nonlinear Baird, gradient projected off the unstable symmetric mode ==")
    phi = torch.tensor(feature_matrix(), dtype=torch.float32)
    gamma = GAMMA

    def semi_grad(net: FuncMLP, theta: torch.Tensor) -> torch.Tensor:
        leaf = theta.detach().requires_grad_(True)
        prediction = net.forward(leaf, phi).squeeze(-1)
        backup = gamma * prediction[6].detach()
        loss = (prediction - backup).pow(2).mean()
        (grad,) = torch.autograd.grad(loss, leaf)
        return grad.detach()

    def symmetric_mode(net: FuncMLP, theta: torch.Tensor, rounds: int = 4) -> torch.Tensor:
        mode = torch.randn(theta.numel())
        mode = mode / mode.norm()
        for _ in range(rounds):
            # Jacobian-vector product of the semi-gradient field, plus its transpose.
            theta_leaf = theta.detach().requires_grad_(True)
            omega = theta.detach().clone().requires_grad_(True)
            prediction = net.forward(theta_leaf, phi).squeeze(-1)
            bootstrap = net.forward(omega, phi[6][None]).squeeze(-1)
            # Broadcast the lower-state value onto every state.
            backup = gamma * bootstrap.expand_as(prediction)
            loss = (prediction - backup).pow(2).mean()
            (grad,) = torch.autograd.grad(loss, theta_leaf, create_graph=True)
            (_, jvp) = torch.autograd.functional.jvp(
                lambda om: _grad_wrt_online(net, theta, om, phi, gamma),
                theta.detach(),
                mode,
            )
            (jtv,) = torch.autograd.grad((grad * mode).sum(), omega, retain_graph=True)
            mixed = 0.5 * (jvp.detach() + jtv.detach())
            # The theta-theta block is symmetric; add its action via another backward.
            (hess,) = torch.autograd.grad((grad * mode).sum(), theta_leaf)
            updated = hess.detach() + mixed
            norm = updated.norm()
            if float(norm) < 1e-8:
                break
            mode = updated / norm
        return mode.detach()

    for lr, project in ((0.01, False), (0.01, True), (0.05, False), (0.05, True)):
        torch.manual_seed(0)
        net = FuncMLP([8, 32, 1], torch.device("cpu"), seed=0)
        theta = net.theta.detach().clone()
        mode = symmetric_mode(net, theta)
        diverged = None
        for step in range(1, 401):
            grad = semi_grad(net, theta)
            if project:
                grad = grad - torch.dot(mode, grad) * mode
            theta = theta - lr * grad
            if step % 40 == 0:
                mode = symmetric_mode(net, theta)
            if float(theta.norm()) > 1e3 or not torch.isfinite(theta).all():
                diverged = step
                break
        with torch.no_grad():
            values = net.forward(theta, phi).squeeze(-1)
        print(
            f"lr {lr:.2f} project {project}: diverged_at {diverged} "
            f"norm {float(theta.norm()):.2f} mean |V| {float(values.abs().mean()):.3f}"
        )


def _grad_wrt_online(net: FuncMLP, theta: torch.Tensor, omega: torch.Tensor, phi: torch.Tensor, gamma: float) -> torch.Tensor:
    theta_fixed = theta.detach().requires_grad_(True)
    prediction = net.forward(theta_fixed, phi).squeeze(-1)
    bootstrap = net.forward(omega, phi[6][None]).squeeze(-1)
    loss = (prediction - gamma * bootstrap.expand_as(prediction)).pow(2).mean()
    (grad,) = torch.autograd.grad(loss, theta_fixed, create_graph=True)
    return grad


def first_rejection_curve() -> dict[float, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(1)
    alpha = 0.05
    trials = 40
    cap = 80
    null = null_mean_on_unit_interval(-1.0, 1.0)
    gaps = [0.0, 0.08, 0.14, 0.20, 0.25]
    curves = {}
    for gap in gaps:
        # Beta(a, b) vs Beta(b, a) with (a-b)/(a+b) = gap, a+b = 24.
        total = 24.0
        left_a = 0.5 * (1.0 + gap) * total
        left_b = total - left_a
        diffs = rng.beta(left_a, left_b, size=(trials, cap)) - rng.beta(left_b, left_a, size=(trials, cap))
        t_hits = np.full(trials, cap + 1)
        cs_hits = np.full(trials, cap + 1)
        for trial in range(trials):
            series = diffs[trial]
            for n in range(2, cap + 1):
                sample = series[:n]
                if float(np.std(sample)) < 1e-8:
                    rejected = abs(float(sample.mean())) > 1e-8
                else:
                    pvalue = float(stats.ttest_1samp(sample, 0.0).pvalue)
                    rejected = pvalue == pvalue and pvalue < alpha
                if rejected:
                    t_hits[trial] = n
                    break
            sequence = BettingCS(alpha=alpha, n_grid=61, n_lambda=7)
            unit = rescale_to_unit(series, -1.0, 1.0)
            for n, value in enumerate(unit, start=1):
                sequence.update(float(value))
                if sequence.excludes(null):
                    cs_hits[trial] = n
                    break
        curves[gap] = (t_hits, cs_hits)
    return curves


def n_for_power(hits: np.ndarray, power: float, cap: int) -> int | None:
    for n in range(2, cap + 1):
        if float(np.mean(hits <= n)) >= power:
            return n
    return None


def unknown_effect() -> None:
    """Fix the budget at the t-test's n for an 0.08 gap, then face larger gaps and a null.

    A lab that sizes the study for the smallest effect it cares about spends that
    many seeds every time. The sequence can stop early when the effect is bigger,
    and must not reject the null.
    """
    print("\n== betting sequence when the study is sized for a gap of 0.08 ==")
    curves = first_rejection_curve()
    cap = 80
    design = n_for_power(curves[0.08][0], 0.8, cap)
    print(f"t-test seeds for 80% power at gap 0.08: {design}")
    if design is None:
        print("t-test never reached 80% power inside the cap")
        return
    print(f"{'gap':>6} {'t_power':>8} {'cs_power':>8} {'t_seeds':>8} {'cs_seeds':>8}")
    for gap, (t_hits, cs_hits) in curves.items():
        t_power = float(np.mean(t_hits <= design))
        cs_power = float(np.mean(cs_hits <= design))
        # Seeds used: stop at rejection, otherwise spend the whole budget.
        t_seeds = float(np.mean(np.minimum(t_hits, design)))
        cs_seeds = float(np.mean(np.minimum(cs_hits, design)))
        print(f"{gap:6.2f} {t_power:8.2f} {cs_power:8.2f} {t_seeds:8.1f} {cs_seeds:8.1f}")


def main() -> None:
    random_systems(150)
    nonlinear_baird()
    unknown_effect()


if __name__ == "__main__":
    main()
