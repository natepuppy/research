"""Checks that do not require a training run.

Confirms the curl estimator, the Lanczos eigenvalue, Baird divergence, and
that the betting confidence sequence covers a known mean often enough.

    python code/self_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent  # code/
sys.path.insert(0, str(HERE))

from lib.anytime import BettingCS  # noqa: E402
from lib.baird import initial_weights, sample_update  # noqa: E402
from lib.curl import curl_of_field, td_curl  # noqa: E402
from lib.nets import FuncMLP  # noqa: E402
from lib.sharpness import top_hessian_eigenvalue  # noqa: E402


def check_curl() -> None:
    symmetric = torch.tensor([[2.0, 0.4], [0.4, 1.0]])
    rotation = torch.tensor([[0.0, 1.0], [-1.0, 0.0]])

    def symmetric_field(theta: torch.Tensor) -> torch.Tensor:
        return symmetric @ theta

    def rotation_field(theta: torch.Tensor) -> torch.Tensor:
        return rotation @ theta

    theta = torch.tensor([0.3, -0.7])
    sym = curl_of_field(symmetric_field, theta, probes=12)
    rot = curl_of_field(rotation_field, theta, probes=12)
    print(f"curl fraction symmetric field {sym['curl_fraction']:.4f} (want ~0)")
    print(f"curl fraction rotation field  {rot['curl_fraction']:.4f} (want ~1)")
    if sym["curl_fraction"] > 0.05 or rot["curl_fraction"] < 0.85:
        raise SystemExit("curl estimator failed the linear-field check")


def check_curl_through_autograd() -> None:
    """The TD field is itself a gradient. That path must still report ~0 curl."""
    matrix = torch.tensor([[2.0, 0.3], [0.3, 1.0]])

    def field(theta: torch.Tensor) -> torch.Tensor:
        loss = 0.5 * theta @ matrix @ theta
        (grad,) = torch.autograd.grad(loss, theta, create_graph=True)
        return grad

    stats = curl_of_field(field, torch.tensor([0.2, -0.4]), probes=8)
    print(f"curl fraction of an explicit gradient {stats['curl_fraction']:.4f} (want ~0)")
    if stats["curl_fraction"] > 0.05:
        raise SystemExit("curl estimator failed when the field is an autograd gradient")


def check_td_semigradient_curl() -> None:
    """Online bootstrap must have curl; a frozen target must not.

    Detaching the backup and then taking a jvp only sees a Hessian, so this
    fails if the estimator forgets the mixed derivative d(grad)/d(target).
    """
    torch.manual_seed(0)
    net = FuncMLP([4, 16, 2], torch.device("cpu"), seed=0)
    theta = net.theta.detach()
    states = torch.randn(16, 4)
    nxt = torch.randn(16, 4)
    actions = torch.randint(0, 2, (16,))
    rewards = torch.rand(16)
    dones = torch.zeros(16)
    online = td_curl(
        net, theta, states, nxt, rewards, dones, 0.99, actions=actions, probes=4, online=True
    )
    frozen = td_curl(
        net,
        theta,
        states,
        nxt,
        rewards,
        dones,
        0.99,
        actions=actions,
        probes=4,
        online=False,
        target=theta,
    )
    print(
        f"TD curl online {online['curl_fraction']:.4f} (want > 0.02), "
        f"frozen target {frozen['curl_fraction']:.4f} (want ~0)"
    )
    if online["curl_fraction"] < 0.02 or frozen["curl_fraction"] > 0.05:
        raise SystemExit("semi-gradient curl estimator is not seeing the bootstrap Jacobian")


def check_lanczos() -> None:
    diagonal = torch.tensor([1.0, -2.0, 5.0, 0.25])

    def hvp(vector: torch.Tensor) -> torch.Tensor:
        return diagonal * vector

    top = top_hessian_eigenvalue(hvp, diagonal.numel(), steps=4)
    print(f"Lanczos top eigenvalue {top:.4f} (want 5)")
    if abs(top - 5.0) > 1e-3:
        raise SystemExit("Lanczos failed to recover the top eigenvalue")


def check_baird() -> None:
    start = float(np.linalg.norm(initial_weights()))
    finals = []
    for seed in range(4):
        rng = np.random.default_rng(seed)
        weights = initial_weights()
        for _ in range(400):
            weights = sample_update(weights, rng, alpha=0.05)
        finals.append(float(np.linalg.norm(weights)))
    print(f"Baird weight norm start {start:.2f}, after 400 steps {['%.1f' % v for v in finals]}")
    if max(finals) < start * 2:
        raise SystemExit("Baird weights did not grow; the divergence fixture is wrong")


def check_coverage() -> None:
    rng = np.random.default_rng(1)
    trials = 40
    covered = 0
    for trial in range(trials):
        sequence = BettingCS(alpha=0.1, n_grid=101, n_lambda=11)
        ok = True
        for _ in range(25):
            observation = float(rng.random() < 0.3)
            lower, upper = sequence.update(observation)
            if not lower - 1e-9 <= 0.3 <= upper + 1e-9:
                ok = False
                break
        covered += int(ok)
        del trial
    rate = covered / trials
    print(f"betting CS path coverage {covered}/{trials} = {rate:.2f} at alpha 0.1 (want at least 0.80)")
    if rate < 0.80:
        raise SystemExit("confidence sequence undercoveres; do not trust s4_anytime until this passes")


def check_capital_positive() -> None:
    """Every safe lambda must keep 1 + lambda (x - m) > 0 for x in {0, 1}."""
    sequence = BettingCS()
    for mean in sequence.grid:
        scale = max(mean, 1.0 - mean, 1e-3)
        for fraction in np.linspace(-0.85, 0.85, sequence.n_lambda):
            if abs(fraction) < 1e-8:
                continue
            lam = fraction / scale
            for value in (0.0, 1.0):
                if 1.0 + lam * (value - mean) <= 1e-8:
                    raise SystemExit(f"unsafe lambda {lam} at mean {mean} value {value}")
    print("betting lambdas stay positive on the boundary")


if __name__ == "__main__":
    check_curl()
    check_curl_through_autograd()
    check_td_semigradient_curl()
    check_lanczos()
    check_baird()
    check_capital_positive()
    check_coverage()
    print("self_check passed")
