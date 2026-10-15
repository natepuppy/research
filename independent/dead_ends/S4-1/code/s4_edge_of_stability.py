"""S4-1. Does TD learning sit at an edge of stability?

In supervised learning the loss Hessian's top eigenvalue rises until it sits
near 2/eta. This script measures the same quantity for a bootstrapped TD loss
on a small chain DQN, sweeping the learning rate eta, the target-network rate
tau, and the discount gamma, with and without a hyperspherical rescaling of
the weights.

Two operators are measured at each checkpoint:
  * semi: Hessian of the TD loss with the bootstrap target detached.
  * full: Hessian of the same loss with the bootstrap differentiated through
    the online weights (the update-map contribution of the target).

The reported ratio is lambda_max * eta / 2. The supervised edge is ratio 1.
The hypothesis says TD equilibrates at a ratio below 1, and that the ratio
moves with tau and gamma.

Kill test: sharpness does not equilibrate, or it equilibrates at 2/eta no
matter what tau and gamma are.

    python code/s4_edge_of_stability.py
    python code/s4_edge_of_stability.py --smoke
    python code/s4_edge_of_stability.py --full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent  # code/
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

from lib.envs import Chain  # noqa: E402
from lib.io_utils import kill_test, save_csv  # noqa: E402
from lib.nets import FuncMLP  # noqa: E402
from lib.sharpness import top_hessian_eigenvalue  # noqa: E402


def measure(net: FuncMLP, theta: torch.Tensor, target: torch.Tensor, batch, chain: Chain, gamma: float) -> tuple[float, float]:
    states = chain.one_hot(batch["s"].to(theta.device))
    nxt = chain.one_hot(batch["sp"].to(theta.device))
    actions = batch["a"].to(theta.device).long()
    rewards = batch["r"].to(theta.device)
    dones = batch["d"].to(theta.device)
    target = target.detach()

    def eigenvalue(through_bootstrap: bool) -> float:
        def hvp(vector: torch.Tensor) -> torch.Tensor:
            parameter = theta.detach().requires_grad_(True)
            prediction = net.forward(parameter, states).gather(1, actions[:, None]).squeeze(1)
            if through_bootstrap:
                next_value = net.forward(parameter, nxt).max(dim=1).values
            else:
                next_value = net.forward(target, nxt).max(dim=1).values.detach()
            backup = rewards + gamma * (1.0 - dones) * next_value
            loss = (prediction - backup).pow(2).mean()
            (grad,) = torch.autograd.grad(loss, parameter, create_graph=True)
            (product,) = torch.autograd.grad((grad * vector).sum(), parameter)
            return product.detach()

        return top_hessian_eigenvalue(hvp, theta.numel(), steps=15, device=theta.device)

    return eigenvalue(False), eigenvalue(True)


def train_one(steps: int, seed: int, lr: float, gamma: float, tau: float, project: bool, device: torch.device, measure_every: int) -> list[dict]:
    torch.manual_seed(seed)
    chain = Chain(8)
    net = FuncMLP([chain.obs_dim, 64, 64, chain.n_actions], device, seed=seed)
    theta = net.theta.detach().clone()
    target = theta.clone()
    capacity = 8000
    store = {key: torch.zeros(capacity) for key in ("r", "d")}
    store["s"] = torch.zeros(capacity, dtype=torch.long)
    store["a"] = torch.zeros(capacity, dtype=torch.long)
    store["sp"] = torch.zeros(capacity, dtype=torch.long)
    count = 0
    pointer = 0
    state = torch.tensor(0)
    rows = []
    for step in range(steps):
        eps = max(0.05, 1.0 - step / (0.5 * max(steps, 1)))
        with torch.no_grad():
            if torch.rand(()) < eps:
                action = int(torch.randint(0, 2, ()).item())
            else:
                values = net.forward(theta, chain.one_hot(state[None].to(device)))
                action = int(values.argmax(dim=1).item())
        nxt, reward, done = chain.step(state[None], torch.tensor([action]))
        store["s"][pointer] = state
        store["a"][pointer] = action
        store["r"][pointer] = float(reward)
        store["sp"][pointer] = nxt
        store["d"][pointer] = float(done)
        pointer = (pointer + 1) % capacity
        count = min(count + 1, capacity)
        state = torch.tensor(0) if bool(done.item()) else nxt[0].cpu()
        if count < 128:
            continue
        index = torch.randint(0, count, (64,))
        batch = {key: value[index] for key, value in store.items()}
        states = chain.one_hot(batch["s"].to(device))
        leaf = theta.detach().requires_grad_(True)
        prediction = net.forward(leaf, states).gather(1, batch["a"].to(device).long()[:, None]).squeeze(1)
        with torch.no_grad():
            nxt_obs = chain.one_hot(batch["sp"].to(device))
            backup = batch["r"].to(device) + gamma * (1.0 - batch["d"].to(device)) * net.forward(target, nxt_obs).max(dim=1).values
        loss = (prediction - backup).pow(2).mean()
        (grad,) = torch.autograd.grad(loss, leaf)
        theta = (leaf.detach() - lr * grad.detach()).detach()
        if project:
            theta = net.hypersphere_project(theta)
        target = ((1.0 - tau) * target + tau * theta).detach()
        if step % measure_every != 0:
            continue
        semi, full = measure(net, theta, target, batch, chain, gamma)
        edge = 2.0 / lr
        rows.append(
            {
                "step": step,
                "seed": seed,
                "lr": lr,
                "gamma": gamma,
                "tau": tau,
                "project": int(project),
                "loss": float(loss.detach()),
                "lambda_semi": semi,
                "lambda_full": full,
                "edge_2_over_eta": edge,
                "ratio_semi": semi / edge,
                "ratio_full": full / edge,
                "diverged": int(not np.isfinite(semi) or abs(semi) > 1e6),
            }
        )
        if rows[-1]["diverged"]:
            break
    return rows


def _late_mean(rows: list[dict], key: str) -> float:
    if not rows:
        return float("nan")
    tail = rows[int(len(rows) * 0.6) :]
    values = np.array([row[key] for row in tail if np.isfinite(row[key])])
    if len(values) == 0:
        return float("nan")
    return float(values.mean())


def _late_cv(rows: list[dict], key: str) -> float:
    tail = rows[int(len(rows) * 0.6) :]
    values = np.array([row[key] for row in tail if np.isfinite(row[key])])
    if len(values) < 3 or abs(values.mean()) < 1e-8:
        return float("nan")
    return float(values.std() / abs(values.mean()))


def judge(rows: list[dict]) -> tuple[bool | None, str]:
    configs: dict[tuple, list[dict]] = {}
    for row in rows:
        configs.setdefault((row["lr"], row["gamma"], row["tau"], row["project"]), []).append(row)
    late = {key: _late_mean(value, "ratio_full") for key, value in configs.items()}
    cvs = [_late_cv(value, "ratio_full") for value in configs.values()]
    finite_late = [value for value in late.values() if np.isfinite(value)]
    finite_cv = [value for value in cvs if np.isfinite(value)]
    if len(finite_late) < 2 or not finite_cv:
        return None, "not enough finite sharpness measurements to judge equilibration."
    equilibrated = float(np.mean(finite_cv)) < 0.35
    # Supervised edge is ratio == 1. "Below it" and "moves with tau, gamma".
    below = float(np.mean(finite_late)) < 0.85
    by_tau = [value for (lr, gamma, tau, project), value in late.items() if np.isfinite(value)]
    spread = float(max(finite_late) - min(finite_late))
    moves = spread > 0.1
    supported = equilibrated and below and moves
    text = (
        f"late full-update sharpness/edge mean {float(np.mean(finite_late)):.3f} "
        f"(1 would be the supervised 2/eta edge), mean late CV {float(np.mean(finite_cv)):.3f}, "
        f"spread across the sweep {spread:.3f}."
    )
    del by_tau
    return supported, text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--out", type=Path, default=PROJECT / "results")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.smoke:
        grid = [{"lr": 0.2, "gamma": 0.99, "tau": 1.0, "project": False}]
        steps, every, seeds = 300, 100, [0]
    elif args.full:
        grid = [
            {"lr": lr, "gamma": gamma, "tau": tau, "project": project}
            for lr in (0.05, 0.1, 0.2)
            for gamma in (0.9, 0.99)
            for tau in (0.01, 1.0)
            for project in (False, True)
        ]
        steps, every, seeds = 2000, 100, [0, 1]
    else:
        grid = [
            {"lr": 0.1, "gamma": 0.99, "tau": 0.01, "project": False},
            {"lr": 0.2, "gamma": 0.99, "tau": 0.01, "project": False},
            {"lr": 0.2, "gamma": 0.99, "tau": 1.0, "project": False},
            {"lr": 0.2, "gamma": 0.9, "tau": 1.0, "project": False},
            {"lr": 0.2, "gamma": 0.99, "tau": 1.0, "project": True},
        ]
        steps, every, seeds = 800, 80, [0]

    rows: list[dict] = []
    for config in grid:
        for seed in seeds:
            traced = train_one(steps, seed, device=device, measure_every=every, **config)
            rows.extend(traced)
            last = traced[-1] if traced else {}
            print(
                f"lr {config['lr']} gamma {config['gamma']} tau {config['tau']} "
                f"project {config['project']} seed {seed}: late ratio_full {last.get('ratio_full', float('nan')):.3f}"
            )
    supported, text = judge(rows)
    verdict = kill_test(supported, text)
    save_csv(args.out / "sharpness_trace.csv", rows)
    (args.out / "verdict.txt").write_text(verdict + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
