"""N1-1. The TD update is not a gradient. Measure its curl.

Hypothesis: the rotational fraction of the semi-gradient TD vector field is
higher on runs that diverge than on runs that stay stable, and it separates
them better than TD error or |Q|.

The contrast that can actually diverge is Baird's counterexample. Classical
semi-gradient TD diverges there; the same update with a frozen target is
gradient descent on a quadratic and stays bounded. A symplectic correction
that drops the skew part of the Jacobian is also run: on Baird that skew part
is not what makes the iteration unstable, so the correction is not assumed to
help. Predictors are read at the first checkpoint, before the weights have
blown up, and scored against whether the run eventually diverges.

A deep Q-network on an 8-state chain, and a TD critic on the pendulum, are
logged with the same curl estimator. On-policy TD on that chain does not
diverge, so those traces are not the kill test.

Kill test: at the first checkpoint, curl fraction does not separate the runs
that later diverge from the runs that stay bounded any better than TD error
or |Q| do.

    python code/n1_td_curl.py            # default, a few minutes
    python code/n1_td_curl.py --smoke    # plumbing check
    python code/n1_td_curl.py --full
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

from lib.baird import expected_semi_gradient, initial_weights, sample_update  # noqa: E402
from lib.curl import td_curl, td_rotational_component  # noqa: E402
from lib.envs import Chain, Pendulum  # noqa: E402
from lib.io_utils import kill_test, save_csv  # noqa: E402
from lib.nets import FuncMLP  # noqa: E402


def _point_biserial(feature: np.ndarray, label: np.ndarray) -> float:
    if label.min() == label.max() or feature.std() < 1e-12:
        return float("nan")
    label = label.astype(np.float64)
    return float(np.corrcoef(feature, label)[0, 1])


def _exact_curl(field, theta: torch.Tensor) -> dict[str, float]:
    """Curl fraction from the full Jacobian. Baird's field is 8-dimensional."""
    columns = []
    for index in range(theta.numel()):
        direction = torch.zeros_like(theta)
        direction[index] = 1.0
        _, tangent = torch.autograd.functional.jvp(field, theta, direction)
        columns.append(tangent)
    jacobian = torch.stack(columns, dim=1)
    skew = 0.5 * (jacobian - jacobian.T)
    anti_sq = float((skew * skew).sum())
    jac_sq = float((jacobian * jacobian).sum())
    return {
        "curl_frob": anti_sq**0.5,
        "curl_fraction": anti_sq / (jac_sq + 1e-12),
        "jacobian_frob": jac_sq**0.5,
    }


def _frozen_baird_field(target: torch.Tensor):
    from lib.baird import feature_matrix

    phi = torch.tensor(feature_matrix(), dtype=target.dtype)
    lower = phi[6]

    def field(weights: torch.Tensor) -> torch.Tensor:
        acc = torch.zeros_like(weights)
        bootstrap = torch.dot(lower, target)
        for state in range(7):
            acc = acc + (0.99 * bootstrap - torch.dot(phi[state], weights)) * phi[state]
        return acc / 7.0

    return field


def _baird_jacobian() -> torch.Tensor:
    """Jacobian of the semi-gradient field. It does not depend on the weights."""
    origin = torch.zeros(8, dtype=torch.float64)
    columns = []
    for index in range(8):
        direction = torch.zeros(8, dtype=torch.float64)
        direction[index] = 1.0
        _, tangent = torch.autograd.functional.jvp(expected_semi_gradient, origin, direction)
        columns.append(tangent)
    return torch.stack(columns, dim=1)


_BAIRD_JACOBIAN = None


def _baird_jacobian_cached() -> torch.Tensor:
    global _BAIRD_JACOBIAN
    if _BAIRD_JACOBIAN is None:
        _BAIRD_JACOBIAN = _baird_jacobian()
    return _BAIRD_JACOBIAN


def _symplectic_baird_field(weights: torch.Tensor) -> torch.Tensor:
    """(J + J^T) w / 2. The skew part of J is removed, and the map stays differentiable."""
    symmetric = 0.5 * (_baird_jacobian_cached() + _baird_jacobian_cached().T)
    return symmetric @ weights


def _baird_td_abs(weights: torch.Tensor, bootstrap: torch.Tensor) -> float:
    from lib.baird import feature_matrix

    phi = torch.tensor(feature_matrix(), dtype=weights.dtype)
    errors = [
        abs(float(0.99 * torch.dot(phi[6], bootstrap) - torch.dot(phi[state], weights)))
        for state in range(7)
    ]
    return float(np.mean(errors))


def _baird_q_abs(weights: torch.Tensor) -> float:
    from lib.baird import feature_matrix

    phi = torch.tensor(feature_matrix(), dtype=weights.dtype)
    values = phi @ weights
    return float(values.abs().mean())


def run_baird_intervention(
    steps: int,
    alpha: float,
    seed: int,
    measure_every: int,
    tau: float,
    correct: bool,
    adaptive_tau: bool,
    condition: str,
) -> list[dict]:
    """Stochastic Baird updates. `tau >= 1` is classical semi-gradient TD.

    `correct` subtracts the rotational part of the mean field. `adaptive_tau`
    freezes the target after the first checkpoint whose curl exceeds 0.2.
    """
    rng = np.random.default_rng(seed)
    weights = torch.tensor(initial_weights(), dtype=torch.float64)
    target = weights.clone()
    current_tau = tau
    rows = []
    diverged = False
    for step in range(steps + 1):
        if step % measure_every == 0:
            if correct:
                field = _symplectic_baird_field
                bootstrap = weights
            elif current_tau >= 1.0:
                field = expected_semi_gradient
                bootstrap = weights
            else:
                field = _frozen_baird_field(target.detach())
                bootstrap = target
            stats = _exact_curl(field, weights.detach())
            norm = float(weights.norm())
            diverged = diverged or (not np.isfinite(norm)) or norm > 40.0
            rows.append(
                {
                    "system": "baird",
                    "condition": condition,
                    "seed": seed,
                    "step": step,
                    "tau": current_tau,
                    "correct": int(correct),
                    "adaptive_tau": int(adaptive_tau),
                    "weight_norm": norm,
                    "td_abs": _baird_td_abs(weights, bootstrap),
                    "q_abs": _baird_q_abs(weights),
                    **stats,
                    "diverged": int(diverged),
                }
            )
            if adaptive_tau and stats["curl_fraction"] > 0.2:
                current_tau = 0.0
                target = weights.detach().clone()
            if diverged:
                break
        if step == steps:
            break
        numpy_w = weights.detach().cpu().numpy()
        if correct:
            updated = sample_update(numpy_w, rng, alpha=alpha)
            # sample_update adds alpha * (stochastic semi-gradient). Subtract
            # alpha * A w so the expected step is the symmetric part S w.
            skew = 0.5 * (_baird_jacobian_cached() - _baird_jacobian_cached().T)
            rotation = skew @ weights.detach()
            weights = torch.tensor(updated, dtype=torch.float64) - alpha * rotation
        elif current_tau >= 1.0:
            weights = torch.tensor(sample_update(numpy_w, rng, alpha=alpha), dtype=torch.float64)
        else:
            weights = _frozen_sample(weights, target, rng, alpha)
            target = ((1.0 - current_tau) * target + current_tau * weights).detach()
        weights = weights.detach()
    return rows


def _frozen_sample(weights: torch.Tensor, target: torch.Tensor, rng: np.random.Generator, alpha: float) -> torch.Tensor:
    """One behavior-policy step whose bootstrap uses `target` rather than the online weights."""
    from lib.baird import LOWER, SOLID_PROBABILITY, feature_matrix

    phi = feature_matrix()
    state = int(rng.integers(0, 7))
    solid = bool(rng.random() < SOLID_PROBABILITY)
    nxt = LOWER if solid else int(rng.integers(0, LOWER))
    rho = (1.0 / SOLID_PROBABILITY) if solid else 0.0
    online = weights.detach().cpu().numpy()
    frozen = target.detach().cpu().numpy()
    delta = 0.99 * float(phi[nxt] @ frozen) - float(phi[state] @ online)
    updated = online + alpha * rho * delta * phi[state]
    return torch.tensor(updated, dtype=torch.float64)


def _batch_tensors(chain: Chain, batch: dict, device: torch.device):
    states = chain.one_hot(batch["s"].to(device))
    nxt = chain.one_hot(batch["sp"].to(device))
    actions = batch["a"].to(device).long()
    rewards = batch["r"].to(device)
    dones = batch["d"].to(device)
    return states, actions, rewards, nxt, dones


def td_gradient(net: FuncMLP, theta: torch.Tensor, bootstrap_weights: torch.Tensor, chain: Chain, batch: dict, gamma: float) -> torch.Tensor:
    """Semi-gradient using whatever bootstrap weights the run is actually following."""
    states, actions, rewards, nxt, dones = _batch_tensors(chain, batch, net.device)
    leaf = theta.detach().requires_grad_(True)
    prediction = net.forward(leaf, states).gather(1, actions[:, None]).squeeze(1)
    next_value = net.forward(bootstrap_weights.detach(), nxt).max(dim=1).values
    backup = rewards + gamma * (1.0 - dones) * next_value
    loss = (prediction - backup).pow(2).mean()
    (grad,) = torch.autograd.grad(loss, leaf)
    return grad.detach()


def measure_chain_curl(net: FuncMLP, theta: torch.Tensor, target: torch.Tensor, chain: Chain, batch: dict, gamma: float, probes: int) -> dict[str, float]:
    """Curl of the semi-gradient field whose bootstrap tracks theta.

    A frozen target network makes this Jacobian a Hessian, so its curl is ~0
    even when the online field is rotational. Both numbers are logged.
    """
    states, actions, rewards, nxt, dones = _batch_tensors(chain, batch, net.device)
    online = td_curl(
        net, theta, states, nxt, rewards, dones, gamma, actions=actions, probes=probes, online=True
    )
    frozen = td_curl(
        net,
        theta,
        states,
        nxt,
        rewards,
        dones,
        gamma,
        actions=actions,
        probes=probes,
        online=False,
        target=target,
    )
    return {
        "curl_frob": online["curl_frob"],
        "curl_fraction": online["curl_fraction"],
        "jacobian_frob": online["jacobian_frob"],
        "curl_fraction_frozen_target": frozen["curl_fraction"],
    }


def train_chain(
    steps: int,
    seed: int,
    lr: float,
    gamma: float,
    tau: float,
    device: torch.device,
    measure_every: int,
    correct: bool,
    adaptive_tau: bool,
    hidden: int,
) -> list[dict]:
    torch.manual_seed(seed)
    chain = Chain(8)
    net = FuncMLP([chain.obs_dim, hidden, hidden, chain.n_actions], device, seed=seed)
    theta = net.theta.detach().clone().requires_grad_(False)
    target = theta.detach().clone()
    capacity = 8000
    store = {
        "s": torch.zeros(capacity, dtype=torch.long),
        "a": torch.zeros(capacity, dtype=torch.long),
        "r": torch.zeros(capacity),
        "sp": torch.zeros(capacity, dtype=torch.long),
        "d": torch.zeros(capacity),
    }
    count = 0
    pointer = 0
    state = torch.tensor(0)
    rows = []
    diverged = False
    current_tau = tau
    for step in range(steps):
        eps = max(0.05, 1.0 - step / (0.6 * steps))
        with torch.no_grad():
            if torch.rand(()) < eps:
                action = int(torch.randint(0, 2, ()).item())
            else:
                q_values = net.forward(theta, chain.one_hot(state[None].to(device)))
                action = int(q_values.argmax(dim=1).item())
        nxt, reward, done = chain.step(state[None], torch.tensor([action]))
        store["s"][pointer] = state
        store["a"][pointer] = action
        store["r"][pointer] = reward
        store["sp"][pointer] = nxt
        store["d"][pointer] = done.float()
        pointer = (pointer + 1) % capacity
        count = min(count + 1, capacity)
        state = torch.tensor(0) if bool(done.item()) else nxt[0].cpu()
        if count < 128:
            continue
        index = torch.randint(0, count, (64,))
        batch = {key: value[index] for key, value in store.items()}
        # tau = 1 copies the online weights into the target before the update,
        # which is classical semi-gradient TD. A smaller tau freezes the backup.
        bootstrap_weights = theta if current_tau >= 1.0 else target
        grad = td_gradient(net, theta, bootstrap_weights, chain, batch, gamma)
        if correct:
            states, actions, rewards, nxt, dones = _batch_tensors(chain, batch, device)
            rotation = td_rotational_component(
                net, theta, grad, states, nxt, rewards, dones, gamma, actions=actions, online=True
            )
            grad = grad - rotation
        theta = (theta - lr * grad).detach()
        target = ((1.0 - current_tau) * target + current_tau * theta).detach()
        if step % measure_every != 0:
            continue
        stats = measure_chain_curl(net, theta, target, chain, batch, gamma, probes=2)
        with torch.no_grad():
            q_sa = net.forward(theta, chain.one_hot(batch["s"].to(device)))
            q_abs = float(q_sa.abs().mean())
            next_value = net.forward(target, chain.one_hot(batch["sp"].to(device))).max(dim=1).values
            backup = batch["r"].to(device) + gamma * (1.0 - batch["d"].to(device)) * next_value
            chosen = q_sa.gather(1, batch["a"].to(device).long()[:, None]).squeeze(1)
            td_abs = float((chosen - backup).abs().mean())
        bad = (not np.isfinite(stats["curl_fraction"])) or q_abs > 1e3 or float(theta.norm()) > 1e3
        diverged = diverged or bad
        rows.append(
            {
                "system": "chain_dqn",
                "seed": seed,
                "step": step,
                "lr": lr,
                "gamma": gamma,
                "tau": current_tau,
                "correct": int(correct),
                "adaptive_tau": int(adaptive_tau),
                "td_abs": td_abs,
                "q_abs": q_abs,
                "theta_norm": float(theta.norm()),
                **stats,
                "diverged": int(diverged),
            }
        )
        if adaptive_tau and stats["curl_fraction"] > 0.2:
            current_tau = min(current_tau, 0.02)
        if diverged:
            break
    return rows


def run_critic_curl_on_pendulum(steps: int, seed: int, device: torch.device) -> list[dict]:
    """SAC-style TD(0) critic on the pendulum, so the deep continuous case is covered."""
    torch.manual_seed(seed)
    env = Pendulum()
    net = FuncMLP([env.obs_dim + env.act_dim, 64, 64, 1], device, seed=seed)
    theta = net.theta.detach().clone()
    target = theta.clone()
    obs = env.reset(32, device)
    rows = []
    gamma = 0.99
    for step in range(steps):
        action = torch.tanh(torch.randn(obs.shape[0], env.act_dim, device=device))
        nxt, reward = env.step(obs, action)
        states = torch.cat([obs, action], dim=-1).detach()
        next_action = torch.tanh(torch.randn_like(action))
        next_states = torch.cat([nxt, next_action], dim=-1).detach()

        leaf = theta.detach().requires_grad_(True)
        prediction = net.forward(leaf, states).squeeze(-1)
        bootstrap = net.forward(target, next_states).squeeze(-1)
        loss = (prediction - (reward + gamma * bootstrap).detach()).pow(2).mean()
        (grad,) = torch.autograd.grad(loss, leaf)
        theta = (leaf.detach() - 1e-3 * grad.detach()).detach()
        target = (0.995 * target + 0.005 * theta).detach()
        obs = nxt.detach()
        if step % 25 == 0:
            dones = torch.zeros(states.shape[0], device=device)
            stats = td_curl(
                net,
                theta,
                states,
                next_states,
                reward,
                dones,
                gamma,
                pred_mode="scalar",
                boot_mode="scalar",
                probes=2,
                online=True,
            )
            rows.append({"system": "pendulum_td", "seed": seed, "step": step, "diverged": 0, **stats})
    return rows


def _beats(primary: float, others: list[float]) -> bool:
    if primary != primary:
        return False
    finite = [value for value in others if value == value]
    if not finite:
        return primary > 0.0
    return primary > max(finite)


def summarize(baird_rows: list[dict]) -> tuple[bool | None, str, list[dict]]:
    """Score curl at the first checkpoint against eventual divergence.

    Only the stable (frozen target) and unstable (semi-gradient) runs enter
    the correlation. Symplectic and adaptive runs are reported beside that,
    because they answer a different question: does removing the rotation, or
    dropping tau when the rotation is large, stop the divergence.
    """
    grouped: dict[tuple, list[dict]] = {}
    for row in baird_rows:
        grouped.setdefault((row["condition"], row["seed"]), []).append(row)
    per_run = []
    for (condition, seed), sequence in grouped.items():
        first, last = sequence[0], sequence[-1]
        per_run.append(
            {
                "condition": condition,
                "seed": seed,
                "diverged": last["diverged"],
                "curl_fraction": first["curl_fraction"],
                "td_abs": first["td_abs"],
                "q_abs": first["q_abs"],
                "final_norm": last["weight_norm"],
                "final_step": last["step"],
            }
        )
    contrast = [row for row in per_run if row["condition"] in ("stable", "unstable")]
    labels = np.array([row["diverged"] for row in contrast], dtype=np.float64)
    curl = np.array([row["curl_fraction"] for row in contrast], dtype=np.float64)
    td = np.array([row["td_abs"] for row in contrast], dtype=np.float64)
    magnitude = np.array([row["q_abs"] for row in contrast], dtype=np.float64)
    corr_curl = _point_biserial(curl, labels)
    corr_td = _point_biserial(td, labels)
    corr_q = _point_biserial(magnitude, labels)
    if len(contrast) < 2 or np.isnan(corr_curl) or labels.min() == labels.max():
        return None, "stable and unstable Baird runs did not produce both outcomes.", per_run
    gap = float(curl[labels == 1].mean() - curl[labels == 0].mean())
    supported = gap > 0.02 and _beats(corr_curl, [corr_td, corr_q])
    extras = []
    for condition in ("symplectic", "adaptive_tau"):
        subset = [row for row in per_run if row["condition"] == condition]
        if not subset:
            continue
        extras.append(
            f"{condition} step-0 curl {np.mean([row['curl_fraction'] for row in subset]):.3f}, "
            f"diverged {np.mean([row['diverged'] for row in subset]):.2f}, "
            f"final norm {np.mean([row['final_norm'] for row in subset]):.1f}"
        )
    td_text = "undefined (no variance at step 0)" if np.isnan(corr_td) else f"{corr_td:.3f}"
    q_text = "undefined (no variance at step 0)" if np.isnan(corr_q) else f"{corr_q:.3f}"
    text = (
        f"at step 0, curl/divergence correlation {corr_curl:.3f} vs TD-error {td_text} vs |Q| {q_text}; "
        f"mean curl fraction diverged-minus-stable = {gap:.3f}. "
        + ("; ".join(extras) + "." if extras else "")
    )
    return supported, text, per_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--out", type=Path, default=PROJECT / "results")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.smoke:
        baird_steps, chain_steps, seeds, hidden, pendulum_steps = 200, 250, [0], 32, 0
    elif args.full:
        baird_steps, chain_steps, seeds, hidden, pendulum_steps = 2000, 2500, [0, 1, 2, 3], 64, 400
    else:
        baird_steps, chain_steps, seeds, hidden, pendulum_steps = 800, 1200, [0, 1], 64, 200

    baird_conditions = [
        {"condition": "stable", "tau": 0.0, "correct": False, "adaptive_tau": False},
        {"condition": "unstable", "tau": 1.0, "correct": False, "adaptive_tau": False},
        {"condition": "symplectic", "tau": 1.0, "correct": True, "adaptive_tau": False},
        {"condition": "adaptive_tau", "tau": 1.0, "correct": False, "adaptive_tau": True},
    ]
    baird_rows: list[dict] = []
    for condition in baird_conditions:
        for seed in seeds:
            rows = run_baird_intervention(
                steps=baird_steps,
                alpha=0.05,
                seed=seed,
                measure_every=20 if not args.smoke else 40,
                **condition,
            )
            baird_rows.extend(rows)
            last = rows[-1]
            print(
                f"baird {condition['condition']} seed {seed}: step {last['step']} "
                f"norm {last['weight_norm']:.1f} curl {rows[0]['curl_fraction']:.3f} diverged {last['diverged']}"
            )

    conditions = [
        {"name": "stable", "lr": 1e-3, "gamma": 0.9, "tau": 0.02, "correct": False, "adaptive_tau": False},
        {"name": "unstable", "lr": 5e-2, "gamma": 0.99, "tau": 1.0, "correct": False, "adaptive_tau": False},
        {"name": "symplectic", "lr": 5e-2, "gamma": 0.99, "tau": 1.0, "correct": True, "adaptive_tau": False},
        {"name": "adaptive_tau", "lr": 5e-2, "gamma": 0.99, "tau": 1.0, "correct": False, "adaptive_tau": True},
    ]
    chain_rows: list[dict] = []
    for condition in conditions:
        for seed in seeds:
            rows = train_chain(
                steps=chain_steps,
                seed=seed,
                lr=condition["lr"],
                gamma=condition["gamma"],
                tau=condition["tau"],
                device=device,
                measure_every=20 if not args.smoke else 50,
                correct=condition["correct"],
                adaptive_tau=condition["adaptive_tau"],
                hidden=hidden,
            )
            for row in rows:
                row["condition"] = condition["name"]
            chain_rows.extend(rows)
            last = rows[-1] if rows else {}
            print(
                f"chain {condition['name']} seed {seed}: steps {last.get('step')} "
                f"curl {last.get('curl_fraction', float('nan')):.3f} diverged {last.get('diverged')}"
            )

    pendulum_rows: list[dict] = []
    if pendulum_steps:
        pendulum_rows = run_critic_curl_on_pendulum(pendulum_steps, seed=0, device=device)
        print(f"pendulum critic curl fraction, last: {pendulum_rows[-1]['curl_fraction']:.3f}")

    supported, text, per_run = summarize(baird_rows)
    verdict = kill_test(supported, text)
    args.out.mkdir(parents=True, exist_ok=True)
    save_csv(args.out / "baird_trace.csv", baird_rows)
    save_csv(args.out / "baird_runs.csv", per_run)
    save_csv(args.out / "chain_trace.csv", chain_rows)
    save_csv(args.out / "pendulum_trace.csv", pendulum_rows)
    (args.out / "verdict.txt").write_text(verdict + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
