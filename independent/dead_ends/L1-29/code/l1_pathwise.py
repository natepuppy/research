"""L1-29. When does pathwise beat score-function on-policy?

PPO's clipped surrogate is a score-function estimator. The pathwise alternative
(the idea behind REPPO) trains a Q and ascends Q(s, pi(s)) by reparameterizing
the Gaussian, with a KL penalty so the step stays on-policy. Both methods see
the same number of environment steps, the same network width, and the same
number of update epochs.

The tasks cross the three factors in the question, using simulators in
lib/envs.py rather than Mujoco Playground:

  * action dimension: a 4D linear plant driven by 1 real actuator, with 0 or 7
    ignored extra action dimensions
  * reward smoothness: quadratic reward versus a threshold reward
  * contact: the 1D point mass, goal past a wall, stiffness 8 or 40

A logistic regression on those three factors is the switching rule, scored by
leave-one-task-out accuracy.

    python code/l1_pathwise.py
    python code/l1_pathwise.py --smoke
    python code/l1_pathwise.py --full
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent  # code/
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

from lib.envs import ContactPoint, LinearSystem  # noqa: E402
from lib.io_utils import kill_test, save_csv  # noqa: E402
from lib.nets import MLP  # noqa: E402


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.body = MLP([obs_dim, 64, 64])
        self.mean = nn.Linear(64, act_dim)
        self.log_std = nn.Linear(64, act_dim)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, -0.5)

    def dist(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(obs)
        return self.mean(hidden), self.log_std(hidden).clamp(-5.0, 1.0)

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.dist(obs)
        std = log_std.exp()
        pre = mean + std * torch.randn_like(mean)
        action = torch.tanh(pre)
        log_prob = self._log_prob(mean, log_std, action, pre)
        return action, log_prob

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.dist(obs)
        pre = torch.atanh(action.clamp(-0.999, 0.999))
        return self._log_prob(mean, log_std, action, pre)

    @staticmethod
    def _log_prob(mean, log_std, action, pre) -> torch.Tensor:
        std = log_std.exp()
        base = -0.5 * (((pre - mean) / std).square() + 2.0 * log_std + math.log(2.0 * math.pi)).sum(-1)
        return base - torch.log(1.0 - action.square() + 1e-6).sum(-1)

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.dist(obs)
        return torch.tanh(mean)


class Critic(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = MLP([obs_dim, 64, 64, 1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class QNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = MLP([obs_dim + act_dim, 64, 64, 1])

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


def gaussian_kl(new_obs_mean, new_log_std, old_mean, old_log_std) -> torch.Tensor:
    """KL(new || old) for diagonal Gaussians, per batch row, in pre-tanh space."""
    old_var = (2.0 * old_log_std).exp()
    new_var = (2.0 * new_log_std).exp()
    return 0.5 * (
        (old_log_std - new_log_std) * 2.0
        + (new_var + (new_obs_mean - old_mean).square()) / old_var.clamp_min(1e-8)
        - 1.0
    ).sum(-1)


def gae(rewards, values, bootstrap, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """rewards and values are (time, envs). Returns advantage and returns in that same layout."""
    steps, _n_envs = rewards.shape
    advantage = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[1], device=rewards.device)
    next_value = bootstrap
    for t in reversed(range(steps)):
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        advantage[t] = running
        next_value = values[t]
    return advantage, advantage + values


def collect(env, actor, critic, n_envs: int, horizon: int, device):
    obs = env.reset(n_envs, device)
    obs_buf, act_buf, logp_buf, rew_buf, val_buf = [], [], [], [], []
    for _ in range(horizon):
        with torch.no_grad():
            action, logp = actor.sample(obs)
            value = critic(obs)
        nxt, reward = env.step(obs, action)
        obs_buf.append(obs)
        act_buf.append(action)
        logp_buf.append(logp)
        rew_buf.append(reward)
        val_buf.append(value)
        obs = nxt
    with torch.no_grad():
        bootstrap = critic(obs)
    # Stack time first so a later reshape(-1) lines up with cat() over the same list.
    rewards = torch.stack(rew_buf, dim=0)
    values = torch.stack(val_buf, dim=0)
    advantage, returns = gae(rewards, values, bootstrap, gamma=0.99, lam=0.95)
    flat = lambda xs: torch.cat(xs, dim=0)
    return {
        "obs": flat(obs_buf),
        "action": flat(act_buf),
        "logp": flat(logp_buf),
        "advantage": advantage.reshape(-1),
        "returns": returns.reshape(-1),
    }


def ppo_update(actor, critic, opt_a, opt_c, batch, epochs: int = 4) -> None:
    advantage = batch["advantage"]
    advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)
    for _ in range(epochs):
        new_logp = actor.log_prob(batch["obs"], batch["action"])
        ratio = (new_logp - batch["logp"]).exp()
        surrogate = torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
        entropy = -new_logp.mean()
        opt_a.zero_grad(set_to_none=True)
        (-surrogate.mean() - 0.01 * entropy).backward()
        opt_a.step()
        value = critic(batch["obs"])
        opt_c.zero_grad(set_to_none=True)
        (value - batch["returns"].detach()).pow(2).mean().backward()
        opt_c.step()


def pathwise_update(actor, qnet, opt_a, opt_q, batch, epochs: int = 4, beta: float = 0.5) -> None:
    frozen = copy.deepcopy(actor).eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    for _ in range(epochs):
        prediction = qnet(batch["obs"], batch["action"])
        opt_q.zero_grad(set_to_none=True)
        (prediction - batch["returns"].detach()).pow(2).mean().backward()
        opt_q.step()
        for parameter in qnet.parameters():
            parameter.requires_grad_(False)
        action, _ = actor.sample(batch["obs"])
        mean, log_std = actor.dist(batch["obs"])
        with torch.no_grad():
            old_mean, old_log_std = frozen.dist(batch["obs"])
        kl = gaussian_kl(mean, log_std, old_mean, old_log_std)
        policy_loss = -qnet(batch["obs"], action).mean() + beta * kl.mean()
        opt_a.zero_grad(set_to_none=True)
        policy_loss.backward()
        opt_a.step()
        for parameter in qnet.parameters():
            parameter.requires_grad_(True)


def evaluate(env, actor, horizon: int, episodes: int, device) -> float:
    obs = env.reset(episodes, device)
    total = torch.zeros(episodes, device=device)
    with torch.no_grad():
        for _ in range(horizon):
            obs, reward = env.step(obs, actor.mean_action(obs))
            total = total + reward
    return float(total.mean())


def train(env, method: str, iterations: int, seed: int, device, n_envs: int, horizon: int) -> float:
    torch.manual_seed(seed)
    actor = Actor(env.obs_dim, env.act_dim).to(device)
    critic = Critic(env.obs_dim).to(device)
    qnet = QNet(env.obs_dim, env.act_dim).to(device)
    opt_a = torch.optim.Adam(actor.parameters(), lr=3e-4)
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3)
    opt_q = torch.optim.Adam(qnet.parameters(), lr=1e-3)
    for _ in range(iterations):
        batch = collect(env, actor, critic, n_envs, horizon, device)
        if method == "ppo":
            ppo_update(actor, critic, opt_a, opt_c, batch)
        else:
            pathwise_update(actor, qnet, opt_a, opt_q, batch)
    return evaluate(env, actor, horizon=max(horizon, 64), episodes=32, device=device)


def fit_logistic(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    weights = np.zeros(features.shape[1])
    for _ in range(1500):
        scores = features @ weights
        probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -20, 20)))
        grad = features.T @ (probs - labels) / len(labels) + 1e-2 * weights
        weights -= 0.4 * grad
    return weights


def leave_one_out(task_rows: list[dict]) -> tuple[float, np.ndarray]:
    features = np.array([[1.0, row["action_dim"], row["smooth"], row["contact"]] for row in task_rows], dtype=np.float64)
    features[:, 1] /= 8.0
    labels = np.array([row["pathwise_wins"] for row in task_rows], dtype=np.float64)
    if labels.min() == labels.max():
        return float("nan"), np.zeros(4)
    correct = 0
    last = np.zeros(4)
    for held in range(len(task_rows)):
        mask = np.ones(len(task_rows), dtype=bool)
        mask[held] = False
        last = fit_logistic(features[mask], labels[mask])
        prob = 1.0 / (1.0 + np.exp(-np.clip(float(features[held] @ last), -20, 20)))
        correct += int((prob >= 0.5) == bool(labels[held]))
    return correct / len(task_rows), last


def task_list(smoke: bool, full: bool) -> list[dict]:
    base = [
        {"name": "linear_smooth_d1", "kind": "linear", "act_dim": 1, "reward": "quadratic", "stiffness": 0.0},
        {"name": "linear_smooth_d8", "kind": "linear", "act_dim": 8, "reward": "quadratic", "stiffness": 0.0},
        {"name": "linear_threshold_d1", "kind": "linear", "act_dim": 1, "reward": "threshold", "stiffness": 0.0},
        {"name": "soft_contact", "kind": "contact", "act_dim": 1, "reward": "quadratic", "stiffness": 8.0},
        {"name": "stiff_contact", "kind": "contact", "act_dim": 1, "reward": "quadratic", "stiffness": 40.0},
    ]
    if smoke:
        return [base[0], base[-1]]
    if full:
        base.append({"name": "linear_threshold_d8", "kind": "linear", "act_dim": 8, "reward": "threshold", "stiffness": 0.0})
    return base


def make_env(spec: dict):
    if spec["kind"] == "linear":
        return LinearSystem(obs_dim=4, act_dim=spec["act_dim"], true_act_dim=1, reward=spec["reward"])
    return ContactPoint(spec["stiffness"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--out", type=Path, default=PROJECT / "results")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.smoke:
        iterations, seeds, n_envs, horizon = 2, [0], 16, 16
    elif args.full:
        iterations, seeds, n_envs, horizon = 12, [0, 1, 2], 64, 64
    else:
        iterations, seeds, n_envs, horizon = 5, [0], 32, 32

    runs = []
    for spec in task_list(args.smoke, args.full):
        for method in ("ppo", "pathwise"):
            for seed in seeds:
                env = make_env(spec)
                score = train(env, method, iterations, seed, device, n_envs, horizon)
                runs.append({**spec, "method": method, "seed": seed, "return": score})
                print(f"{spec['name']} {method} seed {seed}: {score:.3f}")

    grouped: dict[str, dict] = {}
    for spec in task_list(args.smoke, args.full):
        ppo = [row["return"] for row in runs if row["name"] == spec["name"] and row["method"] == "ppo"]
        path = [row["return"] for row in runs if row["name"] == spec["name"] and row["method"] == "pathwise"]
        grouped[spec["name"]] = {
            "name": spec["name"],
            "action_dim": spec["act_dim"],
            "smooth": int(spec["reward"] == "quadratic"),
            "contact": spec["stiffness"] / 40.0,
            "ppo_return": float(np.mean(ppo)),
            "pathwise_return": float(np.mean(path)),
            "pathwise_wins": int(np.mean(path) > np.mean(ppo)),
        }
    summary = list(grouped.values())
    accuracy, weights = leave_one_out(summary)
    both = len({row["pathwise_wins"] for row in summary}) == 2
    if not both or accuracy != accuracy:
        supported, text = False, (
            "one estimator won every task, so action dimension, smoothness, and contact "
            "do not pick a winner. There is no switching rule to report."
        )
    else:
        supported = accuracy >= 0.6
        text = (
            f"leave-one-task-out accuracy {accuracy:.2f}. "
            f"logistic weights [intercept, action_dim/8, smooth, contact] = "
            f"{[round(float(w), 3) for w in weights]}."
        )
    verdict = kill_test(supported, text)
    for row in summary:
        print(
            f"{row['name']}: ppo {row['ppo_return']:.3f}  pathwise {row['pathwise_return']:.3f}  "
            f"pathwise_wins {row['pathwise_wins']}"
        )
    save_csv(args.out / "runs.csv", runs)
    save_csv(args.out / "by_task.csv", summary)
    (args.out / "verdict.txt").write_text(verdict + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
