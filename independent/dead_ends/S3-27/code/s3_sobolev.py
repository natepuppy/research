"""S3-27. Sobolev critics from a differentiable simulator.

The one-step Bellman identity says

    dQ/da = dr/da + gamma * (ds'/da)^T dV/ds'

when the bootstrap value is V. A differentiable step gives dr/da and ds'/da
analytically. The Sobolev critic regresses its own dQ/da onto that target,
instead of only regressing Q onto the scalar backup.

The simulators are a smooth pendulum and a 1D point mass whose wall stiffness
is the contact-richness dial (0, soft, stiff). They are pure torch functions,
so no MuJoCo license and no MJX install are required. MJX would only change
the body of `env.step`; the loss does not care.

Logged for every run: episodic return, cosine between dQ/da and the
horizon-unrolled ground-truth action gradient, and the same run with noise
injected into the simulator gradient.

Kill test: the Sobolev critic is no better, or it gets worse as contact stiffens.

    python code/s3_sobolev.py
    python code/s3_sobolev.py --smoke
    python code/s3_sobolev.py --full
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent  # code/
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

from lib.envs import ContactPoint, Pendulum  # noqa: E402
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
        nn.init.orthogonal_(self.log_std.weight, gain=0.01)
        nn.init.zeros_(self.log_std.bias)

    def _dist(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(obs)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(-5.0, 2.0)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self._dist(obs)
        std = log_std.exp()
        pre_tanh = mean + std * torch.randn_like(mean)
        action = torch.tanh(pre_tanh)
        log_base = -0.5 * (((pre_tanh - mean) / std).square() + 2.0 * log_std + math.log(2.0 * math.pi)).sum(-1)
        log_prob = log_base - torch.log(1.0 - action.square() + 1e-6).sum(-1)
        return action, log_prob

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self._dist(obs)
        return torch.tanh(mean)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = MLP([obs_dim + act_dim, 64, 64, 1])

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


class Replay:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: torch.device):
        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.action = torch.zeros(capacity, act_dim, device=device)
        self.reward = torch.zeros(capacity, device=device)
        self.nxt = torch.zeros(capacity, obs_dim, device=device)
        self.pointer = 0
        self.count = 0
        self.capacity = capacity

    def add(self, obs, action, reward, nxt) -> None:
        n = obs.shape[0]
        index = (torch.arange(n, device=obs.device) + self.pointer) % self.capacity
        self.obs[index] = obs.detach()
        self.action[index] = action.detach()
        self.reward[index] = reward.detach()
        self.nxt[index] = nxt.detach()
        self.pointer = int((self.pointer + n) % self.capacity)
        self.count = min(self.capacity, self.count + n)

    def sample(self, batch: int):
        index = torch.randint(0, self.count, (batch,), device=self.obs.device)
        return self.obs[index], self.action[index], self.reward[index], self.nxt[index]


def jacobian_wrt_action(output: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """output (B,) or (B, k) -> tensor (B, k, act_dim)."""
    if output.ndim == 1:
        output = output[:, None]
    columns = []
    for component in range(output.shape[-1]):
        (grad,) = torch.autograd.grad(output[:, component].sum(), action, retain_graph=True, create_graph=True)
        columns.append(grad)
    return torch.stack(columns, dim=1)


def sobolev_target(env, target_a: Critic, target_b: Critic, actor: Actor, obs, action, gamma: float, noise_std: float):
    """dr/da + gamma (ds'/da)^T dV/ds', optionally with multiplicative gradient noise."""
    nxt, reward = env.step(obs.detach(), action)
    dr_da = jacobian_wrt_action(reward, action).squeeze(1)
    ds_da = jacobian_wrt_action(nxt, action)
    nxt_leaf = nxt.detach().requires_grad_(True)
    next_action = actor.mean_action(nxt_leaf).detach()
    value = torch.minimum(target_a(nxt_leaf, next_action), target_b(nxt_leaf, next_action))
    (grad_s,) = torch.autograd.grad(value.sum(), nxt_leaf)
    if noise_std > 0.0:
        dr_da = dr_da * (1.0 + noise_std * torch.randn_like(dr_da))
        ds_da = ds_da * (1.0 + noise_std * torch.randn_like(ds_da))
    chain = torch.einsum("bi,bij->bj", grad_s.detach(), ds_da)
    return dr_da + gamma * chain


def horizon_action_grad(env, actor: Actor, obs: torch.Tensor, action: torch.Tensor, gamma: float, horizon: int) -> torch.Tensor:
    action = action.detach().requires_grad_(True)
    state = obs.detach()
    total = torch.zeros(obs.shape[0], device=obs.device)
    discount = 1.0
    for step in range(horizon):
        act = action if step == 0 else actor.mean_action(state)
        state, reward = env.step(state, act)
        total = total + discount * reward
        discount *= gamma
    (grad,) = torch.autograd.grad(total.sum(), action)
    return grad.detach()


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.reshape(left.shape[0], -1)
    right = right.reshape(right.shape[0], -1)
    num = (left * right).sum(dim=-1)
    den = left.norm(dim=-1).clamp_min(1e-8) * right.norm(dim=-1).clamp_min(1e-8)
    return float((num / den).mean())


def evaluate(env, actor: Actor, horizon: int, episodes: int, device: torch.device) -> float:
    obs = env.reset(episodes, device)
    total = torch.zeros(episodes, device=device)
    with torch.no_grad():
        for _ in range(horizon):
            obs, reward = env.step(obs, actor.mean_action(obs))
            total = total + reward
    return float(total.mean())


def polyak(source: nn.Module, dest: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for online, frozen in zip(source.parameters(), dest.parameters()):
            frozen.mul_(1.0 - tau).add_(online, alpha=tau)


def train_one(env, updates: int, seed: int, device: torch.device, sobolev: bool, noise_std: float, gamma: float = 0.99) -> dict:
    torch.manual_seed(seed)
    actor = Actor(env.obs_dim, env.act_dim).to(device)
    critic_a = Critic(env.obs_dim, env.act_dim).to(device)
    critic_b = Critic(env.obs_dim, env.act_dim).to(device)
    target_a = Critic(env.obs_dim, env.act_dim).to(device)
    target_b = Critic(env.obs_dim, env.act_dim).to(device)
    target_a.load_state_dict(critic_a.state_dict())
    target_b.load_state_dict(critic_b.state_dict())
    opt_actor = torch.optim.Adam(actor.parameters(), lr=3e-4)
    opt_critic = torch.optim.Adam(list(critic_a.parameters()) + list(critic_b.parameters()), lr=3e-4)
    replay = Replay(20000, env.obs_dim, env.act_dim, device)
    obs = env.reset(64, device)
    alpha = 0.2
    last_cosine = float("nan")
    for update in range(updates):
        with torch.no_grad():
            action, _ = actor.sample(obs)
        nxt, reward = env.step(obs, action)
        replay.add(obs, action, reward, nxt)
        obs = nxt.detach()
        if update > 0 and update % 100 == 0:
            obs = env.reset(obs.shape[0], device)
        if replay.count < 256 or update % 2 != 0:
            continue
        s, a, r, sp = replay.sample(128)
        with torch.no_grad():
            next_action, next_logp = actor.sample(sp)
            backup = r + gamma * (torch.minimum(target_a(sp, next_action), target_b(sp, next_action)) - alpha * next_logp)
        a_leaf = a.detach().requires_grad_(sobolev)
        q1 = critic_a(s, a_leaf)
        q2 = critic_b(s, a_leaf)
        td = (q1 - backup).pow(2).mean() + (q2 - backup).pow(2).mean()
        loss = td
        if sobolev:
            target_grad = sobolev_target(env, target_a, target_b, actor, s, a_leaf, gamma, noise_std).detach()
            pred_a = jacobian_wrt_action(q1, a_leaf).squeeze(1)
            pred_b = jacobian_wrt_action(q2, a_leaf).squeeze(1)
            loss = loss + (pred_a - target_grad).pow(2).mean() + (pred_b - target_grad).pow(2).mean()
        opt_critic.zero_grad(set_to_none=True)
        loss.backward()
        opt_critic.step()
        for parameter in list(critic_a.parameters()) + list(critic_b.parameters()):
            parameter.requires_grad_(False)
        new_action, logp = actor.sample(s)
        policy_loss = (alpha * logp - torch.minimum(critic_a(s, new_action), critic_b(s, new_action))).mean()
        opt_actor.zero_grad(set_to_none=True)
        policy_loss.backward()
        opt_actor.step()
        for parameter in list(critic_a.parameters()) + list(critic_b.parameters()):
            parameter.requires_grad_(True)
        polyak(critic_a, target_a, 0.005)
        polyak(critic_b, target_b, 0.005)
        if update % 100 == 0 and replay.count >= 256:
            sample_obs = replay.obs[torch.randint(0, replay.count, (32,))]
            sample_act = actor.mean_action(sample_obs).detach()
            true_grad = horizon_action_grad(env, actor, sample_obs, sample_act, gamma, horizon=12)
            leaf = sample_act.detach().requires_grad_(True)
            (pred,) = torch.autograd.grad(critic_a(sample_obs, leaf).sum(), leaf)
            last_cosine = cosine(pred, true_grad)
    score = evaluate(env, actor, horizon=150, episodes=16, device=device)
    return {"return": score, "cosine": last_cosine}


def build_env(kind: str, stiffness: float):
    if kind == "pendulum":
        return Pendulum(), "pendulum", 0.0
    return ContactPoint(stiffness), "contact", stiffness


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--out", type=Path, default=PROJECT / "results")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.smoke:
        specs = [("pendulum", 0.0), ("contact", 30.0)]
        methods = [(False, 0.0), (True, 0.0)]
        updates, seeds = 120, [0]
    elif args.full:
        specs = [("pendulum", 0.0), ("contact", 0.0), ("contact", 8.0), ("contact", 40.0)]
        methods = [(False, 0.0), (True, 0.0), (True, 0.25), (True, 1.0)]
        updates, seeds = 1500, [0, 1]
    else:
        specs = [("pendulum", 0.0), ("contact", 0.0), ("contact", 40.0)]
        methods = [(False, 0.0), (True, 0.0), (True, 0.5)]
        updates, seeds = 400, [0]

    rows = []
    for kind, stiffness in specs:
        for sobolev, noise in methods:
            if (not sobolev) and noise > 0.0:
                continue
            for seed in seeds:
                env, name, stiff = build_env(kind, stiffness)
                result = train_one(env, updates, seed, device, sobolev, noise)
                row = {
                    "env": name,
                    "stiffness": stiff,
                    "sobolev": int(sobolev),
                    "grad_noise": noise,
                    "seed": seed,
                    **result,
                }
                rows.append(row)
                print(
                    f"{name} stiffness {stiff} sobolev {sobolev} noise {noise} seed {seed}: "
                    f"return {result['return']:.3f} cosine {result['cosine']:.3f}"
                )
    # Kill test uses the zero-noise pairs. Sobolev must beat plain SAC on the
    # smooth task and not fall behind on the stiffest contact task.
    def mean_return(env: str, stiffness: float, sobolev: int) -> float:
        values = [row["return"] for row in rows if row["env"] == env and row["stiffness"] == stiffness and row["sobolev"] == sobolev and row["grad_noise"] == 0.0]
        return float(sum(values) / len(values)) if values else float("nan")

    smooth_gain = mean_return("pendulum", 0.0, 1) - mean_return("pendulum", 0.0, 0)
    stiff_values = [row["stiffness"] for row in rows if row["env"] == "contact"]
    stiffest = max(stiff_values) if stiff_values else float("nan")
    contact_gain = mean_return("contact", stiffest, 1) - mean_return("contact", stiffest, 0)
    if any(map(lambda v: v != v, (smooth_gain, contact_gain))):  # NaN
        supported, text = None, "a paired condition is missing, so the kill test cannot be scored."
    else:
        supported = smooth_gain > 0.0 and contact_gain > -0.05 * max(1.0, abs(mean_return("contact", stiffest, 0)))
        text = (
            f"Sobolev minus SAC return on the pendulum is {smooth_gain:.3f}; "
            f"on contact stiffness {stiffest} it is {contact_gain:.3f}."
        )
    verdict = kill_test(bool(supported) if supported is not None else None, text) if supported is not None else kill_test(None, text)
    # kill_test(bool) when supported is False is fine; the line above double-calls. Fix:
    # I already called it. Don't call twice. Let me look... I assigned verdict with a
    # conditional that always calls once. Good. Wait, `kill_test(bool(supported) if supported is not None else None` 
    # when supported is False, bool(False) is False. Good.
    # when supported is None we go to else branch only if nan. The if supported is not None in the ternary
    # is inside the else-not structure. Let me re-read.
    # I wrote:
    # if nan: supported, text = None, ...
    # else: supported = ...; text = ...
    # verdict = kill_test(...) if supported is not None else kill_test(None, text)
    # That's one call. Good.
    save_csv(args.out / "runs.csv", rows)
    (args.out / "verdict.txt").write_text(verdict + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
