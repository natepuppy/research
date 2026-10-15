"""T1. Why do flow policies beat Gaussians in online RL? A controlled decomposition.

Pilot version. One PPO trainer, eight policy heads that each add one
mechanism credited to generative policies, three point-mass tasks from
lib/control_tasks.py that each isolate one mechanism:

  heads
    gdiag      diagonal Gaussian                      (baseline)
    glowrank   Gaussian, covariance D + U U^T, rank 4  (correlated exploration)
    gou        diagonal Gaussian, OU-correlated noise   (temporal correlation)
    gmip       two-step residual mean + diag noise      (iterative computation)
    ggmm       mixture of 4 diagonal Gaussians           (multimodality)
    flow1      one-step pushforward, FPO surrogate       (surrogate, no iteration)
    flow8      8-step Euler flow, FPO surrogate          (full generative policy)
    flowfrozen 8-step flow centre + diagonal noise       (expressivity without flow exploration)

  tasks
    reach      no mechanism should matter (control)
    fork       bimodal optimal action at the start
    corridor   8 actuators, rank-2 effect, null-space penalty: correlated
               exploration should matter

All heads share the critic, the trainer, the network width, the number of
environment steps, and the number of gradient steps. Gaussian heads use the
exact log-likelihood ratio; flow heads use the FPO ratio
exp(-(L_cfm(theta) - L_cfm(theta_old))) with the (t, eps) draws fixed at
collection time. Per run we log the final return (sampled and deterministic),
the mean |cross-dimension correlation| of sampled actions, the fraction of
states where sampled actions split into two clusters, and the effective
rank of the exploration covariance.

Kill test (pilot form): on every task where flow8 beats gdiag by more than
the seed spread, some Gaussian head closes at least half the gap. KILLED if
on such a task no Gaussian head does and flow8 is measurably bimodal.
INCONCLUSIVE if flow8 never beats gdiag here; then these tasks do not
reproduce the humanoid gap and the question needs MJX.

    python code/t1_policy_heads.py --smoke
    python code/t1_policy_heads.py
    python code/t1_policy_heads.py --full
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent  # code/
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

from lib.control_tasks import make_task  # noqa: E402
from lib.io_utils import kill_test, load_csv, save_csv  # noqa: E402
from lib.nets import MLP  # noqa: E402

LOG2PI = math.log(2.0 * math.pi)
HEADS = ["gdiag", "glowrank", "gou", "gmip", "ggmm", "flow1", "flow8", "flowfrozen"]
GAUSSIAN_HEADS = ["gdiag", "glowrank", "gou", "gmip", "ggmm"]
TASKS = ["reach", "fork", "corridor"]
HIDDEN = 128


def diag_logp(x: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    return -0.5 * (((x - mean) / log_std.exp()).square() + 2.0 * log_std + LOG2PI).sum(-1)


class Head(nn.Module):
    """Interface: act(obs, state) -> action, aux, new_state; score(obs, action, aux) -> tensor.

    ratio = exp(score_new - score_old). aux holds whatever noise the score
    must be conditioned on. state is per-env carry (OU noise), reset by reset_state.
    """

    needs_entropy_bonus = True
    frozen_sigma = None

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.body = MLP([obs_dim, HIDDEN, HIDDEN])

    def reset_state(self, n: int, device) -> dict:
        return {}

    def aux_loss(self, obs: torch.Tensor, action: torch.Tensor, aux: dict) -> torch.Tensor:
        return torch.zeros((), device=obs.device)

    def set_progress(self, frac: float) -> None:
        pass


class GDiag(Head):
    def __init__(self, obs_dim, act_dim):
        super().__init__(obs_dim, act_dim)
        self.mean = nn.Linear(HIDDEN, act_dim)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def act(self, obs, state):
        mean = self.mean(self.body(obs))
        action = mean + self.log_std.exp() * torch.randn_like(mean)
        return action, {}, state

    def score(self, obs, action, aux):
        return diag_logp(action, self.mean(self.body(obs)), self.log_std.expand(action.shape))

    def mean_action(self, obs):
        return self.mean(self.body(obs))


class GLowRank(Head):
    def __init__(self, obs_dim, act_dim, rank: int = 4):
        super().__init__(obs_dim, act_dim)
        self.mean = nn.Linear(HIDDEN, act_dim)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        self.rank = min(rank, act_dim)
        self.log_diag = nn.Parameter(torch.full((act_dim,), -1.5))
        self.factor = nn.Parameter(torch.randn(act_dim, self.rank) * 0.3)

    def dist(self, obs):
        mean = self.mean(self.body(obs))
        B = mean.shape[0]
        return torch.distributions.LowRankMultivariateNormal(mean, self.factor.expand(B, -1, -1), self.log_diag.exp().square().expand(B, -1))

    def act(self, obs, state):
        return self.dist(obs).rsample(), {}, state

    def score(self, obs, action, aux):
        return self.dist(obs).log_prob(action)

    def mean_action(self, obs):
        return self.mean(self.body(obs))


class GOU(Head):
    """Diagonal Gaussian with temporally correlated standardised noise eps_t = rho eps_{t-1} + sqrt(1-rho^2) xi."""

    rho = 0.8

    def __init__(self, obs_dim, act_dim):
        super().__init__(obs_dim, act_dim)
        self.mean = nn.Linear(HIDDEN, act_dim)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def reset_state(self, n, device):
        return {"eps": torch.randn(n, self.act_dim, device=device)}

    def act(self, obs, state):
        eps_prev = state["eps"]
        xi = torch.randn_like(eps_prev)
        eps = self.rho * eps_prev + math.sqrt(1.0 - self.rho**2) * xi
        mean = self.mean(self.body(obs))
        action = mean + self.log_std.exp() * eps
        return action, {"eps_prev": eps_prev}, {"eps": eps}

    def score(self, obs, action, aux):
        mean = self.mean(self.body(obs))
        std = self.log_std.exp()
        cond_mean = mean + std * self.rho * aux["eps_prev"]
        cond_log_std = (self.log_std + 0.5 * math.log(1.0 - self.rho**2)).expand(action.shape)
        return diag_logp(action, cond_mean, cond_log_std)

    def mean_action(self, obs):
        return self.mean(self.body(obs))


class GMIP(Head):
    """Two-step residual mean with intermediate noise (Much-Ado-About-Noising style) and diagonal exploration noise."""

    inter_noise = 0.3

    def __init__(self, obs_dim, act_dim):
        super().__init__(obs_dim, act_dim)
        self.step1 = nn.Linear(HIDDEN, act_dim)
        nn.init.orthogonal_(self.step1.weight, gain=0.01)
        nn.init.zeros_(self.step1.bias)
        self.step2 = MLP([HIDDEN + act_dim, HIDDEN, act_dim])
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def means(self, obs, z):
        h = self.body(obs)
        a1 = self.step1(h)
        a2 = a1 + 0.1 * self.step2(torch.cat([h, a1 + z], -1))
        return a1, a2

    def act(self, obs, state):
        z = self.inter_noise * torch.randn(obs.shape[0], self.act_dim, device=obs.device)
        _, a2 = self.means(obs, z)
        action = a2 + self.log_std.exp() * torch.randn_like(a2)
        return action, {"z": z}, state

    def score(self, obs, action, aux):
        _, a2 = self.means(obs, aux["z"])
        return diag_logp(action, a2, self.log_std.expand(action.shape))

    def aux_loss(self, obs, action, aux):
        a1, a2 = self.means(obs, aux["z"])
        return 0.5 * (a1 - a2.detach()).square().sum(-1).mean()

    def mean_action(self, obs):
        _, a2 = self.means(obs, torch.zeros(obs.shape[0], self.act_dim, device=obs.device))
        return a2


class GGMM(Head):
    def __init__(self, obs_dim, act_dim, k: int = 4):
        super().__init__(obs_dim, act_dim)
        self.k = k
        self.means = nn.Linear(HIDDEN, k * act_dim)
        nn.init.orthogonal_(self.means.weight, gain=0.01)
        # spread the component means so the mixture starts multimodal
        self.means.bias.data = (torch.randn(k, act_dim) * 0.3).reshape(-1)
        self.logits = nn.Linear(HIDDEN, k)
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)
        self.log_std = nn.Parameter(torch.full((k, act_dim), -0.5))

    def params(self, obs):
        h = self.body(obs)
        means = self.means(h).view(-1, self.k, self.act_dim)
        return means, self.logits(h).log_softmax(-1)

    def act(self, obs, state):
        means, logw = self.params(obs)
        comp = torch.multinomial(logw.exp(), 1).squeeze(1)
        mean = means[torch.arange(obs.shape[0]), comp]
        action = mean + self.log_std[comp].exp() * torch.randn_like(mean)
        return action, {}, state

    def score(self, obs, action, aux):
        means, logw = self.params(obs)
        lp = diag_logp(action[:, None, :].expand_as(means), means, self.log_std[None].expand_as(means))
        return torch.logsumexp(logw + lp, dim=-1)

    def mean_action(self, obs):
        means, logw = self.params(obs)
        comp = logw.argmax(-1)
        return means[torch.arange(obs.shape[0]), comp]


class Flow(Head):
    """Rectified-flow policy a = Euler_K(z ~ N(0,I)), trained with the FPO surrogate (score = -L_cfm)."""

    needs_entropy_bonus = False
    n_cfm = 4

    def __init__(self, obs_dim, act_dim, steps: int):
        super().__init__(obs_dim, act_dim)
        self.steps = steps
        self.vel = MLP([HIDDEN + act_dim + 1, HIDDEN, act_dim])

    def velocity(self, h, x, t):
        return self.vel(torch.cat([h, x, t], -1))

    def integrate(self, obs, z):
        h = self.body(obs)
        x = z
        for k in range(self.steps):
            t = torch.full((obs.shape[0], 1), k / self.steps, device=obs.device)
            x = x + self.velocity(h, x, t) / self.steps
        return x

    def sample(self, obs):
        return self.integrate(obs, torch.randn(obs.shape[0], self.act_dim, device=obs.device))

    def act(self, obs, state):
        action = self.sample(obs)
        t = torch.rand(obs.shape[0], self.n_cfm, 1, device=obs.device)
        eps = torch.randn(obs.shape[0], self.n_cfm, self.act_dim, device=obs.device)
        return action, {"t": t, "eps": eps}, state

    def score(self, obs, action, aux):
        h = self.body(obs)
        t, eps = aux["t"], aux["eps"]
        M = t.shape[1]
        a = action[:, None, :].expand(-1, M, -1)
        x_t = (1.0 - t) * eps + t * a
        target = a - eps
        pred = self.velocity(h[:, None, :].expand(-1, M, -1), x_t, t)
        return -(pred - target).square().sum(-1).mean(1)

    def mean_action(self, obs):
        with torch.no_grad():
            reps = torch.cat([self.sample(obs) for _ in range(16)], 0).view(16, obs.shape[0], -1)
        return reps.mean(0)


class FlowFrozenExplore(Flow):
    """Same flow and surrogate, but exploration is diagonal Gaussian noise around the flow centre (z = 0)."""

    def __init__(self, obs_dim, act_dim):
        super().__init__(obs_dim, act_dim, steps=8)
        self.sigma = 0.6

    def set_progress(self, frac):
        self.sigma = 0.6 * (1.0 - frac) + 0.1 * frac

    def centre(self, obs):
        return self.integrate(obs, torch.zeros(obs.shape[0], self.act_dim, device=obs.device))

    def act(self, obs, state):
        action = self.centre(obs) + self.sigma * torch.randn(obs.shape[0], self.act_dim, device=obs.device)
        t = torch.rand(obs.shape[0], self.n_cfm, 1, device=obs.device)
        eps = torch.randn(obs.shape[0], self.n_cfm, self.act_dim, device=obs.device)
        return action, {"t": t, "eps": eps}, state

    def mean_action(self, obs):
        return self.centre(obs)


def make_head(name: str, obs_dim: int, act_dim: int) -> Head:
    return {
        "gdiag": lambda: GDiag(obs_dim, act_dim),
        "glowrank": lambda: GLowRank(obs_dim, act_dim),
        "gou": lambda: GOU(obs_dim, act_dim),
        "gmip": lambda: GMIP(obs_dim, act_dim),
        "ggmm": lambda: GGMM(obs_dim, act_dim),
        "flow1": lambda: Flow(obs_dim, act_dim, steps=1),
        "flow8": lambda: Flow(obs_dim, act_dim, steps=8),
        "flowfrozen": lambda: FlowFrozenExplore(obs_dim, act_dim),
    }[name]()


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = MLP([obs_dim, HIDDEN, HIDDEN, 1])

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


def gae(rewards, values, gamma=0.99, lam=0.95):
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    next_value = torch.zeros_like(rewards[0])  # fixed horizon, episode ends
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        adv[t] = running
        next_value = values[t]
    return adv, adv + values


def stack_aux(aux_list: list[dict]) -> dict:
    if not aux_list or not aux_list[0]:
        return {}
    return {k: torch.cat([a[k] for a in aux_list], 0) for k in aux_list[0]}


def index_aux(aux: dict, idx: torch.Tensor) -> dict:
    return {k: v[idx] for k, v in aux.items()}


def collect(task, head, critic, n_envs, device):
    obs = task.reset(n_envs, device)
    state = head.reset_state(n_envs, device)
    obs_b, act_b, score_b, rew_b, val_b, aux_b = [], [], [], [], [], []
    with torch.no_grad():
        for _ in range(task.horizon):
            action, aux, state = head.act(obs, state)
            score = head.score(obs, action, aux)
            value = critic(obs)
            nxt, reward = task.step(obs, action)
            obs_b.append(obs)
            act_b.append(action)
            score_b.append(score)
            rew_b.append(reward)
            val_b.append(value)
            aux_b.append(aux)
            obs = nxt
    rewards = torch.stack(rew_b)
    values = torch.stack(val_b)
    adv, ret = gae(rewards, values)
    return {
        "obs": torch.cat(obs_b),
        "action": torch.cat(act_b),
        "score": torch.cat(score_b),
        "adv": adv.reshape(-1),
        "ret": ret.reshape(-1),
        "aux": stack_aux(aux_b),
        "episode_return": float(rewards.sum(0).mean()),
    }


def ppo_update(head, critic, opt, batch, epochs=4, minibatch=4096, clip=0.2, ent_coef=0.003):
    N = batch["obs"].shape[0]
    adv = batch["adv"]
    adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
    for _ in range(epochs):
        perm = torch.randperm(N, device=adv.device)
        for start in range(0, N, minibatch):
            idx = perm[start : start + minibatch]
            obs, action = batch["obs"][idx], batch["action"][idx]
            aux = index_aux(batch["aux"], idx)
            new_score = head.score(obs, action, aux)
            ratio = (new_score - batch["score"][idx]).exp()
            a = adv[idx]
            surrogate = torch.minimum(ratio * a, ratio.clamp(1.0 - clip, 1.0 + clip) * a).mean()
            loss = -surrogate + head.aux_loss(obs, action, aux)
            if head.needs_entropy_bonus:
                loss = loss + ent_coef * new_score.mean()  # -entropy proxy
            value = critic(obs)
            loss = loss + 0.5 * (value - batch["ret"][idx]).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(head.parameters()) + list(critic.parameters()), 1.0)
            opt.step()


@torch.no_grad()
def evaluate(task, head, n: int, device, deterministic: bool) -> float:
    obs = task.reset(n, device)
    state = head.reset_state(n, device)
    total = torch.zeros(n, device=device)
    for _ in range(task.horizon):
        if deterministic:
            action = head.mean_action(obs)
        else:
            action, _, state = head.act(obs, state)
        obs, reward = task.step(obs, action)
        total += reward
    return float(total.mean())


@torch.no_grad()
def diagnostics(task, head, device, n_states: int = 128, n_samples: int = 64) -> dict:
    """Action-distribution diagnostics at states visited by the current policy."""
    obs = task.reset(n_states, device)
    state = head.reset_state(n_states, device)
    visited = [obs]
    for _ in range(task.horizon // 2):
        action, _, state = head.act(obs, state)
        obs, _ = task.step(obs, action)
        visited.append(obs)
    states = torch.cat(visited)[torch.randperm(len(visited) * n_states)[:n_states]]
    samples = []
    for _ in range(n_samples):
        st = head.reset_state(n_states, device)
        a, _, _ = head.act(states, st)
        samples.append(a)
    A = torch.stack(samples, 1)  # (states, samples, act_dim)
    centred = A - A.mean(1, keepdim=True)
    cov = torch.einsum("sni,snj->sij", centred, centred) / (n_samples - 1)
    std = cov.diagonal(dim1=-2, dim2=-1).sqrt().clamp_min(1e-6)
    corr = cov / (std[:, :, None] * std[:, None, :])
    d = A.shape[-1]
    off = corr[:, ~torch.eye(d, dtype=torch.bool, device=device)].abs().mean() if d > 1 else torch.zeros(())
    mean_cov = torch.nan_to_num(cov.mean(0), nan=0.0, posinf=1e6, neginf=-1e6).double()
    mean_cov = 0.5 * (mean_cov + mean_cov.T) + 1e-9 * torch.eye(d, device=device, dtype=torch.float64)
    try:
        eig = torch.linalg.eigvalsh(mean_cov).clamp_min(1e-12)
    except RuntimeError:  # ill-conditioned (diverged policy); fall back to the diagonal
        eig = mean_cov.diagonal().clamp_min(1e-12)
    p = eig / eig.sum()
    eff_rank = float(torch.exp(-(p * p.log()).sum()))
    A = torch.nan_to_num(A, nan=0.0, posinf=1e3, neginf=-1e3)
    cov = torch.nan_to_num(cov, nan=0.0, posinf=1e6, neginf=0.0)
    # two-means split statistic, calibrated against a Gaussian with the same
    # mean and covariance (an elongated Gaussian also splits cleanly, so the
    # raw between/within ratio is not enough).
    def split_ratio(X: torch.Tensor) -> tuple[float, bool]:
        c = X[torch.randperm(X.shape[0])[:2]].clone()
        for _ in range(10):
            assign = torch.cdist(X, c).argmin(1)
            for k in range(2):
                if (assign == k).any():
                    c[k] = X[assign == k].mean(0)
        within = torch.stack([(X[assign == k] - c[k]).norm(dim=1).mean() if (assign == k).any() else torch.zeros(()) for k in range(2)]).mean()
        between = (c[0] - c[1]).norm()
        balanced = min(float((assign == 0).float().mean()), float((assign == 1).float().mean())) > 0.2
        return float(between / within.clamp_min(1e-6)), balanced

    bimodal = 0
    for s in range(n_states):
        X = A[s]
        ratio, balanced = split_ratio(X)
        mu = X.mean(0)
        cov_s = cov[s] + 1e-6 * torch.eye(d, device=device)
        try:
            null = torch.distributions.MultivariateNormal(mu, cov_s).sample((n_samples,))
        except (RuntimeError, ValueError):
            null = mu + cov_s.diagonal().clamp_min(0).sqrt() * torch.randn(n_samples, d, device=device)
        ratio_null, _ = split_ratio(null)
        if balanced and ratio > 1.5 * ratio_null:
            bimodal += 1
    return {"cross_corr": float(off), "eff_rank": eff_rank, "bimodal_frac": bimodal / n_states}


def train(task_name: str, head_name: str, iterations: int, seed: int, n_envs: int, device) -> dict:
    torch.manual_seed(seed)
    task = make_task(task_name)
    head = make_head(head_name, task.obs_dim, task.act_dim).to(device)
    critic = Critic(task.obs_dim).to(device)
    opt = torch.optim.Adam(list(head.parameters()) + list(critic.parameters()), lr=1e-3)
    curve = []
    start = time.time()
    for it in range(iterations):
        head.set_progress(it / max(iterations - 1, 1))
        batch = collect(task, head, critic, n_envs, device)
        curve.append(batch["episode_return"])
        ppo_update(head, critic, opt, batch)
    sampled = evaluate(task, head, 512, device, deterministic=False)
    determ = evaluate(task, head, 512, device, deterministic=True)
    diag = diagnostics(task, head, device)
    half = len(curve) // 2
    return {
        "task": task_name,
        "head": head_name,
        "seed": seed,
        "return_sampled": sampled,
        "return_det": determ,
        "det_gap": determ - sampled,
        "auc_second_half": float(np.mean(curve[half:])) if curve else float("nan"),
        "seconds": time.time() - start,
        **diag,
    }


def summarise(runs: list[dict]) -> tuple[list[dict], str, bool | None]:
    rows = []
    gap_tasks, closed, killed = [], [], []
    for task in TASKS:
        per_head = {}
        for head in HEADS:
            vals = [r["return_det"] for r in runs if r["task"] == task and r["head"] == head]
            if not vals:
                continue
            vals = np.nan_to_num(np.array(vals, dtype=float), nan=-1e3)
            per_head[head] = (float(np.mean(vals)), float(np.std(vals)), np.mean([r["bimodal_frac"] for r in runs if r["task"] == task and r["head"] == head]), np.nanmean([r["cross_corr"] for r in runs if r["task"] == task and r["head"] == head]), float(np.min(vals)), float(np.median(vals)))
        if "gdiag" not in per_head or "flow8" not in per_head:
            continue
        base, flow = per_head["gdiag"], per_head["flow8"]
        gap = flow[0] - base[0]
        noise = max(base[1], flow[1], 1e-6)
        for head, (m, s, bim, cc, lo, med) in per_head.items():
            rows.append({"task": task, "head": head, "return_det_mean": m, "return_det_std": s, "return_det_min": lo, "return_det_median": med, "gap_vs_gdiag": m - base[0], "gap_closed_frac": (m - base[0]) / gap if abs(gap) > 1e-9 else float("nan"), "bimodal_frac": bim, "cross_corr": cc})
        if flow[4] > base[4] + noise / 2 and gap <= noise:
            closed.append(f"{task}: flow8 is more reliable than gdiag (worst seed {flow[4]:.1f} vs {base[4]:.1f}) but not better on average; heads with worst seed >= flow8's: " + ", ".join(h for h in GAUSSIAN_HEADS if h in per_head and per_head[h][4] >= flow[4] - 1.0))
        if gap > noise:
            gap_tasks.append(task)
            best = max(((per_head[h][0] - base[0]) / gap, h) for h in GAUSSIAN_HEADS if h in per_head and h != "gdiag")
            if best[0] >= 0.5:
                closed.append(f"{task}: {best[1]} closes {best[0]:.0%}")
            elif flow[2] > 0.2:
                killed.append(f"{task}: best Gaussian {best[1]} closes {best[0]:.0%}, flow8 bimodal at {flow[2]:.0%} of states")
            else:
                closed.append(f"{task}: best Gaussian {best[1]} closes only {best[0]:.0%} but flow8 is unimodal, mechanism unresolved")
    if not gap_tasks:
        text = "flow8 did not beat gdiag on average beyond seed spread on any task; these tasks do not reproduce the humanoid gap, so the pilot cannot attribute it."
        if closed:
            text += " Reliability note: " + "; ".join(closed) + "."
        return rows, text, None
    if killed:
        return rows, "; ".join(killed + closed), False
    return rows, "; ".join(closed), True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=PROJECT / "results")
    parser.add_argument("--heads", default=",".join(HEADS))
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    parser.add_argument("--resume", action="store_true", help="skip (task, head, seed) runs already in results/runs.csv")
    parser.add_argument("--summarize", action="store_true", help="only re-summarise existing results")
    args = parser.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.smoke:
        iterations, seeds, n_envs = 2, [0], 32
    elif args.full:
        iterations, seeds, n_envs = 150, list(range(10)), 512
    else:
        iterations, seeds, n_envs = 80, [0, 1, 2], 256
    heads = args.heads.split(",")
    tasks = args.tasks.split(",")
    runs = load_csv(args.out / "runs.csv") if (args.resume or args.summarize) else []
    done = {(r["task"], r["head"], r["seed"]) for r in runs}
    for task in tasks:
        for head in heads:
            for seed in seeds:
                if args.summarize or (task, head, seed) in done:
                    continue
                row = train(task, head, iterations, seed, n_envs, device)
                runs.append(row)
                print(f"{task:9s} {head:11s} seed {seed}: det {row['return_det']:7.2f} sampled {row['return_sampled']:7.2f} bimodal {row['bimodal_frac']:.2f} corr {row['cross_corr']:.2f} rank {row['eff_rank']:.2f} ({row['seconds']:.0f}s)", flush=True)
                save_csv(args.out / "runs.csv", runs)
    summary, text, supported = summarise(runs)
    verdict = kill_test(supported, text)
    for row in summary:
        print(f"{row['task']:9s} {row['head']:11s} det {row['return_det_mean']:7.2f} ± {row['return_det_std']:5.2f}  closed {row['gap_closed_frac']:+.2f}  bimodal {row['bimodal_frac']:.2f}")
    save_csv(args.out / "summary.csv", summary)
    (args.out / "verdict.txt").write_text(verdict + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
