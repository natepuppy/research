"""Batched deterministic actor-critic (DDPG with target networks) on Pendulum: R independent runs
stepped in lockstep as one tensor program, each with its own actor, critic, Adam states, replay
buffer, environment state and generators. Per-run learning rates are supported so a sweep over
actor/critic learning rates runs as one batch.

Also provides the ground-truth machinery for the critic-lag measurement: deterministic Pendulum
rollouts from arbitrary states under arbitrary (possibly past) actors, which give Monte-Carlo
Q^{pi}(s, a) exactly up to truncation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

OBS, ACT = 3, 1
MAX_STEPS = 200
MAX_SPEED, MAX_TORQUE, DT, G, M, L = 8.0, 2.0, 0.05, 10.0, 1.0, 1.0


# ----------------------------------------------------------------------------- env (gym Pendulum-v1 numbers)

def pend_reset(n: int, gens: list[torch.Generator]) -> torch.Tensor:
    """Internal state (theta, thetadot) with gym's reset ranges."""
    th = torch.stack([(torch.rand(1, generator=g) * 2 - 1) * math.pi for g in gens])[:, 0]
    thd = torch.stack([(torch.rand(1, generator=g) * 2 - 1) * 1.0 for g in gens])[:, 0]
    return torch.stack([th, thd], -1)


def angle_norm(x):
    return ((x + math.pi) % (2 * math.pi)) - math.pi


def pend_step(x: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    th, thd = x.unbind(-1)
    u = u.clamp(-MAX_TORQUE, MAX_TORQUE)[..., 0]
    cost = angle_norm(th) ** 2 + 0.1 * thd ** 2 + 0.001 * u ** 2
    thd2 = thd + (3 * G / (2 * L) * torch.sin(th) + 3.0 / (M * L ** 2) * u) * DT
    thd2 = thd2.clamp(-MAX_SPEED, MAX_SPEED)
    th2 = th + thd2 * DT
    return torch.stack([th2, thd2], -1), -cost


def obs_of(x: torch.Tensor) -> torch.Tensor:
    th, thd = x.unbind(-1)
    return torch.stack([torch.cos(th), torch.sin(th), thd], -1)


# ----------------------------------------------------------------------------- batched MLP + Adam

class MLP:
    """List of (W, b) per layer, each with a leading run dimension."""

    def __init__(self, sizes: list[int], gens: list[torch.Generator]):
        self.sizes = sizes
        self.params: list[torch.Tensor] = []
        for i, o in zip(sizes[:-1], sizes[1:]):
            bound = 1 / math.sqrt(i)
            W = torch.stack([(torch.rand(i, o, generator=g) * 2 - 1) * bound for g in gens])
            b = torch.stack([(torch.rand(o, generator=g) * 2 - 1) * bound for g in gens])
            self.params += [W, b]

    @staticmethod
    def forward(params: list[torch.Tensor], x: torch.Tensor, out_tanh: bool = False) -> torch.Tensor:
        """x: (R, B, in) -> (R, B, out)"""
        h = x
        n = len(params) // 2
        for k in range(n):
            W, b = params[2 * k], params[2 * k + 1]
            h = torch.bmm(h, W) + b[:, None]
            if k < n - 1:
                h = torch.relu(h)
        return torch.tanh(h) if out_tanh else h

    def clone_params(self):
        return [t.clone() for t in self.params]

    def index(self, idx: torch.Tensor):
        return [t[idx].clone() for t in self.params]


class Adam:
    def __init__(self, params: list[torch.Tensor], lr: torch.Tensor, b1=0.9, b2=0.999, eps=1e-8):
        self.m = [torch.zeros_like(t) for t in params]
        self.v = [torch.zeros_like(t) for t in params]
        self.t = torch.zeros(params[0].shape[0])
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps   # lr: (R,) per-run learning rate

    def step(self, params: list[torch.Tensor], grads: list[torch.Tensor]):
        self.t += 1
        for k, (w, g) in enumerate(zip(params, grads)):
            self.m[k].mul_(self.b1).add_(g, alpha=1 - self.b1)
            self.v[k].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            shape = (-1,) + (1,) * (w.dim() - 1)
            mhat = self.m[k] / (1 - self.b1 ** self.t).view(shape)
            vhat = self.v[k] / (1 - self.b2 ** self.t).view(shape)
            w.sub_(self.lr.view(shape) * mhat / (vhat.sqrt() + self.eps))


# ----------------------------------------------------------------------------- learner

@dataclass
class Config:
    hidden: int = 64
    gamma: float = 0.98
    buffer: int = 20_000
    batch: int = 64
    tau: float = 0.005
    expl_sigma: float = 0.2          # in torque units
    learn_start: int = 1_000
    total_steps: int = 20_000
    eval_every: int = 1_000
    probe_states: int = 64
    mc_horizon: int = 250
    max_lag: int = 6                 # how many past actors to test against


class BatchedDDPG:
    def __init__(self, cfg: Config, lr_actor: list[float], lr_critic: list[float], seeds: list[int]):
        self.cfg = cfg
        R = len(seeds)
        g_init = [torch.Generator().manual_seed(int(s)) for s in seeds]
        self.g_env = [torch.Generator().manual_seed(int(s) + 10_000) for s in seeds]
        self.g_batch = [torch.Generator().manual_seed(int(s) + 20_000) for s in seeds]
        self.actor = MLP([OBS, cfg.hidden, cfg.hidden, ACT], g_init)
        self.critic = MLP([OBS + ACT, cfg.hidden, cfg.hidden, 1], g_init)
        self.actor_t = self.actor.clone_params(); self.critic_t = self.critic.clone_params()
        self.opt_a = Adam(self.actor.params, torch.tensor(lr_actor, dtype=torch.float32))
        self.opt_c = Adam(self.critic.params, torch.tensor(lr_critic, dtype=torch.float32))
        self.lr_actor, self.lr_critic = lr_actor, lr_critic
        self.buf_x = torch.zeros(R, cfg.buffer, 2); self.buf_a = torch.zeros(R, cfg.buffer, ACT)
        self.buf_r = torch.zeros(R, cfg.buffer); self.buf_x2 = torch.zeros(R, cfg.buffer, 2)
        self.buf_n = torch.zeros(R, dtype=torch.long)
        self.step_count = 0
        self.x = pend_reset(R, self.g_env)
        self.ep_t = 0
        self.actor_history: list[list[torch.Tensor]] = []     # snapshots of actor params at each eval
        self.log: list[dict] = []

    @property
    def R(self):
        return self.actor.params[0].shape[0]

    def act(self, params, x):
        return MLP.forward(params, obs_of(x)[:, None], out_tanh=True)[:, 0] * MAX_TORQUE

    def step(self):
        R, cfg = self.R, self.cfg
        with torch.no_grad():
            a = self.act(self.actor.params, self.x)
            noise = torch.stack([torch.randn(ACT, generator=g) for g in self.g_env]) * cfg.expl_sigma
            if self.step_count < cfg.learn_start:
                a = torch.stack([(torch.rand(ACT, generator=g) * 2 - 1) for g in self.g_env]) * MAX_TORQUE
            else:
                a = (a + noise).clamp(-MAX_TORQUE, MAX_TORQUE)
        x2, r = pend_step(self.x, a)
        slot = self.buf_n % cfg.buffer
        ar = torch.arange(R)
        self.buf_x[ar, slot] = self.x; self.buf_a[ar, slot] = a; self.buf_r[ar, slot] = r; self.buf_x2[ar, slot] = x2
        self.buf_n += 1; self.step_count += 1; self.ep_t += 1
        if self.ep_t >= MAX_STEPS:
            x2 = pend_reset(R, self.g_env); self.ep_t = 0
        self.x = x2
        if self.step_count >= cfg.learn_start:
            self._update()
        if self.step_count % cfg.eval_every == 0:
            self.measure()

    def _update(self):
        R, cfg = self.R, self.cfg
        n = torch.minimum(self.buf_n, torch.tensor(cfg.buffer))
        idx = torch.stack([(torch.rand(cfg.batch, generator=g) * n[i]).long() for i, g in enumerate(self.g_batch)])
        ar = torch.arange(R)[:, None]
        x, a, r, x2 = self.buf_x[ar, idx], self.buf_a[ar, idx], self.buf_r[ar, idx], self.buf_x2[ar, idx]
        o, o2 = obs_of(x), obs_of(x2)
        with torch.no_grad():
            a2 = MLP.forward(self.actor_t, o2, out_tanh=True) * MAX_TORQUE
            q2 = MLP.forward(self.critic_t, torch.cat([o2, a2 / MAX_TORQUE], -1))[..., 0]
            y = r + cfg.gamma * q2                                   # no terminals in Pendulum
        # critic
        pc = [t.detach().requires_grad_(True) for t in self.critic.params]
        q = MLP.forward(pc, torch.cat([o, a / MAX_TORQUE], -1))[..., 0]
        loss_c = ((q - y) ** 2).mean(-1).sum()
        gc = torch.autograd.grad(loss_c, pc)
        self.opt_c.step(self.critic.params, list(gc))
        # actor
        pa = [t.detach().requires_grad_(True) for t in self.actor.params]
        a_pi = MLP.forward(pa, o, out_tanh=True)
        q_pi = MLP.forward([t.detach() for t in self.critic.params], torch.cat([o, a_pi], -1))[..., 0]
        loss_a = (-q_pi).mean(-1).sum()
        ga = torch.autograd.grad(loss_a, pa)
        self.opt_a.step(self.actor.params, list(ga))
        # targets
        with torch.no_grad():
            for t, s in zip(self.actor_t, self.actor.params):
                t.mul_(1 - cfg.tau).add_(s, alpha=cfg.tau)
            for t, s in zip(self.critic_t, self.critic.params):
                t.mul_(1 - cfg.tau).add_(s, alpha=cfg.tau)

    # ------------------------------------------------------------------ ground truth
    @torch.no_grad()
    def mc_q(self, actor_params: list[torch.Tensor], x0: torch.Tensor, a0: torch.Tensor) -> torch.Tensor:
        """Monte-Carlo Q^{pi}(x0, a0): take a0 then follow the given actor for mc_horizon steps.
        x0: (R, P, 2), a0: (R, P, 1) -> (R, P). Deterministic dynamics, so this is exact up to truncation."""
        cfg = self.cfg
        R, P = x0.shape[:2]
        x = x0.clone(); total = torch.zeros(R, P); disc = 1.0
        a = a0
        for t in range(cfg.mc_horizon):
            x, r = pend_step(x, a)
            total += disc * r; disc *= cfg.gamma
            a = MLP.forward(actor_params, obs_of(x), out_tanh=True) * MAX_TORQUE
        return total

    @torch.no_grad()
    def measure(self):
        """Eval return, critic error against the current policy's true Q, and the best-matching past policy."""
        R, cfg = self.R, self.cfg
        self.actor_history.append(self.actor.clone_params())
        # probe states: fixed per eval call (same for all runs), drawn from the state space uniformly
        g = torch.Generator().manual_seed(999 + len(self.actor_history))
        th = (torch.rand(cfg.probe_states, generator=g) * 2 - 1) * math.pi
        thd = (torch.rand(cfg.probe_states, generator=g) * 2 - 1) * MAX_SPEED * 0.5
        x0 = torch.stack([th, thd], -1)[None].expand(R, -1, -1).clone()
        o0 = obs_of(x0)
        a0 = MLP.forward(self.actor.params, o0, out_tanh=True) * MAX_TORQUE
        q_c = MLP.forward(self.critic.params, torch.cat([o0, a0 / MAX_TORQUE], -1))[..., 0]       # (R, P)
        errs = []   # errs[k] = (R,) mean abs error vs Q^{pi_{t-k}}(s, pi_t(s))
        biases = []
        truths = []
        for k in range(min(cfg.max_lag + 1, len(self.actor_history))):
            past = self.actor_history[-1 - k]
            q_true = self.mc_q(past, x0, a0)
            truths.append(q_true)
            errs.append((q_c - q_true).abs().mean(-1)); biases.append((q_c - q_true).mean(-1))
        # also: return of the current policy from standard resets (8 fixed starts)
        g2 = torch.Generator().manual_seed(4242)
        xs = pend_reset(8, [g2] * 8)[None].expand(R, -1, -1).clone()
        ret = self.mc_q(self.actor.params, xs, MLP.forward(self.actor.params, obs_of(xs), out_tanh=True) * MAX_TORQUE)
        # undiscounted-ish: report discounted return from resets (what the critic is supposed to estimate)
        E = torch.stack(errs, 0)         # (K, R)
        kstar = E.argmin(0)              # (R,)
        # policy change magnitude: mean |Q^{pi_t} - Q^{pi_{t-1}}| on probe states (how fast the target moves)
        drift = (truths[0] - truths[1]).abs().mean(-1) if len(truths) > 1 else torch.zeros(R)
        for i in range(R):
            self.log.append(dict(step=self.step_count, run=i, lr_actor=self.lr_actor[i], lr_critic=self.lr_critic[i],
                                 ret=float(ret[i].mean()), err0=float(errs[0][i]), bias0=float(biases[0][i]),
                                 err_kstar=float(E[kstar[i], i]), kstar=int(kstar[i]), n_lags=int(E.shape[0]),
                                 drift=float(drift[i]), qscale=float(truths[0][i].abs().mean()),
                                 **{f"err_lag{k}": float(errs[k][i]) for k in range(E.shape[0])}))
