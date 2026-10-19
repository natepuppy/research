"""A batched DQN: R independent runs, each with its own network, Adam state, replay buffer, env,
and three separately seeded noise sources, all stepped in lockstep as one tensor program.

Noise sources per run (each a torch.Generator):
    g_env    : environment resets + epsilon-greedy draws        ('explore' noise)
    g_batch  : minibatch index sampling from the replay buffer   ('batch' noise)
    (init noise is consumed once, at construction)

`snapshot(idx)` / `fork(parent_state, n_children, reseed=...)` copy everything a run owns, so a child
continues the parent's trajectory exactly unless one of its generators is reseeded.

Environment: CartPole with a 200-step cap (classic -v0 numbers), reward 1 per step, vectorised.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

OBS, ACT = 4, 2
MAX_STEPS = 200


# ----------------------------------------------------------------------------- env

def cartpole_reset(n: int, gen: list[torch.Generator]) -> torch.Tensor:
    return torch.stack([(torch.rand(OBS, generator=g) - 0.5) * 0.1 for g in gen])


def cartpole_step(s: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x, xd, th, thd = s.unbind(-1)
    force = torch.where(a == 1, 10.0, -10.0)
    cos, sin = torch.cos(th), torch.sin(th)
    temp = (force + 0.05 * thd ** 2 * sin) / 1.1
    thacc = (9.8 * sin - cos * temp) / (0.5 * (4.0 / 3.0 - 0.1 * cos ** 2 / 1.1))
    xacc = temp - 0.05 * thacc * cos / 1.1
    x = x + 0.02 * xd; xd = xd + 0.02 * xacc; th = th + 0.02 * thd; thd = thd + 0.02 * thacc
    s2 = torch.stack([x, xd, th, thd], -1)
    done = (x.abs() > 2.4) | (th.abs() > 12 * 2 * math.pi / 360)
    return s2, done


# ----------------------------------------------------------------------------- batched MLP + Adam

@dataclass
class Params:
    W1: torch.Tensor; b1: torch.Tensor; W2: torch.Tensor; b2: torch.Tensor; W3: torch.Tensor; b3: torch.Tensor

    def tensors(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def clone(self):
        return Params(*[t.clone() for t in self.tensors()])

    def index(self, idx):
        return Params(*[t[idx].clone() for t in self.tensors()])

    @staticmethod
    def cat(ps: list["Params"]):
        return Params(*[torch.cat([p.tensors()[i] for p in ps]) for i in range(6)])


def init_params(R: int, hidden: int, gens: list[torch.Generator]) -> Params:
    def layer(i, o):
        bound = 1 / math.sqrt(i)
        W = torch.stack([(torch.rand(i, o, generator=g) * 2 - 1) * bound for g in gens])
        b = torch.stack([(torch.rand(o, generator=g) * 2 - 1) * bound for g in gens])
        return W, b
    W1, b1 = layer(OBS, hidden); W2, b2 = layer(hidden, hidden); W3, b3 = layer(hidden, ACT)
    return Params(W1, b1, W2, b2, W3, b3)


def forward(p: Params, x: torch.Tensor) -> torch.Tensor:
    """x: (R, B, OBS) -> (R, B, ACT)"""
    h = torch.relu(torch.bmm(x, p.W1) + p.b1[:, None])
    h = torch.relu(torch.bmm(h, p.W2) + p.b2[:, None])
    return torch.bmm(h, p.W3) + p.b3[:, None]


class Adam:
    def __init__(self, p: Params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.m = [torch.zeros_like(t) for t in p.tensors()]
        self.v = [torch.zeros_like(t) for t in p.tensors()]
        self.t = torch.zeros(p.W1.shape[0])
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps

    def step(self, p: Params, grads: list[torch.Tensor]):
        self.t += 1
        for k, (w, g) in enumerate(zip(p.tensors(), grads)):
            self.m[k].mul_(self.b1).add_(g, alpha=1 - self.b1)
            self.v[k].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            shape = (-1,) + (1,) * (w.dim() - 1)
            mhat = self.m[k] / (1 - self.b1 ** self.t).view(shape)
            vhat = self.v[k] / (1 - self.b2 ** self.t).view(shape)
            w.sub_(self.lr * mhat / (vhat.sqrt() + self.eps))

    def index(self, idx):
        o = Adam.__new__(Adam)
        o.m = [t[idx].clone() for t in self.m]; o.v = [t[idx].clone() for t in self.v]; o.t = self.t[idx].clone()
        o.lr, o.b1, o.b2, o.eps = self.lr, self.b1, self.b2, self.eps
        return o

    @staticmethod
    def cat(os: list["Adam"]):
        o = Adam.__new__(Adam)
        o.m = [torch.cat([x.m[k] for x in os]) for k in range(6)]; o.v = [torch.cat([x.v[k] for x in os]) for k in range(6)]
        o.t = torch.cat([x.t for x in os]); o.lr, o.b1, o.b2, o.eps = os[0].lr, os[0].b1, os[0].b2, os[0].eps
        return o


# ----------------------------------------------------------------------------- the batched learner

@dataclass
class Config:
    hidden: int = 64
    lr: float = 1e-3
    gamma: float = 0.99
    buffer: int = 10_000
    batch: int = 32
    target_every: int = 500
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_steps: int = 10_000
    learn_start: int = 500
    total_steps: int = 30_000
    eval_every: int = 2_500
    eval_episodes: int = 4


class BatchedDQN:
    def __init__(self, cfg: Config, seeds_init: list[int], seeds_env: list[int], seeds_batch: list[int]):
        self.cfg = cfg
        R = len(seeds_init)
        g_init = [torch.Generator().manual_seed(int(s)) for s in seeds_init]
        self.g_env = [torch.Generator().manual_seed(int(s)) for s in seeds_env]
        self.g_batch = [torch.Generator().manual_seed(int(s)) for s in seeds_batch]
        self.p = init_params(R, cfg.hidden, g_init)
        self.p_targ = self.p.clone()
        self.opt = Adam(self.p, lr=cfg.lr)
        self.buf_s = torch.zeros(R, cfg.buffer, OBS); self.buf_a = torch.zeros(R, cfg.buffer, dtype=torch.long)
        self.buf_r = torch.zeros(R, cfg.buffer); self.buf_s2 = torch.zeros(R, cfg.buffer, OBS); self.buf_d = torch.zeros(R, cfg.buffer)
        self.buf_n = torch.zeros(R, dtype=torch.long)       # transitions written (per run)
        self.step_count = torch.zeros(R, dtype=torch.long)  # env steps taken (per run)
        self.s = cartpole_reset(R, self.g_env)
        self.ep_t = torch.zeros(R, dtype=torch.long)
        self.ep_ret = torch.zeros(R)
        self.recent_returns = [[] for _ in range(R)]          # training-episode returns (last 20)
        self.eval_log: list[list[tuple[int, float]]] = [[] for _ in range(R)]

    @property
    def R(self):
        return self.p.W1.shape[0]

    # -- one lockstep env step + one gradient step for every run
    def step(self):
        R, cfg = self.R, self.cfg
        eps = torch.clamp(cfg.eps_start - (cfg.eps_start - cfg.eps_end) * self.step_count.float() / cfg.eps_steps, min=cfg.eps_end)
        with torch.no_grad():
            q = forward(self.p, self.s[:, None])[:, 0]
        greedy = q.argmax(-1)
        u = torch.stack([torch.rand(2, generator=g) for g in self.g_env])          # (R, 2): explore?, which action
        a = torch.where(u[:, 0] < eps, (u[:, 1] * ACT).long().clamp(max=ACT - 1), greedy)
        s2, term = cartpole_step(self.s, a)
        self.ep_t += 1; self.ep_ret += 1.0
        trunc = self.ep_t >= MAX_STEPS
        done = term | trunc
        # store
        slot = self.buf_n % cfg.buffer
        ar = torch.arange(R)
        self.buf_s[ar, slot] = self.s; self.buf_a[ar, slot] = a; self.buf_r[ar, slot] = 1.0
        self.buf_s2[ar, slot] = s2; self.buf_d[ar, slot] = term.float()          # truncation is not terminal
        self.buf_n += 1; self.step_count += 1
        # resets
        if done.any():
            idx = torch.nonzero(done).flatten().tolist()
            fresh = cartpole_reset(len(idx), [self.g_env[i] for i in idx])
            s2 = s2.clone(); s2[idx] = fresh
            for i in idx:
                self.recent_returns[i].append(float(self.ep_ret[i]))
                self.recent_returns[i] = self.recent_returns[i][-20:]
            self.ep_t[idx] = 0; self.ep_ret[idx] = 0.0
        self.s = s2
        # learn
        if int(self.step_count.min()) >= cfg.learn_start:
            self._update()
        tu = (self.step_count % cfg.target_every == 0)
        if tu.any():
            for k, (t, src) in enumerate(zip(self.p_targ.tensors(), self.p.tensors())):
                t[tu] = src[tu].clone()
        ev = (self.step_count % cfg.eval_every == 0)
        if ev.any():
            scores = self.evaluate()
            for i in torch.nonzero(ev).flatten().tolist():
                self.eval_log[i].append((int(self.step_count[i]), scores[i]))

    def _update(self):
        R, cfg = self.R, self.cfg
        n = torch.minimum(self.buf_n, torch.tensor(cfg.buffer))
        idx = torch.stack([(torch.rand(cfg.batch, generator=g) * n[i]).long() for i, g in enumerate(self.g_batch)])
        ar = torch.arange(R)[:, None]
        s, a, r, s2, d = self.buf_s[ar, idx], self.buf_a[ar, idx], self.buf_r[ar, idx], self.buf_s2[ar, idx], self.buf_d[ar, idx]
        with torch.no_grad():
            q2 = forward(self.p_targ, s2).max(-1).values
            y = r + cfg.gamma * (1 - d) * q2
        ps = [t.detach().requires_grad_(True) for t in self.p.tensors()]
        p = Params(*ps)
        q = forward(p, s).gather(-1, a[..., None])[..., 0]
        loss = torch.nn.functional.smooth_l1_loss(q, y, reduction="none").mean(-1).sum()   # sum over runs: independent grads
        grads = torch.autograd.grad(loss, ps)
        # per-run global-norm clip at 10
        norms = torch.sqrt(sum((g.flatten(1) ** 2).sum(1) for g in grads))
        scale = torch.clamp(10.0 / (norms + 1e-6), max=1.0)
        grads = [g * scale.view((-1,) + (1,) * (g.dim() - 1)) for g in grads]
        self.opt.step(self.p, grads)

    @torch.no_grad()
    def evaluate(self) -> list[float]:
        """Greedy policy, deterministic eval starts (fixed generator per call), eval_episodes episodes, mean return."""
        R, cfg = self.R, self.cfg
        g = [torch.Generator().manual_seed(12345 + k) for k in range(cfg.eval_episodes)]
        total = torch.zeros(R)
        for k in range(cfg.eval_episodes):
            s = cartpole_reset(1, [g[k]]).expand(R, OBS).clone()
            alive = torch.ones(R, dtype=torch.bool)
            for _ in range(MAX_STEPS):
                a = forward(self.p, s[:, None])[:, 0].argmax(-1)
                s, term = cartpole_step(s, a)
                total += alive.float()
                alive &= ~term
                if not alive.any():
                    break
        return (total / cfg.eval_episodes).tolist()

    # -- snapshot / fork
    def snapshot(self, idx: list[int]) -> dict:
        t = torch.tensor(idx)
        return dict(
            p=self.p.index(t), p_targ=self.p_targ.index(t), opt=self.opt.index(t),
            buf=[b[t].clone() for b in (self.buf_s, self.buf_a, self.buf_r, self.buf_s2, self.buf_d)],
            buf_n=self.buf_n[t].clone(), step_count=self.step_count[t].clone(), s=self.s[t].clone(),
            ep_t=self.ep_t[t].clone(), ep_ret=self.ep_ret[t].clone(),
            g_env=[self.g_env[i].get_state().clone() for i in idx], g_batch=[self.g_batch[i].get_state().clone() for i in idx],
            recent=[list(self.recent_returns[i]) for i in idx], eval_log=[list(self.eval_log[i]) for i in idx],
        )

    @classmethod
    def from_snapshots(cls, cfg: Config, snaps: list[dict], reps: list[int], reseed_env: list[int | None], reseed_batch: list[int | None]):
        """Build a batched learner whose runs are children: snaps[j] repeated reps[j] times.
        reseed_env/reseed_batch are flat lists (one per child): None = copy the parent's generator state."""
        o = cls.__new__(cls)
        o.cfg = cfg
        o.p = Params.cat([Params(*[t.repeat((r,) + (1,) * (t.dim() - 1)) for t in s["p"].tensors()]) for s, r in zip(snaps, reps)])
        o.p_targ = Params.cat([Params(*[t.repeat((r,) + (1,) * (t.dim() - 1)) for t in s["p_targ"].tensors()]) for s, r in zip(snaps, reps)])
        opts = []
        for s, r in zip(snaps, reps):
            a = Adam.__new__(Adam)
            a.m = [t.repeat((r,) + (1,) * (t.dim() - 1)) for t in s["opt"].m]; a.v = [t.repeat((r,) + (1,) * (t.dim() - 1)) for t in s["opt"].v]
            a.t = s["opt"].t.repeat(r); a.lr, a.b1, a.b2, a.eps = s["opt"].lr, s["opt"].b1, s["opt"].b2, s["opt"].eps
            opts.append(a)
        o.opt = Adam.cat(opts)
        bufs = list(zip(*[[b.repeat((r,) + (1,) * (b.dim() - 1)) for b in s["buf"]] for s, r in zip(snaps, reps)]))
        o.buf_s, o.buf_a, o.buf_r, o.buf_s2, o.buf_d = [torch.cat(list(b)) for b in bufs]
        o.buf_n = torch.cat([s["buf_n"].repeat(r) for s, r in zip(snaps, reps)])
        o.step_count = torch.cat([s["step_count"].repeat(r) for s, r in zip(snaps, reps)])
        o.s = torch.cat([s["s"].repeat(r, 1) for s, r in zip(snaps, reps)])
        o.ep_t = torch.cat([s["ep_t"].repeat(r) for s, r in zip(snaps, reps)])
        o.ep_ret = torch.cat([s["ep_ret"].repeat(r) for s, r in zip(snaps, reps)])
        o.g_env, o.g_batch, o.recent_returns, o.eval_log = [], [], [], []
        j = 0
        for s, r in zip(snaps, reps):
            for _ in range(r):
                ge = torch.Generator(); gb = torch.Generator()
                if reseed_env[j] is None:
                    ge.set_state(s["g_env"][0])
                else:
                    ge.manual_seed(int(reseed_env[j]))
                if reseed_batch[j] is None:
                    gb.set_state(s["g_batch"][0])
                else:
                    gb.manual_seed(int(reseed_batch[j]))
                o.g_env.append(ge); o.g_batch.append(gb)
                o.recent_returns.append(list(s["recent"][0])); o.eval_log.append(list(s["eval_log"][0]))
                j += 1
        return o

    def reset_last_layer(self, idx: list[int]):
        """Re-initialise the output layer of the chosen runs (Nikishin-style partial reset), fresh Adam state for it."""
        t = torch.tensor(idx)
        gens = [torch.Generator().manual_seed(777 + i) for i in idx]
        bound = 1 / math.sqrt(self.cfg.hidden)
        self.p.W3[t] = torch.stack([(torch.rand(self.cfg.hidden, ACT, generator=g) * 2 - 1) * bound for g in gens])
        self.p.b3[t] = torch.stack([(torch.rand(ACT, generator=g) * 2 - 1) * bound for g in gens])
        self.p_targ.W3[t] = self.p.W3[t].clone(); self.p_targ.b3[t] = self.p.b3[t].clone()
        for k in (4, 5):
            self.opt.m[k][t] = 0.0; self.opt.v[k][t] = 0.0

    def final_score(self, i: int, last_k: int = 3) -> float:
        ev = [v for _, v in self.eval_log[i]]
        return float(np.mean(ev[-last_k:])) if ev else float("nan")
