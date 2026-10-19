"""Batched DQN with per-run target period and per-run constant epsilon, plus a policy-churn meter.

Churn (Schaul et al. 2022): the fraction of a fixed probe set of states whose greedy action changes across one
gradient update. Measured on every update and averaged per evaluation interval.
"""

from __future__ import annotations

import torch

from . import dqn_batched as D
from .dqn_batched import OBS, ACT, forward, cartpole_reset, cartpole_step


def make_probe_states(n: int, seed: int = 4711) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    S = []
    s = cartpole_reset(1, [g]); t = 0
    while len(S) < n:
        a = (torch.rand(1, generator=g) * ACT).long()
        s2, term = cartpole_step(s, a); t += 1
        S.append(s[0].clone())
        if bool(term[0]) or t >= D.MAX_STEPS:
            s = cartpole_reset(1, [g]); t = 0
        else:
            s = s2
    return torch.stack(S)


class ChurnDQN(D.BatchedDQN):
    def __init__(self, cfg: D.Config, seeds_init, seeds_env, seeds_batch, target_every: list[int], eps: list[float], probe_n: int = 256):
        super().__init__(cfg, seeds_init, seeds_env, seeds_batch)
        self.K = torch.tensor(target_every)
        self.eps_run = torch.tensor(eps, dtype=torch.float32)
        self.probe = make_probe_states(probe_n)
        self.churn_acc = torch.zeros(self.R); self.churn_n = 0
        self.churn_log: list[list[tuple[int, float]]] = [[] for _ in range(self.R)]

    def step(self):
        R, cfg = self.R, self.cfg
        eps = self.eps_run
        with torch.no_grad():
            q = forward(self.p, self.s[:, None])[:, 0]
        greedy = q.argmax(-1)
        u = torch.stack([torch.rand(2, generator=g) for g in self.g_env])
        a = torch.where(u[:, 0] < eps, (u[:, 1] * ACT).long().clamp(max=ACT - 1), greedy)
        s2, term = cartpole_step(self.s, a)
        self.ep_t += 1; self.ep_ret += 1.0
        trunc = self.ep_t >= D.MAX_STEPS
        done = term | trunc
        slot = self.buf_n % cfg.buffer
        ar = torch.arange(R)
        self.buf_s[ar, slot] = self.s; self.buf_a[ar, slot] = a; self.buf_r[ar, slot] = 1.0
        self.buf_s2[ar, slot] = s2; self.buf_d[ar, slot] = term.float()
        self.buf_n += 1; self.step_count += 1
        if done.any():
            idx = torch.nonzero(done).flatten().tolist()
            fresh = cartpole_reset(len(idx), [self.g_env[i] for i in idx])
            s2 = s2.clone(); s2[idx] = fresh
            for i in idx:
                self.recent_returns[i].append(float(self.ep_ret[i])); self.recent_returns[i] = self.recent_returns[i][-20:]
            self.ep_t[idx] = 0; self.ep_ret[idx] = 0.0
        self.s = s2
        if int(self.step_count.min()) >= cfg.learn_start:
            with torch.no_grad():
                g_before = forward(self.p, self.probe[None].expand(R, -1, -1)).argmax(-1)
            self._update()
            with torch.no_grad():
                g_after = forward(self.p, self.probe[None].expand(R, -1, -1)).argmax(-1)
            self.churn_acc += (g_before != g_after).float().mean(-1); self.churn_n += 1
        tu = (self.step_count % self.K == 0)
        if tu.any():
            for t, src in zip(self.p_targ.tensors(), self.p.tensors()):
                t[tu] = src[tu].clone()
        ev = (self.step_count % cfg.eval_every == 0)
        if ev.any():
            scores = self.evaluate()
            ch = (self.churn_acc / max(self.churn_n, 1)).tolist()
            for i in torch.nonzero(ev).flatten().tolist():
                self.eval_log[i].append((int(self.step_count[i]), scores[i]))
                self.churn_log[i].append((int(self.step_count[i]), ch[i]))
            self.churn_acc.zero_(); self.churn_n = 0
