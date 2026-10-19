"""BatchedDQN with a per-run loss (0 = Huber kappa=1, 1 = squared) and R11's failure-Q probe."""

from __future__ import annotations

import torch

from . import dqn_batched as D
from .dqn_batched import ACT, OBS, Params, cartpole_reset, cartpole_step, forward


def make_probe_set(n: int, seed: int = 31337) -> tuple[torch.Tensor, torch.Tensor]:
    """Random-policy CartPole; collect (s, a) with cartpole_step(s, a) terminal, so Q*(s, a) = 1 exactly."""
    g = torch.Generator().manual_seed(seed)
    S, A = [], []
    s = cartpole_reset(1, [g]); t = 0
    while len(S) < n:
        a = (torch.rand(1, generator=g) * ACT).long()
        s2, term = cartpole_step(s, a); t += 1
        if bool(term[0]):
            S.append(s[0].clone()); A.append(a[0].clone())
        if bool(term[0]) or t >= D.MAX_STEPS:
            s = cartpole_reset(1, [g]); t = 0
        else:
            s = s2
    return torch.stack(S), torch.stack(A)


class LossDQN(D.BatchedDQN):
    def __init__(self, cfg: D.Config, seeds_init, seeds_env, seeds_batch, loss_kind: list[int], probe_n: int = 256):
        super().__init__(cfg, seeds_init, seeds_env, seeds_batch)
        self.kind = torch.tensor(loss_kind).float()          # 0 huber, 1 mse
        self.probe_s, self.probe_a = make_probe_set(probe_n)
        self.probe_log: list[list[tuple[int, float, float]]] = [[] for _ in range(self.R)]   # (step, failure-Q mean, mean |clipped frac|)
        self.clip_frac = torch.zeros(self.R); self.clip_n = 0

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
        huber = torch.nn.functional.smooth_l1_loss(q, y, reduction="none").mean(-1)
        mse = 0.5 * ((q - y) ** 2).mean(-1)
        loss = (huber * (1 - self.kind) + mse * self.kind).sum()
        grads = torch.autograd.grad(loss, ps)
        with torch.no_grad():
            self.clip_frac += ((q - y).abs() > 1.0).float().mean(-1); self.clip_n += 1
        norms = torch.sqrt(sum((g.flatten(1) ** 2).sum(1) for g in grads))
        scale = torch.clamp(10.0 / (norms + 1e-6), max=1.0)
        grads = [g * scale.view((-1,) + (1,) * (g.dim() - 1)) for g in grads]
        self.opt.step(self.p, grads)

    @torch.no_grad()
    def probe(self):
        R = self.R
        q = forward(self.p, self.probe_s[None].expand(R, -1, -1)).gather(-1, self.probe_a[None].expand(R, -1)[..., None])[..., 0]
        cf = self.clip_frac / max(1, self.clip_n)
        for i in range(R):
            self.probe_log[i].append((int(self.step_count[i]), float(q[i].mean()), float(cf[i])))
        self.clip_frac.zero_(); self.clip_n = 0
