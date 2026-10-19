"""Batched DQN with (a) a protected reservoir of transitions mixed into every minibatch and (b) a fixed
probe set of failure states whose true action value is known exactly.

Reservoir kinds (per run):
    0 = none         : control, every minibatch sample comes from the FIFO buffer
    1 = terminal     : the first M terminal transitions the run experiences are kept forever
    2 = random       : every 4th transition from the start is kept until M are collected (control for "old data")
Each minibatch is `batch - res_batch` FIFO samples plus `res_batch` reservoir samples (control: all FIFO).

Probe set: (s, a) pairs from a random policy where taking a in s terminates the episode. With reward 1 per
step and termination, Q*(s, a) = 1 exactly, for any policy. The mean of the network's Q on the probe set
minus 1 is the "failure-Q inflation": how far the agent has drifted from knowing that these actions end
the episode.
"""

from __future__ import annotations

import torch

from . import dqn_batched as D
from .dqn_batched import OBS, ACT, Params, forward, cartpole_reset, cartpole_step


def make_probe_set(n: int, seed: int = 31337) -> tuple[torch.Tensor, torch.Tensor]:
    """Random-policy CartPole; collect (s, a) with cartpole_step(s, a) terminal."""
    g = torch.Generator().manual_seed(seed)
    S, A = [], []
    s = cartpole_reset(1, [g])
    t = 0
    while len(S) < n:
        a = (torch.rand(1, generator=g) * ACT).long()
        s2, term = cartpole_step(s, a)
        t += 1
        if bool(term[0]):
            S.append(s[0].clone()); A.append(a[0].clone())
        if bool(term[0]) or t >= D.MAX_STEPS:
            s = cartpole_reset(1, [g]); t = 0
        else:
            s = s2
    return torch.stack(S), torch.stack(A)


class ReservoirDQN(D.BatchedDQN):
    def __init__(self, cfg: D.Config, seeds_init, seeds_env, seeds_batch, kinds: list[int], res_size: int = 256, res_batch: int = 4,
                 probe_n: int = 256):
        super().__init__(cfg, seeds_init, seeds_env, seeds_batch)
        R = self.R
        self.kind = torch.tensor(kinds)
        self.res_size, self.res_batch = res_size, res_batch
        self.res_s = torch.zeros(R, res_size, OBS); self.res_a = torch.zeros(R, res_size, dtype=torch.long)
        self.res_r = torch.zeros(R, res_size); self.res_s2 = torch.zeros(R, res_size, OBS); self.res_d = torch.zeros(R, res_size)
        self.res_n = torch.zeros(R, dtype=torch.long)
        self.probe_s, self.probe_a = make_probe_set(probe_n)
        self.probe_log: list[list[tuple[int, float, float]]] = [[] for _ in range(R)]   # (step, failure-Q mean, buffer terminal fraction)

    # Capture transitions into the reservoir right after the parent stores them.
    def step(self):
        R, cfg = self.R, self.cfg
        pre_n = self.buf_n.clone()
        super().step()
        slot = pre_n % cfg.buffer
        ar = torch.arange(R)
        s, a, r, s2, d = self.buf_s[ar, slot], self.buf_a[ar, slot], self.buf_r[ar, slot], self.buf_s2[ar, slot], self.buf_d[ar, slot]
        want = torch.zeros(R, dtype=torch.bool)
        want |= (self.kind == 1) & (d > 0.5)
        want |= (self.kind == 2) & (pre_n % 4 == 0)
        want &= self.res_n < self.res_size
        if want.any():
            idx = torch.nonzero(want).flatten()
            rs = self.res_n[idx]
            self.res_s[idx, rs] = s[idx]; self.res_a[idx, rs] = a[idx]; self.res_r[idx, rs] = r[idx]
            self.res_s2[idx, rs] = s2[idx]; self.res_d[idx, rs] = d[idx]
            self.res_n[idx] += 1
        if int(self.step_count[0]) % cfg.eval_every == 0:
            self._probe()

    @torch.no_grad()
    def _probe(self):
        R = self.R
        q = forward(self.p, self.probe_s[None].expand(R, -1, -1)).gather(-1, self.probe_a[None].expand(R, -1)[..., None])[..., 0]
        n = torch.minimum(self.buf_n, torch.tensor(self.cfg.buffer)).float()
        term_frac = self.buf_d.sum(1) / n
        for i in range(R):
            self.probe_log[i].append((int(self.step_count[i]), float(q[i].mean()), float(term_frac[i])))

    def _update(self):
        R, cfg = self.R, self.cfg
        n = torch.minimum(self.buf_n, torch.tensor(cfg.buffer))
        nb = cfg.batch - self.res_batch
        idx = torch.stack([(torch.rand(cfg.batch, generator=g) * n[i]).long() for i, g in enumerate(self.g_batch)])
        ar = torch.arange(R)[:, None]
        s, a, r, s2, d = self.buf_s[ar, idx], self.buf_a[ar, idx], self.buf_r[ar, idx], self.buf_s2[ar, idx], self.buf_d[ar, idx]
        # replace the last res_batch samples with reservoir samples for runs that have a (non-empty) reservoir
        use = (self.kind > 0) & (self.res_n > 0)
        if use.any():
            ridx = torch.stack([(torch.rand(self.res_batch, generator=g) * max(int(self.res_n[i]), 1)).long() for i, g in enumerate(self.g_batch)])
            rs, ra, rr, rs2, rd = self.res_s[ar, ridx], self.res_a[ar, ridx], self.res_r[ar, ridx], self.res_s2[ar, ridx], self.res_d[ar, ridx]
            m = use[:, None]
            s = torch.cat([s[:, :nb], torch.where(m[..., None], rs, s[:, nb:])], 1)
            a = torch.cat([a[:, :nb], torch.where(m, ra, a[:, nb:])], 1)
            r = torch.cat([r[:, :nb], torch.where(m, rr, r[:, nb:])], 1)
            s2 = torch.cat([s2[:, :nb], torch.where(m[..., None], rs2, s2[:, nb:])], 1)
            d = torch.cat([d[:, :nb], torch.where(m, rd, d[:, nb:])], 1)
        with torch.no_grad():
            q2 = forward(self.p_targ, s2).max(-1).values
            y = r + cfg.gamma * (1 - d) * q2
        ps = [t.detach().requires_grad_(True) for t in self.p.tensors()]
        p = Params(*ps)
        q = forward(p, s).gather(-1, a[..., None])[..., 0]
        loss = torch.nn.functional.smooth_l1_loss(q, y, reduction="none").mean(-1).sum()
        grads = torch.autograd.grad(loss, ps)
        norms = torch.sqrt(sum((g.flatten(1) ** 2).sum(1) for g in grads))
        scale = torch.clamp(10.0 / (norms + 1e-6), max=1.0)
        grads = [g * scale.view((-1,) + (1,) * (g.dim() - 1)) for g in grads]
        self.opt.step(self.p, grads)
