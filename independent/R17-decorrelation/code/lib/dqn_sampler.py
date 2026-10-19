"""BatchedDQN with a per-run minibatch sampler and per-run target period, plus a gradient-cosine meter.

Samplers (per run, integer code):
    0 iid     : batch independent uniform draws from the buffer (standard)
    1 win4    : 4 uniform anchors, each extended to batch/4 consecutive transitions
    2 win1    : 1 uniform anchor, batch consecutive transitions
    3 recent  : the last `batch` transitions written (no reuse)
Windows wrap in the circular buffer and are clamped to written transitions.

Gradient cosine: every `cos_every` steps, cosine between the minibatch gradient actually applied and the gradient on a
fixed uniform sample of `ref_n` buffer transitions (same target network), per run.
"""

from __future__ import annotations

import torch

from . import dqn_batched as D
from .dqn_batched import Params, forward


class SamplerDQN(D.BatchedDQN):
    def __init__(self, cfg: D.Config, seeds_init, seeds_env, seeds_batch, samplers: list[int], target_every: list[int],
                 cos_every: int = 1000, ref_n: int = 1024):
        super().__init__(cfg, seeds_init, seeds_env, seeds_batch)
        R = self.R
        self.sampler = torch.tensor(samplers)
        self.K = torch.tensor(target_every)
        self.cos_every, self.ref_n = cos_every, ref_n
        self.cos_log: list[list[tuple[int, float]]] = [[] for _ in range(R)]
        self._pending_cos = False

    def _indices(self) -> torch.Tensor:
        R, cfg = self.R, self.cfg
        n = torch.minimum(self.buf_n, torch.tensor(cfg.buffer))
        B = cfg.batch
        out = torch.zeros(R, B, dtype=torch.long)
        for i, g in enumerate(self.g_batch):
            ni = int(n[i]); k = int(self.sampler[i])
            if k == 0:
                out[i] = (torch.rand(B, generator=g) * ni).long()
            elif k in (1, 2):
                nw = 4 if k == 1 else 1
                w = B // nw
                anchors = (torch.rand(nw, generator=g) * ni).long()
                # consecutive slots in write order; wrap in the circular buffer only once the buffer is full
                offs = torch.arange(w)
                idx = (anchors[:, None] + offs[None]).flatten()
                if ni < cfg.buffer:
                    idx = idx.clamp(max=ni - 1)
                else:
                    idx = idx % cfg.buffer
                out[i] = idx
            else:
                last = int(self.buf_n[i])
                idx = torch.arange(last - B, last)
                idx = idx.clamp(min=0) % cfg.buffer
                out[i] = idx
                _ = torch.rand(1, generator=g)   # keep generator streams aligned across samplers
        return out

    def _grads(self, idx: torch.Tensor):
        R, cfg = self.R, self.cfg
        ar = torch.arange(R)[:, None]
        s, a, r, s2, d = self.buf_s[ar, idx], self.buf_a[ar, idx], self.buf_r[ar, idx], self.buf_s2[ar, idx], self.buf_d[ar, idx]
        with torch.no_grad():
            q2 = forward(self.p_targ, s2).max(-1).values
            y = r + cfg.gamma * (1 - d) * q2
        ps = [t.detach().requires_grad_(True) for t in self.p.tensors()]
        p = Params(*ps)
        q = forward(p, s).gather(-1, a[..., None])[..., 0]
        loss = torch.nn.functional.smooth_l1_loss(q, y, reduction="none").mean(-1).sum()
        return list(torch.autograd.grad(loss, ps))

    def _update(self):
        R, cfg = self.R, self.cfg
        idx = self._indices()
        grads = self._grads(idx)
        if self._pending_cos:
            self._pending_cos = False
            n = torch.minimum(self.buf_n, torch.tensor(cfg.buffer))
            gref = torch.Generator().manual_seed(4242)
            ref = torch.stack([(torch.rand(self.ref_n, generator=gref) * int(n[i])).long() for i in range(R)])
            rg = self._grads(ref)
            flat = torch.cat([g.flatten(1) for g in grads], 1); rflat = torch.cat([g.flatten(1) for g in rg], 1)
            cos = (flat * rflat).sum(1) / (flat.norm(dim=1) * rflat.norm(dim=1) + 1e-12)
            for i in range(R):
                self.cos_log[i].append((int(self.step_count[i]), float(cos[i])))
        norms = torch.sqrt(sum((g.flatten(1) ** 2).sum(1) for g in grads))
        scale = torch.clamp(10.0 / (norms + 1e-6), max=1.0)
        grads = [g * scale.view((-1,) + (1,) * (g.dim() - 1)) for g in grads]
        self.opt.step(self.p, grads)

    def step(self):
        """Same as the parent's step, with a per-run target period and the cosine flag."""
        R, cfg = self.R, self.cfg
        eps = torch.clamp(cfg.eps_start - (cfg.eps_start - cfg.eps_end) * self.step_count.float() / cfg.eps_steps, min=cfg.eps_end)
        with torch.no_grad():
            q = forward(self.p, self.s[:, None])[:, 0]
        greedy = q.argmax(-1)
        u = torch.stack([torch.rand(2, generator=g) for g in self.g_env])
        a = torch.where(u[:, 0] < eps, (u[:, 1] * D.ACT).long().clamp(max=D.ACT - 1), greedy)
        s2, term = D.cartpole_step(self.s, a)
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
            fresh = D.cartpole_reset(len(idx), [self.g_env[i] for i in idx])
            s2 = s2.clone(); s2[idx] = fresh
            for i in idx:
                self.recent_returns[i].append(float(self.ep_ret[i]))
                self.recent_returns[i] = self.recent_returns[i][-20:]
            self.ep_t[idx] = 0; self.ep_ret[idx] = 0.0
        self.s = s2
        if int(self.step_count.min()) >= cfg.learn_start:
            self._pending_cos = bool((self.step_count[0] % self.cos_every) == 0)
            self._update()
        tu = (self.step_count % self.K == 0)
        if tu.any():
            for t, src in zip(self.p_targ.tensors(), self.p.tensors()):
                t[tu] = src[tu].clone()
        ev = (self.step_count % cfg.eval_every == 0)
        if ev.any():
            scores = self.evaluate()
            for i in torch.nonzero(ev).flatten().tolist():
                self.eval_log[i].append((int(self.step_count[i]), scores[i]))
