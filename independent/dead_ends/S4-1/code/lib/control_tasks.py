"""Three vectorised point-mass tasks for the policy-head decomposition (T1).

Each task isolates one mechanism that generative policies are credited with:

  * Reach    : 2D point mass to a goal. Control task; no head should win.
  * Fork     : same, but a wall sits between start and goal with a gap on
               either side. The optimal action at the start is bimodal, so
               this is where multimodality (GMM, flow) could matter.
  * Corridor : 8 actuators drive the 2D mass through a rank-2 matrix B. Any
               action component in the 6-dim null space of B is penalised.
               Diagonal exploration noise wastes 6/8 of its variance there,
               so this is where a correlated (low-rank) exploration
               covariance could matter.

All dynamics are pure torch functions of (obs, action), fixed horizon, no
termination. Observations are [pos, vel, goal - pos].
"""

from __future__ import annotations

import math

import torch


class PointMassTask:
    obs_dim = 6
    dt = 0.1
    damping = 0.9
    goal = (0.0, 1.0)
    horizon = 64

    def __init__(self, act_dim: int = 2):
        self.act_dim = act_dim

    def reset(self, n: int, device: torch.device) -> torch.Tensor:
        pos = torch.stack([(torch.rand(n, device=device) - 0.5) * 0.6, -1.0 + (torch.rand(n, device=device) - 0.5) * 0.2], -1)
        vel = torch.zeros(n, 2, device=device)
        return self._obs(pos, vel)

    def _obs(self, pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        goal = torch.tensor(self.goal, device=pos.device).expand_as(pos)
        return torch.cat([pos, vel, goal - pos], -1)

    def force(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (2D force, per-row action penalty)."""
        a = action.clamp(-1.0, 1.0)
        return a, 0.01 * a.square().sum(-1)

    def collide(self, pos: torch.Tensor, nxt: torch.Tensor, vel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return nxt, vel, torch.zeros(pos.shape[0], device=pos.device)

    def step(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos, vel = obs[:, :2], obs[:, 2:4]
        force, penalty = self.force(action)
        vel = (self.damping * vel + self.dt * force).clamp(-2.0, 2.0)
        nxt = pos + self.dt * vel
        nxt, vel, hit = self.collide(pos, nxt, vel)
        nxt = nxt.clamp(-1.5, 1.5)
        goal = torch.tensor(self.goal, device=pos.device)
        dist = (nxt - goal).norm(dim=-1)
        reward = -0.1 * dist + (dist < 0.15).float() - penalty - hit
        return self._obs(nxt, vel), reward


class Reach(PointMassTask):
    name = "reach"


class Fork(PointMassTask):
    """Wall on y = 0 for |x| < half_width. Crossing it is blocked and penalised."""

    name = "fork"
    half_width = 0.5

    def collide(self, pos, nxt, vel):
        crossing = (pos[:, 1] < 0.0) & (nxt[:, 1] >= 0.0) & (nxt[:, 0].abs() < self.half_width)
        nxt = nxt.clone()
        vel = vel.clone()
        nxt[crossing, 1] = -0.02
        vel[crossing] = 0.0
        return nxt, vel, 0.2 * crossing.float()


class Corridor(PointMassTask):
    """8 actuators, rank-2 effect, null-space penalty."""

    name = "corridor"
    null_penalty = 0.5

    def __init__(self, act_dim: int = 8, seed: int = 0):
        super().__init__(act_dim)
        generator = torch.Generator().manual_seed(seed)
        raw = torch.randn(act_dim, act_dim, generator=generator)
        q, _ = torch.linalg.qr(raw)
        self.B = q[:, :2].T.contiguous() * math.sqrt(act_dim / 2.0)  # (2, act_dim), rows orthogonal
        self.P_null = torch.eye(act_dim) - q[:, :2] @ q[:, :2].T

    def force(self, action):
        a = action.clamp(-1.0, 1.0)
        B = self.B.to(a.device)
        P = self.P_null.to(a.device)
        null = a @ P
        force = a @ B.T
        return force, 0.01 * a.square().sum(-1) + self.null_penalty * null.square().sum(-1)


def make_task(name: str):
    return {"reach": Reach, "fork": Fork, "corridor": Corridor}[name]()
