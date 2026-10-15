"""Differentiable and tabular simulators. No external assets.

All torch environments are pure functions of (state, action), so analytic
action-derivatives exist. That is the stand-in for MJX / Brax in S3-27, and
the controlled factor sweep (smoothness, contact, action dimension) in L1-29.
"""

from __future__ import annotations

import math

import torch


class Pendulum:
    """Standard torque pendulum. Smooth dynamics, smooth quadratic reward. obs is (cos, sin, thdot)."""

    obs_dim = 3
    act_dim = 1
    dt = 0.05

    def reset(self, n: int, device: torch.device) -> torch.Tensor:
        theta = (torch.rand(n, device=device) - 0.5) * 2 * math.pi
        theta_dot = (torch.rand(n, device=device) - 0.5) * 2.0
        return torch.stack([torch.cos(theta), torch.sin(theta), theta_dot], dim=-1)

    def step(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos_th, sin_th, theta_dot = obs[:, 0], obs[:, 1], obs[:, 2]
        torque = action[:, 0].clamp(-2.0, 2.0)
        gravity = 10.0
        theta_dot_next = theta_dot + self.dt * (-1.5 * gravity * sin_th + 3.0 * torque)
        theta_dot_next = theta_dot_next.clamp(-8.0, 8.0)
        delta = theta_dot_next * self.dt
        cos_next = cos_th * torch.cos(delta) - sin_th * torch.sin(delta)
        sin_next = sin_th * torch.cos(delta) + cos_th * torch.sin(delta)
        norm = (cos_next.square() + sin_next.square()).sqrt().clamp_min(1e-8)
        nxt = torch.stack([cos_next / norm, sin_next / norm, theta_dot_next], dim=-1)
        theta = torch.atan2(sin_th, cos_th)
        reward = -(theta.square() + 0.1 * theta_dot.square() + 0.001 * torque.square())
        return nxt, reward


class ContactPoint:
    """1D point mass with a soft wall. `stiffness` is the contact-richness dial.

    stiffness = 0 is free motion. Larger values add a softplus wall whose
    sharpness grows with stiffness, so the dynamics approach a hard contact.
    """

    obs_dim = 2
    act_dim = 1
    dt = 0.05
    wall = 0.2
    goal = 0.55

    def __init__(self, stiffness: float):
        self.stiffness = float(stiffness)

    def reset(self, n: int, device: torch.device) -> torch.Tensor:
        # Starts on the near side of the wall. The goal is past the wall, so a
        # stiffer contact is a different control problem from free motion.
        position = -0.05 + (torch.rand(n, device=device) - 0.5) * 0.3
        velocity = (torch.rand(n, device=device) - 0.5) * 0.4
        return torch.stack([position, velocity], dim=-1)

    def step(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        position, velocity = obs[:, 0], obs[:, 1]
        force = action[:, 0].clamp(-1.0, 1.0)
        if self.stiffness <= 0.0:
            contact = torch.zeros_like(position)
        else:
            gain = 2.0 + self.stiffness
            penetration = torch.nn.functional.softplus(gain * (position - self.wall)) / gain
            contact = -self.stiffness * penetration
        velocity_next = (velocity + self.dt * (force + contact - 0.2 * velocity)).clamp(-8.0, 8.0)
        position_next = position + self.dt * velocity_next
        nxt = torch.stack([position_next, velocity_next], dim=-1)
        reward = -((position - self.goal).square() + 0.01 * force.square())
        return nxt, reward


class LinearSystem:
    """Linear plant x' = 0.9 x + B u + noise, with optional ignored action dims.

    `reward` is "quadratic" or "threshold". Extra action dimensions past
    `true_act_dim` do not enter the dynamics, which is the action-dimension dial.
    """

    def __init__(self, obs_dim: int, act_dim: int, true_act_dim: int, reward: str, noise: float = 0.0):
        if true_act_dim > act_dim:
            raise ValueError("true_act_dim cannot exceed act_dim")
        if reward not in ("quadratic", "threshold"):
            raise ValueError(reward)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.true_act_dim = true_act_dim
        self.reward_kind = reward
        self.noise = noise

    def reset(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.randn(n, self.obs_dim, device=device) * 0.5

    def step(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        used = action[:, : self.true_act_dim].clamp(-1.0, 1.0)
        drive = torch.zeros_like(obs)
        drive[:, : self.true_act_dim] = used
        noise = 0.0
        if self.noise > 0.0:
            noise = self.noise * torch.randn_like(obs)
        nxt = 0.9 * obs + 0.3 * drive + noise
        radius = obs.square().mean(dim=-1)
        if self.reward_kind == "quadratic":
            reward = -radius - 0.01 * used.square().mean(dim=-1)
        else:
            reward = (radius < 0.25).float() - 0.01 * used.square().mean(dim=-1)
        return nxt, reward


class Chain:
    """Deterministic chain. Reward 1 only in the last state. One-hot observations.

    Right moves toward the goal, left moves away. Used for TD curl and sharpness,
    where a tiny exact MDP is what makes divergence attributable to bootstrapping.
    """

    def __init__(self, n_states: int = 8):
        self.n_states = n_states
        self.obs_dim = n_states
        self.n_actions = 2

    def one_hot(self, states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(states.long(), self.n_states).float()

    def step(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        delta = torch.where(actions.long() == 1, torch.ones_like(states), -torch.ones_like(states))
        nxt = (states + delta).clamp(0, self.n_states - 1)
        reward = (nxt == self.n_states - 1).float()
        done = nxt == self.n_states - 1
        return nxt, reward, done
