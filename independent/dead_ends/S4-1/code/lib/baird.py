"""Baird's counterexample, as specified in Sutton & Barto (2nd ed.), Section 11.2.

Seven states. State 6 is the lower state; states 0-5 are the upper states.
The solid action transitions to the lower state. The dashed action transitions
to a uniformly random upper state. Reward is always 0. Discount is 0.99.

The target policy always takes solid. The behavior policy takes solid with
probability 1/7 (and dashed otherwise), so the importance ratio is 7 on solid
and 0 on dashed. Features are the standard 8-dimensional encoding: upper state
i has a 2 in component i and a 1 in component 7; the lower state has a 1 in
component 6 and a 2 in component 7. Weights start at [1, 1, 1, 1, 1, 1, 10, 1].

Under the semi-gradient, those weights diverge. `expected_semi_gradient` is the
deterministic dynamic-programming form of the same update (Figure 11.2, right),
which is the vector field whose curl we measure.
"""

from __future__ import annotations

import numpy as np
import torch

LOWER = 6
SOLID_PROBABILITY = 1.0 / 7.0
GAMMA = 0.99


def feature_matrix() -> np.ndarray:
    phi = np.zeros((7, 8), dtype=np.float64)
    for state in range(LOWER):
        phi[state, state] = 2.0
        phi[state, 7] = 1.0
    phi[LOWER, 6] = 1.0
    phi[LOWER, 7] = 2.0
    return phi


def initial_weights() -> np.ndarray:
    weights = np.ones(8, dtype=np.float64)
    weights[6] = 10.0
    return weights


def _step(action_is_solid: bool, rng: np.random.Generator) -> int:
    if action_is_solid:
        return LOWER
    return int(rng.integers(0, LOWER))


def sample_update(
    weights: np.ndarray,
    rng: np.random.Generator,
    alpha: float,
    gamma: float = GAMMA,
) -> np.ndarray:
    """One semi-gradient off-policy TD(0) step, following the behavior policy."""
    phi = feature_matrix()
    state = int(rng.integers(0, 7))
    solid = bool(rng.random() < SOLID_PROBABILITY)
    nxt = _step(solid, rng)
    rho = (1.0 / SOLID_PROBABILITY) if solid else 0.0
    delta = gamma * float(phi[nxt] @ weights) - float(phi[state] @ weights)
    return weights + alpha * rho * delta * phi[state]


def expected_semi_gradient(weights: torch.Tensor, gamma: float = GAMMA) -> torch.Tensor:
    """Deterministic semi-gradient DP field. Target policy always moves to state 6.

    f(w) = (1/7) sum_s (gamma phi_6 · w - phi_s · w) phi_s
    The learning update is w <- w + alpha f(w).
    """
    phi = torch.as_tensor(feature_matrix(), dtype=weights.dtype, device=weights.device)
    lower = phi[LOWER]
    field = torch.zeros_like(weights)
    for state in range(7):
        bellman_error = gamma * torch.dot(lower, weights) - torch.dot(phi[state], weights)
        field = field + bellman_error * phi[state]
    return field / 7.0
