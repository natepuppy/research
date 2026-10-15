"""Tabular gridworld used as the seed pool for anytime-valid comparisons.

No gym install. One seed is one randomly initialized Q-learning run; its score
is the mean return of greedy evaluation episodes after training. Slip makes the
score noisy, which is the variance the sequential test has to live with.
"""

from __future__ import annotations

import numpy as np

MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


def return_bounds(horizon: int, step_cost: float, goal_reward: float) -> tuple[float, float]:
    return (-horizon * step_cost, goal_reward)


def run_seed(
    eps: float,
    seed: int,
    size: int = 6,
    horizon: int = 30,
    episodes: int = 40,
    slip: float = 0.15,
    step_cost: float = 0.01,
    goal_reward: float = 1.0,
    gamma: float = 0.97,
    lr: float = 0.5,
    eval_episodes: int = 8,
) -> float:
    rng = np.random.default_rng(seed)
    q = np.zeros((size, size, 4), dtype=np.float64)
    goal = (size - 1, size - 1)

    def transition(pos: tuple[int, int], action: int) -> tuple[tuple[int, int], float, bool]:
        if rng.random() < slip:
            action = int(rng.integers(0, 4))
        dr, dc = MOVES[action]
        nxt = (min(max(pos[0] + dr, 0), size - 1), min(max(pos[1] + dc, 0), size - 1))
        reward = -step_cost
        done = nxt == goal
        if done:
            reward += goal_reward
        return nxt, reward, done

    def rollout(greedy: bool) -> float:
        pos = (0, 0)
        total = 0.0
        for _ in range(horizon):
            if (not greedy) and rng.random() < eps:
                action = int(rng.integers(0, 4))
            else:
                action = int(np.argmax(q[pos]))
            nxt, reward, done = transition(pos, action)
            total += reward
            if not greedy:
                bootstrap = 0.0 if done else gamma * float(q[nxt].max())
                q[pos][action] += lr * (reward + bootstrap - q[pos][action])
            pos = nxt
            if done:
                break
        return total

    for _ in range(episodes):
        rollout(greedy=False)
    scores = [rollout(greedy=True) for _ in range(eval_episodes)]
    return float(np.mean(scores))
