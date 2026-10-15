"""Anytime-valid confidence sequence for a bounded mean, by betting.

For observations in [0, 1] and a candidate mean m, each fixed betting fraction
lambda in a safe range makes

    M_t(m) = prod_i (1 + lambda * (x_i - m))

a test martingale for H0: E[x] = m (Waudby-Smith & Ramdas, betting CSs).
A uniform mixture over a fixed lambda grid is still a test martingale, and so
is one extra predictable lambda, chosen from the observations seen so far and
clipped so the capital cannot go negative. Ville's inequality gives

    P(exists t : M_t(m) >= 1/alpha) <= alpha.

The confidence sequence at time t is every m whose mixture wealth is still
below 1/alpha. Intersecting with earlier intervals keeps the sequence nested
and remains valid.

This is the drop-in tool for S4-46. Feed it paired differences rescaled to
[0, 1] and stop when the interval excludes the null mean.
"""

from __future__ import annotations

import numpy as np


class BettingCS:
    def __init__(self, alpha: float = 0.05, n_grid: int = 201, n_lambda: int = 9):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        self.alpha = float(alpha)
        self.grid = np.linspace(0.0, 1.0, n_grid)
        self.n_lambda = n_lambda
        fractions = np.linspace(-0.85, 0.85, n_lambda)
        self.fractions = fractions[np.abs(fractions) > 1e-8]
        scale = np.maximum(np.maximum(self.grid, 1.0 - self.grid), 1e-3)
        # |lambda| <= 0.9 / max(m, 1-m) keeps 1 + lambda (x - m) >= 0.1 on [0, 1].
        self.bounds = 0.9 / scale
        self.lambdas = self.fractions[None, :] / scale[:, None]
        self.log_wealth = np.zeros((n_grid, self.fractions.size), dtype=np.float64)
        self.pred_log = np.zeros(n_grid, dtype=np.float64)
        self.n = 0
        self.total = 0.0
        self.sumsq = 0.0
        self._lower = 0.0
        self._upper = 1.0

    def update(self, x: float) -> tuple[float, float]:
        if not 0.0 <= x <= 1.0:
            raise ValueError(f"observation {x} is outside [0, 1]; rescale first")
        x = float(x)
        if self.n == 0:
            predictable = np.zeros_like(self.grid)
        else:
            mu = self.total / self.n
            moment = self.sumsq / self.n - 2.0 * self.grid * mu + self.grid**2
            raw = (mu - self.grid) / np.maximum(moment, 1e-4)
            predictable = np.clip(raw, -self.bounds, self.bounds)
        centered = x - self.grid
        self.pred_log += np.log(1.0 + predictable * centered)
        self.log_wealth += np.log(1.0 + self.lambdas * centered[:, None])
        self.n += 1
        self.total += x
        self.sumsq += x * x
        lower, upper = self._interval_now()
        self._lower = max(self._lower, lower)
        self._upper = min(self._upper, upper)
        if self._lower > self._upper:
            self._lower, self._upper = lower, upper
        return self._lower, self._upper

    @property
    def interval(self) -> tuple[float, float]:
        return self._lower, self._upper

    def excludes(self, value: float) -> bool:
        return value < self._lower or value > self._upper

    def _interval_now(self) -> tuple[float, float]:
        combined = np.concatenate([self.log_wealth, self.pred_log[:, None]], axis=1)
        top = combined.max(axis=1)
        mixture = top + np.log(np.exp(combined - top[:, None]).mean(axis=1))
        keep = self.grid[mixture < np.log(1.0 / self.alpha)]
        if keep.size == 0:
            point = float(self.grid[int(np.argmin(mixture))])
            return point, point
        return float(keep[0]), float(keep[-1])


def rescale_to_unit(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        raise ValueError("need high > low")
    scaled = (np.asarray(values, dtype=np.float64) - low) / (high - low)
    if np.any(scaled < -1e-6) or np.any(scaled > 1.0 + 1e-6):
        raise ValueError("values fall outside the declared [low, high] bounds")
    return np.clip(scaled, 0.0, 1.0)


def null_mean_on_unit_interval(difference_low: float, difference_high: float) -> float:
    """Where a paired difference of 0 lands after rescaling to [0, 1]."""
    return float((0.0 - difference_low) / (difference_high - difference_low))
