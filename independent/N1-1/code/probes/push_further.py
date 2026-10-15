"""Push the two results that still have room.

1. Off-policy linear TD, many transition structures. Split systems that diverge
   into those whose symmetric part is already unstable and those that diverge
   only because the step size meets a large imaginary part. Damp every positive
   eigenmode of the symmetric part, and compare with dropping the skew part.
   Then repeat the damping from a key matrix estimated by samples.

2. Replace the lambda mixture in the betting sequence with one predictable
   lambda. That mixture was diluting the capital. Check coverage, then seeds
   to 80% power against a t-test.

    python probes/push_further.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.anytime import BettingCS, null_mean_on_unit_interval, rescale_to_unit  # noqa: E402


def softmax(logits: np.ndarray, axis: int) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=axis, keepdims=True)


def offpolicy_key(phi: np.ndarray, transitions: np.ndarray, target: np.ndarray, behavior: np.ndarray, gamma: float) -> np.ndarray:
    n_s, n_a = behavior.shape
    behavior_kernel = np.einsum("sa,sap->sp", behavior, transitions)
    eigvals, eigvecs = np.linalg.eig(behavior_kernel.T)
    index = int(np.argmin(np.abs(eigvals - 1.0)))
    stationary = np.abs(np.real(eigvecs[:, index]))
    stationary = stationary / max(stationary.sum(), 1e-12)
    dimension = phi.shape[1]
    key = np.zeros((dimension, dimension))
    for state in range(n_s):
        for action in range(n_a):
            weight = stationary[state] * target[state, action]
            next_phi = transitions[state, action] @ phi
            key += weight * np.outer(phi[state], gamma * next_phi - phi[state])
    return key


def step_radius(operator: np.ndarray, alpha: float) -> float:
    eye = np.eye(operator.shape[0])
    return float(np.max(np.abs(np.linalg.eigvals(eye + alpha * operator))))


def discrete_diverges(operator: np.ndarray, alpha: float) -> bool:
    return step_radius(operator, alpha) > 1.0 + 1e-8


def damp_positive_modes(operator: np.ndarray, margin: float = 0.05) -> np.ndarray:
    symmetric = 0.5 * (operator + operator.T)
    evals, evecs = np.linalg.eigh(symmetric)
    damped = operator.copy()
    for value, vector in zip(evals, evecs.T):
        if value > 1e-8:
            damped -= (value + margin) * np.outer(vector, vector)
    return damped


def policies(n_states: int, solid: float) -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros((n_states, 2))
    target[:, 0] = 1.0
    behavior = np.zeros((n_states, 2))
    behavior[:, 0] = solid
    behavior[:, 1] = 1.0 - solid
    return target, behavior


def family_baird(rng: np.random.Generator, n: int) -> list[np.ndarray]:
    keys = []
    for _ in range(n):
        n_states = int(rng.choice([4, 5, 6, 7]))
        phi = rng.normal(size=(n_states, n_states + 1))
        transitions = np.zeros((n_states, 2, n_states))
        transitions[:, 0, -1] = 1.0
        transitions[:, 1, :-1] = 1.0 / (n_states - 1)
        target, behavior = policies(n_states, 1.0 / n_states)
        keys.append(offpolicy_key(phi, transitions, target, behavior, 0.99))
    return keys


def family_random(rng: np.random.Generator, n: int) -> list[np.ndarray]:
    keys = []
    for _ in range(n):
        n_states = int(rng.integers(3, 8))
        dimension = int(rng.integers(2, n_states + 2))
        phi = rng.normal(size=(n_states, dimension))
        transitions = softmax(rng.normal(size=(n_states, 2, n_states)), axis=2)
        target, behavior = policies(n_states, 0.05)
        gamma = float(rng.choice([0.9, 0.99, 0.999]))
        keys.append(offpolicy_key(phi, transitions, target, behavior, gamma))
    return keys


def family_cycle(rng: np.random.Generator, n: int) -> list[np.ndarray]:
    """Target walks a cycle. Behavior walks the other way. Features are random."""
    keys = []
    for _ in range(n):
        n_states = int(rng.integers(3, 8))
        dimension = int(rng.integers(2, n_states + 1))
        phi = rng.normal(size=(n_states, dimension))
        transitions = np.zeros((n_states, 2, n_states))
        for state in range(n_states):
            transitions[state, 0, (state + 1) % n_states] = 1.0
            transitions[state, 1, (state - 1) % n_states] = 1.0
        target, behavior = policies(n_states, 0.05)
        keys.append(offpolicy_key(phi, transitions, target, behavior, 0.99))
    return keys


def family_self_loop(rng: np.random.Generator, n: int) -> list[np.ndarray]:
    """Target jumps to one state. Behavior stays put."""
    keys = []
    for _ in range(n):
        n_states = int(rng.integers(3, 8))
        dimension = int(rng.integers(2, n_states + 1))
        phi = rng.normal(size=(n_states, dimension))
        transitions = np.zeros((n_states, 2, n_states))
        transitions[:, 0, -1] = 1.0
        for state in range(n_states):
            transitions[state, 1, state] = 1.0
        target, behavior = policies(n_states, 0.05)
        keys.append(offpolicy_key(phi, transitions, target, behavior, 0.99))
    return keys


def family_aliased(rng: np.random.Generator, n: int) -> list[np.ndarray]:
    """Binary features shared across states, Baird backups or random backups."""
    keys = []
    for _ in range(n):
        n_states = int(rng.integers(4, 8))
        dimension = max(2, n_states // 2)
        phi = rng.integers(0, 2, size=(n_states, dimension)).astype(np.float64)
        if rng.random() < 0.5:
            transitions = np.zeros((n_states, 2, n_states))
            transitions[:, 0, -1] = 1.0
            transitions[:, 1, :-1] = 1.0 / (n_states - 1)
        else:
            transitions = softmax(rng.normal(size=(n_states, 2, n_states)), axis=2)
        target, behavior = policies(n_states, 0.08)
        keys.append(offpolicy_key(phi, transitions, target, behavior, 0.99))
    return keys


def summarize_family(name: str, keys: list[np.ndarray], alpha: float) -> list[np.ndarray]:
    diverging = []
    symmetric_cause = 0
    rotation_cause = 0
    damp_fixes = 0
    skew_fixes = 0
    for key in keys:
        if not discrete_diverges(key, alpha):
            continue
        diverging.append(key)
        symmetric = 0.5 * (key + key.T)
        if discrete_diverges(symmetric, alpha):
            symmetric_cause += 1
        else:
            rotation_cause += 1
        damp_fixes += int(not discrete_diverges(damp_positive_modes(key), alpha))
        skew_fixes += int(not discrete_diverges(symmetric, alpha))
    total = len(keys)
    n_div = len(diverging)
    print(
        f"{name:12} n {total:4d}  diverge {n_div:4d}  "
        f"symmetric-cause {symmetric_cause:4d}  rotation-cause {rotation_cause:4d}  "
        f"damp-fixes {damp_fixes:4d}  skew-fixes {skew_fixes:4d}"
    )
    return diverging


def estimate_key(key_shape_source: dict, samples: int, rng: np.random.Generator) -> np.ndarray:
    phi = key_shape_source["phi"]
    transitions = key_shape_source["transitions"]
    target = key_shape_source["target"]
    behavior = key_shape_source["behavior"]
    gamma = key_shape_source["gamma"]
    stationary = key_shape_source["stationary"]
    dimension = phi.shape[1]
    estimate = np.zeros((dimension, dimension))
    flat_p = stationary / stationary.sum()
    for column in range(dimension):
        weights = np.zeros(dimension)
        weights[column] = 1.0
        acc = np.zeros(dimension)
        for _ in range(samples):
            state = int(rng.choice(phi.shape[0], p=flat_p))
            action = int(rng.choice(2, p=behavior[state]))
            rho = target[state, action] / max(behavior[state, action], 1e-8)
            nxt = int(rng.choice(phi.shape[0], p=transitions[state, action]))
            delta = rho * ((gamma * phi[nxt] - phi[state]) @ weights)
            acc += delta * phi[state]
        estimate[:, column] = acc / samples
    return estimate


def stationary_of(behavior: np.ndarray, transitions: np.ndarray) -> np.ndarray:
    kernel = np.einsum("sa,sap->sp", behavior, transitions)
    eigvals, eigvecs = np.linalg.eig(kernel.T)
    index = int(np.argmin(np.abs(eigvals - 1.0)))
    stationary = np.abs(np.real(eigvecs[:, index]))
    return stationary / max(stationary.sum(), 1e-12)


def sampled_damping(rng: np.random.Generator, alpha: float) -> None:
    """Baird-structure systems, key estimated from trajectories, then damped."""
    print("\n== damping from a sampled key, Baird transitions, random features ==")
    tried = 0
    caught = 0
    fixed = 0
    cosines = []
    while caught < 12 and tried < 80:
        tried += 1
        n_states = int(rng.choice([4, 5, 7]))
        phi = rng.normal(size=(n_states, n_states + 1))
        transitions = np.zeros((n_states, 2, n_states))
        transitions[:, 0, -1] = 1.0
        transitions[:, 1, :-1] = 1.0 / (n_states - 1)
        target, behavior = policies(n_states, 1.0 / n_states)
        gamma = 0.99
        key = offpolicy_key(phi, transitions, target, behavior, gamma)
        if not discrete_diverges(key, alpha):
            continue
        caught += 1
        pack = {
            "phi": phi,
            "transitions": transitions,
            "target": target,
            "behavior": behavior,
            "gamma": gamma,
            "stationary": stationary_of(behavior, transitions),
        }
        estimate = estimate_key(pack, samples=2500, rng=rng)
        true_s = 0.5 * (key + key.T)
        est_s = 0.5 * (estimate + estimate.T)
        _, true_vecs = np.linalg.eigh(true_s)
        _, est_vecs = np.linalg.eigh(est_s)
        true_mode = true_vecs[:, -1]
        est_mode = est_vecs[:, -1]
        cosines.append(abs(float(true_mode @ est_mode)))
        # Apply the estimated correction to the true operator.
        evals, evecs = np.linalg.eigh(est_s)
        corrected = key.copy()
        for value, vector in zip(evals, evecs.T):
            if value > 1e-8:
                corrected -= (value + 0.05) * np.outer(vector, vector)
        fixed += int(not discrete_diverges(corrected, alpha))
    print(f"diverging systems tested {caught}  stabilized by the sample estimate {fixed}")
    if cosines:
        print(f"cosine of estimated top mode with the truth  min {min(cosines):.2f}  median {float(np.median(cosines)):.2f}")


class PlugInTest:
    """Test one null mean with a single predictable betting fraction.

    lambda is chosen from past data only, so the capital is a test martingale
    and Ville's inequality applies. No mixture over unused fractions.
    """

    def __init__(self, null: float, alpha: float = 0.05, bound_fraction: float = 0.99):
        self.null = float(null)
        self.alpha = float(alpha)
        self.threshold = float(np.log(1.0 / alpha))
        scale = max(self.null, 1.0 - self.null, 1e-3)
        self.bound = bound_fraction / scale
        self.n = 0
        self.total = 0.0
        self.sumsq = 0.0
        self.log_wealth = 0.0

    def update(self, x: float) -> None:
        x = float(x)
        if self.n == 0:
            lam = 0.0
        else:
            mu = self.total / self.n
            moment = self.sumsq / self.n - 2.0 * self.null * mu + self.null**2
            raw = (mu - self.null) / max(moment, 1e-6)
            lam = float(np.clip(raw, -self.bound, self.bound))
        capital = 1.0 + lam * (x - self.null)
        self.log_wealth += float(np.log(max(capital, 1e-12)))
        self.n += 1
        self.total += x
        self.sumsq += x * x

    def rejects(self) -> bool:
        return self.log_wealth >= self.threshold


def coverage_check() -> None:
    print("\n== plug-in test coverage, alpha 0.05, 300 paths, 200 steps ==")
    rng = np.random.default_rng(0)
    alpha = 0.05
    steps = 200
    paths = 300

    for name, true_mean, draw in (
        ("bernoulli_0.10", 0.10, lambda: rng.random(steps) < 0.10),
        ("bernoulli_0.50", 0.50, lambda: rng.random(steps) < 0.50),
        ("bernoulli_0.90", 0.90, lambda: rng.random(steps) < 0.90),
    ):
        # The closure above can't see true_mean unless we bind it. Do it directly.
        misses = 0
        for _ in range(paths):
            values = draw().astype(np.float64)
            test = PlugInTest(null=true_mean, alpha=alpha)
            hit = False
            for value in values:
                test.update(float(value))
                if test.rejects():
                    hit = True
                    break
            misses += int(hit)
        print(f"{name:16} false rejections {misses / paths:.3f}  (cap {alpha:.2f})")

    misses = 0
    for _ in range(paths):
        diffs = rng.beta(12, 12, size=steps) - rng.beta(12, 12, size=steps)
        unit = rescale_to_unit(diffs, -1.0, 1.0)
        test = PlugInTest(null=0.5, alpha=alpha)
        hit = False
        for value in unit:
            test.update(float(value))
            if test.rejects():
                hit = True
                break
        misses += int(hit)
    print(f"{'beta_null':16} false rejections {misses / paths:.3f}  (cap {alpha:.2f})")
    del rate


def seeds_to_power() -> None:
    print("\n== seeds to 80% power: t-test, lambda mixture, plug-in bet ==")
    rng = np.random.default_rng(1)
    alpha = 0.05
    trials = 80
    cap = 160
    null = null_mean_on_unit_interval(-1.0, 1.0)
    print(f"{'gap':>6} {'t':>6} {'mixture':>8} {'plugin':>8} {'mix/t':>7} {'plug/t':>7}")
    for gap in (0.0, 0.08, 0.14, 0.20, 0.25):
        total = 24.0
        left_a = 0.5 * (1.0 + gap) * total
        left_b = total - left_a
        diffs = rng.beta(left_a, left_b, size=(trials, cap)) - rng.beta(left_b, left_a, size=(trials, cap))
        t_hits = np.full(trials, cap + 1)
        mix_hits = np.full(trials, cap + 1)
        plug_hits = np.full(trials, cap + 1)
        for trial in range(trials):
            series = diffs[trial]
            for n in range(2, cap + 1):
                sample = series[:n]
                if float(np.std(sample)) < 1e-8:
                    rejected = abs(float(sample.mean())) > 1e-8
                else:
                    pvalue = float(stats.ttest_1samp(sample, 0.0).pvalue)
                    rejected = pvalue == pvalue and pvalue < alpha
                if rejected:
                    t_hits[trial] = n
                    break
            unit = rescale_to_unit(series, -1.0, 1.0)
            mixture = BettingCS(alpha=alpha, n_grid=81, n_lambda=7)
            plugin = PlugInTest(null=null, alpha=alpha)
            for n, value in enumerate(unit, start=1):
                if mix_hits[trial] > cap:
                    mixture.update(float(value))
                    if mixture.excludes(null):
                        mix_hits[trial] = n
                if plug_hits[trial] > cap:
                    plugin.update(float(value))
                    if plugin.rejects():
                        plug_hits[trial] = n
                if mix_hits[trial] <= cap and plug_hits[trial] <= cap:
                    break

        def first(hits: np.ndarray) -> str:
            for n in range(2, cap + 1):
                if float(np.mean(hits <= n)) >= 0.8:
                    return str(n)
            return "none"

        t_n = first(t_hits)
        mix_n = first(mix_hits)
        plug_n = first(plug_hits)

        def ratio(a: str, b: str) -> str:
            if a == "none" or b == "none" or b == "0":
                return "na"
            return f"{int(a) / int(b):.2f}"

        print(f"{gap:6.2f} {t_n:>6} {mix_n:>8} {plug_n:>8} {ratio(mix_n, t_n):>7} {ratio(plug_n, t_n):>7}")


def main() -> None:
    rng = np.random.default_rng(0)
    alpha = 0.05
    print(f"== who diverges at step size {alpha}, and which fix works ==")
    print("damp-fixes: remove every positive eigenmode of the symmetric part")
    print("skew-fixes: drop the skew part and keep the symmetric part")
    families = {
        "baird": family_baird(rng, 200),
        "random": family_random(rng, 400),
        "cycle": family_cycle(rng, 200),
        "self-loop": family_self_loop(rng, 200),
        "aliased": family_aliased(rng, 200),
    }
    for name, keys in families.items():
        summarize_family(name, keys, alpha)
    print("\n== same families at step size 0.5 ==")
    for name, keys in families.items():
        summarize_family(name, keys, alpha=0.5)
    sampled_damping(rng, alpha)
    coverage_check()
    seeds_to_power()


if __name__ == "__main__":
    main()
