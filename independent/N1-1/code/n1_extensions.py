"""N1-1 extensions. Sample-based spectral damping against GTD2, TDC, ETD; theory checks;
a deep off-policy diagnostic on CartPole; the nonlinear Baird sweep with more seeds.

Parts (select with --part, default runs all):

  theory    Bendixson's inequality and the two step-size propositions, checked
            numerically on every key matrix in the sweep.
  linear    One-transition-per-step learning on trajectories of the behavior
            policy. Algorithms: TD(0), GTD2, TDC, ETD(0), and SD-TD (TD(0) with
            online Oja tracking of the top-k eigenmodes of the symmetric part of
            the sampled Jacobian; positive modes are damped). Systems: Baird,
            Baird with random rewards, Baird-structure random features, random
            softmax kernels, and on-policy chains where TD(0) is already stable.
            Reports divergence rate, RMSVE at the best step size, and the RMSVE
            of each method's fixed point.
  cartpole  DQN on CartPole without a target network, plain SGD. Logs the top
            eigenvalue of the symmetric part of the semi-gradient Jacobian and
            the curl fraction before Q-values blow up. Then the deep SD-TD
            variant (damp the tracked top mode) at the diverging learning rate.
  nonlinear SiLU value network on Baird's features, 20 seeds, two learning
            rates: no correction, hard projection of k modes re-estimated every
            50 steps (the earlier method), and tracked damping.

    python code/n1_extensions.py --part linear --quick
    python code/n1_extensions.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # code/
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

from probes.push_further import (  # noqa: E402
    damp_positive_modes,
    discrete_diverges,
    family_aliased,
    family_baird,
    family_cycle,
    family_random,
    family_self_loop,
    offpolicy_key,
    policies,
    softmax,
    stationary_of,
)

OUT = PROJECT / "results"


# ----------------------------------------------------------------------------
# Part 1: theory checks
# ----------------------------------------------------------------------------


def sym(k: np.ndarray) -> np.ndarray:
    return 0.5 * (k + k.T)


def theory_checks(rng: np.random.Generator) -> dict:
    """Bendixson: max Re lambda(K) <= lambda_max(S). Explicit bound: ||I + a K_e||_2 < 1
    when a < 2 eps / ||K_e||_2^2. Implicit bound: ||(I - a K_e)^{-1}||_2 <= (1 + 2 a eps)^{-1/2}."""
    keys = (
        family_baird(rng, 150)
        + family_random(rng, 300)
        + family_cycle(rng, 150)
        + family_self_loop(rng, 150)
        + family_aliased(rng, 150)
    )
    eps = 0.05
    bendixson_violations = 0
    explicit_violations = 0
    implicit_violations = 0
    worst_gap = 0.0
    n_unstable_s = 0
    n_unstable_k = 0
    for key in keys:
        s = sym(key)
        lam_s = float(np.linalg.eigvalsh(s).max())
        re_k = float(np.real(np.linalg.eigvals(key)).max())
        if re_k > lam_s + 1e-9:
            bendixson_violations += 1
        worst_gap = max(worst_gap, re_k - lam_s)
        n_unstable_s += int(lam_s > 0)
        n_unstable_k += int(re_k > 0)
        damped = damp_positive_modes(key, margin=eps)
        norm_sq = float(np.linalg.norm(damped, 2) ** 2)
        if norm_sq < 1e-12:
            continue
        # Realized margin: damping only the positive modes leaves eigenvalues in
        # (-eps, 0] alone, so the margin can be smaller than eps (or zero).
        margin = max(-float(np.linalg.eigvalsh(sym(damped)).max()), 0.0)
        for alpha in (0.05, 0.5):
            explicit = np.linalg.norm(np.eye(len(key)) + alpha * damped, 2)
            bound = math.sqrt(max(1.0 - 2.0 * alpha * margin + alpha**2 * norm_sq, 0.0))
            if explicit > bound + 1e-9:
                explicit_violations += 1
        for alpha in (0.05, 0.5, 5.0):
            resolvent = np.linalg.inv(np.eye(len(key)) - alpha * damped)
            bound = (1.0 + 2.0 * alpha * margin) ** -0.5
            if np.linalg.norm(resolvent, 2) > bound + 1e-9:
                implicit_violations += 1
    result = {
        "systems": len(keys),
        "unstable_symmetric_part": n_unstable_s,
        "unstable_field": n_unstable_k,
        "bendixson_violations": bendixson_violations,
        "max(ReK - lamS)": worst_gap,
        "explicit_bound_violations": explicit_violations,
        "implicit_bound_violations": implicit_violations,
    }
    print("theory checks:", json.dumps(result, indent=1))
    return result


# ----------------------------------------------------------------------------
# Part 2: sample-based linear algorithms
# ----------------------------------------------------------------------------


@dataclass
class LinearSystem:
    name: str
    phi: np.ndarray  # (n_states, d)
    transitions: np.ndarray  # (n_states, n_actions, n_states)
    rewards: np.ndarray  # (n_states, n_actions)
    target: np.ndarray  # (n_states, n_actions)
    behavior: np.ndarray  # (n_states, n_actions)
    gamma: float
    key: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    d_mu: np.ndarray = field(init=False)
    v_pi: np.ndarray = field(init=False)
    w_init: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.key = offpolicy_key(self.phi, self.transitions, self.target, self.behavior, self.gamma)
        self.d_mu = stationary_of(self.behavior, self.transitions)
        r_pi = np.einsum("sa,sa->s", self.target, self.rewards)
        p_pi = np.einsum("sa,sap->sp", self.target, self.transitions)
        self.v_pi = np.linalg.solve(np.eye(len(r_pi)) - self.gamma * p_pi, r_pi)
        self.b = (self.phi * (self.d_mu * r_pi)[:, None]).sum(axis=0)
        self.w_init = np.zeros(self.phi.shape[1])

    def rmsve(self, w: np.ndarray) -> float:
        err = self.phi @ w - self.v_pi
        return float(np.sqrt(np.sum(self.d_mu * err**2)))

    def best_rmsve(self) -> float:
        """d_mu-weighted least squares onto the feature span."""
        sqrt_d = np.sqrt(self.d_mu)
        w, *_ = np.linalg.lstsq(self.phi * sqrt_d[:, None], self.v_pi * sqrt_d, rcond=None)
        return self.rmsve(w)

    def fixed_point_rmsve(self, operator: np.ndarray) -> float:
        """RMSVE of the solution of operator w + b = 0 (least-norm if singular)."""
        w, *_ = np.linalg.lstsq(operator, -self.b, rcond=None)
        return self.rmsve(w)

    def emphatic_key(self) -> tuple[np.ndarray, np.ndarray]:
        """Expected ETD(0) operator with interest 1: m = d_mu-weighted followon."""
        p_mu = np.einsum("sa,sap->sp", self.behavior, self.transitions)
        p_pi = np.einsum("sa,sap->sp", self.target, self.transitions)
        del p_mu
        m = np.linalg.solve(np.eye(len(self.d_mu)) - self.gamma * p_pi.T, self.d_mu)
        d = self.phi.shape[1]
        key = np.zeros((d, d))
        b = np.zeros(d)
        for s in range(len(m)):
            next_phi = p_pi[s] @ self.phi
            key += m[s] * np.outer(self.phi[s], self.gamma * next_phi - self.phi[s])
            b += m[s] * (self.target[s] @ self.rewards[s]) * self.phi[s]
        return key, b


def baird_system(rng: np.random.Generator | None, reward_scale: float) -> LinearSystem:
    from lib.baird import feature_matrix, initial_weights

    n = 7
    phi = feature_matrix()
    transitions = np.zeros((n, 2, n))
    transitions[:, 0, 6] = 1.0
    transitions[:, 1, :6] = 1.0 / 6.0
    target, behavior = policies(n, 1.0 / 7.0)
    rewards = np.zeros((n, 2))
    if reward_scale > 0 and rng is not None:
        # Scale by (1 - gamma) so the true values are O(reward_scale) rather than O(reward_scale / (1 - gamma)).
        rewards = rng.normal(size=(n, 2)) * reward_scale * (1.0 - 0.99)
    system = LinearSystem("baird" if reward_scale == 0 else "baird_reward", phi, transitions, rewards, target, behavior, 0.99)
    system.w_init = initial_weights()
    return system


def baird_structure_system(rng: np.random.Generator) -> LinearSystem:
    n_states = int(rng.choice([4, 5, 6, 7]))
    phi = rng.normal(size=(n_states, n_states + 1))
    transitions = np.zeros((n_states, 2, n_states))
    transitions[:, 0, -1] = 1.0
    transitions[:, 1, :-1] = 1.0 / (n_states - 1)
    target, behavior = policies(n_states, 1.0 / n_states)
    rewards = rng.normal(size=(n_states, 2)) * (1.0 - 0.99)
    system = LinearSystem("baird_structure", phi, transitions, rewards, target, behavior, 0.99)
    system.w_init = rng.normal(size=n_states + 1)
    return system


def random_system(rng: np.random.Generator) -> LinearSystem:
    n_states = int(rng.integers(3, 8))
    dimension = int(rng.integers(2, n_states + 2))
    phi = rng.normal(size=(n_states, dimension))
    transitions = softmax(rng.normal(size=(n_states, 2, n_states)), axis=2)
    target, behavior = policies(n_states, 0.05)
    gamma = float(rng.choice([0.9, 0.99]))
    rewards = rng.normal(size=(n_states, 2)) * (1.0 - gamma)
    system = LinearSystem("random", phi, transitions, rewards, target, behavior, gamma)
    system.w_init = rng.normal(size=dimension)
    return system


def on_policy_system(rng: np.random.Generator) -> LinearSystem:
    """Behavior equals target. TD(0) is stable; this measures the cost of a correction that is never needed."""
    n_states = int(rng.integers(4, 9))
    dimension = int(rng.integers(2, n_states))
    phi = rng.normal(size=(n_states, dimension))
    transitions = softmax(rng.normal(size=(n_states, 2, n_states)) * 2.0, axis=2)
    target = softmax(rng.normal(size=(n_states, 2)), axis=1)
    rewards = rng.normal(size=(n_states, 2)) * (1.0 - 0.95)
    system = LinearSystem("on_policy", phi, transitions, rewards, target, target.copy(), 0.95)
    system.w_init = rng.normal(size=dimension)
    return system


def sample_trajectory(system: LinearSystem, steps: int, rng: np.random.Generator):
    """States, actions, rewards, next states, and importance ratios from the behavior policy."""
    n_states, n_actions = system.behavior.shape
    cum_b = np.cumsum(system.behavior, axis=1)
    cum_p = np.cumsum(system.transitions, axis=2)
    s = int(rng.choice(n_states, p=system.d_mu))
    states = np.empty(steps, dtype=np.int64)
    actions = np.empty(steps, dtype=np.int64)
    nexts = np.empty(steps, dtype=np.int64)
    u_a = rng.random(steps)
    u_s = rng.random(steps)
    for t in range(steps):
        a = int(np.searchsorted(cum_b[s], u_a[t]))
        a = min(a, n_actions - 1)
        sp = int(np.searchsorted(cum_p[s, a], u_s[t]))
        sp = min(sp, n_states - 1)
        states[t], actions[t], nexts[t] = s, a, sp
        s = sp
    rewards = system.rewards[states, actions]
    rho = system.target[states, actions] / np.maximum(system.behavior[states, actions], 1e-12)
    return states, actions, rewards, nexts, rho


def run_algorithm(
    system: LinearSystem,
    algo: str,
    alpha: float,
    beta_ratio: float,
    steps: int,
    seed: int,
    k_modes: int = 2,
    oja_lr: float = 0.02,
    margin: float = 0.05,
    eval_every: int = 50,
) -> dict:
    rng = np.random.default_rng(seed)
    states, actions, rewards, nexts, rho = sample_trajectory(system, steps, rng)
    phi = system.phi
    gamma = system.gamma
    d = phi.shape[1]
    w = system.w_init.astype(np.float64).copy()
    w_ref = w.copy()
    h = np.zeros(d)
    beta = beta_ratio * alpha
    follow = 1.0
    prev_rho = 1.0
    # Oja state for SD-TD.
    modes = rng.normal(size=(k_modes, d))
    modes /= np.linalg.norm(modes, axis=1, keepdims=True)
    lam = np.zeros(k_modes)
    curve = []
    diverged = False
    bound = 1e3 * (1.0 + np.linalg.norm(system.v_pi))
    damped_steps = 0
    for t in range(steps):
        s, a, r, sp, p = states[t], actions[t], rewards[t], nexts[t], rho[t]
        x, xp = phi[s], phi[sp]
        delta = r + gamma * (xp @ w) - (x @ w)
        if algo == "td":
            w = w + alpha * p * delta * x
        elif algo == "gtd2":
            hx = x @ h
            w = w + alpha * p * (x - gamma * xp) * hx
            h = h + beta * (p * delta - hx) * x
        elif algo == "tdc":
            hx = x @ h
            w = w + alpha * p * (delta * x - gamma * xp * hx)
            h = h + beta * (p * delta - hx) * x
        elif algo == "etd":
            follow = 1.0 + gamma * prev_rho * follow
            w = w + alpha * follow * p * delta * x
            prev_rho = p
        elif algo in ("sdtd", "sdtd_anchor"):
            if algo == "sdtd_anchor":
                # Reference copy w_ref follows TDC (guaranteed convergent, unbiased); it does not depend on w.
                delta_ref = r + gamma * (xp @ w_ref) - (x @ w_ref)
                hx = x @ h
                w_ref = w_ref + alpha * p * (delta_ref * x - gamma * xp * hx)
                h = h + beta * (p * delta_ref - hx) * x
                target_vec = w - w_ref
            else:
                target_vec = w
            # Sampled Jacobian K_hat = rho * x (gamma xp - x)^T; symmetric part applied to each mode.
            diff = gamma * xp - x
            damp = np.zeros(d)
            for i in range(k_modes):
                u = modes[i]
                su = 0.5 * p * (x * (diff @ u) + diff * (x @ u))
                for j in range(i):
                    su = su - lam[j] * modes[j] * (modes[j] @ u)
                rayleigh = float(u @ su)
                lam[i] = (1.0 - oja_lr) * lam[i] + oja_lr * rayleigh
                u = u + oja_lr * (su - rayleigh * u)
                for j in range(i):
                    u = u - (modes[j] @ u) * modes[j]
                modes[i] = u / max(np.linalg.norm(u), 1e-12)
                if lam[i] > 0.0:
                    damp = damp + (lam[i] + margin) * modes[i] * (modes[i] @ target_vec)
            if np.any(lam > 0.0):
                damped_steps += 1
            w = w + alpha * (p * delta * x - damp)
        else:
            raise ValueError(algo)
        if t % eval_every == 0 or t == steps - 1:
            norm = float(np.linalg.norm(w))
            if not np.isfinite(norm) or norm > bound:
                diverged = True
                curve.append((t, float("nan")))
                break
            curve.append((t, system.rmsve(w)))
    final = float("nan") if diverged else system.rmsve(w)
    tail = [v for _, v in curve[-max(1, len(curve) // 5):] if np.isfinite(v)]
    return {
        "system": system.name,
        "algo": algo,
        "alpha": alpha,
        "beta_ratio": beta_ratio,
        "seed": seed,
        "diverged": int(diverged),
        "final_rmsve": final,
        "tail_rmsve": float(np.mean(tail)) if tail and not diverged else float("nan"),
        "damped_fraction": damped_steps / steps,
        "lam_top": float(lam[0]),
        "curve": curve,
    }


def _run_job(args):
    idx, system, algo, alpha, beta_ratio, steps, seed = args
    out = run_algorithm(system, algo, alpha, beta_ratio, steps, seed)
    out["system_index"] = idx
    return out


ALGO_CONFIGS = [
    ("td", 1.0),
    ("gtd2", 1.0),
    ("gtd2", 10.0),
    ("tdc", 1.0),
    ("tdc", 10.0),
    ("etd", 1.0),
    ("sdtd", 1.0),
]
ALPHAS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]


def linear_experiment(rng: np.random.Generator, quick: bool, workers: int) -> dict:
    steps = 4000 if quick else 20000
    n_seeds = 3 if quick else 6
    n_family = 6 if quick else 20
    systems: list[LinearSystem] = [baird_system(None, 0.0)]
    systems += [baird_system(rng, 1.0) for _ in range(2 if quick else 5)]
    # Keep only diverging Baird-structure systems (that is what the family is for).
    found = 0
    while found < n_family:
        system = baird_structure_system(rng)
        if discrete_diverges(system.key, 0.05):
            systems.append(system)
            found += 1
    systems += [random_system(rng) for _ in range(n_family)]
    systems += [on_policy_system(rng) for _ in range(n_family)]

    jobs = []
    for idx, system in enumerate(systems):
        for algo, beta_ratio in ALGO_CONFIGS:
            for alpha in ALPHAS:
                for seed in range(n_seeds):
                    jobs.append((idx, system, algo, alpha, beta_ratio, steps, 1000 * idx + seed))
    t0 = time.time()
    with Pool(workers) as pool:
        rows = pool.map(_run_job, jobs, chunksize=8)
    print(f"linear: {len(rows)} runs in {time.time() - t0:.0f}s")
    for row in rows:
        system = systems[row["system_index"]]
        row["expected_unstable"] = int(discrete_diverges(system.key, 0.05))
        row["s_unstable"] = int(np.linalg.eigvalsh(sym(system.key)).max() > 0)

    # Fixed-point quality per system.
    fixed_points = []
    for idx, system in enumerate(systems):
        e_key, e_b = system.emphatic_key()
        damped = damp_positive_modes(system.key, 0.05)
        row = {
            "system_index": idx,
            "system": system.name,
            "lam_max_S": float(np.linalg.eigvalsh(sym(system.key)).max()),
            "re_max_K": float(np.real(np.linalg.eigvals(system.key)).max()),
            "diverges_0.05": int(discrete_diverges(system.key, 0.05)),
            "rmsve_best": system.best_rmsve(),
            "rmsve_td_fixed_point": system.fixed_point_rmsve(system.key),
            "rmsve_damped_fixed_point": system.fixed_point_rmsve(damped),
            "rmsve_zero": system.rmsve(np.zeros(system.phi.shape[1])),
        }
        w_e, *_ = np.linalg.lstsq(e_key, -e_b, rcond=None)
        row["rmsve_etd_fixed_point"] = system.rmsve(w_e)
        fixed_points.append(row)
    return {"rows": rows, "fixed_points": fixed_points, "steps": steps, "n_seeds": n_seeds}


def summarize_linear(result: dict) -> list[dict]:
    """Best step size per (system family, algorithm) by median tail RMSVE with divergence counted as +inf."""
    rows = result["rows"]
    families = sorted({r["system"] for r in rows})
    summary = []
    for fam in families:
        fam_rows = [r for r in rows if r["system"] == fam]
        for algo, beta_ratio in ALGO_CONFIGS:
            cand = [r for r in fam_rows if r["algo"] == algo and r["beta_ratio"] == beta_ratio]
            best = None
            for alpha in ALPHAS:
                sub = [r for r in cand if r["alpha"] == alpha]
                div = float(np.mean([r["diverged"] for r in sub]))
                vals = [r["tail_rmsve"] for r in sub if r["diverged"] == 0]
                score = (div, float(np.median(vals)) if vals else float("inf"))
                if best is None or score < best[0]:
                    best = (score, alpha, div, float(np.median(vals)) if vals else float("nan"), float(np.mean(vals)) if vals else float("nan"))
            # Divergence rate across *all* step sizes too.
            div_all = float(np.mean([r["diverged"] for r in cand]))
            summary.append(
                {
                    "family": fam,
                    "algo": algo if beta_ratio == 1.0 else f"{algo}(beta={beta_ratio:g}a)",
                    "best_alpha": best[1],
                    "diverged_at_best": best[2],
                    "median_tail_rmsve_at_best": best[3],
                    "mean_tail_rmsve_at_best": best[4],
                    "diverged_over_all_alphas": div_all,
                    "n_runs": len(cand),
                }
            )
    return summary


# ----------------------------------------------------------------------------
# Shared: most unstable mode of the TD field when all we can apply is the
# symmetric Jacobian of the *loss gradient* g = -f. Unstable directions of f are
# the most negative eigenvalues of sym(Dg). Shifted power iteration finds them.
# ----------------------------------------------------------------------------


def unstable_modes_from_loss_jacobian(apply_s_grad, dim: int, k: int, iters: int, warm=None, sigma_warm=None):
    """Return (modes, field_eigenvalues, sigma_vec). field_eigenvalue > 0 means unstable.

    apply_s_grad(v) must return sym(Dg) v for the loss-gradient field g. We first
    estimate sigma = largest |eigenvalue| of sym(Dg) by power iteration, then power
    iterate on sigma*I - sym(Dg) (deflated) whose top eigenvectors are the most
    negative eigenvectors of sym(Dg), i.e. the most positive of the TD field.
    """
    import torch

    v = torch.randn(dim) if sigma_warm is None else sigma_warm.clone()
    v = v / v.norm()
    sigma = 0.0
    for _ in range(iters):
        sv = apply_s_grad(v)
        sigma = float(v @ sv)  # signed: the dominant eigenvalue of sym(Dg); positive = sharpness of the loss
        v = sv / sv.norm().clamp_min(1e-12)
    unstable_modes_from_loss_jacobian.last_sharpness = sigma
    sigma = max(abs(sigma), 1e-6) * 1.05
    modes, lams = [], []
    for i in range(k):
        u = torch.randn(dim) if warm is None else warm[i].clone()
        u = u / u.norm()
        top = 0.0
        for _ in range(iters):
            mu = sigma * u - apply_s_grad(u)
            for j in range(i):
                mu = mu - (sigma - (-lams[j])) * modes[j] * (modes[j] @ u)
            top = float(u @ mu)
            for j in range(i):
                mu = mu - (modes[j] @ mu) * modes[j]
            u = mu / mu.norm().clamp_min(1e-12)
        lam_grad = sigma - top  # eigenvalue of sym(Dg) along u
        modes.append(u.detach())
        lams.append(-lam_grad)  # eigenvalue of sym(Df), f = -g
    return modes, lams, v.detach()


# ----------------------------------------------------------------------------
# Part 3: CartPole DQN without a target network
# ----------------------------------------------------------------------------


class CartPole:
    """Standard CartPole-v1 dynamics. Observations are scaled by [2.4, 3, 0.21, 3] for the network."""

    g, mc, mp, length, force, tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    scale = np.array([2.4, 3.0, 0.21, 3.0])

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(-0.05, 0.05, size=4)

    def step(self, s: np.ndarray, a: int) -> tuple[np.ndarray, float, bool]:
        x, xd, th, thd = s
        f = self.force if a == 1 else -self.force
        ct, st = math.cos(th), math.sin(th)
        total = self.mc + self.mp
        tmp = (f + self.mp * self.length * thd**2 * st) / total
        thacc = (self.g * st - ct * tmp) / (self.length * (4.0 / 3.0 - self.mp * ct**2 / total))
        xacc = tmp - self.mp * self.length * thacc * ct / total
        s2 = np.array([x + self.tau * xd, xd + self.tau * xacc, th + self.tau * thd, thd + self.tau * thacc])
        done = bool(abs(s2[0]) > 2.4 or abs(s2[2]) > 12 * math.pi / 360)
        return s2, 1.0, done


def _cartpole_job(args):
    seed, cfg, steps = args
    return cartpole_run(seed, steps=steps, **cfg)


def _cartpole_products(net, theta, batch, gamma, probe):
    """(loss gradient, sym(Dg) probe, skew(Dg) probe) for the online semi-gradient field on this batch."""
    from lib.curl import _td_products

    s, a, r, sp, d = batch
    grad, jvp, vjp = _td_products(net, theta, s, sp, r, d, gamma, probe, a, "gather", "max", True, None)
    return grad, 0.5 * (jvp + vjp), 0.5 * (jvp - vjp)


def _cartpole_apply_factory(net, theta, batch, gamma):
    def apply(v):
        return _cartpole_products(net, theta, batch, gamma, v)[1]

    return apply


def _cartpole_curl_fraction(net, theta, batch, gamma, probes=3):
    import torch

    rot, jac = 0.0, 0.0
    for _ in range(probes):
        p = torch.randn_like(theta)
        _, sp_, ap_ = _cartpole_products(net, theta, batch, gamma, p)
        rot += float(ap_.pow(2).sum())
        jac += float((sp_ + ap_).pow(2).sum())
    return rot / max(jac, 1e-12)


def cartpole_run(seed: int, lr: float, tau: float, damp_k: int, steps: int, gamma: float = 0.99, margin: float = 0.05, measure_every: int = 100) -> dict:
    """DQN on CartPole with plain SGD. tau = 1 bootstraps from the online weights (no target network).
    damp_k > 0 tracks the k most unstable modes of the symmetric Jacobian of the online field and damps them."""
    import torch

    from lib.nets import FuncMLP

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = CartPole()
    net = FuncMLP([4, 64, 64, 2], torch.device("cpu"), seed=seed)
    theta = net.theta.detach().clone()
    target = theta.clone()
    cap = 20000
    buf_s = np.zeros((cap, 4), dtype=np.float32)
    buf_a = np.zeros(cap, dtype=np.int64)
    buf_r = np.zeros(cap, dtype=np.float32)
    buf_sp = np.zeros((cap, 4), dtype=np.float32)
    buf_d = np.zeros(cap, dtype=np.float32)
    n, ptr = 0, 0
    s = env.reset(rng)
    ep_ret, returns = 0.0, []
    modes = [torch.randn(theta.numel()) for _ in range(max(damp_k, 1))]
    lams = [0.0] * max(damp_k, 1)
    sigma_vec = torch.randn(theta.numel())
    log = []
    diverged_at = None

    def make_batch(idx):
        return (
            torch.tensor(buf_s[idx] / env.scale, dtype=torch.float32),
            torch.tensor(buf_a[idx]),
            torch.tensor(buf_r[idx]),
            torch.tensor(buf_sp[idx] / env.scale, dtype=torch.float32),
            torch.tensor(buf_d[idx]),
        )

    for t in range(steps):
        eps = max(0.05, 1.0 - t / (0.5 * steps))
        if rng.random() < eps:
            a = int(rng.integers(0, 2))
        else:
            with torch.no_grad():
                q = net.forward(theta, torch.tensor(s / env.scale, dtype=torch.float32)[None])
            a = int(q.argmax().item())
        sp, r, done = env.step(s, a)
        ep_ret += r
        truncated = ep_ret >= 500
        buf_s[ptr], buf_a[ptr], buf_r[ptr], buf_sp[ptr], buf_d[ptr] = s, a, r, sp, float(done)
        ptr = (ptr + 1) % cap
        n = min(n + 1, cap)
        if done or truncated:
            returns.append(ep_ret)
            ep_ret = 0.0
            s = env.reset(rng)
        else:
            s = sp
        if n < 256:
            continue
        batch = make_batch(rng.integers(0, n, size=64))
        boot_theta = theta if tau >= 1.0 else target
        leaf = theta.detach().requires_grad_(True)
        pred = net.forward(leaf, batch[0]).gather(1, batch[1][:, None]).squeeze(1)
        with torch.no_grad():
            boot = net.forward(boot_theta, batch[3]).max(dim=1).values
            backup = batch[2] + gamma * (1 - batch[4]) * boot
        loss = (pred - backup).pow(2).mean()
        (grad,) = torch.autograd.grad(loss, leaf)
        grad = grad.detach()
        if damp_k > 0:
            # One warm-started shifted power iteration per mode per step, on the current batch.
            new_modes, new_lams, sigma_vec = unstable_modes_from_loss_jacobian(
                _cartpole_apply_factory(net, theta, batch, gamma), theta.numel(), damp_k, iters=1, warm=modes, sigma_warm=sigma_vec
            )
            damp = torch.zeros_like(theta)
            for i in range(damp_k):
                modes[i] = new_modes[i]
                lams[i] = 0.8 * lams[i] + 0.2 * new_lams[i]
                if lams[i] > 0:
                    # Field eigenvalue > 0 is unstable. The step is theta - lr*grad with grad = -field,
                    # so damping the field by (lam+margin) u u^T theta adds that term to grad.
                    damp = damp + (lams[i] + margin) * modes[i] * (modes[i] @ theta)
            grad = grad + damp
        theta = (theta - lr * grad).detach()
        target = ((1 - tau) * target + tau * theta).detach()
        if t % measure_every == 0:
            with torch.no_grad():
                q_abs = float(net.forward(theta, batch[0]).abs().mean())
            big = make_batch(rng.integers(0, n, size=256))
            _, lam_list, _ = unstable_modes_from_loss_jacobian(_cartpole_apply_factory(net, theta, big, gamma), theta.numel(), 1, iters=10)
            sharpness = unstable_modes_from_loss_jacobian.last_sharpness
            cf = _cartpole_curl_fraction(net, theta, big, gamma)
            bad = (not np.isfinite(q_abs)) or q_abs > 1e4 or not torch.isfinite(theta).all()
            log.append(
                {
                    "step": t,
                    "q_abs": q_abs,
                    "lam_max_S": lam_list[0],
                    "sharpness": sharpness,
                    "lr_times_sharpness": lr * sharpness,
                    "curl_fraction": cf,
                    "theta_norm": float(theta.norm()),
                    "recent_return": float(np.mean(returns[-10:])) if returns else 0.0,
                    "tracked_lam": float(lams[0]) if damp_k else None,
                }
            )
            if bad:
                diverged_at = t
                break
    return {
        "seed": seed,
        "lr": lr,
        "tau": tau,
        "damp_k": damp_k,
        "gamma": gamma,
        "diverged_at": diverged_at,
        "log": log,
        "final_return": float(np.mean(returns[-10:])) if returns else 0.0,
        "first_positive_lam": next((row["step"] for row in log if row["lam_max_S"] > 0), None),
    }


CARTPOLE_CONFIGS = [
    {"lr": 1e-3, "tau": 1.0, "damp_k": 0, "gamma": 0.99},
    {"lr": 3e-3, "tau": 1.0, "damp_k": 0, "gamma": 0.99},
    {"lr": 1e-2, "tau": 1.0, "damp_k": 0, "gamma": 0.99},
    {"lr": 3e-3, "tau": 0.005, "damp_k": 0, "gamma": 0.99},
    {"lr": 3e-3, "tau": 1.0, "damp_k": 0, "gamma": 0.9},
    {"lr": 1e-2, "tau": 1.0, "damp_k": 0, "gamma": 0.9},
    {"lr": 1e-3, "tau": 1.0, "damp_k": 1, "gamma": 0.99},
    {"lr": 1e-3, "tau": 1.0, "damp_k": 2, "gamma": 0.99},
    {"lr": 3e-3, "tau": 1.0, "damp_k": 1, "gamma": 0.99},
    {"lr": 3e-3, "tau": 1.0, "damp_k": 2, "gamma": 0.99},
]


def cartpole_experiment(quick: bool, seeds: list[int], workers: int) -> dict:
    steps = 4000 if quick else 10000
    jobs = [(seed, cfg, steps) for cfg in CARTPOLE_CONFIGS for seed in seeds]
    t0 = time.time()
    with Pool(workers) as pool:
        results = pool.map(_cartpole_job, jobs, chunksize=1)
    print(f"cartpole: {len(results)} runs in {time.time() - t0:.0f}s")
    for out in results:
        last = out["log"][-1] if out["log"] else {}
        print(
            f"cartpole lr {out['lr']:g} tau {out['tau']:g} gamma {out['gamma']:g} damp_k {out['damp_k']} seed {out['seed']}: "
            f"diverged_at {out['diverged_at']} first lam_max(S)>0 at {out['first_positive_lam']} "
            f"final q_abs {last.get('q_abs', float('nan')):.2f} lr*sharpness {last.get('lr_times_sharpness', float('nan')):.2f} "
            f"return {out['final_return']:.0f}"
        )
    return {"runs": results, "steps": steps}


# ----------------------------------------------------------------------------
# Part 4: nonlinear Baird, more seeds
# ----------------------------------------------------------------------------


def nonlinear_experiment(quick: bool) -> list[dict]:
    import torch

    from lib.baird import GAMMA, feature_matrix
    from lib.nets import FuncMLP
    from probes.push_again import semi_grad, top_symmetric_modes, apply_symmetric

    torch.set_num_threads(1)
    phi = torch.tensor(feature_matrix(), dtype=torch.float32)
    n_seeds = 6 if quick else 20
    steps = 300 if quick else 600
    rows = []
    for lr in (0.01, 0.05):
        for method, k in (("none", 0), ("project_every50", 1), ("project_every50", 3), ("track_damp", 1), ("track_damp", 2), ("track_damp", 3)):
            n_stable = 0
            final_norms = []
            for seed in range(n_seeds):
                torch.manual_seed(seed)
                net = FuncMLP([8, 32, 1], torch.device("cpu"), seed=seed)
                theta = net.theta.detach().clone()
                apply = lambda v, net=net, theta=theta: apply_symmetric(net, theta, v, phi, GAMMA)  # noqa: E731
                if method == "project_every50":
                    modes = top_symmetric_modes(net, theta, phi, GAMMA, k)
                    lams = [0.0] * k
                    sigma_vec = None
                elif method == "track_damp":
                    modes, lams, sigma_vec = unstable_modes_from_loss_jacobian(apply, theta.numel(), k, iters=12)
                else:
                    modes, lams, sigma_vec = [], [], None
                diverged = None
                for step in range(1, steps + 1):
                    grad = semi_grad(net, theta, phi, GAMMA)
                    if method == "project_every50":
                        for mode in modes:
                            grad = grad - torch.dot(mode, grad) * mode
                        if step % 50 == 0:
                            modes = top_symmetric_modes(net, theta, phi, GAMMA, k)
                    elif method == "track_damp":
                        apply = lambda v, net=net, theta=theta: apply_symmetric(net, theta, v, phi, GAMMA)  # noqa: E731
                        new_modes, new_lams, sigma_vec = unstable_modes_from_loss_jacobian(
                            apply, theta.numel(), k, iters=2, warm=modes, sigma_warm=sigma_vec
                        )
                        damp = torch.zeros_like(theta)
                        for i in range(k):
                            modes[i] = new_modes[i]
                            lams[i] = 0.7 * lams[i] + 0.3 * new_lams[i]
                            if lams[i] > 0:
                                damp = damp + (lams[i] + 0.05) * modes[i] * (modes[i] @ theta)
                        grad = grad + damp
                    theta = theta - lr * grad
                    if float(theta.norm()) > 1e3 or not torch.isfinite(theta).all():
                        diverged = step
                        break
                final_norms.append(float(theta.norm()))
                n_stable += int(diverged is None)
            rows.append({"lr": lr, "method": method, "k": k, "n_seeds": n_seeds, "stable": n_stable, "median_final_norm": float(np.median(final_norms))})
            print(f"nonlinear lr {lr} {method} k={k}: stable {n_stable}/{n_seeds}  median final norm {np.median(final_norms):.1f}")
    return rows


# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", default="all", choices=["all", "theory", "linear", "cartpole", "nonlinear"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    if args.part in ("all", "theory"):
        res = theory_checks(rng)
        (OUT / "theory_checks.json").write_text(json.dumps(res, indent=1))
    if args.part in ("all", "linear"):
        res = linear_experiment(np.random.default_rng(1), args.quick, args.workers)
        summary = summarize_linear(res)
        rows_out = [{k: v for k, v in r.items() if k != "curve"} for r in res["rows"]]
        (OUT / "linear_runs.json").write_text(json.dumps(rows_out))
        curves = [{"system_index": r["system_index"], "system": r["system"], "algo": r["algo"], "beta_ratio": r["beta_ratio"], "alpha": r["alpha"], "seed": r["seed"], "curve": r["curve"]} for r in res["rows"] if r["system"] in ("baird", "baird_reward")]
        (OUT / "linear_curves_baird.json").write_text(json.dumps(curves))
        (OUT / "linear_fixed_points.json").write_text(json.dumps(res["fixed_points"], indent=1))
        (OUT / "linear_summary.json").write_text(json.dumps(summary, indent=1))
        print(f"{'family':16} {'algo':16} {'alpha*':>7} {'div@best':>9} {'med RMSVE':>10} {'div(all a)':>10}")
        for r in summary:
            print(f"{r['family']:16} {r['algo']:16} {r['best_alpha']:7.3f} {r['diverged_at_best']:9.2f} {r['median_tail_rmsve_at_best']:10.3f} {r['diverged_over_all_alphas']:10.2f}")
    if args.part in ("all", "cartpole"):
        res = cartpole_experiment(args.quick, list(range(args.seeds)), args.workers)
        (OUT / "cartpole_runs.json").write_text(json.dumps(res))
    if args.part in ("all", "nonlinear"):
        rows = nonlinear_experiment(args.quick)
        (OUT / "nonlinear_sweep.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
