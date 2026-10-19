"""Preprocessing pipelines phi applied to the same raw logs, each followed by the *same* tabular
offline-RL step (model-based Q-iteration with a visit-count constraint) and the same FQE on
phi(logs). The deployed policy is then run in the simulator for its true value, and evaluated
again by timing-aware FQE on the *raw* logs.

phi = (time, state, action)
    time   : 'raw' (decision points are the logged checks, action includes the wait), 'bin1h', 'bin4h'
    state  : 'fine' (6 glucose bins), 'coarse' (3: <70, 70-180, >180), 'binary' (2: <=180, >180)
    action : 'fine' (rescue / none / 2u / 4u / 6u), 'coarse' (rescue / none / insulin=4u)
"""

from __future__ import annotations

import numpy as np

from . import ctsim as cs
from . import ope_timing as ope

G_EDGES_BY = dict(fine=cs.G_EDGES, coarse=np.array([70.0, 180.0]), binary=np.array([180.0]))
# fine dose idx (0 rescue, 1 none, 2..4 = 2/4/6 u)  ->  coarse idx (0 rescue, 1 none, 2 insulin)
DOSE_TO_COARSE = np.array([0, 1, 2, 2, 2])
COARSE_TO_DOSE = np.array([0, 1, 3])
N_DOSE_BY = dict(fine=cs.N_DOSE, coarse=3)
MIN_COUNT = 30


def dose_map(action_mode: str) -> tuple[np.ndarray, np.ndarray]:
    if action_mode == "fine":
        return np.arange(cs.N_DOSE), np.arange(cs.N_DOSE)
    return DOSE_TO_COARSE, COARSE_TO_DOSE


def phi_transitions(ev: cs.Events, time_mode: str, state_mode: str, action_mode: str) -> tuple[dict, int, int]:
    """Transitions in the fictional MDP phi(logs). Returns (transitions, n_states, n_actions);
    transitions carry 'a' (phi action index) and 'gpow' (discount for the bootstrap)."""
    g_edges = G_EDGES_BY[state_mode]
    to_c, _ = dose_map(action_mode)
    n_d = N_DOSE_BY[action_mode]
    n_s = cs.n_states(g_edges=g_edges)
    if time_mode == "raw":
        tr = ope.raw_transitions(ev, g_edges)
        tr["a"] = to_c[tr["dose"]] * cs.N_WAIT + tr["wait"]
        tr["gpow"] = cs.GAMMA_H ** tr["tau"]
        return tr, n_s, n_d * cs.N_WAIT
    delta = 1.0 if time_mode == "bin1h" else 4.0
    tr = ope.binned_transitions(ev, delta, g_edges)
    tr["a"] = to_c[tr["dose"]]
    tr["gpow"] = np.full(len(tr["s"]), cs.GAMMA_H ** delta)
    return tr, n_s, n_d


def q_iteration(tr: dict, n_s: int, n_a: int, n_iter: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Model-based Q-iteration on the fictional MDP. Actions seen fewer than MIN_COUNT times in a
    state are not allowed; a state with no allowed action uses its most frequent logged action."""
    s, a, r, s_next, done, gpow = tr["s"], tr["a"], tr["r"], tr["s_next"], tr["done"], tr["gpow"]
    idx = s * n_a + a
    cnt = np.bincount(idx, minlength=n_s * n_a).reshape(n_s, n_a)
    allowed = cnt >= MIN_COUNT
    most_freq = cnt.argmax(1)
    no_allowed = ~allowed.any(1)
    allowed[no_allowed, most_freq[no_allowed]] = True
    Q = np.zeros(n_s * n_a)
    for _ in range(n_iter):
        Qm = Q.reshape(n_s, n_a)
        V = np.where(allowed, Qm, -np.inf).max(1)
        V = np.where(np.isfinite(V), V, 0.0)
        target = r + np.where(done, 0.0, gpow * V[s_next])
        num = np.bincount(idx, weights=target, minlength=n_s * n_a)
        Q = np.where(cnt.ravel() > 0, num / np.maximum(cnt.ravel(), 1), 0.0)
    Qm = Q.reshape(n_s, n_a)
    greedy = np.where(allowed, Qm, -np.inf).argmax(1)
    return greedy, Qm


def fqe_in_phi(tr: dict, n_s: int, n_a: int, greedy: np.ndarray) -> dict:
    Q, stats = ope.tabular_fqe(tr, n_s, n_a, tr["a"], greedy, tr["gpow"])
    first = np.flatnonzero(tr["t"] == 0.0)
    s0 = tr["s"][first]
    return dict(value=float(np.mean(Q[s0, greedy[s0]])), **stats)


def deploy_policy(greedy: np.ndarray, time_mode: str, state_mode: str, action_mode: str):
    """Turn the greedy table into a simulator policy obs -> (dose_idx, wait_idx)."""
    g_edges = G_EDGES_BY[state_mode]
    _, to_f = dose_map(action_mode)
    n_d = N_DOSE_BY[action_mode]
    if time_mode == "raw":
        def pi(obs, rng):
            s = cs.disc_state(obs["g_obs"], obs["iob"], obs["t"], g_edges=g_edges)
            a = greedy[s]
            return to_f[a // cs.N_WAIT], a % cs.N_WAIT
        return pi
    wait_idx = int(np.argmin(np.abs(cs.WAITS - (1.0 if time_mode == "bin1h" else 4.0))))

    def pi_grid(obs, rng):
        s = cs.disc_state(obs["g_obs"], obs["iob"], obs["t"], g_edges=g_edges)
        return to_f[greedy[s]], np.full(len(s), wait_idx)
    return pi_grid


def raw_target_table(greedy: np.ndarray, time_mode: str, state_mode: str, action_mode: str) -> np.ndarray:
    """The deployed policy expressed as a (dose, wait) action per *fine raw* state, for raw-log FQE."""
    g_edges = G_EDGES_BY[state_mode]
    _, to_f = dose_map(action_mode)
    n_i, n_t = len(cs.IOB_EDGES) + 1, len(cs.T_EDGES) + 1
    n_fine = cs.n_states()
    fine_gb = np.arange(n_fine) // (n_i * n_t)
    rest = np.arange(n_fine) % (n_i * n_t)
    centres = np.r_[cs.G_EDGES[0] - 20, (cs.G_EDGES[:-1] + cs.G_EDGES[1:]) / 2, cs.G_EDGES[-1] + 40]
    phi_gb = np.digitize(centres[fine_gb], g_edges)
    phi_s = phi_gb * (n_i * n_t) + rest
    a = greedy[phi_s]
    if time_mode == "raw":
        return to_f[a // cs.N_WAIT] * cs.N_WAIT + (a % cs.N_WAIT)
    wait_idx = int(np.argmin(np.abs(cs.WAITS - (1.0 if time_mode == "bin1h" else 4.0))))
    return to_f[a] * cs.N_WAIT + wait_idx


def raw_fqe_of(ev: cs.Events, target_act: np.ndarray) -> dict:
    """Timing-aware FQE on the raw logs (fine discretisation) of an arbitrary deterministic (dose, wait) table."""
    tr = ope.raw_transitions(ev)
    a_of = tr["dose"] * cs.N_WAIT + tr["wait"]
    Q, stats = ope.tabular_fqe(tr, cs.n_states(), cs.N_ACT, a_of, target_act, cs.GAMMA_H ** tr["tau"])
    return dict(value=ope.initial_value(Q, tr, target_act), **stats)
