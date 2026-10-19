"""Tabular OPE estimators for irregular-time logs: binned FQE, exogenous-timing FQE, and
timing-aware FQE with the 'mask the extra checks' augmentation (Robins et al. 2008 brought
into an SMDP Q-evaluation), with and without inverse-probability weights for the masked path.

All estimators are model-based tabular FQE on a coarse discretisation, so differences come
from *what MDP each one evaluates*, not from function-approximation luck.
"""

from __future__ import annotations

import numpy as np

from . import ctsim as cs


# ----------------------------------------------------------------------------- transitions

def raw_transitions(ev: cs.Events, g_edges=cs.G_EDGES) -> dict:
    s = cs.disc_state(ev.g_obs, ev.iob, ev.t, g_edges=g_edges)
    s_next = np.where(ev.done, 0, cs.disc_state(np.nan_to_num(ev.g_obs_next, nan=100.0), np.nan_to_num(ev.iob_next, nan=0.0), np.nan_to_num(ev.t_next, nan=0.0), g_edges=g_edges))
    return dict(s=s, dose=ev.dose_idx.copy(), wait=ev.wait_idx.copy(), r=ev.r_soj.copy(), tau=ev.tau.copy(),
                s_next=s_next, done=ev.done.copy(), w=np.ones(len(s)), ep=ev.ep.copy(), t=ev.t.copy(),
                g_band=np.digitize(ev.g_obs, cs.G_EDGES))


def behaviour_table(ev: cs.Events) -> np.ndarray:
    """Empirical P(dose, wait | glucose band) from the logs, Laplace-smoothed. Shape (bands, N_DOSE, N_WAIT)."""
    nb = len(cs.G_EDGES) + 1
    counts = np.ones((nb, cs.N_DOSE, cs.N_WAIT)) * 0.5
    np.add.at(counts, (np.digitize(ev.g_obs, cs.G_EDGES), ev.dose_idx, ev.wait_idx), 1.0)
    return counts / counts.sum(axis=(1, 2), keepdims=True)


def masked_transitions(ev: cs.Events, beh: np.ndarray, g_edges=cs.G_EDGES, tol: float = 0.3, ipw: bool = True, max_w: float = 50.0) -> dict:
    """Compound transitions i -> i+k (k >= 2) whose intermediate checks all gave dose 0.

    Under 'no direct effect of measuring', these are what the log would have looked like had the
    policy not checked in between, *provided* we undo the logger's selection: the logger only kept
    waiting because the intermediate readings looked fine. The IPW weight divides by the logger's
    probability of the intermediate (dose = 0, wait) decisions given what it saw.
    """
    order = np.lexsort((ev.t, ev.ep))
    ep, t, g, iob, dose, wait, r, done = (ev.ep[order], ev.t[order], ev.g_obs[order], ev.iob[order], ev.dose_idx[order],
                                          ev.wait_idx[order], ev.r_soj[order], ev.done[order])
    g_next, iob_next, t_next = ev.g_obs_next[order], ev.iob_next[order], ev.t_next[order]
    band = np.digitize(g, cs.G_EDGES)
    out = {k: [] for k in ("s", "dose", "wait", "r", "tau", "s_next", "done", "w", "ep", "t", "g_band")}
    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    ends = np.r_[starts[1:], len(ep)]
    for a, b in zip(starts, ends):
        for i in range(a, b):
            R = r[i]
            w = 1.0
            for j in range(i + 1, b):
                if dose[j] != cs.NO_DOSE_IDX:
                    break
                # j is an intermediate check we mask: refund its cost, accumulate its sojourn reward
                dt_ij = t[j] - t[i]
                R += (r[j] + cs.C_CHECK) * cs.GAMMA_H ** dt_ij
                if ipw:
                    p_stay = beh[band[j], cs.NO_DOSE_IDX, wait[j]]
                    w = min(w / max(p_stay, 1e-3), max_w)
                total = t_next[j] - t[i]
                k_w = int(np.argmin(np.abs(cs.WAITS - total)))
                if abs(cs.WAITS[k_w] - total) <= tol and k_w > wait[i]:
                    out["s"].append(cs.disc_state(np.array([g[i]]), np.array([iob[i]]), np.array([t[i]]), g_edges=g_edges)[0])
                    out["dose"].append(dose[i]); out["wait"].append(k_w); out["r"].append(R); out["tau"].append(total)
                    if done[j]:
                        out["s_next"].append(0); out["done"].append(True)
                    else:
                        out["s_next"].append(cs.disc_state(np.array([g_next[j]]), np.array([iob_next[j]]), np.array([t_next[j]]), g_edges=g_edges)[0])
                        out["done"].append(False)
                    out["w"].append(w); out["ep"].append(ep[i]); out["t"].append(t[i]); out["g_band"].append(band[i])
                if total > cs.WAITS[-1] + tol:
                    break
    return {k: np.asarray(v, dtype=(bool if k == "done" else (int if k in ("s", "dose", "wait", "s_next", "ep", "g_band") else float))) for k, v in out.items()}


def concat(*ts: dict) -> dict:
    return {k: np.concatenate([t[k] for t in ts]) for k in ts[0]}


# ----------------------------------------------------------------------------- tabular FQE

def tabular_fqe(tr: dict, n_s: int, n_a: int, a_of: np.ndarray, target_act: np.ndarray, gamma_pow: np.ndarray,
                n_iter: int = 60) -> tuple[np.ndarray, dict]:
    """Weighted tabular FQE.  a_of: action index per transition.  target_act[s]: the target's action in s.
    gamma_pow: per-transition discount applied to the bootstrap (gamma**tau).  Returns Q and support stats."""
    s, s_next, r, done, w = tr["s"], tr["s_next"], tr["r"], tr["done"], tr["w"]
    idx = s * n_a + a_of
    wsum = np.bincount(idx, weights=w, minlength=n_s * n_a)
    visited = wsum > 0
    Q = np.zeros(n_s * n_a)
    state_mean = np.zeros(n_s)
    for _ in range(n_iter):
        boot = np.where(done, 0.0, gamma_pow * Q[s_next * n_a + target_act[s_next]])
        num = np.bincount(idx, weights=w * (r + boot), minlength=n_s * n_a)
        Q_new = np.where(visited, num / np.maximum(wsum, 1e-12), 0.0)
        # unvisited (s,a): fall back to the mean of visited actions in that state (neutral extrapolation)
        Qm = Q_new.reshape(n_s, n_a)
        vm = visited.reshape(n_s, n_a)
        has = vm.any(axis=1)
        state_mean = np.where(has, (Qm * vm).sum(1) / np.maximum(vm.sum(1), 1), Q_new[visited].mean() if visited.any() else 0.0)
        Q_new = np.where(visited, Q_new, np.repeat(state_mean, n_a))
        Q = Q_new
    # support diagnostic: weight mass of target lookups that landed on unvisited (s', a*)
    look = s_next * n_a + target_act[s_next]
    unsupported = float(np.sum(w * (~done) * (~visited[look])) / max(np.sum(w * (~done)), 1e-12))
    return Q.reshape(n_s, n_a), dict(unsupported_lookup_frac=unsupported, n_visited_sa=int(visited.sum()))


def initial_value(Q: np.ndarray, tr: dict, target_act: np.ndarray) -> float:
    first = np.flatnonzero(tr["t"] == 0.0)
    s0 = tr["s"][first]
    return float(np.mean(Q[s0, target_act[s0]]))


# ----------------------------------------------------------------------------- the estimators

def _target_tables(dose_shift: int, wait_shift: int, g_edges) -> tuple[np.ndarray, np.ndarray]:
    """Target (dose, wait) per discretised state. The target reads the glucose band off the state."""
    n_s = cs.n_states(g_edges=g_edges)
    n_i, n_t = len(cs.IOB_EDGES) + 1, len(cs.T_EDGES) + 1
    gb = np.arange(n_s) // (n_i * n_t)
    # map the (possibly coarser) state glucose bin back to the behaviour band via bin centres
    centres = np.r_[g_edges[0] - 20, (g_edges[:-1] + g_edges[1:]) / 2, g_edges[-1] + 40]
    band = np.digitize(centres[gb], cs.G_EDGES)
    d_tab, w_tab = cs.target_action_table(dose_shift, wait_shift)
    return d_tab[band], w_tab[band]


def est_timing_aware(ev: cs.Events, dose_shift: int, wait_shift: int, mask: str = "none", g_edges=cs.G_EDGES) -> dict:
    """mask in {'none', 'naive', 'ipw'}."""
    tr = raw_transitions(ev, g_edges)
    if mask != "none":
        beh = behaviour_table(ev)
        tr = concat(tr, masked_transitions(ev, beh, g_edges, ipw=(mask == "ipw")))
    n_s = cs.n_states(g_edges=g_edges)
    d_t, w_t = _target_tables(dose_shift, wait_shift, g_edges)
    target_act = d_t * cs.N_WAIT + w_t
    a_of = tr["dose"] * cs.N_WAIT + tr["wait"]
    Q, stats = tabular_fqe(tr, n_s, cs.N_ACT, a_of, target_act, cs.GAMMA_H ** tr["tau"])
    return dict(value=initial_value(Q, tr, target_act), **stats, n_transitions=len(tr["s"]))


def est_exogenous(ev: cs.Events, dose_shift: int, wait_shift: int, g_edges=cs.G_EDGES) -> dict:
    """Insulin4RL-style: decision points are the logged checks, the action is the dose only, timing is the log's."""
    tr = raw_transitions(ev, g_edges)
    n_s = cs.n_states(g_edges=g_edges)
    d_t, _ = _target_tables(dose_shift, wait_shift, g_edges)
    Q, stats = tabular_fqe(tr, n_s, cs.N_DOSE, tr["dose"], d_t, cs.GAMMA_H ** tr["tau"])
    return dict(value=initial_value(Q, tr, d_t), **stats, n_transitions=len(tr["s"]))


def binned_transitions(ev: cs.Events, delta_h: float, g_edges=cs.G_EDGES) -> dict:
    """Resample every episode to a fixed grid: carry-forward observation, total dose per bin, reward per bin."""
    n_bins = int(round(cs.T_END / delta_h))
    n_ep = ev.n_episodes()
    order = np.lexsort((ev.t, ev.ep))
    ep, t, g, iob, dose = ev.ep[order], ev.t[order], ev.g_obs[order], ev.iob[order], ev.dose_idx[order]
    S = np.zeros((n_ep, n_bins), dtype=int); A = np.zeros((n_ep, n_bins), dtype=int); R = np.zeros((n_ep, n_bins))
    starts = np.flatnonzero(np.r_[True, ep[1:] != ep[:-1]])
    ends = np.r_[starts[1:], len(ep)]
    hours_per_bin = int(round(delta_h))
    for e, (a, b) in enumerate(zip(starts, ends)):
        te, ge, ie, de = t[a:b], g[a:b], iob[a:b], dose[a:b]
        for k in range(n_bins):
            t0 = k * delta_h
            j = np.searchsorted(te, t0, side="right") - 1          # last check at or before the bin start
            j = max(j, 0)
            S[e, k] = cs.disc_state(np.array([ge[j]]), np.array([ie[j] + (cs.DOSES[de[j]] if te[j] < t0 else 0.0)]), np.array([t0]), g_edges=g_edges)[0]
            in_bin = (te >= t0) & (te < t0 + delta_h)
            units = cs.DOSES[de[in_bin]].sum()
            if (de[in_bin] == cs.RESCUE_IDX).any() and units == 0.0:
                A[e, k] = cs.RESCUE_IDX
            else:
                A[e, k] = int(cs.NO_DOSE_IDX + np.argmin(np.abs(cs.DOSES[cs.NO_DOSE_IDX:] - min(units, cs.DOSES[-1]))))
            R[e, k] = ev.reward_grid[ev.ep[order][a], k * hours_per_bin:(k + 1) * hours_per_bin].sum() / cs.GAMMA_H ** t0
    s = S[:, :-1].ravel(); s_next = S[:, 1:].ravel()
    done = np.zeros_like(s, dtype=bool)
    # include the last bin as a terminal transition
    s = np.r_[s, S[:, -1]]; s_next = np.r_[s_next, np.zeros(n_ep, dtype=int)]
    a = np.r_[A[:, :-1].ravel(), A[:, -1]]; r = np.r_[R[:, :-1].ravel(), R[:, -1]]
    done = np.r_[done, np.ones(n_ep, dtype=bool)]
    tt = np.r_[np.tile(np.arange(n_bins - 1) * delta_h, n_ep), np.full(n_ep, (n_bins - 1) * delta_h)]
    return dict(s=s, dose=a, r=r, tau=np.full(len(s), delta_h), s_next=s_next, done=done, w=np.ones(len(s)), t=tt)


def est_binned(ev: cs.Events, dose_shift: int, wait_shift: int, delta_h: float, g_edges=cs.G_EDGES) -> dict:
    tr = binned_transitions(ev, delta_h, g_edges)
    n_s = cs.n_states(g_edges=g_edges)
    d_t, _ = _target_tables(dose_shift, wait_shift, g_edges)
    Q, stats = tabular_fqe(tr, n_s, cs.N_DOSE, tr["dose"], d_t, np.full(len(tr["s"]), cs.GAMMA_H ** delta_h))
    return dict(value=initial_value(Q, tr, d_t), **stats, n_transitions=len(tr["s"]))
