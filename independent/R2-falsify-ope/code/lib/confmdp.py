"""Finite-horizon tabular MDPs with a hidden per-episode variable u, inert states, and a
negative-control outcome, plus the standard OPE estimators and the falsification tests.

Everything is tabular so the ground-truth value of any target is exact (dynamic programming
over (s, u)) and thousands of configurations run in minutes on a laptop.

Failure modes that can be injected (one at a time):
    none         u is logged as part of the state; behaviour is estimated correctly.
    confounding  u is hidden; behaviour depends on u (confounding by indication).
    miscal       u is logged, but the behaviour model used for IS/WIS/PDIS/DR is wrong
                 (smoothed toward uniform), as when a behaviour-cloning model is poorly fit.
    abstraction  u is logged, but states are merged in groups of three before estimation (non-Markov).
    support      u is logged, but the target acts where the behaviour never did.

Falsification tests (all computed from the logs, never from the truth):
    agree    spread across estimators                      (the usual 'they agree' check)
    ess      effective sample size of the WIS weights      (the usual overlap check)
    self     model-based self-evaluation of the behaviour policy vs the empirical return
    nco      negative-control outcome: a baseline covariate no policy can change
    ncp      negative-control policy: the target perturbed only in inert states
    placebo  rewards shuffled across episodes; every policy must look the same
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

S, A, U, H = 12, 3, 2, 10
INERT = np.array([9, 10, 11])          # states where neither reward nor transition depends on the action
AGG = 3                                # 'abstraction' failure merges states in groups of this size
EPS_W = 1e-12


@dataclass
class MDP:
    P: np.ndarray        # (U, S, A, S)
    R: np.ndarray        # (U, S, A)
    p0: np.ndarray       # (S,) initial state distribution
    pu: float            # P(u = 1)
    beh: np.ndarray      # (U, S, A) behaviour policy, depends on u
    kappa: float         # confounding strength


def make_mdp(seed: int, kappa: float = 1.5, support_gap: bool = False) -> MDP:
    rng = np.random.default_rng(seed)
    base = rng.dirichlet(np.full(S, 0.3), size=(S, A))              # (S, A, S)
    # u = 1 ('frail') shifts transitions toward the low-reward half of the state space
    tilt = np.exp(-1.0 * (np.arange(S) < S // 2))                   # favour states >= S/2 when u = 0 ...
    P = np.zeros((U, S, A, S))
    P[0] = base * tilt
    P[1] = base / tilt
    for u in range(U):
        P[u] /= P[u].sum(-1, keepdims=True)
    Rsa = rng.uniform(0.0, 1.0, size=(S, A))
    R = np.stack([Rsa, Rsa - 0.5])                                   # frail patients do worse under every action
    R = np.clip(R, 0.0, 1.0)
    for s in INERT:                                                  # inert: the action does nothing here
        P[:, s, :, :] = P[:, s, :1, :]
        R[:, s, :] = R[:, s, :1]
    # behaviour: a soft preference per state plus an action tilt that depends on u (confounding by indication)
    theta = rng.normal(0.0, 0.7, size=(S, A))
    phi = np.array([1.0, 0.0, -1.0])
    beh = np.zeros((U, S, A))
    for u in range(U):
        logits = theta + kappa * (2 * u - 1) * phi
        beh[u] = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    if support_gap:                                                  # the behaviour never takes action 2 in states 0..3
        beh[:, :4, 2] = 0.0
        beh /= beh.sum(-1, keepdims=True)
    p0 = rng.dirichlet(np.ones(S))
    return MDP(P=P, R=R, p0=p0, pu=0.5, beh=beh, kappa=kappa)


def true_value(m: MDP, pi: np.ndarray) -> float:
    """Exact value of a target policy pi (S, A) that does not see u."""
    V = np.zeros((U, S))
    for _ in range(H):
        Q = m.R + np.einsum("usat,ut->usa", m.P, V)
        V = np.einsum("sa,usa->us", pi, Q)
    return float(np.sum(m.p0 * ((1 - m.pu) * V[0] + m.pu * V[1])))


def true_value_u(m: MDP, pi_u: np.ndarray) -> float:
    """Exact value of a policy that may depend on u (used for the behaviour itself)."""
    V = np.zeros((U, S))
    for _ in range(H):
        Q = m.R + np.einsum("usat,ut->usa", m.P, V)
        V = np.einsum("usa,usa->us", pi_u, Q)
    return float(np.sum(m.p0 * ((1 - m.pu) * V[0] + m.pu * V[1])))


def simulate(m: MDP, n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    u = (rng.uniform(size=n) < m.pu).astype(int)
    s = rng.choice(S, size=n, p=m.p0)
    S_, A_, R_ = np.zeros((n, H), int), np.zeros((n, H), int), np.zeros((n, H))
    for t in range(H):
        probs = m.beh[u, s]
        a = (rng.uniform(size=(n, 1)) > np.cumsum(probs, 1)).sum(1)
        a = np.minimum(a, A - 1)
        S_[:, t], A_[:, t] = s, a
        R_[:, t] = m.R[u, s, a]
        pn = m.P[u, s, a]
        s = np.minimum((rng.uniform(size=(n, 1)) > np.cumsum(pn, 1)).sum(1), S - 1)
    nco = u + rng.normal(0.0, 0.7, n)               # a baseline covariate (e.g. admission severity) correlated with u
    return dict(s=S_, a=A_, r=R_, u=u, nco=nco)


# ----------------------------------------------------------------------------- views of the logs

def observed_state(data: dict, mode: str) -> tuple[np.ndarray, int]:
    """Return the state index the estimator gets to see, and its cardinality."""
    s = data["s"]
    if mode == "confounding":
        return s, S
    if mode == "abstraction":
        return s // AGG + (data["u"][:, None] * (S // AGG)), 2 * (S // AGG)   # states merged in groups of AGG, u kept
    return s + data["u"][:, None] * S, 2 * S                        # (s, u) jointly


def lift_target(pi: np.ndarray, mode: str) -> np.ndarray:
    """Express a target policy (S, A) on the observed-state space of the estimator."""
    if mode == "confounding":
        return pi
    if mode == "abstraction":
        # the target is defined on the merged state (first state of each group decides)
        coarse = pi[::AGG]
        return np.concatenate([coarse, coarse], axis=0)
    return np.concatenate([pi, pi], axis=0)


def unlift_target(pi_obs: np.ndarray, mode: str) -> np.ndarray:
    """Policy on the true (s) space implied by a policy on the observed space (for the truth)."""
    if mode == "confounding":
        return pi_obs
    if mode == "abstraction":
        return np.repeat(pi_obs[: S // AGG], AGG, axis=0)
    return pi_obs[:S]


def estimate_behaviour(x: np.ndarray, a: np.ndarray, n_x: int, smooth: float = 0.5) -> np.ndarray:
    c = np.full((n_x, A), smooth)
    np.add.at(c, (x.ravel(), a.ravel()), 1.0)
    return c / c.sum(1, keepdims=True)


# ----------------------------------------------------------------------------- estimators

def fit_model(x: np.ndarray, a: np.ndarray, r: np.ndarray, n_x: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, as_, xn = x[:, :-1].ravel(), a[:, :-1].ravel(), x[:, 1:].ravel()
    cnt = np.zeros((n_x, A, n_x))
    np.add.at(cnt, (xs, as_, xn), 1.0)
    n_sa = cnt.sum(-1)
    rs = np.zeros((n_x, A))
    np.add.at(rs, (x.ravel(), a.ravel()), r.ravel())
    n_r = np.zeros((n_x, A))
    np.add.at(n_r, (x.ravel(), a.ravel()), 1.0)
    P_hat = np.where(n_sa[..., None] > 0, cnt / np.maximum(n_sa[..., None], 1), np.nan)
    # unvisited (x, a): use the state's average transition over visited actions (neutral), else uniform
    vis = n_sa > 0
    row_P = np.where(vis.any(1)[:, None], np.nansum(np.where(vis[..., None], P_hat, 0.0), axis=1) / np.maximum(vis.sum(1), 1)[:, None], 1.0 / n_x)
    P_hat = np.where(vis[..., None], P_hat, row_P[:, None, :])
    R_hat = np.where(n_r > 0, rs / np.maximum(n_r, 1), np.nan)
    # fall back: unvisited (x, a) take the state's mean reward over visited actions (neutral extrapolation)
    row_mean = np.nanmean(np.where(np.isnan(R_hat), np.nan, R_hat), axis=1)
    row_mean = np.where(np.isnan(row_mean), np.nanmean(R_hat), row_mean)
    R_hat = np.where(np.isnan(R_hat), row_mean[:, None], R_hat)
    return P_hat, R_hat, n_r


def fqe(P_hat, R_hat, pi_obs, x0) -> tuple[float, np.ndarray]:
    n_x = P_hat.shape[0]
    V = np.zeros(n_x)
    Qs = []
    for _ in range(H):
        Q = R_hat + P_hat @ V
        V = (pi_obs * Q).sum(1)
        Qs.append(Q)
    Qs = Qs[::-1]                                     # Qs[t] is the Q-function at step t
    return float(V[x0].mean()), np.array(Qs)


def is_family(x, a, r, pi_obs, beh_hat, Q_t=None, clip: float = 1e6) -> dict:
    n = len(x)
    rho = pi_obs[x, a] / np.maximum(beh_hat[x, a], EPS_W)
    cum = np.cumprod(rho, axis=1)
    cum = np.minimum(cum, clip)
    w_traj = cum[:, -1]
    G = r.sum(1)
    out = dict(IS=float(np.mean(w_traj * G)), WIS=float(np.sum(w_traj * G) / max(np.sum(w_traj), EPS_W)),
               PDIS=float(np.mean((cum * r).sum(1))), ess_frac=float(np.sum(w_traj) ** 2 / max(np.sum(w_traj ** 2), EPS_W) / n),
               w_traj=w_traj)
    if Q_t is not None:
        # doubly robust (per-decision, Jiang & Li 2016)
        tt = np.arange(H)
        Qsa = Q_t[tt[None, :], x, a]
        Vs = (pi_obs[x] * Q_t[tt[None, :], x]).sum(-1)
        prev = np.concatenate([np.ones((n, 1)), cum[:, :-1]], axis=1)
        dr = (prev * Vs).sum(1) + (cum * (r - Qsa)).sum(1)
        out["DR"] = float(np.mean(dr))
    return out


ESTIMATORS = ["FQE", "IS", "WIS", "PDIS", "DR"]


def all_estimates(x, a, r, n_x, pi_obs, beh_hat, x0) -> dict:
    P_hat, R_hat, _ = fit_model(x, a, r, n_x)
    v_fqe, Q_t = fqe(P_hat, R_hat, pi_obs, x0)
    isf = is_family(x, a, r, pi_obs, beh_hat, Q_t)
    return dict(FQE=v_fqe, IS=isf["IS"], WIS=isf["WIS"], PDIS=isf["PDIS"], DR=isf["DR"], ess_frac=isf["ess_frac"], w_traj=isf["w_traj"])


# ----------------------------------------------------------------------------- tests

def _se(vals: np.ndarray) -> float:
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 1.0


def boot_se(fn, n: int, rng: np.random.Generator, B: int = 25) -> float:
    vals = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return float(np.std(vals, ddof=1) + 1e-9)


def falsification_tests(x, a, r, n_x, pi_obs, pi_nc_obs, beh_hat, nco, rng, est_vals: dict) -> dict:
    """Return one statistic per test. Larger = more suspicious. All computed without the truth."""
    n = len(x)
    x0 = x[:, 0]
    G = r.sum(1)
    out = {}
    vals = np.array([est_vals[e] for e in ESTIMATORS])
    out["agree"] = float((vals.max() - vals.min()) / (abs(vals.mean()) + 1.0))
    out["ess"] = float(1.0 - est_vals["ess_frac"])                       # 1 - ESS fraction, so larger is worse
    # self-evaluation: model-based value of the (estimated) behaviour vs the empirical mean return
    P_hat, R_hat, _ = fit_model(x, a, r, n_x)
    v_self, _ = fqe(P_hat, R_hat, beh_hat, x0)
    out["self"] = abs(v_self - G.mean()) / _se(G)
    # negative-control outcome, IS form: reweighted baseline covariate vs its plain mean
    w = est_vals["w_traj"]
    nco_w = np.sum(w * nco) / max(np.sum(w), EPS_W)
    se_nco = boot_se(lambda idx: np.sum(w[idx] * nco[idx]) / max(np.sum(w[idx]), EPS_W) - nco[idx].mean(), n, rng)
    out["nco_is"] = abs(nco_w - nco.mean()) / se_nco
    # negative-control outcome, regression form: does E[nco | x0, a0] depend on a0 once we average under the target?
    cnt = np.zeros((n_x, A)); tot = np.zeros((n_x, A))
    np.add.at(cnt, (x0, a[:, 0]), 1.0); np.add.at(tot, (x0, a[:, 0]), nco)
    Y_hat = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    row = np.nanmean(Y_hat, axis=1); row = np.where(np.isnan(row), nco.mean(), row)
    Y_hat = np.where(np.isnan(Y_hat), row[:, None], Y_hat)
    v_nco_model = float(np.mean((pi_obs[x0] * Y_hat[x0]).sum(1)))

    def nco_model_boot(idx):
        c = np.zeros((n_x, A)); t_ = np.zeros((n_x, A))
        np.add.at(c, (x0[idx], a[idx, 0]), 1.0); np.add.at(t_, (x0[idx], a[idx, 0]), nco[idx])
        Yh = np.where(c > 0, t_ / np.maximum(c, 1), np.nan)
        rw = np.nanmean(Yh, axis=1); rw = np.where(np.isnan(rw), nco[idx].mean(), rw)
        Yh = np.where(np.isnan(Yh), rw[:, None], Yh)
        return float(np.mean((pi_obs[x0[idx]] * Yh[x0[idx]]).sum(1))) - nco[idx].mean()
    out["nco_model"] = abs(v_nco_model - nco.mean()) / boot_se(nco_model_boot, n, rng)
    # negative-control policy: perturb the target only in inert states; every estimator must not move
    ests_nc = all_estimates(x, a, r, n_x, pi_nc_obs, beh_hat, x0)
    diffs = {}
    for e in ESTIMATORS:
        def fn(idx, e=e):
            ev = all_estimates(x[idx], a[idx], r[idx], n_x, pi_obs, beh_hat, x0[idx])
            ev2 = all_estimates(x[idx], a[idx], r[idx], n_x, pi_nc_obs, beh_hat, x0[idx])
            return ev2[e] - ev[e]
        diffs[e] = abs(ests_nc[e] - est_vals[e]) / boot_se(fn, n, rng, B=12)
    out["ncp"] = float(max(diffs.values()))
    out.update({f"ncp_{e}": v for e, v in diffs.items()})
    # placebo: shuffle whole reward sequences across episodes; the target must look like the behaviour
    perm = rng.permutation(n)
    ev_pl = all_estimates(x, a, r[perm], n_x, pi_obs, beh_hat, x0)
    Gp = r[perm].sum(1)
    out["placebo"] = float(max(abs(ev_pl[e] - Gp.mean()) for e in ("WIS", "PDIS", "DR", "FQE")) / _se(Gp))
    return out


TESTS = ["agree", "ess", "self", "nco_is", "nco_model", "ncp", "placebo"]
