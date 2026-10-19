"""A small continuous-time 'ICU glucose' simulator with irregular, policy-chosen check times.

The point is not physiological fidelity. The point is a process that has the three
properties Frost et al. (2602.06603) and Insulin4RL (2606.19481) say matter and that
fixed-interval binning destroys:

1. decisions happen at irregular times chosen by the policy (the *wait* action),
2. the policy only observes the patient when it checks (partial, irregular observation),
3. checking has a cost, so 'look less often' is a real trade-off.

Dynamics (all per hour, integrated with an Euler step of DT hours):
    d   : OU disturbance (meals / stress)      d'  = -k_d d + sigma_d dW
    iob : insulin on board                      iob'= -k_i iob          (+ dose at a check)
    g   : glucose, mg/dL                        g'  = -k_g (g - g_b) - s iob + d + sigma_g dW

Reward rate (per hour): +1 in range [70, 180], -4 below 70, -0.5 in (180, 250], -2 above 250.
Each check costs C_CHECK. Return is discounted at GAMMA_H per hour over a 48 h stay.

A policy is a vectorised function  obs -> (dose_idx, wait_idx)  where
    obs = dict(g_obs, iob, t, last_wait_idx)     (arrays of equal length)
    dose_idx 0 = hypo rescue (dextrose), 1..4 = [0, 2, 4, 6] insulin units; WAITS = [1, 2, 4] hours.

`simulate` returns per-check event logs plus the discounted return of each episode, so the
same function gives ground truth (run the target policy) and logged data (run the behaviour
policy).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DT = 5.0 / 60.0           # hours
T_END = 48.0              # hours
GAMMA_H = 0.98            # discount per hour
C_CHECK = 0.25            # cost of one glucose check (in reward units)
# action 0 is a hypoglycaemia rescue (dextrose: +G_RESCUE mg/dL at once); 1..4 are insulin units
DOSES = np.array([0.0, 0.0, 2.0, 4.0, 6.0])
RESCUE_IDX = 0
NO_DOSE_IDX = 1
G_RESCUE = 60.0
WAITS = np.array([1.0, 2.0, 4.0])
N_DOSE = len(DOSES)
N_WAIT = len(WAITS)
N_ACT = N_DOSE * N_WAIT

# glucose bands used by the behaviour policy (and by the default discretiser)
G_EDGES = np.array([70.0, 110.0, 150.0, 180.0, 250.0])   # -> 6 bins
IOB_EDGES = np.array([0.5, 2.0, 5.0])                    # -> 4 bins
T_EDGES = np.array([12.0, 24.0, 36.0])                   # -> 4 bins


def reward_rate(g: np.ndarray) -> np.ndarray:
    r = np.zeros_like(g)
    r[(g >= 70) & (g <= 180)] = 1.0
    r[g < 70] = -4.0
    r[(g > 180) & (g <= 250)] = -0.5
    r[g > 250] = -2.0
    return r


@dataclass
class Events:
    """Flat arrays, one row per check (decision point)."""
    ep: np.ndarray          # episode id
    t: np.ndarray           # time of the check (h)
    g_obs: np.ndarray       # observed glucose at the check
    iob: np.ndarray         # insulin on board just before dosing
    dose_idx: np.ndarray
    wait_idx: np.ndarray
    tau: np.ndarray         # realised sojourn until the next check or the end (h)
    r_soj: np.ndarray       # discounted reward accumulated during the sojourn, discounted to t
    # (includes the check cost of *this* check, not of the next one)
    done: np.ndarray        # True if the sojourn ended the episode
    g_obs_next: np.ndarray  # observation at the next check (nan if done)
    iob_next: np.ndarray
    t_next: np.ndarray
    wait_idx_prev: np.ndarray  # wait chosen at the previous check (-1 at the first)
    returns: np.ndarray     # per-episode discounted return (length = number of episodes)
    reward_grid: np.ndarray  # (episodes, 48): reward (incl. check costs) per clock hour, discounted to t=0

    def n_episodes(self) -> int:
        return len(self.returns)


def _draw_patients(n: int, rng: np.random.Generator) -> dict:
    return dict(
        g_b=np.clip(rng.normal(160.0, 30.0, n), 90.0, 260.0),
        s=np.exp(rng.normal(np.log(3.5), 0.35, n)),       # mg/dL per hour per unit of iob
        k_g=np.full(n, 0.25),
        k_i=np.exp(rng.normal(np.log(0.35), 0.2, n)),
        k_d=np.full(n, 0.5),
        sigma_d=np.full(n, 25.0),
        sigma_g=np.full(n, 6.0),
        obs_sd=np.full(n, 6.0),
    )


def simulate(policy, n: int, seed: int, first_wait: float = 1.0) -> Events:
    """Run `policy` on n fresh patients. Returns the event log and the per-episode returns."""
    rng = np.random.default_rng(seed)
    p = _draw_patients(n, rng)
    g = p["g_b"] + rng.normal(0.0, 40.0, n)
    g = np.clip(g, 40.0, 450.0)
    iob = np.zeros(n)
    d = rng.normal(0.0, 20.0, n)
    t = 0.0
    next_check = np.full(n, 0.0)         # everybody is checked at admission
    last_wait = np.full(n, -1, dtype=int)
    ret = np.zeros(n)
    # per-episode bookkeeping of the open sojourn (vectorised)
    open_idx = np.full(n, -1, dtype=int)
    t_open = np.zeros(n)
    r_open = np.zeros(n)
    rows: list[list] = []                # one row per check, filled lazily
    grid = np.zeros((n, int(T_END)))

    n_steps = int(round(T_END / DT))
    for step in range(n_steps + 1):
        t = step * DT
        disc = GAMMA_H ** t
        due = next_check <= t + 1e-9
        if step == n_steps:
            due = np.zeros(n, dtype=bool)
        if due.any():
            idx = np.flatnonzero(due)
            g_obs = g[idx] + rng.normal(0.0, 1.0, len(idx)) * p["obs_sd"][idx]
            obs = dict(g_obs=g_obs, iob=iob[idx].copy(), t=np.full(len(idx), t), last_wait_idx=last_wait[idx].copy())
            dose_idx, wait_idx = policy(obs, rng)
            dose_idx = np.asarray(dose_idx, dtype=int)
            wait_idx = np.asarray(wait_idx, dtype=int)
            # close the previous sojourn of these episodes
            for j, e in enumerate(idx):
                k = open_idx[e]
                if k >= 0:
                    rows[k][6] = t - rows[k][1]            # tau
                    rows[k][7] = r_open[e]                 # r_soj
                    rows[k][8] = False                     # done
                    rows[k][9] = g_obs[j]
                    rows[k][10] = iob[e]
                    rows[k][11] = t
                new_k = len(rows)
                rows.append([e, t, g_obs[j], iob[e], dose_idx[j], wait_idx[j], np.nan, 0.0, True, np.nan, np.nan, np.nan, last_wait[e]])
                open_idx[e] = new_k
            t_open[idx] = t
            r_open[idx] = -C_CHECK                         # check cost, discounted to the check time -> factor 1
            ret[idx] -= C_CHECK * disc
            grid[idx, min(int(t), int(T_END) - 1)] -= C_CHECK * disc
            iob[idx] += DOSES[dose_idx]
            g[idx] += np.where(dose_idx == RESCUE_IDX, G_RESCUE, 0.0)
            jitter = rng.uniform(-0.1, 0.1, len(idx))
            next_check[idx] = t + WAITS[wait_idx] + jitter
            last_wait[idx] = wait_idx
        if step == n_steps:
            break
        # integrate one step
        r = reward_rate(g) * DT
        ret += r * disc
        grid[:, min(int(t), int(T_END) - 1)] += r * disc
        r_open += r * (GAMMA_H ** (t - t_open))            # discounted to the sojourn start
        noise_d = rng.normal(0.0, 1.0, n)
        noise_g = rng.normal(0.0, 1.0, n)
        d = d + (-p["k_d"] * d) * DT + p["sigma_d"] * np.sqrt(DT) * noise_d
        iob = iob * np.exp(-p["k_i"] * DT)
        g = g + (-p["k_g"] * (g - p["g_b"]) - p["s"] * iob + d) * DT + p["sigma_g"] * np.sqrt(DT) * noise_g
        g = np.clip(g, 20.0, 500.0)
    # close the open sojourns at the end
    for e in range(n):
        k = open_idx[e]
        rows[k][6] = T_END - rows[k][1]
        rows[k][7] = r_open[e]
        rows[k][8] = True
        rows[k][11] = T_END
    arr = np.array(rows, dtype=float)
    return Events(
        ep=arr[:, 0].astype(int), t=arr[:, 1], g_obs=arr[:, 2], iob=arr[:, 3],
        dose_idx=arr[:, 4].astype(int), wait_idx=arr[:, 5].astype(int), tau=arr[:, 6], r_soj=arr[:, 7],
        done=arr[:, 8].astype(bool), g_obs_next=arr[:, 9], iob_next=arr[:, 10], t_next=arr[:, 11],
        wait_idx_prev=arr[:, 12].astype(int), returns=ret, reward_grid=grid,
    )


# ----------------------------------------------------------------------------- policies

def _band(g_obs: np.ndarray) -> np.ndarray:
    return np.digitize(g_obs, G_EDGES)      # 0:<70 1:70-110 2:110-150 3:150-180 4:180-250 5:>250


# behaviour 'sliding scale': modal dose and modal wait per glucose band
BEH_DOSE_MODE = np.array([0, 1, 1, 2, 3, 4])     # dose index per band (rescue if <70, nothing in range, then 2/4/6 u)
BEH_WAIT_MODE = np.array([0, 2, 2, 1, 1, 0])     # wait index per band (1h / 4h / 4h / 2h / 2h / 1h)
EPS_DOSE = 0.15
EPS_WAIT = 0.30


def behaviour_probs(g_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P(dose | obs) and P(wait | obs) for the logging clinician. Shape (n, N_DOSE), (n, N_WAIT)."""
    b = _band(g_obs)
    pd = np.full((len(b), N_DOSE), EPS_DOSE / (N_DOSE - 1))
    pd[np.arange(len(b)), BEH_DOSE_MODE[b]] = 1.0 - EPS_DOSE
    pw = np.full((len(b), N_WAIT), EPS_WAIT / (N_WAIT - 1))
    pw[np.arange(len(b)), BEH_WAIT_MODE[b]] = 1.0 - EPS_WAIT
    return pd, pw


def _sample_rows(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    c = np.cumsum(p, axis=1)
    u = rng.uniform(size=(len(p), 1))
    return np.minimum((u > c).sum(axis=1), p.shape[1] - 1)


def pi_behaviour(obs, rng):
    pd, pw = behaviour_probs(obs["g_obs"])
    return _sample_rows(pd, rng), _sample_rows(pw, rng)


def make_target(dose_shift: int = 0, wait_shift: int = 0, stochastic_dose: bool = False):
    """Deterministic target policies built from the behaviour's modal rule.

    dose_shift: +1 = one dose level higher than the clinician's modal dose.
    wait_shift: +1 = one wait class longer (look less often), -1 = shorter (look more often).
    """
    def pi(obs, rng):
        b = _band(obs["g_obs"])
        if stochastic_dose:
            pd, _ = behaviour_probs(obs["g_obs"])
            dose = _sample_rows(pd, rng)
        else:
            dose = np.clip(BEH_DOSE_MODE[b] + dose_shift, 0, N_DOSE - 1)
        wait = np.clip(BEH_WAIT_MODE[b] + wait_shift, 0, N_WAIT - 1)
        return dose, wait
    pi.dose_shift = dose_shift
    pi.wait_shift = wait_shift
    return pi


def target_action_table(dose_shift: int, wait_shift: int) -> tuple[np.ndarray, np.ndarray]:
    """(dose_idx, wait_idx) per glucose band for a deterministic target. Used by the estimators."""
    bands = np.arange(len(G_EDGES) + 1)
    return (np.clip(BEH_DOSE_MODE[bands] + dose_shift, 0, N_DOSE - 1),
            np.clip(BEH_WAIT_MODE[bands] + wait_shift, 0, N_WAIT - 1))


def make_grid_policy(dose_shift: int, every_h: float):
    """The 'degenerate timing policy' that binning silently evaluates: the dose rule applied
    at a fixed interval, checking (and paying) every `every_h` hours."""
    def pi(obs, rng):
        b = _band(obs["g_obs"])
        dose = np.clip(BEH_DOSE_MODE[b] + dose_shift, 0, N_DOSE - 1)
        wait = np.full(len(b), int(np.argmin(np.abs(WAITS - every_h))))
        return dose, wait
    return pi


# ----------------------------------------------------------------------------- discretiser

def disc_state(g_obs: np.ndarray, iob: np.ndarray, t: np.ndarray,
               g_edges: np.ndarray = G_EDGES, iob_edges: np.ndarray = IOB_EDGES, t_edges: np.ndarray = T_EDGES) -> np.ndarray:
    gb = np.digitize(g_obs, g_edges)
    ib = np.digitize(iob, iob_edges)
    tb = np.digitize(t, t_edges)
    n_i = len(iob_edges) + 1
    n_t = len(t_edges) + 1
    return (gb * n_i + ib) * n_t + tb


def n_states(g_edges=G_EDGES, iob_edges=IOB_EDGES, t_edges=T_EDGES) -> int:
    return (len(g_edges) + 1) * (len(iob_edges) + 1) * (len(t_edges) + 1)
