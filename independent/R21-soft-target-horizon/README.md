# R21: Soft target updates are an implicit discount schedule

**Question.** DDPG, TD3 and SAC update their target networks by Polyak averaging, θ̄ ← (1 − τ)θ̄ + τθ, with τ = 0.005 as a universal default that nobody derives. We claim τ is a horizon parameter. With one-step targets, the online critic at a state is a first-order low-pass filter (rate α, the fit rate) of the target critic one step downstream, and the target critic is a first-order filter (rate τ) of the online critic. A reward k steps away therefore passes through k stages of rate α and k stages of rate τ before it reaches a decision, and the arrival time is a sum of 2k exponentials with mean k·(1/τ + 1/α) updates. With n-step targets the stage count is k/n. The consequence is that the critic's effective horizon grows linearly with the update count,

  h(t) ≈ n·t / (1/τ + 1/α),

so the agent is greedy with respect to a horizon-h(t) problem until h(t) reaches the task's horizon: Polyak averaging is an implicit discount schedule. At τ = 0.005, n = 1, one update per environment step and a well-fit critic, the horizon reaches 100 steps after ~20 000 updates; at τ = 0.005 with α-limited fitting it is slower by the α term. The same cascade argument gives the hard-sync law of R12 as the limit where the τ-stage is a deterministic K-step delay, and matches a hard period K to a soft rate τ = 1/K. Piché et al. (ICLR 2022, "Beyond target networks", functional regularisation) state the hard-sync N·H chain latency qualitatively and note that Polyak averaging also slows propagation; nobody has written the Polyak law, the α-symmetry, or the horizon interpretation.

**Status.** Laptop pilot done: **SUPPORTED** on all four tests. The horizon grows at n/(1/τ + 1/α) per update in 18/18 cells (R² = 1.000); the patience switch lands on the closed form with log–log slope 1.011, R² = 0.9997, 54/54 cells within [0.79, 1.03]×; τ and α are symmetric (12/12); hard sync at K = 1/τ matches soft τ within 1.08× (15/15). At τ = 0.01 the agent is a γ_eff ≈ 0.9 agent for its first 1000 updates. See `results/verdict.txt`, `results/posthoc.txt`, `notes/pilot_notes.md`. Paper version on the 3080: TD3/SAC with τ ∈ {0.05, 0.005, 0.0005} × n ∈ {1, 5} on (a) a patience gridworld with continuous actions, measuring the iteration at which the policy switches from the impatient to the patient option, and (b) Hopper/Walker, measuring the learned critic's effective horizon over training by regressing V against truncated Monte-Carlo returns at horizons h ∈ {5, 10, 20, 50, 100, 200}.

Adjacent prior work: R12 (hard-sync law, this lab); Piché et al. 2022; Lillicrap et al. 2016 (introduces τ, no analysis); Kim et al. 2019 and Zhang et al. 2021 (target networks as stabilisers); Fedus et al. 2020 (n-step makes replay work). Lab stubs: the delayed-reward cluster studies reward delay inside an episode; this is a training-time latency; named for adjacency. Not overlapping with R19 (GAE/PPO) or R20 (accounting) beyond sharing the latency programme.

## Pre-registered kill test (written before the pilot ran)

Design. Deterministic synchronous tabular learning: every state is updated every step (no replay, no exploration), Q ← Q + α(y − Q) with y the n-step target bootstrapped from the Polyak table Q̄ ← Q̄ + τ(Q − Q̄); γ = 0.99. Two environments.

- **Chain (horizon measurement).** D = 200 states in a line, reward 1 per step, terminal after the last. V*(s₀) = (1 − γ^D)/(1 − γ). The effective horizon at update t is h(t) = log(1 − (1 − γ)V_t(s₀)) / log γ, i.e. the truncation horizon whose value equals the current estimate.
- **Patience fork.** Start state with two actions: `now` pays 1 and terminates; `wait` enters a corridor of d steps with reward R at the end and terminates, R·γ^d = 2 (patient is optimal with a 2× margin at full horizon). The switch time t* is the first update at which Q(s₀, wait) > Q(s₀, now).

Grid: τ ∈ {0.1, 0.01, 0.001} × α ∈ {1.0, 0.1, 0.01} × n ∈ {1, 5}; fork depths d ∈ {10, 30, 100}; hard-sync controls with K = 1/τ at α = 1. Predicted cascade: with m = ⌈d/n⌉ segments, the reward passes through m fit stages (rate α) and m − 1 target stages (rate τ): the segment adjacent to the reward needs no bootstrap. Mean arrival (m − 1)/τ + m/α; the switch happens at the median of the arrival distribution because R·γ^d = 2, and the Gamma-median approximation gives

  t*_pred = (m − 1)/τ + m/α − max(1/τ, 1/α)/3.

*Smoke-test correction (before the pilot, 3000-step smoke run):* the first draft of this section counted m target stages instead of m − 1, which over-predicted the n = 5, d = 10 cells (m = 2) by 2.4× because only one τ-stage is on the path. The count was corrected by re-deriving the cascade, not by fitting; the horizon-rate prediction in K1 is unchanged because for the long chain the −1 is negligible against m. K3 was re-expressed in terms of the corrected prediction (its original "2×" was the m → ∞ limit).

- **K1 (horizon grows linearly at the predicted rate).** For each (τ, α, n) cell, OLS slope of h(t) against t over the range h ∈ [5, 100] is within [0.8, 1.2] of n/(1/τ + 1/α) in at least 16 of 18 cells, and R² of the linear fit ≥ 0.98 in each.
- **K2 (the policy switches when the horizon arrives).** Over the 54 fork cells (τ × α × n × d), log t* regressed on log t*_pred has slope in [0.9, 1.1] and R² ≥ 0.98, and the ratio t*/t*_pred lies in [0.7, 1.4] in at least 48 of 54 cells.
- **K3 (τ and α are symmetric).** For cells with τ = α (0.1/0.1, 0.01/0.01), the ratio t*(τ = α) / t*(τ, α = 1) is within [0.8, 1.25] of the predicted ratio (2m − 1 − 1/3)/((m − 1 − 1/3) + m·τ), for every d and n (12 cells).
- **K4 (hard = soft at K = 1/τ).** At α = 1, for m = ⌈d/n⌉ ≥ 6, the hard-sync switch time is within [0.7, 1.3] of the soft t* for every (τ, n, d).

SUPPORTED if K1 and K2 and K3. INCONCLUSIVE if two. KILLED otherwise. K4 is reported as the bridge to R12.

## Layout

| folder | contents |
|---|---|
| `code/` | `r21_soft.py`; `code/lib/io_utils.py` |
| `results/` | `horizon.csv` (h(t) traces), `fork.csv` (switch times), `summary.csv`, `verdict.txt` |
| `logs/` | `pilot.log` |
| `notes/` | `pilot_notes.md` |
| `paper/` | `main.md` draft |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/r21_soft.py --smoke
.venv/bin/python code/r21_soft.py             # pilot, ~5 min CPU
```
