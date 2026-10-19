# Is the Critic Behind? Measuring Which Policy a Deep Critic Evaluates

*Draft v0.1 (pilot scale, negative result). This is written as a short empirical note; the pilot killed the central hypothesis and the honest venue is a workshop (e.g. the NeurIPS workshop on RL theory and practice or "I Can't Believe It's Not Better"). All numbers come from `../results/` and `../results/fine/`.*

## Abstract

Two-timescale analyses of actor-critic require the critic to move faster than the actor, and the informal picture that goes with them is that a slow critic evaluates a *stale* policy: the actor is improved against the value function of a policy it no longer is. We test this picture directly. In a deterministic simulator the true action value of any stored past policy can be computed exactly by Monte-Carlo, so for a critic at time t we can ask which of the policies π_t, π_{t−100}, …, π_{t−6000} it most resembles. Across 96 DDPG runs on Pendulum spanning a 300-fold range of critic-to-actor learning-rate ratios, the answer is always the current one: the error profile over lag is flat to within 2% and, where the policy is still moving, increases with lag. Slow critics are three times less accurate and cost 35–50 return points, and their error tracks the rate of policy change (ρ = 0.90 across cells), but it is inaccuracy about the present policy, not knowledge of a past one. "Lag" is the wrong metaphor for critic error in deep actor-critic; "tracking error" is the right one, and we show it is a cheap, exact diagnostic in any resettable simulator.

## 1. Introduction

Actor-critic methods learn a value function for a policy that is changing under them. Classical convergence results handle this with two timescales (Konda & Tsitsiklis, 2000; Borkar, 2008): the critic's step size is asymptotically larger than the actor's, so that from the actor's point of view the critic has converged to Q^π for the current π. Practice ignores the asymptotics and uses the same learning rate for both, typically 3e-4 (Haarnoja et al., 2018; Fujimoto et al., 2018).

The informal story told to explain why this works, and why it sometimes does not, is one of *lag*: the critic is "behind", it reflects the policy of a few hundred updates ago, and the actor is therefore improved against a slightly wrong objective. Several methods are motivated by this picture, including delayed actor updates (Fujimoto et al., 2018) and proposals to compute targets with an anticipated future actor.

We asked whether the picture is literally true. If a critic at time t lags, its values should be closer to the true action values of some past policy π_{t−k} than to those of π_t. In a deterministic simulator this is checkable: store the actor at every evaluation, roll each stored actor out from the same probe states, and compute Q^{π_{t−k}}(s, π_t(s)) exactly. The lag is the k that minimises the discrepancy.

We ran this measurement at two resolutions (1 000-step and 100-step evaluation intervals) across a grid of actor and critic learning rates, with pre-registered tests for the existence of a lag and its dependence on the learning-rate ratio. Both tests failed. The critic is never closer to a past policy than to the present one.

## 2. Setup

**Agent.** DDPG with target networks (Polyak τ = 0.005), Gaussian exploration noise σ = 0.2, two hidden layers of 64, batch 64, buffer 20k, γ = 0.98, 30k environment steps, one gradient update per step. Actor LR ∈ {1e-4, 3e-4, 1e-3} × critic LR ∈ {1e-4, 3e-4, 1e-3, 3e-3} × 8 seeds = 96 runs, batched as one tensor program with per-run generators.

**Ground truth.** At each evaluation, 64 probe states uniform over (θ, θ̇); the current actor's action a_t = π_t(s); for each stored past actor π_{t−k}, the exact discounted return of taking a_t then following π_{t−k} for 250 steps (truncation error < 1% at γ = 0.98). Error e_k = mean_s |Q_critic(s, a_t) − Q^{π_{t−k}}(s, a_t)|.

**Tests (pre-registered, README).** Pilot: K1, in cells with critic LR ≤ actor LR, k* = argmin_k e_k is positive in ≥ 60% of (run, eval) pairs and e_{k*} is ≥ 20% below e_0; K2, Spearman(log ratio, mean k*) ≤ −0.6; K3, lag or e_0 predicts final return. Fine-resolution follow-up (after the pilot's K1 was found to pass through argmin selection bias): F1, the *mean* profile over k is minimised at k > 0 in every slow-critic cell with bootstrap support ≥ 0.8; F2, Spearman(log ratio, argmin) ≤ −0.6.

## 3. Results

### 3.1 No lag at 100–6000 steps

Table 1 gives the mean error profile at 100-step resolution. In the three slow-critic cells (critic LR 1e-4), where the policy is still moving at 5–10 Q-units per 100 steps, the profile is minimised at k = 0 and rises monotonically with k. In the fast-critic cells the profile is flat to within 2% (the policy has converged, so all past policies are the same policy). F1 and F2 both fail. The pilot at 1 000-step resolution shows the same: the mean profile for critic LR 1e-4 is 47.6, 48.8, 49.5, 51.0, 53.5, 58.0, 64.2 for k = 0..6.

| actor LR | critic LR | e₀ | e₂ | e₅ | e₁₀ | drift / 100 steps | final return |
|---|---|---|---|---|---|---|---|
| 1e-4 | 1e-4 | 46.9 | 47.3 | 47.9 | 48.6 | 5.3 | −176 |
| 3e-4 | 1e-4 | 48.1 | 48.4 | 48.8 | 49.6 | 6.4 | −161 |
| 1e-3 | 1e-4 | 42.7 | 42.9 | 43.3 | 43.8 | 10.2 | −168 |
| 1e-4 | 3e-4 | 23.1 | 23.2 | 23.1 | 23.0 | 0.9 | −129 |
| 3e-4 | 1e-3 | 15.5 | 15.6 | 15.6 | 15.8 | 0.6 | −127 |
| 1e-3 | 3e-3 | 14.4 | 14.4 | 14.4 | 14.4 | 0.5 | −127 |

*Table 1. Mean |Q_critic − Q^{π_{t−k}}(s, π_t(s))| at lags of k × 100 steps, steps ≥ 10k, 8 seeds per cell. Drift = mean |Q^{π_t} − Q^{π_{t−100}}| on the probe states. The critic is closest to the current policy in every cell where the distinction is measurable.*

### 3.2 The argmin artefact

The pilot's K1 passed (k* > 0 in 83% of pairs, 23% error reduction) and this was an artefact: with seven nearly equal errors, the minimum lands on k > 0 six times in seven by chance, and selecting it biases the "improvement" upward. The mean profile, which has no selection step, is minimised at k = 0. We record this because the argmin estimator is the natural first thing to try and it is wrong.

### 3.3 What slow critics do instead

Slow critics are *inaccurate*, not stale. Their error against the current policy is 43–48 against 14–15 for fast critics, and across the 12 cells Spearman(e₀, final return) = −0.78: the three slow-critic cells finish 35–50 points worse. The error tracks the rate of policy change: Spearman(drift, e₀) = +0.90 across cells, and pooled over (run, eval) pairs the log-log correlation of e₀ with drift is 0.67. The two-timescale requirement shows up as a tracking-error requirement: the critic's learning rate must be large relative to how fast Q^π is moving, and when it is not, the error is spread over the state-action space as a smooth bias rather than aligned with any past policy's values.

## 4. Discussion

The lag picture assumes the critic is a delayed, otherwise faithful copy of Q^π. A TD-trained critic on replayed data is not: its targets combine a Polyak-averaged target actor, bootstrapped values that are themselves in error, and actions drawn from the whole history of the buffer. The result is a bias field that does not line up with any single policy. This has a practical consequence for methods motivated by lag (delayed actor updates, anticipatory targets): what they must correct is tracking error, which is governed by critic LR relative to drift, and the measurement in this paper is the way to check whether they do.

The measurement itself is the contribution we would keep. In any simulator with state resets (MuJoCo exposes `set_state`; most JAX environments are functional), exact Q^π for stored policies is a few seconds of rollout and gives a low-variance critic-quality signal with no learned baseline. We use it in a companion project to measure overestimation on non-terminal states.

## 5. Limitations

One environment (Pendulum), one algorithm (DDPG, single critic), small networks. A higher-dimensional task with slower policy convergence could in principle show a lag that Pendulum does not, though the monotone-in-k profiles in the slow-critic cells argue against it. Probe states are uniform rather than on-policy. Lags beyond 6 000 steps were not tested.

## 6. Reproducibility

`code/r10_critic_lag.py` (12 CPU-minutes for the pilot, 31 for the fine-resolution run). The README records both pre-registrations; `results/posthoc.txt` records the analyses that were not.

## References

- Borkar, V. S. (2008). *Stochastic Approximation: A Dynamical Systems Viewpoint.*
- Fujimoto, S., van Hoof, H., Meger, D. (2018). Addressing function approximation error in actor-critic methods. ICML.
- Haarnoja, T. et al. (2018). Soft actor-critic. ICML.
- Konda, V. & Tsitsiklis, J. (2000). Actor-critic algorithms. NeurIPS.
- Lillicrap, T. et al. (2016). Continuous control with deep reinforcement learning. ICLR.
- Nauman, M. et al. (2024). Overestimation, overfitting, and plasticity in actor-critic. ICML.
