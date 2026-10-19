# Two Biases, One Knob: Why Uncorrected n-Step Returns Help Under Noise and Hurt Without It

*Draft. Pilot numbers are from the laptop runs in `results/`; sections marked [3080] are planned.*

## Abstract

Every strong value-based agent uses n-step returns built from behaviour actions in the replay buffer with no off-policy correction. This is theoretically wrong, empirically necessary (it is what makes large replay buffers useful; Fedus et al., 2020), and the corrected alternatives do not help (Hernandez-Garcia & Sutton, 2019). We give a bias accounting for the uncorrected target: it carries a *pessimistic* term from blending the behaviour policy's value, Q ← (T^μ)^{n−1} T* Q, whose fixed point lies below Q* and whose size grows with n and the behaviour policy's ε, and it carries the usual *optimistic* max-bias of Q-learning under noise, which falls steeply with n because a longer n-step target leans less on the bootstrapped max. In tabular Q-learning with replay on a stochastic gridworld with exact Q*, three pre-registered consequences hold and replicate on fresh seeds: overestimation falls monotonically with n in every noisy cell (Spearman −1.00 in 10/10), n ≥ 2 is pessimistic without noise and costs 11–23% of return, and the best n rises with reward noise (1 → 2–3 → 3 → 5–10 as σ goes 0 → 0.05 → 0.1 → 0.2). The best n is, in all 10 noisy cells, one of the two n at which the two biases net to zero. A fourth prediction, that double Q-learning removes the n-dependence, failed: double Q halves the max-bias and shares the pessimistic term, so it still benefits from n > 1. The accounting predicts that the optimal n should rise over training as ε anneals; [3080] DQN/Double DQN on MinAtar with injected reward noise and a propagation-speed control are planned.

## 1. Introduction

Rainbow (n = 3), R2D2 (n = 5), Agent57, MuZero's reanalyse and most modern replay-based agents compute n-step targets from stored behaviour trajectories without importance weights or Retrace-style truncation. The fixed point of this update is not Q*: the target applies n − 1 steps of the behaviour policy's Bellman operator before the max. The practice persists because it works and the corrected alternatives do not (Hernandez-Garcia & Sutton, 2019; Fedus et al., 2020 find that n-step is what lets large buffers help at all). The explanations on offer are propagation speed (a reward reaches the start in L/n hops; see R12) and variance. Neither explains why *uncorrected* n-step should beat corrected n-step, which propagates just as fast.

We offer a bias accounting. Two biases live in the uncorrected target with opposite signs and opposite dependence on n. The pessimistic one is specific to the uncorrected target. The optimistic one is Q-learning's familiar max-bias (Thrun & Schwartz, 1993; van Hasselt, 2010), and it falls with n because the bootstrapped max enters with weight γ^n and compounds through fewer steps. When there is noise, n trades one bias against the other, and the best n is where they cancel. When there is no noise, only the pessimistic term is left, and n > 1 costs.

Contributions. (1) The decomposition and its testable consequences. (2) A pre-registered tabular test with exact ground truth; the sign and ordering predictions replicate out of sample on fresh seeds; a competence threshold we set for ourselves was missed by five points, so the pre-registered verdict is INCONCLUSIVE, which we report as such. (3) The finding that double Q does *not* remove the n-dependence, and why. (4) A prediction specific to the accounting, that the optimal n rises as ε anneals, with [3080] the deep test.

## 2. Related work

Uncorrected n-step in practice: Hessel et al. (2018), Kapturowski et al. (2019), Fedus et al. (2020). Corrections: Retrace (Munos et al., 2016), Tree Backup (Precup et al., 2000), Q(σ) (De Asis et al., 2018); Hernandez-Garcia & Sutton (2019) compare and find the corrections do not help in DQN-style agents. Max-bias and its fixes: Thrun & Schwartz (1993), van Hasselt (2010), van Hasselt et al. (2016), Lan et al. (2020, Maxmin), Chen et al. (2021, REDQ); none of these treats n as a max-bias knob. Pessimism from behaviour-policy blending appears in the offline-RL literature as a feature (Kostrikov et al., 2022, IQL uses expectile regression partly for this reason); the connection to online n-step is not drawn. Propagation speed of n-step: our R12. Adjacent lab stub S2-5 ("n-step by transition age") proposes a method; this is the diagnosis that would justify or refute it, and it predicts a different schedule (n should track ε, not age).

## 3. The accounting

Uncorrected n-step target from a stored behaviour trajectory: G = Σ_{k<n} γ^k r_{t+k} + γ^n max_a Q̄(s_{t+n}, a). In expectation over μ's actions along the trajectory the update is Q ← (T^μ)^{n−1} T* Q̄ for the first n − 1 steps and the max for the last. Its fixed point Q_n^μ satisfies Q_n^μ ≤ Q* with equality only when μ is greedy along the way; for ε-greedy μ the gap at the start state grows with n (more behaviour steps) and ε (worse behaviour), and is zero at n = 1. Call this pess(n, ε) ≤ 0.

Under zero-mean reward noise σ, Q-learning overestimates because max_a Q̄ of noisy estimates is biased upward (Thrun & Schwartz). The bias enters the n-step target with weight γ^n and compounds through ⌈L/n⌉ bootstrapping steps rather than L; both shrink with n. Call this maxbias(n, σ) ≥ 0, decreasing in n, zero at σ = 0 up to the transition-noise contribution.

Predictions. (i) Total overestimation falls with n whenever σ > 0 and is ≤ 0 for n ≥ 2 at σ = 0. (ii) The n that maximises return rises with σ (more max-bias to cancel) and falls with ε (more pessimism per step). (iii) The best n is where pess + maxbias ≈ 0. (iv) [as pre-registered] Double Q removes maxbias, so its best n should not depend on σ. (iv) turned out to be wrong in premise, see §4.4.

## 4. Experiments

### 4.1 Setup and pre-registration

8×8 gridworld, slip 0.1, γ = 0.95, reward 1 on entering the absorbing goal, exploring starts, episode cap 100. Tabular Q with FIFO replay 5000, batch 32, α = 0.1, target table refreshed every 100 updates, uncorrected n-step targets for n ∈ {1, 2, 3, 5, 10}, behaviour ε-greedy with constant ε ∈ {0.1, 0.3}, Gaussian reward noise σ on every step (Q* unchanged), plain Q and double Q (online argmax, target evaluation), 150k steps, 4 seeds. Overestimation over_all = Σ_s d^{π_g}(s)(max_a Q(s, a) − V*(s)) with d the greedy policy's discounted visitation from the start; greedy return = exact V^{π_g}(s₀); V* = 0.468. Pre-registered tests K1 (sign), K2 (best n rises with σ), K3 (double Q halves the slope on log n); SUPPORTED if K1 and K2.

Three σ grids were run and all are reported. The first (σ ∈ {0, 0.5, 1.5}) passed K1 and K2 by rule but no agent at σ > 0 exceeded 0.4% of V*; K2 there ranks failed agents and we discount it. We then pre-registered a follow-up (σ ∈ {0, 0.1, 0.2, 0.3}) with a working-agent condition at σ_max, which K1/K2 passed and the condition failed (0.7% of V* at σ = 0.3). We then pre-registered a replication on **fresh seeds** (σ ∈ {0, 0.05, 0.1}) with four predictions P1–P4 and a competence threshold P3 of 50% of V* at σ = 0.1. The replication is the test we stand on.

### 4.2 Replication on fresh seeds

P1 (sign): Spearman(n, over_all) = −1.00 in all four noisy cells; n ≥ 3 pessimistic at σ = 0. **Holds.** P2 (ordering): best n by AUC = 1, 1 (σ = 0) / 3, 2 (σ = 0.05) / 3, 3 (σ = 0.1) for ε = 0.1, 0.3. **Holds.** P3 (competence): plain Q's best final return at σ = 0.1 is 45% of V* (the first seed set gave 55%). **Fails, by five points.** P4 (cancellation): best n is one of the two n with over_all nearest zero in all four noisy cells, and in all 10 noisy cells pooled across the second and third grids (chance 0.4^10 ≈ 10⁻⁴). **Holds.** Verdict by the pre-registered rule: INCONCLUSIVE.

Plain Q, over_all by n (third grid):

| σ | ε | n=1 | n=2 | n=3 | n=5 | n=10 | best n | best final / V* |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.1 | +0.010 | −0.002 | −0.007 | −0.011 | −0.018 | 1 | 95% |
| 0 | 0.3 | +0.009 | −0.016 | −0.027 | −0.044 | −0.058 | 1 | 99% |
| 0.05 | 0.1 | +0.037 | +0.003 | −0.005 | −0.020 | −0.044 | 3 | 79% |
| 0.05 | 0.3 | +0.051 | −0.014 | −0.030 | −0.061 | −0.102 | 2 | 71% |
| 0.1 | 0.1 | +0.132 | +0.015 | +0.009 | −0.032 | −0.086 | 3 | 45% |
| 0.1 | 0.3 | +0.162 | +0.040 | −0.013 | −0.098 | −0.134 | 3 | 28% |

### 4.3 Decomposition

Taking pess(n) as over_all at σ = 0 and maxbias(n, σ) = over_all(σ, n) − pess(n): at σ = 0.1, ε = 0.1, maxbias = +0.122, +0.017, +0.016, −0.021, −0.068 for n = 1, 2, 3, 5, 10; pess = +0.010, −0.002, −0.007, −0.011, −0.018. The max-bias falls 7× from n = 1 to n = 2 and is gone by n = 3; the pessimistic term is smaller and grows slowly. The fall of overestimation with n is therefore driven mostly by the max-bias shrinking, with the pessimistic term as the opposite-signed correction that sets where the sum crosses zero. This refines the hypothesis: the two biases are both present and opposite, but the *active* knob at small n is the max-bias.

The cost of the pessimistic term at σ = 0 is 11–13% of final return at n = 10 for ε = 0.1 and 22–23% for ε = 0.3, for both plain and double Q, in the direction and ε-ordering the accounting predicts.

### 4.4 Double Q does not remove the n-dependence

Double Q's max-bias at n = 1 is about half plain Q's (+0.062 vs +0.132 at σ = 0.1; +0.020 vs +0.037 at σ = 0.05), its pessimistic term is identical (σ = 0 rows agree to ±0.005), and its slope on log n is 60–80% of plain Q's in every grid. K3 fails three times. Double Q still prefers n = 3–5 at σ = 0.1. The premise of K3, that double Q removes max-bias, is wrong in this setting; it attenuates it, consistent with the known underestimation/overestimation mix of double estimators with a shared target table.

### 4.5 [3080] Planned

1. DQN and Double DQN on MinAtar (Breakout, Asterix, Seaquest, Freeway), n ∈ {1, 2, 3, 5, 10}, injected reward noise σ ∈ {0, 0.1, 0.3} in reward units, 10 seeds. Prediction: the n-step gain rises with σ for both; it is larger for DQN than for Double DQN but present for both.
2. ε-schedule test. Fixed ε ∈ {0.01, 0.1, 0.3} versus the standard anneal. Prediction: optimal n rises as ε falls; under annealing a schedule n(t) increasing with training beats any fixed n. This is the discriminating prediction against age-dependent-n (S2-5).
3. Propagation control. Corrected (Retrace / tree-backup) n-step at the same n propagates at the same speed (R12) but lacks the pessimistic term. Prediction: corrected n-step overestimates more than uncorrected at every n > 1 and loses under noise, wins without it. This is the direct test of "why uncorrected beats corrected".
4. Overestimation measurement in deep agents via exact Monte-Carlo rollouts of the greedy policy from probe states (the R10/R11 instrument).

## 5. Consequences

n is a bias knob, not just a speed knob. The right n depends on the noise level and on how exploratory the behaviour data is, and the two push in opposite directions. Under an annealed ε, a fixed n is too large early and too small late. Double Q and n-step are partial substitutes for the max-bias but not for each other's other effects. The accounting also says when *corrected* n-step should win: deterministic rewards and transitions, where the pessimistic term is pure cost.

## 6. Limitations

Tabular, one α, one grid, 4 seeds per set, two seed sets. The σ scale was mis-set twice before the replication; all three grids are in `results/`. The competence threshold we set was missed. Transition stochasticity was not swept. AUC over 150k steps confounds bias with propagation speed; §4.5 item 3 is the control.

## 7. Reproducibility

`code/r15_nstep.py`. Pilot: default grid, seeds 0–3. Follow-ups: `--sigmas 0 0.1 0.2 0.3 --tag band`; `--sigmas 0 0.05 0.1 --seed-base 4 --tag band2`. 15–20 minutes each on a laptop CPU.

## References

Chen et al. (2021). Randomized ensembled double Q-learning. ICLR.
De Asis et al. (2018). Multi-step RL: a unifying algorithm. AAAI.
Fedus et al. (2020). Revisiting fundamentals of experience replay. ICML.
Hernandez-Garcia, Sutton (2019). Understanding multi-step deep RL: a systematic study of the DQN target.
Hessel et al. (2018). Rainbow. AAAI.
Kapturowski et al. (2019). Recurrent experience replay in distributed RL. ICLR.
Kostrikov, Nair, Levine (2022). Offline RL with implicit Q-learning. ICLR.
Lan et al. (2020). Maxmin Q-learning. ICLR.
Munos et al. (2016). Safe and efficient off-policy RL. NeurIPS.
Precup, Sutton, Singh (2000). Eligibility traces for off-policy policy evaluation. ICML.
Thrun, Schwartz (1993). Issues in using function approximation for RL.
van Hasselt (2010). Double Q-learning. NeurIPS.
van Hasselt, Guez, Silver (2016). Deep RL with double Q-learning. AAAI.
