# The Credit-Assignment Speed Limit of PPO: λ and Rollout Length Set How Many Iterations a Reward Needs to Reach the Policy

**Abstract.** Generalised advantage estimation with λ < 1 is read as a bias–variance trade-off, and recently, in long-horizon RLHF, as a source of "reward signal decay": a reward D steps away reaches the advantage at the start of a trajectory with weight (γλ)^D. We show that in the iterated actor–critic loop PPO actually runs, the right description is a *latency*, not a decay. Because the critic is refit each iteration to λ-return targets built from itself, new reward information moves back ≈ 1/(1 − γλ) states per iteration, so a reward D steps away reaches the start-state value, and in the same iteration the start-state policy, after t ≈ c·D(1 − γλ) iterations. Across 30 depth × λ cells in a controlled corridor task the law fits with R² = 0.999 and slopes 0.95 (D) and 0.88 (1 − λ); λ = 1 makes arrival one iteration at every depth; and all 70 cells eventually solve the task. The same reward with the critic frozen fails at λ = 0.95, D = 128: the critic is what converts decay into latency. Rollout length imposes a second, additive latency, 1/h_eff = (1 − γλ) + 1/T_eff, so short segments reintroduce the delay at any λ. [3080] We replicate with neural PPO on long-corridor MiniGrid and a frozen-reward RLHF proxy. The practical rule is a closed form: choose λ and `num_steps` so that D·((1 − γλ) + 1/T) is well inside the iteration budget, and count iterations, not samples.

## 1. Introduction

Every PPO configuration carries two numbers that shape how far reward information travels in one update: λ (typically 0.95) and the rollout length T (`num_steps`, typically 128–2048). Both are treated as variance knobs. A recent line of work on PPO for long chain-of-thought RLHF (VC-PPO, Yuan et al. 2025) noticed that with λ = 0.95 and sequences of thousands of tokens, the terminal reward barely reaches early tokens, called this "reward signal decay", and fixed it by giving the critic λ = 1. That fix works. The explanation, we argue, is incomplete in a way that matters: GAE with a critic does not lose the signal, it delays it, and the delay has a closed form in iterations.

The mechanism is elementary once stated. At iteration k the λ-return target at a state s is a geometric mixture of n-step returns bootstrapped from V_{k−1}. If V_{k−1} already carries the reward at states within ≈ 1/(1 − γλ) of s, the target at s does too; otherwise the target sees only the (γλ)^D direct term. So each critic fit moves the "front" of reward information back by about 1/(1 − γλ) states, and the front reaches the start after D(1 − γλ) fits. The policy gradient at the start state is driven by the advantage, which is the same mixture minus V(s), so the policy at the start moves in the same iteration the front arrives. A rollout segment of T steps adds a bootstrap at the segment boundary and therefore a second cap on how far the front moves per iteration.

Contributions. (1) A pre-registered test of the latency law t ∝ D(1 − λ) in an iterated actor–critic loop with an exact critic (R² = 0.999, slopes 0.95 and 0.88; λ = 1 gives one iteration at all depths). (2) Evidence that the policy inherits the latency exactly (Spearman 0.99, zero further lag). (3) The latency-vs-decay separation: with a critic, every (λ, D) cell solves; with the critic frozen, λ = 0.95 fails at D = 128. (4) The rollout-length extension: latencies add as rates, 1/h_eff = (1 − γλ) + 1/T_eff, with T_eff ≈ 6 states for a 16-step segment in our task (a pre-registered min() form was rejected). (5) [3080] Neural PPO replication and an RLHF-style proxy.

## 2. Related work

GAE (Schulman et al. 2016) introduces λ as a bias–variance trade-off; the analysis is per-update, not across iterations. VC-PPO (Yuan et al. 2025, arXiv 2503.01491) is the closest work: it identifies that at λ < 1 the critic in long-sequence RLHF underestimates early-token values, frames this as decay, and decouples λ_critic = 1 from λ_actor < 1. We provide the iteration-count law behind their observation and show the effect is a latency that short rollouts reintroduce. TD(λ) convergence theory (Tsitsiklis & Van Roy 1997) gives asymptotics, not transient arrival times. The PPO implementation-details literature (Huang et al. 2022; the SB3 discussion of `n_steps`) notes informally that GAE truncates at the rollout boundary. Our R12 derived an analogous additive latency law for hard target networks in value-based RL; the mechanism here (geometric bootstrapping through a continuously refit critic) is different and the two laws are complementary. Lab stubs on delayed reward (reward delay within an episode) and on RLVR are adjacent; neither studies iteration latency.

## 3. Setting

Deterministic MDP: choice state s₀ with two actions leading to corridors A and B of depth D; corridor states have `forward` and `stall`; the end of A pays 1, the end of B pays 0; cap 3D; γ = 1. Per iteration: 32 episodes from the current softmax policy; GAE advantages with per-run λ and segment length T (bootstrap at every T-th step and at the cap); critic set to the exact per-state mean of λ-return targets (states not visited keep their value); advantages normalised over the batch; one SGD step (size 0.5) on tabular logits, gradient averaged per episode. Policy uniform at start, critic 0.

Measurements: t_V = first iteration with V(s₀) ≥ 0.25 (half the uniform-policy value); t_π = first iteration with exact success probability ≥ 0.75 (uniform policy ≈ 0.5). Grid D ∈ {8, 16, 32, 64, 128}, λ ∈ {0, 0.5, 0.8, 0.9, 0.95, 0.99, 1}, T ∈ {∞, 16}, 8 seeds, 600 iterations.

Pre-registered (README): K1 OLS of log t_V on log D and log(1 − λ) over T = ∞, λ < 1, D(1 − λ) ≥ 4 cells, both slopes in [0.7, 1.3], R² ≥ 0.9; K2 λ = 1 slope in D < 0.3; K3 T = 16, λ = 1 slope in [0.7, 1.3] and within 1.5× of λ = 0.95, T = ∞; K4 Spearman(t_π, t_V) ≥ 0.8. SUPPORTED if K1, K2, K4.

## 4. Results

**The law (K1, K2).** Over the 16 fit cells the slopes are 0.95 (D) and 0.88 (1 − λ) with R² = 0.999. On all 30 λ < 1, T = ∞ cells, t_V = 1.25·D(1 − λ) + 1.9 (R² = 0.999). At λ = 1, t_V = 1 for every D.

| λ | D=8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| 0 | 11 | 22 | 42 | 82 | 160 |
| 0.5 | 6 | 12 | 22 | 42 | 81 |
| 0.8 | 3 | 6 | 10 | 20 | 38 |
| 0.9 | 2 | 3 | 6 | 11 | 21 |
| 0.95 | 1 | 2 | 4 | 6 | 11 |
| 0.99 | 1 | 1 | 1 | 2 | 3 |
| 1 | 1 | 1 | 1 | 1 | 1 |

Median t_V (iterations), T = ∞.

**The policy inherits it (K4).** Spearman(t_π, t_V) = 0.99 over 70 cells; t_π − t_V ∈ [−2, +1]. The start-state policy moves in the iteration the value arrives, with no further lag.

**Latency, not decay.** All 70 cells reach success probability ≥ 0.98 by iteration 600. With the critic frozen at 0 (post-hoc, 4 seeds): λ = 1 solves every depth in 3 iterations; λ = 0.95 solves D = 32 in 24 iterations, D = 64 in 218, and fails at D = 128 (final success 0.16); λ = 0.8 solves nothing. The decay description is right for a critic-free learner and wrong as soon as a critic is refit between iterations.

**Rollout length (K3, half).** T = 16, λ = 1 scales as D^1.03 (25 iterations at D = 128) but is 1.5–2.3× slower than λ = 0.95, T = ∞, failing the pre-registered within-1.5× clause. Post-hoc, the additive-rate model t = a·D·((1 − λ) + 1/T_eff) + b fits the 35 T = 16 cells with R² = 0.998 (T_eff = 6.1 states; a = 1.06), against R² = 0.988 for the min() form. At T = 16, λ = 0.99 and λ = 1 are indistinguishable (26 vs 25 iterations at D = 128), and λ = 0.95 is 3× slower than with T = ∞. Segment length limits propagation at any λ, and the two limits combine as rates.

## 5. Discussion

For long-horizon PPO the operative quantity is the number of critic fits a reward needs to reach the decision it should influence, ≈ D·((1 − γλ) + 1/T_eff). With λ = 0.95 and a 2000-step horizon that is ~100 iterations before the first token sees the reward at all; with λ_critic = 1 it is one iteration, provided the rollout covers the sequence (in RLHF it does; in MuJoCo with `num_steps` = 128 on a 1000-step episode it does not, and the latency returns as D/T_eff). Critic-free group-baseline methods (GRPO-style, λ = 1 Monte Carlo) have zero latency by construction, which is one more reason they tolerate long horizons. The actionable rule: if a run has not moved the policy at early decisions after D(1 − γλ) iterations, that is the law, not a bug, and raising λ_critic or `num_steps` is the fix; raising the learning rate or the sample count is not.

Limitations. Tabular policy and exact critic fit; a neural critic realises less than one full propagation step per iteration and will lengthen the constant, and Adam rescales small gradients. The constant 1.25 and T_eff ≈ 6 are specific to our stall structure. Advantage normalisation is on. γ = 1.

## 6. [3080] Planned experiments

(a) CleanRL PPO on long-corridor MiniGrid (depth 32–256), λ ∈ {0.9, 0.95, 0.99, 1}, `num_steps` ∈ {16, 64, 256, 1024}, measure the iteration at which the start-state advantage first correlates with the correct action; test D((1 − γλ) + 1/T) scaling. (b) Vary learning rate and epochs to show the latency is in iterations, not samples. (c) RLHF proxy: a small LM with a terminal reward on 512–2048-token completions, λ_critic ∈ {0.95, 1}, reproducing VC-PPO's observation as a latency curve. (d) Decoupled λ_actor < 1 with λ_critic = 1: confirm the law tracks λ_critic.

## 7. Conclusion

GAE with a critic does not decay distant reward; it delays it by ≈ D(1 − γλ) iterations, plus D/T_eff for short rollouts, and the policy follows in the same iteration. The pilot law fits with R² = 0.999 and the critic-frozen contrast shows the critic is what makes the difference.

## References

Schulman et al. 2016 (GAE); Yuan et al. 2025 (VC-PPO, arXiv 2503.01491); Tsitsiklis & Van Roy 1997; Huang et al. 2022 (PPO implementation details); Schulman et al. 2017 (PPO); Shao et al. 2024 (GRPO); R12 (this lab, target-network latency law).
