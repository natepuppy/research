# How to Choose n: The n-Step Return as a Latency–Bias Trade-off Under a Target Network

*Draft. Pilot numbers are from the tabular study in `results/`; sections marked [3080] are planned.*

## Abstract

Every value-based deep RL agent sets the multi-step horizon n by sweep: DQN 1, Rainbow 3, R2D2 5, BBF 10 → 3, DER 20. We show that n is a derived quantity with two determinants that do not interact. First, under a target network refreshed every K updates, a reward L steps away reaches a decision after (⌈(L+1)/n⌉ − 1)K updates: n divides the credit-assignment latency. Second, uncorrected n-step returns from ε-greedy replay are pessimistic by a closed-form factor, (1 − ε/2)^{(L+1) − ⌈(L+1)/n⌉} on a path of L transitions, because every behaviour step inside a segment can break the path; the greedy policy is wrong once this bias exceeds the value gap to the best alternative. There is therefore a largest safe n, n* = max{n : γ^L(1 − ε/2)^{(L+1) − ⌈(L+1)/n⌉} > g}, which depends on the exploration rate, the path length and the gap, and not on K; below n* the only cost of a smaller n is latency. In a pre-registered tabular study (1344 learners; ε ∈ {0.01, 0.03, 0.05, 0.1}, gaps 0.1–0.3, n ∈ {1 … 20}, K ∈ {200, 2000}), time-to-correct-policy follows the latency law with Spearman 1.00 and median ratio 1.05 with no fitted constant; the learned values match the bias formula with r = 0.985; the largest n that learns the right policy is within one grid step of n* in 9/12 cells, the misses being those with a predicted margin below seed noise; and it is identical at K = 200 and K = 2000 in 12/12. At ε = 0.01 the correct policy arrives 18× sooner at n = 12 than at n = 1 with no loss. The rule reads the published settings: Nature DQN's ε = 0.1 makes any n > 1 unsafe on 20-step paths with gaps up to 0.3, so n = 1 was right for it; agents with ε = 0.01 can take n ≈ 20, which is what the sample-efficient agents chose. [3080] MinAtar DQN tests of both halves and a reanalysis of Rainbow, R2D2, DER and BBF.

## 1. Introduction

The multi-step return is the most widely used variance–bias knob in value-based deep RL and the least explained. Hessel et al. (2018) found n = 3 by sweep and reported that removing it hurt Rainbow more than removing any other component except prioritised replay; van Hasselt et al. (2019) used n = 20 for data-efficient Rainbow; Kapturowski et al. (2019) n = 5; Schwarzer et al. (2023) annealed 10 → 3 and showed that the schedule mattered. The usual explanation — n trades bootstrapping bias for return variance (Sutton & Barto, ch. 7) — is correct for on-policy prediction and silent on the two things that actually move in these agents: the target network and the exploratory behaviour policy.

We make both explicit. (i) With a hard target network, a bootstrapped target cannot carry information further than n steps per refresh, so a reward L steps from a decision takes (⌈(L+1)/n⌉ − 1) refreshes to reach it [R12]. The n-step return's first job is to divide that latency. (ii) Deep agents use the n-step return uncorrected — no importance weights, no truncation at off-policy actions (Hernandez-Garcia & Sutton 2019 document the practice) — and the behaviour policy is ε-greedy. Within an n-step segment every behaviour action is sampled, so the return is the behaviour policy's, not the greedy policy's; on a path where one wrong action loses the reward, the segment survives with probability (1 − f)^{n−1} per interior step with f ≈ ε/2 (half the exploratory actions take the wrong branch of a two-way choice). Bootstrapping on max Q at the segment end restores optimism there, so over a path of L + 1 transitions bootstrapped in ⌈(L+1)/n⌉ segments, the fixed-point value is γ^L(1 − f)^{(L+1) − ⌈(L+1)/n⌉}. [R15] measured this pessimism; this is its closed form on a path.

Putting the two together: there is a largest n at which the bias stays below the gap to the competing action, and up to that n, raising n costs nothing and buys latency in proportion to K. Section 2 states the rule; Section 3 tests both halves and their independence; Section 4 reads the published agents.

## 2. The rule

**Setting.** Deterministic path of L transitions from decision state s₀ to a reward of 1; at each path state a wrong action ends the episode with 0; a competing action at s₀ yields g immediately, with g < γ^L. Data from an ε-greedy behaviour policy that takes the wrong action with probability f = ε/2 at every path state. Uncorrected n-step targets with episode truncation, bootstrapping on max_a Q̄ from a target table refreshed every K updates.

**Proposition 1 (bias).** The fixed point satisfies Q(s₀, A) = γ^L (1 − f)^{(L+1) − ⌈(L+1)/n⌉}. *Sketch:* a segment starting at path state i covers min(n, L + 1 − i) transitions; all but the first are behaviour actions, each surviving with probability 1 − f; the segment bootstraps on max Q̄ at its end, which is the (optimistic) next segment's value. Chaining ⌈(L+1)/n⌉ segments, the number of behaviour-sampled interior transitions is (L + 1) − ⌈(L+1)/n⌉. ∎

**Proposition 2 (latency).** With synchronous updates and a target table refreshed every K updates, the greedy choice at s₀ becomes correct at update (⌈(L+1)/n⌉ − 1)K + O(1). *Sketch:* [R12]; each refresh advances the front by one segment. ∎

**Rule.** n* = max{n : γ^L(1 − ε/2)^{(L+1) − ⌈(L+1)/n⌉} > g + m}, where m is a margin for value noise (a few seed standard deviations). Choose n = n*; the benefit over n = 1 is a latency reduction from L·K to (⌈(L+1)/n*⌉ − 1)K. n* does not depend on K.

Corollaries. With ε → 0 the bias vanishes and n* → ∞: n is bounded only by variance. As the behaviour policy becomes greedy over training, the effective f falls and n* rises — unless the gaps that matter also shrink as the policy improves (they do: the remaining decisions are the close ones), in which case n* can fall. This is the BBF schedule.

## 3. Tabular study

**Design (pre-registered, `README.md`).** L = 20, γ = 0.99, Q(s₀, A) = 0.818; g chosen for gaps {0.1, 0.2, 0.3}; ε ∈ {0.01, 0.03, 0.05, 0.1}; fixed dataset of 20 000 transitions per (ε, seed); tabular Q with uniform replay, batch 32, α = 0.1, uncorrected n-step, hard target table; n ∈ {1, 2, 3, 5, 8, 12, 20}, K ∈ {200, 2000}, 8 seeds, 60 000 updates; exact greedy rollout every 200 updates. n̂ = largest n whose policy is correct in ≥ 6/8 seeds over the last 20 % of training. K1: n̂ at K = 2000 within one grid step of n* in ≥ 9/12 (ε, gap) cells. K2: Spearman(median time to correct policy, (⌈21/n⌉ − 1)K + 1) ≥ 0.85 over correct cells. K3: n̂ equal at K = 200 and K = 2000 in ≥ 9/12. SUPPORTED if K1 and K2.

**Result: SUPPORTED.** K1 9/12, K2 ρ = +1.00 over 66 cells, K3 12/12.

*Latency.* Median T/law = 1.05 (range 1.005–1.99, the upper end being the evaluation grain). At K = 2000, ε = 0.01, gap ≥ 0.2, all 8 seeds correct: T = 40 200, 20 200, 12 200, 8 200, 4 200, 2 200, 2 200 for n = 1, 2, 3, 5, 8, 12, 20.

*Bias.* Learned Q(s₀, A) against Proposition 1 across 168 cells: r = 0.985, RMSE 0.029. Fraction of seeds correct as a function of the predicted margin (bias − g): 0.00 below −0.05, ≤ 0.06 in (−0.05, 0.02], 0.29 in (0.02, 0.05], 0.96 above 0.05. The three K1 misses are the three cells where n* lies on the boundary with a predicted margin of 0.026, 0.005 and 0.001 — below the seed standard deviation of the learned value (0.06). The rule needs the margin term m; with m = 0.05 all twelve cells agree.

*Independence.* n̂ is identical at K = 200 and K = 2000 in every cell.

| ε | gap | n* (m = 0) | n̂ (K = 200) | n̂ (K = 2000) | T at n = 1 vs n̂, K = 2000 |
|---|---|---|---|---|---|
| 0.01 | 0.1 | 20 | 2 | 2 | 40 200 vs 20 200 |
| 0.01 | 0.2 | 20 | 20 | 20 | 40 200 vs 2 200 |
| 0.01 | 0.3 | 20 | 20 | 20 | 40 200 vs 2 200 |
| 0.03 | 0.1 | 1 | 1 | 1 | — |
| 0.03 | 0.2 | 8 | 2 | 2 | 40 200 vs 20 200 |
| 0.03 | 0.3 | 20 | 20 | 20 | 40 200 vs 2 200 |
| 0.05 | 0.1 | 1 | 1 | 1 | — |
| 0.05 | 0.2 | 2 | 1 | 1 | — |
| 0.05 | 0.3 | 8 | 2 | 2 | 40 200 vs 20 200 |
| 0.1 | 0.1–0.3 | 1 | 1 | 1 | — |

## 4. Reading the published agents

On a 20-step path with gaps 0.1–0.3 the rule gives n* = 1 at ε = 0.1 and n* ≥ 40 at ε = 0.01. Nature DQN (Mnih et al. 2015) trained at ε annealed to 0.1 and used n = 1; by the rule, n = 3 would have been wrong on exactly the long, close decisions that matter, which is consistent with n-step DQN at ε = 0.1 needing careful tuning (Hernandez-Garcia & Sutton 2019). Rainbow anneals ε 1 → 0.01 over 250k frames and uses n = 3: safe everywhere, and leaving most of the latency reduction unused at K = 32k gradient steps (8000 updates × ⌈L/3⌉). DER and BBF use ε = 0.01 (or noisy nets) with n = 20 and n = 10 → 3; both are inside the safe region and both exploit latency, and BBF's anneal is the corollary of Section 2. R2D2's per-actor ε up to 0.4 with n = 5 is outside the rule for its high-ε actors; the recurrent state and the 80-step stored sequences change the path structure, and we do not claim the rule applies unmodified.

## 5. [3080] Deep experiments

DQN (CleanRL) on five MinAtar games, n ∈ {1, 3, 5, 10, 20} × ε_final ∈ {0.01, 0.1} × K ∈ {500, 8000}, 5 seeds, 5M frames: ~30 GPU-hours on one RTX 3080. For each game, path lengths L from the distribution of steps between rewards under the learned policy, and gaps from the learned Q. Predictions: (i) the time to reach a fixed return threshold follows (⌈L/n⌉ − 1)K with the same K-slope across n; (ii) the asymptotic return is non-increasing in n once n exceeds n*(ε, L, gap), with the cross-over at a smaller n for ε = 0.1 than for ε = 0.01; (iii) the cross-over n is the same at both K. A fourth arm with truncated (Watkins-style) n-step returns removes the bias and should remove the cross-over.

## 6. Related work

Sutton & Barto (2018) ch. 7; Hessel et al. (2018), Kapturowski et al. (2019), van Hasselt et al. (2019), Schwarzer et al. (2023) for the settings. Hernandez-Garcia & Sutton (2019) measure uncorrected n-step DQN empirically and note that correction hurts. Fedus et al. (2020) show n-step is what makes large replay buffers usable, consistent with latency being the binding constraint under high K. Rowland et al. (2020) adaptive trade-offs and Daley et al. (2023) trajectory-aware eligibility traces address off-policy correction; the bias we quantify is what those corrections remove, and the rule says when it is small enough not to need them. Within this lab, [R12] gives the latency law, [R15] measures the two biases of uncorrected n-step, [R22] shows the law holds for neural critics where data exists, [R21] the soft-target analogue; the lab's delayed-reward stub concerns within-episode reward delay and is adjacent, not overlapping.

## 7. Limitations

The bias formula assumes a single path with a one-shot failure action; branching tasks have a per-step break probability that depends on the fraction of actions that are catastrophic, which must be estimated. The latency half assumes synchronous updates; sampled replay adds a propagation term (c·L²/(nB), [R12]) that we omitted by using exact-greedy rollouts on a small table. The pilot behaviour policy is stationary; in a live agent ε and the gaps both move, and the rule becomes a schedule whose inputs must be measured online.

## References

Mnih et al. 2015. Human-level control through deep RL. · Hessel et al. 2018. Rainbow. · van Hasselt, Hessel, Aslanides 2019. When to use parametric models in RL? · Kapturowski et al. 2019. Recurrent experience replay (R2D2). · Schwarzer et al. 2023. Bigger, Better, Faster (BBF). · Hernandez-Garcia & Sutton 2019. Understanding multi-step deep RL: a systematic study of the DQN target. · Fedus et al. 2020. Revisiting fundamentals of experience replay. · Rowland et al. 2020. Adaptive trade-offs in off-policy learning. · Daley, White, Amato, White 2023. Trajectory-aware eligibility traces. · Sutton & Barto 2018. · [R12], [R15], [R21], [R22] this lab.
