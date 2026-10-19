# Forgetting How to Fail: Terminal Transitions Anchor Value Learning, and Replay Lets Them Go

*Draft v0.1 (pilot scale). Sections marked [3080] are planned; all numbers come from the CPU pilot in `../results/`.*

## Abstract

Value-based agents on tasks with a time limit have a familiar failure mode: they reach the performance ceiling and then collapse. A folk explanation says that once the agent is good, failures stop happening, failure transitions leave the FIFO replay buffer, and the network forgets what failure looks like. We give this explanation a measurement with exact ground truth. In CartPole, a state-action pair whose action terminates the episode has Q*(s, a) = 1 for every policy, so a fixed probe set of such pairs yields a quantity whose true value is known at every point in training. In 24 DQN runs, the network's value for these actions drifts from 2 to 35 over 40k steps while terminals fall below 1% of the buffer, and every run collapses at least once (2.5 collapses per run). Pinning the first 256 terminal transitions in the minibatch stream holds the probe value at 1.0, cuts collapses to 0.08 per run, and raises the final score by 35 points (19/24 paired wins, p = 0.007). The same amount of random old data has no effect. Inflation does not, however, time individual collapses (AUROC 0.61), so terminal forgetting is a chronic loss of anchoring rather than a trigger. [3080] We separate "anchor" from "boundary knowledge" with a flag-removed reservoir, measure overestimation against Monte-Carlo truth on non-failure states, and test on MinAtar and MuJoCo.

## 1. Introduction

Every bootstrapped value estimate is a chain of estimates that ends somewhere. In an episodic task the chain ends at a terminal transition, whose target is a bare reward with no bootstrap; these are the only places where the value function touches ground truth. In a continuing task, or an episodic one whose agent has learned never to fail, there are no such places: every target is built from another estimate, and the only thing that keeps values finite is the discount.

This suggests a specific mechanism for a specific, widely observed phenomenon. Deep Q-learning on CartPole, and more generally on tasks with a time limit that a good agent always reaches, is known to reach the ceiling and then collapse, often repeatedly (see e.g. the training curves in Nikishin et al., 2022, and countless practitioner reports). The folk account: once the agent is good, episodes end by truncation rather than failure; failure transitions age out of the FIFO buffer; the network, trained on nothing but bootstrapped targets, forgets that certain actions end the episode; it starts to take them; it collapses.

The account has not been tested, because "forgets what failure looks like" has had no measurement. We provide one. With reward 1 per step, a transition (s, a) whose action terminates the episode has Q*(s, a) = 1 exactly, independently of the policy and the discount. A probe set of such pairs, collected once from a random policy and never used for training, gives a scalar with a known true value that can be read off the network at any time. We call its excess over 1 the *failure-Q inflation*.

With this instrument we ask three pre-registered questions. Does inflation rise as terminals leave the buffer, and does it precede collapses? Does pinning terminal transitions in the training stream prevent collapses without costing performance? Is that effect specific to terminals, or does any old data do? The answers are: yes and no; yes; yes.

## 2. Related work

**Collapse and plasticity.** Primacy bias and resets (Nikishin et al., 2022), dormant neurons (Sokar et al., 2023), capacity loss (Lyle et al., 2022), normalisation as a plasticity fix (Lyle et al., 2024). These treat collapse as a property of the network. We treat it as a property of the *data stream*: the network is fine; it has been shown nothing but bootstrapped targets for 30k steps.

**Catastrophic forgetting within a task.** Interference between states in TD (Liu et al., 2020; Fedus et al., 2020b) and selective replay for lifelong learning (Isele & Cosgun, 2018). Our contribution is narrower and sharper: one class of transitions (terminals), one exact ground truth, one intervention.

**Replay composition.** Buffer size and staleness (Fedus et al., 2020a; Zhang & Sutton, 2017); prioritised replay (Schaul et al., 2016), which up-weights high-TD-error transitions and would up-weight terminals only if their error were large, which it is not until it is too late. Time-limit handling (Pardo et al., 2018) is a different issue; we bootstrap through truncation correctly, so our effect is not the truncation-as-terminal bug.

**Overestimation.** Thrun & Schwartz (1993); van Hasselt et al. (2016). Our failure-Q inflation is a form of overestimation localised to the one place where the true value is known exactly; whether it is a symptom of global overestimation is the main open question (Section 6).

## 3. Method

**Instrument.** Run a random policy in CartPole and collect 256 pairs (s, a) such that stepping (s, a) terminates (|θ| > 12° or |x| > 2.4). Q*(s, a) = 1 for all of them. Failure-Q at time t is the mean over the probe set of the network's Q_t(s, a).

**Intervention.** `reservoir_term`: the first 256 terminal transitions the run experiences are stored in a protected buffer that is never overwritten; every minibatch of 32 contains 28 samples from the FIFO buffer and 4 from the reservoir. `reservoir_rand`: the same protected buffer filled with every fourth transition from the start of training (early, mostly random-policy data, almost all non-terminal), to control for "any old data". `control`: all 32 samples from the FIFO buffer.

**Agent.** DQN, two hidden layers of 64, Adam 1e-3, γ = 0.99, FIFO buffer 10k, batch 32, hard target update every 500 steps, ε annealed 1 → 0.05 over 10k steps, Huber loss, gradient-norm clip 10, truncation at 200 steps bootstrapped (not terminal). 24 seeds per condition, identical across conditions. Greedy evaluation every 1k steps on 4 fixed episodes.

**Outcomes.** Collapse: an evaluation ≤ 100 after the run has had an evaluation ≥ 175. Final score: mean of the last 5 evaluations. Pre-registered tests (README): K1 (mechanism), K2 (fix), K3 (specificity).

## 4. Results

### 4.1 The network forgets how to fail

In control runs the terminal fraction of the buffer falls from 3.6% at 4k steps to 0.9% by 16k and stays there. Failure-Q rises monotonically from 2.5 (4k) to 15 (20k) to 35 (40k), against a true value of 1. Rank correlation between failure-Q and terminal fraction over all control (run, eval) pairs: −0.53. With the terminal reservoir, failure-Q is 1.0 ± 0.1 for the entire run. With the random reservoir it rises to 11: the random reservoir contains a handful of terminals (random-policy data is ~4% terminal), which is enough to slow the drift but not stop it.

### 4.2 Pinning terminals prevents collapse

| condition | collapses / run | runs with ≥ 1 collapse | final score (sd) | failure-Q (true 1) |
|---|---|---|---|---|
| control | 2.46 | 24 / 24 | 119 (40) | 21.2 |
| reservoir_term | **0.08** | **2 / 24** | **154 (23)** | **1.0** |
| reservoir_rand | 2.17 | 23 / 24 | 115 (31) | 9.3 |

*Table 1. 24 seeds per condition, 40k steps. Paired on seed, reservoir_term beats control on final score in 19/24 runs, mean +34.5 (two-sided sign test p = 0.007).*

K2 passes: a 97% reduction in collapses and a higher final score. K3 passes: the random reservoir's reduction is 12%, well under half of the terminal reservoir's. The terminal reservoir also halves the across-seed standard deviation of the final score.

### 4.3 Inflation is chronic, not a trigger

K1 half-passes. The correlation with terminal fraction holds, but failure-Q at a good evaluation predicts a collapse within the next two evaluations with AUROC 0.61 (n = 185 good evaluations, 45 followed by collapse), below the pre-registered 0.7; extending the horizon to 5 or 10 evaluations does not help (0.61, 0.53). Across control runs, mean inflation does not predict the number of collapses (Spearman −0.20). The time course explains why: inflation rises smoothly and monotonically in every run, while collapses are scattered through training. Terminal forgetting sets the stage; something else picks the moment.

## 5. Discussion

The result we are confident in: a 256-transition protected memory of terminals, costing nothing in compute, removes a failure mode that otherwise hits every run, and it does so through the terminals specifically. The result we are not yet confident in: *why*. Two readings fit the data.

(a) **Anchor.** Terminals are the only targets without a bootstrap. Removing them from the loss removes the only constraint on the value scale; overestimation then grows everywhere (the discount alone pulls it back only at rate 1 − γ per hop), and a greedy policy on an inflated, drifting value function is brittle in ways not specific to failure states. On this reading failure-Q is a convenient *thermometer* for a global loss of anchoring.

(b) **Boundary.** The network specifically loses the ranking between safe and fatal actions near the failure boundary, and collapses are visits to that boundary. On this reading the fix works because it maintains boundary knowledge.

The pilot cannot separate these. The reading matters for what the method generalises to: (a) says the reservoir should help in any episodic task once the agent stops failing, and should be replaceable by any scale anchor (Pop-Art, reward centering with a known average); (b) says it is a safety-boundary method.

## 6. [3080] Planned experiments

1. **Flag-removed reservoir.** Keep the same 256 terminal transitions but store them with done = 0 (bootstrapped like any other). Same states, no anchor. If (a), collapses return; if (b), they do not.
2. **Global overestimation.** Monte-Carlo returns of the greedy policy from 256 buffer states every 1k steps; Q − MC on non-failure states over time, per condition.
3. **Per-run causality.** Fork a good control run at step t (R7 machinery); add the reservoir to half the children. Collapse rate of children with and without, as a function of t.
4. **Scale.** MinAtar Breakout and Asterix (terminals), Freeway (no terminals: predicted null), MuJoCo Hopper and Walker2d with SAC (falls are terminal). Buffer sizes 10⁴–10⁶; prediction: the effect shrinks as the buffer grows.
5. **Reweighting vs memory.** A condition that prioritises terminals from the FIFO buffer without protecting them separates the "12.5% terminals in the minibatch" effect from the "terminals never leave" effect.
6. **Comparison.** Resets, LayerNorm, Pop-Art on the same seeds; cost and collapse rate.

## 7. Limitations

One environment, one architecture, one buffer size; collapse thresholds (175, 100) were fixed before the run but remain a choice; 4-episode evaluations are noisy. The reservoir changes the minibatch composition as well as its memory. The mechanism question in Section 5 is open.

## 8. Reproducibility

`code/r11_forgetting.py` reproduces the pilot in ~7 CPU-minutes. The README records the pre-registered tests; `results/posthoc.txt` records every analysis that was not pre-registered.

## References

- Fedus, W. et al. (2020a). Revisiting fundamentals of experience replay. ICML.
- Fedus, W. et al. (2020b). Catastrophic interference in reinforcement learning: a solution based on context division. arXiv:2007.03920.
- Isele, D. & Cosgun, A. (2018). Selective experience replay for lifelong learning. AAAI.
- Liu, V. et al. (2020). Towards a practical measure of interference for reinforcement learning. arXiv:2007.03807.
- Lyle, C. et al. (2022). Understanding and preventing capacity loss in reinforcement learning. ICLR.
- Lyle, C. et al. (2024). Normalization and effective learning rates in reinforcement learning. NeurIPS.
- Nikishin, E. et al. (2022). The primacy bias in deep reinforcement learning. ICML.
- Pardo, F. et al. (2018). Time limits in reinforcement learning. ICML.
- Schaul, T. et al. (2016). Prioritized experience replay. ICLR.
- Sokar, G. et al. (2023). The dormant neuron phenomenon in deep reinforcement learning. ICML.
- Thrun, S. & Schwartz, A. (1993). Issues in using function approximation for reinforcement learning.
- van Hasselt, H., Guez, A., Silver, D. (2016). Deep reinforcement learning with double Q-learning. AAAI.
- Zhang, S. & Sutton, R. S. (2017). A deeper look at experience replay. arXiv:1712.01275.
