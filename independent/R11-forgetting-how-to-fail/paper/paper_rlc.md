# Forgetting How to Fail

*Conference-length version (RLC format). Every number here comes from the results in this folder.*

## Abstract

Value-based agents on tasks with a time limit often reach the performance ceiling and then collapse. A folk explanation says that once failures stop, failure transitions leave the FIFO replay buffer and the network forgets what failure looks like. CartPole with reward 1 per step supplies a measurement with exact ground truth: any state-action pair whose action terminates the episode has \(Q^\ast(s, a) = 1\), for every policy. On 24 DQN seeds the probe value on 256 such pairs rises from 2.5 at 4k steps to 34.8 at 40k, while the buffer's terminal fraction falls from 3.55% to under 1%, and every control run collapses (2.46 collapses per run, final score 119). Keeping the first 256 terminal transitions, and drawing 4 of every 32 minibatch samples from them, holds mean failure-Q at 1.0, cuts collapses to 0.08 per run, and raises the final score to 154 (19/24 paired wins, mean difference \(+34.5\), two-sided sign test \(p = 0.0066\)). The same amount of random early data does not (2.17 collapses per run, final score 115). Failure-Q tracks the loss of terminals (Spearman \(-0.53\)) and does not time individual collapses (AUROC 0.61). The pre-registered verdict is INCONCLUSIVE.

## 1 Introduction

Every bootstrapped value is a chain of estimates that ends somewhere. In an episodic task the chain ends at a terminal transition, whose target is a reward with no bootstrap. Those transitions are where the value function touches a number that does not depend on another estimate. When a competent agent stops failing, episodes end by time-limit truncation, failure transitions age out of a FIFO buffer, and every target the network still sees is built from another estimate. The discount is then the only contractive pressure left.

That story is aimed at a familiar CartPole curve: the return reaches the ceiling and then falls, often more than once. It had no measurement with a known true value. With reward 1 per step, a transition \((s, a)\) whose action terminates the episode has \(Q^\ast(s, a) = 1\) exactly, independently of the policy and of \(\gamma\). A probe set of such pairs, collected once from a random policy and never used for training, can be read off the network at any time. We call the probe mean the failure-Q, and its excess over 1 the failure-Q inflation.

Three questions were pre-registered. Does inflation at a good evaluation predict a collapse soon after, and does it move against the buffer's terminal fraction? Does pinning terminal transitions in the minibatch prevent collapses without lowering the final score? Is the effect specific to terminals? The pilot answers them on CartPole, 24 shared seeds, three conditions. The fix and the specificity test pass. The timing test does not. By the rule written before the run, the verdict is INCONCLUSIVE.

**Contributions.** (1) A probe with exact ground truth for "forgetting how to fail": 256 terminating pairs, true value 1, read out during training. (2) A protected terminal reservoir of 256 transitions, 4 of every 32 minibatch rows: collapses fall from 2.46 to 0.08 per run and final score rises from 119 to 154, with failure-Q held at 1.0; a random reservoir of the same size yields 2.17 collapses per run and final score 115. (3) A negative mechanism result that fixes the verdict: Spearman of failure-Q with terminal fraction is \(-0.53\), and the AUROC for a collapse within two evaluations is 0.61 on 185 good evaluations, below the pre-registered 0.7, so the result is INCONCLUSIVE.

## 2 Related work

Collapse after early success is discussed as a property of the network: primacy bias and resets (Nikishin et al., 2022), dormant neurons (Sokar et al., 2023), capacity loss (Lyle et al., 2022), and normalisation (Lyle et al., 2024). The intervention in this paper leaves the network alone and changes one class of rows in the minibatch. Interference among states in TD (Liu et al., 2020; Fedus et al., 2020b) and selective replay for lifelong learning (Isele and Cosgun, 2018) are the neighbouring forgetting literature; the object here is narrower, a single class of transitions whose true action-value is a constant. Replay capacity and staleness are studied by Zhang and Sutton (2017) and Fedus et al. (2020a). Prioritised replay (Schaul et al., 2016) up-weights large TD errors, which terminals need not have while their targets are still correct. Time-limit handling (Pardo et al., 2018) is a different bug; truncation at 200 steps is bootstrapped in every condition here, so a truncation is not stored as a terminal. Overestimation (Thrun and Schwartz, 1993; van Hasselt, Guez, and Silver, 2016) is the family failure-Q belongs to, localised to the state-actions whose true value is known exactly.

## 3 Method

**Probe.** A random policy is rolled out in CartPole. The probe set is 256 pairs \((s, a)\) for which stepping \((s, a)\) terminates the episode (\(|\theta| > 12^\circ\) or \(|x| > 2.4\)). Then \(Q^\ast(s, a) = 1\) for every pair. Failure-Q at a logged time is the mean of the current network's \(Q(s, a)\) over the probe. The probe is not trained on.

**Conditions.** `reservoir_term` stores the first 256 terminal transitions the run sees in a buffer that is never overwritten. Every minibatch of 32 contains 28 samples from the FIFO buffer and 4 from the reservoir. `reservoir_rand` fills a protected buffer of the same size with every fourth transition from the start of training, early data from a policy that is still near random, and samples it the same way. `control` draws all 32 samples from the FIFO buffer. The terminal reservoir is 12.5% of each minibatch; control's buffer terminal fraction ends under 1%.

**Agent.** DQN with two hidden layers of 64, Adam at \(10^{-3}\), \(\gamma = 0.99\), FIFO buffer of 10k, batch 32, hard target updates every 500 steps, \(\varepsilon\) annealed from 1 to 0.05 over 10k steps, Huber loss, gradient-norm clip 10. Episodes truncate at 200 steps and the truncation is bootstrapped. Training runs for 40k steps. Greedy evaluation uses 4 episodes every 1k steps. Seeds are shared across the three conditions, 24 seeds each.

**Outcomes, fixed before the run.** A collapse is an evaluation at most 100 after the run has already recorded an evaluation at least 175. The final score is the mean of the last five evaluations. K1 (mechanism): on control runs, failure-Q at an evaluation where the return is at least 175 predicts a collapse within the next two evaluations with AUROC at least 0.7, and failure-Q is negatively rank-correlated with the buffer's terminal fraction at Spearman at most \(-0.5\). K2 (fix): the terminal reservoir reduces collapses per run by at least 40% relative to control and does not lower the final score by more than 5 points. K3 (specificity): the random reservoir achieves less than half of the terminal reservoir's reduction. SUPPORTED requires K1 and K2. Exactly one of them is INCONCLUSIVE. Neither is KILLED. K3 is reported with the verdict.

## 4 Results

### 4.1 What the control runs do

In control, the terminal fraction of the buffer falls from 3.55% at 4k steps to 0.83% at 16k and stays near 0.9% through 40k. The minimum terminal fraction recorded for control is 0.60%. Failure-Q rises from 2.5 at 4k to 15.2 at 20k to 34.8 at 40k, against a true value of 1. The rank correlation between failure-Q and terminal fraction, over control run-evaluation pairs, is \(-0.53\). Every control run collapses at least once: 2.46 collapses per run, 24/24 runs, mean failure-Q 21.2, final score 119 with standard deviation 40.

The same trace under the terminal reservoir stays between 1.0 and 1.2 at the logged checkpoints, and the run-mean failure-Q is 1.0, while the FIFO terminal fraction still falls (3.51% at 4k, 0.44% at 20k, minimum 0.28%). The random reservoir slows the probe drift and does not stop it: failure-Q is 2.6 at 4k, 8.5 at 20k, and 11.3 at 40k, with mean failure-Q 9.3. Random-policy data is about 4% terminal, so that reservoir contains a handful of terminals. Its minimum buffer terminal fraction is 0.57%.

| Step | Control score / failure-Q / term % | Terminal reservoir | Random reservoir score / failure-Q |
|---:|---|---|---|
| 4k | 155.6 / 2.5 / 3.55 | 157.1 / 1.2 / 3.51 | 156.7 / 2.6 |
| 8k | 150.9 / 3.7 / 2.28 | 173.4 / 1.2 / 2.16 | 169.4 / 3.8 |
| 12k | 121.2 / 5.1 / 1.28 | 154.0 / 1.2 / 1.02 | 111.9 / 5.3 |
| 16k | 122.1 / 7.7 / 0.83 | 167.7 / 1.1 / 0.46 | 112.6 / 6.6 |
| 20k | 122.3 / 15.2 / 0.87 | 167.6 / 1.0 / 0.44 | 100.2 / 8.5 |
| 24k | 126.9 / 21.9 / 0.86 | 165.4 / 1.0 / 0.44 | 100.8 / 10.6 |
| 28k | 109.3 / 25.9 / 0.88 | 152.3 / 1.0 / 0.46 | 112.0 / 9.7 |
| 32k | 108.7 / 29.4 / 0.90 | 152.2 / 1.1 / 0.50 | 115.9 / 11.1 |
| 36k | 113.4 / 33.1 / 0.92 | 157.4 / 1.0 / 0.50 | 120.0 / 11.6 |
| 40k | 134.0 / 34.8 / 0.90 | 147.9 / 1.0 / 0.50 | 108.4 / 11.3 |

*Table 1. Means over 24 seeds. Terminal-reservoir columns are score / failure-Q / FIFO terminal percent. Random-reservoir columns are score / failure-Q.*

By the end of control training the probe says that an action which ends the episode in one step is worth 34.8 steps of reward. Under the terminal reservoir the same probe stays at 1.0, the true value, at the 20k, 24k, 28k, 36k, and 40k means.

### 4.2 Pinning terminals

| Condition | Collapses / run | Runs with \(\ge 1\) collapse | Final score (sd) | Mean failure-Q (true value 1) | Min. buffer terminal % |
|---|---:|---:|---:|---:|---:|
| Control | 2.46 | 24/24 | 119 (40) | 21.2 | 0.60 |
| Terminal reservoir | 0.08 | 2/24 | 154 (23) | 1.0 | 0.28 |
| Random reservoir | 2.17 | 23/24 | 115 (31) | 9.3 | 0.57 |

*Table 2. 24 shared seeds, 40k steps. Paired on seed, the terminal reservoir beats control on final score in 19/24 runs, mean difference \(+34.5\), two-sided sign test \(p = 0.0066\).*

K2 passes. Collapses per run fall from 2.46 to 0.08, a 97% reduction, against a pre-registered bar of 40%, and the final score rises from 119 to 154 rather than falling. K3 passes. The random reservoir's reduction is 12%, from 2.46 to 2.17, which is less than half of the terminal reservoir's reduction. The standard deviation of the final score falls from 40 to 23. Two of 24 terminal-reservoir runs still collapse at least once, against 24 of 24 control runs and 23 of 24 random-reservoir runs.

The final-score gain is a shift of the bulk of the distribution, not a pile-up at the ceiling. Terminal-reservoir finals have minimum 121 and median 150, and the fraction of runs whose final score is at least 175 is 0.17, against 0.12 in control. The learning curves separate earlier than the finals: at 8k the terminal-reservoir mean score is 173.4 against 150.9, and at 12k it is 154.0 against 121.2, when control failure-Q is 5.1 and the control terminal fraction is 1.28%.

### 4.3 Inflation does not time collapses

K1 does not pass. The Spearman clause does: \(-0.53\) against a bar of \(-0.5\). The AUROC clause does not. Failure-Q at a good evaluation (return at least 175) predicts a drop to at most 100 within the next two evaluations with AUROC 0.61, on 185 good evaluations of which 45 are followed by a collapse, against a bar of 0.7. Extending the horizon does not raise it. Within five evaluations the AUROC is 0.61 (\(n = 173\), 79 positives). Within ten it is 0.53 (\(n = 153\), 115 positives).

Across the 24 control runs, mean failure-Q does not predict the number of collapses (Spearman \(-0.20\)) or the final score (Spearman \(-0.11\)). Spearman of the maximum failure-Q with area under the curve is \(+0.35\). All 24 control runs have both a collapse and a crossing of failure-Q above 5. The gap between the first collapse and the first time failure-Q exceeds 5 has median 0, minimum \(-11000\) steps, and maximum \(26000\) steps. The fraction of runs in which the inflation crossing came first is 0.38. Failure-Q rises smoothly in every control run in Table 1. Collapses are scattered through training. The rank correlation with the terminal fraction says the probe moves when terminals leave. The AUROC says that movement does not pick the evaluation at which the return falls.

The rule needs K1 and K2 for SUPPORTED. K2 holds and K1 does not, and K3 holds. The recorded verdict is INCONCLUSIVE.

## 5 Discussion and limitations

The comparison that is determined by this pilot is the intervention. A protected set of 256 terminal transitions, four rows in every minibatch of 32, is associated with a drop from 2.46 collapses per run to 0.08, a final-score increase of 34.5 points on paired seeds (19/24, \(p = 0.0066\)), and a probe value held at the true value 1 while control's probe ends at 34.8. An equal amount of early non-terminal-heavy data does not reproduce the collapse reduction (2.17 per run) or the final score (115) and only slows the probe, to a mean of 9.3 and an endpoint of 11.3. The specificity control is what makes "any old data" and "a slightly different buffer" poor accounts of Table 2. The terminal fraction inside the FIFO buffer still collapses under the reservoir, down to a minimum of 0.28%, so the anchor that remains is the protected set, not a FIFO that happens to retain failures.

The comparison that is not determined is why the return collapses when it does. Two readings fit the same tables. On the anchor reading, terminals are the only targets without a bootstrap; removing them removes the only data constraint on the scale of \(Q\), the discount pulls the scale back only at rate \(1 - \gamma\) per hop, and a greedy policy on a drifting value function is brittle on states that are not themselves failures. Failure-Q is then a thermometer. On the boundary reading, the network loses the ranking between safe and fatal actions near the failure set, and collapses are visits to that set. The pilot cannot separate them. Failure-Q is monotone, the AUROC against the next collapse is 0.61, and in 0.38 of control runs the probe crosses 5 before the first collapse, with the median gap equal to 0. A monotone marker of how many terminals remain will not time an episodic event. That is why K1 fails, and why the verdict is INCONCLUSIVE rather than a confirmation of the folk story's causal step.

The reservoir changes the composition of the minibatch, to 12.5% terminals against under 1% in control, and it changes what is remembered. This design does not separate reweighting from protection. Collapse is defined on 4-episode evaluations with thresholds 175 and 100 that were fixed before the run and are still a choice. The buffer holds 10k transitions, a size the notes describe as turning over every 50 good episodes, at one network size. MinAtar and MuJoCo experiments were not run.

## 6 Conclusion

On CartPole, the action-value of terminating actions is a known constant, 1, and a DQN trained from a FIFO buffer drives its estimate of that constant from 2.5 to 34.8 as terminals fall below 1% of the buffer. Every one of 24 control runs collapses at least once. Putting 256 terminal transitions back into every minibatch holds the estimate at 1.0, cuts collapses from 2.46 to 0.08 per run, and raises paired final scores by 34.5 points, while 256 random early transitions do not. The same probe does not predict which good evaluation is about to collapse (AUROC 0.61). Terminal forgetting, as measured here, is a chronic loss of the only unbootstrapped targets in the loss. The pre-registered verdict is INCONCLUSIVE.

## References

Fedus et al. (2020a). Revisiting fundamentals of experience replay. ICML. · Fedus et al. (2020b). Catastrophic interference in reinforcement learning: a solution based on context division. arXiv:2007.03920. · Isele, Cosgun (2018). Selective experience replay for lifelong learning. AAAI. · Liu et al. (2020). Towards a practical measure of interference for reinforcement learning. arXiv:2007.03807. · Lyle et al. (2022). Understanding and preventing capacity loss in reinforcement learning. ICLR. · Lyle et al. (2024). Normalization and effective learning rates in reinforcement learning. NeurIPS. · Nikishin et al. (2022). The primacy bias in deep reinforcement learning. ICML. · Pardo et al. (2018). Time limits in reinforcement learning. ICML. · Schaul et al. (2016). Prioritized experience replay. ICLR. · Sokar et al. (2023). The dormant neuron phenomenon in deep reinforcement learning. ICML. · Thrun, Schwartz (1993). Issues in using function approximation for reinforcement learning. · van Hasselt, Guez, Silver (2016). Deep reinforcement learning with double Q-learning. AAAI. · Zhang, Sutton (2017). A deeper look at experience replay. arXiv:1712.01275.
