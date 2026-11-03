# A Fork Test for When a Deep-RL Run's Fate Is Sealed

*Conference-length version (RLC format). Every number in this paper comes from this folder's results.*

## Abstract

Deep RL returns vary across seeds, and runs that share their initial weights still diverge, so the variance is produced during training. This pilot measures when. A parent DQN is snapshotted and forked into children that share the snapshot and differ only in the randomness that comes after it. If the children end alike, the fate was sealed by the fork. Sixteen parents train for 30{,}000 steps of batched DQN on CartPole. Forks are at 0, 2{,}500, 5{,}000, 10{,}000, 15{,}000, 20{,}000, and 25{,}000 steps, with three children in each of three conditions: both noise streams reseeded, only exploration reseeded, or only the minibatch stream reseeded. The intraclass correlation of children's final scores, grouped by parent, reaches 0.63 at 15{,}000 steps, with bootstrap 90% interval \([0.35, 0.75]\). That is \(T/2\), the first fork that clears the pre-registered bar (ICC at least 0.5 and interval lower bound above 0.2). At that same fork the parent's own evaluation score explains none of the children's finals (\(R^2 = 0.00\)), against an ICC of 0.63. The two noise sources differ by at least 0.15 at five forks, which passes the letter of the third test, and the sign of the difference flips, so the test does not say which source matters. The recorded verdict is **SUPPORTED** (borderline): lock-in by halfway, and the locked-in state is not the learning curve. Which randomness seals it is unresolved.

## 1 Introduction

Seed variance in deep RL is large enough to change algorithm rankings (Henderson, 2018; Agarwal, 2021). Sharing the initial weights does not remove it: a 2025 Applied Sciences report finds that identical initialisations still diverge, so the randomness that acts during training is enough. What is not measured is the time at which a run's ending becomes a property of its state. Early-stopping and pruning rules act as if the visible score at time \(t\) were that property.

The measurement copied here is Frankle et al.'s instability analysis (2020), moved to RL. Fork a run at step \(t\). Children inherit the network, the optimiser state, and the replay buffer, and then follow independent randomness. The intraclass correlation of their final scores, with the parent as the group, is the fraction of ending variance already fixed by the snapshot. A curve of that correlation against \(t\) is a lock-in curve. Splitting the randomness into the exploration stream and the minibatch stream asks which noise seals the fate. Comparing the correlation to the \(R^2\) of the children's finals on the parent's score at \(t\) asks whether the fate is visible when it is decided. RL has both noise sources; the supervised analysis has the minibatch.

**Contributions.** (1) A batched DQN in which exploration noise and minibatch noise are separate generators, with an exact snapshot and fork, run on CartPole for 16 parents, 3 children, and 7 fork times. (2) Lock-in at \(T/2\): ICC 0.63, interval \([0.35, 0.75]\), at 15{,}000 of 30{,}000 steps. The same quantity at step 0, where children share only the initialisation, is 0.14 with an interval that covers 0. (3) At the lock-in fork the visible score has \(R^2 = 0.00\). The parent's identity accounts for 0.63 of the ending variance; the number on the learning curve accounts for none of it. (4) An honest reading of the noise-source test. It passes, and the sign flips (batch higher at 5{,}000 and 15{,}000, exploration higher at 10{,}000, 20{,}000, and 25{,}000). The verdict is SUPPORTED. The claim it supports is lock-in without visibility. It does not identify a culprit stream.

## 2 Related work

Henderson (2018) documented that deep RL comparisons move under seeds, implementation details, and environment stochasticity. Agarwal (2021) argued for interval estimates of aggregate performance in place of a single seed-mean. Frankle et al. (2020) forked supervised networks to study linear mode connectivity and the instability of a training run under data order. The Applied Sciences (2025) observation cited in the pilot is that identical initial weights still diverge in deep RL. The design here adds the exploration stream, which supervised forking does not have, and it compares the fork statistic to the score the experimenter can see without forking.

## 3 Setup

The agent is a batched DQN: many runs stepped as one tensor program, each with its own network, Adam state, replay buffer, and environment. The network has two hidden layers of 64 units. Adam uses learning rate \(10^{-3}\). The discount is 0.99. The replay buffer holds 10{,}000 transitions, minibatches are 32, and the target network is copied every 500 steps. \(\varepsilon\)-greedy decays from 1.0 to 0.05 over 10{,}000 steps. Learning starts at step 500. CartPole is the classic 200-step capped version, reward 1 per step, so a return of 200 is the ceiling. Evaluation is greedy, 4 episodes, every 2{,}500 steps. The final score is the mean of the last three greedy evaluations.

Three generators are distinct. One draws the initial weights and is consumed at construction. One draws environment resets and \(\varepsilon\)-greedy actions (`explore`). One draws minibatch indices (`batch`). A snapshot copies everything a run owns. A child continues the parent's trajectory exactly unless a generator is reseeded.

Parents: 16 seeds, \(T = 30{,}000\) steps. Fork times: 0, 2{,}500, 5{,}000, 10{,}000, 15{,}000, 20{,}000, 25{,}000. Children: 3 per parent, per fork, per condition. Conditions: `all` (both streams reseeded), `explore` (only exploration reseeded), `batch` (only the minibatch reseeded). The ICC is the standard one-way intraclass correlation of children's final scores grouped by parent. The interval is a bootstrap 90% interval over parents, 300 resamples, 5th to 95th percentile. \(R^2\) is the squared correlation of the child's final score with the parent's evaluation score at the fork (the last logged evaluation at or before the fork). The run took 34 minutes on an M1 Pro, CPU.

- **K1.** \(\mathrm{ICC}_{\mathrm{all}} \ge 0.5\) with interval lower bound \(> 0.2\), at some fork at or before \(T/2\).
- **K2.** \(|\mathrm{ICC}_{\mathrm{explore}} - \mathrm{ICC}_{\mathrm{batch}}| \ge 0.15\) at two or more forks, and those differences have the same sign.
- **K3.** At the first lock-in fork, the \(R^2\) of child final on the parent's visible score is at most \(\mathrm{ICC}_{\mathrm{all}} - 0.2\).

SUPPORTED if K1 and (K2 or K3). INCONCLUSIVE if only K1. KILLED if K1 fails. \(T\) is the training length, 30{,}000, so \(T/2\) is the 15{,}000-step fork. A first call to the summariser took \(T\) to be the last fork, 25{,}000, excluded 15{,}000 from the early window, and printed KILLED. The code was corrected and re-run on the saved runs. Nothing in the runs changed. The recorded verdict is the second one.

## 4 The lock-in curve

| Fork | ICC all (90% CI) | explore | batch | \(R^2\) on visible score | \(R^2\) on parent final | child mean (sd), all |
|---:|---|---:|---:|---:|---:|---|
| 0 | 0.14 \([-0.12, 0.44]\) | \(-0.11\) | 0.36 | — | 0.02 | 115.0 (31.0) |
| 2{,}500 | 0.18 \([-0.02, 0.32]\) | 0.33 | 0.35 | 0.04 | 0.09 | 121.5 (33.1) |
| 5{,}000 | 0.43 \([0.09, 0.66]\) | 0.23 | 0.51 | 0.06 | 0.34 | 123.7 (32.5) |
| 10{,}000 | 0.45 \([0.17, 0.62]\) | 0.53 | 0.32 | 0.00 | 0.21 | 127.6 (38.1) |
| 15{,}000 | **0.63 \([0.35, 0.75]\)** | 0.45 | 0.68 | **0.00** | 0.24 | 122.1 (37.2) |
| 20{,}000 | 0.39 \([0.02, 0.62]\) | 0.70 | 0.25 | 0.31 | 0.38 | 127.3 (35.1) |
| 25{,}000 | 0.76 \([0.53, 0.85]\) | 0.84 | 0.63 | 0.68 | 0.61 | 126.1 (36.9) |

*ICC of children's final scores grouped by parent. The visible-score \(R^2\) and the child mean are for `all`. Brackets are bootstrap 90% intervals.*

**K1, lock-in at halfway.** The first fork that clears both bars is 15{,}000. ICC is 0.63 and the interval starts at 0.35. The recorded verdict marks K1 true with `first such fork = 15000`. At step 0 the children share the initial weights and nothing else, and ICC is 0.14 with interval \([-0.12, 0.44]\), which covers 0. Initialisation, on this environment and this agent, contributes little. By halfway, about 60% of the variance in where a run ends is a property of the snapshot. The curve is not monotone: ICC falls to 0.39 at 20{,}000, with interval \([0.02, 0.62]\). With 16 parents the intervals have width on the order of \(\pm 0.2\), and the 20{,}000 dip sits inside the noise of the 15{,}000 value. The pass is exactly at the \(T/2\) boundary.

**K3, the fate is not on the curve.** At 15{,}000 the \(R^2\) of child final on the parent's evaluation score is 0.00, and the ICC is 0.63. K3 requires the \(R^2\) to be at most the ICC minus 0.2. The recorded comparison is 0.00 vs 0.63. K3 holds. The parent's eventual final score, which the child cannot see, has \(R^2 = 0.24\) at the same fork: even the ending of the parent run is only a weak predictor of the children at the moment the children have become alike. The visible score starts to predict later, 0.31 at 20{,}000 and 0.68 at 25{,}000, when little training remains. An early-stopping rule that reads the evaluation score at halfway is reading a number that, in this run, has no linear association with the ending the fork has already partly fixed.

**K2, the streams are not separated.** The recorded differences \(\mathrm{ICC}_{\mathrm{explore}} - \mathrm{ICC}_{\mathrm{batch}}\) are \(-0.03\) at 2{,}500, \(-0.27\) at 5{,}000, \(+0.21\) at 10{,}000, \(-0.24\) at 15{,}000, \(+0.45\) at 20{,}000, and \(+0.21\) at 25{,}000. Five of them clear 0.15 in absolute value, and each sign occurs at least twice, so the rule as coded passes. The sign is not stable: the minibatch stream has the higher ICC at 5{,}000 (0.51 against 0.23) and at 15{,}000 (0.68 against 0.45); the exploration stream has the higher ICC at 10{,}000 (0.53 against 0.32), at 20{,}000 (0.70 against 0.25), and at 25{,}000 (0.84 against 0.63). Three children and 16 parents do not resolve a winner. The pilot notes say to treat K2 as unanswered in spirit. The verdict file marks it true. Both statements are kept: the rule passes, and the comparison does not identify a noise source.

Under `all`, the child-mean final is 115.0 at the step-0 fork and 122.1 at the lock-in fork, with standard deviation 37.2 there. The ceiling is 200. Two parents hit 200 exactly. The quantity being correlated is a noisy, partly saturated score.

## 5 What the fork is measuring

At the lock-in fork the children of one parent end nearer to each other than to the children of another parent, and the parent's displayed score does not say which cluster. The snapshot contains the network, the Adam moments, the target network, and the replay buffer. This design does not take them apart. A fourth condition that would also refresh the replay buffer, and so separate "state of the network" from "state of the data," was part of the planned follow-up and was not run.

The practical implication the notes draw is narrow and negative. A rule that prunes runs by the visible score at \(T/2\) is, on this evidence, using a feature with \(R^2 = 0.00\) at the time the identity of the run already has ICC 0.63. A fork measurement is a different predictor. Building that predictor from cheap proxies (gradient norm, dormant units, the TD-error distribution) would need the fork statistic as a label. Those proxies were not trained here.

## 6 Limitations

CartPole with a 200-step cap is a poor environment for a fate statistic. Returns saturate, and DQN on CartPole is known to collapse and recover, so the final score is noisy even after three evaluations of four episodes. Sixteen parents and three children are the smallest design that yields an ICC, and the intervals show it. One architecture, one algorithm, one hyperparameter setting, one environment. The pass sits on the boundary fork, and the curve dips at the next fork. K2's pass is a pass of a magnitude rule under a sign that does not settle. The step-0 ICC of 0.14, with an interval covering 0, matches the spirit of the supervised finding that initialisation is not the whole instability, and it is one environment.

The follow-up on MinAtar with DQN and on the DeepMind Control Suite with SAC or TD3, with more parents and more children, was not run. No MinAtar or DMC number appears in these results.

## 7 Conclusion

On 16 CartPole DQNs, forking at 15{,}000 of 30{,}000 steps leaves children whose endings have ICC 0.63, while the score the parent is showing at that moment has \(R^2 = 0.00\). The fate is partly sealed by halfway, and it is not sealed in the learning curve. Exploration noise and minibatch noise both move the ICC by amounts the rule calls large, in both directions, so this sample does not say which stream does the sealing. The kill test is SUPPORTED on lock-in and on invisibility. It is borderline because the lock-in fork is exactly \(T/2\), the curve is not monotone, and the noise-source clause passes without answering its question.

## References

Henderson (2018). Deep reinforcement learning that matters. · Agarwal (2021). Deep reinforcement learning at the edge of the statistical precipice. · Frankle et al. (2020). Instability analysis of trained networks (linear mode connectivity). · Applied Sciences (2025), as cited in the pilot: identical initial weights still diverge during training.
