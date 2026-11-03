# Decorrelation Is Not Why Replay Works: Minibatch Correlation, Reuse, and the Target Period

*Conference-length version (RLC format). Every number here comes from this folder's results.*

## Abstract

Experience replay is justified twice: it reuses data, and uniform sampling decorrelates consecutive updates, which is said to reduce the variance of the updates and to break feedback loops. The two have not been separated, because a replay-versus-no-replay comparison changes both at once. We hold the buffer, the data, and the number of updates fixed in DQN on CartPole, and vary only how the 32 transitions in a minibatch are chosen: 32 independent draws, four contiguous windows of 8, one contiguous window of 32, or the last 32 transitions written. The target period is crossed in, at 100 and at 2500. Verdict: KILLED. At period 100, independent draws beat the fully contiguous window by 8.3 AUC points (12 of 16 seeds), short of a pre-registered bar of 20 points; at period 2500 the gap is 8.1, so a slow target does not mask it. The reuse control comes out backwards at the fast target: the last-32 sampler beats the contiguous window by 48.1 AUC points. Against uniform replay, at period 100, the last 32 transitions win by 39.7 AUC and 69.3 final return (15/16 and 14/16). Correlation does move the minibatch gradient's cosine to the buffer gradient, from 0.66 down to 0.59, and that cosine does not rank AUC (Spearman \(-0.02\)).

## 1 Introduction

Lin (1992) proposed replay so that a transition would be used more than once. Mnih et al. (2015) added a second reason: consecutive samples are strongly correlated, learning from them directly is inefficient, and sampling uniformly from the buffer "breaks these correlations and therefore reduces the variance of the updates." The second reason is in every textbook treatment of DQN. It is also separable from the first. Keep the buffer, keep the update count, and change only the dependence among the 32 indices in the minibatch.

We ran that comparison and crossed it with the target period. The hypothesis was that correlated batches would hurt by chasing a fast-moving target, and that a slow target would make the same correlation harmless. The hypothesis is dead in a specific way. The gap between independent and fully contiguous batches is about 8 AUC points at both periods, with 4 of 16 paired wins for the contiguous batch and \(p = 0.077\). The large effect has the other sign from the textbook's. With a target period of 100, training on the most recent 32 transitions, maximal correlation and no reuse, beats uniform replay, because the replay learners reach a high score and then collapse and the on-policy learner does not. With a target period of 2500 the ordering on AUC flips toward replay, and the final returns meet.

**Contributions.** (1) A controlled split of within-batch correlation from reuse: at fixed data and update count, one contiguous window of 32 costs 8.3 and 8.1 AUC points at the two target periods, below the pre-registered bar and not significant on the paired test. (2) The reversal of the no-replay baseline at a fast target: \(+39.7\) AUC and \(+69.3\) final return against uniform replay, 15/16 and 14/16. (3) A gradient-alignment measurement: correlation worsens the minibatch as an estimate of the buffer gradient, and the alignment does not predict return. The two candidate mechanisms of the collapse were not separated.

## 2 Related work

Lin (1992) and Mnih et al. (2015) are the two justifications. Buffer size and replay ratio have been varied extensively (Zhang and Sutton, 2017; Fedus et al., 2020; D'Oro et al., 2023). Those dials change how much data is reused and how old the oldest policy in the buffer is. They do not change the correlation inside a minibatch at fixed reuse. Fedus et al. (2020) find that \(n\)-step returns are what make large buffers useful. D'Oro et al. (2023) study the replay ratio together with resets.

The closest design to the last-32 sampler is Combined Experience Replay (Zhang and Sutton, 2017), which puts the newest transition into each otherwise uniform minibatch and recovers most of the loss from an oversized buffer. The last-32 sampler is the extreme of that fix: the whole batch is the newest transitions. The collapse of uniform replay at target period 100 is the regime that fix was aimed at. Nikishin et al. (2022) study a related failure mode, primacy, in which early data dominates later training. Non-iid minibatches in supervised SGD are a variance effect that adaptive step-size rules partly absorb; the measurement below asks whether the analogous variance story shows up in the return of a DQN.

## 3 Setting and the pre-registered test

Batched DQN, a \(2 \times 64\) MLP, Adam at \(10^{-3}\), \(\gamma = 0.99\), replay capacity 10,000, batch 32, Huber loss, gradient-norm clip 10. Epsilon anneals from 1 to 0.05 over 10,000 steps. One update per step, learning from step 500. CartPole with a 200-step cap, 30,000 steps, greedy evaluation every 1,000 steps. AUC is the mean of the 30 evaluations. Final return is the mean of the last three.

Samplers, all drawing 32 transitions:

- `iid`: 32 independent uniform draws from the buffer.
- `win4`: four uniform anchors, each extended to eight consecutive transitions.
- `win1`: one anchor, extended to 32 consecutive transitions.
- `recent`: the last 32 transitions written. No reuse of older data, and maximal temporal correlation.

Target period \(K \in \{100, 2500\}\). Sixteen seeds, shared across cells, 128 runs. Every 1,000 steps we record the cosine between the minibatch gradient that was applied and the gradient on a fixed uniform sample of 1,024 transitions from the buffer.

The hypothesis was that within-batch correlation hurts, and that it hurts through the target: a fast target is chased by a correlated batch, and a slow target is fixed for long enough that correlation is harmless.

- **K1.** At \(K = 100\), \(\mathrm{AUC}(\texttt{iid}) - \mathrm{AUC}(\texttt{win1}) \ge 20\), and `iid` wins on at least 12 of 16 paired seeds.
- **K2.** The same gap at \(K = 2500\) is at most half the gap at \(K = 100\).
- **K3.** At both \(K\), \(\mathrm{AUC}(\texttt{win1}) - \mathrm{AUC}(\texttt{recent}) \ge 20\). Same correlation structure, reuse against none. K3 is a control. It does not enter the verdict.

SUPPORTED if K1 and K2. INCONCLUSIVE if exactly one of them holds. KILLED if neither holds. The pre-registration also said that if correlation is irrelevant at both periods, that negative result on the textbook justification is the result to write up.

## 4 Results

### 4.1 Kill test

KILLED. K1: the AUC gap is \(+8.3\), and `iid` wins 12 of 16 seeds. The win count meets its bar. The magnitude does not meet 20, so K1 fails. K2: the gap at \(K = 2500\) is \(+8.1\), against a threshold of half of \(+8.3\), which is \(+4.2\). The gap does not shrink. K3: \(\texttt{win1} - \texttt{recent}\) is \(-48.1\) at \(K = 100\) and \(+23.9\) at \(K = 2500\). The control fails because of the fast-target sign.

### 4.2 Correlation at fixed reuse

Cell means over 16 seeds. Parentheses are standard deviations. The gradient cosine is the run-mean cosine to the 1,024-transition buffer gradient.

| \(K\) | sampler | final (sd) | AUC (sd) | grad cos |
|---:|---|---|---|---:|
| 100 | iid | 113.7 (38.1) | 119.1 (19.1) | 0.66 |
| 100 | win4 | 125.3 (38.5) | 126.6 (23.9) | 0.62 |
| 100 | win1 | 101.2 (50.8) | 110.8 (22.8) | 0.59 |
| 100 | recent | 183.0 (22.6) | 158.9 (14.2) | 0.10 |
| 2500 | iid | 146.2 (30.6) | 122.4 (16.0) | 0.51 |
| 2500 | win4 | 133.2 (17.3) | 111.9 (12.6) | 0.50 |
| 2500 | win1 | 139.2 (23.0) | 114.3 (15.1) | 0.45 |
| 2500 | recent | 139.5 (35.5) | 90.4 (12.8) | 0.15 |

`win1` minus `iid` is \(-8.3\) AUC at \(K = 100\) and \(-8.1\) at \(K = 2500\), with 4 of 16 wins for `win1` at each period (\(p = 0.077\)). On final return the same comparison is \(-12.6\) (5/16, \(p = 0.210\)) and \(-7.0\) (8/16, \(p = 1.000\)). Four windows of eight are indistinguishable from independent draws: `win4` minus `iid` is \(+7.4\) AUC at \(K = 100\) (10/16, \(p = 0.454\)) and \(-10.5\) at \(K = 2500\) (5/16, \(p = 0.210\)).

The gradient cosine does fall as the batch becomes more contiguous: 0.66, 0.62, 0.59 at \(K = 100\), and 0.51, 0.50, 0.45 at \(K = 2500\). A contiguous minibatch is a worse estimate of the gradient of the uniform-buffer objective. That degradation has no detectable ranking of the runs. Per-run Spearman of gradient cosine with AUC is \(-0.02\) over all cells, and \(+0.08\) inside the three replay samplers. The decorrelation effect the textbook describes is visible in the cosine and absent from the return, at the size this grid can see: under 10 AUC points, and the paired test is not significant (\(p = 0.077\)).

### 4.3 The last 32 transitions against replay

Paired differences, 16 seeds. A positive number means the first sampler is ahead.

| \(K\) | comparison | AUC (wins, \(p\)) | final (wins, \(p\)) |
|---:|---|---|---|
| 100 | recent − iid | \(+39.7\) (15/16, 0.001) | \(+69.3\) (14/16, 0.004) |
| 100 | win1 − iid | \(-8.3\) (4/16, 0.077) | \(-12.6\) (5/16, 0.210) |
| 100 | win4 − iid | \(+7.4\) (10/16, 0.454) | \(+11.6\) (9/16, 0.804) |
| 100 | recent − win1 | \(+48.1\) (15/16, 0.001) | \(+81.8\) (15/16, 0.001) |
| 2500 | recent − iid | \(-31.9\) (1/16, 0.001) | \(-6.7\) (7/16, 0.804) |
| 2500 | win1 − iid | \(-8.1\) (4/16, 0.077) | \(-7.0\) (8/16, 1.000) |
| 2500 | win4 − iid | \(-10.5\) (5/16, 0.210) | \(-13.0\) (8/16, 1.000) |
| 2500 | recent − win1 | \(-23.9\) (2/16, 0.004) | \(+0.3\) (8/16, 1.000) |

At \(K = 100\), `recent` beats uniform replay by 39.7 AUC points and 69.3 final-return points. Its gradient cosine is 0.10. At \(K = 2500\) the cosine is 0.15, and the post-hoc note records the `recent` cosines as spanning 0.10–0.15. The minibatch is nearly orthogonal to the buffer objective because it is the gradient of a different objective, the on-policy one, built from the latest transitions only. At the fast target that objective wins on the task. At the slow target, `recent` loses on AUC by 31.9 points (1 of 16 wins) and finishes level: final-return difference \(-6.7\), 7 of 16 wins, \(p = 0.804\).

K3 asked whether reuse helps once correlation is held at the fully contiguous extreme. At \(K = 100\), `win1` minus `recent` is \(-48.1\) AUC: throwing away the older buffer helps. At \(K = 2500\) the same difference is \(+23.9\) AUC, which does clear the control's 20-point bar, and the final-return difference is \(+0.3\). Reuse shows up in the area under a slow-target curve. It does not show up as a better policy at the end of 30,000 steps.

### 4.4 Learning curves

Mean evaluation score at 5,000 / 10,000 / 20,000 / 30,000 steps.

| \(K\) | sampler | 5k | 10k | 20k | 30k |
|---:|---|---:|---:|---:|---:|
| 100 | iid | 170.0 | 113.3 | 90.9 | 117.0 |
| 100 | win4 | 152.8 | 128.0 | 108.4 | 121.0 |
| 100 | win1 | 165.6 | 106.9 | 79.1 | 102.2 |
| 100 | recent | 143.2 | 170.5 | 163.5 | 180.6 |
| 2500 | iid | 18.2 | 107.2 | 159.4 | 150.5 |
| 2500 | win4 | 13.5 | 83.3 | 142.0 | 129.1 |
| 2500 | win1 | 10.7 | 95.1 | 149.3 | 142.1 |
| 2500 | recent | 15.3 | 47.7 | 133.4 | 127.2 |

At \(K = 100\) every replay sampler is high at 5,000 steps (170.0, 152.8, 165.6) and lower at 20,000 steps (90.9, 108.4, 79.1), in the band the notes call 80–120. `recent` moves 143.2 → 170.5 → 163.5 → 180.6. The AUC gap is that collapse, not a failure of `recent` to learn. At \(K = 2500\) all four samplers are between 10.7 and 18.2 at 5,000 steps. The notes read the wait until 10,000 steps as a target-period propagation limit. After that, the replay samplers climb (iid reaches 159.4 at 20,000 steps) and `recent` climbs less far (133.4 at 20,000, 127.2 at 30,000). AUC penalises `recent`, and the last evaluations do not.

### 4.5 Two mechanisms, not separated

The collapse at \(K = 100\) has two readings, and this pilot does not decide between them.

(a) Stale targets. Actions and next states in the buffer come from older, worse policies. With a fast target, bootstrapped values track the current network, and the mismatch between those targets and off-policy data can drive the decline after the ceiling.

(b) Forgotten failures. A competent agent stops failing, its terminal transitions leave a FIFO buffer, and values at failure states inflate until the policy breaks. The `recent` sampler cannot forget a failure: every failure it makes is in the next minibatches.

The decisive comparison, uniform replay plus a reservoir that keeps terminal transitions, at \(K = 100\), was not run. If that agent matched `recent`, the mechanism would be (b). It was not measured here, on CartPole or on anything else. Polyak targets, a replay-ratio sweep, and a MinAtar replication of the same four samplers were specified and not run.

## 5 Discussion and limitations

On this task, at this batch size, decorrelation inside the minibatch is an 8-point effect at both target periods. Replay's large effect is reuse, and it is conditional. With a fast target, on-policy minibatches of the last 32 transitions beat a buffer of 10,000, by the margins in Section 4.3, because the buffer learners collapse after a peak near 5,000 steps. With a slow target, replay wins the area under the curve and ties at the end. A sampler on this evidence is better judged by which data it reuses and how stale those data are than by how independent its draws are. That judgement is from one dense-reward environment. It is not a measurement of prioritised replay, and it is not a measurement on MinAtar.

The limitations are the scope. CartPole, one network, one learning rate, two target periods, 30,000 steps, 16 seeds, about 14 minutes on a laptop CPU. The crossover period, if there is a single one between 100 and 2500, was not located. AUC over 30,000 steps charges \(K = 2500\) for a slow start that the curves show directly, at scores of 10.7–18.2 after 5,000 steps. The gradient cosine is against a uniform buffer sample, which is the right reference for the replay objective and the wrong one for `recent`: the cosine near 0.10 is an objective gap, not an estimate of noise. `recent` at one update per step never reuses a transition older than the batch. Whether that remains an advantage when the current policy's own data are less informative was not tested. The mechanism split of Section 4.5 is open.

## 6 Conclusion

Holding the buffer, the data, and the update count fixed, within-batch correlation does not carry the textbook's weight. Fully contiguous minibatches lose to independent draws by 8.3 AUC points at target period 100 (12 of 16 seeds, \(p = 0.077\)) and by 8.1 points at period 2500, so the pre-registered test is KILLED on both the magnitude and the masking claim. The gradient's cosine to the buffer gradient falls from 0.66 to 0.59 and does not rank AUC. The large effect is the reuse control, backwards at a fast target: the last 32 transitions beat uniform replay by 39.7 AUC and 69.3 final return, because every replay sampler peaks near 5,000 steps and declines, and the on-policy sampler does not. At period 2500 that ordering flips on AUC (\(-31.9\)) and disappears in the final return (\(-6.7\), 7 of 16). Which of two collapse mechanisms produces the fast-target gap was not identified.

## References

D'Oro et al. (2023). Sample-efficient reinforcement learning by breaking the replay ratio barrier. ICLR. · Fedus et al. (2020). Revisiting fundamentals of experience replay. ICML. · Lin (1992). Self-improving reactive agents based on reinforcement learning, planning and teaching. Machine Learning. · Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature. · Nikishin et al. (2022). The primacy bias in deep reinforcement learning. ICML. · Zhang, Sutton (2017). A deeper look at experience replay. NeurIPS Deep RL workshop.
