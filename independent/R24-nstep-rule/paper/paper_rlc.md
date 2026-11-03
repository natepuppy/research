# How to Choose \(n\): Latency against Bias under a Target Network

*RLC-length version; every number is from this folder's results.*

## Abstract

The backup length \(n\) does two jobs that do not interact. Under a hard target refreshed every \(K\) updates, a reward \(L\) steps from a decision arrives after \((\lceil (L+1)/n \rceil - 1)K\) updates, so \(n\) divides a latency. Uncorrected \(n\)-step returns drawn from \(\varepsilon\)-greedy behaviour are pessimistic by a closed factor \((1 - \varepsilon/2)^{(L+1) - \lceil (L+1)/n \rceil}\) on a path, and the greedy policy is wrong once that bias swallows the gap to the best alternative. The largest safe \(n\) is therefore

\(n^\ast = \max\bigl\{ n : \gamma^L (1 - \varepsilon/2)^{(L+1) - \lceil (L+1)/n \rceil} > g \bigr\}\),

which depends on \(\varepsilon\), the path, and the gap, and not on \(K\). Below \(n^\ast\), a smaller \(n\) buys nothing but waiting. In a pre-registered tabular study, 1,344 learners, the time to a correct greedy policy follows \((\lceil 21/n \rceil - 1)K + 1\) at Spearman correlation \(+1.00\) over 66 cells, median ratio 1.050, with no fitted constant. Learned \(Q(s_0, A)\) matches the bias formula at Pearson \(r = 0.985\) and RMSE 0.029 over 168 cells. The largest \(n\) that learns the right policy is within one grid step of \(n^\ast\) in 9 of 12 cells at \(K = 2{,}000\), and it is the same at \(K = 200\) and \(K = 2{,}000\) in 12 of 12. The three misses are the cells whose predicted margin at \(n^\ast\) is \(+0.026\), \(+0.005\), and \(+0.001\), below the within-cell seed noise of the learned value (median standard deviation 0.064). The recorded verdict is **supported**.

## 1 Introduction

Every value-based deep agent sets \(n\) by sweep: DQN at 1, Rainbow at 3, R2D2 at 5, DER at 20, BBF annealed from 10 to 3. The textbook account, a trade of bootstrap bias against return variance, is the right account of on-policy prediction. It does not mention the two objects that move in these agents. The first is the target network. A bootstrapped target cannot carry information further than \(n\) steps per refresh, so a reward \(L\) steps from a decision takes \((\lceil (L+1)/n \rceil - 1)\) refreshes to arrive. The second is the behaviour policy. Deep agents use the \(n\)-step return uncorrected — no importance weights, no cut at the first exploratory action — and the behaviour policy is \(\varepsilon\)-greedy. Inside a segment, every step after the first is a behaviour action. On a path where one wrong action loses the reward, the segment survives with probability \((1-f)^{n-1}\) on its interior steps, with \(f = \varepsilon/2\) the chance that the exploratory half of \(\varepsilon\)-greedy takes the wrong branch of a two-way choice. Bootstrapping on \(\max_a \bar Q\) at the segment boundary restores the optimistic value there. Chaining the segments, the fixed point on a path of \(L\) transitions is \(\gamma^L (1-f)^{(L+1) - \lceil (L+1)/n \rceil}\).

The two pieces give a rule. Raise \(n\) until the pessimistic bias approaches the gap, and stop. Up to that point the only cost of a smaller \(n\) is latency proportional to \(K\). Past that point the greedy action at the decision is the wrong one, at every \(K\).

**Contributions.** (1) The fixed-point formula for the uncorrected return on a path, and the threshold \(n^\ast\) it defines. (2) A pre-registered test of both halves and of their independence: Spearman \(+1.00\) on the latency, 9/12 on the threshold, 12/12 on independence of \(K\). (3) The noise reading of the three misses, from a post-hoc margin table. (4) The reading of the published settings on this 20-step path: at \(\varepsilon = 0.1\) the safe \(n\) are those at most 1, and at \(\varepsilon = 0.01\) the closed form still clears all three gaps out to \(n \le 40\).

## 2 The rule

**Path.** From a decision state \(s_0\), action \(A\) enters a corridor of \(L = 20\) states and action \(B\) terminates with reward \(g\). In the corridor, `forward` advances and pays 0, and the last forward pays 1; `fall` terminates with reward 0. The discount is \(\gamma = 0.99\), so the optimistic value is \(Q(s_0, A) = \gamma^L = 0.818\). The gaps used below are 0.1, 0.2, and 0.3, i.e. \(g \in \{0.718, 0.618, 0.518\}\).

**Proposition 1 (bias).** At the fixed point of uncorrected \(n\)-step TD with bootstrap on \(\max_a \bar Q\),

\(Q(s_0, A) = \gamma^L (1-f)^{(L+1) - \lceil (L+1)/n \rceil}\),

where \(f\) is the per-step probability that behaviour leaves the path. A segment that starts at path index \(i\) covers \(\min(n, L+1-i)\) transitions. All but the first are behaviour actions, each surviving with probability \(1-f\). The segment then bootstraps on the max, which is the next segment's value. Over \(\lceil (L+1)/n \rceil\) segments the number of behaviour-sampled interior transitions is \((L+1) - \lceil (L+1)/n \rceil\).

**Proposition 2 (latency).** With a target table refreshed every \(K\) updates, the greedy choice at \(s_0\) becomes correct at update \((\lceil (L+1)/n \rceil - 1)K\), up to a small additive fitting delay. Each refresh advances the front by one segment. On this path \(L+1 = 21\), and the predictor used in the test is \((\lceil 21/n \rceil - 1)K + 1\).

**Rule.** \(n^\ast\) is the largest \(n\) at which the fixed point still exceeds the competing reward \(g\), optionally by a margin \(m\) for value noise. Choose that \(n\). The benefit relative to \(n = 1\) is a latency that falls from about \(L \cdot K\) to \((\lceil (L+1)/n^\ast \rceil - 1)K\). The threshold does not depend on \(K\). As \(\varepsilon \to 0\) the bias vanishes and \(n^\ast\) is bounded only by variance. If the behaviour policy becomes greedier during training, \(f\) falls and the safe set grows, unless the gaps that remain are the close ones, in which case \(n^\ast\) can shrink. That second case is the shape of an anneal from a large \(n\) to a smaller one.

## 3 Related work

Sutton and Barto (2018, ch. 7) treat \(n\) as a bias–variance knob for on-policy prediction. Hessel et al. (2018) set Rainbow's \(n = 3\) by sweep and report that removing the multi-step return hurt more than removing any other component except prioritised replay. Kapturowski et al. (2019) use \(n = 5\). van Hasselt, Hessel, and Aslanides (2019) use \(n = 20\) in data-efficient Rainbow. Schwarzer et al. (2023) anneal \(n\) from 10 to 3 and show that the schedule matters; the corollary above is the same shape, read off \(\varepsilon\) and the gap rather than off a sweep. Hernandez-Garcia and Sutton (2019) measure uncorrected \(n\)-step DQN and note that correction hurts. Fedus et al. (2020) show that multi-step returns are what make a large replay buffer usable, which is what a latency account predicts when \(K\) is the binding constraint. Rowland et al. (2020) and Daley, White, Amato, and White (2023) build adaptive and trajectory-aware corrections. The bias in Proposition 1 is the quantity those corrections remove, and the margin table below says when it is already smaller than the gap. The latency half is the hard-target hop count on a path of 21 transitions. A separate measurement of uncorrected \(n\)-step bias, and a separate measurement that the hop count survives function approximation where data exist, are not re-run here. The numbers below are this grid's.

## 4 Tabular study

**Design, fixed before the pilot.** For each \((\varepsilon, \mathrm{seed})\) a dataset of 20,000 transitions is drawn from a behaviour policy that picks \(A\) or \(B\) at random and, in the corridor, falls with probability \(f = \varepsilon/2\). The learner is tabular Q, uniform replay from that fixed dataset, batch 32, step size \(\alpha = 0.1\), uncorrected \(n\)-step targets with truncation at episode boundaries, hard target table every \(K\) updates. The grid is \(\varepsilon \in \{0.01, 0.03, 0.05, 0.1\}\), gaps \(\{0.1, 0.2, 0.3\}\), \(n \in \{1, 2, 3, 5, 8, 12, 20\}\), \(K \in \{200, 2000\}\), eight seeds: 168 cells and 1,344 learners, 60,000 updates. The longest predicted latency on the grid is 40,001, which is inside that budget. Every 200 updates the greedy policy is rolled out exactly from \(s_0\); correct means it reaches the goal. \(T\) is the first correct update.

A smoke run of two seeds and 8,000 updates, before the pilot, changed three things and nothing afterwards. The \(\varepsilon\) grid gained 0.05, because with the original three values \(n^\ast\) was 1 or at least 8 in eight of nine cells and the test had almost no interior; at \(\varepsilon = 0.05\), \(n^\ast \in \{1, 2, 8\}\). The pass thresholds on K1 and K3 moved from 7 of 9 cells to 9 of 12. "Final-correct" became: the greedy policy is correct in at least 80% of the evaluations over the last 20% of training. With \(\alpha = 0.1\) and sampled falls, \(Q(s_0, A)\) fluctuates by about \(\pm 0.07\), so a single end-of-training read is a coin flip when the bias sits within 0.07 of the gap. The update budget moved from 100,000 to 60,000.

Let \(\hat n\) be the largest grid \(n\) that is final-correct in at least 6 of 8 seeds, or 0 if none is.

- **K1.** At \(K = 2{,}000\), \(\hat n\) equals \(n^\ast\) or the grid neighbour of \(n^\ast\), in at least 9 of 12 \((\varepsilon, \mathrm{gap})\) cells.
- **K2.** Over all cells with final-correct in at least 6 of 8 seeds, the Spearman correlation of median \(T\) with \((\lceil 21/n \rceil - 1)K + 1\) is at least 0.85.
- **K3.** \(\hat n\) at \(K = 200\) equals \(\hat n\) at \(K = 2{,}000\) in at least 9 of 12 cells.

Supported if K1 and K2. The recorded verdict is **supported**: K1 is 9/12, K2 is \(+1.00\) over 66 cells, K3 is 12/12. The run took 75 minutes on CPU.

### 4.1 Below the threshold, the wait is the hop count

On the 66 cells that end correct, median \(T\) divided by \((\lceil 21/n \rceil - 1)K + 1\) has median 1.050, minimum 1.005, maximum 1.990. There is no fitted constant in the predictor. The upper end of the range is the 200-update evaluation grain at \(K = 200\) and \(n \ge 12\). The Spearman correlation is \(+1.00\).

At \(K = 2{,}000\), \(\varepsilon = 0.01\), and gap at least 0.2, every seed is correct, and the median times are 40,200, 20,200, 12,200, 8,200, 4,200, 2,200, and 2,200 at \(n = 1, 2, 3, 5, 8, 12, 20\). From \(n = 1\) to \(n = 12\) that is an 18× reduction, at 8 of 8 seeds correct on both sides. The recorded times at \(n = 12\) and \(n = 20\) are both 2,200.

### 4.2 The learned value is the closed form, and the policy flips at the gap

Over all 168 cells, median \(Q(s_0, A)\) against Proposition 1 has Pearson \(r = 0.985\), RMSE 0.029, and maximum absolute error 0.086. The within-cell standard deviation of the learned value, across the eight seeds, has median 0.064 and 90th percentile 0.097.

The fraction of seeds that end correct, pooled over \(K\), as a function of the predicted margin (fixed point minus \(g\)):

| Margin | fraction correct | cells |
|---|---:|---:|
| \((-1.00,\,-0.05]\) | 0.00 | 64 |
| \((-0.05,\,-0.02]\) | 0.00 | 4 |
| \((-0.02,\,0.00]\) | 0.04 | 10 |
| \((0.00,\,+0.02]\) | 0.06 | 6 |
| \((+0.02,\,+0.05]\) | 0.29 | 16 |
| \((+0.05,\,+1.00]\) | 0.96 | 68 |

On the two bins at or below \(-0.02\), the fraction correct is 0.00 over 64 cells and 0.00 over 4 cells. Above \(+0.05\), the fraction is 0.96. The change across the bins around zero is on the scale of the seed noise at \(\alpha = 0.1\).

K1's three misses, and the margin of Proposition 1 over \(g\) at the formula's \(n^\ast\), are \(\varepsilon = 0.01\), gap 0.1, \(n^\ast = 20\), margin \(+0.026\); \(\varepsilon = 0.03\), gap 0.2, \(n^\ast = 8\), margin \(+0.005\); \(\varepsilon = 0.05\), gap 0.3, \(n^\ast = 8\), margin \(+0.001\). Each is below the median seed standard deviation 0.064. The formula called \(n^\ast\) "still correct" by less than the noise, and the grid's \(\hat n\) stepped down. The nine cells that hit have margins \(+0.126\), \(+0.226\), \(+0.100\), \(+0.096\), \(+0.100\), \(+0.017\), \(+0.100\), \(+0.200\), and \(+0.300\): the notes record these as 0.10–0.30 except one at 0.017 that happened to land right. K1 passes at exactly its bar of 9 of 12. A margin rule that requires the bias to clear \(g\) by more than a few seed standard deviations is the refinement the misses ask for. The draft of this study records that with a margin \(m = 0.05\) all twelve cells agree. The \(+0.017\) hit shows the boundary is soft at the scale of the noise, which is what the six-bin table already says: fraction correct 0.06 just above zero, 0.29 between 0.02 and 0.05, 0.96 above 0.05.

The full grid of \(n^\ast\) (margin 0) and \(\hat n\):

| \(\varepsilon\) | gap | \(n^\ast\) | \(\hat n\) at \(K=200\) | \(\hat n\) at \(K=2000\) | \(T\) at \(n=1\) vs at \(\hat n\), \(K=2000\) |
|---:|---:|---:|---:|---:|---|
| 0.01 | 0.1 | 20 | 2 | 2 | 40,200 vs 20,200 |
| 0.01 | 0.2 | 20 | 20 | 20 | 40,200 vs 2,200 |
| 0.01 | 0.3 | 20 | 20 | 20 | 40,200 vs 2,200 |
| 0.03 | 0.1 | 1 | 1 | 1 | — |
| 0.03 | 0.2 | 8 | 2 | 2 | 40,200 vs 20,200 |
| 0.03 | 0.3 | 20 | 20 | 20 | 40,200 vs 2,200 |
| 0.05 | 0.1 | 1 | 1 | 1 | — |
| 0.05 | 0.2 | 2 | 1 | 1 | — |
| 0.05 | 0.3 | 8 | 2 | 2 | 40,200 vs 20,200 |
| 0.1 | 0.1–0.3 | 1 | 1 | 1 | — |

### 4.3 The threshold does not depend on \(K\)

\(\hat n\) is identical at \(K = 200\) and at \(K = 2{,}000\) in all 12 cells, against a bar of 9. A tenfold change in the target period changes when a correct policy appears and does not change the largest \(n\) at which it appears. The threshold is a property of the dataset. The latency is a property of the target period.

## 5 What the formula says about the published choices

On this 20-step path, the closed form with margin 0 gives safe \(n \le 1\) at \(\varepsilon = 0.1\) for gaps 0.1, 0.2, and 0.3, and safe \(n \le 40\) at \(\varepsilon = 0.01\) for the same three gaps. The second bound is also read as \(n^\ast \ge 40\): every \(n\) up to 40 stays above the gap, so \(n\) of 10 to 20 lies inside the safe set. Nature DQN annealed \(\varepsilon\) to 0.1 and used \(n = 1\). On this path \(n = 3\) would have put the fixed point on the wrong side of gaps up to 0.3, which is the regime in which uncorrected \(n\)-step DQN at \(\varepsilon = 0.1\) needs careful tuning. Rainbow anneals \(\varepsilon\) from 1 to 0.01 over 250,000 frames and uses \(n = 3\): safe on this path at the final \(\varepsilon\), and, at a target period of 32,000 gradient steps (8,000 updates times \(\lceil L/3 \rceil\)), leaving most of the latency reduction unused. DER and BBF use \(\varepsilon = 0.01\), or noisy nets, with \(n = 20\) and with \(n\) annealed from 10 to 3. Both sit inside the safe set the formula gives at that \(\varepsilon\), and both spend the latency. The anneal is the corollary in Section 2: early, the gap is large relative to the bias and the latency dominates; late, the remaining gaps shrink. R2D2's per-actor \(\varepsilon\) up to 0.4 at \(n = 5\) falls outside the rule for its high-\(\varepsilon\) actors. The recurrent state and the 80-step stored sequences change the path the formula assumes, and the rule is not claimed for that agent unmodified.

The behaviour policy in the pilot is frozen. In a live agent it drifts toward greedy, \(f\) falls, and the bias shrinks during training. The rule then says large \(n\) early and smaller \(n\) late, provided the gaps shrink as well. That is a prediction about a schedule. This grid does not run one.

## 6 Limitations

Proposition 1 is a single path with a one-shot failure action. A branching task has a per-step break probability equal to the fraction of actions that are catastrophic, which this corridor sets to \(1/2\) of \(\varepsilon\) by construction and which a general action space does not. Proposition 2 is the synchronous hop count. Sampled replay adds a further wait proportional to \(L^2/(nB)\) per hop; the fixed dataset and the exact greedy rollout keep that term from being the quantity K2 measures. The dataset's behaviour policy does not change with the learner, so the bias is stationary. Online, \(\varepsilon\) and the gaps both move, and \(n^\ast\) becomes a schedule whose inputs have to be measured. K1 passes at its threshold and not with room: three cells miss, all three on the noise boundary, and one hit sits at margin \(+0.017\). The margin \(m\) is the honest form of the rule, and \(m = 0.05\) is a post-hoc choice on this noise level, not a constant the pre-registration named. MinAtar DQN at \(n \in \{1, 3, 5, 10, 20\}\), \(\varepsilon \in \{0.01, 0.1\}\), and \(K \in \{500, 8000\}\) has not been run.

## 7 Conclusion

On a 20-step path, uncorrected \(n\)-step returns under a hard target have a largest safe backup length set by \(\varepsilon\) and the gap, and a latency \((\lceil 21/n \rceil - 1)K\) below it. The learned values match the bias formula at \(r = 0.985\) over 168 cells. The time to a correct policy matches the latency at Spearman \(+1.00\), median ratio 1.050, with no fitted constant. The largest safe grid point agrees with \(n^\ast\) within one step in 9 of 12 cells, the misses being margins of \(+0.026\), \(+0.005\), and \(+0.001\) against a seed standard deviation of 0.064, and it does not change when \(K\) changes from 200 to 2,000. At \(\varepsilon = 0.01\) and a gap of at least 0.2, \(n = 12\) delivers the correct policy in 2,200 updates and \(n = 1\) in 40,200, an 18× difference at the same correctness. At \(\varepsilon = 0.1\) the same path admits \(n = 1\) and not the neighbours above it.

## References

Daley, White, Amato, White (2023). Trajectory-aware eligibility traces for off-policy reinforcement learning. · Fedus et al. (2020). Revisiting fundamentals of experience replay. ICML. · Hernandez-Garcia and Sutton (2019). Understanding multi-step deep reinforcement learning: a systematic study of the DQN target. arXiv:1901.07510. · Hessel et al. (2018). Rainbow: combining improvements in deep reinforcement learning. AAAI. · Kapturowski et al. (2019). Recurrent experience replay in distributed reinforcement learning. ICLR. · Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature. · Rowland et al. (2020). Adaptive trade-offs in off-policy learning. AISTATS. · Schwarzer et al. (2023). Bigger, better, faster: human-level Atari with human-level efficiency. ICML. · Sutton and Barto (2018). Reinforcement Learning: An Introduction, 2nd ed. · van Hasselt, Hessel, Aslanides (2019). When to use parametric models in reinforcement learning? NeurIPS.
