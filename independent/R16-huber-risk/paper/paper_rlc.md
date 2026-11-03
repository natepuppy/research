# The Huber Loss Is a Risk Preference

*Conference-length version (RLC format). Every number here comes from the results in this folder.*

## Abstract

DQN minimises a Huber loss on the TD error, equivalently clipping the error to \([-\kappa, \kappa]\) before the update, with the usual choice \(\kappa = 1\). The fixed point of that update is the Huber location of the target distribution, not its mean. Wherever a rare outcome carries a large value difference, the rare side is under-weighted: a lottery is under-valued and a hazard is under-weighted, with a bias whose sign is minus the skew and whose size is set by \(\kappa\) relative to the jump. On fork MDPs the closed-form Huber Bellman fixed point was compared with tabular Q-learning. A two-action pilot is KILLED by rule. A pre-registered single-action follow-up predicts the learned bias with identity-fit \(R^2 = 1.000\) and places the greedy error in exactly the cells where the absolute predicted bias exceeds the gap (60/60 cells, and in no squared-loss cell). The sign is \(-\)skew in all 40 active cells and fails in four cells where the predicted bias is at most 0.013, so the follow-up rule returns INCONCLUSIVE. At \(\kappa = 1\) the bias is 2.5× to 9.0× the action gap and the greedy action is wrong in 100% of seeds in every listed cell. A secondary CartPole check finds the learned value of terminating actions higher under Huber than under the squared loss in 13/16 paired seeds.

## 1 Introduction

Huber (1964) introduced his loss as a robust location estimator. For a distribution with outliers it estimates a point between the mean and the median. In deep Q-learning the same loss is applied to TD errors (Mnih et al., 2015). The target at a state-action pair is the distribution of \(r + \gamma \max_{a'} \bar Q(s', a')\). That distribution is skewed when a rare transition carries a large value: a heavy-tailed reward, a rare jump to a high-value state, or a rare termination from a high-value state. The Huber fixed point then differs from the mean, and the rare side is the side that is under-weighted.

The shift is the population fixed point of the update. It remains after the sample average has converged. It is also distinct from reward clipping (van Hasselt et al., 2016), which changes the reward function. The Huber loss changes the estimator of a fixed reward function's return, and it acts on bootstrapped value jumps even when rewards already lie in a bounded range.

The experiments below ask whether that fixed point predicts what tabular Q-learning learns, whether the resulting bias is large enough to flip the greedy action, and whether a deep network with \(\kappa = 1\) shows the same signature on terminal transitions, the one place CartPole produces a large TD error. The answers are tied to pre-registered rules. The two-action pilot is KILLED. The single-action follow-up is INCONCLUSIVE, with the quantitative and policy clauses confirmed and the sign clause failed on four inactive cells.

**Contributions.** (1) The fixed point of clipped TD, \(\mathbb{E}[\mathrm{clip}(Y - q, -\kappa, \kappa)] = 0\), and the two-point solution \(q_\kappa = p\kappa/(1-p)\) when the rare outcome still saturates the clip. (2) A 60-cell tabular test: the pilot is KILLED because a second bias, Q-learning's max over duplicate noisy actions, enters the target; the pre-registered single-action follow-up reaches \(R^2 = 1.000\) and 60/60 policy agreement, and is INCONCLUSIVE because the sign fails where the clip is nearly off. (3) The interaction of the two biases: under the squared loss the max-bias on the noisy arm is \(+0.09\) to \(+0.31\), risk-seeking, and the Huber clip suppresses it. (4) A secondary CartPole comparison, not part of the kill test: failure-Q ends at 26.84 under Huber against 22.15 under the squared loss (13/16 paired seeds), while the fraction of TD errors beyond \(\kappa = 1\) is 0.02.

## 2 Related work

Mnih et al. (2015) introduce the clip. Obando-Ceron and Castro (2021) and Patterson et al. (2023) report that the loss matters empirically, without a fixed-point analysis of the clip. Distributional RL uses the quantile Huber loss on purpose, to estimate quantiles rather than means (Dabney et al., 2018). Robust bandits and robust RL with heavy tails (Bubeck et al., 2013; Zhuang and Sui, 2021) seek a robust estimate of the mean, which is the estimand the clipped TD update does not have. The max-bias of Q-learning is Thrun and Schwartz (1993) and van Hasselt (2010); the two-action pilot below is that bias sitting on one arm. Risk-sensitive RL (Mihatsch and Neuneier, 2002; Tamar et al., 2015) puts a risk preference into the objective by design. The claim here is that a default loss already has one, with the sign of minus the skew and the scale of \(\kappa\).

## 3 The Huber fixed point

Fix a state-action pair whose TD target is a random variable \(Y\). The expected clipped update is zero at the \(q\) solving \(\mathbb{E}[\mathrm{clip}(Y - q, -\kappa, \kappa)] = 0\). The left-hand side is monotone decreasing in \(q\), so the root \(q_\kappa\) is unique. It equals \(\mathbb{E}[Y]\) when \(\kappa\) is at least the largest deviation of \(Y\) from that root, and it tends to the median of \(Y\) as \(\kappa \to 0\). For a two-point lottery \(Y \in \{M, 0\}\) with \(\mathbb{P}(Y = M) = p\) and \(M - q_\kappa > \kappa\), the estimating equation is \(p \cdot \kappa = (1-p) \cdot q_\kappa\), hence \(q_\kappa = p\kappa/(1-p)\). At \(p = 0.05\) and \(\kappa = 1\) this root is 0.053, against a mean \(pM = 1.1\). The rare outcome contributes \(\kappa\) to the estimating equation in place of \(M - q\). The bias \(q_\kappa - \mathbb{E}[Y]\) is then approximately \(-pM\) for a lottery and approximately \(+pM\) for a hazard: the rare side's expected contribution is almost entirely lost. Iterating the root as a Bellman operator produces a Huber-TD fixed point for the whole MDP. In the experiments that fixed point is computed by bisection and compared with the value tabular Q-learning learns.

The bias is zero once \(\kappa\) exceeds the largest deviation of the target from its Huber root, roughly once \(\kappa > M(1-p)\). With rewards clipped to \([-1, 1]\) and \(\kappa = 1\), ordinary reward noise does not activate the clip. Value jumps do: a termination from a high-value state, or a rare large bootstrap.

## 4 Experiments

### 4.1 Design, smoke, and the pre-registered rules

The MDP is a fork. From \(s_0\) the actions are *safe* and *risky*. Each action leads down a deterministic chain of \(L = 3\) states to a terminal reward. The discount is \(\gamma = 0.95\). Safe pays \(m = 1\). Risky pays a skewed lottery whose mean is \(m \pm g\).

- Lottery (positive skew): terminal reward \(M\) with probability \(p\) and 0 otherwise, with \(pM = m + g\). Risky is optimal by the gap \(g\).
- Hazard (negative skew): a sure base \(b = m - g + pM\), minus \(M\) with probability \(p\). The mean is \(m - g\). Safe is optimal by \(g\).

The grid is \(g \in \{0.1, 0.3\}\), \(p \in \{0.05, 0.1, 0.3\}\), \(\kappa \in \{0.1, 0.3, 1, 3, \infty\}\) with \(\infty\) the squared loss, and both skews: 60 cells, 8 seeds. Behaviour is a uniform random action at \(s_0\) together with exploring starts, so both arms are sampled equally. The update is \(Q \leftarrow Q + \alpha \,\mathrm{clip}(\delta, -\kappa, \kappa)\) with \(\alpha = 0.02\) for 300k steps. The learned bias is \(\hat Q(s_0, \mathrm{risky}) - Q^\ast(s_0, \mathrm{risky})\), averaged over the last quarter of training. The prediction is the Huber Bellman fixed point.

A smoke run of 20k steps and 2 seeds at \(p = 0.02\) (payoff \(M \approx 52\)) showed the squared-loss estimate's stationary standard deviation alone, about 0.7 at \(\alpha = 0.02\), flipping a gap of 0.05. The grid was narrowed to \(p \ge 0.05\) and \(g \ge 0.1\), and the horizon set to 300k steps, before the tests were fixed. At the retained lottery \(p = 0.05\), \(M = 22\), the same stationary standard deviation is about 0.5. The clip trades that variance for bias.

The kill test, written before the pilot, has three clauses. K1: in every cell with finite \(\kappa\), the bias is negative under a lottery and positive under a hazard, and within each (skew, \(g\), \(p\)) the absolute bias is non-increasing in \(\kappa\), with Spearman at most \(-0.8\) over the five values of \(\kappa\). K2: identity-fit \(R^2\) of learned bias on predicted bias at least 0.9 over all 60 cells. K3: the cells whose greedy policy is wrong in a majority of seeds are those with absolute predicted bias above \(g\), at cell-wise agreement at least 90%, and no \(\kappa = \infty\) cell is wrong. SUPPORTED requires K1, K2, and K3. INCONCLUSIVE is exactly two. Anything else is KILLED.

### 4.2 Two-action pilot: KILLED

Chain states in the pilot had two identical actions, with Q-values learned independently from noisy targets. The target's \(\max\) therefore carries Q-learning's max-bias on the noisy arm only. Under the squared loss that bias is \(+0.09\) to \(+0.31\).

K2 passes: identity-fit \(R^2 = 0.986\). K1 fails: monotonicity holds, with Spearman in \([-1.00, -0.90]\), and the sign clause does not, including a positive bias in the \(\kappa = 3\) lottery cells. K3 fails: cell-wise agreement with \(|\mathrm{predicted\ bias}| > g\) is 95%, and at least one squared-loss cell is wrong. The squared-loss agent prefers the hazard arm on the strength of the max-bias alone. The rule verdict on this grid is KILLED.

The residual, learned bias minus the Huber fixed-point bias, is positive in every pilot cell and grows with \(\kappa\). Clipped updates have less variance, so the clip suppresses the max-bias that the squared loss lets through.

| \(\kappa\) | Pilot residual, mean (range) | Single-action residual, mean (range) |
|---:|---|---|
| 0.1 | \(+0.002\) (\(+0.001\) to \(+0.005\)) | \(-0.000\) (\(-0.000\) to \(+0.000\)) |
| 0.3 | \(+0.007\) (\(+0.004\) to \(+0.013\)) | \(+0.000\) (\(-0.001\) to \(+0.001\)) |
| 1 | \(+0.024\) (\(+0.012\) to \(+0.042\)) | \(+0.000\) (\(-0.001\) to \(+0.002\)) |
| 3 | \(+0.064\) (\(+0.037\) to \(+0.129\)) | \(-0.001\) (\(-0.022\) to \(+0.018\)) |
| \(\infty\) | \(+0.188\) (\(+0.085\) to \(+0.313\)) | \(+0.000\) (\(-0.006\) to \(+0.021\)) |

### 4.3 Single-action follow-up: INCONCLUSIVE

The follow-up was written down after the pilot and before it was run. The only change is one action per chain state, so the duplicate-action max is gone. The same three clauses apply. The recorded outcome is K2 true at \(R^2 = 1.000\), K3 true at 100% agreement with no squared-loss cell wrong, and K1 false. Monotonicity still holds, Spearman in \([-1.00, -0.90]\). The sign clause fails in four cells where the predicted bias is at most 0.013 in magnitude (\(\kappa = 3\), \(p = 0.3\)), the clip is nearly inactive, and the learned bias has the sign of noise. The sign is \(-\)skew in all 40 active cells. Because K1 fails, the pre-registered follow-up verdict is INCONCLUSIVE. With one action per chain state the residual is within \(\pm 0.03\) everywhere: the Huber fixed point accounts for the learned bias.

At the DQN default \(\kappa = 1\) the bias, relative to the gap, is large, and the greedy action is wrong in every seed. The jump \(M\) is 22 at \(p = 0.05\), 3.7 at lottery \(g = 0.1\), \(p = 0.3\), and 4.3 at \(g = 0.3\), \(p = 0.3\).

| Skew | \(g\) | \(p\) | Bias | Bias / gap | Greedy wrong |
|---|---:|---:|---:|---:|---|
| Hazard | 0.1 | 0.05 | \(+0.899\) | 9.0× | 100% of seeds |
| Hazard | 0.1 | 0.10 | \(+0.847\) | 8.5× | 100% |
| Hazard | 0.1 | 0.30 | \(+0.574\) | 5.7× | 100% |
| Hazard | 0.3 | 0.05 | \(+1.070\) | 3.6× | 100% |
| Hazard | 0.3 | 0.10 | \(+1.019\) | 3.4× | 100% |
| Hazard | 0.3 | 0.30 | \(+0.747\) | 2.5× | 100% |
| Lottery | 0.1 | 0.05 | \(-0.897\) | 9.0× | 100% |
| Lottery | 0.1 | 0.10 | \(-0.846\) | 8.5× | 100% |
| Lottery | 0.1 | 0.30 | \(-0.576\) | 5.8× | 100% |
| Lottery | 0.3 | 0.05 | \(-1.068\) | 3.6× | 100% |
| Lottery | 0.3 | 0.10 | \(-1.018\) | 3.4× | 100% |
| Lottery | 0.3 | 0.30 | \(-0.747\) | 2.5× | 100% |

The \(\kappa = 1\) agent takes the wrong fork in every one of these cells. At \(\kappa = 3\) it is wrong wherever the jump exceeds about 3. At \(p = 0.3\) the jump is 3.7 when \(g = 0.1\), and \(\kappa = 3\) is nearly inactive. At \(p = 0.05\) the jump is 22, and \(\kappa = 3\) still removes 80–95% of the risky arm's expected premium. Under the squared loss, with one action per chain state, the greedy action is right everywhere.

### 4.4 Two biases

The squared loss, in the two-action design, is risk-seeking toward the noisy arm through the max (\(+0.09\) to \(+0.31\), enough to flip a gap of 0.1). Huber is biased against the rare side. Under a hazard the two biases favour the risky arm together. Under a lottery they oppose, and at \(\kappa = 3\) they nearly cancel in the pilot cells. The max-bias is a sampling effect. The Huber bias is a population effect: it is the fixed point, and in the single-action grid it remains after 300k steps, with a residual at \(\kappa = 1\) of mean \(+0.000\).

### 4.5 CartPole, secondary

The deep check was pre-registered as a secondary prediction and is not part of the kill test. Batched DQN on CartPole, episode cap 200, reward 1 per step, \(\gamma = 0.99\), Huber at \(\kappa = 1\) against the squared loss, 16 paired seeds, 30k steps. The probe is 256 state-action pairs whose action terminates, so the true value is 1. Values reach about 100, and the terminal TD error from a high-value state is a large negative number.

K4 asked that the final failure-Q be higher under Huber in at least 12 of 16 paired seeds. It is higher in 13/16. Failure-Q ends at 26.84 (maximum 28.07) under Huber and at 22.15 (maximum 24.26) under the squared loss, against a true value of 1. The fraction of TD errors beyond \(\kappa = 1\) is 0.02; the notes record that 98% of TD errors lie under 1, and that the active errors are the terminal ones. The pilot notes summarise the failure-Q gap as 21% on the rounding 26.8 against 22.2. Final return is 129.1 against 131.2. Area under the learning curve is 125.1 under Huber against 146.4 under the squared loss. On this run the clip's cost is visible in the terminal values, and a higher return is not.

## 5 Discussion and limitations

The follow-up verdict is INCONCLUSIVE by the rule that was written before that run. What the rule's passing clauses say is quantitative. Once duplicate actions are removed, tabular Q-learning learns the Huber Bellman fixed point: \(R^2 = 1.000\), residual within \(\pm 0.03\), and the greedy policy wrong exactly where the absolute predicted bias exceeds the gap, in 60/60 cells, including every \(\kappa = 1\) cell above and no squared-loss cell. The sign is \(-\)skew in all 40 active cells. The clause as written demanded the sign in every finite-\(\kappa\) cell, and four cells with predicted bias at most 0.013 fail it because the learned bias there is noise. That is a real failure of the pre-registered statement. It is also the regime in which the clip is nearly off, which is why the monotonicity clause still holds and why \(\kappa = 3\) at \(p = 0.3\) is the boundary where the bias disappears.

The pilot is KILLED, and the reason is informative rather than a contradiction of the fixed point. Two identical actions on the noisy arm add a max-bias of \(+0.085\) to \(+0.313\) under the squared loss (mean \(+0.188\)). The Huber residual in that design grows monotonically with \(\kappa\), from mean \(+0.002\) at \(\kappa = 0.1\) to \(+0.188\) at the squared loss, because clipping reduces the update variance that feeds the max. A loss can be robust to outliers and still change the estimand, and the same clip can hide a second bias while introducing its own.

The CartPole check is one task, 16 seeds, and one \(\kappa\). It matches the direction predicted for terminal transitions, where \(\delta = 1 - Q(s, a)\) is about \(-(V - 1)\) and \(V\) reaches about 100. It does not show a return gap (129.1 against 131.2), and the squared loss has the higher area under the curve (146.4 against 125.1). Both failure-Q numbers, 26.84 and 22.15, sit far above the true value 1. The squared loss already misses the terminal value on this task, and Huber ends higher still. The tabular MDPs are two-outcome lotteries on a chain of length 3, at one step size. Function approximation shares parameters across states, so a clipped update at one state moves others; the fixed-point argument is per state-action pair. MinAtar runs with injected lotteries and hazards were not run.

## 6 Conclusion

Clipping the TD error replaces the mean of the target with its Huber location. On the single-action forks that location predicts the learned bias at \(R^2 = 1.000\) and predicts every greedy error in a 60-cell grid, and the pre-registered verdict remains INCONCLUSIVE because four nearly inactive cells break a sign clause written for every finite \(\kappa\). For a two-point lottery with \(p = 0.05\) and mean \(pM = 1.1\), the Huber root at \(\kappa = 1\) is 0.053. At the fork, the learned bias in the cell \(g = 0.1\), \(p = 0.05\) is \(-0.897\), 9.0× the gap, and the greedy action is wrong in every seed; the matching hazard cell is \(+0.899\), also 9.0× the gap. The two-action pilot, KILLED by rule, shows the squared loss carrying a risk-seeking max-bias of \(+0.09\) to \(+0.31\) on the noisy arm, which the clip suppresses as it imposes a bias of its own. In the CartPole check the clip is active on 2% of TD errors, the terminal ones, and the value of actions whose true value is 1 ends higher under Huber in 13 of 16 seeds.

## References

Bubeck, Cesa-Bianchi, Lugosi (2013). Bandits with heavy tails. IEEE Trans. Inf. Theory. · Dabney, Rowland, Bellemare, Munos (2018). Distributional reinforcement learning with quantile regression. AAAI. · Huber (1964). Robust estimation of a location parameter. Ann. Math. Stat. · Mihatsch, Neuneier (2002). Risk-sensitive reinforcement learning. Machine Learning. · Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature. · Obando-Ceron, Castro (2021). Revisiting Rainbow: promoting more insightful and inclusive deep reinforcement learning research. ICML. · Patterson, Neumann, White, White (2023). Empirical design in reinforcement learning. · Tamar, Chow, Ghavamzadeh, Mannor (2015). Policy gradients for coherent risk measures. NeurIPS. · Thrun, Schwartz (1993). Issues in using function approximation for reinforcement learning. · van Hasselt (2010). Double Q-learning. NeurIPS. · van Hasselt, Guez, Hessel, Mnih, Silver (2016). Learning values across many orders of magnitude. NeurIPS. · Zhuang, Sui (2021). No-regret reinforcement learning with heavy-tailed rewards. AISTATS.
