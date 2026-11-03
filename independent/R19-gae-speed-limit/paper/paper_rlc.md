# The Credit-Assignment Speed Limit of PPO

*RLC-length version; every number is from this folder's results.*

## Abstract

With a critic refit each iteration to \(\lambda\)-returns built from itself, a reward \(D\) steps away reaches the start-state value after a number of iterations proportional to \(D(1-\lambda)\), and the start-state policy moves in that same iteration. On a controlled corridor, over the 16 cells with \(D(1-\lambda) \ge 4\), \(\log t_V\) has slope 0.95 on \(\log D\) and 0.88 on \(\log(1-\lambda)\), with \(R^2 = 0.999\). At \(\lambda = 1\) the value arrives in one iteration at every depth. All 70 cells reach success probability at least 0.98 by iteration 600. The same \(\lambda = 0.95\) run with the critic frozen never solves \(D = 128\). A pre-registered claim that a rollout of \(T = 16\) matches \(\min(T, 1/(1-\lambda))\) failed the constant: at \(\lambda = 1\) the \(T = 16\) latency scales as \(D^{1.03}\) and is 1.5 to 2.27 times the \(\lambda = 0.95\), \(T = \infty\) latency. A post-hoc model in which the two slowdowns add as rates, \(1/h_{\mathrm{eff}} = (1-\lambda) + 1/T_{\mathrm{eff}}\) with \(T_{\mathrm{eff}} = 6.1\), fits the 35 finite-horizon cells at \(R^2 = 0.998\).

## 1 Introduction

PPO with generalised advantage estimation is the default policy-gradient method from continuous control to long-horizon preference optimisation, and \(\lambda \in [0.95, 0.97]\) is a near-universal setting. Read one update at a time, \(\lambda < 1\) is a bias–variance knob: the advantage at a state is a geometric mixture of \(n\)-step returns and the critic, so a reward \(D\) steps away contributes with weight about \((\gamma\lambda)^D\) unless the critic has already absorbed it. VC-PPO (Yuan et al., 2025) takes that weight as a lost signal in long-horizon RLHF and decouples a critic at \(\lambda = 1\) from an actor at \(\lambda < 1\).

The loop PPO actually runs refits the critic every iteration to \(\lambda\)-return targets built from the previous critic. If that critic already carries the reward within about \(1/(1-\gamma\lambda)\) states, the new target does too. Each fit therefore moves a front, and the front reaches a decision \(D\) steps from the reward after a number of iterations proportional to \(D(1-\gamma\lambda)\). The advantage that trains the policy is the same mixture minus \(V(s)\), so the policy at the start moves when the value does. A rollout segment of \(T\) steps adds a bootstrap at the boundary and a second limit on how far one fit can see.

This paper tests that account on a tabular actor–critic loop with an exact critic. The latency law in \(D\) and \(\lambda\), the disappearance of the latency at \(\lambda = 1\), and the lockstep of policy and value were pre-registered and are supported. The pre-registered identification of the segment cap with \(\min(T, 1/(1-\lambda))\) is not: the scaling in \(D\) holds, and the constant does not. The replacement, addition of the two rates, is post hoc on the same cells.

**Contributions.** (1) A latency law \(t_V \propto D^{0.95}(1-\lambda)^{0.88}\) at \(R^2 = 0.999\), and \(t_V = 1\) at every depth when \(\lambda = 1\). (2) The policy inherits the latency: Spearman correlation 0.99 between the two arrival times, with a further lag of \(-2\) to \(+1\) iterations. (3) The frozen-critic contrast: every \((\lambda, D)\) cell solves when the critic is refit, and \(\lambda = 0.95\) fails at \(D = 128\) when it is held at zero. (4) The segment-length extension, whose pre-registered min form fails and whose post-hoc rate sum fits at \(R^2 = 0.998\).

## 2 Related work

Schulman et al. (2016) introduce \(\lambda\) as a per-update bias–variance trade-off. Schulman et al. (2017) is the PPO update used here in a one-step, unclipped form. Yuan et al. (2025) identify that at \(\lambda < 1\) a critic in long-sequence RLHF underestimates early-token values, frame the gap as decay, and set \(\lambda_{\mathrm{critic}} = 1\) while leaving \(\lambda_{\mathrm{actor}} < 1\). The iteration count below is the quantitative version of that observation, together with the segment length, which puts the delay back at any \(\lambda\). Tsitsiklis and Van Roy (1997) give asymptotic convergence for TD(\(\lambda\)), not the transient arrival time. Huang et al. (2022) record, among PPO implementation details, that GAE stops at the rollout boundary. Critic-free group baselines (Shao et al., 2024) are the \(\lambda = 1\) Monte Carlo end of the same axis. A hard target refreshed on a fixed period imposes a different latency, by a different mechanism; nothing in this experiment shares that setup.

## 3 Setting

The MDP is deterministic. A choice state \(s_0\) has two actions, leading to corridors A and B of depth \(D\). Corridor states have `forward` and `stall`. The end of A pays 1 and terminates; the end of B pays 0 and terminates. The episode cap is \(3D\). The discount is \(\gamma = 1\), so the mixture horizon is \(1/(1-\lambda)\).

Each iteration collects \(N = 32\) episodes from a tabular softmax policy, forms \(\lambda\)-returns with the current critic as the bootstrap at segment boundaries, sets the tabular critic to the per-state mean of those targets (unvisited states keep their value), normalises the GAE advantages over the batch, and takes one SGD step of size 0.5 on the logits. There is no clipping: one step per batch does not engage it. The policy starts uniform and the critic starts at 0. \(T = \infty\) means the only bootstrap inside an episode is at the cap.

Two clocks, chosen before the pilot. \(t_V\) is the first iteration with \(V(s_0) \ge 0.25\). The value of the policy that always enters A and always moves forward has half equal to 0.5; 0.25 is used so the threshold is crossed before the policy is near-deterministic. \(t_\pi\) is the first iteration at which the exact probability that the stochastic policy reaches A's end within the cap exceeds 0.75. A threshold of 0.5 was unusable: the untrained uniform policy already succeeds with probability about 0.5, because it picks corridor A half the time and almost always finishes inside the cap. The threshold was raised to 0.75 before any pilot run. Runs that never cross a threshold are recorded at 600.

The grid is \(D \in \{8, 16, 32, 64, 128\}\), \(\lambda \in \{0, 0.5, 0.8, 0.9, 0.95, 0.99, 1\}\), \(T \in \{\infty, 16\}\), eight seeds, 600 iterations: 70 cells. Because \(t_V\) is an integer with a floor of 1, the regression in K1 uses only cells whose predicted latency \(D(1-\lambda)\) is at least 4. That restriction was fixed with the grid.

- **K1.** \(T = \infty\), \(\lambda < 1\), \(D(1-\lambda) \ge 4\): OLS of \(\log\) median \(t_V\) on \(\log D\) and \(\log(1-\lambda)\) has both slopes in \([0.7, 1.3]\) and \(R^2 \ge 0.9\).
- **K2.** \(T = \infty\), \(\lambda = 1\): the slope of \(\log\) median \(t_V\) on \(\log D\) is below 0.3.
- **K3.** \(T = 16\), \(\lambda = 1\): the slope on \(\log D\) over \(D \ge 32\) is in \([0.7, 1.3]\), and for \(D \ge 32\) the median \(t_V\) is within \(1.5\times\) of the median \(t_V\) at \(T = \infty\), \(\lambda = 0.95\), whose horizon \(1/(1-\lambda) = 20\) is close to 16.
- **K4.** Spearman correlation of median \(t_\pi\) with median \(t_V\), over cells with both medians finite, at least 0.8.

Supported if K1, K2, and K4. K3 is the segment-length extension and does not enter the verdict.

## 4 Results

The recorded verdict is **supported**. K1, K2, and K4 pass. K3 fails. The pilot took about 4 CPU minutes.

### 4.1 Depth and \(\lambda\)

On the 16 cells admitted to K1, the slope on \(\log D\) is 0.95, the slope on \(\log(1-\lambda)\) is 0.88, and \(R^2 = 0.999\). On all 30 cells with \(\lambda < 1\) and \(T = \infty\), the post-hoc line \(t_V = 1.25\, D(1-\lambda) + 1.91\) has \(R^2 = 0.999\). The slope above 1 matches the stall: early in training about half the actions stall, and \(V(s)\) rises geometrically toward \(V(s+1)\) on each fit, which adds a constant fraction of an iteration per state. At \(\lambda = 1\), \(t_V = 1\) at every depth and the slope on \(\log D\) is 0.00.

| \(\lambda\) | \(D=8\) | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|
| 0 | 11 | 22 | 42 | 82 | 160 |
| 0.5 | 6 | 12 | 22 | 42 | 81 |
| 0.8 | 3 | 6 | 10 | 20 | 38 |
| 0.9 | 2 | 3 | 6 | 11 | 21 |
| 0.95 | 1 | 2 | 4 | 6 | 11 |
| 0.99 | 1 | 1 | 1 | 2 | 3 |
| 1 | 1 | 1 | 1 | 1 | 1 |

*Table 1. Median \(t_V\) in iterations, \(T = \infty\).*

At \(\lambda = 0.95\), a reward 128 steps away first moves \(V(s_0)\) at iteration 11. At \(\lambda = 0\), the same reward takes 160 iterations. The practical reading of the fitted coefficient is that a 2,000-step horizon at \(\lambda = 0.95\) is about 100 iterations before the start state sees the reward at all.

### 4.2 The policy moves in the same iteration

Spearman\((t_\pi, t_V) = 0.99\) over the 70 cells. The per-cell median of \(t_\pi - t_V\) has mean \(-0.34\), minimum \(-2\), and maximum \(+1\). There is no second wait between the critic and the action at \(s_0\). One SGD step on the normalised advantage is enough for the start-state policy to move once the advantage carries the reward.

### 4.3 Arrival, with a critic

Every one of the 70 cells reaches a final success probability of at least 0.98 by iteration 600. The minimum across cells is 0.983. No pair \((\lambda, D)\) fails to learn. The delay is a wait, and the task is solved after the wait.

The decay description becomes true when the conversion is removed. With the critic frozen at \(V = 0\) (post hoc, four seeds, 600 iterations, \(T = \infty\)), the policy has only the direct factor \((\gamma\lambda)^D\) on the start-state advantage, after the same batch normalisation:

| \(D\) | \(\lambda\) | median \(t_\pi\) | hit rate | final success |
|---:|---:|---:|---:|---:|
| 32 | 0.80 | 600 | 0.00 | 0.21 |
| 32 | 0.95 | 24 | 1.00 | 1.00 |
| 32 | 1.00 | 3 | 1.00 | 1.00 |
| 64 | 0.80 | 600 | 0.00 | 0.25 |
| 64 | 0.95 | 218 | 1.00 | 0.94 |
| 64 | 1.00 | 3 | 1.00 | 1.00 |
| 128 | 0.80 | 600 | 0.00 | 0.37 |
| 128 | 0.95 | 600 | 0.00 | 0.16 |
| 128 | 1.00 | 3 | 1.00 | 1.00 |

\(\lambda = 1\) still solves every depth in 3 iterations, because the return is Monte Carlo and does not need the critic. \(\lambda = 0.95\) solves \(D = 32\) in 24 iterations and \(D = 64\) in 218, and at \(D = 128\) the hit rate is 0 with final success 0.16. \(\lambda = 0.8\) solves nothing on this grid. The main experiment's \(\lambda = 0.95\), \(D = 128\) cell, with the critic refit, solves in 11 iterations. Refitting the critic is what turns the direct weight into a delay of about \(1.25\, D(1-\lambda)\) iterations.

### 4.4 Segment length: the min form fails, the rates add

K3 asked \(T = 16\) at \(\lambda = 1\) to scale linearly in \(D\) and to land within \(1.5\times\) of \(T = \infty\) at \(\lambda = 0.95\). The slope on \(\log D\) over \(D \ge 32\) is 1.03. The ratios of the two medians are 1.5, 2.17, and 2.27. The scaling passes and the constant fails, so K3 is false as pre-registered. At \(D = 128\) the \(\lambda = 1\), \(T = 16\) run takes 25 iterations.

The pre-registered shape was \(h_{\mathrm{eff}} = \min(T, 1/(1-\lambda))\). On the same 35 cells with \(T = 16\), that min-horizon model fits with \(a = 1.27\), \(T_{\mathrm{eff}} = 6.1\), \(b = 2.08\), and \(R^2 = 0.988\). Adding the rates instead, \(t_V = a\, D\bigl((1-\lambda) + 1/T_{\mathrm{eff}}\bigr) + b\), fits with \(a = 1.06\), \(T_{\mathrm{eff}} = 6.1\), \(b = 0.61\), and \(R^2 = 0.998\). This fit was not pre-registered. It is the summary of the cells K3 already used.

\(T_{\mathrm{eff}} = 6.1\) is below 16 because stalls make 16 time steps cover about 8 states, and because a boundary at a fixed time lands on different states across episodes. Median \(t_V\) at \(T = 16\) for \(\lambda = 0.99\) and \(\lambda = 1\):

| \(D\) | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|
| \(\lambda = 0.99\) | 1 | 3 | 7 | 13 | 26 |
| \(\lambda = 1\) | 1 | 3 | 6 | 13 | 25 |

Once \(\lambda\) is near 1, the segment sets the latency. At \(T = 16\), \(\lambda = 0.95\) is 3 times slower than the same \(\lambda\) with \(T = \infty\). A short `num_steps` puts the delay back at every \(\lambda\), including 1. On a 1,000-step episode with `num_steps` \(= 128\), the same mechanism returns as a latency of order \(D/T_{\mathrm{eff}}\).

## 5 Discussion

The operative count for a reward \(D\) steps from the decision it should change is the number of critic fits, \(D\bigl((1-\gamma\lambda) + 1/T_{\mathrm{eff}}\bigr)\), up to the constants measured here (\(1.25\) and \(T_{\mathrm{eff}} = 6.1\) on this stall structure, \(\gamma = 1\)). Setting \(\lambda = 1\), or a decoupled \(\lambda_{\mathrm{critic}} = 1\), makes arrival one iteration when the segment covers the distance, and leaves a latency proportional to \(D/T_{\mathrm{eff}}\) when it does not. Raising the learning rate or the number of samples per iteration is a different axis: the signal is absent from the advantage until the front arrives, and the policy step in this experiment is already large enough to move \(s_0\) in that iteration.

If a run has not moved the policy at an early decision after \(D(1-\gamma\lambda)\) iterations, that wait is what this law counts. The fix the law points to is a larger \(\lambda\) on the critic, or a longer segment.

## 6 Limitations

The policy is tabular and the critic is an exact fit to the batch targets, so one iteration is one full propagation step of length \(1/(1-\lambda)\) or \(T_{\mathrm{eff}}\). A neural critic that only partly fits its targets would lengthen the constant; Adam rescales small gradients and would change it again. Both leave the \(D(1-\lambda)\) scaling as the claim still to be shown outside this table. The constant 1.25 and \(T_{\mathrm{eff}} = 6.1\) belong to this corridor's stalls. Multiple epochs and PPO's clip change how far the policy moves in the iteration the signal arrives, which is the quantity K4 found to be zero further lag. The discount is \(\gamma = 1\); at \(\gamma < 1\) the horizon in the same derivation is \(1/(1-\gamma\lambda)\). The rate-addition model is post hoc. It beats the pre-registered min on these 35 cells (\(R^2\) 0.998 against 0.988) and has not been confirmed on a fresh grid. Neural PPO on long-corridor MiniGrid, and a frozen-reward proxy for long completions, have not been run.

## 7 Conclusion

In an iterated actor–critic loop, GAE with a refit critic delays a reward \(D\) steps away by a number of iterations proportional to \(D(1-\lambda)\): slopes 0.95 and 0.88, \(R^2 = 0.999\), and one iteration at every depth when \(\lambda = 1\). The start-state policy moves in the iteration the value arrives. Every cell solves. Freezing the critic restores a genuine failure at \(\lambda = 0.95\), \(D = 128\). A segment of 16 steps adds a second latency that scales linearly in \(D\) and does not match \(\min(T, 1/(1-\lambda))\); on these cells the two slowdowns add as rates, with \(T_{\mathrm{eff}} = 6.1\).

## References

Huang et al. (2022). PPO implementation details. · Schulman et al. (2016). Generalised advantage estimation. · Schulman et al. (2017). Proximal policy optimization. · Shao et al. (2024). Group relative policy optimization. · Tsitsiklis and Van Roy (1997). An analysis of temporal-difference learning with function approximation. · Yuan et al. (2025). VC-PPO. arXiv:2503.01491.
