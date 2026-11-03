# Exploration or Propagation? An Accounting of Time-to-Solve on Sparse Corridors

*RLC-length version; every number is from this folder's results.*

## Abstract

On deterministic corridors of depth 10, 20, and 30, a tabular Q-learner with a hard target refreshed every 2,000 updates and one-step backups spends most of its time after it has already found the reward. Median discovery times are 346, 1,862, and 9,324 steps; median times from that discovery to the first successful greedy episode are 18,154, 37,716, and 57,648. The post-discovery shares of time-to-solve are 0.98, 0.95, and 0.86. Across the 18 cells with no exploration bonus, median propagation time tracks \((\lceil D/n \rceil - 1)K + 1\) at Spearman correlation 0.92, and a post-hoc two-term fit is \(T_p = 0.96\,(\lceil D/n \rceil - 1)K + 59.3\, D^2/(nB) + 584\) with \(R^2 = 0.996\). Replacing that cell with \(K = 1\) and \(n = 5\), the same \(\varepsilon\)-greedy behaviour, cuts time-to-solve by the recorded factors 31×, 18×, and 7×. A count bonus at \(\beta = 0.1\), the pre-registered exploration control, does not: it halves discovery at depths 10 and 20, misses at depth 30 (10,055 against 9,324), shortens median discovery in 5 of 18 cells, lengthens median propagation in 16 of 18, and makes total time-to-solve worse in 16 of 18.

## 1 Introduction

A sparse-reward task is called a hard-exploration task, and the methods built for it — count bonuses, random network distillation, archives, episodic novelty — are scored by how soon the reward is found and by the return at the end. Finding the reward once is not the end of a value-based learner. The value of that transition has to travel back to the start through bootstrapping. With a hard target table refreshed every \(K\) updates, a reward \(D\) steps away cannot move the start state before \((\lceil D/n \rceil - 1)K\) updates, plus whatever uniform replay adds while the frontier waits to be sampled. At Nature-DQN settings \(K = 2{,}500\), \(n = 1\), a 30-step path is 72,500 updates of that target term alone, longer than the random-walk discovery time of a short corridor.

This experiment separates the two clocks on corridors where discovery is a random walk of known depth and the greedy policy can be rolled out exactly. Discovery \(T_d\) is the step of the first goal. Propagation \(T_p\) is the number of steps from then until the first successful greedy episode. The pre-registered claim is that under a slow target the second clock is the longer one, and that it follows the target-sync count in \(K\) and \(n\). Both parts are supported. The pre-registered claim that a count bonus would halve discovery at every depth is not.

**Contributions.** (1) The split, under \(K = 2{,}000\), \(n = 1\): propagation is 52×, 20×, and 6× discovery at depths 10, 20, and 30, and it is 0.98, 0.95, and 0.86 of time-to-solve. (2) The identification of that clock with the target-sync count, Spearman 0.92, and a two-term fit with target coefficient 0.96 and \(R^2 = 0.996\). (3) The same exploration with the latency removed (\(K = 1\), \(n = 5\)) cuts time-to-solve by 31×, 18×, and 7×. (4) The bonus control, which fails, and which lengthens the propagation half in 16 of 18 cells.

## 2 Related work

Count-based bonuses (Bellemare et al., 2016; Tang et al., 2017), random network distillation (Burda et al., 2019), Go-Explore (Ecoffet et al., 2021), and NGU and Agent57 (Badia et al., 2020) are evaluated by time-to-reward or by final return. RUDDER (Arjona-Medina et al., 2019) names three sub-problems of delayed reward — discover, keep, and learn — and does not measure their shares of time-to-solve. Go-Explore's second phase, robustification, is the closest acknowledgement that work remains after the first reward. Laidlaw et al. (2023) predict difficulty from one effective-horizon scalar that mixes the two halves. Fedus et al. (2020) find that multi-step returns are what make replay work. Target networks are usually discussed as a stabiliser (Kim et al., 2019; Lyle et al.). Stale intrinsic rewards in a replay buffer are treated as an implementation detail in the follow-ups to NGU and RND; the table below says which half of time-to-solve they land on in this agent. Within-episode reward delay is a different clock from the training-time latency measured here.

## 3 Setting

The environment is a corridor of depth \(D \in \{10, 20, 30\}\). The agent starts at 0. Two actions, left and right. Reaching \(D\) pays 1 and terminates. The episode cap is \(4D\). Slip is 0. The discount is \(\gamma = 0.99\).

The learner is tabular Q-learning. Uniform replay, buffer 20,000, batch 32, one update per environment step after step 200, step size \(\alpha = 0.1\), uncorrected \(n\)-step targets, hard target table refreshed every \(K\) updates. Epsilon is 0.1 and constant. Greedy ties, defined by a Q-range below \(10^{-4}\), are broken uniformly at random, so the untrained agent is a symmetric random walk rather than an agent that breaks ties toward the goal. The count bonus is \(\beta/\sqrt{N(s')}\) added to the stored reward, at \(\beta \in \{0, 0.1\}\).

The factorial is \(K \in \{1, 200, 2000\}\), \(n \in \{1, 5\}\), \(\beta \in \{0, 0.1\}\), eight seeds, 300,000 steps: 288 runs. Every 500 steps the greedy policy is rolled out exactly from the start for \(4D\) steps on the deterministic corridor. Success means it reaches the goal. \(T_d\) is the step of the first goal during training. \(T_p\) is the step of the first successful greedy rollout, minus \(T_d\). The share is \(T_p/(T_d + T_p)\). Runs that never discover, or that discover and never succeed, were to be reported and dropped from the medians. None occurred: all 288 runs discovered and solved. The pilot took about 22 CPU minutes.

- **K1.** In the cell \(K = 2{,}000\), \(n = 1\), \(\beta = 0\), median \(T_p >\) median \(T_d\) at every depth.
- **K2.** Over the \(\beta = 0\) cells with at least four discoveries, the Spearman correlation of median \(T_p\) with \((\lceil D/n \rceil - 1)K + 1\) is at least 0.8.
- **K3.** At \(K = 2{,}000\), \(n = 1\), the bonus cuts median \(T_d\) by at least \(2\times\) at every depth. Its effect on \(T_p\) is reported with no threshold.

Supported if K1 and K2. K3 is the exploration control and does not enter the verdict.

## 4 Results

The recorded verdict is **supported**. K1 and K2 pass. K3 fails at depth 30.

### 4.1 Propagation is the longer half

At \(K = 2{,}000\), \(n = 1\), no bonus, the medians are \(T_d = 346\), \(1{,}862\), \(9{,}324\) and \(T_p = 18{,}154\), \(37{,}716\), \(57{,}648\) at depths 10, 20, and 30. The ratios are 52×, 20×, and 6×. The post-discovery shares are 0.98, 0.95, and 0.86. The share falls as depth grows because discovery, a capped random walk, grows faster than the linear target-sync term. It is still the larger half at depth 30.

Median post-discovery share with no bonus:

| \(D\) | \(K=1, n=1\) | \(K=1, n=5\) | \(K=200, n=1\) | \(K=200, n=5\) | \(K=2000, n=1\) | \(K=2000, n=5\) |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.51 | 0.62 | 0.95 | 0.77 | 0.98 | 0.96 |
| 20 | 0.39 | 0.18 | 0.68 | 0.49 | 0.95 | 0.90 |
| 30 | 0.28 | 0.22 | 0.67 | 0.26 | 0.86 | 0.67 |

At \(K = 1\), \(n = 1\), propagation is already the minority at depth 20 (share 0.39) and depth 30 (share 0.28). At \(K = 2{,}000\) it is the majority at both backup lengths. The \(n = 5\) column cuts the share, and at \(K = 2{,}000\) the share remains 0.96, 0.90, and 0.67: five-step returns shrink a 2,000-update period and do not remove it.

### 4.2 The post-discovery clock is the target-sync count

Over the 18 cells with \(\beta = 0\), Spearman correlation of median \(T_p\) with \((\lceil D/n \rceil - 1)K + 1\) is 0.92. That is K2, against a threshold of 0.8. The predictor is only the target term, plus one update so that a zero-hop cell is defined.

A post-hoc fit of both terms on those same 18 cells is

\(T_p = 0.96\,(\lceil D/n \rceil - 1)K + 59.3\, D^2/(nB) + 584\),

with \(R^2 = 0.996\). The target term alone has \(R^2 = 0.995\): on this grid the replay term is what carries the \(K = 1\) cells, and the slow-target cells are the sync count. The target-term coefficient is 0.96. At \(K = 2{,}000\) the post-discovery time is the number of target syncs times \(K\), to within 4%. The two-term fit was not pre-registered. K2 was.

### 4.3 Removing the latency, without touching exploration

The cell \(K = 1\), \(n = 5\), \(\beta = 0\) uses the same \(\varepsilon\), the same tie-breaking, and the same corridor as \(K = 2{,}000\), \(n = 1\). Median time-to-solve is 500 against 18,500 at depth 10, 2,250 against 39,500 at depth 20, and 9,250 against 67,500 at depth 30. The recorded reduction factors are 31×, 18×, and 7×. The post-discovery shares in the fast cell are 0.62, 0.18, and 0.22. The intervention that changes time-to-solve by that factor does not change the behaviour policy. It changes how many target refreshes stand between the first reward and a greedy policy that can use it.

### 4.4 The bonus control fails, and it lengthens propagation

K3 required the \(\beta = 0.1\) bonus to cut median \(T_d\) by at least \(2\times\) at every depth in the cell \(K = 2{,}000\), \(n = 1\). The three comparisons are true, true, and false. At depth 30 the medians are 9,324 without the bonus and 10,055 with it.

Median discovery, propagation, and total time-to-solve, without and with the bonus. Each entry is a median over seeds; the median of the sum need not equal the sum of the medians.

| \(D\) | \(K\) | \(n\) | \(T_d\) off | \(T_d\) on | \(T_p\) off | \(T_p\) on | total off | total on |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 1 | 1 | 246 | 59 | 290 | 6,205 | 500 | 6,500 |
| 10 | 1 | 5 | 215 | 422 | 381 | 6,384 | 500 | 7,250 |
| 10 | 200 | 1 | 90 | 193 | 1,945 | 18,346 | 2,000 | 18,500 |
| 10 | 200 | 5 | 116 | 214 | 392 | 476 | 500 | 1,250 |
| 10 | 2,000 | 1 | 346 | 106 | 18,154 | 3,682 | 18,500 | 4,000 |
| 10 | 2,000 | 5 | 111 | 148 | 2,389 | 2,202 | 2,500 | 2,500 |
| 20 | 1 | 1 | 1,470 | 4,042 | 856 | 10,159 | 2,250 | 14,000 |
| 20 | 1 | 5 | 1,874 | 3,746 | 376 | 13,480 | 2,250 | 19,000 |
| 20 | 200 | 1 | 1,910 | 4,902 | 3,982 | 20,226 | 6,000 | 26,000 |
| 20 | 200 | 5 | 1,188 | 3,303 | 1,058 | 14,447 | 2,250 | 17,000 |
| 20 | 2,000 | 1 | 1,862 | 586 | 37,716 | 81,020 | 39,500 | 82,500 |
| 20 | 2,000 | 5 | 740 | 918 | 6,037 | 21,095 | 6,750 | 22,750 |
| 30 | 1 | 1 | 6,746 | 8,650 | 3,606 | 10,364 | 11,000 | 20,500 |
| 30 | 1 | 5 | 7,653 | 4,581 | 1,597 | 17,919 | 9,250 | 21,750 |
| 30 | 200 | 1 | 3,229 | 17,922 | 6,822 | 11,653 | 10,000 | 32,000 |
| 30 | 200 | 5 | 8,746 | 12,288 | 2,412 | 11,379 | 12,000 | 23,500 |
| 30 | 2,000 | 1 | 9,324 | 10,055 | 57,648 | 78,346 | 67,500 | 95,000 |
| 30 | 2,000 | 5 | 6,576 | 5,196 | 13,651 | 18,304 | 20,750 | 25,000 |

The bonus shortens median discovery in 5 of 18 cells and lengthens median propagation in 16 of 18. Median total time-to-solve is worse in 16 of 18. At depth 10, \(K = 2{,}000\), \(n = 1\), discovery goes from 346 to 106, propagation from 18,154 to 3,682, and the total from 18,500 to 4,000. At depth 20 in that same \((K, n)\) the bonus cuts discovery from 1,862 to 586 and raises propagation from 37,716 to 81,020, and the total goes from 39,500 to 82,500.

The bonus used here is a weak explorer: \(\beta = 0.1\), and the bonus written into the buffer at collection time is the bonus replayed later. A candidate mechanism, not tested by any arm in this grid, is that a stale stored bonus injects persistent value noise that competes with a true signal attenuated by \(\gamma^D\), and that this noise falls on the propagation half. A stronger bonus or a bonus recomputed at replay time is a different control. This grid does not contain it.

## 5 Discussion

Under the slow target used as the Nature-DQN-like cell of this grid, time-to-solve on these corridors is mostly a wait with a closed form, and the wait is the number of target refreshes between the first reward and the start state. The intervention matched to that form — one-step targets replaced by five-step targets, and the target table refreshed every update — cuts time-to-solve by 31×, 18×, and 7× with the behaviour policy held fixed. The intervention matched to the exploration framing shortens the smaller half in a minority of cells and lengthens the larger half in 16 of 18.

The share is not a constant of sparse-reward learning. It is large when discovery is short beside \((D/n)K\), and the table shows it falling from 0.98 to 0.86 as depth goes from 10 to 30 under the slow one-step target. Discovery in this experiment is hundreds to about 10,000 steps by design. A task whose first reward takes \(10^6\) steps or more would put a smaller fraction on the same propagation term. The claim the grid supports is the decomposition, the law for the propagation half, and the sign of this particular bonus's effect on that half.

Rainbow's \(n = 3\), R2D2's \(n = 5\), and Agent57's long \(n\)-step traces shrink the term \((\lceil D/n \rceil - 1)K\). On this accounting that is the term that dominated time-to-solve. An evaluation that reports first-reward time and final return, and not the gap between them, cannot see a bonus that buys discovery and sells propagation.

## 6 Limitations

The domain is a deterministic one-dimensional corridor and the learner is a table, which is the setting in which a target-sync count is a theorem about indices rather than a statement about a network. The new content relative to that count is the share of time-to-solve and the bonus trade-off. The bonus arm is \(\beta = 0.1\) with the bonus stored at collection time. It is a weak explorer on these corridors, and no non-stale or larger-\(\beta\) arm was run, so the 16/18 result is about this implementation. Greedy success is exact; an \(\varepsilon\)-greedy return would lag it. Epsilon is constant, so the discovery process does not anneal. DQN on MinAtar and on MiniGrid hard-exploration tasks, with the same discovery/propagation split under a factorial of target period, \(n\), and bonus, has not been run.

## 7 Conclusion

Discovery and propagation are different intervals, and on this grid they have different fixes. At a target period of 2,000 updates and one-step backups, the interval after the first reward is 6 to 52 times the interval before it and is 86 to 98 percent of time-to-solve. That interval tracks \((\lceil D/n \rceil - 1)K\) at Spearman 0.92, with a fitted target coefficient of 0.96. Removing it, at \(K = 1\) and \(n = 5\), cuts time-to-solve by 31×, 18×, and 7×. The count bonus that was supposed to halve discovery does so at two depths out of three, and it makes the total worse in 16 of 18 cells.

## References

Arjona-Medina et al. (2019). RUDDER: return decomposition for delayed rewards. · Badia et al. (2020). Never give up, and Agent57. · Bellemare et al. (2016). Unifying count-based exploration and intrinsic motivation. · Burda et al. (2019). Exploration by random network distillation. · Ecoffet et al. (2021). First return, then explore. Nature. · Fedus et al. (2020). Revisiting fundamentals of experience replay. ICML. · Hessel et al. (2018). Rainbow. · Kapturowski et al. (2019). Recurrent experience replay in distributed reinforcement learning. ICLR. · Laidlaw et al. (2023). Bridging RL theory and practice with the effective horizon. · Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature. · Tang et al. (2017). #Exploration: a study of count-based exploration for deep reinforcement learning.
