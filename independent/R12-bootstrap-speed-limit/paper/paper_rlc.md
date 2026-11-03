# The Speed Limit of Bootstrapping

*RLC-length version; every number is from this folder's results.*

## Abstract

A hard target network refreshed every \(K\) updates, and uniform replay of an episode of \(L\) transitions in batches of \(B\), set the pace at which a terminal reward can reach the start state. On a deterministic chain the propagation time is \(t_{\mathrm{prop}} = (\lceil L/n \rceil - 1)\,K + c\,\lceil L/n \rceil\,L/B\). The leading term is a counting lower bound, and in the target-limited regime it is attained: over 60 cells with \(K \ge 100\) and more than one hop, the median of \(t_{\mathrm{prop}} / ((\lceil L/n \rceil - 1)K)\) is 1.002 for per-hit SGD and 1.004 for per-state Adam. The two costs add. A pre-registered competing form, \(\lceil L/n \rceil \cdot \max(K, c\,L/B)\), was killed (SGD \(R^2 = 0.883\) against a threshold of 0.9; the Adam replay-limited slope was undefined). The additive law, with \(c\) fixed at the pilot values 2.63 (SGD) and 4.21 (Adam), was then confirmed on a disjoint grid: identity-fit \(R^2 = 0.997\) for SGD and 0.971 for Adam.

## 1 Introduction

Temporal-difference learning moves information about reward backwards one bootstrap hop at a time. Two design choices that every DQN-style agent makes constrain the speed of a hop, and they are usually chosen independently. A target network frozen for \(K\) updates cannot carry a newly learned value into the next hop until the next refresh. Uniform replay of an episode of \(L\) transitions samples the current frontier about once every \(L/B\) updates. For \(n\)-step targets the number of hops from a reward \(L\) steps away is \(h = \lceil L/n \rceil\).

The pre-registered guess was that the two waits compete, \(t_{\mathrm{prop}} \approx h \cdot \max(K,\, c\,L/B)\), with a crossover near \(K \approx c\,L/B\). The pilot rejected that form in the crossover region, which is where real DQN settings sit, and replaced it with a sum. Because the sum was read off the same data that killed the max, it was held to a fresh grid that shares no values with the pilot, with \(c\) not refit.

**Contributions.** (1) The counting lower bound: with a hard target, the start state stays at zero for \((h-1)K\) updates. (2) The pre-registered max form, killed on the pilot, and the additive refinement. (3) An out-of-sample identity fit of that refinement at fixed \(c\). (4) The design reading that follows on this chain: \(n\) divides both terms, and a long target period is a hard delay of \((h-1)K\) updates before a distant reward is felt at the start.

## 2 Related work

Target networks were introduced for stability (Mnih et al., 2015) and analysed as damped fixed-point iterations and as a piece of the deadly triad (Lee and He, 2019; van Hasselt et al., 2018; Zhang et al., 2021; Fellows et al., 2023). The speed cost of a long period is folklore. The closest prior statement is one sentence in Piché et al. (2022): on an \(N\)-state chain with tabular values and a target refreshed every \(H\) steps, propagating the terminal reward will require \(NH\) training steps. They use it to motivate replacing the target network. They do not test it, and they do not add an \(n\)-step factor or a replay term.

Replay goes back to Lin (1992) and Mnih et al. (2015). Schaul et al. (2016) prioritise by TD error. Fedus et al. (2020) find that a large buffer helps when the backup is multi-step. Reverse and episodic backward replay (Lee et al., 2019; Rotinov, 2019) order updates from the terminal backwards. Multi-step returns in DQN are used at \(n = 3\) and \(n = 5\) as tuned constants (Hessel et al., 2018; Kapturowski et al., 2019); Hernandez-Garcia and Sutton (2019) study \(n\) empirically. Classical horizon bounds (Bertsekas and Tsitsiklis, 1996; Bhandari et al., 2018) concern \(\gamma\). Here \(\gamma\) is held at 0.99 and the mechanics of the update are what vary. The quantity is the time until the signal arrives.

## 3 Setting and the counting argument

**Chain.** States \(0, 1, \ldots, L-1\) and a terminal state. The policy moves right. Reward 1 on the final transition, 0 elsewhere, discount \(\gamma = 0.99\). The true value is \(V(s) = \gamma^{L-1-s}\).

**Learner.** A value table \(V\) and a target table \(\bar V\). Each update draws \(B\) transitions uniformly from a stationary buffer that contains each of the \(L\) episode transitions equally often. For a transition out of \(s\), the \(n\)-step target is the exact discounted return if \(s + n \ge L\), and \(\gamma^n \bar V(s+n)\) otherwise. Two optimisers: per-hit SGD with step \(0.3\) on every sampled copy, and per-state Adam at learning rate \(0.05\) on the batch-mean gradient. \(\bar V \leftarrow V\) every \(K\) updates.

**Propagation time.** \(t_{\mathrm{prop}} = \min\{t : V_t(0) \ge \tfrac12 V(0)\}\). The cap is 400,000 updates. Every cell in both grids converged inside it.

**Proposition (target term).** Let \(h = \lceil L/n \rceil\). With a hard target, \(V_t(0) = 0\) for all \(t < (h-1)K\).

The states within \(n\) of the terminal receive exact targets and become nonzero at the first update that samples them. Any earlier state is regressed toward \(\bar V\) at a state that is still zero until the first refresh after that downstream value moves. By induction the start state, hop \(h\), cannot move before \(h-1\) refreshes. The bound does not depend on the step size, the batch, or the optimiser. The empirical claim is that the bound is attained up to a small additive remainder whenever \(K\) is large enough for the frontier to be fit between refreshes, and that a replay cost \(c \cdot h \cdot L/B\) sits on top of it rather than inside a max.

## 4 Experiments

### 4.1 Pilot: the max form is killed

The pilot grid, fixed before the run, is \(L \in \{8, 16, 32, 64\}\), \(n \in \{1, 2, 4, 8\}\), \(K \in \{1, 10, 100, 1000\}\), \(B \in \{4, 32\}\), both optimisers, three seeds: 256 configurations. It finished in 4 CPU minutes. The constant \(c\) was chosen by maximising \(R^2\). The tests were:

- **K1.** Per optimiser, \(R^2\) of \(\log t_{\mathrm{prop}}\) on \(\log[h \cdot \max(K, c\,L/B)]\) at least 0.9.
- **K2.** Inside groups of fixed \((L, n, B)\): slope of \(\log t_{\mathrm{prop}}\) on \(\log K\) at least 0.8 on target-limited cells (\(K \ge 10 \cdot c\,L/B\)) and at most 0.2 on replay-limited cells (\(K \le 0.1 \cdot c\,L/B\)).
- **K3.** Inside target-limited groups of fixed \((L, K, B)\), the slope of \(\log t_{\mathrm{prop}}\) on \(\log n\) lies in \([-1.3, -0.7]\).

Supported required K1 and K2 for both optimisers. The recorded verdict is **killed**.

SGD K1 is \(R^2 = 0.883\) at \(c^\ast = 7.68\), slope 1.10 (hops-only baseline \(R^2 = 0.44\), hops\(\times K\) baseline \(R^2 = 0.72\)). Adam K1 passes on the point estimate, \(R^2 = 0.902\) at \(c^\ast = 3.38\), slope 0.87 (baselines 0.34 and 0.80), and Adam K2 does not: the target-limited slope is 0.88 over 20 groups, and the replay-limited slope is undefined because that regime contained 0 groups. SGD K2 passes: target-limited slope 0.91 over 12 groups, replay-limited slope 0.03 over 4 groups. K3 fails for both, slopes \(-1.88\) (SGD) and \(-1.47\) (Adam). Those K3 groups include cells with \(n \ge L\), where the backup never bootstraps.

An identity fit of the same max prediction, no free slope or intercept, is worse still: \(R^2 = 0.860\) at \(c = 3.26\) (SGD) and \(R^2 = 0.881\) at \(c = 3.55\) (Adam).

### 4.2 What the pilot cells do instead

On the 60 cells with \(K \ge 100\) and \(h > 1\), \(t_{\mathrm{prop}} / ((h-1)K)\) has median 1.002 (range 1.000 to 1.225) for SGD and median 1.004 (range 1.000 to 1.465) for Adam. At \(L = 64\), \(n = 1\), \(K = 1000\), \(B = 32\), the times are 63,003 (SGD) and 63,005 (Adam) against \((h-1)K = 63{,}000\).

With \(K = 1\), \(\log t_{\mathrm{prop}}\) on \(\log(L^2/(nB))\) over 32 cells has slope 0.99 and \(R^2 = 0.990\) for SGD, and slope 0.61 and \(R^2 = 0.941\) for Adam. SGD matches the \(L^2/(nB)\) replay scaling. Adam is faster than that scaling and noisier, consistent with per-state normalisation taking large steps on the first hits to a frontier state that has just become nonzero.

Four recorded one-parameter forms, identity \(R^2\) in log space, \(c\) fit per optimiser:

| Form | SGD \(R^2\) (\(c\)) | Adam \(R^2\) (\(c\)) |
|---|---|---|
| \(h \cdot \max(K, c\,L/B)\) | 0.860 (3.26) | 0.881 (3.55) |
| \((h-1)K + c\,h\,L/B\) | 0.996 (2.63) | 0.924 (4.21) |
| \(h\sqrt{K^2 + (c\,L/B)^2}\) | 0.860 (2.99) | 0.890 (3.55) |
| \((h-1)(K + c\,L/B)\) | 0.898 (2.99) | 0.897 (3.70) |

The additive form is the one that fits. Restricting the \(n\)-slope to target-limited groups that actually bootstrap (\(h \ge 2\), 16 groups) moves it from the killed values to \(-1.21\) (SGD) and \(-1.18\) (Adam): near \(-1\), and a little steeper, as expected from \(h - 1 = \lceil L/n \rceil - 1\) rather than \(L/n\).

Table 1 is the \(n = 1\) slice (geometric mean over three seeds).

| \(L\) | \(K\) | \(B\) | \(t_{\mathrm{prop}}\) SGD | \(t_{\mathrm{prop}}\) Adam | \((h-1)K\) |
|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 32 | 422 | 186 | 63 |
| 64 | 10 | 32 | 790 | 642 | 630 |
| 64 | 100 | 32 | 6,303 | 6,309 | 6,300 |
| 64 | 1,000 | 32 | 63,003 | 63,005 | 63,000 |
| 64 | 1 | 4 | 3,422 | 1,328 | 63 |
| 64 | 100 | 4 | 7,445 | 6,426 | 6,300 |
| 16 | 1 | 32 | 26 | 35 | 15 |
| 16 | 1,000 | 32 | 15,001 | 15,003 | 15,000 |

At \(K = 1{,}000\), \(B = 32\), the two optimisers finish at 63,003 and 63,005 against \((h-1)K = 63{,}000\), and at \(L = 16\) they finish at 15,001 and 15,003 against 15,000. When \(K = 1\), cutting \(B\) from 32 to 4 moves the SGD time from 422 to 3,422.

### 4.3 Confirmation on a disjoint grid

The additive law was locked before the confirmation run: \(c_{\mathrm{SGD}} = 2.63\), \(c_{\mathrm{Adam}} = 4.21\), no refit, identity fit in log space (no free slope or intercept). The grid shares no coordinate with the pilot: \(L \in \{12, 24, 48, 96\}\), \(n \in \{1, 3, 6\}\), \(K \in \{3, 30, 300, 3000\}\), \(B \in \{8, 16\}\), three new seeds, 192 configurations (96 cells per optimiser). The pre-registered bar was \(R^2 \ge 0.95\) for SGD and \(R^2 \ge 0.90\) for Adam. The run took 1 CPU minute.

**Supported.** \(R^2 = 0.997\) (SGD) and \(0.971\) (Adam). Median \(t_{\mathrm{prop}}/\mathrm{pred}\) is 0.97 and 0.96. By target period the SGD medians are 0.98, 0.87, 0.97, 1.00 at \(K = 3, 30, 300, 3000\); the Adam medians are 0.62, 0.75, 0.96, 1.00. Adam's 10–90% range on the confirmation is 0.47 to 1.00, and the small-\(K\) medians sit below 1: the first hits on a fresh frontier are large under per-state normalisation, so replay-limited Adam arrives sooner than the constant fitted on the pilot predicts. The target-limited end of the grid lands on the prediction for both optimisers.

## 5 What the law says about the usual knobs

**\(n\) divides both terms.** Inside the cells that bootstrap, the slope on \(n\) is \(-1.21\) and \(-1.18\). Doubling \(n\) cuts the hop count in the target term and in the replay term together. Doubling \(K\) doubles only the first term. Lengthening the target period and leaving \(n\) alone pays the full \((h-1)K\).

**A long period is a hard delay.** Nature DQN's target period is \(K = 2{,}500\) updates at \(n = 1\), \(B = 32\). On this chain a reward 100 steps from the start has target term \(99 \times 2{,}500 \approx 250{,}000\) updates, which is 1M environment steps, before the start state can move. The replay term \(c \cdot 100 \cdot 100/32 \approx 800\) updates is negligible beside it. The same accounting at Rainbow's \(n = 3\) is about 83,000 updates, and at R2D2's \(n = 5\) about 50,000. The recorded regime statement for those Nature DQN settings is that they sit in the target-limited regime for any episode shorter than about 80 steps and in the replay-limited regime for longer ones. These figures are counts on a stylised chain. They are the reason the two hyperparameters are one choice.

**Two predictions the pilot does not test.** A frontier transition has the largest TD error, so prioritised replay should drop the replay term and leave \((h-1)K\) untouched, which would matter in the replay-limited regime and not in the target-limited one. Playing an episode's transitions from the terminal backwards inside one target period makes every hop fire in a single pass, leaving \(t_{\mathrm{prop}} \approx K\). A Polyak average at rate \(\tau\) is a natural candidate for an effective period \(1/\tau\). None of the three was run.

## 6 Limitations

The chains are tabular, deterministic, and uniformly replayed, with one terminal reward and a stationary buffer, so the frontier is a single state. The criterion \(V(0) \ge \tfrac12 V(0)\) marks arrival of the signal, which on the target-limited cells is the same event as the first nonzero value, and it does not mark accuracy. The constant \(c\) is one fitted scalar per optimiser, not a derived quantity, and Adam's \(K = 1\) slope of 0.61 shows that \(L^2/(nB)\) is not a universal replay exponent. Cells with \(n \ge L\) are Monte Carlo and fall out of the hop count, which is why the pre-registered K3 failed on the full grid. Stochastic transitions, function approximation, Polyak targets, prioritised replay, and the planned DQN runs on sparsified MinAtar have not been run.

## 7 Conclusion

Bootstrapping under a hard target has a speed limit with two additive pieces. The target piece is \((\lceil L/n \rceil - 1)K\), a counting lower bound, and on this pilot it is the measured time whenever it dominates: median ratio 1.002 (SGD) and 1.004 (Adam) on 60 cells. The replay piece is \(c\,\lceil L/n \rceil\,L/B\), with \(c = 2.63\) for per-hit SGD at step 0.3 and \(c = 4.21\) for per-state Adam at learning rate 0.05. The pre-registered form that takes the max of the two pieces was killed. The sum, with those constants held fixed, matches a disjoint grid at identity-fit \(R^2 = 0.997\) and \(0.971\).

## References

Bertsekas and Tsitsiklis (1996). Neuro-Dynamic Programming. · Bhandari, Russo, Singal (2018). A finite time analysis of temporal difference learning with linear function approximation. COLT. · Fedus et al. (2020). Revisiting fundamentals of experience replay. ICML. · Fellows et al. (2023). Why target networks stabilise temporal difference methods. ICML. · Hernandez-Garcia and Sutton (2019). Understanding multi-step deep reinforcement learning: a systematic study of the DQN target. arXiv:1901.07510. · Hessel et al. (2018). Rainbow. AAAI. · Kapturowski et al. (2019). Recurrent experience replay in distributed reinforcement learning. ICLR. · Lee and He (2019). Target-based temporal-difference learning. ICML. · Lee, Choi, Chung (2019). Sample-efficient deep reinforcement learning via episodic backward update. NeurIPS. · Lin (1992). Self-improving reactive agents based on reinforcement learning, planning and teaching. Machine Learning. · Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature. · Piché et al. (2022). Beyond target networks: functional regularisation. · Rotinov (2019). Reverse experience replay. arXiv:1910.08780. · Schaul, Quan, Antonoglou, Silver (2016). Prioritized experience replay. ICLR. · Sutton (1988). Learning to predict by the methods of temporal differences. Machine Learning. · van Hasselt et al. (2018). Deep reinforcement learning and the deadly triad. arXiv:1812.02648. · Zhang, Yao, Whiteson (2021). Breaking the deadly triad with a target network. ICML.
