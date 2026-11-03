# Policy Churn Is Not Controlled by the Target Network, and Only Matters When Nothing Else Explores

*Conference-length version (RLC format). Every number here comes from this folder's results.*

## Abstract

Schaul, Barreto, Quan and Ostrovski (2022) showed that deep value-based agents change their greedy action in a large fraction of states after every gradient update, and conjectured that this policy churn is an unintended source of exploration. We test two consequences on CartPole, in a \(4 \times 3\) factorial over the target-network period \(K\) and a constant \(\varepsilon\): that a longer target period damps churn, and that explicit exploration becomes more valuable as that damping is applied. Verdict: KILLED. Mean churn rises with \(K\) (Spearman \(+0.88\) over the 12 cells; the bar was \(\le -0.8\)). The AUC gain of \(\varepsilon = 0.2\) over \(\varepsilon = 0\) is \(+0\), \(+79\), \(+65\), \(+73\) at \(K = 20, 100, 500, 2500\) (Spearman \(+0.40\); the bar was \(\ge 0.8\)). The mediation clause passed: among \(\varepsilon = 0\) runs, Spearman of AUC on early churn is \(+0.72\). Dropping the degenerate \(K = 20\) cell, early churn and AUC have Spearman \(+0.70\) to \(+0.83\) inside each remaining \(K\), and \(+0.80\) pooled (\(n = 24\)). The pooled correlation falls to \(+0.67\) at \(\varepsilon = 0.05\) and to \(-0.02\) at \(\varepsilon = 0.2\). Churn and \(\varepsilon\) line up on the seed axis. The target period is the wrong knob, and the correlation does not say which way the causation goes.

## 1 Introduction

In DQN-style agents the greedy action changes after a single gradient step in far more states than a tie-breaking story would suggest. Schaul et al. (2022) put the rate near 10% of states and argued that the flips are a hidden exploration mechanism. Tang and Berseth (2024) regularise churn away (CHAIN) and report gains, which cuts the other way: churn can be harmful. Neither line asks which of the knobs a DQN already has actually moves churn, or tests the substitution a true exploration mechanism implies. Remove the mechanism, and deliberate exploration should matter more.

The natural knob is the target period \(K\). A hard target, copied every \(K\) updates, sets how far the online network can drift between re-anchorings. \(K\) is chosen for stability. If churn is drift, a longer period should produce more of it; the conjecture we pre-registered ran the other way, that \(K\) damps churn, and that the value of \(\varepsilon\) should then rise with \(K\). A one-seed smoke run had already hinted that churn might increase with \(K\). The tests were written in the hypothesis's direction anyway.

The factorial kills that mechanism. Churn rises with \(K\). The gain from \(\varepsilon\) is large and not monotone in \(K\). What survives is a correlation, at \(\varepsilon = 0\), between early churn and area under the learning curve, which disappears once \(\varepsilon\) is supplying exploration of its own.

**Contributions.** (1) A pre-registered factorial on CartPole: Spearman of mean churn on \(\log K\) is \(+0.88\), so the target period does not damp per-update greedy flips. (2) The mediation result that was pre-registered, \(+0.72\), and the post-hoc gradient in \(\varepsilon\): \(+0.80\) at \(\varepsilon = 0\) (\(n = 24\)), \(+0.67\) at \(\varepsilon = 0.05\), \(-0.02\) at \(\varepsilon = 0.2\). (3) A statement of the causal question this correlation leaves open. A direct churn manipulation was not run.

## 2 Related work

Schaul et al. (2022) define churn, show it is not an artefact of near-ties, and conjecture an exploration role. Tang and Berseth (2024) penalise it and report improvements in several settings. Target networks are from Mnih et al. (2015). Analyses of the target period treat it as a stability and propagation trade-off (Lee and He, 2019; Zhang, Yao and Whiteson, 2021). Exploration that arrives through representation change includes the tandem effect (Ostrovski, Castro and Dabney, 2021) and the primacy bias (Nikishin et al., 2022). The present experiment asks whether the largest reported incidental source of exploration is controlled by a standard stability knob.

## 3 Setting and the pre-registered test

Batched DQN on CartPole-v1 with a 200-step cap. The network is a \(2 \times 64\) MLP, trained with Adam, replay capacity 10,000, batch 64, \(\gamma = 0.99\), one update per environment step. The grid is \(K \in \{20, 100, 500, 2500\}\) by constant \(\varepsilon \in \{0, 0.05, 0.2\}\), with no annealing, so \(\varepsilon\) is the only deliberate exploration. Eight seeds are shared across cells. Each run is 30,000 steps. Greedy evaluation every 1,000 steps; AUC is the mean of those scores. Churn is the fraction of 256 fixed probe states whose greedy action changes across one update, averaged inside each evaluation interval, and reported below in percent. Early churn is the mean over the first 10,000 steps, the first third of training.

The grid starts at \(K = 20\) because a one-seed, 5,000-step smoke run with \(K = 1\) diverged to a constant-action policy. That smoke is not part of the factorial.

- **K1.** Spearman of mean churn on \(\log K\), over the 12 cells, \(\le -0.8\).
- **K2.** \(\mathrm{gain}(K) = \mathrm{AUC}(\varepsilon = 0.2) - \mathrm{AUC}(\varepsilon = 0)\). Spearman of gain on \(\log K\) \(\ge 0.8\), and \(\mathrm{gain}(2500) - \mathrm{gain}(20) \ge 20\).
- **K3.** Among \(\varepsilon = 0\) runs, Spearman of AUC on early churn \(\ge 0.4\).

SUPPORTED if K1 and K2. INCONCLUSIVE if K1 and K3 only. KILLED otherwise. K3 alone does not support the mechanism that was registered, which needs \(K\) to control churn.

## 4 Results

### 4.1 Kill test

KILLED. K1 \(= +0.88\). K2 gains are \(+0\), \(+79\), \(+65\), \(+73\) at \(K = 20, 100, 500, 2500\), and the Spearman of those gains on \(\log K\) is \(+0.40\). K3 \(= +0.72\), which clears \(0.4\). The decision rule needs K1. The sign of K1 is the opposite of the one that was registered.

### 4.2 Cell means

Final return is the end-of-training evaluation; the parenthetical is the standard deviation across eight seeds. Churn and early churn are in percent of the 256 probe states.

| \(K\) | \(\varepsilon\) | final (sd) | AUC | churn % | early churn % |
|---:|---:|---|---:|---:|---:|
| 20 | 0 / 0.05 / 0.2 | 9.5 / 8.9 / 8.9 | 9.5 / 9.9 / 9.5 | 0.00 / 0.07 / 0.02 | 0.01 / 0.20 / 0.07 |
| 100 | 0 | 78.0 (86.8) | 58.8 | 0.34 | 0.25 |
| 100 | 0.05 | 68.9 (69.4) | 83.7 | 0.74 | 0.87 |
| 100 | 0.2 | 157.1 (49.7) | 138.1 | 1.83 | 1.52 |
| 500 | 0 | 77.8 (73.6) | 66.7 | 1.11 | 0.46 |
| 500 | 0.05 | 101.3 (24.8) | 114.4 | 2.21 | 1.50 |
| 500 | 0.2 | 113.2 (20.0) | 132.0 | 3.40 | 1.71 |
| 2500 | 0 | 63.7 (71.8) | 51.2 | 0.81 | 1.03 |
| 2500 | 0.05 | 159.7 (20.3) | 132.6 | 2.28 | 3.14 |
| 2500 | 0.2 | 130.4 (11.1) | 123.8 | 3.30 | 3.38 |

The early-churn entry at \(K = 2500\), \(\varepsilon = 0.2\) is the draft's 3.38, the third term of the reported monotone triple 1.52 / 1.71 / 3.38. The notes' cell table prints 5.3 in that position.

### 4.3 \(K = 20\) collapses

All 24 runs at \(K = 20\) reach a constant greedy action inside the first evaluation interval. Final returns lie in 8.8–10.2 at every \(\varepsilon\), and churn is about 0. A near-on-policy target on CartPole lets the max-operator feedback run away before a return signal arrives. These cells carry no information about churn. The post-hoc correlations below drop them. They remain inside the pre-registered K1, K2, and K3, which were computed on the full grid: the \(K = 20\) gain is \(+0\) because nothing learns at any \(\varepsilon\), and the \(\varepsilon = 0\) AUC range in that cell is 9–10.

### 4.4 Churn rises with the target period

Mean churn per update at \(K = 100 / 500 / 2500\), in percent: at \(\varepsilon = 0\), 0.34 / 1.11 / 0.81; at \(\varepsilon = 0.05\), 0.74 / 2.21 / 2.28; at \(\varepsilon = 0.2\), 1.83 / 3.40 / 3.30. The \(\varepsilon = 0\) row peaks at \(K = 500\) rather than at \(K = 2500\). Early churn is monotone in \(K\) on the two slices the draft reports: 0.25 / 0.46 / 1.03 at \(\varepsilon = 0\), and 1.52 / 1.71 / 3.38 at \(\varepsilon = 0.2\). Across runs and all \(\varepsilon\), Spearman of churn on \(\log K\) is \(+0.34\), and Spearman of early churn on \(\log K\) is \(+0.53\).

The reading that fits the direction: a frequently refreshed target keeps the online network close to a target it has mostly fit, and the residual update flips few greedy actions. A stale target leaves a large residual, and fitting that residual flips actions. Churn, as a per-update flip rate, tracks the distance from the current target. \(K\) sets that distance upward. The registered substitution along \(K\) — more \(\varepsilon\) needed as \(K\) removes churn — had the knob's sign backwards, which is why K2's rank correlation could not clear \(+0.8\). The gains at \(K = 100, 500, 2500\) are 79, 65, and 73 AUC points. \(\varepsilon\) still matters. It does not matter more at every step up in \(K\).

At \(\varepsilon = 0\) the outcome is bimodal inside each non-degenerate cell. AUC ranges are 9–146 at \(K = 100\), 9–163 at \(K = 500\), and 9–123 at \(K = 2500\). Final-return standard deviations at \(\varepsilon = 0\) are 86.8, 73.6, and 71.8; at \(\varepsilon = 0.2\) they are 49.7, 20.0, and 11.1. The notes summarise the two regimes as standard deviations 72–87 against 11–50. Some seeds solve the task. Others stay near a return of 10.

### 4.5 Mediation at \(\varepsilon = 0\), and its disappearance

Post hoc. Spearman of AUC on early churn, and on churn over the whole run, for the eight \(\varepsilon = 0\) seeds in each cell:

| \(K\) | early churn | churn | AUC range |
|---:|---:|---:|---|
| 20 | \(+0.05\) | \(+0.05\) | 9–10 |
| 100 | \(+0.78\) | \(+0.81\) | 9–146 |
| 500 | \(+0.70\) | \(+0.78\) | 9–163 |
| 2500 | \(+0.83\) | \(+0.88\) | 9–123 |

Pooled over \(K > 20\) and \(\varepsilon = 0\) (\(n = 24\)): Spearman \(+0.80\), \(p < 0.001\). The same pool at \(\varepsilon = 0.05\) is \(+0.67\) (\(p < 0.001\)). At \(\varepsilon = 0.2\) it is \(-0.02\) (\(p = 0.92\)).

The pre-registered K3 of \(+0.72\) is the correlation on all \(\varepsilon = 0\) runs, including the collapsed \(K = 20\) cell. The within-cell figures above are the same relationship with that cell removed, and they sit between \(+0.70\) and \(+0.83\). Seeds that learn at \(\varepsilon = 0\) are seeds whose networks churned early. Once \(\varepsilon = 0.2\) is flipping actions on its own, early churn no longer ranks AUC.

This is the substitution pattern, on the seed axis rather than the \(K\) axis. It is a correlation inside a run. A network whose values are moving changes its greedy actions, so "learning causes churn" produces the same Spearman. The pilot does not separate the two directions.

## 5 Discussion and limitations

The mechanism that was registered, \(K\) damps churn and thereby makes \(\varepsilon\) more necessary, is dead. Higher \(K\) means more per-update churn, not less. If churn is exploration, a staler target is giving the agent more of it, and the stability choice of \(K\) is also an exploration choice with that sign. The part of the conjecture that survives is narrower: at \(\varepsilon = 0\), and still at \(\varepsilon = 0.05\), early churn ranks learning; at \(\varepsilon = 0.2\) the rank correlation is \(-0.02\). In a small-\(\varepsilon\) regime, churn is a candidate account of seed-level bimodality. Measuring it is warranted before a regulariser removes it. Whether removing it would hurt is exactly the untested direction.

A CHAIN-style penalty crossed with \(\varepsilon\), an accounting of which exploratory actions came from churn rather than from \(\varepsilon\), and a \(K\) sweep on MinAtar with annealed \(\varepsilon\) were specified and not run. Until a manipulation moves churn at fixed learning progress, the \(+0.80\) is compatible with churn-as-exploration and with learning-causes-churn.

The evidence is CartPole, one architecture, one learning rate, constant \(\varepsilon\), 30,000 steps, and eight seeds, in 96 runs and about 11 minutes on a laptop CPU. The \(K = 20\) collapse may be specific to that learning rate. Probe states are fixed early, so their overlap with the late-training distribution is not controlled. Means and standard deviations in the \(\varepsilon = 0\) cells summarise a bimodal outcome. Early churn and AUC are measured on the same run.

## 6 Conclusion

On this CartPole factorial the target period does not damp policy churn. Mean churn and \(\log K\) have Spearman \(+0.88\) across the 12 cells, and the AUC gain of \(\varepsilon = 0.2\) over \(\varepsilon = 0\) has Spearman \(+0.40\) on \(\log K\). The pre-registered mediation test passed, at \(+0.72\). With the degenerate \(K = 20\) runs removed, early churn predicts AUC at \(\varepsilon = 0\) inside every remaining target period (\(+0.70\) to \(+0.83\); pooled \(+0.80\), \(n = 24\)), the pooled correlation falls to \(+0.67\) at \(\varepsilon = 0.05\), and it is \(-0.02\) at \(\varepsilon = 0.2\). Churn tracks learning only when nothing else is exploring. The causal direction of that tracking was not identified.

## References

Lee, He (2019). Target-based temporal-difference learning. ICML. · Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature. · Nikishin et al. (2022). The primacy bias in deep reinforcement learning. ICML. · Ostrovski, Castro, Dabney (2021). The difficulty of passive learning in deep reinforcement learning. NeurIPS. · Schaul, Barreto, Quan, Ostrovski (2022). The phenomenon of policy churn. NeurIPS. · Tang, Berseth (2024). Improving deep reinforcement learning by reducing the chain effect of value and policy churn. NeurIPS. · Zhang, Yao, Whiteson (2021). Breaking the deadly triad with a target network. ICML.
