# The Value Support of C51 Is a Clipped Bellman Fixed Point

*Conference-length version (RLC format). Every number in this paper comes from this folder's results.*

## Abstract

C51 represents the return distribution on a fixed support \([-V_{\max}, V_{\max}]\). The categorical projection clips every bootstrapped atom, so the learned mean is the fixed point of a clipped Bellman operator. On a deterministic fork, wherever both actions' continuations saturate, that fixed point prefers the immediate reward by exactly \((1 - \gamma)V_{\max}\). A tabular study with the exact C51 projection, 51 atoms, \(\gamma = 0.99\), and 20{,}000 synchronous sweeps on 40 fork cells confirms the values and kills the pre-registered choice rule. The C51 mean matches clipped value iteration in 80/80 (cell, action) pairs, to within 5% of \(V_{\max}\) and in fact to within one atom width. The choice tests do not: agreement with the clipped prediction is 35/40, the myopia rule holds in 28/33 saturation cells, and a scalar clipped-Q control agrees in 39/40. The recorded verdict is **KILLED**. The misses are the threshold. The pre-registered indifference band of 0.05 sits on the margin it was meant to split: that margin is 0.03 at \(V_{\max} = 3\) and 0.05 at \(V_{\max} = 5\), and at \(V_{\max} = 10\) a 0.4 atom spacing leaves the top mean at 9.93. Zero of 18 saturated cells chose the patient action. The agent turns patient when \(V_{\max}\) exceeds the myopic action's true value, in all four forks, as the clipped operator predicts. The rule failed because the threshold was mis-set. The mechanism is confirmed. The test was not re-run.

## 1 Introduction

C51 and the agents that inherit its support (Rainbow, and the Dopamine defaults that carry it into later distributional agents) place the return distribution on 51 atoms in \([-10, 10]\) (Bellemare et al., 2017; Hessel et al., 2018; Castro et al., 2018). Atari rewards are clipped to \(\{-1, 0, 1\}\) and \(\gamma = 0.99\), so a competent discounted return in a dense-reward game is tens of reward units. The C51 paper notes that returns outside the support are projected onto the boundary. Tutorials observe that \(\pm 10\) does not bracket the achievable \(\pm 100\) and treat the mismatch as tolerable. QR-DQN removed the fixed support for flexibility (Dabney et al., 2018). The fixed point the clip induces, and the policy it selects, are the subject of this pilot.

The operator is \(T_c Q(s, a) = \mathbb E[\min(V_{\max},\, r + \gamma \max_{a'} Q(s', a'))]\). It is a \(\gamma\)-contraction in the sup norm, because a minimum with a constant is 1-Lipschitz, so it has a unique fixed point \(Q_c\), and \(Q_c \le \min(V_{\max}, Q^\ast)\) pointwise. On a fork where one action pays 1 now and the other pays 0 now, and both continuations are worth at least \(V_{\max}\),

\[
Q_c(\text{myopic}) = \min(V_{\max},\, 1 + \gamma V_{\max}) = V_{\max}, \qquad Q_c(\text{patient}) = \min(V_{\max},\, \gamma V_{\max}) = \gamma V_{\max}.
\]

The preference for the immediate reward is \((1 - \gamma)V_{\max}\), whatever the true values are. With the Atari pair \((V_{\max}, \gamma) = (10, 0.99)\) that preference is 0.1. The planning horizon of the clipped agent, in exact arithmetic, is "until \(V_{\max}\) reward units are in view."

**Contributions.** (1) The saturated preference \((1 - \gamma)V_{\max}\), and the patient boundary \(V_{\max} > 1 + \gamma V_M\), written for this fork and checked against clipped value iteration. (2) A deterministic tabular C51, with the exact projection, on 40 cells: values match the clipped fixed point in 80/80 pairs. (3) The pre-registered choice rule, which required 95% agreement and myopia exactly on the saturated cells. It scores 35/40 and 28/33. The verdict is KILLED, and the post-hoc account of every miss is the threshold, not a patient choice in a saturated cell (0/18). (4) A scalar clipped-Q control agrees in 39/40 cells, so the effect is the clip. The single miss is the same atom-resolution cell.

## 2 Related work

Bellemare, Dabney, and Munos (2017) introduce the categorical projection and note the boundary projection. Hessel et al. (2018) adopt \(V_{\max} = 10\) in Rainbow; Dopamine carries the default forward. Dabney et al. (2018) motivate QR-DQN in part by the fixed support. van Hasselt et al. (2016) show that reward clipping changes the objective and propose Pop-Art. Clipping and the support interact: clipping piles return mass into a few reward units, which is what pushes values through a support of width 10. Rowland et al. (2018) treat truncation of a return distribution as a technical device in the analysis of categorical distributional RL. Jiang et al. (2015) study the discount as a horizon. The support is a second horizon, in reward units, sitting on top of the discount's time horizon. A separate lab note treats the Huber loss as a risk preference; that is a different distortion of the same default agent.

## 3 The fork and the pre-registered tests

The MDP is deterministic, \(\gamma = 0.99\). From \(s_0\), `myopic` pays 1 and then a corridor of 1 per step for \(M\) steps. `patient` pays 0 and then a corridor of 1 per step for \(P\) steps, with \(P > M + 1\). The continuations are \(V_M = (1 - \gamma^M)/(1 - \gamma)\) and \(V_P = (1 - \gamma^P)/(1 - \gamma)\), so the true action values are \(1 + \gamma V_M\) and \(\gamma V_P\), and patient is optimal under the true return. The grid is \((M, P) \in \{(5, 12), (10, 25), (20, 50), (40, 100)\}\) and \(V_{\max} \in \{3, 5, 8, 10, 15, 20, 30, 50, 100, 200\}\): 40 cells.

Two learners see every state-action pair on every sweep, with step size 0.1, for 20{,}000 sweeps. Tabular C51 uses 51 atoms on \([-V_{\max}, V_{\max}]\) and the exact categorical projection, \(p \leftarrow p + 0.1(\Pi_{\text{target}} - p)\). Scalar clipped Q-learning uses the target \(\min(V_{\max},\, r + \gamma \max Q')\). The prediction is clipped value iteration, run to a maximum change below \(10^{-12}\). A cell is called indifferent when \(|\Delta Q| < 0.05\), for the prediction and for both learners. An indifferent prediction agrees only with an indifferent learned choice. There are no seeds: the MDPs are deterministic and the sweeps are exact.

The threshold and the sweep count were fixed before the pilot. A smoke run had used \(10^{-9}\) for the prediction, which labelled \(V_{\max} = 3\) cells myopic by 0.03 while the learner's 0.05 band called them indifferent; both sides were set to 0.05 before the recorded run. Sweeps were raised from 5{,}000 to 20{,}000 because a 100-state corridor had not converged in a short smoke.

- **K1.** C51's greedy choice at \(s_0\) agrees with the clipped prediction in at least 95% of 40 cells (38 cells).
- **K2.** The C51 mean is within 5% of \(V_{\max}\) of \(Q_c\), for both actions, in at least 95% of 80 pairs.
- **K3.** In every cell with \(V_{\max} < 1 + \gamma V_M\) (both sides saturate: 18 cells) C51 picks `myopic`. In every cell with \(V_{\max} > \gamma V_P\) (nothing saturates: 15 cells) C51 picks `patient`.
- **K4**, reported. The scalar control's choice agrees with C51 in every cell.

SUPPORTED if K1, K2, and K3 all hold. INCONCLUSIVE if exactly two hold. KILLED otherwise.

## 4 What the operator predicts, and what the run did

**Saturated preference.** In every both-saturate cell the clipped fixed point prefers the immediate reward by exactly \((1 - \gamma)V_{\max}\): 18/18 cells. The recorded examples are \(V_{\max} = 10 \to 0.10\) and \(V_{\max} = 3 \to 0.03\). At \(V_{\max} = 5\) the margin is 0.05, the indifference threshold itself.

**Patient boundary.** The clipped agent selects patient once \(V_{\max}\) exceeds the myopic action's true value \(1 + \gamma V_M\), so the myopic action no longer saturates while the patient action still can. The recorded boundaries are \(M = 10 \to 10.47\) (patient from \(V_{\max} = 15\) on) and \(M = 40 \to 33.77\) (patient from \(V_{\max} = 50\) on). The same switch matches clipped value iteration in all four forks.

The projection does not implement \(T_c\) exactly. An atom above \(V_{\max}\) is placed on the top atom, which is the clip. An atom strictly inside the support is split between neighbours, which shaves the mean by up to half an atom. Clipped value iteration is the prediction; C51 is the learner.

**The kill test.** K2 holds: 80/80. K1 is 35/40, K3 is 28/33, K4 is 39/40. One of K1–K3 holds. The verdict line is `KILL TEST: KILLED`.

The five K1 misses, recorded as (corridor length, \(V_{\max}\), prediction, C51 choice, margin), are \((5, 5,\ \text{indifferent},\ \text{myopic},\ 0.05)\), \((10, 5,\ \text{indifferent},\ \text{myopic},\ 0.05)\), \((10, 10,\ \text{myopic},\ \text{indifferent},\ 0.033)\), \((20, 5,\ \text{indifferent},\ \text{myopic},\ 0.05)\), and \((40, 5,\ \text{indifferent},\ \text{myopic},\ 0.05)\). Four are \(V_{\max} = 5\) cells where the prediction's margin falls under 0.05 by floating point and the learner's margin is 0.04999, so the labels split "indifferent" from "myopic" on a margin the operator set equal to the threshold. The fifth is \((M, P) = (10, 25)\) at \(V_{\max} = 10\): the 0.4 atom spacing leaves the top mean at 9.93, and the learned margin is 0.033 rather than 0.10. That cell is also the only K4 miss. The scalar control says myopic by 0.10; C51 says indifferent by 0.03. The five K3 misses are these same cells. Patient choices in saturated cells: 0/18.

The post-hoc records the C51 learned margins, in this order: 0.030, 0.050, 0.030, 0.050, 0.033, 0.030, 0.050, 0.100, 0.030, 0.050, 0.100, 0.159, tagged with \(V_{\max}\) equal to 3, 5, 3, 5, 10, 3, 5, 10, 3, 5, 10, 30. The clipped preference in all 18 both-saturate cells is exactly \((1 - \gamma)V_{\max}\).

| \((M, P)\) | true \(Q\), myopic / patient | \(V_{\max} = 10\): clipped VI | \(V_{\max} = 10\): C51 | C51 choice | \(V_{\max} = 100\): C51 |
|---|---|---|---|---|---|
| (5, 12) | 5.85 / 11.25 | 5.85 / 9.90 | 5.85 / 9.90 | patient | patient (5.85 / 11.25) |
| (10, 25) | 10.47 / 22.00 | 10.00 / 9.90 | 9.93 / 9.90 | indifferent | patient (10.47 / 22.00) |
| (20, 50) | 19.03 / 39.10 | 10.00 / 9.90 | 10.00 / 9.90 | myopic | patient (19.03 / 39.10) |
| (40, 100) | 33.77 / 62.76 | 10.00 / 9.90 | 10.00 / 9.90 | myopic | patient (33.77 / 62.76) |

At \(V_{\max} = 10\) the two longer forks are myopic by 0.10, the margin \((1 - 0.99) \times 10\). The shortest fork never saturates the myopic action (true value 5.85) and is patient. The middle fork is the atom-resolution miss. At \(V_{\max} = 100\) every fork is patient and the C51 means sit on the true values. Neighbouring logged cells agree with the boundary: \(M = 20\) is myopic at \(V_{\max} = 15\) (clipped 15.00 / 14.85) and patient at \(V_{\max} = 20\) (clipped 19.03 / 19.80; C51 18.92 / 19.80); \(M = 40\) is still myopic at \(V_{\max} = 30\) (C51 29.86 / 29.70) and patient at \(V_{\max} = 50\) (C51 33.77 / 49.48).

## 5 What the margin says, and what was not run

For \(V_{\max} = 10\), \(\gamma = 0.99\), and rewards clipped to \(\{-1, 0, 1\}\), any two actions whose continuations are both worth at least 10 are separated by at most 0.1 in the learned value, whatever their true values. In the longest fork those true values are 33.8 and 62.8. A continuation exceeds 10 when the clipped reward rate over the next ~100 steps exceeds about 0.1 per step. Under function-approximation noise larger than 0.1, a saturated state is not a myopic state in any robust sense: the greedy action among saturated actions is set by the noise. That is a statement about the size of the margin. It is not a measurement on a deep agent.

A re-run with the indifference threshold set to \((1 - \gamma)V_{\max}/2\), and with 101 atoms, is the correction the post-hoc names. It was not run. The recorded position is that the pre-registered threshold was mis-specified, and that the values, the 0/18 patient count, and the four boundaries already show the mechanism.

The Rainbow comparison that would carry a deep claim — \(V_{\max} = 10\) against a wider support, a saturation audit of trained agents, MinAtar, Atari — was not run. Nothing in this paper is a deep-RL result.

## 6 Limitations

The evidence is one deterministic fork family, tabular, with exact sweeps and no function approximation. The preference \((1 - \gamma)V_{\max}\) is exact when both backups sit above \(V_{\max}\); with stochastic rewards the saturated preference depends on the part of the return that still falls inside the support. The 0.05 band was a property of the test, and it made "myopic" and "indifferent" the same event at \(V_{\max} \le 5\). Reporting the verdict as KILLED is the pre-registered rule, applied once. Dopamine's Rainbow also uses \(n = 3\) multi-step targets, which the draft notes would replace the one-step margin by roughly \((1 - \gamma^3)V_{\max} \approx 0.3\); that wider margin was not measured here. The deep follow-up was not run.

## 7 Conclusion

The categorical support clips the Bellman operator. On these forks the clipped fixed point prefers the immediate reward by \((1 - \gamma)V_{\max}\) in every saturated cell, the tabular C51 mean matches that fixed point in 80/80 pairs, and the patient action is never chosen in those 18 cells. The pre-registered choice test is KILLED, because an indifference threshold of 0.05 was placed on a margin of 0.03 at \(V_{\max} = 3\) and of 0.05 at \(V_{\max} = 5\). The mechanism stands. The rule, as written, does not, and it was not rewritten after the fact.

## References

Bellemare, Dabney, Munos (2017). A distributional perspective on reinforcement learning. · Hessel et al. (2018). Rainbow. · Dabney et al. (2018). Distributional reinforcement learning with quantile regression. · van Hasselt et al. (2016). Learning values across many orders of magnitude. · Rowland et al. (2018). An analysis of categorical distributional reinforcement learning. · Jiang et al. (2015). The dependence of effective planning horizon on model accuracy. · Castro et al. (2018). Dopamine.
