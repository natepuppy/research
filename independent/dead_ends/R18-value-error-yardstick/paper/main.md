# Which Value-Error Statistic Predicts the Greedy Policy? A Negative Result on Diffuse Errors and a Caution About Overestimation Plots

*Draft. Pilot numbers are from the laptop run in `results/`; sections marked [3080] are planned.*

## Abstract

Theory bounds a greedy policy's suboptimality by the sup-norm of the value error; practice reports value accuracy as RMS error or as "overestimation" and argues that less of it means better policies. We asked whether any such statistic predicts exact greedy-policy suboptimality on a corpus of 5760 Q-functions produced by tabular Q-learning on 24 random MDPs, comparing eleven statistics with training time held fixed so that progress alone cannot carry the correlation. The pre-registered hypothesis, that norms fail and a gap-crossing statistic (the visitation mass of states where the error flips the argmax) succeeds, was killed: every norm predicts moderately (Spearman 0.72–0.78), the gap-crossing statistic predicts better (0.93) but not categorically, and weighting it by the policy's own visitation adds 0.02. The errors in this corpus are diffuse (sup/RMS ratio 2.9, the i.i.d. value), which is why all norms are proportional; the regime the hypothesis is about, errors concentrated off the policy's path, has to be produced deliberately. Two secondary findings stand: the Bellman error is the weakest predictor (0.58), and the overestimation statistic has the *wrong sign* at fixed training time (ρ = −0.65): learners that are further behind under-estimate more and perform worse, so "more overestimation" accompanies better policies. Overestimation plots compared across methods at matched training time measure progress, not bias. [3080] A corpus with localised errors (narrow behaviour data, linear and deep FA with exact-MC truth) is planned.

## 1. Introduction

The performance guarantee for a greedy policy, V* − V^{π̂} ≤ 2‖Q̂ − Q*‖∞/(1−γ) (Singh & Yee, 1994), is the canonical link between value accuracy and control. In practice, value accuracy is reported as a plot of estimated against true Q (van Hasselt et al., 2016; Fujimoto et al., 2018) and the argument "our method overestimates less, therefore performs better" is standard. Fujimoto et al. (2022) showed that the quantity training actually minimises, the Bellman error, does not track the value error. We ask the next question: does any of these value-error statistics track performance, at fixed task and fixed training time, across learners that differ in their hyperparameters and noise? And is there a statistic that does?

The hypothesis we pre-registered was that no norm would, because only errors that flip an argmax at states the policy visits matter, and that a gap-crossing statistic would. The pilot killed the first half: on the corpus we built, every norm predicts. The reason is instructive and the two secondary results are useful.

Contributions. (1) A controlled horse race of eleven value-accuracy statistics as predictors of exact greedy suboptimality, with time held fixed. (2) The negative result that in a diffuse-error regime all norms predict at 0.72–0.78 and the best gap-aware statistic at 0.93. (3) The sign result: at matched training time, overestimation correlates *negatively* with suboptimality, because under-estimation is what an upward-learning learner has more of when it is behind. (4) [3080] The localised-error corpus.

## 2. Related work

Sup-norm bounds: Singh & Yee (1994); Bertsekas & Tsitsiklis (1996). Action-gap-aware bounds: Farahmand (2011); Bellemare et al. (2016) increase the gap by design. Bellman error versus value error: Fujimoto et al. (2022). Overestimation as a reported diagnostic: van Hasselt et al. (2016), Fujimoto et al. (2018), Chen et al. (2021). Our R15 and R16 use overestimation at convergence as a target variable; the present result says the time-of-learning confound must be removed before overestimation is read as bias.

## 3. Setting

24 Garnet MDPs (|S| = 50, |A| = 4, branching 3, γ = 0.9), 12 dense (rewards N(0,1) per (s, a)) and 12 sparse (5% nonzero); ρ uniform. Learners: online Q-learning, ε-greedy, restart to a uniform state every 50 steps, α ∈ {0.05, 0.2} × ε ∈ {0.1, 0.5} × reward noise σ ∈ {0, 0.5} × 3 seeds, 100k steps, Q̂ at 10 checkpoints. Per Q̂: greedy π̂, exact V^{π̂}, Δ = Σ ρ(V* − V^{π̂}), d^{π*}, d^{π̂}. Statistics: sup, RMS, RMS weighted by d^{π*} and d^{π̂}, Bellman sup and weighted RMS, overestimation under d^{π̂} and its absolute value, and gap-crossing mass (fraction of states where π̂ is suboptimal) unweighted, under d^{π*}, under d^{π̂}. Spearman with Δ within each (MDP, checkpoint) group of 24 learners, averaged over 240 groups.

Pre-registered: K1 ρ(sup) ≤ 0.5; K2 ρ(cross_dhat) ≥ 0.8; K3 cross_dhat beats the best norm by ≥ 0.1 and cross_unif by ≥ 0.05. SUPPORTED if all three.

## 4. Results

KILLED: K1 fails (ρ = 0.72), K2 passes (0.93), K3 half-passes (margin 0.15 over the best norm; 0.02 over the unweighted crossing).

| statistic | time-controlled ρ | pooled ρ |
|---|---|---|
| sup | 0.72 | 0.76 |
| RMS | 0.75 | 0.79 |
| RMS, d^{π*} | 0.78 | 0.82 |
| RMS, d^{π̂} | 0.77 | 0.81 |
| Bellman sup | 0.58 | 0.62 |
| Bellman RMS, d^{π̂} | 0.77 | 0.83 |
| overestimation, d^{π̂} | −0.65 | −0.73 |
| gap-crossing, uniform | 0.91 | 0.93 |
| gap-crossing, d^{π*} | 0.92 | 0.94 |
| gap-crossing, d^{π̂} | 0.93 | 0.95 |

### 4.1 Why the norms work here

The sup/RMS ratio of the error is 2.9 (median), which is what i.i.d. Gaussian noise over 200 entries gives: the errors are spread over the state-action space, not concentrated. ε-greedy with periodic restarts visits every state, and every state's error shrinks together. When errors are diffuse, all norms are proportional to one another (ρ(sup, RMS) = 0.95) and each is a proxy for how far along the learner is.

### 4.2 What the gap-crossing statistic adds

0.93 against 0.78: it discards error that does not change an action. The weighting by where the policy goes adds 0.02, because with diffuse errors and uniform ρ, where the policy goes does not select a very different set of states.

### 4.3 Overestimation has the wrong sign at fixed time

88% of the Q̂s under-estimate (zero initialisation, values learned upward). At fixed time, the learners that are behind (small α, small ε) under-estimate more and have worse policies, so overestimation correlates with *lower* suboptimality (ρ = −0.65). A plot of estimated-vs-true Q at matched training time, compared across methods, measures progress, and the method that is further along looks "more overestimating". The reading "less overestimation → better" requires measuring at convergence or at matched progress.

### 4.4 [3080] Planned

1. Localised-error corpus: the same MDPs with narrow behaviour (ε = 0.05, no restarts, fixed start), linear FA with aliasing (random features of width 10–40), and DQN on CartPole/MinAtar with exact-MC truth at probe states. Prediction: in this corpus the sup-norm's ρ falls below 0.5 while the gap-crossing statistic under d^{π̂} stays above 0.8 and the d^{π̂} weighting now matters.
2. Overestimation at matched progress: re-plot Double DQN vs DQN on MinAtar with the x-axis as greedy return rather than steps, to see how much of the published overestimation gap survives.

## 5. Consequences

For the regime the pilot covers, the norms are fine and the sup-norm is merely the weakest of them; the gap-crossing statistic is a better diagnostic when Q* is available. The transferable result is the sign of overestimation at fixed time: comparisons of overestimation across methods should be made at matched progress.

## 6. Limitations

Tabular, uniform ρ, one γ, small MDPs, and a corpus whose error structure did not test the hypothesis's intended regime. Within-group Spearman on 24 points is noisy per group.

## 7. Reproducibility

`code/r18_yardstick.py`, 16 seconds on a laptop CPU. `results/runs.csv` has every Q̂'s statistics and Δ.

## References

Bellemare, Ostrovski, Guez, Thomas, Munos (2016). Increasing the action gap. AAAI.
Bertsekas, Tsitsiklis (1996). Neuro-dynamic programming.
Chen et al. (2021). Randomized ensembled double Q-learning. ICLR.
Farahmand (2011). Action-gap phenomenon in RL. NeurIPS.
Fujimoto, van Hoof, Meger (2018). Addressing function approximation error in actor-critic methods. ICML.
Fujimoto, Meger, Precup, Nachum, Gu (2022). Why should I trust you, Bellman? ICLR.
Singh, Yee (1994). An upper bound on the loss from approximate optimal-value functions. Machine Learning.
van Hasselt, Guez, Silver (2016). Deep RL with double Q-learning. AAAI.
