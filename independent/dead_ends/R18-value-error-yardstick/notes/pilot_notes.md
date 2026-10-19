# R18 pilot notes

One run. 24 Garnet MDPs (|S| = 50, |A| = 4, branching 3, γ = 0.9; half dense, half sparse rewards) × 24 Q-learning configurations (α × ε × reward noise × 3 seeds) × 10 checkpoints = 5760 Q̂s with exact Q*, V^{π̂}, d^{π*}, d^{π̂}. Eleven statistics per Q̂, Spearman with the exact suboptimality Δ within each (MDP, checkpoint) group of 24 learners, averaged over 240 groups. CPU, 16 s. Log: `logs/pilot.log`. Raw: `results/runs.csv`. Post hoc: `results/posthoc.txt`.

## Verdict

**KILLED.** K1 (sup-norm fails, ρ ≤ 0.5): ρ = **+0.72**. K2 (gap-crossing under d^{π̂} works, ρ ≥ 0.8): **+0.93**, passed. K3 (beats the best norm by ≥ 0.1 and the unweighted crossing by ≥ 0.05): best norm +0.78, margin 0.15, passed that half; cross_unif +0.91, margin 0.02, failed. Secondary: |ρ(overestimation, Δ)| = 0.65, not ≤ 0.5; the sign is negative (more overestimation, better policy).

## Table (time-controlled Spearman with Δ; pooled within MDP; dense / sparse)

| statistic | time-ctrl | pooled | dense | sparse |
|---|---|---|---|---|
| sup ‖Q̂−Q*‖∞ | +0.72 | +0.76 | +0.80 | +0.63 |
| rms | +0.75 | +0.79 | +0.83 | +0.68 |
| rms weighted d^{π*} | +0.78 | +0.82 | +0.84 | +0.72 |
| rms weighted d^{π̂} | +0.77 | +0.81 | +0.83 | +0.72 |
| Bellman sup | +0.58 | +0.62 | +0.68 | +0.48 |
| Bellman rms d^{π̂} | +0.77 | +0.83 | +0.84 | +0.71 |
| overestimation under d^{π̂} | −0.65 | −0.73 | −0.74 | −0.55 |
| gap-crossing, uniform | +0.91 | +0.93 | +0.92 | +0.89 |
| gap-crossing, d^{π*} | +0.92 | +0.94 | +0.92 | +0.92 |
| gap-crossing, d^{π̂} | +0.93 | +0.95 | +0.91 | +0.94 |

## What happened

1. **Every norm predicts, moderately.** At fixed task and training time, the learners that are further along have smaller errors in every norm and better policies, so the sup-norm, RMS and weighted RMS all sit at 0.72–0.78. The sup/RMS ratio is 2.9, what i.i.d. Gaussian errors over 200 entries would give: the errors Q-learning with ε-greedy and restarts produces here are diffuse, and when errors are diffuse all norms are proportional.
2. **The gap-crossing statistic is better, not categorically better.** 0.93 vs 0.78: a real margin, because it discards error that does not flip an action. But the pre-registered claim was a regime change (norms fail), and they do not.
3. **Visitation weighting adds almost nothing.** Uniform, d^{π*} and d^{π̂} weighting of the crossing indicator give 0.91 / 0.92 / 0.93. With diffuse errors and a uniform evaluation distribution, where the policy goes does not matter much for which states' errors count.
4. **The overestimation plot has the wrong sign here.** 88% of the Q̂s under-estimate (zero initialisation, learning upward), and the learners that are further behind under-estimate more and have worse policies, so "more overestimation" goes with *better* performance (ρ = −0.65). This is a time-of-learning effect, not a bias effect, and it is the opposite of the deep-RL reading of such plots. It is also a caution for R15's and R16's use of overestimation as a target variable: there it was measured at convergence, where the time effect is gone.
5. **The Bellman sup-norm is the weakest predictor (0.58).** Consistent with Fujimoto et al. (2022).

## What this says

The pilot did not produce the error structure the hypothesis is about. Under ε-greedy with restarts on a 50-state MDP, errors are spread over the whole state space, so any norm is a proxy for progress. The hypothesis was motivated by function approximation and narrow behaviour data, where error concentrates away from the policy's path and a sup-norm is dominated by irrelevant states. That regime has to be produced deliberately (narrow behaviour, no restarts, or linear FA with aliasing) and was not in the pre-registered corpus. The gap-crossing statistic's 0.15 margin is the piece worth keeping, together with the sign result on overestimation.

## Caveats

- Tabular, uniform ρ, one γ, 24 small MDPs. The interesting regime (localised errors) was not in the corpus.
- Spearman within groups of 24 learners is noisy per group (sd 0.04–0.17 across groups), fine on average.
- Q-learning's α and ε are the dominant drivers of Δ at every time (`posthoc.txt` §1); the statistics are all partly proxies for those two factors.
