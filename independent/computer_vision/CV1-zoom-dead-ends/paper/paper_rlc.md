# A Wrong First Crop Is Sometimes a Dead-End

*RLC-length version of the synthetic pilot. Every number is from `results/verdict.txt` and the summary printed with it. The run is `code/cv1_zoom.py` (8 seeds). No vision-language model was run.*

## Abstract

Zoom policies issue a short sequence of crops before answering. A prefix is a dead-end when the policy's own continuations almost never recover, while budget remains. On an 8×8 synthetic grid with a fixed stochastic zoom policy, Monte-Carlo continuations, and a logistic detector on cue margins, the pre-registered rule is INCONCLUSIVE. Dead-ends are real and early: over the cells that never re-crop on their own, 0.67 of failed episodes pass through a prefix whose continuation success is at most 0.15, and the median first dead-end is at depth 1. The detector does not clear the bar against a one-step margin baseline in every noise cell (AUROC 0.975 / 0.790 / 0.690 against margins 0.851 / 0.611 / 0.536). Gated re-crop gains +0.094, +0.051 and +0.036, against oracle gains +0.140, +0.125 and +0.108, and only the first of those three cells meets the joint action rule. A policy that already re-crops with probability 0.3 has almost no dead-ends at low noise, and the intervention adds nothing.

## 1 The question

DeepEyes, MGPO, ACTIVE-o3 and VIS-GRPO train a model to zoom before it answers, with a budget of a handful of crops. VIS-GRPO reads the collapse toward never zooming as an advantage-normalisation artefact. The alternative is irrecoverability (Fatemi and Killian, 2021): a prefix from which the probability of a correct answer under the policy's own continuation is near chance, while a re-crop would still fit in the budget. If that is the case, a detector on the current crop's features can fire before the budget is spent, and re-cropping when it fires should recover a share of what an oracle re-crop recovers.

The experiment that would decide this on V\* Bench, with a frozen zoom model, was not run. What was run is the estimator, on a grid where the irrecoverability of a prefix can be checked by continuing the same policy many times.

## 2 Design

The image is an 8×8 grid with one target cell and 10 symbols. A view starts as the full image. Zoom halves the side into one quadrant; re-crop returns to the full image; the symbol can be read only at side 2, with probability 0.95. The budget is 5 actions, so one re-crop-and-retry fits. Cues are the indicator that the target lies in a quadrant, plus noise of scale \(\sigma \cdot 0.7^{d}\). The base policy zooms into the loudest cue (softmax temperature 0.3). At a readable view that missed the target it re-crops with probability \(\rho\) and otherwise guesses. The grid is \(\sigma \in \{0.5, 1.0, 1.5\} \times \rho \in \{0, 0.3\}\), 8 seeds, 1,000 training episodes, 500 held-out episodes, 2,000 evaluation episodes.

A prefix at depth 0 or 1 with budget at least 3 is a dead-end when at most 15% of 32 continuations succeed. The detector is logistic regression on depth, budget, re-crop count, and cue margins. The baseline is the current margin alone. The intervention re-crops when the detector's probability exceeds 0.5. The oracle re-crops whenever the target is outside the view.

K1 requires, on \(\rho = 0\), at least half of failed episodes to pass through a dead-end and a median depth of at most 1. K2 requires held-out AUROC at least 0.80 and at least 0.05 above the margin baseline in all three \(\rho = 0\) cells. K3 requires a gain of at least 0.05, at least half the oracle's gain, and at most one extra action, in at least two of those three cells. Supported if K1 and K3.

## 3 Result

K1 passes. The fraction of failed episodes that pass through a dead-end, averaged as the rule states it, is 0.67, and the median first dead-end is at depth 1. Per cell the fail-through-dead-end rates at \(\rho = 0\) are 0.62, 0.68 and 0.72.

K2 fails. Detector against margin: 0.975 against 0.851 at \(\sigma = 0.5\), 0.790 against 0.611 at \(\sigma = 1.0\), 0.690 against 0.536 at \(\sigma = 1.5\). The detector is above the margin in every cell and above 0.80 only in the easiest one.

K3 fails the count of cells. Gated minus base is +0.094, +0.051 and +0.036, with extra actions +0.41, +0.49 and +0.72. Oracle gains are +0.140, +0.125 and +0.108. The heuristic that re-crops on a small margin gets +0.127, +0.050 and +0.027, which beats the detector at \(\sigma = 0.5\). At \(\rho = 0.3\) the dead-end rate collapses (0.00, 0.04, 0.22) and the gated gain is 0.

The recorded verdict is INCONCLUSIVE: K1 and not K3.

## 4 What this does and does not say

On this grid, a zoom that follows a noisy cue into the wrong quadrant is usually not recovered by the same policy, and that happens on the first crop. A linear detector on the cue margin sees some of it and does not see enough, at high noise, to beat a one-step threshold by the margin the rule required. The intervention helps where the cues are clean and does not recover half the oracle once they are not. A policy that re-crops on its own has little left for a detector to find.

Nothing here is a vision-language model, a real image, or a learned zoom policy. K1 was expected to be close to true by construction once \(\rho = 0\). The 3080 measurement, a frozen DeepEyes-class policy on V\* Bench with the same continuation estimator, was not run. The synthetic result says the estimator and the intervention are worth that measurement only where the base policy does not already re-crop and the cues are not dominated by noise. It does not say the measurement would pass.

## References

Fatemi, Killian et al. (2021). Medical dead-ends and irrecoverability. · DeepEyes, Zheng et al. (ICLR 2026). · MGPO, Huang et al. (2025). · ACTIVE-o3, Zhu et al. (2025). · VIS-GRPO, Zhu et al. (ECCV 2026).
