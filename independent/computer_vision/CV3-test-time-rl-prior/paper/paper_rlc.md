# Test-Time Self-Consistency Did Not Amplify the Prior in This Classifier

*RLC-length version of the synthetic pilot. Every number is from the summary printed by `code/cv3_ttrl.py` and from `results/verdict.txt`. Eight seeds. No vision-language model was run.*

## Abstract

Test-time reinforcement learning that rewards an answer for agreeing with the model's own samples can only sharpen the current mode. On a linear softmax classifier with a spurious cue of strength \(s\) and a fraction \(\phi\) of cue-conflicting test points, that prediction was pre-registered and killed. At \(\phi = 0.25\) the conflicting subset gets worse in 7/8 seeds at \(s = 0.6\) (median \(-0.029\)) and in 1/8 and 2/8 seeds at \(s = 0.8\) and \(s = 0.95\) (medians \(+0.024\) and \(+0.017\)), so the harm is not monotone in the prior. The net change is already negative when no test point conflicts: \(-0.033\), \(-0.008\) and \(-0.002\) at \(\phi = 0\). An augmentation-instability gate has pooled AUROC 0.872 for later harm, below the confidence baseline at 0.912, and at the pre-registered cell it removes 0.59 of the harm while keeping 0.37 of the aligned gain. The recorded verdict is KILLED.

## 1 The question

TTRV adapts a vision-language model at test time with no labels, rewarding each sampled answer by its frequency in the sample and rewarding a peaked answer distribution. In expectation that update raises the probability of the current mode. Gains have to come from examples the model already gets right in the mode but not in the greedy answer, or from weight sharing across a batch. The same weight sharing is the harm wherever confidence rests on a prior the test data do not share. Entropy-minimisation test-time adaptation has this failure, which is why EATA and SAR filter samples. A frequency reward is the policy-gradient version of the same objective.

The pilot is a linear classifier, not a VLM. It is the setting in which a spurious cue is the whole prior and the harm should be easiest to see.

## 2 Design

Five classes, 10 core features with class means drawn from \(\mathcal N(0, 0.5^2)\), and a 3-dimensional spurious block that matches the class with probability \(s \in \{0.6, 0.8, 0.95\}\) in training. The test set shifts every class mean by a shared vector of norm 1, and a fraction \(\phi \in \{0, 0.25, 0.5\}\) of test points have a contradictory cue. The base model is multinomial logistic regression, 400 Adam steps, on 5,000 examples. Adaptation, for 50 steps on 2,000 unlabelled test points, samples 16 answers, rewards each by its frequency, standardises advantages inside the group, and takes an Adam step at learning rate \(10^{-2}\). Eight seeds.

Harm is a point the model had right and then gets wrong. The gate is the fraction of 16 noisy copies whose argmax differs from the clean argmax. The baseline is one minus the maximum softmax probability. Gated adaptation drops the 30% of points with the highest gate score from the update.

K1 requires the conflicting subset to lose accuracy in at least 6 of 8 seeds at every \(s\), and the median loss to grow with \(s\). K2 requires, at \(s = 0.8\), a non-negative net change at \(\phi = 0\) and a non-positive net change at \(\phi = 0.5\). Supported if both.

A smoke run with larger class means put base accuracy at 0.98 and left the cue with nothing to do. Means were reduced and the shift was raised before the pilot, on the accuracy range only. Thresholds were not changed. The same smoke run had confidence ahead of the gate. That is what the pilot found.

## 3 Result

| \(s\) | \(\phi\) | accuracy before | after | \(\Delta\) | conflicting \(\Delta\) | aligned \(\Delta\) | harm | gain | AUROC gate / confidence |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.6 | 0 | 0.830 | 0.802 | \(-0.033\) | — | \(-0.033\) | 122 | 72 | 0.872 / 0.890 |
| 0.6 | 0.25 | 0.746 | 0.682 | \(-0.064\) | \(-0.029\) | \(-0.074\) | 218 | 96 | 0.799 / 0.844 |
| 0.6 | 0.5 | 0.692 | 0.620 | \(-0.066\) | \(-0.047\) | \(-0.077\) | 228 | 76 | 0.851 / 0.889 |
| 0.8 | 0 | 0.881 | 0.879 | \(-0.008\) | — | \(-0.008\) | 73 | 50 | 0.904 / 0.930 |
| 0.8 | 0.25 | 0.780 | 0.749 | \(-0.022\) | \(+0.024\) | \(-0.043\) | 108 | 69 | 0.883 / 0.916 |
| 0.8 | 0.5 | 0.648 | 0.611 | \(-0.021\) | \(-0.005\) | \(-0.050\) | 117 | 88 | 0.871 / 0.901 |
| 0.95 | 0 | 0.953 | 0.950 | \(-0.002\) | — | \(-0.002\) | 32 | 27 | 0.949 / 0.965 |
| 0.95 | 0.25 | 0.769 | 0.756 | \(-0.009\) | \(+0.017\) | \(-0.018\) | 70 | 39 | 0.898 / 0.932 |
| 0.95 | 0.5 | 0.590 | 0.556 | \(-0.023\) | \(-0.006\) | \(-0.032\) | 111 | 64 | 0.872 / 0.912 |

*Table 1. Medians over 8 seeds. Harm and gain are counts of test points.*

K1 fails. Only \(s = 0.6\) hurts the conflicting subset in 7/8 seeds. At the two stronger priors the conflicting median moves the other way. K2 fails because the net change at \(\phi = 0\) is already negative, \(-0.008\) at \(s = 0.8\), so there is no sign change to find. The aligned subset is where the accuracy is lost. K3 fails: pooled gate AUROC is 0.872, under confidence at 0.912, and at \((s, \phi) = (0.8, 0.25)\) the gate keeps 0.37 of the aligned gain against a required 0.70. The recorded verdict is KILLED.

## 4 What survived, and what was not run

Self-consistency adaptation did not help this classifier in any cell. The loss sits on the aligned points, which is not the prior-amplification story, and it does not grow with the cue. Confidence predicts the points that will be harmed better than augmentation disagreement does. Both of those are reasons not to write the VLM paper this pilot was a gate for.

Qwen3-VL-2B and 4B, QLoRA, and TTRV's recognition suite were not run. The README set that run behind a pilot that had to show harm on the conflicting subset scaling with the prior. It did not.

## References

Singh et al. (CVPR 2026). TTRV. · Wang et al. (ICLR 2021). Tent. · Niu et al. (ICML 2022). EATA. · Niu et al. (ICLR 2023). SAR. · Zuo et al. (2025). TTRL. · Shao et al. (2024). GRPO.
