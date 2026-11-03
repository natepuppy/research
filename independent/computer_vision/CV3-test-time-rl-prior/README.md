# CV3: Test-time RL amplifies the prior — when TTRV helps, when it hurts, and a gate that tells them apart

**Question.** TTRV (CVPR 2026) adapts a VLM at test time with GRPO and no labels: sample several answers per image, reward each by its frequency among the samples, and reward low entropy of the empirical answer distribution. It reports large gains on recognition and VQA. The reward is self-consistency, so the update sharpens the model toward whatever it already believes — in the single-example bandit view the expected update increases the probability of the current mode and nothing else. Gains therefore cannot come from fixing an example whose sampled mode is wrong; they must come from (a) examples where the greedy answer and the sampled mode differ, and (b) weight sharing across examples in a batch, where sharpening on confidently-right examples moves the decision boundary for others. The same weight sharing is the harm: where the base model's confidence is driven by a prior that does not hold on the test data (a spurious cue, a class imbalance, a style), test-time RL pushes further into it. This is the confirmation-bias failure known for entropy-minimisation test-time adaptation (Tent) and the reason EATA and SAR filter samples; TTRV is a policy-gradient instance of the same objective and should inherit the failure. The questions: where do TTRV's gains come from, does the harm concentrate on prior-conflicting examples and scale with prior strength, does the net effect change sign as the conflicting fraction grows, and does a label-free gate — the instability of an example's sampled mode under input augmentation — separate the examples that will be helped from those that will be hurt better than the model's own confidence does?

**Status.** Laptop pilot run. Verdict: `KILL TEST: KILLED` (`results/verdict.txt`). Harm did not scale with the prior, and the net change was negative at every cell. RLC note: `paper/paper_rlc.md`. No VLM was adapted.

Laptop pilot (CPU, numpy, a linear softmax classifier trained on data with a spurious cue of controllable strength, tested under a covariate shift plus a controllable fraction of cue-conflicting examples; TTRV's frequency reward with GRPO advantages, batch adaptation on the unlabelled test set). Paper version on the 3080: Qwen3-VL-2B/4B with the TTRV recipe on its own benchmark suite (recognition and VQA), harm/gain split by whether the base model's mode is correct, prior strength manipulated through class-imbalanced and style-shifted test splits (e.g. ImageNet-R/Sketch subsets, Waterbirds-style spurious splits, long-tailed label shift), and the augmentation-disagreement gate against confidence and entropy filtering.

Prior work: Singh et al. CVPR 2026 (TTRV), Wang et al. ICLR 2021 (Tent; entropy minimisation at test time), Niu et al. ICML 2022 (EATA; filtering unreliable samples) and ICLR 2023 (SAR; collapse under label shift and mixed shifts — the closest diagnosis, for Tent), Zuo et al. 2025 (TTRL for LLMs; majority-vote reward; reports that the method depends on the base model's prior), Zhao et al. 2025 ("Learning to reason without external rewards"; self-consistency rewards and their limits). The pilot's contribution relative to EATA/SAR is the TTRV framing (policy gradient, group advantages, frequency reward), the harm/gain decomposition by mode-correctness, and the augmentation-instability gate rather than an entropy threshold. Novelty is the weak point of this project and the README says so; the paper has to earn it with the decomposition and the gate. Lab stubs: "RLVR when you cannot trust the grader" (the frequency reward is an untrustworthy grader by construction); "One value function instead of an ensemble" (adjacent, on confidence from a single model).

## Pre-registered kill test (written before the pilot ran)

Design. Classes C = 5. Core features: 10 dimensions, class means drawn once per seed from N(0, 0.5²), unit noise — weak enough that the classifier has to lean on the cue. Spurious block: 3 dimensions carrying a fixed per-class pattern; on training data the block matches the true class with probability s (**prior strength**, s ∈ {0.6, 0.8, 0.95}) and a random other class otherwise. Training set 5 000 examples; base model: multinomial logistic regression, 400 Adam steps, L2 1e-3. Test set 2 000 examples under a covariate shift (all core means moved by a common random vector of norm 1.0, which adaptation can in principle correct) with a fraction φ ∈ {0, 0.25, 0.5} of **conflicting** examples whose spurious block points at a wrong class; the rest **aligned**. Adaptation (TTRV recipe): for T = 50 steps over the whole unlabelled test set, sample K = 16 answers per example from the model, reward each sample by its frequency among the 16, standardise advantages within the group, take an Adam step (lr 1e-2) on the policy-gradient loss; no labels. 8 seeds.

Measurements. Accuracy before/after, overall and on the aligned and conflicting subsets. Per example: **harm** = correct before and wrong after; **gain** = wrong before and correct after. **Gate score** = mode instability: 16 augmented copies of the input (Gaussian noise σ = 0.5 on all features), the fraction whose argmax differs from the clean argmax. Baseline score = 1 − max softmax probability. Gated adaptation: exclude the 30 % of test examples with the highest gate score from the adaptation set (they are still evaluated).

- **K1 (harm is on the prior-conflicting examples and scales with the prior).** At φ = 0.25: Δaccuracy on the conflicting subset is negative in ≥ 6/8 seeds for every s, and the median Δaccuracy on the conflicting subset is monotonically more negative in s over {0.6, 0.8, 0.95}.
- **K2 (the net effect changes sign with the conflicting fraction).** At s = 0.8: median net Δaccuracy ≥ 0 at φ = 0 and ≤ 0 at φ = 0.5 (report the crossing).
- **K3 (the gate).** Pooled over cells with ≥ 20 harmed examples: AUROC of the gate score for harm among base-correct examples ≥ 0.70 and ≥ the confidence baseline's AUROC; at (s = 0.8, φ = 0.25) gated adaptation removes ≥ 50 % of the harm while keeping ≥ 70 % of the gain on the aligned subset (medians over seeds).

SUPPORTED if K1 and K2. INCONCLUSIVE if exactly one. KILLED if neither. K3 is reported; it is the part the paper would be about, and if the gate does no better than confidence the paper is a re-statement of SAR for TTRV and should not be written.

*Smoke-test change (before the pilot, 2 seeds × 20 adaptation steps):* the first draft drew class means from N(0, 1.2²) with a shift of norm 0.5. Base accuracy was 0.98 in every cell: the core features solved the task alone, the cue was irrelevant, and TTRV had nothing to amplify or fix (≤ 6 harmed examples per cell). Means were reduced to N(0, 0.5²) and the shift raised to norm 1.0, which puts base accuracy between 0.57 and 0.95 across cells with a visible accuracy drop on the conflicting subset. The change was chosen on the accuracy range alone; thresholds unchanged. The same smoke run showed the confidence baseline ahead of the gate score on AUROC in every cell, which the pilot may confirm — K3 is written so that this kills the gate, not the diagnosis.

What would make this a null: TTRV not hurting the conflicting subset (the batch generalisation corrects the spurious reliance rather than amplifying it), or the sign not flipping with φ. Either falsifies "test-time RL amplifies the prior" in the one setting where it should be easiest to see.

## Layout

| folder | contents |
|---|---|
| `code/` | `cv3_ttrl.py`; `code/lib/io_utils.py` |
| `results/` | `runs.csv` (per cell × seed), `summary.csv` (per cell), `verdict.txt` |
| `logs/` | `pilot.log` |
| `notes/` | `pilot_notes.md` (after the pilot) |
| `paper/` | `main.md` draft |

## Run

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python code/cv3_ttrl.py --smoke
.venv/bin/python code/cv3_ttrl.py             # pilot, ~5 min CPU
.venv/bin/python code/cv3_ttrl.py --summarize
```
