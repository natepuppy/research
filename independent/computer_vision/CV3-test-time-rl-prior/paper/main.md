# Test-Time RL Amplifies the Prior: Where TTRV's Gains Come From, Where It Hurts, and a Gate

*Draft skeleton, written before the pilot. Numbers in [brackets] are to be filled from `results/`; sections marked [3080] are the paper-scale experiments.*

**Abstract.** TTRV adapts a vision-language model at test time with GRPO and no labels, rewarding each sampled answer by its frequency among the samples and rewarding low entropy of the empirical answer distribution, and reports gains of up to 50 % on recognition. The reward is self-consistency: in expectation the update raises the probability of the current mode and nothing else, so it cannot repair an example the model already gets wrong. We ask where the gains come from and when the same mechanism hurts. Decomposing per-example outcomes into gains (wrong → right) and harms (right → wrong), we find in a controlled setting that gains arise from [weight sharing across examples under a covariate shift / greedy-vs-mode disagreement] and that harms concentrate on examples where the model's confidence rests on a prior that does not hold on the test data, growing with the strength of that prior ([numbers]). The net effect changes sign as the fraction of prior-conflicting examples grows past [φ*]. A label-free gate — the instability of an example's answer under input augmentation — predicts harm with AUROC [0.xx] against [0.xx] for the model's own confidence, and excluding gated examples from adaptation removes [X %] of the harm while keeping [Y %] of the gain. [3080] On Qwen3-VL-2B/4B with the TTRV recipe across its benchmark suite and prior-manipulated splits (class imbalance, style shift, spurious correlations), the same decomposition and gate yield [results]. Test-time RL is entropy minimisation by policy gradient and inherits its confirmation bias; the decomposition tells a practitioner when to turn it off.

## 1. Introduction

Test-time adaptation without labels has a long history in vision (Tent and its descendants) and a known failure: minimising prediction entropy sharpens whatever the model already believes, so under label shift or spurious correlation it drifts toward confidently wrong answers; EATA and SAR filter samples to contain this. TTRV brings the same objective to VLMs through GRPO — the frequency reward is the empirical mode's probability, the entropy reward is Tent's objective — and reports large gains. We take the gains as given and ask two questions a practitioner needs answered: which examples produce them, and which examples are being sacrificed.

The bandit view is sharp. For a single example with answer distribution p, a sample a rewarded by its frequency has advantage increasing in p(a); the expected policy-gradient step raises p at the mode. Gains therefore come from somewhere else: from the greedy answer differing from the sampled mode, and from weight sharing across examples in a batch. The latter is also the harm. We measure both, manipulate the prior's strength and the fraction of examples that conflict with it, and test a gate that uses only the base model.

Contributions. (1) The gain/harm decomposition for test-time RL by mode-correctness and prior-conflict [pilot; 3080]. (2) The dependence of harm on prior strength and of the net sign on the conflicting fraction [pilot; 3080]. (3) The augmentation-instability gate against confidence and entropy filtering [pilot; 3080]. (4) The identification of TTRV with entropy-minimisation TTA and the transfer of its known failure modes.

## 2. Related work

TTRV (Singh et al., CVPR 2026). Tent (Wang et al. 2021); EATA (Niu et al. 2022); SAR (Niu et al. 2023) — the closest diagnosis, for Tent under label shift. TTRL for LLMs (Zuo et al. 2025) and self-consistency rewards (Zhao et al. 2025), including the observation that they depend on the base model's prior. Spurious correlations: Waterbirds / Group DRO (Sagawa et al. 2020). Lab stubs: RLVR when you cannot trust the grader.

## 3. Setting

**Pilot.** Linear softmax classifier, five classes, 10 weak core dimensions plus a 3-dimensional spurious block matching the class with probability s ∈ {0.6, 0.8, 0.95} on training data. Test: covariate shift of norm 1.0 plus a conflicting fraction φ ∈ {0, 0.25, 0.5}. TTRV: 16 samples per example, frequency reward, standardised group advantages, Adam 1e-2, 50 steps over the unlabelled test set. Gate: fraction of 16 noise-augmented copies whose argmax differs from the clean argmax; exclude the top 30 %. 8 seeds.

**[3080] Paper setting.** Qwen3-VL-2B and 4B (QLoRA), the TTRV recipe (frequency + entropy rewards, GRPO) on TTRV's recognition and VQA suite; prior-manipulated splits: class-imbalanced subsets, ImageNet-R/Sketch style shifts, Waterbirds-style spurious splits. Decomposition by base-model mode-correctness and by split group. Gate: answer instability under standard image augmentations; baselines: softmax confidence, sample entropy (EATA/SAR-style), no gate.

Pre-registered (see README): K1 harm on the conflicting subset at φ = 0.25, negative in ≥ 6/8 seeds for every s and monotone in s; K2 net Δ ≥ 0 at φ = 0 and ≤ 0 at φ = 0.5 for s = 0.8; K3 gate AUROC ≥ 0.70 and ≥ confidence, ≥ 50 % harm removed with ≥ 70 % aligned gain kept. SUPPORTED if K1 ∧ K2.

## 4. Results

[pilot] Accuracy before/after, split by aligned/conflicting; harm and gain counts; harm versus s; net Δ versus φ with the crossing; gate and confidence AUROC; gated adaptation.

[3080] The same on the VLM suite and prior-manipulated splits.

## 5. Discussion

If K1 and K2 hold, TTRV's reported gains are a property of the benchmark's agreement with the model's prior, and the method should ship with a diagnostic. If K3 holds, the diagnostic is cheap and label-free. If K3 fails against confidence, the honest paper is shorter: TTRV inherits Tent's failure, SAR's remedy transfers, and the new content is the decomposition.

Limitations. The pilot's prior is a single spurious block; real priors are distributed. Batch adaptation is one TTRV mode; the single-example mode has no cross-example gain by construction and its reported gains need a separate explanation (greedy-vs-mode disagreement), which the 3080 version measures directly.

## 6. [3080] Planned experiments

(a) TTRV recipe on Qwen3-VL-2B/4B, decomposition on the original suite; (b) prior-manipulated splits; (c) single-example mode: gains attributable to greedy ≠ mode; (d) gate versus confidence and entropy filtering; (e) sensitivity to the entropy reward weight.

## 7. Conclusion

[After the pilot.]

## References

Singh et al. 2026; Wang et al. 2021; Niu et al. 2022, 2023; Zuo et al. 2025; Zhao et al. 2025; Sagawa et al. 2020; Shao et al. 2024.
